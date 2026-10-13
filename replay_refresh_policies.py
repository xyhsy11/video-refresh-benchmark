"""Replay fixed-budget refresh policies on saved training/validation artifacts."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from collect_cache_drift import AGE_INPUTS, MOTION_INPUTS, fresh_directory
from refresh_policy_utils import (LEARNED, POLICIES, SCORED, TOL, check_protected, distance_matrix,
    evaluate_plan, exact_oracle, fit_models, load_development, make_training_pairs, oracle_plan,
    plan_policy, protected_files, protocol, threshold_grids, verify_plan)
from research_utils import ROOT, now, relative_path, sha256_file, write_json
from test_refresh_policies import run_focused_checks

SOURCE_FILES=["replay_refresh_policies.py","refresh_policy_utils.py","test_refresh_policies.py","refresh_replay_report.py"]


def strict_json(path: Path, value: dict) -> None:
    json.dumps(value,allow_nan=False)
    write_json(path,value)


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path,index=False,float_format="%.17g",na_rep="")


def threshold_search(videos: list, matrices: dict, models: dict, grids: dict, budgets: list[int], output: Path) -> tuple[dict,pd.DataFrame]:
    rows=[]
    winners={}
    for policy in SCORED:
        winners[policy]={}
        for budget in budgets:
            candidates=[]
            for order,setting in enumerate(grids[policy]):
                errors=[]
                for video in videos:
                    plan=plan_policy(video.cheap,policy,budget,setting,models.get(policy))
                    trace,metrics=evaluate_plan(matrices[video.entry["video_id"]],plan)
                    verify_plan(video.cheap,plan,trace,budget)
                    errors.append(metrics["mean_output_error"])
                    rows.append({"video_id":video.entry["video_id"],"source_video_id":video.cheap.source_video_id,
                        "policy":policy,"budget":budget,"grid_order":order,"threshold_id":setting["threshold_id"],
                        "threshold_kind":setting["kind"],"threshold_value":setting["value"],
                        **{k:v for k,v in metrics.items() if k!="refresh_positions"},
                        "refresh_positions":json.dumps(metrics["refresh_positions"]),
                        "threshold_equal_video_mean_error":None})
                candidates.append({**setting,"grid_order":order,"equal_video_mean_error":float(np.mean(errors))})
            minimum=min(c["equal_video_mean_error"] for c in candidates)
            selected=next(c for c in candidates if c["equal_video_mean_error"]<=minimum+TOL)
            winners[policy][str(budget)]=selected
            summaries={c["threshold_id"]:c["equal_video_mean_error"] for c in candidates}
            for row in rows:
                if row["policy"]==policy and row["budget"]==budget:
                    row["threshold_equal_video_mean_error"]=summaries[row["threshold_id"]]
            print(f"Threshold selected {policy} B={budget}: {selected['threshold_id']}; developmental error {selected['equal_video_mean_error']:.6f}",flush=True)
    table=pd.DataFrame(rows)
    assert (table.groupby(["policy","budget","threshold_id"]).size()==40).all()
    save_csv(output/"threshold_search.csv",table)
    strict_json(output/"selected_thresholds.json",{"selected_at_utc":now(),"scope":"global validation-tuned development thresholds",
        "validation_video_count":40,"rule":protocol(budgets,0)["threshold_selection"],"selected":winners})
    return winners,table


def replay_validation(videos: list, matrices: dict, models: dict, thresholds: dict, budgets: list[int], seeds: list[int]) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    traces,metrics,oracles=[],[],{}
    for number,video in enumerate(videos,1):
        video_id=video.entry["video_id"]
        matrix=matrices[video_id]
        for budget in [*budgets,16]:
            policies=["dense"] if budget==16 else POLICIES
            if budget!=16:
                oracles[(video_id,budget)]=exact_oracle(matrix,budget)
            for policy in policies:
                for seed in seeds if policy=="random" else [None]:
                    setting=thresholds[policy][str(budget)] if policy in SCORED else None
                    plan=oracle_plan(video.cheap,oracles[(video_id,budget)]["positions"]) if policy=="oracle" else plan_policy(video.cheap,policy,budget,setting,models.get(policy),seed)
                    evaluated,summary=evaluate_plan(matrix,plan)
                    verify_plan(video.cheap,plan,evaluated,budget)
                    if budget!=16:
                        assert oracles[(video_id,budget)]["mean_error"]<=summary["mean_output_error"]+TOL
                    else:
                        assert summary["max_candidate_error"]<=TOL
                    identity={"trajectory_id":f"{video_id}:{policy}:B{budget}:seed{seed if seed is not None else 'none'}",
                        "video_id":video_id,"source_video_id":video.cheap.source_video_id,"label":video.entry["label"],
                        "policy":policy,"budget":budget,"seed_id":seed if seed is not None else -1,
                        "threshold_id":setting["threshold_id"] if setting else "not_applicable",
                        "score_units":"codec-displacement pixels" if policy=="simple_motion" else "predicted drift" if policy in LEARNED else "not_applicable"}
                    for row in evaluated:
                        traces.append({**identity,"decoded_frame_index":video.cheap.indices[row["candidate_ordinal"]],
                            "timestamp_seconds":video.cheap.timestamps[row["candidate_ordinal"]],**row})
                    metrics.append({**identity,**{k:v for k,v in summary.items() if k!="refresh_positions"},
                        "refresh_positions":json.dumps(summary["refresh_positions"]),
                        "clipped_predicted_score_count":sum(r["score_raw"]!=r["score_clipped"] for r in plan if r["score_raw"] is not None) if policy in LEARNED else 0,
                        "oracle_schedules_enumerated":oracles[(video_id,budget)]["enumerated_schedules"] if policy=="oracle" else 0})
        print(f"Replayed {number}/{len(videos)} validation videos: {video_id}",flush=True)
    return pd.DataFrame(traces),pd.DataFrame(metrics),oracles


def per_video_metrics(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (video_id,budget,policy),group in trajectories.groupby(["video_id","budget","policy"],sort=True):
        assert len(group)==(10 if policy=="random" else 1)
        assert group.source_video_id.nunique()==1
        assert (group.simulated_refreshes==budget).all() and (group.reuses==16-budget).all() and (group.feature_outputs==16).all()
        rows.append({"video_id":video_id,"source_video_id":group.source_video_id.iloc[0],"label":group.label.iloc[0],
            "budget":int(budget),"policy":policy,"trajectory_count":len(group),
            "mean_output_error":float(group.mean_output_error.mean()),
            "max_candidate_error":float(group.max_candidate_error.mean()),
            "simulated_refreshes":int(budget),"reuses":16-int(budget),"feature_outputs":16,
            "mean_forced_refreshes":float(group.forced_refreshes.mean()),
            "fraction_trajectories_with_forced_refresh":float((group.forced_refreshes>0).mean()),
            "forced_share_of_nonmandatory_refreshes":float(group.forced_refreshes.mean()/(budget-1)),
            "mean_clipped_predicted_scores":float(group.clipped_predicted_score_count.mean()),
            "random_seed_error_std_population":float(group.mean_output_error.std(ddof=0)) if policy=="random" else None,
            "random_seed_error_min":float(group.mean_output_error.min()) if policy=="random" else None,
            "random_seed_error_max":float(group.mean_output_error.max()) if policy=="random" else None,
            "random_max_candidate_error_over_seeds":float(group.max_candidate_error.max()) if policy=="random" else None,
            "refresh_positions":json.dumps({str(s):json.loads(positions) for s,positions in zip(group.seed_id,group.refresh_positions)})})
    result=pd.DataFrame(rows)
    for budget in sorted(result.budget.unique()):
        if budget==16: continue
        indices=result.index[result.budget==budget]
        by_video=result.loc[indices].pivot(index="video_id",columns="policy",values="mean_output_error")
        for index in indices:
            row=result.loc[index]
            periodic=float(by_video.loc[row.video_id,"periodic"])
            oracle=float(by_video.loc[row.video_id,"oracle"])
            reduction=periodic-row.mean_output_error
            result.loc[index,"absolute_reduction_vs_periodic"]=reduction
            result.loc[index,"percentage_reduction_vs_periodic"]=100*reduction/periodic if periodic>TOL else np.nan
            result.loc[index,"oracle_gap_closure"]=reduction/(periodic-oracle) if periodic-oracle>TOL else np.nan
            result.loc[index,"comparison_outcome"]="improved" if reduction>TOL else "worsened" if reduction<-TOL else "tied"
            result.loc[index,"oracle_gap_note"]="defined" if periodic-oracle>TOL else "undefined: periodic-oracle denominator <=1e-12"
    assert len(result)==880
    return result


def aggregate_results(per_video: pd.DataFrame, trajectories: pd.DataFrame) -> tuple[dict,pd.DataFrame]:
    summaries=[]
    for (budget,policy),group in per_video.groupby(["budget","policy"],sort=True):
        assert len(group)==40
        def statistic(column,method="mean"):
            values=group[column].dropna()
            return float(getattr(values,method)()) if len(values) else None
        summaries.append({"budget":int(budget),"policy":policy,"videos":40,
            "mean_output_error":statistic("mean_output_error"),"median_output_error":statistic("mean_output_error","median"),
            "mean_max_candidate_error":statistic("max_candidate_error"),
            "mean_absolute_reduction_vs_periodic":statistic("absolute_reduction_vs_periodic"),
            "mean_per_video_percentage_reduction":statistic("percentage_reduction_vs_periodic"),
            "median_per_video_percentage_reduction":statistic("percentage_reduction_vs_periodic","median"),
            "mean_defined_oracle_gap_closure":statistic("oracle_gap_closure"),
            "undefined_oracle_gap_videos":int(group.oracle_gap_closure.isna().sum()) if budget!=16 else None,
            "improved":int((group.comparison_outcome=="improved").sum()) if budget!=16 else None,
            "tied":int((group.comparison_outcome=="tied").sum()) if budget!=16 else None,
            "worsened":int((group.comparison_outcome=="worsened").sum()) if budget!=16 else None,
            "mean_forced_refreshes":statistic("mean_forced_refreshes"),
            "fraction_trajectories_with_forced_refresh":statistic("fraction_trajectories_with_forced_refresh"),
            "mean_forced_share_of_nonmandatory_refreshes":statistic("forced_share_of_nonmandatory_refreshes"),
            "mean_random_seed_error_std":statistic("random_seed_error_std_population")})
    pairs=[]
    for budget in [2,4,8]:
        pivot=per_video.loc[per_video.budget==budget].pivot(index="video_id",columns="policy",values="mean_output_error")
        comparisons=[(p,"periodic") for p in POLICIES if p!="periodic"]+[("age_plus_motion",p) for p in ["random","simple_motion","motion_only","age_only"]]
        for candidate,reference in comparisons:
            for video_id in pivot.index:
                delta=float(pivot.loc[video_id,reference]-pivot.loc[video_id,candidate])
                denominator=float(pivot.loc[video_id,reference])
                pairs.append({"video_id":video_id,"budget":budget,"candidate":candidate,"reference":reference,
                    "absolute_error_reduction":delta,"percentage_error_reduction":100*delta/denominator if denominator>TOL else None,
                    "outcome":"improved" if delta>TOL else "worsened" if delta<-TOL else "tied"})
    paired=pd.DataFrame(pairs)
    comparisons=[]
    for (budget,candidate,reference),group in paired.groupby(["budget","candidate","reference"],sort=True):
        comparisons.append({"budget":int(budget),"candidate":candidate,"reference":reference,
            "mean_absolute_reduction":float(group.absolute_error_reduction.mean()),
            "mean_per_video_percentage_reduction":float(group.percentage_error_reduction.mean()) if group.percentage_error_reduction.notna().any() else None,
            "improved":int((group.outcome=="improved").sum()),"tied":int((group.outcome=="tied").sum()),"worsened":int((group.outcome=="worsened").sum())})
    random_variability=[]
    for budget,group in trajectories.loc[trajectories.policy=="random"].groupby("budget"):
        seed_means=group.groupby("seed_id").mean_output_error.mean()
        random_variability.append({"budget":int(budget),"mean_video_error_by_seed":{str(int(s)):float(v) for s,v in seed_means.items()},
            "min_seed_equal_video_mean_error":float(seed_means.min()),"max_seed_equal_video_mean_error":float(seed_means.max()),
            "note":"descriptive variability over all ten fixed random seeds; no confidence interval"})
    return {"scope":"validation-tuned developmental policy evaluation; not unseen-test evidence",
        "aggregation":"random averaged within video first; 40 source videos equal weight; no independent candidate/seed sampling",
        "policy_metrics":summaries,"paired_comparisons":comparisons,"random_seed_variability":random_variability,
        "random_max_error_definition":"max_candidate_error is mean of ten trajectory maxima within video; worst seed is separately recorded"},paired


def verify_saved(output: Path, expected_traces: pd.DataFrame, expected_trajectories: pd.DataFrame,
                 expected_per_video: pd.DataFrame, expected_aggregate: dict) -> dict:
    traces=pd.read_csv(output/"policy_traces.csv",float_precision="round_trip")
    trajectories=pd.read_csv(output/"validation_trajectories.csv",float_precision="round_trip")
    saved_per_video=pd.read_csv(output/"validation_per_video_metrics.csv",float_precision="round_trip")
    assert len(traces)==31360 and len(trajectories)==1960
    assert not traces.duplicated(["trajectory_id","candidate_ordinal"]).any()
    for identity,group in traces.groupby("trajectory_id",sort=False):
        group=group.sort_values("candidate_ordinal")
        assert group.candidate_ordinal.tolist()==list(range(16)) and group.refresh.iloc[0]
        budget=int(group.budget.iloc[0])
        assert sum(group.refresh)==budget and group.remaining_budget_after.iloc[-1]==0
        cached=0
        for row in group.itertuples():
            if row.candidate_ordinal:
                assert int(row.cached_before_ordinal)==cached
            if row.refresh:
                cached=row.candidate_ordinal
                assert abs(row.output_feature_error)<=TOL
            assert row.cached_output_ordinal==cached
        metrics=trajectories.loc[trajectories.trajectory_id==identity].iloc[0]
        assert abs(group.output_feature_error.mean()-metrics.mean_output_error)<=TOL
        assert abs(group.output_feature_error.max()-metrics.max_candidate_error)<=TOL
        assert json.loads(metrics.refresh_positions)==group.loc[group.refresh,"candidate_ordinal"].tolist()
        assert int(group.forced_refresh.sum())==metrics.forced_refreshes
    regenerated=per_video_metrics(trajectories)
    pd.testing.assert_frame_equal(saved_per_video,regenerated,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    aggregate,_=aggregate_results(regenerated,trajectories)
    assert aggregate==expected_aggregate
    search=pd.read_csv(output/"threshold_search.csv",float_precision="round_trip")
    winners=json.loads((output/"selected_thresholds.json").read_text())["selected"]
    for policy,by_budget in winners.items():
        for budget,selected in by_budget.items():
            group=search.loc[(search.policy==policy)&(search.budget==int(budget))]
            means=group.groupby("grid_order").mean_output_error.mean()
            order=int(means.index[means<=means.min()+TOL][0])
            assert order==selected["grid_order"]
            assert abs(means.loc[order]-selected["equal_video_mean_error"])<=TOL
    return {"saved_traces_latest_actual_cache_and_counters":True,"saved_traces_reproduce_trajectory_quality":True,
        "saved_per_video_seed_averaging_and_aggregates_reproduced":True,"all_threshold_results_reproduce_global_selection":True,
        "trajectory_rows":len(trajectories),"trace_rows":len(traces),"per_video_metric_rows":len(saved_per_video)}


def run(args,output: Path) -> int:
    started=now()
    frozen=protocol(args.budgets,args.seed)
    strict_json(output/"protocol.json",{"frozen_at_utc":started,"protocol":frozen})
    protected=protected_files()
    sources={name:sha256_file(ROOT/name) for name in SOURCE_FILES}
    metadata={"experiment":"adaptive_refresh_v1","status":"running","started_at_utc":started,
        "input_directory":relative_path(args.input),"output_directory":relative_path(output),
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "source_code_sha256":sources,"protected_original_sha256":protected,
        "protocol_sha256":sha256_file(output/"protocol.json"),"no_RGB_inference_or_GPU_timing":True,
        "actual_encoder_calls":0,"video_downloads":0,"encoded_video_reads":0,
        "environment":{"python":platform.python_version(),"os":platform.platform(),
            "execution":"CPU NumPy/Pandas replay of saved arrays; no CUDA operation",
            "packages":{name:importlib.metadata.version(name) for name in ["numpy","pandas","matplotlib","torch","torchvision","av","Pillow"]}}}
    strict_json(output/"metadata.json",metadata)
    software=run_focused_checks()
    assert software["passed"]
    print(f"Focused software verification PASS: {len(software['checks'])}",flush=True)
    train,validation,identities=load_development(args.input)
    metadata.update({"input_identities":identities,"training_videos":len(train),"validation_videos":len(validation),"test_videos_used":0})
    pairs=make_training_pairs(train)
    assert len(pairs)==14400
    save_csv(output/"train_pairs.csv",pairs)
    saved_pairs=pd.read_csv(output/"train_pairs.csv",float_precision="round_trip")
    pd.testing.assert_frame_equal(pairs,saved_pairs,check_dtype=False,check_exact=True)
    # Fit each actual model exactly once; the CSV round-trip was checked above.
    models=fit_models(saved_pairs)
    fit_record={"fitted_at_utc":now(),"training_observations":len(pairs),"training_source_video_ids":identities["training_sources"],
        "train_pairs_sha256":sha256_file(output/"train_pairs.csv"),"models":models,
        "input_definitions":{k:frozen[k] for k in ["age_inputs","motion_inputs","motion_interval","missing_motion"]}}
    strict_json(output/"fitted_predictors.json",fit_record)
    fitted_hash=sha256_file(output/"fitted_predictors.json")
    grids=threshold_grids(pairs,models)
    strict_json(output/"threshold_grid.json",{"built_from_training_only_at_utc":now(),"settings":grids,"quantile_method":"linear"})
    metadata.update({"training_pairs":len(pairs),"fit_completed_before_validation_at_utc":now(),
                     "fitted_predictors_sha256":fitted_hash})
    strict_json(output/"metadata.json",metadata)
    print("Real inputs verified: 120 training, 40 validation; 14,400 all-history training pairs. Test data excluded.",flush=True)
    matrices={v.entry["video_id"]:distance_matrix(v.references) for v in validation}
    thresholds,search=threshold_search(validation,matrices,models,grids,args.budgets,output)
    assert sha256_file(output/"fitted_predictors.json")==fitted_hash
    assert models==json.loads((output/"fitted_predictors.json").read_text())["models"]
    seeds=list(range(args.seed,args.seed+10))
    traces,trajectories,oracles=replay_validation(validation,matrices,models,thresholds,args.budgets,seeds)
    assert models==json.loads((output/"fitted_predictors.json").read_text())["models"]
    per_video=per_video_metrics(trajectories)
    aggregate,paired=aggregate_results(per_video,trajectories)
    save_csv(output/"policy_traces.csv",traces)
    save_csv(output/"validation_trajectories.csv",trajectories)
    save_csv(output/"validation_per_video_metrics.csv",per_video)
    save_csv(output/"validation_pairwise_comparisons.csv",paired)
    strict_json(output/"validation_aggregate_metrics.json",aggregate)
    strict_json(output/"oracle_schedules.json",{"offline_only":True,"videos":
        [{"video_id":v,"budget":b,**record} for (v,b),record in sorted(oracles.items())]})
    metadata.update({"random_seeds":seeds,"oracle_schedules_checked":len(oracles),
        "validation_trajectory_count":len(trajectories),"validation_trace_count":len(traces),
        "collection_model_and_numerics":{k:json.loads((args.input/"collection_metadata.json").read_text())[k] for k in ["model","numerical_settings","decoder","motion_descriptor_definition"]}})
    from refresh_replay_report import create_report
    examples=create_report(output,per_video,trajectories,traces,aggregate,metadata)
    saved_checks=verify_saved(output,traces,trajectories,per_video,aggregate)
    assert set(pairs.source_video_id)==set(identities["training_sources"])
    assert set(traces.source_video_id)==set(identities["validation_sources"])
    assert set(pairs.source_video_id).isdisjoint(set(traces.source_video_id))
    assert sha256_file(output/"fitted_predictors.json")==fitted_hash
    check_protected(protected)
    release=subprocess.run([sys.executable,str(ROOT/"validate_release.py")],cwd=ROOT,capture_output=True,text=True)
    assert release.returncode==0,release.stdout+release.stderr
    verification={"passed":True,"verified_at_utc":now(),"software_checks":software,"training_videos":120,
        "validation_videos":40,"training_pairs":14400,"pairs_per_training_video":120,"test_videos_used":0,
        "train_validation_disjoint":True,"parsed_development_input_hashes_verified":True,
        "models_unchanged_after_threshold_selection":True,"all_real_plans_budget_cache_interval_checks":True,
        "in_memory_models_match_saved_parameters_after_search_and_replay":True,"fits_per_actual_training_model":1,
        "real_oracle_no_worse_than_all_replayed_same_budget_policies":True,
        "dense_max_error_approximately_zero":True,"original_sources_and_snapshots_unchanged":len(protected),
        "release_validator":{"returncode":release.returncode,"stdout":release.stdout.strip()},**saved_checks}
    strict_json(output/"verification.json",verification)
    metadata.update({"status":"complete","completed_at_utc":now(),"example_videos":examples,
        "saved_artifact_sha256":{p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!="metadata.json"}})
    strict_json(output/"metadata.json",metadata)
    checksum_files=sorted(p for p in output.rglob("*") if p.is_file() and p.name!="SHA256SUMS.txt")
    (output/"SHA256SUMS.txt").write_text("".join(f"{sha256_file(p)}  {p.relative_to(output).as_posix()}\n" for p in checksum_files),encoding="utf-8")
    print("Replay PASS: 120 training / 40 validation / 0 test videos; 14,400 training pairs; 1,960 trajectories and 31,360 outputs.",flush=True)
    for budget in args.budgets:
        print(f"B={budget}: "+", ".join(f"{r['policy']}={r['mean_output_error']:.6f}" for r in aggregate["policy_metrics"] if r["budget"]==budget),flush=True)
    print(f"Saved {relative_path(output)}",flush=True)
    return 0


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("results/cache_drift_v2"))
    parser.add_argument("--output",type=Path,default=Path("results/adaptive_refresh_v1"))
    parser.add_argument("--budgets",type=int,nargs="+",default=[2,4,8])
    parser.add_argument("--seed",type=int,default=20261013)
    args=parser.parse_args()
    if sorted(args.budgets)!=[2,4,8]:
        parser.error("This frozen protocol evaluates budgets 2, 4, and 8 exactly once")
    args.budgets=sorted(args.budgets)
    output=fresh_directory(args.output)
    try:
        return run(args,output)
    except Exception as exc:
        metadata_path=output/"metadata.json"
        metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        metadata.update({"status":"incomplete","failed_at_utc":now(),"blocker":f"{type(exc).__name__}: {exc}"})
        strict_json(metadata_path,metadata)
        print(f"Replay incomplete: {type(exc).__name__}: {exc}; retained {relative_path(output)}",file=sys.stderr)
        raise


if __name__=="__main__":
    raise SystemExit(main())
