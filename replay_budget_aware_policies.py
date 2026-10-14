"""Budget-aware adaptive refresh: saved-feature offline development replay."""
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

import budget_aware_refresh_utils as u
from collect_cache_drift import fresh_directory
from research_utils import ROOT, now, relative_path, sha256_file, write_json
from test_budget_aware_refresh import run_checks

SOURCES=['replay_budget_aware_policies.py','budget_aware_refresh_utils.py','test_budget_aware_refresh.py','budget_aware_refresh_report.py']


def strict_json(path:Path,value:dict) -> None:
    json.dumps(value,allow_nan=False)
    write_json(path,value)


def csv(path:Path,frame:pd.DataFrame) -> None:
    frame.to_csv(path,index=False,float_format='%.17g',na_rep='')


def protocol(budgets:list[int],seed:int) -> dict:
    return {'experiment':u.EXPERIMENT,'client_date':'2026-10-14','scope':'offline development study; no original validation/test features, motion, predictions or evaluation results loaded',
        'development_split':{'original_membership':'cache_drift_v2 train only','seed':seed,'classes':20,'per_class':{'train':4,'development':2},'totals':{'train':80,'development':40},
            'rule':'sorted six source IDs/class; class SHA256 seed documented in development_split.json; random.Random.shuffle; first four train'},
        'arithmetic':'float64 distances, population scaling, ridge fitting and metrics; original saved reference arrays float32',
        'age_inputs':u.AGE_INPUTS,'motion_inputs':u.MOTION_INPUTS,'budget_inputs':u.BUDGET_INPUTS,'action_value_columns':u.VALUE_COLUMNS,
        'ridge_alpha':1.0,'scaling':'train-only population mean/std, std<=1e-12 -> scale=1; unpenalized intercept',
        'drift_training':'all r<t pairs: 120 per video, 9,600 from 80 videos; original appearance-feature drift target',
        'action_value_training':'all reachable feasible states from (1,0,B-1); only states with both actions feasible; separate models per B',
        'V':'minimum SUM from t onward with exactly b remaining refreshes; terminal V(16,r,0)=0, other terminal b infeasible',
        'Q_reuse':'D[t,r]+V(t+1,r,b), feasible iff b<=16-t-1','Q_refresh':'D[t,t]+V(t+1,t,b-1), feasible iff b>=1',
        'action_value_target':'refresh_advantage=(Q_reuse-Q_refresh)/16; signed, distinct from drift; future RGB is permitted for TRAINING labels only',
        'value_decision':'optional refresh iff raw signed predicted advantage >0; threshold fixed zero, no clipping or tuning',
        'exact_budget':'first refresh mandatory and counted; b=0 requires reuse; b=16-t requires refresh; cache/motion interval reset on actual refresh',
        'candidates':16,'outputs':16,'budgets':budgets,'primary_budget':4,'dense_budget':16,'sparse_policies':u.POLICIES,
        'threshold_policies':u.THRESHOLD_POLICIES,'threshold_grid':'unique TRAINING-score linear quantiles 0%,5%,...,100%; always/wait kind with value=null',
        'drift_score':'clip predictions to [0,2]; recompute age/motion from current actual cache; optional refresh iff score >= threshold',
        'privileged_score':'actual pre-decision D[t,r], with grid from training-pair actual drift; unavailable without fresh RGB, not an upper bound',
        'pacing':{'lambda_grid':u.LAMBDAS,'rule':'threshold_t=base_threshold+lambda*((B-1)/15-b/(16-t))','extremes':'always/wait once, lambda=0'},
        'selection':'lowest equal-video mean development error; within 1e-12 of minimum prefer smaller lambda, then earlier base grid order; coefficients frozen',
        'periodic':'range(0,16,16//B)','random_seeds':list(range(seed,seed+10)),
        'random_rule':'mandatory 0 + random.Random.sample(1..15,B-1); seed first16 SHA256 hex of budget_aware_refresh_v2|random|seed|sourceID; all10 seeds averaged within video',
        'motion_intervals':'strictly after actual cached frame through current frame inclusive; equal frame weights when available; missing magnitudes zero with explicit availability',
        'metric':'mean POST-decision output-feature cosine error over all16 candidates; equal weight/video',
        'comparisons':'periodic-policy per video; percentages per video before means; oracle gap closure=(periodic-policy)/(periodic-oracle); null if denominator<=1e-12; no clipping',
        'action_regret':'(Q_chosen-min_feasible_Q)/16, evaluator only; summed regrets equal mean-error excess over exact oracle',
        'oracle':'existing exhaustive full-future schedules cross-checked with exact DP; lexicographic tie within1e-12',
        'example_rule':'B4 value_age_motion_budget largest absolute gain and largest loss versus periodic, source-ID tie break; descriptive only',
        'limitations':['development settings selected on evaluated development videos','appearance fidelity, not recognition accuracy','zero actual encoder calls and no latency measurements','offline: full-clip candidates and future-reference codec vectors','privileged future RGB trains oracle labels','states from one video are correlated'],
        'fresh_test':'later stage only:100 new videos,five per same class,exclude allprior source IDs and duplicate hashes; fixed fitted models/settings; paired whole-source-video uncertainty,no retuning; no acquisition now'}


def settings_search(dev:list,oracles:dict,models:dict,grids:dict,budgets:list[int],output:Path) -> tuple[dict,pd.DataFrame]:
    all_rows=[]; selected={}
    for policy in u.THRESHOLD_POLICIES:
        selected[policy]={}
        for B in budgets:
            candidates=[]
            for setting in grids[policy]:
                local=[]
                for video in dev:
                    vid=video.entry['video_id']; oracle=oracles[vid]
                    plan=u.plan_privileged(video.cheap,B,oracle.d,setting) if policy=='privileged_actual_drift' else u.plan_deployable(video.cheap,policy,B,models,setting)
                    _,m=u.evaluate(video.cheap,plan,oracle,B)
                    local.append({'video_id':vid,'source_video_id':video.cheap.source_video_id,'policy':policy,'budget':B,**setting,
                                  **{k:v for k,v in m.items() if k!='refresh_positions'},'refresh_positions':json.dumps(m['refresh_positions'])})
                average=float(np.mean([row['mean_output_error'] for row in local]))
                all_rows.extend({**row,'setting_equal_video_mean_error':average} for row in local)
                candidates.append({**setting,'equal_video_mean_error':average})
            minimum=min(c['equal_video_mean_error'] for c in candidates)
            winner=min((c for c in candidates if c['equal_video_mean_error']<=minimum+u.TOL),key=lambda c:(c['lambda'],c['grid_order']))
            selected[policy][str(B)]=winner
            print(f"Selected {policy} B={B}: {winner['setting_id']}; development error {winner['equal_video_mean_error']:.6f}",flush=True)
    table=pd.DataFrame(all_rows)
    assert (table.groupby(['policy','budget','setting_id']).size()==40).all()
    # Lambda=0 is the same predictor, grid and decisions as the static baseline.
    static=table[table.policy=='age_plus_motion'].sort_values(['budget','setting_id','video_id'])
    paced=table[(table.policy=='paced_age_plus_motion')&(table['lambda']==0)].sort_values(['budget','setting_id','video_id'])
    assert static[['budget','setting_id','video_id']].to_numpy().tolist()==paced[['budget','setting_id','video_id']].to_numpy().tolist()
    assert static.refresh_positions.tolist()==paced.refresh_positions.tolist()
    np.testing.assert_array_equal(static.mean_output_error.to_numpy(),paced.mean_output_error.to_numpy())
    csv(output/'threshold_search.csv',table)
    strict_json(output/'selected_thresholds.json',{'selected_at_utc':now(),'scope':'development-tuned global settings','fixed_value_threshold':0.0,'selected':selected})
    return selected,table


def replay(dev:list,oracles:dict,models:dict,selected:dict,budgets:list[int],seed:int) -> tuple[pd.DataFrame,pd.DataFrame,list]:
    traces=[]; trajectories=[]; checks=[]
    for index,video in enumerate(dev,1):
        vid=video.entry['video_id']; oracle=oracles[vid]
        for B in [*budgets,16]:
            policies=['dense'] if B==16 else u.POLICIES
            if B!=16:
                exact=u.exact_oracle(oracle.d,B)
                optimum=(oracle.d[0,0]+oracle.value(1,0,B-1))/16
                assert abs(optimum-exact['exact_minimum_error'])<=u.TOL
                checks.append({'video_id':vid,'budget':B,'dp_mean_error':float(optimum),'exhaustive_mean_error':exact['exact_minimum_error']})
            for policy in policies:
                setting=selected[policy][str(B)] if policy in selected else None
                for run_seed in range(seed,seed+10) if policy=='random' else [None]:
                    if policy=='oracle': plan=u.plan_oracle(video.cheap,B,oracle.d)
                    elif policy=='privileged_actual_drift': plan=u.plan_privileged(video.cheap,B,oracle.d,setting)
                    else: plan=u.plan_deployable(video.cheap,policy,B,models,setting,run_seed)
                    evaluated,m=u.evaluate(video.cheap,plan,oracle,B)
                    identity={'trajectory_id':f'{vid}:{policy}:B{B}:seed{run_seed if run_seed is not None else "none"}',
                        'video_id':vid,'source_video_id':video.cheap.source_video_id,'label':video.entry['label'],
                        'policy':policy,'budget':B,'seed_id':run_seed if run_seed is not None else -1,
                        'setting_id':setting['setting_id'] if setting else 'not_applicable','privileged':policy in ['oracle','privileged_actual_drift'],
                        'score_units':'signed whole-clip mean-error advantage' if policy in u.VALUE_COLUMNS else 'actual appearance drift (privileged)' if policy=='privileged_actual_drift' else 'predicted appearance drift' if policy in u.THRESHOLD_POLICIES else 'not_applicable'}
                    traces.extend({**identity,'decoded_frame_index':video.cheap.indices[row['candidate_ordinal']],
                                   'timestamp_seconds':video.cheap.timestamps[row['candidate_ordinal']],**row} for row in evaluated)
                    trajectories.append({**identity,**{k:v for k,v in m.items() if k!='refresh_positions'},'refresh_positions':json.dumps(m['refresh_positions'])})
        print(f'Replayed {index}/40 development videos: {vid}',flush=True)
    traces=pd.DataFrame(traces); trajectories=pd.DataFrame(trajectories)
    assert len(trajectories)==2200 and len(traces)==35200
    return traces,trajectories,checks


def summarize(trajectories:pd.DataFrame) -> tuple[pd.DataFrame,dict,pd.DataFrame]:
    per=[]
    for (vid,B,policy),g in trajectories.groupby(['video_id','budget','policy'],sort=True):
        assert len(g)==(10 if policy=='random' else 1)
        assert (g.simulated_refreshes==B).all() and (g.cache_reuses==16-B).all() and (g.feature_outputs==16).all()
        per.append({'video_id':vid,'source_video_id':g.source_video_id.iloc[0],'label':g.label.iloc[0],'budget':int(B),'policy':policy,
            'privileged':bool(g.privileged.iloc[0]),'trajectory_count':len(g),
            **{f'mean_{c}' if c not in ['mean_output_error'] else c:float(g[c].mean()) for c in ['mean_output_error','max_output_error','forced_refreshes','last_refresh_position','outputs_after_budget_exhausted','sum_action_regret']},
            'simulated_refreshes':int(B),'cache_reuses':16-int(B),'feature_outputs':16,
            'fraction_trajectories_with_forced_refresh':float((g.forced_refreshes>0).mean()),
            'random_seed_error_std_population':float(g.mean_output_error.std(ddof=0)) if policy=='random' else None,
            'refresh_positions_by_seed':json.dumps({str(s):json.loads(p) for s,p in zip(g.seed_id,g.refresh_positions)})})
    per=pd.DataFrame(per); paired=[]
    comparisons=[(p,'periodic') for p in u.POLICIES if p!='periodic']+[
        ('value_age_motion_budget','paced_age_plus_motion'),('value_age_motion_budget','value_age_budget'),
        ('value_age_motion_budget','age_plus_motion'),('privileged_actual_drift','age_plus_motion'),('value_age_budget','paced_age_plus_motion')]
    for B in [2,4,8]:
        pivot=per[per.budget==B].pivot(index='video_id',columns='policy',values='mean_output_error')
        for i in per.index[per.budget==B]:
            row=per.loc[i]; periodic=float(pivot.loc[row.video_id,'periodic']); exact=float(pivot.loc[row.video_id,'oracle'])
            delta=periodic-row.mean_output_error
            per.loc[i,'absolute_improvement_vs_periodic']=delta
            per.loc[i,'percentage_improvement_vs_periodic']=100*delta/periodic if periodic>u.TOL else np.nan
            per.loc[i,'oracle_headroom']=periodic-exact
            per.loc[i,'oracle_gap_closure']=delta/(periodic-exact) if periodic-exact>u.TOL else np.nan
            per.loc[i,'gap_reason']='defined' if periodic-exact>u.TOL else 'undefined: denominator <=1e-12'
            per.loc[i,'outcome']='improved' if delta>u.TOL else 'worsened' if delta<-u.TOL else 'tied'
        for candidate,reference in comparisons:
            for vid in pivot.index:
                denominator=float(pivot.loc[vid,reference]); delta=denominator-float(pivot.loc[vid,candidate])
                paired.append({'video_id':vid,'budget':B,'candidate':candidate,'reference':reference,'absolute_improvement':delta,
                    'percentage_improvement':100*delta/denominator if denominator>u.TOL else None,
                    'outcome':'improved' if delta>u.TOL else 'worsened' if delta<-u.TOL else 'tied'})
    summaries=[]
    for (B,policy),g in per.groupby(['budget','policy'],sort=True):
        assert len(g)==40
        def mean(c):
            data=g[c].dropna(); return float(data.mean()) if len(data) else None
        summaries.append({'budget':int(B),'policy':policy,'videos':40,'mean_output_error':mean('mean_output_error'),
            'median_output_error':float(g.mean_output_error.median()),'mean_absolute_improvement_vs_periodic':mean('absolute_improvement_vs_periodic'),
            'mean_per_video_percentage_improvement':mean('percentage_improvement_vs_periodic'),'mean_defined_oracle_gap_closure':mean('oracle_gap_closure'),
            'undefined_oracle_gaps':int(g.oracle_gap_closure.isna().sum()) if B!=16 else None,
            'improved':int((g.outcome=='improved').sum()) if B!=16 else None,'tied':int((g.outcome=='tied').sum()) if B!=16 else None,
            'worsened':int((g.outcome=='worsened').sum()) if B!=16 else None,
            **{c:mean(c) for c in ['mean_forced_refreshes','mean_last_refresh_position','mean_outputs_after_budget_exhausted','mean_sum_action_regret','fraction_trajectories_with_forced_refresh']}})
    paired=pd.DataFrame(paired); aggregates=[]
    for (B,candidate,reference),g in paired.groupby(['budget','candidate','reference'],sort=True):
        percentages=g.percentage_improvement.dropna()
        aggregates.append({'budget':int(B),'candidate':candidate,'reference':reference,'mean_absolute_improvement':float(g.absolute_improvement.mean()),
            'mean_per_video_percentage_improvement':float(percentages.mean()) if len(percentages) else None,
            **{k:int((g.outcome==k).sum()) for k in ['improved','tied','worsened']}})
    variability=[]
    for B,g in trajectories[trajectories.policy=='random'].groupby('budget'):
        means=g.groupby('seed_id').mean_output_error.mean()
        variability.append({'budget':int(B),'equal_video_mean_error_by_seed':{str(int(k)):float(v) for k,v in means.items()},
            'min_seed_mean_error':float(means.min()),'max_seed_mean_error':float(means.max())})
    assert len(per)==1120
    return per,{'scope':'development-tuned, equal weight/video, random averaged within video; no CIs or significance',
                'policy_metrics':summaries,'paired_comparisons':aggregates,'random_seed_variability':variability},paired


def verify_saved(output:Path,models:dict,selected:dict) -> dict:
    traces=pd.read_csv(output/'development_policy_traces.csv',float_precision='round_trip')
    trajectories=pd.read_csv(output/'development_trajectories.csv',float_precision='round_trip')
    assert traces.trajectory_id.nunique()==len(trajectories)==2200
    summaries=trajectories.set_index('trajectory_id')
    for tid,g in traces.groupby('trajectory_id',sort=False):
        g=g.sort_values('candidate_ordinal'); m=summaries.loc[tid]
        assert len(g)==16 and g.candidate_ordinal.tolist()==list(range(16))
        assert int(g.refresh.sum())==int(m.budget) and bool(g.refresh.iloc[0])
        latest=0
        for row in g.itertuples():
            t=int(row.candidate_ordinal)
            if t: assert row.cached_before_ordinal==latest
            if row.refresh: latest=t
            assert row.cached_output_ordinal==latest
            if t:
                feasible=[q for q in [row.evaluator_Q_reuse_sum,row.evaluator_Q_refresh_sum] if pd.notna(q)]
                chosen=row.evaluator_Q_refresh_sum if row.refresh else row.evaluator_Q_reuse_sum
                assert pd.notna(chosen)
                assert abs(row.evaluator_action_regret-(chosen-min(feasible))/16)<=u.TOL
        assert abs(g.output_feature_error.mean()-m.mean_output_error)<=u.TOL
        assert abs(g.evaluator_action_regret.sum()-m.sum_action_regret)<=u.TOL
        assert abs(m.sum_action_regret-(m.mean_output_error-m.oracle_optimum_error))<=u.TOL
        assert json.loads(m.refresh_positions)==g.loc[g.refresh,'candidate_ordinal'].tolist()
    expected,aggregate,paired=summarize(trajectories)
    saved=pd.read_csv(output/'development_per_video_metrics.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(expected,saved,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    assert aggregate==json.loads((output/'development_aggregate_metrics.json').read_text())
    search=pd.read_csv(output/'threshold_search.csv',float_precision='round_trip')
    for policy in u.THRESHOLD_POLICIES:
        for B in [2,4,8]:
            g=search[(search.policy==policy)&(search.budget==B)]
            averages=g.groupby('setting_id').mean_output_error.mean()
            unique=g.drop_duplicates('setting_id').copy()
            unique['recomputed']=unique.setting_id.map(averages)
            minimum=unique.recomputed.min()
            winner=unique[unique.recomputed<=minimum+u.TOL].sort_values(['lambda','grid_order']).iloc[0]
            assert selected[policy][str(B)]['setting_id']==winner.setting_id
    assert models==json.loads((output/'fitted_predictors.json').read_text())['models']
    for path in output.glob('*.json'):
        json.loads(path.read_text(),parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f'Invalid scientific number {s}')))
    assert np.isfinite(traces[['output_feature_error','evaluator_action_regret']]).all().all()
    return {'saved_traces_reproduce_cache_counters_errors_and_regret':True,'saved_metrics_and_aggregates_reproduced':True,
            'saved_threshold_results_reproduce_selection':True,'scientific_JSON_finite_and_missing_values_explicit':True}


def run(args,output:Path) -> int:
    frozen=protocol(args.budgets,args.seed)
    strict_json(output/'protocol.json',{'frozen_at_utc':now(),'protocol':frozen})
    protected=u.protected_inventory()
    metadata={'experiment':u.EXPERIMENT,'status':'running','started_at_utc':now(),'client_date':'2026-10-14',
        'input_directory':relative_path(args.input),'output_directory':relative_path(output),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_code_sha256':{name:sha256_file(ROOT/name) for name in SOURCES},'protected_original_sha256':protected,
        'actual_encoder_executions':0,'video_downloads':0,'video_decodes':0,'GPU_latency_measurements':0,
        'environment':{'python':platform.python_version(),'os':platform.platform(),'execution':'CPU offline NumPy/Pandas saved-artifact replay',
            'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','matplotlib','torch','torchvision','av','Pillow']}}}
    strict_json(output/'metadata.json',metadata)
    software=run_checks(); print(f"Focused software checks PASS: {len(software['checks'])}",flush=True)
    manifest_path=args.input/'split_manifest.json'
    if not manifest_path.exists(): raise FileNotFoundError(f'Required frozen manifest missing: {relative_path(manifest_path)}')
    manifest=u.read_json(manifest_path)
    split=u.make_development_split(manifest,args.seed)
    strict_json(output/'development_split.json',{'frozen_at_utc':now(),**split,'original_manifest_sha256':sha256_file(manifest_path)})
    train,dev,identities=u.load_training_artifacts(args.input,manifest,split)
    drift,values,dp_checks=u.training_tables(train,args.budgets)
    assert len(drift)==9600 and set(drift.source_video_id)=={v.cheap.source_video_id for v in train}
    csv(output/'training_drift_pairs.csv',drift); csv(output/'training_action_values.csv',values)
    for name,table in [('training_drift_pairs.csv',drift),('training_action_values.csv',values)]:
        pd.testing.assert_frame_equal(table,pd.read_csv(output/name,float_precision='round_trip'),check_dtype=False,check_exact=True)
    models=u.fit_predictors(drift,values,args.budgets)
    strict_json(output/'fitted_predictors.json',{'fitted_at_utc':now(),'training_source_ids':sorted(v.cheap.source_video_id for v in train),
        'training_videos':80,'models':models,'action_state_counts':dp_checks['counts_by_budget']})
    fixed_hash=sha256_file(output/'fitted_predictors.json')
    grids=u.threshold_grids(drift,models)
    strict_json(output/'threshold_grid.json',{'built_from_training_only_at_utc':now(),'settings':grids})
    metadata.update(input_identities=identities,training_videos=80,development_videos=40,original_validation_test_videos_used=0,
                    drift_training_pairs=len(drift),action_value_training_states=len(values),action_state_counts=dp_checks['counts_by_budget'],
                    models_frozen_before_development_replay_at_utc=now(),random_seeds=list(range(args.seed,args.seed+10)))
    strict_json(output/'metadata.json',metadata)
    print(f'Inputs PASS:80 training /40 development /0 original held-out videos;9,600 drift pairs;{len(values):,} action states.',flush=True)
    oracles={v.entry['video_id']:u.ActionOracle(u.distance_matrix(v.references)) for v in dev}
    selected,search=settings_search(dev,oracles,models,grids,args.budgets,output)
    assert sha256_file(output/'fitted_predictors.json')==fixed_hash
    traces,trajectories,dev_dp=replay(dev,oracles,models,selected,args.budgets,args.seed)
    per,aggregate,paired=summarize(trajectories)
    for name,table in [('development_policy_traces.csv',traces),('development_trajectories.csv',trajectories),
                       ('development_per_video_metrics.csv',per),('development_pairwise_comparisons.csv',paired)]: csv(output/name,table)
    strict_json(output/'development_aggregate_metrics.json',aggregate)
    strict_json(output/'oracle_verification.json',{'training':dp_checks,'development':dev_dp})
    assert sha256_file(output/'fitted_predictors.json')==fixed_hash
    saved_checks=verify_saved(output,models,selected)
    assert protected==u.protected_inventory()
    release=subprocess.run([sys.executable,str(ROOT/'validate_release.py')],cwd=ROOT,capture_output=True,text=True)
    assert release.returncode==0,release.stdout+release.stderr
    verification={'passed':True,'verified_at_utc':now(),'software':software,'training_videos':80,'development_videos':40,
        'class_balance_4_train_2_development_per_class':True,'source_and_file_hashes_disjoint':True,'original_validation_test_artifacts_loaded':0,
        'drift_pairs':9600,'action_value_states':len(values),'equal_supervised_state_counts_per_training_video':dp_checks['counts_by_budget'],
        'all_360_initial_DPs_match_existing_exhaustive_oracle':True,'all_real_trajectories_exact_budget_latest_cache_and_reset':True,
        'all_real_trajectory_regrets_telescope_to_oracle_excess':True,'lambda_zero_matches_static_on_all_real_search_settings':True,
        'actual_models_fitted_once_training_only_and_unchanged_after_search':True,'protected_files_unchanged':len(protected),
        'release_validator':{'returncode':release.returncode,'stdout':release.stdout.strip()},**saved_checks}
    strict_json(output/'verification.json',verification)
    from budget_aware_refresh_report import create_report
    examples=create_report(output,per,trajectories,traces,aggregate,metadata)
    metadata.update(status='complete',completed_at_utc=now(),trajectory_count=len(trajectories),output_trace_rows=len(traces),
        examples=examples,protocol_sha256=sha256_file(output/'protocol.json'),development_split_sha256=sha256_file(output/'development_split.json'))
    metadata['saved_artifact_sha256']={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!='metadata.json'}
    strict_json(output/'metadata.json',metadata)
    files=sorted(p for p in output.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt')
    (output/'SHA256SUMS.txt').write_text(''.join(f'{sha256_file(p)}  {p.name}\n' for p in files),encoding='utf-8')
    print(f'Replay COMPLETE: {relative_path(output)}; 80 train /40 development;9600 drift pairs;{len(values)} action states;2200 trajectories;35200 outputs.',flush=True)
    for B in args.budgets: print(f'B={B}: '+', '.join(f"{r['policy']}={r['mean_output_error']:.6f}" for r in aggregate['policy_metrics'] if r['budget']==B),flush=True)
    return 0


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/cache_drift_v2'))
    parser.add_argument('--output',type=Path,default=Path('results/budget_aware_refresh_v2'))
    parser.add_argument('--budgets',type=int,nargs='+',default=[2,4,8]); parser.add_argument('--seed',type=int,default=20261014)
    args=parser.parse_args()
    if sorted(args.budgets)!=[2,4,8]: parser.error('Frozen experiment requires budgets 2,4,8 exactly once')
    args.budgets=sorted(args.budgets); output=fresh_directory(args.output)
    try: return run(args,output)
    except Exception as exc:
        path=output/'metadata.json'; metadata=u.read_json(path) if path.exists() else {}
        metadata.update(status='incomplete',failed_at_utc=now(),blocker=f'{type(exc).__name__}: {exc}')
        strict_json(path,metadata)
        print(f'Experiment INCOMPLETE: {exc}; preserved {relative_path(output)}',file=sys.stderr)
        raise


if __name__=='__main__': raise SystemExit(main())
