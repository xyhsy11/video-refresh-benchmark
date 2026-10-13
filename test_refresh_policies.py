"""Isolated synthetic correctness fixtures; never experimental evidence."""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from analyze_cache_drift import TARGET, focused_checks
from collect_cache_drift import AGE_INPUTS, MOTION_INPUTS
from refresh_policy_utils import (CheapVideo, ReplayVideo, SCORED, distance_matrix, evaluate_plan,
    exact_oracle, fit_models, load_development, make_training_pairs, oracle_plan, pair_inputs, plan_policy,
    random_schedule, threshold_grids, verify_plan)


def fixture(n: int = 16) -> ReplayVideo:
    motion=tuple({"decoded_frame_index":i,"mv_available":bool(i%3),
                  "mean_magnitude_px":float(i%7) if i%3 else 0.0,
                  "p90_magnitude_px":float(2*(i%7)) if i%3 else 0.0} for i in range(2*n-1))
    rng=np.random.default_rng(20261013)
    features=rng.uniform(0.01,1,size=(n,512)).astype(np.float32)
    cheap=CheapVideo("synthetic_source",tuple(range(0,2*n,2)),tuple(i*0.5 for i in range(n)),motion)
    return ReplayVideo({"video_id":"software_fixture","experiment_split":"train"},cheap,features)


def run_focused_checks() -> dict:
    checks=focused_checks()
    video=fixture()
    pairs=make_training_pairs([video])
    assert len(pairs)==120 and set(zip(pairs.cached_candidate_ordinal,pairs.candidate_ordinal))=={(r,t) for t in range(1,16) for r in range(t)}
    checks["all_120_training_cache_histories_without_duplicates"] = True
    models=fit_models(pairs)
    grids=threshold_grids(pairs,models)
    matrix=distance_matrix(video.references)
    for budget in [2,4,8]:
        for policy in ["periodic","random",*SCORED]:
            settings=[None] if policy not in SCORED else [grids[policy][0],grids[policy][-1],grids[policy][len(grids[policy])//2]]
            for setting in settings:
                plan=plan_policy(video.cheap,policy,budget,setting,models.get(policy),20261013)
                rows,metrics=evaluate_plan(matrix,plan)
                verify_plan(video.cheap,plan,rows,budget)
                assert metrics["simulated_refreshes"]==budget and metrics["feature_outputs"]==16 and metrics["reuses"]==16-budget
    checks["mandatory_first_exact_budget_latest_cache_outputs_and_interval_reset"] = True
    dense=plan_policy(video.cheap,"dense",16)
    assert evaluate_plan(matrix,dense)[1]["max_candidate_error"]<1e-12
    checks["dense_and_all_refreshed_outputs_have_zero_error"] = True
    periodic=plan_policy(video.cheap,"periodic",4)
    assert [r["candidate_ordinal"] for r in periodic if r["refresh"]]==[0,4,8,12]
    checks["periodic_B4_schedule_reproduced"] = True
    for budget in [2,4,8]:
        wait=plan_policy(video.cheap,"age_only",budget,{"kind":"wait"},models["age_only"])
        always=plan_policy(video.cheap,"age_only",budget,{"kind":"always"},models["age_only"])
        assert [r["candidate_ordinal"] for r in wait if r["refresh"]]==[0,*range(17-budget,16)]
        assert sum(r["forced_refresh"] for r in wait)==budget-1
        assert [r["candidate_ordinal"] for r in always if r["refresh"]]==list(range(budget))
        assert all(not r["refresh"] for r in always[budget:])
    checks["budget_exhaustion_and_forced_final_refresh_exact_rule"] = True
    schedules=[random_schedule("synthetic_source",16,4,s) for s in range(20261013,20261023)]
    assert schedules==[random_schedule("synthetic_source",16,4,s) for s in range(20261013,20261023)]
    assert len({tuple(s) for s in schedules})>1
    for schedule in schedules:
        assert schedule[0]==0 and len(schedule)==len(set(schedule))==4
    checks["all_ten_random_seeds_reproducible_with_stable_video_derivation"] = True
    cutoff=video.cheap.indices[8]
    changed=replace(video.cheap,motion=tuple({**f,"mean_magnitude_px":1e9,"p90_magnitude_px":1e9} if f["decoded_frame_index"]>cutoff else f for f in video.cheap.motion))
    changed_rgb=video.references[::-1].copy()
    for policy in SCORED:
        setting=grids[policy][len(grids[policy])//2]
        plan=plan_policy(video.cheap,policy,4,setting,models.get(policy))
        future=plan_policy(changed,policy,4,setting,models.get(policy))
        assert plan[:9]==future[:9]
        # An alternate evaluator cannot provide features to the decision API.
        rows_a,_=evaluate_plan(matrix,plan)
        rows_b,_=evaluate_plan(distance_matrix(changed_rgb),plan)
        repeat=plan_policy(video.cheap,policy,4,setting,models.get(policy))
        assert repeat==plan
        assert all(a["score_raw"]==b["score_raw"] and a["refresh"]==b["refresh"] for a,b in zip(rows_a,rows_b))
    assert "references" not in inspect.signature(plan_policy).parameters
    checks["future_motion_cannot_change_prior_scores_or_decisions"] = True
    checks["fixed_learned_scores_and_schedules_invariant_to_evaluator_RGB"] = True
    small=fixture(6)
    small_matrix=distance_matrix(small.references)
    for budget in [2,3,4]:
        oracle=exact_oracle(small_matrix,budget)
        independent=[]
        for rest in combinations(range(1,6),budget-1):
            schedule=(0,*rest)
            errors=[]
            for t in range(6):
                r=max(i for i in schedule if i<=t)
                a,b=small.references[t].astype(float),small.references[r].astype(float)
                errors.append(1-float(np.clip(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
            independent.append((float(np.mean(errors)),list(schedule)))
        minimum=min(m for m,_ in independent)
        chosen=next(s for m,s in independent if m<=minimum+1e-12)
        assert abs(oracle["mean_error"]-minimum)<1e-12 and oracle["positions"]==chosen
        plan=oracle_plan(small.cheap,oracle["positions"])
        rows,_=evaluate_plan(small_matrix,plan)
        verify_plan(small.cheap,plan,rows,budget)
    constant=np.ones((6,512),dtype=np.float32)
    assert exact_oracle(distance_matrix(constant),3)["positions"]==[0,1,2]
    assert exact_oracle(matrix,4)["enumerated_schedules"]==455
    assert exact_oracle(matrix,8)["enumerated_schedules"]==6435
    checks["exact_oracle_matches_independent_all_schedule_enumeration_and_lex_ties"] = True
    zero_motion=replace(video.cheap,motion=tuple({**f,"mv_available":False,"mean_magnitude_px":0.0,"p90_magnitude_px":0.0} for f in video.cheap.motion))
    observed_zero=replace(zero_motion,motion=tuple({**f,"mv_available":True} for f in zero_motion.motion))
    assert pair_inputs(zero_motion,0,4)["interval_mv_available_fraction"]==0
    assert pair_inputs(observed_zero,0,4)["interval_mv_available_fraction"]==1
    constant_pairs=make_training_pairs([replace(video,cheap=zero_motion)])
    constant_models=fit_models(constant_pairs)
    assert all(s==1 for s in constant_models["motion_only"]["scales"])
    for policy in SCORED:
        plan=plan_policy(zero_motion,policy,4,{"kind":"threshold","value":0.1},constant_models.get(policy))
        assert sum(r["refresh"] for r in plan)==4
    checks["all_missing_motion_valid_distinct_from_observed_zero_and_constant_inputs_safe"] = True
    for name,state in models.items():
        assert set(state["columns"]).issubset(set(AGE_INPUTS+MOTION_INPUTS))
        assert TARGET not in state["columns"] and state["training_rows"]==120
    assert models==fit_models(pairs.copy())
    checks["three_models_fit_same_training_pairs_only_and_input_whitelist"] = True
    checks["threshold_grids_training_quantiles_valid_JSON_extremes"] = all(
        grid[0]["kind"]=="always" and grid[-1]["kind"]=="wait" and
        grid[0]["value"] is None and grid[-1]["value"] is None and
        len({s["value"] for s in grid[1:-1]})==len(grid)-2 for grid in grids.values())
    entry={"video_id":"software_missing","source_video_id":"software_source","experiment_split":"train"}
    fake_json={"split_manifest.json":{"status":"frozen","videos":[entry]},
               "collection_metadata.json":{"status":"collected","split_manifest_sha256":"fixture"},
               "collection_progress.json":{"videos":[{"entry":entry,"artifact_sha256":{"features/software_missing.npz":"expected"}}]}}
    with patch("refresh_policy_utils.read_json",side_effect=lambda path:fake_json[path.name]), \
         patch("refresh_policy_utils.validate_split"), patch("refresh_policy_utils.sha256_file",return_value="fixture"):
        try:
            load_development(Path("results/__isolated_software_missing_fixture__"))
        except FileNotFoundError as exc:
            assert "features/software_missing.npz" in str(exc)
        else:
            raise AssertionError("Missing artifact was silently accepted")
        with patch.object(Path,"exists",return_value=True):
            try:
                load_development(Path("results/__isolated_software_missing_fixture__"))
            except ValueError as exc:
                assert "features/software_missing.npz" in str(exc) and "hash mismatch" in str(exc)
            else:
                raise AssertionError("Invalid artifact was silently accepted")
    checks["missing_and_corrupt_artifacts_report_affected_file_and_stop"] = True
    return {"passed":all(checks.values()),"scope":"synthetic software fixtures only; not experimental evidence","checks":checks}


if __name__=="__main__":
    result=run_focused_checks()
    assert result["passed"]
    print(f"Focused replay software checks PASS: {len(result['checks'])}")
