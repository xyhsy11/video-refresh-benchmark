"""Focused synthetic fixtures, separate from real developmental evidence."""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

import budget_aware_refresh_utils as u
from refresh_policy_utils import ReplayVideo, distance_matrix, exact_oracle
from test_refresh_policies import fixture


def run_checks() -> dict:
    checks={}
    original=fixture()
    video=replace(original,entry={**original.entry,"development_role":"train"})
    drift,values,audits=u.training_tables([video],[2,4,8])
    assert len(drift)==120
    assert not values.duplicated(["video_id","budget","t","r","remaining_budget"]).any()
    checks['unique_reachable_optional_action_states_and_120_drift_pairs']=True
    small=fixture(6); d=distance_matrix(small.references); oracle=u.ActionOracle(d)
    for B in [2,3,4]:
        for t,r,b in oracle.reachable(B):
            alternatives=[]
            for positions in combinations(range(t,6),b):
                cached=r; cost=0.0
                for i in range(t,6):
                    if i in positions: cached=i
                    cost+=d[i,cached]
                alternatives.append((t in positions,cost))
            assert abs(oracle.value(t,r,b)-min(cost for action,cost in alternatives))<1e-12
            q=oracle.q(t,r,b)
            for action,name in [(False,'reuse'),(True,'refresh')]:
                costs=[cost for a,cost in alternatives if a==action]
                assert (q[name] is None)==(not costs)
                if costs: assert abs(q[name]-min(costs))<1e-12
        initial=(d[0,0]+oracle.value(1,0,B-1))/6
        assert abs(initial-exact_oracle(d,B)['exact_minimum_error'])<1e-12
    assert oracle.value(6,0,0)==0 and oracle.value(6,0,1) is None
    checks['all_small_intermediate_Q_and_V_match_independent_schedule_enumeration']=True
    checks['terminal_infeasibility_explicit_without_infinite_values']=True
    d=distance_matrix(video.references)
    for B in [2,4,8]:
        expected=set()
        for rest in combinations(range(1,16),B-1):
            positions={0,*rest}; r=0; b=B-1
            for t in range(1,16):
                expected.add((t,r,b))
                if t in positions: r=t; b-=1
        assert expected==set(u.ActionOracle(d).reachable(B))
    checks['reachable_states_match_union_of_every_feasible_full_schedule']=True
    models=u.fit_predictors(drift,values,[2,4,8]); grids=u.threshold_grids(drift,models)
    frozen=json.dumps(models,sort_keys=True,allow_nan=False)
    for B in [2,4,8]:
        for policy in ["periodic","random",*u.VALUE_COLUMNS,*u.THRESHOLD_POLICIES]:
            settings=grids[policy] if policy in u.THRESHOLD_POLICIES else [None]
            for setting in settings:
                plan=u.plan_privileged(video.cheap,B,d,setting) if policy=='privileged_actual_drift' else u.plan_deployable(video.cheap,policy,B,models,setting,20261014)
                rows,m=u.evaluate(video.cheap,plan,u.ActionOracle(d),B)
                assert m['simulated_refreshes']==B and m['feature_outputs']==16 and m['cache_reuses']==16-B
                assert abs(m['sum_action_regret']-(m['mean_output_error']-m['oracle_optimum_error']))<1e-12
        for setting in grids['age_plus_motion']:
            a=u.plan_deployable(video.cheap,'age_plus_motion',B,models,setting)
            paced={**setting,'lambda':0.0}
            b=u.plan_deployable(video.cheap,'paced_age_plus_motion',B,models,paced)
            assert a==b
    checks['exact_budget_mandatory_first_latest_cache_interval_resets']=True
    checks['sum_action_regret_telescopes_to_mean_error_excess']=True
    checks['lambda_zero_reproduces_static_for_all_predefined_settings_and_budgets']=True
    dense=u.plan_deployable(video.cheap,'dense',16,models)
    assert u.evaluate(video.cheap,dense,u.ActionOracle(d),16)[1]['max_output_error']<1e-12
    assert [r['candidate_ordinal'] for r in u.plan_deployable(video.cheap,'periodic',4,models) if r['refresh']]==[0,4,8,12]
    checks['dense_near_zero_and_periodic_four_schedule']=True
    for B in [2,4,8]:
        always=u.plan_deployable(video.cheap,'age_only',B,models,{'kind':'always'})
        wait=u.plan_deployable(video.cheap,'age_only',B,models,{'kind':'wait'})
        assert [r['candidate_ordinal'] for r in always if r['refresh']]==list(range(B))
        assert [r['candidate_ordinal'] for r in wait if r['refresh']]==[0,*range(17-B,16)]
        assert sum(r['forced_refresh'] for r in wait)==B-1
    checks['forced_final_spending_and_exhaustion_rule']=True
    for policy in u.VALUE_COLUMNS:
        for B in [2,4,8]:
            table=values[values.budget==B]; state=models['action_value'][str(B)][policy]
            cols=state['columns']; x=table[cols].to_numpy(dtype=np.float64); y=table[u.VALUE_TARGET].to_numpy()
            np.testing.assert_allclose(state['means'],x.mean(0),atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(state['raw_standard_deviations'],x.std(0),atol=1e-12,rtol=1e-12)
            z=(x-np.array(state['means']))/np.array(state['scales'])
            design=np.column_stack([np.ones(len(z)),z]); penalty=np.eye(design.shape[1]); penalty[0,0]=0
            independent=np.linalg.solve(design.T@design+penalty,design.T@y)
            np.testing.assert_allclose([state['intercept'],*state['coefficients']],independent,atol=1e-10,rtol=1e-8)
            bad={c:1.0 for c in cols}; bad['true_drift']=0.1
            try: u.controller_score(policy,bad,state)
            except ValueError: pass
            else: raise AssertionError('Target-derived inputs accepted')
    checks['population_training_scaling_ridge_and_unpenalized_intercept_independent_check']=True
    checks['action_value_input_whitelist_rejects_evaluator_information']=True
    constant=values[values.budget==4].copy()
    constant[u.AGE_INPUTS+u.BUDGET_INPUTS+u.MOTION_INPUTS]=0.0
    state=u.fit_action_ridge(constant,u.VALUE_COLUMNS['value_age_motion_budget'])
    assert all(scale==1 for scale in state['scales']) and np.isfinite(u.predict_array(constant,state)).all()
    missing=replace(video.cheap,motion=tuple({**f,'mv_available':False,'mean_magnitude_px':0.0,'p90_magnitude_px':0.0} for f in video.cheap.motion))
    plan=u.plan_deployable(missing,'value_age_motion_budget',4,models)
    assert all(r['interval_mv_available_fraction']==0 for r in plan[1:])
    observed_zero=replace(missing,motion=tuple({**f,'mv_available':True} for f in missing.motion))
    assert u.pair_inputs(observed_zero,0,3)['interval_mv_available_fraction']==1
    checks['constant_columns_and_missing_vs_observed_zero_motion']=True
    negative={**state,'intercept':-3.0,'coefficients':[0.0]*len(state['columns'])}
    assert u.controller_score('value_age_motion_budget',{c:0.0 for c in negative['columns']},negative)==(-3.0,-3.0)
    zero={**negative,'intercept':0.0}
    zero_models=json.loads(frozen); zero_models['action_value']['4']['value_age_motion_budget']=zero
    assert [r['candidate_ordinal'] for r in u.plan_deployable(missing,'value_age_motion_budget',4,zero_models) if r['refresh']]==[0,13,14,15]
    checks['signed_advantage_unclipped_strictly_positive_threshold']=True
    cutoff=video.cheap.indices[8]
    future=replace(video.cheap,motion=tuple({**f,'mean_magnitude_px':1e8,'p90_magnitude_px':1e9} if f['decoded_frame_index']>cutoff else f for f in video.cheap.motion))
    for policy in ['age_only','age_plus_motion','paced_age_plus_motion',*u.VALUE_COLUMNS]:
        setting=grids[policy][len(grids[policy])//2] if policy in grids else None
        a=u.plan_deployable(video.cheap,policy,4,models,setting)
        b=u.plan_deployable(future,policy,4,models,setting)
        assert a[:9]==b[:9]
        altered=distance_matrix(video.references[::-1])
        evaluated,_=u.evaluate(video.cheap,a,u.ActionOracle(altered),4)
        assert [r['score_raw'] for r in evaluated]==[r['score_raw'] for r in a]
        assert u.plan_deployable(video.cheap,policy,4,models,setting)==a
    checks['future_motion_cannot_change_earlier_deployable_scores_or_actions']=True
    checks['fixed_deployable_plans_isolated_from_mutated_evaluator_RGB']=True
    heldout=values[values.budget==4].copy(); first=u.predict_array(heldout,models['action_value']['4']['value_age_motion_budget'])
    heldout[u.VALUE_TARGET]+=1e8
    assert np.array_equal(first,u.predict_array(heldout,models['action_value']['4']['value_age_motion_budget']))
    assert models==u.fit_predictors(drift,values,[2,4,8])
    assert json.dumps(models,sort_keys=True,allow_nan=False)==frozen
    checks['changed_development_targets_cannot_change_training_fit_or_predictions']=True
    synthetic={'videos':[{'video_id':f'{c}-{i}','source_video_id':f'{c}-{i}','label':f'class{c:02d}',
                         'experiment_split':'train' if i<6 else 'validation' if i<8 else 'test',
                         'file_identity':{'sha256':f'hash{c}-{i}'}} for c in range(20) for i in range(10)]}
    with patch.object(u,'validate_split'):
        split=u.make_development_split(synthetic,20261014)
        inverse=u.make_development_split({'videos':synthetic['videos'][::-1]},20261014)
    assert split==inverse and split['counts']=={'train':80,'development':40}
    assert all(e['original_experiment_split']=='train' for e in split['videos'])
    checks['deterministic_80_40_split_class_balance_excludes_original_holdouts']=True
    schedules=[u.random_positions('source',4,s) for s in range(20261014,20261024)]
    assert schedules==[u.random_positions('source',4,s) for s in range(20261014,20261024)] and len({tuple(s) for s in schedules})>1
    assert len(grids['paced_age_plus_motion'])==4*(len(grids['age_plus_motion'])-2)+2
    assert [s['lambda'] for s in grids['paced_age_plus_motion'] if s['kind']!='threshold']==[0.0,0.0]
    json.dumps({'grids':grids,'models':models},allow_nan=False)
    checks['all_random_seeds_reproducible_no_best_seed_selection']=True
    checks['fixed_training_quantile_lambda_grid_valid_finite_JSON']=True
    return {'passed':True,'scope':'synthetic software fixtures only, never experimental observations','checks':checks}


if __name__=='__main__':
    result=run_checks()
    print(f"Focused budget-aware verification PASS: {len(result['checks'])}")
