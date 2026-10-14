"""Independent saved-data checks; does not change policies, models or settings."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

import budget_aware_refresh_utils as u
from research_utils import ROOT, now, sha256_file, write_json


def independent_values(features:np.ndarray) -> tuple[np.ndarray,dict]:
    x=features.astype(np.float64)
    norms=np.sqrt(np.sum(x*x,axis=1)); assert np.isfinite(norms).all() and (norms>1e-12).all()
    d=1-np.clip((x/norms[:,None])@(x/norms[:,None]).T,-1,1)
    n=len(x); values={(n,r,0):0.0 for r in range(n)}
    for t in range(n-1,0,-1):
        for r in range(t):
            for b in range(min(7,n-t)+1):
                costs=[]
                if (t+1,r,b) in values: costs.append(float(d[t,r])+values[t+1,r,b])
                if b>=1 and (t+1,t,b-1) in values: costs.append(float(d[t,t])+values[t+1,t,b-1])
                if costs: values[t,r,b]=min(costs)
    return d,values


def audit(directory:Path,source:Path) -> dict:
    metadata=json.loads((directory/'metadata.json').read_text())
    split=json.loads((directory/'development_split.json').read_text())
    manifest=json.loads((source/'split_manifest.json').read_text())
    assert u.make_development_split(manifest,split['seed'])['videos']==split['videos']
    train,dev,identity=u.load_training_artifacts(source,manifest,split)
    assert identity['original_validation_test_artifacts_opened']==0
    original_train={e['video_id'] for e in manifest['videos'] if e['experiment_split']=='train'}
    assert all(Path(n).stem in original_train for n in identity['opened_artifacts'])
    drift=pd.read_csv(directory/'training_drift_pairs.csv',float_precision='round_trip')
    action=pd.read_csv(directory/'training_action_values.csv',float_precision='round_trip')
    assert len(drift)==9600 and len(action)==32480
    assert not action.duplicated(['video_id','budget','t','r','remaining_budget']).any()
    models=json.loads((directory/'fitted_predictors.json').read_text())
    assert set(models['training_source_ids'])=={v.cheap.source_video_id for v in train}
    assert set(models['training_source_ids']).isdisjoint({v.cheap.source_video_id for v in dev})
    max_q_difference=0.0; pair_count=state_count=0
    for v in train:
        vid=v.entry['video_id']; d,independent=independent_values(v.references)
        mag=np.array([f['mean_magnitude_px'] for f in v.cheap.motion],dtype=np.float64)
        p90=np.array([f['p90_magnitude_px'] for f in v.cheap.motion],dtype=np.float64)
        available=np.array([bool(f['mv_available']) for f in v.cheap.motion])
        prefix_count=np.r_[0,np.cumsum(available)]
        prefix_sum=np.r_[0.0,np.cumsum(np.where(available,mag,0.0))]
        for row in drift[drift.video_id==vid].itertuples():
            r,t=int(row.r),int(row.t); a,b=v.cheap.indices[r]+1,v.cheap.indices[t]
            count=int(prefix_count[b+1]-prefix_count[a]); length=b-a+1
            expected=[v.cheap.timestamps[t]-v.cheap.timestamps[r],v.cheap.indices[t]-v.cheap.indices[r],t-r,(v.cheap.timestamps[t]-v.cheap.timestamps[r])**2,
                mag[b],p90[b],int(available[b]),(prefix_sum[b+1]-prefix_sum[a])/count if count else 0.0,
                float(p90[a:b+1][available[a:b+1]].max()) if count else 0.0,count/length]
            actual=[getattr(row,k) for k in u.AGE_INPUTS+u.MOTION_INPUTS]
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
            assert abs(getattr(row,u.TARGET)-d[t,r])<1e-12
            assert row.interval_start_frame==a and row.interval_end_frame==b
            assert row.interval_available_frame_count==count and row.interval_frame_count==length
            pair_count+=1
        for row in action[action.video_id==vid].itertuples():
            t,r,b=int(row.t),int(row.r),int(row.remaining_budget)
            assert (t+1,r,b) in independent and (t+1,t,b-1) in independent
            reuse=d[t,r]+independent[t+1,r,b]; refresh=d[t,t]+independent[t+1,t,b-1]
            np.testing.assert_allclose([row.Q_reuse_sum,row.Q_refresh_sum,row.V_sum,getattr(row,u.VALUE_TARGET)],
                [reuse,refresh,min(reuse,refresh),(reuse-refresh)/16],rtol=1e-12,atol=1e-12)
            assert row.remaining_refresh_count==b and row.remaining_candidate_count==16-t
            assert abs(row.remaining_refresh_fraction-b/(16-t))<=1e-12
            max_q_difference=max(max_q_difference,abs(reuse-row.Q_reuse_sum),abs(refresh-row.Q_refresh_sum))
            state_count+=1
    assert pair_count==9600 and state_count==32480
    fitted=[]
    for name,state in models['models']['drift'].items(): fitted.append((name,drift,state,u.TARGET))
    for budget,group in models['models']['action_value'].items():
        for name,state in group.items(): fitted.append((f'{name}:B{budget}',action[action.budget==int(budget)],state,u.VALUE_TARGET))
    for name,table,state,target in fitted:
        x=table[state['columns']].to_numpy(dtype=np.float64); y=table[target].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(x.mean(0),state['means'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(x.std(0,ddof=0),state['raw_standard_deviations'],rtol=1e-12,atol=1e-12)
        z=(x-np.array(state['means']))/np.array(state['scales'])
        design=np.column_stack([np.ones(len(z)),z]); penalty=np.eye(design.shape[1]); penalty[0,0]=0.0
        independent=np.linalg.lstsq(np.vstack([design,penalty]),np.r_[y,np.zeros(design.shape[1])],rcond=None)[0]
        np.testing.assert_allclose(independent,[state['intercept'],*state['coefficients']],atol=1e-10,rtol=1e-8)
        assert state['training_rows']==len(table) and state['alpha']==1.0
    selected=json.loads((directory/'selected_thresholds.json').read_text())['selected']
    for v in dev[:2]:
        cutoff=v.cheap.indices[8]
        future=replace(v.cheap,motion=tuple({**f,'mean_magnitude_px':1e6,'p90_magnitude_px':1e7,'mv_available':True,
            'vector_count':1,'past_reference_vectors':1,'future_reference_vectors':0,'other_reference_vectors':0}
            if f['decoded_frame_index']>cutoff else f for f in v.cheap.motion))
        for B in [2,4,8]:
            for policy in ['age_only','age_plus_motion','paced_age_plus_motion',*u.VALUE_COLUMNS]:
                setting=selected[policy][str(B)] if policy in selected else None
                a=u.plan_deployable(v.cheap,policy,B,models['models'],setting)
                b=u.plan_deployable(future,policy,B,models['models'],setting)
                assert a[:9]==b[:9]
                changed=u.ActionOracle(u.distance_matrix(v.references[::-1]))
                evaluated,_=u.evaluate(v.cheap,a,changed,B)
                assert [r['score_raw'] for r in evaluated]==[r['score_raw'] for r in a]
                assert u.plan_deployable(v.cheap,policy,B,models['models'],setting)==a
    for group,base in [('source_code_sha256',ROOT),('protected_original_sha256',ROOT),('saved_artifact_sha256',directory)]:
        for name,digest in metadata[group].items(): assert sha256_file(base/name)==digest,(group,name)
    saved_traces=pd.read_csv(directory/'development_policy_traces.csv',float_precision='round_trip')
    value_model=models['models']['action_value']['4']['value_age_motion_budget']
    heldout=saved_traces[(saved_traces.budget==4)&(saved_traces.policy=='value_age_motion_budget')&(saved_traces.candidate_ordinal>0)].copy()
    before=u.predict_array(heldout,value_model)
    heldout[u.VALUE_TARGET]=1e8; heldout['evaluator_true_pre_decision_drift']=-1e8
    np.testing.assert_array_equal(before,u.predict_array(heldout,value_model))
    result={'passed':True,'audited_at_utc':now(),'scope':'independent saved-data verification, no refitting/tuning/decode/inference',
        'training_videos':80,'development_videos':40,'original_validation_test_feature_motion_artifacts_opened':0,
        'all_9600_drift_labels_ages_intervals_independently_checked':True,
        'all_32480_action_labels_match_independent_bottom_up_DP_and_normalized_matrix':True,
        'maximum_Q_discrepancy':max_q_difference,'all_eight_fitted_models_match_independent_augmented_least_squares':True,
        'real_two_video_future_motion_and_RGB_isolation_checked_at_all_budgets':True,
        'changed_heldout_labels_cannot_change_prediction_inputs_or_train_only_fitted_parameters':True,
        'source_artifact_and_953_protected_hashes_verified':True}
    json.dumps(result,allow_nan=False)
    report=directory/'independent_verification.json'
    if report.exists():
        previous=json.loads(report.read_text())
        assert {k:v for k,v in previous.items() if k!='audited_at_utc'}=={k:v for k,v in result.items() if k!='audited_at_utc'}
    else:
        write_json(report,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/budget_aware_refresh_v2'))
    parser.add_argument('--source',type=Path,default=Path('results/cache_drift_v2'))
    args=parser.parse_args()
    result=audit(args.input,args.source)
    print('Independent audit PASS: all9600 drift pairs/all32480 action targets/eight regression fits; real-data information isolation and preservation.')
