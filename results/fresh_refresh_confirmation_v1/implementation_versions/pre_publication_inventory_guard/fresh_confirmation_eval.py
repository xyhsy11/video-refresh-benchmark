"""Dedicated frozen evaluation: no fitting, threshold grids or selection."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import budget_aware_refresh_utils as u
from refresh_policy_utils import CheapVideo, ReplayVideo
from fresh_confirmation_data import strict_json
from research_utils import now, sha256_file


def no_fitting():
    stack=ExitStack()
    for name in ['fit_predictors','fit_action_ridge','fit_predict','threshold_grids']:
        stack.enter_context(patch.object(u,name,side_effect=AssertionError('Frozen evaluation must not fit or select thresholds')))
    stack.enter_context(patch('analyze_cache_drift.fit_predict',side_effect=AssertionError('No fitting allowed')))
    stack.enter_context(patch('replay_budget_aware_policies.settings_search',side_effect=AssertionError('No threshold selection allowed')))
    return stack


def load_references(entries:list,directory:Path,hashes:dict) -> list:
    result=[]
    for entry in sorted(entries,key=lambda e:e['source_video_id']):
        vid=entry['video_id']; names=[f'features/{vid}.npz',f'features/{vid}.json',f'motion_frames/{vid}.csv']
        for name in names:
            if not (directory/name).exists(): raise FileNotFoundError(str(directory/name))
            assert sha256_file(directory/name)==hashes[name],name
        side=json.loads((directory/names[1]).read_text(encoding='utf-8'))
        assert side['entry']=={k:entry[k] for k in side['entry']}
        with np.load(directory/names[0],allow_pickle=False) as data:
            features=data['features'].copy(); indices=data['frame_indices'].tolist(); stamps=data['timestamps_seconds'].tolist()
        assert features.shape==(16,512) and features.dtype==np.float32 and np.isfinite(features).all()
        assert np.min(np.linalg.norm(features.astype(np.float64),axis=1))>u.TOL
        assert indices==entry['candidate_indices']==[c['decoded_frame_index'] for c in side['candidates']]
        assert len(indices)==16 and indices==sorted(set(indices)) and indices[0]==0 and indices[-1]==entry['decoded_frame_count']-1
        assert stamps==entry['candidate_timestamps_seconds']==[c['timestamp_seconds'] for c in side['candidates']]
        assert np.isfinite(stamps).all() and (np.diff(stamps)>0).all()
        frames=pd.read_csv(directory/names[2],float_precision='round_trip')
        assert len(frames)==entry['decoded_frame_count'] and frames.decoded_frame_index.tolist()==list(range(len(frames)))
        assert set(frames.video_id)=={vid} and frames.frame_type.value_counts().to_dict()=={k:v for k,v in entry['frame_type_counts'].items() if v}
        assert (frames.mv_available==(frames.vector_count>0)).all()
        counts=frames[['vector_count','past_reference_vectors','future_reference_vectors','other_reference_vectors']]
        assert np.isfinite(counts).all().all() and (counts>=0).all().all() and (counts%1==0).all().all()
        assert (counts.iloc[:,1:].sum(axis=1)==frames.vector_count).all()
        mag=frames[['mean_magnitude_px','p90_magnitude_px']]
        assert np.isfinite(mag).all().all() and (mag>=0).all().all() and (mag[~frames.mv_available]==0).all().all()
        features.setflags(write=False)
        result.append(ReplayVideo(entry,CheapVideo(entry['source_video_id'],tuple(indices),tuple(stamps),tuple(frames.to_dict('records'))),features))
    return result


def replay(videos:list,models:dict,settings:dict,protocol:dict) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    traces=[]; trajectories=[]; oracle_checks=[]; frozen=json.dumps({'models':models,'settings':settings},sort_keys=True,allow_nan=False)
    policies=protocol['policies']; budgets=protocol['budgets']; seeds=list(range(protocol['sampling_seed'],protocol['sampling_seed']+10))
    with no_fitting():
        for index,v in enumerate(videos,1):
            oracle=u.ActionOracle(u.distance_matrix(v.references)); vid=v.entry['video_id']
            for B in [*budgets,16]:
                if B!=16:
                    exhaustive=u.exact_oracle(oracle.d,B)
                    initial=(oracle.d[0,0]+oracle.value(1,0,B-1))/16
                    assert abs(initial-exhaustive['exact_minimum_error'])<=u.TOL
                    oracle_checks.append({'video_id':vid,'budget':B,'DP_optimum':float(initial),'exhaustive_optimum':exhaustive['exact_minimum_error']})
                for policy in ['dense'] if B==16 else policies:
                    setting=settings[policy][str(B)] if policy in settings else None
                    for seed in seeds if policy=='random' else [None]:
                        if policy=='oracle': plan=u.plan_oracle(v.cheap,B,oracle.d)
                        elif policy=='privileged_actual_drift': plan=u.plan_privileged(v.cheap,B,oracle.d,setting)
                        else: plan=u.plan_deployable(v.cheap,policy,B,models,setting,seed)
                        evaluated,metrics=u.evaluate(v.cheap,plan,oracle,B)
                        if B!=16: assert metrics['mean_output_error']>=exhaustive['exact_minimum_error']-u.TOL
                        else: assert metrics['max_output_error']<=u.TOL
                        identity={'trajectory_id':f'{vid}:{policy}:B{B}:seed{seed if seed is not None else "none"}',
                            'video_id':vid,'source_video_id':v.cheap.source_video_id,'label':v.entry['label'],'policy':policy,'budget':B,
                            'seed_id':seed if seed is not None else -1,'setting_id':setting['setting_id'] if setting else 'not_applicable',
                            'privileged':policy in ['oracle','privileged_actual_drift']}
                        traces.extend({**identity,'decoded_frame_index':v.cheap.indices[row['candidate_ordinal']],
                                       'timestamp_seconds':v.cheap.timestamps[row['candidate_ordinal']],**row} for row in evaluated)
                        trajectories.append({**identity,**{k:value for k,value in metrics.items() if k!='refresh_positions'},
                                             'refresh_positions':json.dumps(metrics['refresh_positions'])})
            print(f'Frozen replay completed {index}/{len(videos)} videos (no fitting or encoder calls).',flush=True)
    expected_per_video=1+len(budgets)*(len(policies)-1+len(seeds))
    assert len(trajectories)==expected_per_video*len(videos) and len(traces)==16*len(trajectories)
    assert frozen==json.dumps({'models':models,'settings':settings},sort_keys=True,allow_nan=False)
    return pd.DataFrame(traces),pd.DataFrame(trajectories),{'initial_oracle_checks':oracle_checks,'actual_encoder_calls':0,'fit_or_selection_calls':0,
        'expected_trajectories':expected_per_video*len(videos),'expected_trace_rows':16*expected_per_video*len(videos)}


def per_video(trajectories:pd.DataFrame,n:int,protocol:dict) -> pd.DataFrame:
    rows=[]
    for (vid,B,policy),g in trajectories.groupby(['video_id','budget','policy'],sort=True):
        assert len(g)==(10 if policy=='random' else 1)
        assert (g.simulated_refreshes==B).all() and (g.cache_reuses==16-B).all() and (g.feature_outputs==16).all()
        if policy=='random': assert set(g.seed_id)==set(range(protocol['sampling_seed'],protocol['sampling_seed']+10))
        rows.append({'video_id':vid,'source_video_id':g.source_video_id.iloc[0],'label':g.label.iloc[0],'budget':int(B),'policy':policy,
            'privileged':bool(g.privileged.iloc[0]),'trajectory_count':len(g),'mean_output_error':float(g.mean_output_error.mean()),
            'max_output_error_mean_over_seeds':float(g.max_output_error.mean()),'median_output_error_over_seeds':float(g.mean_output_error.median()),
            'simulated_refreshes':int(B),'cache_reuses':16-int(B),'feature_outputs':16,
            'mean_forced_refreshes':float(g.forced_refreshes.mean()),'mean_last_refresh_position':float(g.last_refresh_position.mean()),
            'mean_outputs_after_budget_exhausted':float(g.outputs_after_budget_exhausted.mean()),'mean_sum_action_regret':float(g.sum_action_regret.mean()),
            'fraction_trajectories_forced':float((g.forced_refreshes>0).mean()),'random_seed_error_std_population':float(g.mean_output_error.std(ddof=0)) if policy=='random' else None,
            'random_seed_error_min':float(g.mean_output_error.min()) if policy=='random' else None,'random_seed_error_max':float(g.mean_output_error.max()) if policy=='random' else None,
            'refresh_positions_by_seed':json.dumps({str(s):json.loads(p) for s,p in zip(g.seed_id,g.refresh_positions)})})
    table=pd.DataFrame(rows)
    for B in protocol['budgets']:
        pivot=table[table.budget==B].pivot(index='video_id',columns='policy',values='mean_output_error')
        for i in table.index[table.budget==B]:
            row=table.loc[i]; reference=float(pivot.loc[row.video_id,'periodic']); optimum=float(pivot.loc[row.video_id,'oracle'])
            delta=reference-row.mean_output_error; gap=reference-optimum
            table.loc[i,'absolute_improvement_vs_periodic']=delta
            table.loc[i,'percentage_improvement_vs_periodic']=100*delta/reference if reference>u.TOL else np.nan
            table.loc[i,'percentage_reason']='defined' if reference>u.TOL else 'undefined: periodic error <=1e-12'
            table.loc[i,'oracle_headroom']=gap
            table.loc[i,'oracle_gap_closure']=delta/gap if gap>u.TOL else np.nan
            table.loc[i,'oracle_gap_reason']='defined' if gap>u.TOL else 'undefined: oracle headroom <=1e-12'
            table.loc[i,'outcome']='improved' if delta>u.TOL else 'worsened' if delta<-u.TOL else 'tied'
    assert len(table)==n*(len(protocol['budgets'])*len(protocol['policies'])+1)
    return table


def comparisons(protocol:dict) -> list:
    return [(B,'value_age_motion_budget',reference,'primary' if B==4 and reference=='periodic' else 'secondary_marginal_unadjusted')
        for B in protocol['budgets'] for reference in ['periodic','paced_age_plus_motion','value_age_budget']]+[
        (B,'value_age_budget','periodic','secondary_marginal_unadjusted') for B in protocol['budgets']]


def statistics(table:pd.DataFrame,trajectories:pd.DataFrame,protocol:dict) -> tuple[dict,pd.DataFrame,dict,pd.DataFrame]:
    n=table.video_id.nunique(); rng=np.random.Generator(np.random.PCG64(protocol['uncertainty']['seed']))
    indices=rng.integers(0,n,size=(protocol['uncertainty']['resamples'],n),dtype=np.int64)
    resampling_sha=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
    paired_rows=[]; intervals=[]; metrics=[]; class_rows=[]
    for B,candidate,reference,kind in comparisons(protocol):
        pivot=table[table.budget==B].pivot(index='source_video_id',columns='policy',values='mean_output_error').sort_index()
        assert len(pivot)==n
        delta=pivot[reference].to_numpy()-pivot[candidate].to_numpy()
        sampled=delta[indices].mean(axis=1)
        low,high=np.percentile(sampled,[2.5,97.5],method='linear')
        percentage=np.divide(100*delta,pivot[reference].to_numpy(),out=np.full(n,np.nan),where=pivot[reference].to_numpy()>u.TOL)
        for i,sid in enumerate(pivot.index):
            paired_rows.append({'source_video_id':sid,'budget':B,'candidate':candidate,'reference':reference,'comparison_type':kind,
                'absolute_improvement':float(delta[i]),'percentage_improvement':float(percentage[i]) if np.isfinite(percentage[i]) else None,
                'percentage_reason':'defined' if np.isfinite(percentage[i]) else 'undefined: reference error<=1e-12',
                'outcome':'improved' if delta[i]>u.TOL else 'worsened' if delta[i]<-u.TOL else 'tied'})
        intervals.append({'budget':B,'candidate':candidate,'reference':reference,'comparison_type':kind,'videos':n,
            'mean_paired_absolute_improvement':float(delta.mean()),'median_paired_absolute_improvement':float(np.median(delta)),
            'CI95_lower':float(low),'CI95_upper':float(high),'interpretation':'lower error evidence' if low>0 else 'worse error evidence' if high<0 else 'inconclusive; interval includes zero',
            'improved':int((delta>u.TOL).sum()),'tied':int((abs(delta)<=u.TOL).sum()),'worsened':int((delta<-u.TOL).sum()),
            'mean_per_video_percentage_improvement':float(np.nanmean(percentage)) if np.isfinite(percentage).any() else None,
            'percentage_of_aggregate_means_separate':float(100*delta.mean()/pivot[reference].mean()) if pivot[reference].mean()>u.TOL else None,
            'resampling_indices_sha256':resampling_sha})
    for (B,policy),g in table.groupby(['budget','policy'],sort=True):
        def mean(column):
            valid=g[column].dropna(); return float(valid.mean()) if len(valid) else None
        metrics.append({'budget':int(B),'policy':policy,'videos':n,'mean_output_error':mean('mean_output_error'),'median_per_video_output_error':float(g.mean_output_error.median()),
            'mean_absolute_improvement_vs_periodic':mean('absolute_improvement_vs_periodic'),'mean_per_video_percentage_improvement':mean('percentage_improvement_vs_periodic'),
            'mean_oracle_headroom':mean('oracle_headroom'),'mean_defined_oracle_gap_closure':mean('oracle_gap_closure'),
            'undefined_oracle_gaps':int(g.oracle_gap_closure.isna().sum()) if B!=16 else None,
            **{key:int((g.outcome==key).sum()) if B!=16 else None for key in ['improved','tied','worsened']},
            **{key:mean(key) for key in ['mean_forced_refreshes','mean_last_refresh_position','mean_outputs_after_budget_exhausted','mean_sum_action_regret','fraction_trajectories_forced']}})
    for (label,B,policy),g in table.groupby(['label','budget','policy'],sort=True):
        class_rows.append({'label':label,'budget':int(B),'policy':policy,'videos':len(g),'mean_output_error':float(g.mean_output_error.mean()),
            'mean_absolute_improvement_vs_periodic':float(g.absolute_improvement_vs_periodic.mean()) if B!=16 else None,
            'improved':int((g.outcome=='improved').sum()) if B!=16 else None,'worsened':int((g.outcome=='worsened').sum()) if B!=16 else None})
    random=[]
    for B,g in trajectories[trajectories.policy=='random'].groupby('budget'):
        seed_means=g.groupby('seed_id').mean_output_error.mean()
        random.append({'budget':int(B),'equal_video_mean_error_by_seed':{str(int(k)):float(v) for k,v in seed_means.items()},
            'min_seed_mean':float(seed_means.min()),'max_seed_mean':float(seed_means.max())})
    primary=table[(table.budget==4)&(table.policy=='value_age_motion_budget')]
    positives=primary.absolute_improvement_vs_periodic[primary.absolute_improvement_vs_periodic>0].sort_values(ascending=False)
    net=float(primary.absolute_improvement_vs_periodic.sum())
    concentration={'net_summed_gain':net,'top_two_positive_gains':float(positives.head(2).sum()),
        'top_two_positive_gains_percent_of_positive_net':100*float(positives.head(2).sum())/net if net>u.TOL else None,
        'reason':'defined: net sum positive; can exceed100% when losses offset gains' if net>u.TOL else 'undefined: net gain nonpositive',
        'primary_classes_improved_in_mean':int((pd.DataFrame(class_rows).query('budget==4 and policy=="value_age_motion_budget"').mean_absolute_improvement_vs_periodic>u.TOL).sum())}
    aggregate={'scope':'held-out new source videos from fixed existing classes; equal video weight; random averaged within video',
        'policy_metrics':metrics,'random_seed_variability':random,'concentration':concentration}
    bootstrap={'RNG':'numpy.random.Generator(PCG64)','seed':protocol['uncertainty']['seed'],'resamples':protocol['uncertainty']['resamples'],
        'percentile_method':'numpy.percentile(method="linear")','resampling_unit':'whole source videos, all paired methods together',
        'source_order':'lexicographically sorted source_video_id','shared_index_shape':list(indices.shape),'shared_indices_sha256':resampling_sha,
        'limits':'conditional on fixed models and sampled classes; no training uncertainty; secondary intervals marginal/unadjusted',
        'comparisons':intervals}
    return aggregate,pd.DataFrame(paired_rows),bootstrap,pd.DataFrame(class_rows)


def compatibility(frozen_run:Path,protocol:dict,models:dict,settings:dict,output:Path) -> dict:
    split=json.loads((frozen_run/'development_split.json').read_text(encoding='utf-8')); old=json.loads((frozen_run/'metadata.json').read_text(encoding='utf-8'))
    entries=[e for e in split['videos'] if e['development_role']=='development']
    videos=load_references(entries,Path('results/cache_drift_v2'),old['input_identities']['artifact_sha256'])
    with no_fitting():
        traces,trajectories,checks=replay(videos,models,settings,protocol)
        per=per_video(trajectories,40,protocol)
    published=pd.read_csv(frozen_run/'development_per_video_metrics.csv',float_precision='round_trip')
    join=per.merge(published,on=['video_id','budget','policy'],suffixes=('_new','_published'),validate='one_to_one')
    assert len(join)==1120
    maximum=float(abs(join.mean_output_error_new-join.mean_output_error_published).max())
    assert maximum<=u.TOL
    original=pd.read_csv(frozen_run/'development_trajectories.csv',float_precision='round_trip')
    merged=trajectories.merge(original,on=['trajectory_id'],suffixes=('_new','_published'),validate='one_to_one')
    assert len(merged)==2200 and merged.refresh_positions_new.tolist()==merged.refresh_positions_published.tolist()
    result={'passed':True,'checked_at_utc':now(),'videos':40,'policy_budget_rows':1120,'trajectories':2200,
        'maximum_mean_error_difference':maximum,'all_frozen_refresh_schedules_reproduced':True,'fit_or_selection_calls':0,'encoder_calls':0}
    strict_json(output/'development_compatibility.json',result); return result


def focused_checks(models:dict,settings:dict,protocol:dict) -> dict:
    from test_refresh_policies import fixture
    from itertools import combinations
    v=fixture(); checks={}; state=json.dumps({'models':models,'settings':settings},sort_keys=True)
    with no_fitting():
        oracle=u.ActionOracle(u.distance_matrix(v.references))
        for B in protocol['budgets']:
            for policy in protocol['policies']:
                setting=settings[policy][str(B)] if policy in settings else None
                plans=[u.plan_deployable(v.cheap,policy,B,models,setting,s) for s in range(protocol['sampling_seed'],protocol['sampling_seed']+10)] if policy=='random' else [u.plan_oracle(v.cheap,B,oracle.d) if policy=='oracle' else u.plan_privileged(v.cheap,B,oracle.d,setting) if policy=='privileged_actual_drift' else u.plan_deployable(v.cheap,policy,B,models,setting)]
                for plan in plans: u.evaluate(v.cheap,plan,oracle,B)
        cutoff=v.cheap.indices[8]
        future=replace(v.cheap,motion=tuple({**f,'mean_magnitude_px':1e8,'p90_magnitude_px':1e9,'mv_available':True} if f['decoded_frame_index']>cutoff else f for f in v.cheap.motion))
        for B in protocol['budgets']:
            for policy in ['age_only','age_plus_motion','paced_age_plus_motion',*u.VALUE_COLUMNS]:
                setting=settings[policy][str(B)] if policy in settings else None
                original=u.plan_deployable(v.cheap,policy,B,models,setting)
                assert original[:9]==u.plan_deployable(future,policy,B,models,setting)[:9]
                u.evaluate(v.cheap,original,u.ActionOracle(u.distance_matrix(v.references[::-1])),B)
                assert original==u.plan_deployable(v.cheap,policy,B,models,setting)
        missing=replace(v.cheap,motion=tuple({**f,'mv_available':False,'mean_magnitude_px':0.0,'p90_magnitude_px':0.0} for f in v.cheap.motion))
        u.evaluate(missing,u.plan_deployable(missing,'value_age_motion_budget',4,models),oracle,4)
    assert state==json.dumps({'models':models,'settings':settings},sort_keys=True)
    checks.update(exact_budgets_outputs_latest_cache_and_interval_resets=True,action_regret_identity=True,missing_motion_valid=True,
        future_motion_isolation=True,evaluator_RGB_and_targets_isolation=True,parameters_and_settings_unchanged=True,no_fit_or_threshold_selection=True)
    small=fixture(6); d=u.distance_matrix(small.references); oracle=u.ActionOracle(d)
    for B in [2,3,4]:
        for t,r,b in oracle.reachable(B):
            costs=[]
            for rest in combinations(range(t,6),b):
                cache=r; total=0
                for i in range(t,6):
                    if i in rest: cache=i
                    total+=d[i,cache]
                costs.append(total)
            assert abs(oracle.value(t,r,b)-min(costs))<=u.TOL
    checks['independent_small_DP_enumeration']=True
    return {'passed':True,'scope':'synthetic correctness only, not fresh evidence','checks':checks}
