"""Independent read-only verification of committed safe-density full reports.

Does not use the analyzer's metric, aggregation or bootstrap functions. Raw
authenticated feature/head replay is separately provided by --audit-only.
Prints JSON to stdout; never modifies any input or report.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from refresh_pooling_utils import schedules


ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/safe_schedule_density_v1'
TOL = 1e-10


def near(actual,expected):
    if expected is None:
        assert pd.isna(actual), (actual,expected)
    else:
        assert abs(float(actual)-float(expected)) < 5e-13, (actual,expected)


def main():
    cohort = json.loads((OUT/'cohort_manifest.json').read_text())
    videos,classes = cohort['clips'],cohort['classes']
    ids = [v['clip'] for v in videos]
    labels = {v['clip']:classes.index(v['label']) for v in videos}
    nc = Counter(v['label'] for v in videos)
    ng = Counter(c for c,g in {(v['label'],v['group']) for v in videos})
    nclip = Counter((v['label'],v['group']) for v in videos)
    w = np.array([1/len(nc)/ng[v['label']]/nclip[v['label'],v['group']] for v in videos])
    np.testing.assert_array_equal(w,cohort['weights'])
    stage = OUT/'full'
    def csv(name):
        return pd.read_csv(stage/name,float_precision='round_trip')
    raw,methods = csv('per_video.csv'),csv('frozen_methods_per_video.csv')
    tables = {k:schedules(16,k) for k in (4,8)}
    lookup = {k:{tuple(p):i for i,p in enumerate(t['positions'])} for k,t in tables.items()}
    oracle = []
    checked = 0
    for clip,subset in raw.groupby('clip',sort=True):
        y = labels[clip]
        with np.load(stage/'clips'/(clip+'.npz'),allow_pickle=False) as saved:
            for r in subset.to_dict('records'):
                k,pool,eps = r['K'],r['pool'],r['epsilon']
                d,pred = saved[f'delta_K{k}_{pool}'],saved[f'prediction_K{k}_{pool}']
                correct,lossok = pred == y,d <= eps+TOL
                joint = correct & lossok
                expected = {'correct':correct,'CE_nondegrade':lossok,'correct_and_CE':joint,
                            'dense_agreement':pred == r['dense_prediction']}
                assert r['total'] == len(d)
                for name,mask in expected.items():
                    assert r[name+'_count'] == int(mask.sum())
                    near(r['p_'+name],float(mask.mean()))
                near(r['P_safe'],float(joint.mean()) if r['dense_correct'] else None)
                near(r['RandomExact_CE'],float(d.mean()+r['dense_CE']))
                for name,q in zip(['min','q05','q25','median','q75','q95','max'],[0,.05,.25,.5,.75,.95,1]):
                    near(r['delta_'+name],float(np.quantile(d,q,method='linear')))
                for dense in (0,1):
                    for sched in (0,1):
                        assert r[f'transition_D{dense}_S{sched}_count'] == int(r['dense_correct'] == dense)*int((correct == sched).sum())
                pid = lookup[k][tuple(range(0,16,16//k))]
                assert r['periodic_joint'] == bool(joint[pid])
                assert r['any_joint'] == bool(joint.any())
                improve = (d <= d[pid]-.01+TOL) & (correct | (not bool(correct[pid])))
                assert r['improve_periodic_001_count'] == int(improve.sum())
                if (k,pool,eps) == (4,'C',0.):
                    oracle.append((clip,float(d.min()+r['dense_CE'])))
                checked += 1
            for r in methods[methods['clip'] == clip].to_dict('records'):
                k,pool,eps = r['K'],r['pool'],r['epsilon']
                index = lookup[k][tuple(json.loads(r['positions']))]
                d,pred = saved[f'delta_K{k}_{pool}'],saved[f'prediction_K{k}_{pool}']
                base = subset[(subset.K == k)&(subset.pool == pool)&(subset.epsilon == eps)].iloc[0]
                passed = bool(pred[index] == y and d[index] <= eps+TOL)
                near(r['CE'],float(d[index]+base.dense_CE))
                assert r['joint'] == passed
                assert r['missed_safe'] == (not passed and bool(((pred == y)&(d <= eps+TOL)).any()))
    groups = sorted({(v['label'],v['group']) for v in videos})
    rng = np.random.Generator(np.random.PCG64(20261021))
    blocks = []
    for c in sorted(nc):
        members = [i for i,(cl,g) in enumerate(groups) if cl == c]
        blocks.append(np.array(members)[rng.integers(0,len(members),size=(2000,len(members)))])
    indices = np.concatenate(blocks,axis=1)
    np.testing.assert_array_equal(np.load(stage/'bootstrap_group_indices.npy',allow_pickle=False),indices)
    multiplier = np.array([[np.count_nonzero(row == i) for i in range(len(groups))] for row in indices])
    gi = [groups.index((v['label'],v['group'])) for v in videos]
    bw = multiplier[:,gi]*w
    def estimate(values,condition):
        values,condition = np.asarray(values,float),np.asarray(condition,bool)
        mass = float(w@condition)
        denom = bw@condition.astype(float)
        valid = denom > 0
        samples = (bw@(values*condition))[valid]/denom[valid]
        interval = np.percentile(samples,[2.5,97.5],method='linear') if len(samples) else [None,None]
        return dict(estimate=None if mass == 0 else float(w@(values*condition)/mass),weighted_mass=mass,
                    CI95_lower=interval[0],CI95_upper=interval[1],valid_bootstrap_draws=int(valid.sum()))
    def condition(frame,stratum):
        d = frame.dense_correct.to_numpy(bool)
        return d if stratum == 'Dense_correct' else ~d if stratum == 'Dense_wrong' else np.ones(400,bool)
    estimates_checked = 0
    for file in ['summaries.csv','frozen_method_comparisons.csv']:
        for r in csv(file).to_dict('records'):
            source = raw if file == 'summaries.csv' else methods
            subset = source[(source.K == r['K'])&(source.pool == r['pool'])&(source.epsilon == r['epsilon'])]
            if file != 'summaries.csv':
                subset = subset[subset.method == r['method']]
            subset = subset.set_index('clip').loc[ids]
            mask = condition(subset,r['stratum'])
            metric = r['metric']
            if metric == 'P_safe':
                values = subset.p_correct_and_CE
            elif metric == 'missed_safe_given_failure':
                values = subset.missed_safe
                mask = mask & ~subset.joint.to_numpy(bool)
            else:
                values = subset[metric]
            for key,value in estimate(values,mask).items():
                near(r[key],value)
            estimates_checked += 1
    for r in csv('paired_method_contrasts.csv').to_dict('records'):
        sub = methods[(methods.K == r['K'])&(methods.pool == r['pool'])&(methods.epsilon == r['epsilon'])]
        x = sub[sub.method == r['candidate']].set_index('clip').loc[ids][r['metric']].to_numpy(float)
        y = sub[sub.method == r['reference']].set_index('clip').loc[ids][r['metric']].to_numpy(float)
        for key,value in estimate(x-y,np.ones(400,bool)).items():
            near(r[key],value)
        estimates_checked += 1
    for r in json.loads((stage/'periodic_rescue.json').read_text())['rows']:
        sub = raw[(raw.K == r['K'])&(raw.pool == r['pool'])&(raw.epsilon == r['epsilon'])].set_index('clip').loc[ids]
        fail = sub.dense_correct.to_numpy(bool)&~sub.periodic_joint.to_numpy(bool)
        safe = sub.any_joint.to_numpy(bool)
        assert r['clips'] == int(fail.sum())
        near(r['weighted_mass'],float(w@fail))
        for name,val,mask in [('any_safe_fraction',safe,fail),('no_safe_fraction',~safe,fail),
            ('safe_density',sub.p_correct_and_CE,fail),('whole_cohort_rescuable_mass',fail&safe,np.ones(400,bool)),
            ('periodic_failure_mass',fail,np.ones(400,bool))]:
            for key,value in estimate(val,mask).items():
                near(r[name][key],value)
            estimates_checked += 1
    min_by_clip = dict(oracle)
    replay_oracle = float(w@np.array([min_by_clip[clip] for clip in ids]))
    archived = json.loads((ROOT/'results/task_utility_observability_v1/offline_headroom.json').read_text())
    ref = next(r for r in archived['roles'] if r['role'] == 'execution')['task_oracle_CE']
    near(replay_oracle,ref)
    result = dict(passed=True,per_video_rows_recomputed=checked,conditional_and_paired_estimates_and_intervals_checked=estimates_checked,
        exact_schedule_distributions_checked=400*4,bootstrap_indices_identical=True,
        oracle_CE_reconstructed_from_actual_schedule_minima=replay_oracle,archived_oracle_CE=ref,
        verification_only_no_new_estimands=True,new_encoder_calls=0,fits=0,decodes=0,downloads=0)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
