"""Independent reconstruction from saved outcomes;no main pooling helpers."""
from __future__ import annotations
from itertools import combinations
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def frame(path):return pd.read_csv(path,float_precision='round_trip')


def ce(z,y):
    high=z.max(axis=-1)
    return high+np.log(np.exp(z-high[...,None]).sum(axis=-1))-z[...,y]


def affine(x,head):
    return ((x-np.array(head['means']))/np.array(head['scales']))@np.array(head['weights']).T+np.array(head['bias'])


def explicit_pool(features,positions,head):
    """Cache copying,unique averaging and affine application,independent code."""
    output=[];cache=None
    for index in range(len(features)):
        if index in positions:cache=features[index].copy()
        assert cache is not None
        output.append(cache.copy())
        np.testing.assert_array_equal(output[-1],features[max(t for t in positions if t<=index)])
    cache_features=np.array(output)
    a=affine(cache_features.mean(0),head)
    np.testing.assert_allclose(a,np.mean(affine(cache_features,head),axis=0),atol=1e-10,rtol=1e-10)
    b=affine(features[positions].mean(0),head)
    np.testing.assert_allclose(b,np.mean(affine(features[positions],head),axis=0),atol=1e-10,rtol=1e-10)
    return a,b


def contrasts(cells):
    pc,pu=cells[('periodic','C')].CE,cells[('periodic','U')].CE
    cc,cu=cells[('cache_task_oracle','C')].CE,cells[('cache_task_oracle','U')].CE
    uc,uu=cells[('unique_task_oracle','C')].CE,cells[('unique_task_oracle','U')].CE
    values={'H_C':pc-cc,'H_U':pu-uu,'Delta':uu-cc,'W_C':cu-cc,'W_U':uu-uc,
        'selection_gain_U_on_SC':pu-cu,'L_C_SC':cc,'L_U_SC':cu,'L_C_SU':uc,'L_U_SU':uu,
        'feature_oracle_C_loss_reduction':pc-cells[('feature_oracle','C')].CE,
        'feature_oracle_U_loss_reduction':pu-cells[('feature_oracle','U')].CE}
    assert abs(pc-pu)<=1e-12 and min(pc-cc,pu-uu)>=-1e-12
    assert abs((pc-cc)-((pu-cu)+(cu-cc)))<=1e-12
    for policy in ['periodic','feature_oracle','cache_task_oracle','unique_task_oracle']:
        values[policy+'_accuracy_U_minus_C']=100*(cells[(policy,'U')].correct-cells[(policy,'C')].correct)
    for policy,pool in [('feature_oracle','C'),('feature_oracle','U'),('cache_task_oracle','C'),('cache_task_oracle','U'),('unique_task_oracle','C'),('unique_task_oracle','U')]:
        values[policy+'_'+pool+'_accuracy_vs_periodic']=100*(cells[(policy,pool)].correct-cells[('periodic',pool)].correct)
    return values


def audit(output:Path,directory:Path,p:dict,verify_ledger=True):
    import hashlib
    root=Path(__file__).resolve().parent
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    for name,digest in {**p['source_sha256'],**p['input_sha256']}.items():assert sha(root/name)==digest,name
    if verify_ledger:
        ledger={name:digest for digest,name in (s.split('  ',1) for s in (output/'SHA256SUMS.txt').read_text().splitlines())}
        for name,digest in ledger.items():assert sha(output/name)==digest,name
        assert set(ledger)=={f.relative_to(output).as_posix() for f in output.rglob('*') if f.is_file() and f.name!='SHA256SUMS.txt'}
    raw=frame(output/'per_video_metrics.csv');pairs=frame(output/'paired_comparisons.csv')
    assert len(raw)==4340 and not raw.duplicated(['source_video_id','budget','policy','pool']).any()
    manifest=read(directory/'dataset_manifest.json');head=read(directory/'classifiers/20/parameters.json')
    entries=[e for e in manifest['videos'] if e['recognition_role'] in ['development','exploratory']]
    archived=frame(directory/'per_video_metrics.csv');archived=archived[archived.training_per_class==20].set_index(['cohort','source_video_id','policy'])
    selected={v['source_video_id']:v for v in read(output/'selected_schedules.json')['videos']}
    tables={}
    with np.load(output/'canonical_schedules.npz',allow_pickle=False) as canonical:
        for b in p['budgets']:
            pos=np.array([(0,*rest) for rest in combinations(range(1,16),b-1)])
            assert len(pos)==comb(15,b-1)
            caches=[];weights=[];durations=[]
            for s in pos:
                last=0;timeline=[];w=np.zeros(16)
                for t in range(16):
                    if t in s:last=t
                    timeline.append(last);w[last]+=1/16
                caches.append(timeline);weights.append(w)
                durations.append([int((list(s)+[16])[j+1]-s[j]) for j in range(b)])
            np.testing.assert_array_equal(pos,canonical[f'b{b}_positions'])
            np.testing.assert_array_equal(caches,canonical[f'b{b}_cache'])
            np.testing.assert_array_equal(weights,canonical[f'b{b}_cache_weights'])
            np.testing.assert_array_equal(durations,canonical[f'b{b}_durations'])
            unique=np.zeros((len(pos),16));unique[np.arange(len(pos))[:,None],pos]=1/b
            np.testing.assert_array_equal(unique,canonical[f'b{b}_unique_weights'])
            tables[b]=(pos,np.array(caches),np.array(durations))
            saved=frame(output/f'schedules_b{b}.csv')
            assert [json.loads(v) for v in saved.positions]==pos.tolist()
    checked=archived_checks=outcomes=0;max_logit=max_loss=max_error=0.
    for entry in entries:
        source=entry['source_video_id'];y=p['classes'].index(entry['label']);cohort=entry['recognition_role']
        fp=root/entry['reference_directory']/'features'/(entry['video_id']+'.npz')
        assert sha(fp)==entry['reference_artifact_sha256'][f"features/{entry['video_id']}.npz"]
        with np.load(fp,allow_pickle=False) as stored:x=stored['features'].astype(np.float64)
        z=affine(x,head)
        norm=np.linalg.norm(x,axis=1)
        d=np.array([[1.-np.clip(float(np.dot(a,b)/(na*nb)),-1.,1.) for b,nb in zip(x,norm)] for a,na in zip(x,norm)])
        local=raw[raw.source_video_id==source]
        assert len(local)==31
        with np.load(output/'per_schedule'/(entry['video_id']+'.npz'),allow_pickle=False) as dist:
            for b in p['budgets']:
                positions,caches,duration=tables[b]
                # Independent selected-index summation,not the main full16 weight-matrix multiply.
                zc=(z[positions]*duration[:,:,None]).sum(axis=1)/16
                zu=z[positions].mean(axis=1)
                losses={'C':ce(zc,y),'U':ce(zu,y)}
                feature_error=d[np.arange(16)[None,:],caches].mean(1)
                for pool,zs in [('C',zc),('U',zu)]:
                    np.testing.assert_allclose(losses[pool],dist[f'b{b}_loss_{pool}'],atol=1e-10,rtol=1e-10)
                    np.testing.assert_array_equal(zs.argmax(1),dist[f'b{b}_prediction_{pool}'])
                    np.testing.assert_array_equal(zs.argmax(1)==y,dist[f'b{b}_correct_{pool}'])
                    max_loss=max(max_loss,float(abs(losses[pool]-dist[f'b{b}_loss_{pool}']).max()))
                np.testing.assert_allclose(feature_error,dist[f'b{b}_feature_error'],atol=1e-10,rtol=1e-10)
                rows=local[local.budget==b];cells={(r.policy,r.pool):r for r in rows.itertuples()}
                for policy in ['periodic','feature_oracle','cache_task_oracle','unique_task_oracle']:
                    info=selected[source]['budgets'][str(b)][policy];index=info['index'];schedule=positions[index].tolist()
                    assert schedule==info['positions'] and len(schedule)==b and schedule[0]==0
                    for pool,vector in zip(['C','U'],explicit_pool(x,schedule,head)):
                        row=cells[(policy,pool)];saved_logits=np.array(json.loads(row.logits))
                        np.testing.assert_allclose(vector,saved_logits,atol=1e-10,rtol=1e-10)
                        max_logit=max(max_logit,float(abs(vector-saved_logits).max()))
                        np.testing.assert_allclose(ce(vector,y),row.CE,atol=1e-10,rtol=1e-10)
                        assert int(vector.argmax())==row.prediction_index and float(vector.argmax()==y)==row.correct
                        np.testing.assert_allclose(feature_error[index],row.feature_error,atol=1e-10,rtol=1e-10)
                        weights=np.array(json.loads(row.weights));assert len(weights)==b
                        np.testing.assert_allclose(weights.sum(),1.,atol=1e-12,rtol=0)
                        np.testing.assert_allclose(row.effective_weight_count,1/sum(w*w for w in weights),atol=1e-12,rtol=0)
                        checked+=1
                    if policy!='periodic':
                        values=dist[f'b{b}_feature_error'] if policy=='feature_oracle' else dist[f'b{b}_loss_C'] if policy=='cache_task_oracle' else dist[f'b{b}_loss_U']
                        minimum=min(float(v) for v in values);tie=[i for i,v in enumerate(values) if v<=minimum+1e-12]
                        assert index==tie[0] and info['oracle_objective']['tie_count']==len(tie)
                        assert info['oracle_objective']['true_minimum']==minimum
                        assert info['oracle_objective']['achieved_objective']==values[index]
                        assert info['oracle_objective']['tolerance']==1e-12
                for pool in ['C','U']:
                    row=cells[('RandomExact',pool)]
                    np.testing.assert_allclose(row.CE,np.mean(dist[f'b{b}_loss_{pool}']),atol=1e-12,rtol=0)
                    np.testing.assert_allclose(row.correct,np.mean(dist[f'b{b}_correct_{pool}']),atol=1e-12,rtol=0)
                    assert pd.isna(row.logits) and pd.isna(row.refresh_positions) and row.correctness_is_expectation
                np.testing.assert_allclose(cells[('periodic','C')].CE,cells[('periodic','U')].CE,atol=1e-12,rtol=0)
                independent=contrasts(cells)
                saved_pairs=pairs[(pairs.source_video_id==source)&(pairs.budget==b)]
                assert len(saved_pairs)==len(independent)
                for r in saved_pairs.itertuples():np.testing.assert_allclose(r.paired_value,independent[r.contrast],atol=1e-12,rtol=0)
                if b==4:
                    for policy,old_policy in [('periodic','periodic'),('feature_oracle','feature_oracle'),('cache_task_oracle','task_oracle')]:
                        row=cells[(policy,'C')];old=archived.loc[(cohort,source,old_policy)]
                        assert json.loads(row.refresh_positions)==json.loads(old.refresh_positions)
                        np.testing.assert_allclose(json.loads(row.logits),json.loads(old.logits),atol=1e-10,rtol=1e-10)
                        np.testing.assert_allclose([row.CE,row.feature_error],[old.cross_entropy,old.feature_cosine_error],atol=1e-10,rtol=1e-10)
                        assert row.prediction_index==old.prediction_index and row.correct==old.correct
                        assert old.label==entry['label'] and old.label_index==y
                        archived_checks+=1
                outcomes+=len(positions)
        row=local[local.policy=='dense'].iloc[0];a,b=explicit_pool(x,list(range(16)),head)
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
        np.testing.assert_allclose(a,json.loads(row.logits),atol=1e-10,rtol=1e-10)
        old=archived.loc[(cohort,source,'dense')]
        np.testing.assert_allclose(row.CE,old.cross_entropy,atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(json.loads(row.logits),json.loads(old.logits),atol=1e-10,rtol=1e-10)
        assert row.correct==old.correct and row.prediction_index==old.prediction_index
        archived_checks+=1;checked+=1
    aggregate=read(output/'aggregate_metrics.json')['rows'];intervals=read(output/'bootstrap_intervals.json')['contrasts']
    with np.load(output/'bootstrap_indices.npz',allow_pickle=False) as matrices:
        for cohort in ['development','exploratory']:
            group=raw[raw.cohort==cohort];sources=sorted(group.source_video_id.unique())
            labels=group.drop_duplicates('source_video_id').set_index('source_video_id').label.to_dict()
            idx=matrices[cohort];assert idx.shape==(p['bootstrap']['resamples'],len(sources))
            rng=np.random.Generator(np.random.PCG64(p['bootstrap']['seed']));blocks=[]
            for label in p['classes']:
                members=[i for i,s in enumerate(sources) if labels[s]==label]
                assert len(members)==(5 if cohort=='development' else 2)
                blocks.append(np.array(members)[rng.integers(0,len(members),(len(idx),len(members)))])
            np.testing.assert_array_equal(idx,np.concatenate(blocks,axis=1))
            for row in [r for r in aggregate if r['cohort']==cohort]:
                g=group[(group.budget==row['budget'])&(group.policy==row['policy'])&(group.pool==row['pool'])].set_index('source_video_id').loc[sources]
                assert len(g)==row['sources']
                np.testing.assert_allclose([g.correct.mean(),g.groupby('label').correct.mean().mean(),g.correct.sum(),g.feature_error.mean(),g.effective_weight_count.mean()],
                    [row['top1_accuracy'],row['macro_top1_accuracy'],row['correct_count_or_expectation'],row['mean_last_cache_feature_error'],row['mean_effective_weight_count']],atol=1e-12,rtol=0)
                for key,values in [('CE',g.CE.to_numpy()),('accuracy_pp',100*g.correct.to_numpy())]:
                    quantiles=np.quantile(values[idx].mean(axis=1),[.025,.975],method='linear')
                    np.testing.assert_allclose([values.mean(),*quantiles],[row[key]['mean'],row[key]['CI95_lower'],row[key]['CI95_upper']],atol=1e-12,rtol=0)
            for row in [r for r in intervals if r['cohort']==cohort]:
                group=pairs[(pairs.cohort==cohort)&(pairs.budget==row['budget'])&(pairs.contrast==row['contrast'])].set_index('source_video_id').loc[sources]
                values=group.paired_value.to_numpy();quantiles=np.quantile(values[idx].mean(1),[.025,.975],method='linear')
                np.testing.assert_allclose([values.mean(),*quantiles],[row['mean'],row['CI95_lower'],row['CI95_upper']],atol=1e-12,rtol=0)
                assert int((values>1e-12).sum())==row['positive_sources']
                assert int((abs(values)<=1e-12).sum())==row['tied_sources']
                assert int((values< -1e-12).sum())==row['negative_sources']
    pdf_checked=0
    if (output/'figures').exists():
        for path in (output/'figures').glob('*.pdf'):
            data=path.read_bytes();assert data.startswith(b'%PDF-') and b'%%EOF' in data and b'/Count 1' in data;pdf_checked+=1
        assert pdf_checked==4 and len(list((output/'figures').glob('*.png')))==4
    return {'passed':True,'archived_primary_head_B4_rows_reproduced':archived_checks,
        'independent_explicit_selected_pool_checks':checked,'all_schedule_outcomes_checked':outcomes,
        'maximum_independent_logit_discrepancy':max_logit,'maximum_independent_schedule_loss_discrepancy':max_loss,
        'schedule_counts':{str(b):comb(15,b-1) for b in p['budgets']},'null_RandomExact_outputs_checked':840,
        'source_class_stratification_and_saved_index_reproduction':True,'all_aggregates_contrasts_and_bootstrap_endpoints_reproduced':True,
        'affine_cache_and_unique_pooling_and_accounting_identities':True,'periodic_and_dense_pooling_equal':True,
        'PDF_single_page_structure_checks':pdf_checked,'training_reference_arrays_loaded':0,'protected_confirmation_outcomes_arrays_or_motion_tables_loaded':0,
        'actual_new_encoder_calls':0,'video_decodes':0,'fits':0,'video_or_weight_downloads':0}
