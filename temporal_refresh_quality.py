"""Fixed-head fitting, exact schedule evaluation and source-level statistics."""
from __future__ import annotations
import math
import time

import numpy as np
import pandas as pd

from temporal_refresh_common import *
import refresh_pooling_utils as pool


def write_csv(path,rows):
    pd.DataFrame(rows).to_csv(path,index=False,float_format='%.17g')


def references(output,entry,phase):
    ledger=read(output/f'{phase}_collection.json')['videos'][entry['source_video_id']]
    path=ROOT/ledger['path']; assert sha(path)==ledger['sha256']
    assert ledger['input_sha256']==entry['file_identity']['sha256']
    with np.load(path,allow_pickle=False) as saved: arrays={r:saved[r] for r in REPS}
    for rep,array in arrays.items():
        assert array.dtype==np.float32 and array.shape==(16,2048 if rep=='x3d_xs' else 512)
        assert np.isfinite(array).all()
    return arrays


def head_fitting(output,entries,classes):
    if (output/'head_selection.json').exists(): return read(output/'head_selection.json')
    from task_aware_oracle_utils import fit_classifier
    means={r:{'train':[],'development':[]} for r in REPS}; labels={'train':[],'development':[]}
    for v in entries:
        arrays=references(output,v,'train_development'); role=v['recognition_role']
        labels[role].append(classes.index(v['label']))
        for rep in REPS: means[rep][role].append(arrays[rep].mean(0,dtype=np.float64))
    selection={}; grid_rows=[]
    for rep in REPS:
        x=np.stack(means[rep]['train']); dev=np.stack(means[rep]['development'])
        y=np.array(labels['train']); dy=np.array(labels['development']); options=[]
        for order,alpha in enumerate([.001,.01,.1,1.]):
            head,fit=fit_classifier(x,y,alpha,SEED)
            tr=pool.logits(x,head); dv=pool.logits(dev,head)
            ce=lambda z,ys:float(np.mean(np.logaddexp.reduce(z,axis=1)-z[np.arange(len(z)),ys]))
            record={'representation':rep,'alpha':alpha,'grid_order':order,'training_videos':len(y),'development_videos':len(dy),
                'train_CE':ce(tr,y),'train_accuracy':float((tr.argmax(1)==y).mean()),
                'development_CE':ce(dv,dy),'development_accuracy':float((dv.argmax(1)==dy).mean()),'solver':fit}
            save(output/f'heads/{rep}/grid_{order}.json',{'parameters':head,'fit':record,'classes':classes})
            options.append(record); grid_rows.append(record)
        chosen=min(options,key=lambda e:(e['development_CE'],-e['development_accuracy'],e['grid_order']))
        path=output/f'heads/{rep}/parameters.json'
        save(path,read(output/f'heads/{rep}/grid_{chosen["grid_order"]}.json'))
        selection[rep]={**chosen,'path':relative(path),'sha256':sha(path)}
        print('Frozen selected head',rep,'alpha',chosen['alpha'],'dev CE',chosen['development_CE'],flush=True)
    save(output/'head_grid_results.json',grid_rows); save(output/'head_selection.json',selection)
    return selection


def evaluate_cohort(output,entries,cohort,classes,budgets):
    if cohort=='evaluation':require_evaluation_seal(output)
    target=output/f'{cohort}_quality.csv'
    if target.exists():
        quality_sidecars(output,cohort,budgets)
        return pool.frame(target)
    rows=[]; selected=[]; schedules={b:pool.schedules(16,b) for b in budgets}
    phase='evaluation' if cohort=='evaluation' else 'train_development'
    for i,v in enumerate(entries):
        arrays=references(output,v,phase); y=classes.index(v['label'])
        for rep,features in arrays.items():
            head=read(output/f'heads/{rep}/parameters.json')['parameters']
            z=pool.logits(features,head); dense_z=z.mean(0); dense_loss=float(pool.loss(dense_z[None],y)[0])
            base={'cohort':cohort,'representation':rep,'source_video_id':v['source_video_id'],'label':v['label'],'label_index':y}
            rows.append({**base,'budget':16,'policy':'dense16','pool':'C=U','CE':dense_loss,'correct':int(dense_z.argmax()==y),
                'prediction_index':int(dense_z.argmax()),'feature_error':0.,'positions':json.dumps(list(range(16))),
                'effective_weight_count':16.,'true_minimum':None,'achieved_objective':None,'tie_count':None})
            for b,table in schedules.items():
                value=pool.evaluate(features,head,y,table)
                periodic=list(range(0,16,16//b)); pindex=int(np.flatnonzero((table['positions']==periodic).all(1))[0])
                choices={'periodic':{'index':pindex},'feature_oracle':value['F'],
                    'cache_task_oracle':value['SC'],'unique_task_oracle':value['SU']}
                for policy,choice in choices.items():
                    j=choice['index']; positions=table['positions'][j]
                    for pooling in ['C','U']:
                        weights=table['cache_weights' if pooling=='C' else 'unique_weights'][j]
                        logits=value[f'logits_{pooling}'][j]
                        row={**base,'budget':b,'policy':policy,'pool':pooling,'CE':float(value[f'loss_{pooling}'][j]),
                            'correct':int(value[f'correct_{pooling}'][j]),'prediction_index':int(value[f'prediction_{pooling}'][j]),
                            'feature_error':float(value['feature_error'][j]),'positions':json.dumps(positions.tolist()),
                            'effective_weight_count':float(1/np.square(weights).sum()),
                            **{k:choice.get(k) for k in ['true_minimum','achieved_objective','tie_count']}}
                        rows.append(row); selected.append({**row,'weights':weights.tolist(),'logits':logits.tolist()})
                        # Independent scalar cache-copy or unique average; no pooling helper.
                        if pooling=='C':
                            explicit=[]; last=None
                            for t in range(16):
                                if t in positions: last=features[t]
                                explicit.append(last)
                            mean=np.asarray(explicit,dtype=np.float64).mean(0)
                        else: mean=np.asarray([features[t] for t in positions],dtype=np.float64).mean(0)
                        reference=((mean-np.array(head['means']))/head['scales'])@np.array(head['weights']).T+head['bias']
                        np.testing.assert_allclose(reference,logits,atol=1e-10,rtol=1e-10)
                for pooling in ['C','U']:
                    rows.append({**base,'budget':b,'policy':'RandomExact','pool':pooling,
                        'CE':float(value[f'loss_{pooling}'].mean()),'correct':float(value[f'correct_{pooling}'].mean()),
                        'prediction_index':None,'feature_error':float(value['feature_error'].mean()),'positions':None,
                        'effective_weight_count':float((1/np.square(table['cache_weights' if pooling=='C' else 'unique_weights']).sum(1)).mean()),
                        'true_minimum':None,'achieved_objective':None,'tie_count':None})
                dest=ASSETS/'schedule_outcomes'/cohort/rep/f'{v["source_video_id"]}_B{b}.npz'
                dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,**{k:value[k] for k in ['loss_C','loss_U','feature_error','correct_C','correct_U']})
        if (i+1)%10==0: print(f'{cohort} exact quality {i+1}/{len(entries)}',flush=True)
    write_csv(target,rows); save(output/f'{cohort}_selected_schedules.json',selected)
    quality_sidecars(output,cohort,budgets)
    return pd.DataFrame(rows)


def quality_sidecars(output,cohort,budgets):
    selected=read(output/f'{cohort}_selected_schedules.json')
    diagnostics=[];selected_frame=pd.DataFrame(selected)
    for (rep,source,b),group in selected_frame[selected_frame.pool=='C'].groupby(['representation','source_video_id','budget']):
        s={row.policy:set(json.loads(row.positions)) for row in group.itertuples()}
        diagnostics.append({'cohort':cohort,'representation':rep,'source_video_id':source,'budget':int(b),
            'task_schedule_agreement':s['cache_task_oracle']==s['unique_task_oracle'],
            'task_schedule_jaccard':len(s['cache_task_oracle']&s['unique_task_oracle'])/len(s['cache_task_oracle']|s['unique_task_oracle']),
            'feature_cache_task_jaccard':len(s['feature_oracle']&s['cache_task_oracle'])/len(s['feature_oracle']|s['cache_task_oracle'])})
    write_csv(output/f'{cohort}_schedule_diagnostics.csv',diagnostics)
    artifacts=[]
    for path in sorted((ASSETS/'schedule_outcomes'/cohort).glob('*/*.npz')):
        artifacts.append({'path':relative(path),'sha256':sha(path),'bytes':path.stat().st_size})
    save(output/f'{cohort}_schedule_artifacts.json',artifacts)
    for budget in budgets:
        path=output/f'schedules_B{budget}.npz'
        if not path.exists():np.savez_compressed(path,**pool.schedules(16,budget))


def paired_records(raw):
    rows=[]
    for (cohort,rep,source,b),g in raw[raw.budget!=16].groupby(['cohort','representation','source_video_id','budget']):
        cells=g.set_index(['policy','pool']); label=g.label.iloc[0]
        ce=lambda policy,p:float(cells.loc[(policy,p)].CE)
        pc=ce('periodic','C'); pu=ce('periodic','U')
        cc=ce('cache_task_oracle','C'); cu=ce('cache_task_oracle','U')
        uc=ce('unique_task_oracle','C'); uu=ce('unique_task_oracle','U')
        assert abs(pc-pu)<1e-10 and min(pc-cc,pu-uu)>=-TOL
        values={'feature_minus_periodic_C':ce('feature_oracle','C')-pc,
            'feature_minus_periodic_U':ce('feature_oracle','U')-pu,
            'H_C':pc-cc,'H_U':pu-uu,'H_C_minus_H_U':uu-cc,'W_C':cu-cc,'W_U':uu-uc,
            'L_C_SC':cc,'L_U_SC':cu,'L_C_SU':uc,'L_U_SU':uu}
        for policy in ['feature_oracle','cache_task_oracle','unique_task_oracle']:
            for p in ['C','U']:
                values[f'{policy}_{p}_accuracy_minus_periodic_pp']=100*(float(cells.loc[(policy,p)].correct)-float(cells.loc[('periodic',p)].correct))
        for name,value in values.items():
            rows.append({'cohort':cohort,'representation':rep,'source_video_id':source,'label':label,'budget':int(b),
                'contrast':name,'value':value,'primary':cohort=='evaluation' and rep=='x3d_xs' and b==4 and name=='feature_minus_periodic_C'})
    return pd.DataFrame(rows)


def summarize(output,raw,classes,bootstrap,seed):
    pairs=paired_records(raw); write_csv(output/'per_source_deltas.csv',pairs)
    aggregates=[]; intervals=[]; perclass=[]
    for cohort,g in raw.groupby('cohort',sort=True):
        sources=sorted(g.source_video_id.unique()); labels=g.drop_duplicates('source_video_id').set_index('source_video_id').label.to_dict()
        idx=pool.stratified_indices(sources,labels,classes,bootstrap,seed)
        np.save(output/f'{cohort}_bootstrap_indices.npy',idx)
        save(output/f'{cohort}_bootstrap_sources.json',sources)
        for (rep,b,policy,p),block in g.groupby(['representation','budget','policy','pool']):
            q=block.set_index('source_video_id').loc[sources]
            row={'cohort':cohort,'representation':rep,'budget':int(b),'policy':policy,'pool':p,'sources':len(q),
                'CE':pool.describe(q.CE.to_numpy(),idx),'accuracy_pp':pool.describe(100*q.correct.to_numpy(),idx),
                'macro_accuracy_pp':float(100*q.groupby('label').correct.mean().mean()),'feature_error':float(q.feature_error.mean()),
                'correct_count_or_expectation':float(q.correct.sum()),'expectation':policy=='RandomExact'}
            aggregates.append(row)
            for label,cl in q.groupby('label'):
                perclass.append({**{k:row[k] for k in ['cohort','representation','budget','policy','pool']},
                    'label':label,'sources':len(cl),'CE':float(cl.CE.mean()),'accuracy_pp':float(100*cl.correct.mean())})
        subset=pairs[pairs.cohort==cohort]
        for (rep,b,name),block in subset.groupby(['representation','budget','contrast']):
            values=block.set_index('source_video_id').loc[sources].value.to_numpy()
            intervals.append({'cohort':cohort,'representation':rep,'budget':int(b),'contrast':name,
                'sources':len(values),'primary':bool(block.primary.any()),**pool.describe(values,idx),
                'positive_count':int((values>TOL).sum()),'tied_count':int((abs(values)<=TOL).sum()),'negative_count':int((values<-TOL).sum())})
    save(output/'aggregate_metrics.json',{'rows':aggregates,'unit':'equal source videos; RandomExact correctness is expectation'})
    save(output/'bootstrap_intervals.json',{'rows':intervals,'seed':seed,'resamples':bootstrap,'RNG':'PCG64','percentile':'linear',
        'unit':'class-stratified whole source; shared saved indices across representations/budgets/pools',
        'limits':'fixed fitted heads/classes; excludes training/model selection/dataset uncertainty; development reused; secondary marginal unadjusted'})
    write_csv(output/'paired_comparisons.csv',intervals); write_csv(output/'per_class_metrics.csv',perclass)
    return intervals


def precision(output,dev):
    pairs=paired_records(dev)
    d=pairs[(pairs.representation=='x3d_xs')&(pairs.budget==4)&(pairs.contrast=='feature_minus_periodic_C')]
    # Equal20-class weights, planned5 evaluation sources/class. Within-class
    # sample variance estimated from reused development (5/class), head fixed.
    variance=sum(float(g.value.var(ddof=1))/(20**2*5) for _,g in d.groupby('label'))
    value={'N_fixed':100,'per_class':5,'development_mean_delta':float(d.value.mean()),
        'estimated_SE':math.sqrt(variance),'normal_approx_expected_CI_halfwidth':1.96*math.sqrt(variance),
        'assumptions':'development within-class variance represents new sources; fixed head/classes; normal approximation; only5 development sources/class; no power guarantee; N remains100 regardless of evaluation interval'}
    save(output/'development_precision_estimate.json',value); return value
