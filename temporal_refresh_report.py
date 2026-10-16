"""Saved-result independent audit, figures and bounded interpretation."""
from __future__ import annotations
from collections import Counter
import itertools
import math
import subprocess

import numpy as np
import pandas as pd

from temporal_refresh_common import *
from temporal_refresh_quality import paired_records
import refresh_pooling_utils as pool


def audit(output):
    manifest=read(output/'evaluation_manifest.json'); videos=manifest['videos']; classes=manifest['classes']
    assert len(videos)==100 and all(n==5 for n in Counter(v['label'] for v in videos).values())
    train=read(output/'train_development_manifest.json')['videos']
    assert Counter(v['recognition_role'] for v in train)=={'train':400,'development':100}
    allvideos=train+videos
    assert len({v['source_video_id'] for v in allvideos})==600
    assert len({v['file_identity']['sha256'] for v in allvideos})==600
    excluded=read(output/'source_exclusions.json')
    assert not {v['source_video_id'] for v in videos}&set(excluded['source_ids'])
    assert not {v['file_identity']['sha256'] for v in videos}&set(excluded['encoded_sha256'])
    for name,digest in read(output/'evaluation_seal.json')['sha256'].items(): assert sha(ROOT/name)==digest,name
    assert read(output/'evaluation_seal.json')['sealed_at_utc']<read(output/'evaluation_exposure.json')['started_at_utc']
    feature_count=0; grid_checks=0
    for phase,entries in [('train_development',train),('evaluation',videos)]:
        ledger=read(output/f'{phase}_collection.json')['videos']; assert len(ledger)==len(entries)
        for v in entries:
            data=ledger[v['source_video_id']]; path=ROOT/data['path']; assert sha(path)==data['sha256']
            assert data['input_sha256']==v['file_identity']['sha256']
            assert sha(ROOT/v['relative_path'])==v['file_identity']['sha256']
            with np.load(path,allow_pickle=False) as saved:
                for rep in REPS:
                    f=saved[rep]; assert f.shape==(16,2048 if rep=='x3d_xs' else 512)
                    assert f.dtype==np.float32 and np.isfinite(f).all(); feature_count+=16
            g=v['snippet_grid']; req=np.array(g['requested_timestamps']); actual=np.array(g['actual_timestamps']); indices=np.array(g['frame_indices'])
            assert (actual<=req+TIME_TOL).all() and (np.diff(actual,axis=1)>0).all()
            assert len(set(g['endpoint_indices']))==16 and (np.diff(g['endpoint_timestamps'])>0).all()
            assert all(len(set(row))==4 for row in indices); grid_checks+=1
    raw=pool.frame(output/'per_video_quality.csv'); assert len(raw)==200*3*31
    assert not raw.duplicated(['cohort','representation','source_video_id','budget','policy','pool']).any()
    assert np.isfinite(raw[['CE','correct','feature_error']]).all().all()
    # Independent re-read of per-source summaries and selected logits. No main
    # encoder, fitting, acquisition or quality-evaluation path is called here.
    explicit_checks=0; max_difference=0.
    canonical={b:pool.schedules(16,b) for b in [2,4,8]}
    position_index={b:{tuple(row):i for i,row in enumerate(tab['positions'])} for b,tab in canonical.items()}
    raw_index=raw.set_index(['cohort','representation','source_video_id','budget','policy','pool'])
    entry_lookup={v['source_video_id']:v for v in allvideos}
    for cohort in ['development','evaluation']:
        selected=read(output/f'{cohort}_selected_schedules.json')
        phase='evaluation' if cohort=='evaluation' else 'train_development'
        ledger=read(output/f'{phase}_collection.json')['videos']
        for artifact in read(output/f'{cohort}_schedule_artifacts.json'):
            assert sha(ROOT/artifact['path'])==artifact['sha256']
        # One-source-at-a-time avoids retaining full cohort reference tensors.
        for source,items in itertools.groupby(sorted(selected,key=lambda r:r['source_video_id']),key=lambda r:r['source_video_id']):
            with np.load(ROOT/ledger[source]['path'],allow_pickle=False) as saved:
                arrays={r:saved[r] for r in REPS}
            for item in items:
                f=arrays[item['representation']]; s=json.loads(item['positions']); assert s==sorted(set(s)) and s[0]==0 and len(s)==item['budget']
                if item['pool']=='C':
                    copies=[]; last=0
                    for t in range(16):
                        if t in s: last=t
                        copies.append(f[last])
                    mean=np.mean(copies,axis=0,dtype=np.float64)
                else: mean=np.mean([f[t] for t in s],axis=0,dtype=np.float64)
                head=read(output/f'heads/{item["representation"]}/parameters.json')['parameters']
                z=((mean-head['means'])/head['scales'])@np.array(head['weights']).T+head['bias']
                np.testing.assert_allclose(z,item['logits'],atol=1e-10,rtol=1e-10)
                ce=max(z)+math.log(sum(math.exp(x-max(z)) for x in z))-z[item['label_index']]
                max_difference=max(max_difference,abs(ce-item['CE']))
                assert abs(ce-item['CE'])<1e-10; explicit_checks+=1
                q=raw_index.loc[(cohort,item['representation'],source,item['budget'],item['policy'],item['pool'])]
                assert abs(q.CE-ce)<1e-10 and q.correct==int(z.argmax()==item['label_index'])
                assert json.loads(q.positions)==s
                if 'oracle' in item['policy']:
                    j=position_index[item['budget']][tuple(s)]
                    with np.load(ASSETS/'schedule_outcomes'/cohort/item['representation']/f'{source}_B{item["budget"]}.npz') as outcomes:
                        objective=outcomes['feature_error' if item['policy']=='feature_oracle' else 'loss_C' if item['policy']=='cache_task_oracle' else 'loss_U']
                        minimum=float(objective.min());ties=np.flatnonzero(objective<=minimum+TOL)
                        assert j==int(ties[0]) and len(ties)==item['tie_count']
                        assert abs(minimum-item['true_minimum'])<TOL
                # Reconstruction stays a last-cache timeline property even forU.
                cache=[];last=0
                for t in range(16):
                    if t in s:last=t
                    cache.append(last)
                error=np.mean([1-np.clip(np.dot(f[t].astype(float),f[r].astype(float))/(np.linalg.norm(f[t].astype(float))*np.linalg.norm(f[r].astype(float))),-1,1) for t,r in enumerate(cache)])
                assert abs(error-q.feature_error)<1e-10
            for rep in REPS:
                head=read(output/f'heads/{rep}/parameters.json')['parameters'];mean=arrays[rep].mean(0,dtype=np.float64)
                z=((mean-head['means'])/head['scales'])@np.array(head['weights']).T+head['bias']
                ce=max(z)+math.log(sum(math.exp(x-max(z)) for x in z))-z[classes.index(entry_lookup[source]['label'])]
                q=raw_index.loc[(cohort,rep,source,16,'dense16','C=U')]
                assert abs(q.CE-ce)<1e-10 and q.correct==int(z.argmax()==q.label_index)
                for b in [2,4,8]:
                    with np.load(ASSETS/'schedule_outcomes'/cohort/rep/f'{source}_B{b}.npz') as outcomes:
                        for p in ['C','U']:
                            q=raw_index.loc[(cohort,rep,source,b,'RandomExact',p)]
                            np.testing.assert_allclose([q.CE,q.correct,q.feature_error],
                                [outcomes[f'loss_{p}'].mean(),outcomes[f'correct_{p}'].mean(),outcomes['feature_error'].mean()],atol=1e-12,rtol=0)
    real_exhaustive_checks=0
    for cohort in ['development','evaluation']:
        phase='evaluation' if cohort=='evaluation' else 'train_development'
        ledger=read(output/f'{phase}_collection.json')['videos']
        source=min(v['source_video_id'] for v in (videos if cohort=='evaluation' else train) if v['recognition_role']==cohort)
        entry=entry_lookup[source];y=classes.index(entry['label'])
        with np.load(ROOT/ledger[source]['path']) as saved:
            for rep in REPS:
                f=saved[rep].astype(np.float64);head=read(output/f'heads/{rep}/parameters.json')['parameters']
                z=np.array([((row-head['means'])/head['scales'])@np.array(head['weights']).T+head['bias'] for row in f])
                d=np.empty((16,16))
                for t in range(16):
                    for r in range(16): d[t,r]=1-np.clip(sum(f[t]*f[r])/(math.sqrt(sum(f[t]**2))*math.sqrt(sum(f[r]**2))),-1,1)
                for b in [2,4,8]:
                    cache_loss=[];unique_loss=[];error=[]
                    for tail in itertools.combinations(range(1,16),b-1):
                        schedule=(0,*tail);last=0;cached=[];err=0
                        for t in range(16):
                            if t in schedule:last=t
                            cached.append(z[last]);err+=d[t,last]
                        for output_values,dest in [(cached,cache_loss),([z[t] for t in schedule],unique_loss)]:
                            values=np.mean(output_values,axis=0);maximum=max(values)
                            dest.append(maximum+math.log(sum(math.exp(x-maximum) for x in values))-values[y])
                        error.append(err/16)
                    with np.load(ASSETS/'schedule_outcomes'/cohort/rep/f'{source}_B{b}.npz') as saved_values:
                        for name,values in [('loss_C',cache_loss),('loss_U',unique_loss),('feature_error',error)]:
                            np.testing.assert_allclose(values,saved_values[name],atol=1e-10,rtol=1e-10)
                        real_exhaustive_checks+=len(error)
    # Reconstruct contrasts and bootstrap endpoints from saved source cells.
    endpoint_checks=0;bootstrap_cache={};saved_pairs=pool.frame(output/'per_source_deltas.csv')
    for cohort in ['development','evaluation']:
        sources=read(output/f'{cohort}_bootstrap_sources.json');idx=np.load(output/f'{cohort}_bootstrap_indices.npy')
        label_lookup=raw[raw.cohort==cohort].drop_duplicates('source_video_id').set_index('source_video_id').label.to_dict()
        codes=np.array([classes.index(label_lookup[s]) for s in sources]);rng=np.random.Generator(np.random.PCG64(SEED));blocks=[]
        for code in range(20):
            members=np.flatnonzero(codes==code);assert len(members)==5
            blocks.append(members[rng.integers(0,5,size=(10000,5))])
        np.testing.assert_array_equal(idx,np.concatenate(blocks,axis=1))
        for code in range(20):assert ((codes[idx]==code).sum(1)==5).all()
        bootstrap_cache[cohort]=(sources,idx)
    for interval in read(output/'bootstrap_intervals.json')['rows']:
        cohort=interval['cohort']; sources,idx=bootstrap_cache[cohort]
        q=raw[(raw.cohort==cohort)&(raw.representation==interval['representation'])&(raw.budget==interval['budget'])]
        cells=q.pivot(index='source_video_id',columns=['policy','pool'],values='CE').loc[sources]
        pc=cells[('periodic','C')].to_numpy(); pu=cells[('periodic','U')].to_numpy()
        cc=cells[('cache_task_oracle','C')].to_numpy(); cu=cells[('cache_task_oracle','U')].to_numpy()
        uc=cells[('unique_task_oracle','C')].to_numpy(); uu=cells[('unique_task_oracle','U')].to_numpy()
        name=interval['contrast']
        values={'feature_minus_periodic_C':cells[('feature_oracle','C')].to_numpy()-pc,
            'feature_minus_periodic_U':cells[('feature_oracle','U')].to_numpy()-pu,'H_C':pc-cc,'H_U':pu-uu,
            'H_C_minus_H_U':uu-cc,'W_C':cu-cc,'W_U':uu-uc,'L_C_SC':cc,'L_U_SC':cu,'L_C_SU':uc,'L_U_SU':uu}
        if name in values: d=values[name]
        else:
            policy,p=name.removesuffix('_accuracy_minus_periodic_pp').rsplit('_',1)
            a=q.pivot(index='source_video_id',columns=['policy','pool'],values='correct').loc[sources]
            d=100*(a[(policy,p)].to_numpy()-a[('periodic',p)].to_numpy())
        bounds=np.percentile(d[idx].mean(1),[2.5,97.5],method='linear')
        stored=saved_pairs[(saved_pairs.cohort==cohort)&(saved_pairs.representation==interval['representation'])&
            (saved_pairs.budget==interval['budget'])&(saved_pairs.contrast==name)].set_index('source_video_id').loc[sources].value.to_numpy()
        np.testing.assert_allclose(stored,d,atol=1e-12,rtol=0)
        np.testing.assert_allclose([d.mean(),*bounds],[interval['mean'],interval['CI95_lower'],interval['CI95_upper']],atol=1e-12,rtol=0)
        endpoint_checks+=1
    for result in read(output/'aggregate_metrics.json')['rows']:
        q=raw[(raw.cohort==result['cohort'])&(raw.representation==result['representation'])&(raw.budget==result['budget'])&(raw.policy==result['policy'])&(raw.pool==result['pool'])]
        assert len(q)==100
        np.testing.assert_allclose([q.CE.mean(),100*q.correct.mean(),q.feature_error.mean()],
            [result['CE']['mean'],result['accuracy_pp']['mean'],result['feature_error']],atol=1e-12,rtol=0)
    timings=pool.frame(output/'raw_timing_trials.csv'); assert len(timings)==600
    assert (timings.groupby(['source_video_id','representation','mode']).size()==5).all()
    assert set(timings.source_video_id)==set(manifest['timing_sources'])
    for r in timings.itertuples():
        assert r.decoded_frames==entry_lookup[r.source_video_id]['decoded_frame_count']
        assert r.encoder_calls==r.refreshed_candidates*(4 if r.representation==REPS[1] else 1)
    for rep in REPS:
        selection=read(output/'head_selection.json')[rep]; head=read(ROOT/selection['path']); assert sha(ROOT/selection['path'])==selection['sha256']
        # Independently verify normalization uses all400 training dense means only.
        ledger=read(output/'train_development_collection.json')['videos']; training=[]
        for v in train:
            if v['recognition_role']=='train':
                with np.load(ROOT/ledger[v['source_video_id']]['path']) as f: training.append(f[rep].mean(0,dtype=np.float64))
        x=np.array(training); means=x.mean(0); std=x.std(0)
        np.testing.assert_allclose(means,head['parameters']['means'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.where(std<=1e-12,1,std),head['parameters']['scales'],atol=1e-12,rtol=0)
    # Historical tracked scientific files must have no worktree modification.
    changed=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).splitlines()
    allowed={'README.md','.gitignore','THIRD_PARTY_NOTICES.md'}
    historical=set(read(output/'run_configuration.json')['initial_tracked_files'])
    assert not (set(changed)&historical)-allowed, (set(changed)&historical)-allowed
    release=output/'release_validation.json'
    if not release.exists():
        check=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'validate_release.py')],cwd=ROOT,text=True,capture_output=True)
        (output/'release_validation.txt').write_text(check.stdout+'\n'+check.stderr,encoding='utf-8')
        save(release,{'exit_code':check.returncode,'executed_at_utc':now(),'historical_environment':True})
    assert read(release)['exit_code']==0,'Existing release validator failed; see preserved log'
    return {'status':'passed','verified_at_utc':now(),'source_counts':{'train':400,'development':100,'fresh_evaluation':100},
        'source_and_exact_hash_disjointness':True,'evaluation_class_counts':dict(Counter(v['label'] for v in videos)),
        'protected_outcome_access':False,'frozen_seal_and_pre_exposure_order':True,'reference_features':feature_count,
        'real_temporal_grid_checks':grid_checks,'independent_selected_pool_checks':explicit_checks,
        'maximum_independent_CE_difference':max_difference,'independent_bootstrap_endpoint_checks':endpoint_checks,
        'independent_real_schedule_outcomes_checked':real_exhaustive_checks,
        'quality_rows':len(raw),'timing_measured_rows':len(timings),'training_only_normalization':True,
        'historical_scientific_files_unchanged':True,'release_validation_passed':True,
        'limits':'hashes/parent IDs do not establish no reuploads, full checkpoint pretraining independence or universal dataset independence'}


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    aggregate=read(output/'aggregate_metrics.json')['rows']; intervals=read(output/'bootstrap_intervals.json')['rows']
    raw=pool.frame(output/'per_video_quality.csv'); timings=pool.frame(output/'timing_per_video.csv')
    figure_dir=output/'figures'; figure_dir.mkdir(exist_ok=True)
    lookup=lambda cohort,rep,b,name:next(r for r in intervals if r['cohort']==cohort and r['representation']==rep and r['budget']==b and r['contrast']==name)
    agg=lambda cohort,rep,b,policy,p:next(r for r in aggregate if (r['cohort'],r['representation'],r['budget'],r['policy'],r['pool'])==(cohort,rep,b,policy,p))
    names=['RN18 endpoint','RN18 four-frame mean','X3D-XS']
    def finish(fig,name):
        fig.tight_layout(); fig.savefig(figure_dir/f'{name}.png',dpi=160);fig.savefig(figure_dir/f'{name}.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,cohort in zip(axes,['development','evaluation']):
        values=[lookup(cohort,r,4,'feature_minus_periodic_C') for r in REPS]
        means=np.array([r['mean'] for r in values]); lo=np.array([r['CI95_lower'] for r in values]); hi=np.array([r['CI95_upper'] for r in values])
        ax.vlines(np.arange(3),lo,hi,color='C0');ax.scatter(np.arange(3),means,color='C0')
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),names,rotation=15)
        ax.set_title('Reused development (100)' if cohort=='development' else 'New-source evaluation (100)')
        ax.set_ylabel('CE(feature oracle) − CE(periodic), nats')
    fig.suptitle('B4 cache pooling: positive = worse CE for privileged feature selection\nX3D-XS evaluation is primary; other cells secondary; paired95% intervals')
    finish(fig,'primary_paired_effects')
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,cohort in zip(axes,['development','evaluation']):
        for j,name in enumerate(['H_C','H_U']):
            vals=[lookup(cohort,r,4,name) for r in REPS]; m=np.array([v['mean'] for v in vals]);lo=np.array([v['CI95_lower'] for v in vals]);hi=np.array([v['CI95_upper'] for v in vals])
            x=np.arange(3)+j*.15;ax.vlines(x,lo,hi,color=f'C{j}');ax.scatter(x,m,color=f'C{j}',label='Cache task oracle' if j==0 else 'Unique task oracle')
        ax.set_xticks(range(3),names,rotation=15);ax.set_title(cohort);ax.set_ylabel('Periodic CE − own-pool task-oracle CE (nats)');ax.legend(fontsize=8)
    fig.suptitle('B=4 privileged label-aware headroom, not deployable performance');finish(fig,'oracle_headroom')
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for i,cohort in enumerate(['development','evaluation']):
        for j,rep in enumerate(REPS):
            matrix=np.array([[lookup(cohort,rep,4,n)['mean'] for n in ['L_C_SC','L_U_SC']],
                             [lookup(cohort,rep,4,n)['mean'] for n in ['L_C_SU','L_U_SU']]])
            vmax=max(v['mean'] for v in intervals if v['contrast'].startswith('L_'))
            ax=axes[i,j]; ax.imshow(matrix,cmap='Blues',vmin=0,vmax=vmax)
            for (row,col),value in np.ndenumerate(matrix):ax.text(col,row,f'{value:.3f}',ha='center',va='center',color='white' if value>.5*vmax else 'black')
            ax.set_xticks([0,1],['Cache pool','Unique pool']);ax.set_yticks([0,1],['C-optimal frames','U-optimal frames']);ax.set_title(f'{cohort}: {names[j]}')
    fig.suptitle('B=4 privileged selection/pooling cross-evaluation; mean CE in nats');finish(fig,'pooling_cross_evaluation')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    timing_ids=set(timings.source_video_id)
    for j,rep in enumerate(REPS):
        for mode,policy,b,p,marker in [('dense16','dense16',16,'C=U','o'),('periodic_B4','periodic',4,'C','s')]:
            t=timings[(timings.representation==rep)&(timings['mode']==mode)]
            q=raw[(raw.cohort=='evaluation')&raw.source_video_id.isin(timing_ids)&(raw.representation==rep)&(raw.policy==policy)&(raw.budget==b)&(raw.pool==p)]
            axes[0].scatter(t.median_ms.mean(),100*q.correct.mean(),marker=marker,label=f'{names[j]} {mode}')
            axes[1].bar(j*3+(0 if b==16 else 1),t.median_ms.mean(),yerr=[[t.median_ms.mean()-t.min_ms.mean()],[t.max_ms.mean()-t.median_ms.mean()]],capsize=3)
    axes[0].set_xlabel('Mean of per-video median encoded-memory latency (ms)');axes[0].set_ylabel('Top1 on same20 evaluation sources (%)');axes[0].legend(fontsize=7)
    axes[1].set_xticks([.5,3.5,6.5],names,rotation=12);axes[1].set_ylabel('Latency (ms); dense then periodic\nWhiskers: mean trial min/max')
    fig.suptitle('Actual dense/periodic systems only;20 sources,1/class; no free-oracle latency');finish(fig,'measured_quality_latency')
    primary=lookup('evaluation','x3d_xs',4,'feature_minus_periodic_C')
    interpretation=('supports harmful reconstruction/recognition mismatch in this temporal setting' if primary['CI95_lower']>0 else
        'supports feature-fidelity selection here and contradicts a broad mismatch claim' if primary['CI95_upper']<0 else 'is inconclusive at this precision; it is not evidence of equivalence')
    resource=read(output/'download_accounting.json');exclusions=read(output/'source_exclusions.json');acquisition=read(output/'acquisition.json')
    rejected={e['source_video_id'] for e in acquisition['events'] if not e['eligible']}
    selected_heads=read(output/'head_selection.json')
    lines=['# temporal_refresh_generalization_v1 findings','',
        f"Primary new-source X3D-XS B4 cache effect: **{primary['mean']:+.6f} nats/video**, paired95%CI [{primary['CI95_lower']:+.6f}, {primary['CI95_upper']:+.6f}],100 sources5/class. Positive means CE(feature oracle)>CE(periodic), hence worse classification loss. This {interpretation}.",'',
        f"Actual starting revision: {read(output/'run_configuration.json')['starting_commit']}; tracked inputs match the cited design revision. All400 training and100 development source roles are retained. New evaluation has100 distinct parents,5/class;{len(exclusions['source_ids'])} prior/reserved parent IDs and{len(exclusions['encoded_sha256'])} encoded hashes excluded. {len(rejected)} distinct technically ineligible candidates were rejected before freeze. New HTTP body downloads: {resource['actual_body_bytes']/1024**3:.6f}GiB; conservative interruption charge: {resource['interruption_charge_bytes']}bytes;20GiB cap. Reused compressed archives are accounted separately in data_plan.json.",'',
        '## Frozen recognition heads','',
        '|Representation|Selected alpha|Train dense accuracy|Train CE|Development dense accuracy|Development CE|LBFGS iterations|Maximum objective gradient|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for rep,head in selected_heads.items():
        lines.append(f"|{rep}|{head['alpha']}|{100*head['train_accuracy']:.1f}%|{head['train_CE']:.6f}|{100*head['development_accuracy']:.1f}%|{head['development_CE']:.6f}|{head['solver']['iterations']}|{head['solver']['gradient_max_abs']:.3g}|")
    lines+=['','All grids and solver records are retained. Dense sanity checks are reported without filtering sources or requiring an arbitrary favorable accuracy.','',
        '## Quality, kept separate by cohort','',
        '|Cohort|Representation|Dense accuracy|Periodic B4 accuracy|Feature-oracle B4 accuracy|Periodic CE|Feature-oracle CE|d and95%CI|',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for cohort in ['development','evaluation']:
        for rep in REPS:
            dense=agg(cohort,rep,16,'dense16','C=U');p=agg(cohort,rep,4,'periodic','C');f=agg(cohort,rep,4,'feature_oracle','C');d=lookup(cohort,rep,4,'feature_minus_periodic_C')
            lines.append(f"|{cohort}|{rep}|{dense['accuracy_pp']['mean']:.1f}%|{p['accuracy_pp']['mean']:.1f}%|{f['accuracy_pp']['mean']:.1f}%|{p['CE']['mean']:.6f}|{f['CE']['mean']:.6f}|{d['mean']:+.6f} [{d['CI95_lower']:+.6f},{d['CI95_upper']:+.6f}]|")
    lines+=['','Development is reused/tuned; evaluation uses new source IDs, frozen heads/settings/code and historical exclusions. It is not pooled with exploratory data. All20 classes are the existing classes. Official training annotation parent overlap is excluded, but reuploads and universal checkpoint pretraining independence cannot be proved.','',
        '## Privileged task selection and pooling','',
        '|Cohort|Representation|H_C|H_U|L_C(S_C)|L_U(S_C)|L_C(S_U)|L_U(S_U)|','|---|---|---:|---:|---:|---:|---:|---:|']
    for cohort in ['development','evaluation']:
        for rep in REPS:
            vals=[lookup(cohort,rep,4,n)['mean'] for n in ['H_C','H_U','L_C_SC','L_U_SC','L_C_SU','L_U_SU']]
            lines.append('|'+cohort+'|'+rep+'|'+'|'.join(f'{v:.6f}' for v in vals)+'|')
    lines+=['','SC/SU use true labels and every candidate feature. Cross-cells hold selected positions fixed; reoptimizing under the other pool changes both selection and weighting. Feature reconstruction error remains a last-cache-timeline metric. Positive task-oracle headroom is guaranteed by including periodic, not evidence a practical controller recovers it. CE minimization does not maximize top1.','',
        '## Predetermined budget sensitivities','', '|Representation|B|New-source d (nats)|95%CI|','|---|---:|---:|---|']
    for rep in REPS:
        for b in [2,8]:
            d=lookup('evaluation',rep,b,'feature_minus_periodic_C');lines.append(f"|{rep}|{b}|{d['mean']:+.6f}|[{d['CI95_lower']:+.6f},{d['CI95_upper']:+.6f}]|")
    lines+=['','These are secondary marginal/unadjusted comparisons. The primary representation,budget,pool and sign were fixed before evaluation.','',
        '## Measured system costs','', '|Representation|Dense16 mean per-video median ms|PeriodicB4 mean per-video median ms|','|---|---:|---:|']
    summary=[]
    for rep in REPS:
        vals=[]
        for mode in ['dense16','periodic_B4']:
            q=timings[(timings.representation==rep)&(timings['mode']==mode)];value=float(q.median_ms.mean());vals.append(value)
            summary.append({'representation':rep,'mode':mode,'sources':20,'mean_of_per_video_medians_ms':value,'median_of_per_video_medians_ms':float(q.median_ms.median()),'mean_trial_min_ms':float(q.min_ms.mean()),'mean_trial_max_ms':float(q.max_ms.mean())})
        lines.append(f'|{rep}|{vals[0]:.3f}|{vals[1]:.3f}|')
    pd.DataFrame(summary).to_csv(output/'timing_summary.csv',index=False,float_format='%.17g')
    latency_pairs=[]
    for (source,rep),g in timings.groupby(['source_video_id','representation']):
        cells=g.set_index('mode');a=float(cells.loc['dense16'].median_ms);b=float(cells.loc['periodic_B4'].median_ms)
        latency_pairs.append({'source_video_id':source,'representation':rep,'dense16_median_ms':a,'periodic_B4_median_ms':b,
            'dense_minus_periodic_ms':a-b,'measured_dense_over_periodic_ratio':a/b})
    pd.DataFrame(latency_pairs).to_csv(output/'timing_paired_comparisons.csv',index=False,float_format='%.17g')
    events=Counter()
    for line in (output/'encoder_execution_events.jsonl').read_text().splitlines():
        e=json.loads(line);events[f"{e['phase']}:{e['model']}"]+=1
    save(output/'execution_accounting.json',{'logged_collection_smoke_forward_starts':dict(events),
        'timing_forwards_including_warmups':read(output/'timing_protocol.json')['forward_counts_total_including_warmups'],
        'note':'extra smoke/native-verification forwards counted separately, not just retained reference count; timing records no per-forward file I/O'})
    next_action=('Investigate recognition-aligned selection signals in a separately frozen study with new-source evaluation; retain feature-fidelity selection as a diagnostic comparator, not an assumed task-quality proxy.' if primary['CI95_lower']>0 else
        'Replicate feature-to-task alignment on a separately frozen temporal task before making a broad mismatch claim; this result favors feature-fidelity selection in the tested setting, but the oracle still requires dense features.' if primary['CI95_upper']<0 else
        'Plan one precision-justified new-source replication of this fixed primary comparison before building another controller; the present interval does not resolve the proxy question.')
    lines+=['','The boundary starts with encoded bytes in memory and includes a fresh full sequential decode,within-trial frame buffering and timestamp candidate planning,RGB preparation,transfers,float32 encoders,float64 pooling/head and completed prediction. Disk loading/setup are separate. No predecoded input or free candidate inspection is used. C also executes its native400-class projection before discarding those logits; that implementation cost is included. Two warmups and5 measured trials/mode/source;20 fixed evaluation sources1/class. Raw decoded/call/image counters,trial variability,throughput and memory scopes are saved. A uses16/4 frame calls;B64/16 independent frame calls;C16/4 four-frame calls. Equal refresh budgets are not equal compute. Memory reports torch allocated resident and per-trial incremental peaks,plus process working set; it is not whole-device physical or exact per-trial CPU peak memory. No oracle is assigned measured deployable latency.','',
        '## Limits and next decision','',
        'The400/100 reused source roles replace the older600-new-source proposal. Three new heads select regularization by lowest dense development CE (historically accuracy first), then accuracy/grid order. Training-only normalization and solver diagnostics are saved. Confidence intervals use10000 class-stratified whole-source resamples,PCG64 seed20261024,linear percentiles,shared indices. They condition on fixed heads/classes and omit training/model-selection/repeated-research uncertainty. Development-derived precision was estimated before evaluation;100 sources did not guarantee a conclusive interval.','',
        'A/B/C use the same endpoints and four-frame support. B-versus-C does not isolate a pure temporal-model effect: architecture,pretraining,resolution,feature dimension,accuracy and calibration differ. Full-clip candidate construction is offline even with closed trailing snippets. This study says nothing definitive about whether earlier gain predictors work on X3D. These exposed evaluation sources cannot become untouched confirmation for a later controller. No adaptive speedup,streaming system,recognition guarantee or publication acceptance is established.','',
        'Next: '+next_action+' No follow-up acquisition or controller training is launched.','',
        '## Reproduction','', '```powershell',
        '& .\\.venv\\Scripts\\python.exe .\\run_temporal_refresh_generalization.py --device cuda --max-download-gib 20 --resume',
        '& .\\.venv_temporal\\Scripts\\python.exe -m unittest test_temporal_refresh_generalization -v',
        '& .\\.venv_temporal\\Scripts\\python.exe .\\run_temporal_refresh_generalization.py --audit-only', '```']
    (output/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'README.md').write_text('# Temporal refresh generalization v1\n\nSee findings.md for the primary sign/results and protocol.md plus evaluation_seal.json for the frozen scope.\n\nLarge encoded inputs,checkpoint,dependencies and reference arrays remain under ignored data/source/temporal_refresh_generalization_v1/,with relative paths and SHA256 identities in manifests/collection ledgers. Reproduction requires those assets or unchanged source acquisition. No protected outcomes are used.\n',encoding='utf-8')
