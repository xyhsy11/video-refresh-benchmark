"""Larger fixed-feature recognition baseline: acquire, collect, fit and audit."""
from __future__ import annotations

import os
for variable in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]: os.environ[variable]="1"

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

import recognition_baseline_data as data
import task_aware_oracle_utils as u
from fresh_confirmation_data import strict_json
from research_utils import ROOT,now,relative_path,sha256_file

SOURCES=["run_recognition_baseline_v2.py","recognition_baseline_data.py","recognition_baseline_report.py","test_recognition_baseline_v2.py",
         "task_aware_oracle_utils.py","fresh_confirmation_data.py","fresh_confirmation_eval.py","collect_cache_drift_v2.py","prepare_cache_drift_v2.py","benchmark.py","motion_utils.py","research_utils.py"]
POLICIES=["dense","periodic","feature_oracle","task_oracle"]
COMPARISONS=[("task_oracle","periodic"),("feature_oracle","periodic"),("task_oracle","feature_oracle"),("periodic","dense")]


def csv(path:Path,table:pd.DataFrame): table.to_csv(path,index=False,float_format="%.17g",na_rep="")


def aggregate(raw:pd.DataFrame) -> dict:
    rows=[]
    for (size,role,policy),g in raw.groupby(['training_per_class','cohort','policy'],sort=True):
        rows.append({'training_per_class':int(size),'training_videos':int(size)*20,'cohort':role,'policy':policy,'videos':len(g),
                     'correct_videos':int(g.correct.sum()),'top1_accuracy':float(g.correct.mean()),'macro_class_accuracy':float(g.groupby('label').correct.mean().mean()),
                     'cross_entropy_nats':float(g.cross_entropy.mean()),'mean_feature_cosine_error':float(g.feature_cosine_error.mean()),'simulated_refreshes':16 if policy=='dense' else 4})
    return {'rows':rows,'aggregation':'equal source-video weight; cohorts reported separately'}


def statistics(raw:pd.DataFrame) -> tuple[dict,pd.DataFrame]:
    rows=[];pairs=[]
    primary=raw[raw.training_per_class==20]
    for cohort,g in primary.groupby('cohort',sort=True):
        sources=sorted(g.source_video_id.unique()); n=len(sources)
        idx=np.random.Generator(np.random.PCG64(data.SEED)).integers(0,n,size=(10000,n),dtype=np.int64)
        digest=hashlib.sha256(idx.tobytes()).hexdigest()
        for candidate,reference in COMPARISONS:
            for metric in ['correct','cross_entropy']:
                pivot=g.pivot(index='source_video_id',columns='policy',values=metric).loc[sources]
                delta=(pivot[candidate]-pivot[reference]) if metric=='correct' else (pivot[reference]-pivot[candidate])
                result=u.bootstrap_difference(delta.to_numpy(),idx)
                rows.append({'training_per_class':20,'cohort':cohort,'candidate':candidate,'reference':reference,'metric':metric,
                             'primary':candidate=='task_oracle' and reference=='periodic','videos':n,'resampling_indices_sha256':digest,
                             'accuracy_difference_pp':100*result['mean'] if metric=='correct' else None,**result})
                pairs.extend({'cohort':cohort,'source_video_id':source,'candidate':candidate,'reference':reference,'metric':metric,'paired_gain':float(delta.loc[source])} for source in sources)
    return {'seed':data.SEED,'resamples':10000,'rng':'PCG64','percentile_method':'linear','unit':'whole source-video pairs; fixed models; shared indices within cohort',
            'interpretation':'conditional on fitted classifiers and sampled classes; no training uncertainty; development tuned; exploratory previously evaluated; oracle privileged', 'comparisons':rows},pd.DataFrame(pairs)


def evaluate(videos:list,parameters:dict,classes:list,size:int,cohort:str) -> list:
    rows=[]
    for v in videos:
        logits=u.feature_logits(v.references,parameters);y=classes.index(v.entry['label'])
        d=u.budget_utils.distance_matrix(v.references)
        feature=u.budget_utils.exact_oracle(d,4);task=u.exact_task_oracle(logits,y,4)
        assert feature['enumerated_schedules']==task['enumerated_schedules']==455
        schedules={'dense':list(range(16)),'periodic':[0,4,8,12],'feature_oracle':feature['positions'],'task_oracle':task['positions']}
        for policy,positions in schedules.items():
            cached=u.cache_indices(positions); output=logits[cached].mean(axis=0)
            explicit=u.feature_logits(v.references[cached].mean(axis=0,dtype=np.float64)[None],parameters)[0]
            np.testing.assert_allclose(output,explicit,rtol=1e-10,atol=1e-10)
            loss=float(u.classification_loss(output[None],y)[0]);error=float(d[np.arange(16),cached].mean())
            assert len(positions)==(16 if policy=='dense' else 4)
            rows.append({'training_per_class':size,'cohort':cohort,'video_id':v.entry['video_id'],'source_video_id':v.entry['source_video_id'],
                         'label':v.entry['label'],'label_index':y,'policy':policy,'correct':int(output.argmax()==y),'prediction_index':int(output.argmax()),
                         'cross_entropy':loss,'feature_cosine_error':error,'simulated_refreshes':len(positions),'feature_outputs':16,'cache_reuses':16-len(positions),
                         'refresh_positions':json.dumps(positions),'logits':json.dumps(output.tolist()),'oracle_enumerated_schedules':455 if 'oracle' in policy else 0})
        local={r['policy']:r for r in rows[-4:]}
        assert local['task_oracle']['cross_entropy']<=local['periodic']['cross_entropy']+u.TOL
        assert local['feature_oracle']['feature_cosine_error']<=local['periodic']['feature_cosine_error']+u.TOL
    return rows


def audit(output:Path) -> dict:
    manifest=data.read(output/'dataset_manifest.json');entries=manifest['videos'];data.validate_membership(entries)
    subset=data.read(output/'training_subsets.json')['subsets']
    assert subset==data.subsets(entries,manifest['classes'])
    metadata=data.read(output/'metadata.json')
    assert sha256_file(output/'dataset_manifest.json')==metadata['dataset_manifest_sha256']
    for name,digest in metadata['protected_original_sha256'].items(): assert sha256_file(ROOT/name)==digest,name
    for name,digest in metadata['source_code_sha256'].items(): assert sha256_file(ROOT/name)==digest,name
    raw=pd.read_csv(output/'per_video_metrics.csv',float_precision='round_trip')
    assert len(raw)==1680 and not raw.duplicated(['training_per_class','cohort','video_id','policy']).any()
    regenerated=[]
    for size in [4,10,20]:
        path=output/'classifiers'/str(size)/'parameters.json';parameters=data.read(path)
        assert sha256_file(path)==metadata['classifier_sha256'][str(size)]
        training=data.load_role(entries,'train');selected=[v for v in training if v.entry['source_video_id'] in subset[str(size)]]
        x=np.stack([v.references.mean(axis=0,dtype=np.float64) for v in selected]);means=x.mean(axis=0);std=x.std(axis=0,ddof=0)
        np.testing.assert_allclose(parameters['means'],means,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(parameters['scales'],np.where(std<=u.TOL,1,std),atol=1e-12,rtol=1e-12)
        for cohort in ['development','exploratory']:
            regenerated.extend(evaluate(data.load_role(entries,cohort),parameters,manifest['classes'],size,cohort))
    recomputed=pd.DataFrame(regenerated).sort_values(['training_per_class','cohort','video_id','policy']).reset_index(drop=True)
    saved=raw.sort_values(['training_per_class','cohort','video_id','policy']).reset_index(drop=True)
    pd.testing.assert_frame_equal(recomputed,saved,check_dtype=False,rtol=1e-10,atol=1e-10)
    # Independent loss from saved logits (subtract-max formula, not logaddexp).
    for row in raw.itertuples():
        logits=np.array(json.loads(row.logits));high=logits.max()
        assert abs(np.log(np.exp(logits-high).sum())+high-logits[row.label_index]-row.cross_entropy)<=1e-10
        assert int(logits.argmax()==row.label_index)==row.correct
    assert aggregate(raw)==data.read(output/'aggregate_metrics.json')
    intervals,pairs=statistics(raw);assert intervals==data.read(output/'bootstrap_intervals.json')
    for r in intervals['comparisons']:
        deltas=pairs[(pairs.cohort==r['cohort'])&(pairs.candidate==r['candidate'])&(pairs.reference==r['reference'])&(pairs.metric==r['metric'])].sort_values('source_video_id').paired_gain.to_numpy()
        idx=np.random.default_rng(data.SEED).integers(0,len(deltas),size=(10000,len(deltas)),dtype=np.int64)
        bounds=np.quantile([sum(deltas[s])/len(deltas) for s in idx],[.025,.975],method='linear')
        np.testing.assert_allclose(bounds,[r['CI95_lower'],r['CI95_upper']],atol=1e-12,rtol=1e-12)
    return {'passed':True,'source_hash_separation_and_class_balance':True,'nested_subsets_and_train_only_scalers':True,'same_classifier_all_policies':True,
            'outputs_refreshes_latest_cache_exact_oracles':True,'all_saved_predictions_and_aggregate_metrics_reproduced':True,'bootstrap_independent_loop_reproduced':True,
            'evaluation_rows':1680,'prior_artifacts_preserved':len(metadata['protected_original_sha256'])}


def run(args):
    output=args.output
    if output.exists() and not args.resume and not args.audit_only and not args.preflight_only:
        raise RuntimeError('Existing run preserved; use --resume or --audit-only')
    if args.audit_only: print(audit(output));return
    protocol=data.initialize(output,args.max_download_gib,SOURCES)
    status_path=output/'status.json'
    if not status_path.exists():
        old=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
        strict_json(status_path,{'status':'preflight','started_at_utc':now(),'protected_original_sha256':{n:sha256_file(ROOT/n) for n in old if n and n not in ['README.md','.gitignore']}})
    status=data.read(status_path)
    for name,digest in status['protected_original_sha256'].items(): assert sha256_file(ROOT/name)==digest,name
    print(json.dumps(data.read(output/'preflight.json'),indent=2),flush=True)
    if args.preflight_only: return
    if status['status']=='complete': print(audit(output));return
    try:
        entries=data.read(output/'dataset_manifest.json')['videos'] if (output/'dataset_manifest.json').exists() else data.acquire(output,protocol,args.download_workers)
        data.validate_membership(entries)
        chosen=data.subsets(entries,protocol['classes']);strict_json(output/'training_subsets.json',{'seed':data.SEED,'rule':'per-class SHA256 seeded source ordering,first4/10/20; newly nested80 differs from v1', 'subsets':chosen})
        train=data.load_role(entries,'train');dev=data.load_role(entries,'development')
        models={};curve=[];all_grid=[];hashes={}
        for size in [4,10,20]:
            folder=output/'classifiers'/str(size);folder.mkdir(parents=True,exist_ok=True)
            subset=[v for v in train if v.entry['source_video_id'] in chosen[str(size)]]
            x=np.stack([v.references.mean(axis=0,dtype=np.float64) for v in subset]);y=np.array([protocol['classes'].index(v.entry['label']) for v in subset])
            options=[];grid=[]
            if (folder/'selection.json').exists():
                model=data.read(folder/'parameters.json');selection=data.read(folder/'selection.json');grid=selection['grid']
            else:
                for order,alpha in enumerate(protocol['regularization_grid']):
                    p,info=u.fit_classifier(x,y,alpha,data.SEED);tr,_=u.dense_metrics(subset,protocol['classes'],p);dv,_=u.dense_metrics(dev,protocol['classes'],p)
                    grid.append({'training_per_class':size,'grid_order':order,**info,'training_accuracy':tr['top1_accuracy'],'training_CE':tr['cross_entropy'],'development_accuracy':dv['top1_accuracy'],'development_CE':dv['cross_entropy']});options.append(p)
                    print(f"N/class={size},lambda={alpha}: train {tr['top1_accuracy']:.3f},dev {dv['top1_accuracy']:.3f}",flush=True)
                winner=min(range(4),key=lambda i:(-grid[i]['development_accuracy'],grid[i]['development_CE'],i))
                model=options[winner]
                strict_json(folder/'parameters.json',model)
                selection={'frozen_at_utc':now(),'grid':grid,'selected_grid_order':winner,'selected_regularization':grid[winner]['regularization'],'training_source_ids':chosen[str(size)],'training_manifest_sha256':sha256_file(output/'training_subsets.json'),'parameters_sha256':sha256_file(folder/'parameters.json'),'exploratory_metrics_used':False}
                strict_json(folder/'selection.json',selection)
            assert selection['training_source_ids']==chosen[str(size)] and selection['parameters_sha256']==sha256_file(folder/'parameters.json')
            models[size]=model;hashes[str(size)]=sha256_file(folder/'parameters.json');all_grid.extend(grid)
            for cohort,videos in [('training',subset),('development',dev)]:
                result,rows=u.dense_metrics(videos,protocol['classes'],model)
                curve.append({'training_per_class':size,'cohort':cohort,**result,'selected_regularization':selection['selected_regularization']})
        strict_json(output/'frozen_models.json',{'frozen_at_utc':now(),'parameter_sha256':hashes,'all_models_frozen_before_exploratory_data_evaluation':True})
        # Original exploratory feature arrays only loaded after all twelve grid fits/three choices freeze.
        exploratory=data.load_role(entries,'exploratory');rows=[]
        with patch.object(u,'fit_classifier',side_effect=AssertionError('All classifiers frozen; no fitting in evaluation')):
            for size,p in models.items():
                result,_=u.dense_metrics(exploratory,protocol['classes'],p);curve.append({'training_per_class':size,'cohort':'exploratory',**result})
                for cohort,videos in [('development',dev),('exploratory',exploratory)]:rows.extend(evaluate(videos,p,protocol['classes'],size,cohort))
        raw=pd.DataFrame(rows);csv(output/'per_video_metrics.csv',raw);csv(output/'regularization_grid.csv',pd.DataFrame(all_grid))
        strict_json(output/'learning_curve.json',{'rows':curve,'majority_and_chance_accuracy':.05})
        strict_json(output/'aggregate_metrics.json',aggregate(raw));intervals,pairs=statistics(raw)
        strict_json(output/'bootstrap_intervals.json',intervals);csv(output/'paired_comparisons.csv',pairs)
        metadata={'status':'evaluated_pending_verification','starting_commit':protocol['starting_commit'],'source_code_sha256':{n:sha256_file(ROOT/n) for n in SOURCES},
                  'protected_original_sha256':status['protected_original_sha256'],'dataset_manifest_sha256':sha256_file(output/'dataset_manifest.json'),
                  'classifier_sha256':hashes,'reference_collection_encoder_calls':data.read(output/'acquisition.json')['actual_new_reference_forward_calls'],'replay_encoder_calls':0,'completed_evaluation_at_utc':now()}
        strict_json(output/'metadata.json',metadata)
        verification=audit(output)
        from recognition_baseline_report import report
        report(output)
        tests=subprocess.run([sys.executable,'-m','unittest','test_recognition_baseline_v2','-v'],cwd=ROOT,capture_output=True,text=True)
        assert tests.returncode==0,tests.stdout+tests.stderr
        release=subprocess.run([sys.executable,'validate_release.py'],cwd=ROOT,capture_output=True,text=True)
        assert release.returncode==0,release.stdout+release.stderr
        verification.update(focused_tests={'returncode':0,'stdout':tests.stdout,'stderr':tests.stderr},release_validator={'returncode':0,'stdout':release.stdout},frozen_before_exploratory_evaluation=True)
        strict_json(output/'verification.json',verification)
        status.update(status='complete',completed_at_utc=now(),blocker=None);strict_json(status_path,status)
        metadata.update(status='complete');strict_json(output/'metadata.json',metadata)
        files=sorted(p for p in output.rglob('*') if p.is_file() and p.suffix not in ['.tmp','.part'] and p.name!='SHA256SUMS.txt')
        (output/'SHA256SUMS.txt').write_text(''.join(f'{sha256_file(p)}  {p.relative_to(output).as_posix()}\n' for p in files),encoding='utf-8')
        print('Complete:',relative_path(output),flush=True)
    except Exception as exc:
        status.update(status='incomplete',blocked_at_utc=now(),blocker=f'{type(exc).__name__}: {str(exc).replace(str(ROOT),".")}')
        strict_json(status_path,status)
        from recognition_baseline_report import incomplete_report
        incomplete_report(output,status)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/recognition_baseline_v2'))
    parser.add_argument('--max-download-gib',type=float,default=20)
    parser.add_argument('--download-workers',type=int,default=8)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if args.max_download_gib<=0 or not 1<=args.download_workers<=16:parser.error('Positive download cap,1..16 workers required')
    run(args)
