"""Exact CPU saved-feature pooling control;no acquisition,encoder or fitting."""
from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import importlib.metadata
from math import comb
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import refresh_pooling_utils as u

ROOT=Path(__file__).resolve().parent
REVIEWED='bb1c6343d60e1872d458aaab519c8245634eb019'
SOURCES=['analyze_refresh_pooling_control.py','refresh_pooling_utils.py','verify_refresh_pooling_control.py',
         'refresh_pooling_report.py','test_refresh_pooling_control.py']
LEGACY=['task_aware_oracle_utils.py','refresh_policy_utils.py','collect_cache_drift.py',
        'recognition_baseline_data.py','run_recognition_baseline_v2.py']


def now():return datetime.now(timezone.utc).isoformat()
def rel(path):return path.relative_to(ROOT).as_posix()


def eligible_entries(directory:Path):
    manifest=u.read(directory/'dataset_manifest.json')
    assert u.sha(directory/'dataset_manifest.json')==u.read(directory/'metadata.json')['dataset_manifest_sha256']
    entries=[e for e in manifest['videos'] if e['recognition_role'] in ['development','exploratory']]
    assert len(entries)==len({e['source_video_id'] for e in entries})==len({e['file_identity']['sha256'] for e in entries})==140
    classes=manifest['classes'];assert len(classes)==len(set(classes))==20
    assert u.read(directory/'protocol.json')['class_to_index']=={c:i for i,c in enumerate(classes)}
    for role,count in [('development',5),('exploratory',2)]:
        group=[e for e in entries if e['recognition_role']==role]
        assert len(group)==count*20
        assert all(sum(e['label']==c for e in group)==count for c in classes)
    # Metadata-only check of separation;no training arrays loaded.
    tr={e['source_video_id'] for e in manifest['videos'] if e['recognition_role']=='train'}
    assert len(tr)==400 and tr.isdisjoint(e['source_video_id'] for e in entries)
    training_hashes={e['file_identity']['sha256'] for e in manifest['videos'] if e['recognition_role']=='train'}
    assert training_hashes.isdisjoint(e['file_identity']['sha256'] for e in entries)
    return manifest,sorted(entries,key=lambda e:(e['recognition_role'],e['source_video_id']))


def reference_path(entry):
    path=(ROOT/entry['reference_directory']/'features'/(entry['video_id']+'.npz')).resolve()
    assert path.is_relative_to(ROOT) and 'fresh_refresh_confirmation_v1' not in path.as_posix()
    return path


def load_reference(entry):
    path=reference_path(entry)
    expected=entry['reference_artifact_sha256'][f"features/{entry['video_id']}.npz"]
    assert u.sha(path)==expected,(rel(path),'reference hash mismatch')
    assert entry['video_id'].rsplit('_',2)[0]==entry['source_video_id']
    with np.load(path,allow_pickle=False) as saved:
        x=saved['features'];indices=saved['frame_indices'];timestamps=saved['timestamps_seconds']
        assert x.shape==(16,512) and x.dtype==np.dtype('float32') and np.isfinite(x).all()
        assert indices.shape==timestamps.shape==(16,) and np.isfinite(timestamps).all()
        assert np.issubdtype(indices.dtype,np.integer) and np.issubdtype(timestamps.dtype,np.floating)
        np.testing.assert_array_equal(indices,entry['candidate_indices'])
        np.testing.assert_array_equal(timestamps,entry['candidate_timestamps_seconds'])
        assert indices[0]==0 and indices[-1]==entry['decoded_frame_count']-1
        assert (np.diff(indices)>0).all() and (np.diff(timestamps)>0).all()
        assert np.linalg.norm(x.astype(float),axis=1).min()>u.TOL
        return x.copy()


def preflight(directory:Path):
    manifest,entries=eligible_entries(directory)
    head_path=directory/'classifiers/20/parameters.json'
    digest=u.sha(head_path);selection=u.read(directory/'classifiers/20/selection.json')
    assert digest==selection['parameters_sha256']==u.read(directory/'frozen_models.json')['parameter_sha256']['20']
    assert digest==u.read(directory/'metadata.json')['classifier_sha256']['20']
    assert selection['selected_regularization']==.01
    head=u.read(head_path)
    assert np.asarray(head['means']).shape==np.asarray(head['scales']).shape==(512,)
    assert np.asarray(head['weights']).shape==(20,512) and np.asarray(head['bias']).shape==(20,)
    assert all(np.isfinite(head[k]).all() for k in ['means','scales','weights','bias'])
    assert (np.asarray(head['scales'])>0).all()
    paths=[directory/f for f in ['dataset_manifest.json','protocol.json','metadata.json','frozen_models.json',
        'classifiers/20/parameters.json','classifiers/20/selection.json','per_video_metrics.csv','verification.json']]
    paths += [ROOT/name for name in LEGACY]
    paths += [ROOT/'results/refresh_generalization_audit_v1_rerun1'/name for name in ['findings.md','protocol.json','verification.json']]
    assert u.read(directory/'verification.json')['passed']
    provenance=[]
    for entry in entries:
        features=load_reference(entry)
        path=reference_path(entry);paths.append(path)
        provenance.append({'source_video_id':entry['source_video_id'],'video_id':entry['video_id'],
            'cohort':entry['recognition_role'],'label':entry['label'],'label_index':manifest['classes'].index(entry['label']),
            'encoded_sha256_from_manifest':entry['file_identity']['sha256'],
            'feature_path':rel(path),'feature_sha256':u.sha(path),'feature_shape':list(features.shape),
            'feature_dtype':str(features.dtype),'candidate_indices':entry['candidate_indices'],
            'candidate_timestamps_seconds':entry['candidate_timestamps_seconds']})
    diff=subprocess.check_output(['git','diff','--name-status',REVIEWED,'HEAD','--',*LEGACY,
        rel(directory), 'results/refresh_generalization_audit_v1_rerun1'],cwd=ROOT,text=True)
    return manifest,entries,head,{'head_sha256':digest,'regularization':.01,'class_order':manifest['classes'],
        'videos':provenance,'input_sha256':{rel(p):u.sha(p) for p in sorted(set(paths))},
        'reviewed_revision':REVIEWED,'relevant_changes_since_reviewed_revision':diff.splitlines(),
        'choice':'use actual starting-tree frozen artifacts after validating saved identities;no reset',
        'training_reference_arrays_opened':0,'protected_confirmation_arrays_or_outcomes_opened':0}


def protocol(args,manifest,provenance):
    tracked=subprocess.check_output(['git','ls-files','-s'],cwd=ROOT,text=True)
    old={line.split('\t',1)[1]:line.split()[1] for line in tracked.splitlines() if line.split('\t',1)[1] not in ['README.md','.gitignore']}
    return {'experiment':'refresh_pooling_control_v1','frozen_at_utc':now(),
        'scope':'frozen analysis protocol on reused data;not independent preregistration',
        'starting_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'reviewed_revision':REVIEWED,'reviewed_input_differences':provenance['relevant_changes_since_reviewed_revision'],
        'starting_worktree':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        'repository_instructions':'No AGENTS.md found in repository/ancestors;README and requested code/findings inspected',
        'previous_tracked_blob_identities':old,'source_sha256':{n:u.sha(ROOT/n) for n in SOURCES},
        'input_sha256':provenance['input_sha256'],'input':rel(args.input.resolve()),'head_sha256':provenance['head_sha256'],
        'recognition_head':'400-training-video primary head;20 means videos/class;regularization0.01;unchanged affine map/normalization',
        'classes':manifest['classes'],'cohorts':{c:[v['source_video_id'] for v in provenance['videos'] if v['cohort']==c] for c in ['development','exploratory']},
        'cohort_roles':'development100 used recognition regularization;exploratory40 previously inspected;neither fresh confirmation',
        'budgets':args.budgets,'primary_budget':4,'dense_budget':16,'candidates':16,
        'schedule_counts':{str(b):comb(15,b-1) for b in args.budgets},
        'feasible':'exact sorted distinct positions,mandatory0;lexicographic enumeration',
        'cache_pool':'z_C=sum_j((r_next-r_j)/16)*z_r;durations count candidate outputs,not elapsed seconds',
        'unique_pool':'z_U=mean logits at unique selected positions;not equivalent16 time-indexed outputs',
        'feature_error':'original mean all16 last-cache 1-clipped cosine;float64 safe norms>1e-12;independent of final classification pool',
        'policies':u.POLICIES,'oracles':'SF minimizes last-cache error;SC minimizes cache CE;SU minimizes unique CE;all privileged;task oracles use true label and future reference features',
        'RandomExact':'uniform over all feasible schedules;mean of individual CE/correctness/errors,not loss/accuracy of mean logits;not ensemble',
        'ties':{'tolerance':u.TOL,'rule':'lexicographically first within1e-12 of true minimum;save minimum,achieved value,tie count'},
        'numerics':{'dtype':'float64','CE':'stable logaddexp reduction as archived;independent subtract-max check',
            'top1':'argmax first saved class index in tie','reference_atol':u.ATOL,'reference_rtol':u.RTOL,
            'CSV':'round-trip float parser;exact null masks;no NaN/Infinity JSON'},
        'primary':'development B4 mean Delta=H_C-H_U=L_U(S_U)-L_C(S_C);positive means larger cache-optimized headroom',
        'contrasts':{'H_C':'L_C(P)-L_C(S_C)','H_U':'L_U(P)-L_U(S_U)','Delta':'H_C-H_U',
            'W_C':'L_U(S_C)-L_C(S_C)','W_U':'L_U(S_U)-L_C(S_U)',
            'accounting':'H_C=[L_U(P)-L_U(S_C)]+W_C;path-dependent algebra,not causal attribution',
            'matrix':['L_C(S_C)','L_U(S_C)','L_C(S_U)','L_U(S_U)'],
            'secondary':'exploratory B4 and both cohorts B2/B8;fixed-schedule CE/accuracy contrasts;feature oracle mismatch;schedule overlap/effective count'},
        'bootstrap':{'resamples':args.bootstrap,'seed':args.seed,'rng':'PCG64','percentile':'linear2.5/97.5',
            'unit':'whole original source videos,class-stratified,5/class development,2/class exploratory',
            'sharing':'one saved index matrix across all schedules,budgets,pools,comparisons within cohort;separate matrices',
            'limits':'frozen head/chosen classes;excludes training/model-selection/repeated-research uncertainty;two exploratory sources/class limited;secondary marginal unadjusted'},
        'feature_schema':'existing float32[16,512] references castfloat64;only140 permitted reference arrays;no training arrays or motion tables',
        'verification':'archived B4 choices/logits/losses/accuracy;explicit independent cache/unique feature replay for every selected schedule;scalar synthetic exhaustive minima/ties/random;independent saved-table bootstrap check',
        'figure_review':'completion requires explicit review of rendered figures and hashed review record after all scientific checks',
        'limits':['head trained on dense mean features;sparse pooling changes representation distribution',
            'no retrained best-case sparse heads','true-label hindsight CE optimization guarantees nonnegative headroom within numerical tolerance,not deployable accuracy',
            'full-clip candidates/offline;no latency or streaming claims','no unstable headroom ratios or semantic-importance percentage'],
        'actual_new_encoder_calls':0,'video_decodes':0,'video_downloads':0,'weight_downloads':0,'model_scaler_or_threshold_fits':0,'package_upgrades':0}


def unchanged(p):
    for n,h in p['source_sha256'].items():assert u.sha(ROOT/n)==h,n
    for n,h in p['input_sha256'].items():assert u.sha(ROOT/n)==h,n
    changed=subprocess.check_output(['git','diff','--name-only',p['starting_revision']],cwd=ROOT,text=True).splitlines()
    assert not [n for n in changed if n in p['previous_tracked_blob_identities']]


def ledger(output):
    files=sorted(f for f in output.rglob('*') if f.is_file() and f.name!='SHA256SUMS.txt')
    (output/'SHA256SUMS.txt').write_text(''.join(f'{u.sha(f)}  {f.relative_to(output).as_posix()}\n' for f in files),encoding='utf-8')


def selected_rows(entry,head,features,budget,table,result):
    positions=table['positions'];periodic=list(range(0,16,16//budget))
    pi=int(np.flatnonzero((positions==periodic).all(1))[0])
    choices={'periodic':pi,'feature_oracle':result['F']['index'],
             'cache_task_oracle':result['SC']['index'],'unique_task_oracle':result['SU']['index']}
    records,selection=[],{}
    base={'cohort':entry['recognition_role'],'source_video_id':entry['source_video_id'],'video_id':entry['video_id'],
          'label':entry['label'],'budget':budget}
    for policy,index in choices.items():
        schedule=positions[index].tolist();wc=table['cache_weights'][index,schedule];wu=table['unique_weights'][index,schedule]
        oracle=result['F'] if policy=='feature_oracle' else result['SC'] if policy=='cache_task_oracle' else result['SU'] if policy=='unique_task_oracle' else None
        selection[policy]={'index':index,'positions':schedule,'candidate_output_counts':table['durations'][index].tolist(),
            'cache_weights':wc.tolist(),'unique_weights':wu.tolist(),'cache_effective_count':float(1/(wc@wc)),
            'unique_effective_count':float(1/(wu@wu)),'oracle_objective':oracle,
            'logits_C':result['logits_C'][index].tolist(),'logits_U':result['logits_U'][index].tolist()}
        for pool,weights in [('C',wc),('U',wu)]:
            records.append({**base,'policy':policy,'pool':pool,'privileged':policy!='periodic','correctness_is_expectation':False,
                'schedule_index':index,'refresh_positions':json_string(schedule),'weights':json_string(weights.tolist()),
                'logits':json_string(result['logits_'+pool][index].tolist()),'undefined_output_reason':None,
                'CE':float(result['loss_'+pool][index]),'correct':float(result['correct_'+pool][index]),
                'prediction_index':int(result['prediction_'+pool][index]),'feature_error':float(result['feature_error'][index]),
                'feature_error_definition':'last-cache reconstruction;unchanged by final pool',
                'effective_weight_count':float(1/(weights@weights)),
                'simulated_refreshes':budget,'reconstruction_reference_timeline_length':16,
                'classification_pool_items':16 if pool=='C' else budget,'final_classification_outputs':1})
    for pool in ['C','U']:
        weights=table['cache_weights'] if pool=='C' else table['unique_weights']
        records.append({**base,'policy':'RandomExact','pool':pool,'privileged':False,'correctness_is_expectation':True,
            'schedule_index':None,'refresh_positions':None,'weights':None,'logits':None,'prediction_index':None,
            'undefined_output_reason':'uniform schedule distribution;no ensemble prediction or single schedule',
            'CE':float(result['loss_'+pool].mean()),'correct':float(result['correct_'+pool].mean()),
            'feature_error':float(result['feature_error'].mean()),'feature_error_definition':'expected last-cache reconstruction',
            'effective_weight_count':float((1/(weights*weights).sum(1)).mean()),
            'simulated_refreshes':budget,'reconstruction_reference_timeline_length':16,
            'classification_pool_items':16 if pool=='C' else budget,'final_classification_outputs':1})
    overlaps=[]
    for first,second in [('feature_oracle','cache_task_oracle'),('feature_oracle','unique_task_oracle'),('cache_task_oracle','unique_task_oracle')]:
        a,b=set(selection[first]['positions']),set(selection[second]['positions'])
        overlaps.append({**base,'first':first,'second':second,'overlap':len(a&b),'Jaccard':len(a&b)/len(a|b),'same_schedule':a==b})
    return records,selection,overlaps


def json_string(value):
    import json
    return json.dumps(value,separators=(',',':'),allow_nan=False)


def compute(output,entries,head,p):
    tables={b:u.schedules(16,b) for b in p['budgets']};canonical={}
    for b,table in tables.items():
        assert len(table['positions'])==comb(15,b-1)
        for key,value in table.items():canonical[f'b{b}_{key}']=value
        rows=[{'schedule_index':i,'positions':json_string(s.tolist()),'candidate_output_counts':json_string(table['durations'][i].tolist())} for i,s in enumerate(table['positions'])]
        u.csv(output/f'schedules_b{b}.csv',pd.DataFrame(rows))
    np.savez_compressed(output/'canonical_schedules.npz',**canonical)
    rows,selected,overlaps,random=[],[],[],[]
    folder=output/'per_schedule';folder.mkdir()
    for count,entry in enumerate(entries,1):
        x=load_reference(entry).astype(float);y=p['classes'].index(entry['label']);stored={}
        info={'cohort':entry['recognition_role'],'source_video_id':entry['source_video_id'],'video_id':entry['video_id'],'budgets':{}}
        for b,table in tables.items():
            result=u.evaluate(x,head,y,table)
            new,choices,common=selected_rows(entry,head,x,b,table,result)
            rows.extend(new);overlaps.extend(common);info['budgets'][str(b)]=choices
            for key in ['loss_C','loss_U','correct_C','correct_U','prediction_C','prediction_U','feature_error']:
                stored[f'b{b}_{key}']=result[key]
            for pool in ['C','U']:
                values=result['loss_'+pool]
                random.append({'cohort':entry['recognition_role'],'source_video_id':entry['source_video_id'],'budget':b,'pool':pool,
                    'schedules':len(values),'expected_CE':float(values.mean()),'expected_correctness':float(result['correct_'+pool].mean()),
                    'min_CE':float(values.min()),'q05_CE':float(np.quantile(values,.05)),'median_CE':float(np.median(values)),
                    'q95_CE':float(np.quantile(values,.95)),'max_CE':float(values.max()),'definition':'distribution of individual schedule outcomes;not ensemble'})
        z=u.logits(x,head).mean(0);prediction=int(z.argmax())
        rows.append({'cohort':entry['recognition_role'],'source_video_id':entry['source_video_id'],'video_id':entry['video_id'],
            'label':entry['label'],'budget':16,'policy':'dense','pool':'C=U','privileged':False,'correctness_is_expectation':False,
            'schedule_index':0,'refresh_positions':json_string(list(range(16))),'weights':json_string([1/16]*16),
            'logits':json_string(z.tolist()),'undefined_output_reason':None,'CE':float(u.loss(z[None],y)[0]),'correct':float(prediction==y),
            'prediction_index':prediction,'feature_error':float(np.diag(u.distances(x)).mean()),
            'feature_error_definition':'last-cache reconstruction;dense numerical roundoff','effective_weight_count':16.,
            'simulated_refreshes':16,'reconstruction_reference_timeline_length':16,'classification_pool_items':16,'final_classification_outputs':1})
        np.savez_compressed(folder/(entry['video_id']+'.npz'),**stored)
        selected.append(info)
        if count%20==0:print(f'Exact schedule distributions saved:{count}/140 sources',flush=True)
    raw=pd.DataFrame(rows);u.csv(output/'per_video_metrics.csv',raw)
    u.save(output/'selected_schedules.json',{'videos':selected,'scope':'privileged oracle schedules;periodic baseline;both pools held fixed per selected set'})
    u.csv(output/'schedule_overlap.csv',pd.DataFrame(overlaps));u.csv(output/'random_exact_distributions.csv',pd.DataFrame(random))
    pairs=u.per_source_contrasts(raw);u.csv(output/'paired_comparisons.csv',pairs)
    aggregate,intervals,matrices=u.summarize(raw,pairs,p['classes'],p['bootstrap']['seed'],p['bootstrap']['resamples'])
    u.save(output/'aggregate_metrics.json',aggregate);u.save(output/'bootstrap_intervals.json',intervals)
    np.savez_compressed(output/'bootstrap_indices.npz',**matrices)


def main(args):
    args.input=(ROOT/args.input).resolve();output=(ROOT/args.output).resolve()
    if args.audit_only or args.finalize_reviewed or (output/'status.json').exists() and u.read(output/'status.json')['status']=='complete':
        p=u.read(output/'protocol.json');unchanged(p)
        assert p['bootstrap']['seed']==args.seed and p['bootstrap']['resamples']==args.bootstrap and p['budgets']==args.budgets,'Changed scientific settings require a fresh output directory'
        assert p['input']==rel(args.input),'Changed frozen input requires a fresh output directory'
        from verify_refresh_pooling_control import audit
        result=audit(output,args.input,p)
        if args.finalize_reviewed and u.read(output/'status.json')['status']!='complete':
            assert u.read(output/'status.json')['status']=='awaiting_figure_review'
            figures=sorted(f for f in (output/'figures').iterdir() if f.suffix in ['.png','.pdf'])
            assert len(figures)==8
            u.save(output/'figure_review.json',{'reviewed_at_utc':now(),'review':'Rendered four PNG figures inspected for labels,axes,cohort separation and clipping;PDF counterparts share the same figure canvas and are structurally checked',
                'figure_sha256':{f.relative_to(output).as_posix():u.sha(f) for f in figures}})
            verification=u.read(output/'verification.json')
            verification['final_saved_artifact_audit']=result
            verification['figure_inspection_record']='figure_review.json'
            u.save(output/'verification.json',verification)
            u.save(output/'status.json',{'status':'complete','completed_at_utc':now(),'blocker':None})
            ledger(output)
        print(json_string(result));return
    manifest,entries,head,provenance=preflight(args.input)
    if args.preflight_only:
        print(json_string({'passed':True,'sources':140,'head_sha256':provenance['head_sha256'],'relevant_reviewed_revision_differences':provenance['relevant_changes_since_reviewed_revision'],
            'training_arrays_opened':0,'protected_arrays_or_outcomes_opened':0,'estimated_schedule_outcomes':140*sum(comb(15,b-1) for b in args.budgets)}));return
    assert not output.exists() or not any(output.iterdir()),'Existing incomplete results preserved;choose a fresh versioned directory'
    output.mkdir(parents=True)
    p=protocol(args,manifest,provenance);u.save(output/'protocol.json',p);u.save(output/'input_provenance.json',provenance)
    u.save(output/'status.json',{'status':'running','started_at_utc':now()})
    started=time.perf_counter()
    try:
        print('Frozen reused-data analysis protocol saved;starting exact CPU enumeration.',flush=True)
        compute(output,entries,head,p)
        from verify_refresh_pooling_control import audit
        checks=audit(output,args.input,p,verify_ledger=False)
        tests=subprocess.run([sys.executable,'-m','unittest','test_refresh_pooling_control','-v'],cwd=ROOT,capture_output=True,text=True)
        assert tests.returncode==0,tests.stdout+tests.stderr
        from refresh_pooling_report import report
        report(output,p)
        release=subprocess.run([sys.executable,'validate_release.py'],cwd=ROOT,capture_output=True,text=True)
        assert release.returncode==0,release.stdout+release.stderr
        assert not any(n in sys.modules for n in ['torch','torchvision','av','sklearn'])
        unchanged(p)
        checks['focused_tests']={'returncode':tests.returncode,'output':tests.stdout+tests.stderr}
        checks['release_validator']={'returncode':release.returncode,'output':release.stdout+release.stderr}
        checks['forbidden_runtime_modules_not_imported']=True
        checks['actual_new_encoder_decoder_or_fitting_calls']=0
        u.save(output/'verification.json',checks)
        u.save(output/'environment.json',{'recorded_at_utc':now(),'os':platform.platform(),'python':platform.python_version(),
            'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            'threads':{n:os.environ[n] for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
            'device':'CPU;no torch/av/sklearn import or GPU initialization','new_dependencies_or_upgrades':[]})
        u.save(output/'runtime.json',{'offline_analysis_elapsed_seconds':time.perf_counter()-started,
            'boundary':'CPU enumeration,saving,verification,tests,figures,release validation;excludes preflight and manual visual review;not inference latency',
            'encoder_calls':0,'decoder_calls':0,'fit_calls':0,'video_or_weight_downloads':0})
        u.save(output/'status.json',{'status':'awaiting_figure_review','computed_at_utc':now(),'blocker':None})
        ledger(output)
        print('Scientific computation/verification passed;inspect figures then use --finalize-reviewed. Output:'+rel(output),flush=True)
    except Exception as exc:
        u.save(output/'status.json',{'status':'incomplete','failed_at_utc':now(),'blocker':f'{type(exc).__name__}:{exc}'})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/recognition_baseline_v2'))
    parser.add_argument('--output',type=Path,default=Path('results/refresh_pooling_control_v1'))
    parser.add_argument('--budgets',type=int,nargs='+',default=[2,4,8])
    parser.add_argument('--bootstrap',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20261023)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--finalize-reviewed',action='store_true',help='Only after inspecting all rendered figures;records the visual-review attestation')
    args=parser.parse_args()
    if args.budgets!=[2,4,8] or args.bootstrap<1:parser.error('Frozen study requires --budgets 2 4 8 and positive bootstrap count')
    main(args)
