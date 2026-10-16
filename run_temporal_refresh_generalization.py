"""Resumable temporal-backbone study with an enforced evaluation seal."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

import argparse
import importlib.metadata
import platform
import subprocess
import sys
import time

from temporal_refresh_common import *

SOURCE_FILES=['run_temporal_refresh_generalization.py','temporal_refresh_common.py','temporal_refresh_data.py',
    'temporal_refresh_encoders.py','temporal_refresh_quality.py','temporal_refresh_timing.py','temporal_refresh_report.py',
    'test_temporal_refresh_generalization.py','setup_temporal_refresh.py','refresh_pooling_utils.py',
    'task_aware_oracle_utils.py','research_utils.py','benchmark.py']


def initialize(output,args):
    if (output/'run_configuration.json').exists():
        config=read(output/'run_configuration.json')
        assert config['seed']==args.seed and config['budgets']==args.budgets and config['bootstrap']==args.bootstrap
        assert config['max_download_bytes']==int(args.max_download_gib*1024**3)
        return config
    output.mkdir(parents=True,exist_ok=True)
    config={'experiment':'temporal_refresh_generalization_v1','created_at_utc':now(),'starting_commit':commit(),
        'design_commit':'ed59684e77bc481d627eb6942a52a4bfeae9dbc3',
        'design_revision_difference':subprocess.check_output(['git','diff','--stat','ed59684e77bc481d627eb6942a52a4bfeae9dbc3','HEAD'],cwd=ROOT,text=True),
        'seed':args.seed,'budgets':args.budgets,'bootstrap':args.bootstrap,'max_download_bytes':int(args.max_download_gib*1024**3),
        'initial_tracked_files':subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines(),
        'initial_worktree_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)}
    save(output/'run_configuration.json',config)
    (output/'protocol.md').write_text(f'''# Frozen analysis protocol: temporal_refresh_generalization_v1

Written {now()}, before evaluation inference. This protocol and subsequent hash seal
apply to new-source evaluation, conditional on reused development choices; not a
universal pretraining-independence guarantee. Actual starting commit: {config['starting_commit']}.
Design revision: {config['design_commit']}; tracked difference recorded in run_configuration.json.

## Question and primary endpoint
X3D-XS, budget4, cache-duration pooling: d_i = CE(S_F) - CE(periodic).
The primary endpoint is equal-source mean d over exactly100 new official K400
validation parents,5 per each existing20 classes. Positive means reconstruction-
optimal selection harms classification; negative means helps. Above-zero95%CI
supports mismatch here; below-zero supports feature selection and contradicts a
broad mismatch claim; crossing zero is inconclusive, never equivalence.

## Roles, eligibility and amendment
Reuse exactly400 existing training and100 existing development parents from the
recognition baseline manifest. Re-extract every representation on a new snippet
grid. This resource-conscious amendment replaces the earlier600-entirely-new-
source proposal. Development previously selected recognition regularization and
is reused deliberately. Do not load old exploratory or protected confirmation
features/outcomes. Regenerate historical source/hash exclusions from membership
documents, including protected/reserved cohorts; exclude official K400 training
annotation parent IDs from new evaluation. Original encoded bytes preserved.
Deterministic cached-first archive SHA order and per-archive candidate SHA order,
seed20261024. Technical failures only: H264,5..15sec, valid increasing PTS,
16 distinct complete snippets, labels/access. No recognition-driven replacement.
Freeze100 membership before inference. No extending N after results. Parent IDs
exclude adjacent windows; unknown reuploads/near-duplicates remain unresolved.

## Inputs and encoders
T=16 desired uniformly spaced endpoints from earliest complete snippet through
last decoded PTS. For endpoint u request [u-47/30,u-32/30,u-16/30,u]. Resolve
greatest actual decoded PTS<=target+1e-9sec. Require four distinct frames and
strictly increasing endpoints; no nearest rounding, repeats or padding.
A: original ImageNet RN18 endpoint512. B: separate RN18 forwards for the same
four RGB frames then mean512. C: pinned X3D-XS revision{REVISION}, official
X3D_XS.pyth, shortside/crop182, /255, mean.45/std.225, factorycrop160 unchanged.
Use pre-projection/pre-classification-activation features, mean residual axes.
Verify native400 path, strict checkpoint load and affine mean-logit equivalence.
Batch1,float32 encoders,eval/inference_mode, deterministic settings; no state
between snippets/videos. B-v-C matches observed images, not architecture,
resolution,pretraining or dimensionality; descriptive, not causal temporal effect.

## Recognition heads
One affine20-class head per representation, trained on400 dense feature means.
Training-only population standardization; std<=1e-12 replaced by1. Reuse float64
CPU multinomial meanCE + alpha/2||W||², unpenalized bias; deterministic LBFGS
max_iter400,max_eval800,grad1e-9/change1e-12,strong_wolfe,history50. Grid
[.001,.01,.1,1]. Select lowest DEVELOPMENT DENSE CE, highest accuracy, grid order.
This differs from historical accuracy-first selection. No sparse/evaluation
model selection. Freeze heads/scalers and code before evaluation inference.
Record solver convergence and dense train/development sanity, no favorable
accuracy-based source or study filtering. Estimate N100 precision using reused
development within-class variance; state approximations, no power guarantee.

## Exact schedules and pooling
Require0 and exactlyB selected candidates; B4primary,B2/8secondary; lex enumeration
15/455/6435. Periodic range(0,16,16//B). Dense16 separate.
C weights each selected feature by subsequent candidate reuse count/16.
U weights selected features equally1/B. Pool features/logits, not probabilities.
Cosine error averages raw-feature distance1-clamp(cos,-1,1) on16 last-cache
outputs. Historical convention REJECTS norms<=1e-12 rather than imputing distance.
SF minimizes this error without labels (still privileged full dense references).
SC/SU minimize true-label CE under their own pools; both privileged hindsight.
Lex first within1e-12 of true numerical minimum; retain minimum and achieved
objective/tie counts. ExactRandom averages individual schedule CE/correctness,
not ensemble logits. Cross-evaluate every chosen schedule under both pools.
Record LC(SC),LU(SC),LC(SU),LU(SU), W_C=LU(SC)-LC(SC), W_U=LU(SU)-LC(SU),
H_C=LC(P)-LC(SC),H_U=LU(P)-LU(SU), schedule agreement/Jaccard. Headroom is
nonnegative by construction, not evidence that a practical policy recovers it.
CE optimization does not guarantee top1 improvement.

## Statistics and secondary analyses
10000 class-stratified whole-source paired resamples,PCG64 seed20261024,
5/class in evaluation and reused development. Share saved indices across all
representations/budgets/pools/comparisons within cohort; separate cohort files.
Linear2.5/97.5 percentiles. Report raw source deltas, mean CE,top1/macro accuracy,
per-class descriptions, oracle headroom, selection/pooling cross-evaluation,
other representations and B2/8 as secondary marginal/unadjusted analyses.
No unstable per-video oracle-gap ratios. Intervals condition on heads/classes,
exclude training,selection,dataset/repeated-research uncertainty. No pooling
of development and fresh evaluation. No new endpoint after exposure.

## Timing and resources
LocalCUDA only;20GiB hard cap all new HTTP body bytes incl deps/weights/retries;
verified cache reuse free. Separate add-on environment preserves historicalenv.
Training-only smoke estimates runtime,disk,memory before bulk collection.
After quality seal, time20 evaluation sources1/class by frozen stablehash,
six actual systems A/B/C dense16/periodicB4. Two warmups,five measurements,
seeded interleaving,CPUthreads1,batch1,CUDA sync only at boundaries. Fresh
BytesIO,full sequential decode each trial. Encoded-memory→decode/preprocessing/
transfer/encoder/pooling/head/completed prediction. Disk load/setup/downloads
excluded and recorded. B makes4 independent frame forwards/refreshed snippet,
no cross-snippet feature caching. Equal B is not equal compute. Record counts,
per-video medians/min/max,offline clip throughput and scoped GPU/CPU memory.
Never place free-oracle quality at measured deployable latency.

## Verification, outputs, limits
Timestamp floor/prefix and pixel isolation; distinct frames; independent scalar
oracle/pooling fixtures; checkpoint/native/affine checks; source/hash isolation;
independent selected-schedule cache/unique replays and saved-output/bootstrap
reproduction; historical release validator once; immutable old scientificfiles.
Save cohort/exclusion/eligibility/acquisition records, identities, environment,
head grids,seal,features outsideGit with hashes, per-video quality/cross-pooling,
bootstrap arrays,raw timing,verification,figures,findings and reproductioncommands.
Offline candidates use whole clips; no streaming claim. Cross-backbone changes
can involve accuracy/calibration/pretraining/aggregation. Does not establish old
predictors fail on X3D. Newly exposed evaluation cannot later confirm a controller
designed after reading it. No architecture search/new refresh predictor, model
redesign,recognition accuracy guarantee,adaptive speedup or publication promise.
''',encoding='utf-8')
    return config


def environment(output):
    import torch,av
    packages={}
    for name in ['torch','torchvision','av','numpy','pandas','matplotlib','Pillow','fvcore','iopath','einops','scipy']:
        packages[name]=importlib.metadata.version(name)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; collection/timing blocked, never substitute CPU')
    test=(torch.arange(4,device='cuda')**2).cpu().tolist()
    if not (output/'hardware_details.json').exists():
        details=subprocess.check_output(['powershell','-NoProfile','-Command',
            '$cpu=Get-CimInstance Win32_Processor; $ram=Get-CimInstance Win32_ComputerSystem; '+
            '@{CPU=$cpu.Name;PhysicalCores=$cpu.NumberOfCores;LogicalCores=$cpu.NumberOfLogicalProcessors;RAMBytes=$ram.TotalPhysicalMemory}|ConvertTo-Json -Compress'],text=True)
        save(output/'hardware_details.json',json.loads(details))
    value={'recorded_at_utc':now(),'OS':platform.platform(),'CPU':platform.processor(),'python':platform.python_version(),
        'packages':packages,'torch_version_cuda':torch.version.cuda,'cuda_available':True,'gpu':torch.cuda.get_device_name(0),
        'GPU_operation':test,'FFmpeg':{k:list(v) for k,v in av.library_versions.items()},
        'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        'note':'NVIDIA driver version and torch.version.cuda are distinct; isolated add-on dependencies, historical CUDA stack unchanged'}
    if (output/'environment.json').exists():
        original=read(output/'environment.json')
        for key in ['packages','torch_version_cuda','gpu','python','nvidia_smi']:
            assert original[key]==value[key],f'Resume environment changed: {key}'
        return original
    save(output/'environment.json',value); return value


def seal(output,heads,eval_entries):
    path=output/'evaluation_seal.json'
    identities={relative(output/n):sha(output/n) for n in ['protocol.md','data_plan.json','source_exclusions.json','evaluation_manifest.json',
        'train_development_manifest.json','encoder_provenance.json','head_selection.json','development_precision_estimate.json',
        'environment.json','hardware_details.json','dependency_setup.json','train_development_collection.json','real_input_isolation.json','training_timing_smoke.json',
        'development_quality.csv','development_schedule_artifacts.json']}
    identities.update({relative(ROOT/name):sha(ROOT/name) for name in SOURCE_FILES})
    identities.update({v['path']:v['sha256'] for v in heads.values()})
    if path.exists():
        identities[relative(output/'input_provenance.json')]=sha(output/'input_provenance.json')
        assert read(path)['sha256']==identities,'Frozen protocol/code/head/input identity changed: do not resume exposed evaluation'
        return
    assert not (output/'evaluation_collection.json').exists(),'Cannot retroactively seal after evaluation inference'
    plan=read(output/'data_plan.json')
    save(output/'input_provenance.json',{'recorded_at_utc':now(),'starting_commit':read(output/'run_configuration.json')['starting_commit'],
        'historical_source_roles_manifest':{'path':'results/recognition_baseline_v2/dataset_manifest.json','sha256':plan['source_manifest_sha256']},
        'source_exclusions_sha256':sha(output/'source_exclusions.json'),'annotation_sha256':plan['annotation_sha256'],
        'official_train_annotations_sha256':sha(ASSETS/'k400_train.csv'),
        'official_train_annotations_url':'https://s3.amazonaws.com/kinetics/400/annotations/train.csv',
        'official_val_annotations_url':'https://s3.amazonaws.com/kinetics/400/annotations/val.csv',
        'archive_list_url':'https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt',
        'encoder_provenance_sha256':sha(output/'encoder_provenance.json'),
        'training_development_reference_ledger_sha256':sha(output/'train_development_collection.json'),
        'selected_heads':heads,'evaluation_membership_sha256':sha(output/'evaluation_manifest.json'),
        'immutable_input_and_code_hashes':identities,'protected_access':'membership identities only; no protected predictions,feature arrays or motion tables'})
    identities[relative(output/'input_provenance.json')]=sha(output/'input_provenance.json')
    save(path,{'sealed_at_utc':now(),'sha256':identities,'evaluation_videos':100,
        'membership_sha256':sha(output/'evaluation_manifest.json'),'no_evaluation_predictions_inspected':True,
        'preprocessing_model_and_code_frozen':True})


def run(args):
    output=(ROOT/args.output).resolve()
    if (output/'evaluation_seal.json').exists():
        for name,digest in read(output/'evaluation_seal.json')['sha256'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError(f'Frozen resume identity changed: {name}')
    if args.audit_only or (output/'status.json').exists() and read(output/'status.json').get('status')=='complete':
        from temporal_refresh_report import audit
        print(json.dumps(audit(output),indent=2)); return
    initialize(output,args)
    from temporal_refresh_data import freeze_data_plan,prepare_existing,acquire_evaluation
    from temporal_refresh_encoders import Encoders,load_rgb_frames,collect,verify_real_isolation
    from temporal_refresh_quality import head_fitting,evaluate_cohort,precision,summarize
    from temporal_refresh_timing import run_timing,training_timing_smoke
    from temporal_refresh_report import report,audit
    budget=Budget(output/'download_accounting.json',int(args.max_download_gib*1024**3))
    environment(output); plan=freeze_data_plan(output)
    save(output/'status.json',{'status':'in_progress','stage':'technical_preparation','timestamp':now()})
    existing=prepare_existing(output,plan)
    if args.preflight_only:
        print('Preflight complete:500 unchanged sources; no evaluation inference'); return
    setup_start=time.perf_counter();encoders=Encoders(output,args.device)
    import torch
    torch.cuda.synchronize()
    setup_sessions=read(output/'setup_sessions.json') if (output/'setup_sessions.json').exists() else []
    setup_sessions.append({'at_utc':now(),'model_construction_and_device_transfer_seconds':time.perf_counter()-setup_start,
        'scope':'pinned model imports/factories,checkpoint reads,strict loading and completed CUDA transfers; global CLI imports/preflight excluded',
        'resident_parameter_buffer_bytes':{r:sum(x.numel()*x.element_size() for x in (list(m.parameters())+list(m.buffers())))
            for r,m in [('resnet18',encoders.rn),('x3d_xs_native400',encoders.x3d)]}})
    save(output/'setup_sessions.json',setup_sessions)
    encoders.verify_preprocessing()
    smoke_path=output/'smoke_test.json'
    if not smoke_path.exists():
        smoke=sorted([v for v in existing if v['recognition_role']=='train'],key=lambda v:rank('smoke',v['source_video_id']))[:2]
        start=time.perf_counter(); smoke_ledger=collect(output,smoke,encoders,'training_smoke')
        frames=load_rgb_frames((ROOT/smoke[0]['relative_path']).read_bytes(),smoke[0]); indices=smoke[0]['snippet_grid']['frame_indices'][0]
        native=encoders.native_check([frames[i] for i in indices]); save(output/'native_head_verification.json',native)
        seconds=time.perf_counter()-start
        import torch
        save(smoke_path,{'sources':[v['source_video_id'] for v in smoke],'videos':2,'seconds':seconds,
            'estimated600_video_seconds_linear':seconds/2*600,'estimated_feature_bytes600':sum((ROOT/x['path']).stat().st_size for x in smoke_ledger['videos'].values())/2*600,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'disk_free_bytes':__import__('shutil').disk_usage(ROOT).free,
            'native_head':native,'feasible':True,'limits':'two training sources; estimate excludes archive traversal,heads,exact diagnostics and timing'} )
    save(output/'status.json',{'status':'in_progress','stage':'train_development_collection','timestamp':now()})
    collect(output,existing,encoders,'train_development')
    verify_real_isolation(output,existing,encoders)
    if args.stop_after=='collection': return
    save(output/'status.json',{'status':'in_progress','stage':'head_fitting_and_development','timestamp':now()})
    heads=head_fitting(output,existing,plan['classes'])
    dev=evaluate_cohort(output,[v for v in existing if v['recognition_role']=='development'],'development',plan['classes'],args.budgets)
    precision(output,dev)
    if args.stop_after=='heads': return
    training_timing_smoke(output,existing,encoders)
    evaluation=acquire_evaluation(output,plan,budget)
    seal(output,heads,evaluation)
    save(output/'evaluation_exposure.json',{'started_at_utc':now(),'seal_sha256':sha(output/'evaluation_seal.json'),
        'evaluation_predictions_allowed_only_after_seal':True}) if not (output/'evaluation_exposure.json').exists() else None
    save(output/'status.json',{'status':'in_progress','stage':'sealed_evaluation_collection','timestamp':now()})
    collect(output,evaluation,encoders,'evaluation')
    fresh=evaluate_cohort(output,evaluation,'evaluation',plan['classes'],args.budgets)
    import pandas as pd
    raw=pd.concat([dev,fresh],ignore_index=True); raw.to_csv(output/'per_video_quality.csv',index=False,float_format='%.17g')
    summarize(output,raw,plan['classes'],args.bootstrap,args.seed)
    save(output/'status.json',{'status':'in_progress','stage':'actual_dense_periodic_timing','timestamp':now()})
    run_timing(output,evaluation,encoders)
    report(output)
    verified=audit(output); save(output/'verification.json',verified)
    save(output/'status.json',{'status':'awaiting_figure_review','timestamp':now(),'all_computational_stages':True,
        'next':'inspect all rendered figures, then --finalize-reviewed'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='results/recognition_baseline_v2')
    parser.add_argument('--output',default='results/temporal_refresh_generalization_v1')
    parser.add_argument('--device',default='cuda');parser.add_argument('--max-download-gib',type=float,default=20)
    parser.add_argument('--seed',type=int,default=SEED);parser.add_argument('--bootstrap',type=int,default=10000)
    parser.add_argument('--budgets',type=int,nargs='+',default=[2,4,8]);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--finalize-reviewed',action='store_true')
    parser.add_argument('--stop-after',choices=['collection','heads'],default=None,help='Resume boundary for development-only debugging')
    args=parser.parse_args()
    assert args.seed==SEED and args.bootstrap==10000 and args.budgets==[2,4,8]
    assert 0<args.max_download_gib<=20,'Hard download ceiling is20GiB'
    assert args.input=='results/recognition_baseline_v2','Frozen source roles required'
    output=(ROOT/args.output).resolve()
    if not args.audit_only and Path(sys.prefix).name!='.venv_temporal':
        if not (output/'dependency_setup.json').exists() or not (ROOT/'.venv_temporal/Scripts/python.exe').exists():
            subprocess.run([sys.executable,str(ROOT/'setup_temporal_refresh.py'),'--output',args.output,'--max-download-gib',str(args.max_download_gib)],check=True)
        raise SystemExit(subprocess.call([str(ROOT/'.venv_temporal/Scripts/python.exe'),__file__,*sys.argv[1:]]))
    if args.finalize_reviewed:
        from temporal_refresh_report import audit
        if (output/'status.json').exists() and read(output/'status.json').get('status')=='complete':
            print(json.dumps(audit(output),indent=2));return
        save(output/'verification.json',audit(output))
        save(output/'figure_review.json',{'reviewed_at_utc':now(),'method':'human/assistant inspection of rendered PNGs for signs,units,cohorts,labels and clipping'})
        save(output/'status.json',{'status':'complete','completed_at_utc':now(),'all_stages_and_figure_review':True})
        lines=[f'{sha(p)}  {p.relative_to(output).as_posix()}' for p in sorted(output.rglob('*'))
            if p.is_file() and p.name!='SHA256SUMS.txt']
        (output/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return
    try: run(args)
    except Exception as exc:
        save(output/'status.json',{'status':'incomplete','timestamp':now(),'error_type':type(exc).__name__,'blocker':str(exc),
            'resume':'& .\\.venv_temporal\\Scripts\\python.exe .\\run_temporal_refresh_generalization.py --resume'})
        raise


if __name__=='__main__': main()
