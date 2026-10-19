"""Experiment19 checkpoint/resume runner; population gates precede empirical fitting."""
from __future__ import annotations
import os
for thread_name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[thread_name]='1'
import argparse
import subprocess
import sys
import time
import numpy as np
from causal_observability_common import *


def tests():
    if (OUT/'evaluation_seal.json').exists():return
    r=subprocess.run([sys.executable,'-B','-m','unittest','test_causal_observability','-v'],capture_output=True,text=True,cwd=ROOT)
    save(OUT/'unit_tests.json',dict(at_utc=now(),returncode=r.returncode,output=r.stdout+r.stderr,fixtures='Explicitly SYNTHETIC unit tests, not empirical findings',source_sha256=sha(ROOT/'test_causal_observability.py')))
    if r.returncode:raise Blocker('Unit tests failed; evidence retained in unit_tests.json')
    state('tests','verified')


def native_smoke(encoder=None):
    from causal_observability_runtime import Encoder,live
    from causal_observability_policy import Policy
    path=OUT/'GPU_smoke.json'
    if path.exists():
        from causal_observability_verify import smoke_source_check
        r=read(path);smoke_source_check(r);verify({v['path']:v['sha256'] for v in r['results']});return encoder
    rows=read(DATA/'eligible_input_population.json')['videos']
    candidates=sorted([v for v in rows if v['official_flag']==1],key=lambda v:rank('GPU_smoke',v['label'],v['clip']))
    chosen=[]
    for v in candidates:
        if v['source_family'] not in {s['source_family'] for s in chosen} and v['label'] not in {s['label'] for s in chosen}:chosen.append(v)
        if len(chosen)==2:break
    if len(chosen)!=2:raise Blocker('Insufficient isolated real training videos for GPU smoke')
    member=OUT/'GPU_smoke_membership.json'
    identities=[dict(label=v['label'],clip=v['clip'],source_family=v['source_family'],sha256=v['sha256'],official_flag=v['official_flag']) for v in chosen]
    if member.exists():
        if read(member)['clips']!=identities:raise Blocker('Preselected smoke membership changed')
    else:save(member,dict(at_utc=now(),clips=identities,selection='Input-only hash of seed20261019/class/clip after source-family quarantine'))
    state('GPU_smoke','running',new51_head='pending population design/training')
    setup=time.perf_counter();encoder=encoder or Encoder();setup_seconds=time.perf_counter()-setup
    layer=encoder.model.blocks[-1].proj
    head=dict(means=np.zeros(2048).tolist(),scales=np.ones(2048).tolist(),weights=layer.weight.detach().cpu().numpy().tolist(),bias=layer.bias.detach().cpu().numpy().tolist())
    save(DATA/'native400_smoke_head.json',head);records=[]
    for v in chosen:
        raw=(ROOT/v['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=v['sha256']:raise Blocker('Smoke input bytes changed')
        for method in ['periodic','motion']:
            encoder.phase='pretest_native400_smoke';encoder.clip=v['label']+'/'+v['clip']
            r=live(raw,v,encoder,Policy(method,head,threshold=.05,clip=v['clip']),head)
            r.update(native_projection_check=encoder.native_check(),label=v['label'],clip=v['clip'],method=method,
                scope='Real native400-head integration, not a fitted HMDB51 head or learned-policy quality result')
            p=DATA/'smoke'/f'{rank(v["label"],v["clip"],method)}.json';save(p,r)
            records.append(dict(path=rel(p),sha256=sha(p),clip=v['clip'],method=method,positions=r['positions'],
                encoder_starts=r['encoder_starts'],encoder_completions=r['encoder_completions'],grayscale_conversions=r['grayscale_conversions'],RGB_conversions=r['RGB_conversions'],seconds=r['seconds']))
            print('Real two-video integration:',v['clip'],method,r['positions'],'actual forwards',r['encoder_completions'],flush=True)
    save(path,dict(at_utc=now(),status='native400_integration_passed_new51_head_integration_pending',results=records,
        native_head_path=rel(DATA/'native400_smoke_head.json'),native_head_sha256=sha(DATA/'native400_smoke_head.json'),
        official_test_calls=0,completed_smoke_calls=sum(r['encoder_completions'] for r in records),setup_seconds_excluded=setup_seconds,
        new51_head_integration='NOT RUN until feasible population and fitting',
        code_sha256={rel(ROOT/n):sha(ROOT/n) for n in ['causal_observability_runtime.py','causal_observability_policy.py']},
        projected_B4_seconds_per_video=float(np.mean([r['seconds'] for r in records])),projection_limit='Two real sources/native400 head; not final latency benchmark or simulated measurements'))
    state('GPU_smoke','partial',native400='verified',new51_head='pending population design',actual_completed_calls=16)
    return encoder


def policy(method,head,selection,clip):
    from causal_observability_policy import Policy
    if method in ['history','history_motion']:
        r=selection[method];verify({r['path']:r['sha256']});return Policy(method,head,read(ROOT/r['path']),r['threshold'],clip)
    return Policy(method,head,threshold=selection.get('motion',{}).get('threshold',.5),clip=clip)


def timing_membership(manifest):
    target=OUT/'timing_membership.json'
    if target.exists():return read(target)
    selected=[];used=set();missing=[]
    for c in sorted(manifest['classes'],key=lambda c:rank('timing_class',c)):
        candidates=sorted([v for v in manifest['videos'] if v['role']=='test' and v['label']==c and v['source_family'] not in used],key=lambda v:rank('timing',v['label'],v['clip']))
        if not candidates:missing.append(c);continue
        v=candidates[0];selected.append(v['label']+'/'+v['clip']);used.add(v['source_family'])
    result=dict(at_utc=now(),keys=selected,source_families=len(used),missing_classes=missing,selection='Input-only seed hash; one distinct source family/class when available')
    save(target,result);return result


def new_head_integration(manifest,encoder,head,selection):
    if (OUT/'new51_integration.json').exists():
        r=read(OUT/'new51_integration.json')
        if r['head_sha256']!=sha(OUT/'head/parameters.json') or r['selection_sha256']!=sha(OUT/'controller_selection.json'):
            raise Blocker('New51 integration model changed')
        verify({v['path']:v['sha256'] for v in r['results']});return
    from causal_observability_runtime import live
    candidates=sorted([v for v in manifest['videos'] if v['role']=='head_train'],key=lambda v:rank('new_head_smoke',v['clip']))[:2]
    records=[]
    for v in candidates:
        for method in ['periodic','history_motion']:
            encoder.phase='pretest_new51_integration';encoder.clip=v['label']+'/'+v['clip']
            r=live((ROOT/v['path']).read_bytes(),v,encoder,policy(method,head,selection,v['clip']),head)
            p=DATA/'new51_integration'/f'{rank(v["clip"],method)}.json';save(p,r);records.append(dict(path=rel(p),sha256=sha(p)))
    save(OUT/'new51_integration.json',dict(at_utc=now(),passed=True,head_sha256=sha(OUT/'head/parameters.json'),selection_sha256=sha(OUT/'controller_selection.json'),results=records,official_test_calls=0))
    state('GPU_smoke','verified',native400='verified',new51_head='verified')


def freeze(manifest):
    p=OUT/'evaluation_seal.json'
    if p.exists():seal_check((OUT/'evaluation_exposure.json').exists());return
    if (OUT/'test_rollouts.json').exists() or (OUT/'test_feature_collection.json').exists():raise Blocker('Test artifact exists before seal; preserve evidence')
    code=list(ROOT.glob('*.py'))
    artifacts=[OUT/n for n in ['protocol.md','eligible_manifest.json','official_membership.json','encoder_identity.json',
        'head/parameters.json','head_selection.json','head_grid.json','controller_selection.json','controller_training_identity.json',
        'controller_training_development.json','timing_membership.json','new51_integration.json','unit_tests.json']]
    artifacts+=list((OUT/'controllers').glob('*.json'))
    artifacts+=[DATA/'hmdb51_org.rar.identity.json',DATA/'test_train_splits.rar',DATA/'official_page.html',DATA/'distribution_metadata.json']
    ident=read(OUT/'encoder_identity.json');artifacts.append(ROOT/ident['checkpoint'])
    root=ROOT/'data/source/temporal_refresh_generalization_v1'
    artifacts+=list((root/'pytorchvideo-f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0/pytorchvideo').rglob('*.py'))
    save(p,dict(at_utc=now(),HEAD=git('rev-parse','HEAD'),sha256={rel(q):sha(q) for q in code+artifacts},
        primary='class-balanced CE(periodic)-CE(history_motion); positive better',seed=SEED,bootstrap_repeats=10000,
        bootstrap='paired whole source families, stratified by class-support signature; class-balanced clip ratios',
        no_prior_test_model_calls=True,protocol='Feature-causal batch inference; complete-clip timestamp planning'))
    state('freeze','frozen')


def rollouts(manifest,encoder,head,selection):
    from causal_observability_policy import METHODS
    from causal_observability_runtime import live
    seal_check()
    exposure=OUT/'evaluation_exposure.json'
    if not exposure.exists():save(exposure,dict(at_utc=now(),seal_sha256=sha(OUT/'evaluation_seal.json'),boundary='Immediately before any official test feature/prediction call'))
    seal_check(True);records={}
    entries=[v for v in manifest['videos'] if v['role']=='test']
    identity=dict(head=sha(OUT/'head/parameters.json'),controller=sha(OUT/'controller_selection.json'),seal=sha(OUT/'evaluation_seal.json'))
    for i,v in enumerate(entries):
        raw=(ROOT/v['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=v['sha256']:raise Blocker('Test encoded bytes changed')
        for method in METHODS:
            key=v['label']+'/'+v['clip']+'/'+method;p=DATA/'test_rollouts'/f'{rank(key)}.json'
            encoder.phase='test_'+method;encoder.clip=v['label']+'/'+v['clip']
            r=checkpoint(p,dict(**identity,input=v['sha256'],method=method),lambda:live(raw,v,encoder,policy(method,head,selection,v['clip']),head))
            if r['encoder_starts']!=4 or r['encoder_completions']!=4:raise Blocker('Recorded deployment violates exact budget')
            records[key]=dict(path=rel(p),sha256=sha(p))
        if (i+1)%25==0:
            save(OUT/'test_rollouts.json',dict(all_deployable_rollouts_complete=False,records=records));print(f'Actual B4 decisions {i+1}/{len(entries)} videos, all5 policies',flush=True)
    save(OUT/'test_rollouts.json',dict(all_deployable_rollouts_complete=True,records=records,clips=len(entries),methods=METHODS))
    state('test_rollouts','evaluated',clips=len(entries),actual_encodings_per_policy_video=4)


def diagnostics(manifest,head):
    from causal_observability_runtime import features
    from causal_observability_policy import affine
    from causal_observability_learning import teacher
    from causal_observability_analysis import score
    import refresh_pooling_utils as pool
    import pandas as pd
    target=OUT/'per_source_metrics.csv'
    if target.exists():return
    seal_check(True);ledger=read(OUT/'test_rollouts.json')
    if not ledger['all_deployable_rollouts_complete']:raise Blocker('Privileged diagnostics called before all deployed decisions')
    table=pool.schedules(16,4);rows=[];decisions=[]
    for v in [v for v in manifest['videos'] if v['role']=='test']:
        key=v['label']+'/'+v['clip'];y=manifest['classes'].index(v['label']);f,_=features(v,'test');z=affine(f,head);dense=z.mean(0)
        schedules=table['cache_weights'] @ z;loss=pool.loss(schedules,y);q=z-z.mean(1,keepdims=True)
        objective=np.square(table['cache_weights'] @ q-q.mean(0)).mean(1);j=pool.first_tie(objective)['index'];oracle=pool.first_tie(loss)['index']
        base=dict(key=key,clip=v['clip'],label=v['label'],source_family=v['source_family'])
        rows.append(dict(**base,method='dense16',**score(dense,y,dense)))
        for name,index in [('Z_pool_offline',j),('task_C_hindsight',oracle)]:rows.append(dict(**base,method=name,**score(schedules[index],y,dense)))
        average=[score(a,y,dense) for a in schedules]
        rows.append(dict(**base,method='RandomExact_expectation_privileged',**{m:float(np.mean([a[m] for a in average])) for m in average[0]}))
        for method in ledger['methods']:
            record=ledger['records'][key+'/'+method];verify({record['path']:record['sha256']});r=read(ROOT/record['path'])['payload']
            replay=affine(np.asarray([f[t] for t in r['cache_indices']],dtype=float).mean(0),head)
            np.testing.assert_allclose(replay,r['logits'],atol=1e-6,rtol=1e-6)
            rows.append(dict(**base,method=method,**score(r['logits'],y,dense),encoder_starts=r['encoder_starts'],encoder_completions=r['encoder_completions']))
            last=0
            for t in range(1,16):
                tr=r['trace'][t]
                if not tr['forced'] and tr['remaining_before']>0:
                    agreement=int(tr['selected']==bool(teacher(f,head,r['trace'],t)))
                    gain=score(z[last],y)['CE']-score(z[t],y)['CE']
                    decisions.append(dict(**base,method=method,t=t,selected=tr['selected'],teacher_agreement=agreement,
                        immediate_refresh_gain=gain,helpful=bool(tr['selected'] and gain>1e-12),harmful=bool(tr['selected'] and gain< -1e-12),
                        scope='retrospective local true-label diagnostic; not cumulative causal effect'))
                if tr['selected']:last=t
    pd.DataFrame(rows).to_csv(target,index=False,float_format='%.17g')
    pd.DataFrame(decisions).to_csv(OUT/'per_decision_diagnostics.csv',index=False,float_format='%.17g')
    if decisions:
        pd.DataFrame(decisions).groupby(['key','label','source_family','method'])[['teacher_agreement','helpful','harmful']].mean().reset_index().to_csv(OUT/'per_source_decisions.csv',index=False,float_format='%.17g')
    state('diagnostics','evaluated')


def timing(manifest,encoder,head,selection):
    from causal_observability_runtime import live,decode,LiveProvider
    from causal_observability_policy import affine
    from temporal_refresh_timing import working_set
    import pandas as pd
    seal_check(True);entries={v['label']+'/'+v['clip']:v for v in manifest['videos']};cohort=read(OUT/'timing_membership.json')
    progress=OUT/'timing_progress.json';records=read(progress) if progress.exists() else []
    completed={(r['key'],r['method'],r['repetition']) for r in records};modes=['dense16','periodic','periodic_unused_gate','RandomB4_seeded','motion','history','history_motion']
    encoder.log=False
    try:
        for key in cohort['keys']:
            v=entries[key];tick=time.perf_counter();raw=(ROOT/v['path']).read_bytes();load_seconds=time.perf_counter()-tick
            rng=np.random.Generator(np.random.PCG64(int(rank('timing_order',key)[:16],16)))
            for repetition in range(-2,5):
                for method in rng.permutation(modes):
                    if (key,method,repetition) in completed:continue
                    encoder.phase='test_timing';encoder.clip=key
                    if method=='dense16':
                        torch=encoder.torch;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated();before=encoder.starts
                        tick=time.perf_counter();frames,grid=decode(raw,v);provider=LiveProvider(frames,grid,encoder);values=[]
                        for t in range(16):provider.advance(t);values.append(provider.refresh())
                        z=affine(np.asarray(values,dtype=float).mean(0),head);prediction=int(z.argmax());torch.cuda.synchronize();seconds=time.perf_counter()-tick
                        r=dict(logits=z.tolist(),prediction=prediction,seconds=seconds,encoder_starts=encoder.starts-before,encoder_completions=provider.forward_completions,
                            decoded_frames=len(frames),RGB_conversions=provider.RGB_conversions,grayscale_conversions=0,cpu_decision_wall_seconds=0,
                            resident_cuda_bytes=resident,peak_cuda_bytes=torch.cuda.max_memory_allocated())
                        from causal_observability_runtime import features
                        reference=affine(features(v,'test')[0].mean(0,dtype=float),head)
                    else:
                        r=live(raw,v,encoder,policy(method,head,selection,v['clip']),head);r.pop('refreshed_features',None)
                        original='periodic' if method=='periodic_unused_gate' else method
                        ref=read(OUT/'test_rollouts.json')['records'][key+'/'+original];reference=read(ROOT/ref['path'])['payload']['logits']
                    discrepancy=float(np.max(abs(np.asarray(r['logits'])-reference)));np.testing.assert_allclose(r['logits'],reference,atol=1e-6,rtol=1e-6)
                    records.append(dict(key=key,label=v['label'],source_family=v['source_family'],method=str(method),repetition=repetition,warmup=repetition<0,
                        disk_load_seconds_excluded=load_seconds,reference_logit_max_difference=discrepancy,**r));save(progress,records)
            print('Real timing source complete',key,flush=True)
    finally:encoder.log=True
    frame=pd.DataFrame(records);frame.to_csv(OUT/'timing_trials.csv',index=False,float_format='%.17g')
    measured=frame[~frame.warmup];measured.groupby(['key','label','source_family','method']).seconds.agg(['median','min','max','std']).rename(columns={'median':'median_seconds'}).reset_index().to_csv(OUT/'timing_per_video.csv',index=False,float_format='%.17g')
    save(OUT/'timing_scope.json',dict(boundary='encoded bytes in memory -> fresh full decode/plan -> encountered cheap observations/decisions -> selected RGB/transfer -> native X3D -> pooled head -> CUDA-completed prediction',
        modes=modes,cohort=cohort,warmups=2,measured_repetitions=5,CPU_threads=1,encoder='batch1 FP32; native400 projection executes',memory='allocated GPU resident/peak, process working set; lifetime CPU peak not an exact scoped peak',offline_quality='never assigned periodic latency'))
    state('timing','evaluated')


def run():
    from causal_observability_data import metadata,eligibility
    from causal_observability_runtime import Encoder,collect_dense
    from causal_observability_learning import recognition,controllers
    from causal_observability_analysis import summarize,figures
    from causal_observability_verify import audit,report
    for name,url in [('official_page.html','https://serre.lab.brown.edu/hmdb51.html'),('distribution_README.md',BASE+'README.md'),
            ('distribution_metadata.json','https://huggingface.co/api/datasets/Serrelab/hmdb51/revision/'+REVISION+'?blobs=true')]:
        acquire(url,DATA/name,2**20)
    for name,(url,digest,limit) in SOURCES.items():acquire(url,DATA/name,limit,digest)
    state('data','prepared');gate=None
    try:manifest=eligibility(metadata());state('eligibility','prepared')
    except Blocker as e:gate=str(e)
    tests();encoder=native_smoke()
    if gate:
        state('eligibility','blocked',reason=gate);report(gate);save(OUT/'verification.json',audit());return 2
    timing_membership(manifest);encoder=encoder or Encoder()
    for role in ['head_train','controller_train','development']:
        collect_dense([v for v in manifest['videos'] if v['role']==role],role,encoder)
    head=recognition(manifest);selection=controllers(manifest,head);new_head_integration(manifest,encoder,head,selection);state('training','prepared')
    freeze(manifest);rollouts(manifest,encoder,head,selection)
    collect_dense([v for v in manifest['videos'] if v['role']=='test'],'test',encoder)
    diagnostics(manifest,head);summarize(manifest);timing(manifest,encoder,head,selection);figures()
    state('figures','awaiting_visual_review');save(OUT/'verification.json',audit());report();return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true');p.add_argument('--audit-only',action='store_true');p.add_argument('--finalize-reviewed',action='store_true');args=p.parse_args()
    if args.audit_only:
        from causal_observability_verify import audit
        print(json.dumps(audit(),indent=2));return
    if (OUT/'status.json').exists() and read(OUT/'status.json').get('current_stage')=='final' and read(OUT/'status.json')['status']=='verified':
        from causal_observability_verify import audit
        print(json.dumps(audit(),indent=2));return
    if args.finalize_reviewed:
        if read(OUT/'status.json')['status']!='awaiting_visual_review':raise Blocker('Outcome figures/results incomplete; cannot finalize')
        save(OUT/'figure_review.json',dict(at_utc=now(),actually_inspected=True));state('final','verified')
        from causal_observability_verify import report
        report();save(OUT/'SHA256SUMS.json',{rel(q):sha(q) for q in OUT.rglob('*') if q.is_file() and q.name not in ['SHA256SUMS.json','run.lock']});return
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'run.lock'
    try:
        with lock.open('x') as f:f.write(str(os.getpid()))
    except FileExistsError:raise Blocker('Runner lock exists; confirm recorded process exit before removing stale lock')
    try:code=run()
    except Exception as exc:
        state('execution','blocked' if isinstance(exc,Blocker) else 'incomplete',reason=type(exc).__name__+': '+str(exc))
        from causal_observability_verify import report
        report(str(exc));raise
    finally:lock.unlink(missing_ok=True)
    raise SystemExit(code)


if __name__=='__main__':main()
