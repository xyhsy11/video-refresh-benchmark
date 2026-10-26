"""E27 fitting-only entrypoint; frozen scientific modules are never rewritten.

No acquisition, final fitting, sealing, evaluation, or final timing entrypoint is
reachable from this module. Unfinished atomic work blocks for explicit audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time
import uuid

import numpy as np
from threadpoolctl import threadpool_limits

from native_head_value_confirmation import (read, save, sha, rank, folds,
    precision_plan, validate_components, VIEWS, CHECKPOINT_SHA)
from run_native_head_value_confirmation import ROOT, OUT, CACHE, CAPS, CODE, event, now, run_lock, git, network_ledger
from native_head_confirmation_state import totals, metered, retained_bytes, FREE_RESERVE, state, advance, memory_snapshot
from native_head_confirmation_runtime import (NativeEncoder, collect, feature_path,
    identities, fit_models, source_values, atomic_npz)

TASK = OUT/'task1'
SESSION_LIMIT = 21600
FORBIDDEN = ('joint_source_certificate.json','evaluation_seal.json','evaluation_exposure.json',
             'test_manifest.json.gz','eval_source_summaries.json','summary.json','bootstrap_draws.npz')


def freeze_operational_controls():
    names=('fit_plan_native_head_confirmation.py','verify_native_head_task1.py','test_native_head_task1.py')
    current=dict(sha256={n:sha(ROOT/n) for n in names},frozen_config_sha256=sha(OUT/'frozen_config.json'),
                 stop_stage='OOF_COMPLETE',maximum_session_seconds=SESSION_LIMIT,caps=CAPS)
    p=TASK/'operational_controls.json'
    if p.exists():
        require(read(p)==current,'BLOCKED_IDENTITY: task1 operational code changed; preserve and independently reconcile before resume')
    else:
        require(not list((TASK/'operations').glob('*.json')),'BLOCKED_IDENTITY: task1 controls missing after work')
        save(p,current)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def events():
    raw=(OUT/'resource_exposure_ledger.jsonl').read_bytes()
    require(raw.endswith(b'\n'), 'BLOCKED_ACCOUNTING: partial ledger record')
    return [json.loads(line) for line in raw.splitlines()]


def boundary():
    require(state()['stage'] in ('FIT_CERTIFIED','OOF_COMPLETE'), 'BLOCKED_INPUT: fitting certification required; no automatic acquisition')
    require(not any((OUT/n).exists() for n in FORBIDDEN), 'BLOCKED_STAGE_BOUNDARY: evaluation preparation/exposure already present')
    require(not list((OUT/'probes').glob('final_*')), 'BLOCKED_STAGE_BOUNDARY: final probes already present')
    require(not any(e.get('phase')=='EVAL' for e in events()), 'BLOCKED_STAGE_BOUNDARY: evaluation event present')
    require(not totals()['uncertain_operations'] and not totals()['uncertain_encoder_calls'], 'BLOCKED_ACCOUNTING: unresolved operations; no automatic recomputation')
    for p in (TASK/'operations').glob('*.json'):
        operation=read(p)
        require(operation['status']=='complete', 'BLOCKED_ACCOUNTING: incomplete atomic operation '+p.name)
        for name,digest in operation.get('sha256',{}).items():
            require(sha(ROOT/name)==digest,'BLOCKED_IDENTITY: completed atomic output changed '+name)
    config=read(OUT/'frozen_config.json')
    require(config['implementation']=={n:sha(ROOT/n) for n in CODE}, 'BLOCKED_IDENTITY: frozen scientific implementation changed')
    require(config['caps']==CAPS, 'BLOCKED_IDENTITY: cumulative caps changed')
    require(config['checkpoint_sha256']==CHECKPOINT_SHA, 'BLOCKED_IDENTITY: checkpoint changed')
    for receipt in state()['receipts']:
        for name,digest in receipt['sha256'].items():
            require(sha(ROOT/name)==digest, 'BLOCKED_IDENTITY: completed stage artifact changed '+name)
    return config


def authenticate_reference(row):
    """Authenticate original bytes as well as frozen recipe and output bytes."""
    path=ROOT/row['path']
    require(path.is_file(), 'BLOCKED_INPUT: required original missing '+row['filename'])
    require(sha(path)==row['encoded_sha256'], 'BLOCKED_INPUT: original byte mismatch '+row['filename'])
    p=feature_path(row); receipt=p.with_suffix('.json')
    if not p.exists() and not receipt.exists():
        return False
    require(p.exists() and receipt.exists(), 'BLOCKED_ACCOUNTING: partial reference requires audit')
    r=read(receipt)
    expected=dict(encoded_sha256=row['encoded_sha256'],grid_sha256=rank('grid',json.dumps(row['grid'],sort_keys=True)),
                  checkpoint_sha256=CHECKPOINT_SHA,phase='PROBE_FIT',implementation=identities())
    require(r['input']==expected and sha(p)==r['array_sha256'], 'BLOCKED_IDENTITY: reference mismatch '+row['filename'])
    require(r['encoder_calls']==r['completed_calls']==16 and r.get('genuine_resource_elapsed_seconds',0)>0,
            'BLOCKED_ACCOUNTING: reference completion receipt absent')
    with np.load(p,allow_pickle=False) as a:
        for name,shape in [('features',(16,2048)),('logits',(16,400)),('thumbnails',(16,32,32))]:
            require(a[name].shape==shape and np.isfinite(a[name]).all(), 'BLOCKED_INPUT: invalid reference arrays')
    return True


def ordinary_profile():
    """Independently reconcile saved timing with fresh-forward ledger receipts."""
    report=read(OUT/'normal_reference_throughput_diagnosis.json')
    smoke=read(OUT/'smoke_manifest.json'); log=events()
    require(report['implementation_sha256']==sha(ROOT/'profile_native_head_reference.py'), 'BLOCKED_IDENTITY: profile implementation')
    require(len(smoke)==len(report['receipts'])==8 and report['genuine_native_forwards']==report['completed']==128,
            'BLOCKED_THROUGHPUT: eight completed genuine profile sources required')
    require(report['cached_vector_replay_used_for_throughput'] is False, 'BLOCKED_THROUGHPUT: cached throughput forbidden')
    times=[]; evidence={}
    for row,r in zip(smoke,report['receipts']):
        require(authenticate_reference(row), 'BLOCKED_INPUT: smoke reference missing')
        require(row['filename']==r['filename'], 'BLOCKED_IDENTITY: profile source order')
        p=ROOT/r['array_path']; receipt=read(p.with_suffix('.json'))
        require(sha(p)==r['array_sha256']==receipt['array_sha256'], 'BLOCKED_IDENTITY: profile arrays')
        require(receipt['encoded_sha256']==row['encoded_sha256'] and receipt['checkpoint_sha256']==CHECKPOINT_SHA,
                'BLOCKED_IDENTITY: profile input/model')
        require(receipt['profile_implementation_sha256']==report['implementation_sha256'] and
                receipt['native_validation_inside_measurement'] is False and receipt['cached_vector_replay'] is False,
                'BLOCKED_THROUGHPUT: profile timing scope')
        with np.load(p,allow_pickle=False) as a,np.load(feature_path(row),allow_pickle=False) as b:
            require(all(np.array_equal(a[k],b[k]) for k in ('features','logits','thumbnails')), 'BLOCKED_INPUT: ordinary/smoke parity failed')
        attempts=[e for e in log if e['kind']=='normal_reference_profile_attempt' and e['source']==row['component']]
        require(len(attempts)==1 and attempts[0]['encoder_starts']==attempts[0]['encoder_completions']==16,
                'BLOCKED_ACCOUNTING: ordinary profile genuine forwards')
        starts=[e for e in log if e['kind']=='operation_start' and e.get('label')=='genuine-normal-reference-profile' and e.get('source')==row['component']]
        require(len(starts)==1, 'BLOCKED_ACCOUNTING: profile operation identity')
        ends=[e for e in log if e['kind']=='operation_finish' and e['operation_id']==starts[0]['operation_id']]
        require(len(ends)==1 and ends[0]['success'] and 0<=r['elapsed_seconds']-ends[0]['collection_seconds']<.1,
                'BLOCKED_ACCOUNTING: profile elapsed boundary')
        times.append(r['elapsed_seconds'])
        evidence[r['array_path']]=sha(p)
        evidence[str(p.with_suffix('.json').relative_to(ROOT).as_posix())]=sha(p.with_suffix('.json'))
    require(abs(sum(times)-report['total_seconds'])<1e-9, 'BLOCKED_ACCOUNTING: total profile time')
    return dict(passed=True,sources=8,genuine_forwards=128,seconds=times,mean_seconds=statistics.mean(times),
        sample_sd_seconds=statistics.stdev(times),minimum_seconds=min(times),maximum_seconds=max(times),
        median_seconds=statistics.median(times),safety_factor=1.5,array_parity=True,evidence_sha256=evidence,
        profile_sha256=sha(OUT/'normal_reference_throughput_diagnosis.json'),
        boundary='Includes source hashing/read, decode, RGB preparation, CUDA-synchronized native inference, output writes and collection guard. Model setup separate. Smoke affine assertions and offline features/targets excluded and separately charged.',
        limitation='Eight preselected smoke sources are not a representative throughput study. Component times were not separately instrumented; no retrospective timing decomposition is claimed. Cached authentication/parity replay is CPU audit, never GPU throughput.')


def resource_projection(rows,profile):
    c=totals(); log=events()
    remaining=sum(not feature_path(r).exists() for r in rows)
    ordinary=[e['collection_seconds'] for e in log if e['kind']=='operation_finish' and e.get('label')=='native-reference' and e.get('success')]
    # Historical first eight are smoke; use only subsequent ordinary source work.
    observed=ordinary[8:]
    rate=max(profile['mean_seconds'],statistics.mean(observed) if observed else 0)
    cpu={label:[e['cpu_seconds'] for e in log if e['kind']=='operation_finish' and e.get('label')==label and e.get('success')]
         for label in ('offline-source-enumeration','source-input-controls','smoke-scalar-verification')}
    source_cpu=sum(statistics.mean(cpu[k]) for k in ('offline-source-enumeration','source-input-controls'))
    collection=remaining*rate*1.5
    scalar=statistics.mean(cpu['smoke-scalar-verification'])/8*len(rows)*1.5
    # Explicit unmeasured reserve; first actual fold must validate remaining fit work.
    cpu_needed=len(rows)*source_cpu*1.5+scalar+1800
    smoke=read(OUT/'smoke_manifest.json')
    bytes_per_source=max(feature_path(r).stat().st_size+feature_path(r).with_suffix('.json').stat().st_size for r in smoke)
    retained_projection=remaining*bytes_per_source*1.5+len(rows)*12*496*8*2+128*2**20
    result=dict(scope='Stage 1 only; complete-study gate remains unchanged in frozen integrated runner',
        remaining_sources=remaining,reference_safety_factor=1.5,reference_seconds_per_source=rate,
        remaining_collection_seconds=collection,remaining_cpu_projection_seconds=cpu_needed,
        scalar_verification_projection_seconds=scalar,unmeasured_oof_and_audit_reserve_seconds=1800,
        additional_retained_projection_bytes=int(retained_projection),scratch_reserve_bytes=2*2**30,
        current=c,available_disk_bytes=shutil.disk_usage(ROOT).free)
    result['fits_collection_cap']=c['collection_seconds']+collection<=CAPS['collection_seconds']
    result['fits_cpu_cap']=c['cpu_seconds']+cpu_needed<=CAPS['cpu_seconds']
    result['fits_retained_cap']=retained_bytes()+retained_projection<=CAPS['retained_bytes']
    result['fits_free_disk']=result['available_disk_bytes']>=FREE_RESERVE+result['scratch_reserve_bytes']+retained_projection
    return result


class PausedSession(Exception):
    pass


class Session:
    def __init__(self,seconds):
        require(0<seconds<=SESSION_LIMIT, 'Session allowance must be >0 and <=21600 seconds')
        self.start=time.perf_counter();self.allowance=seconds;self.id=uuid.uuid4().hex
        self.next_operation='initial audit'
        self.record=dict(id=self.id,started_at_utc=now(),allowance_seconds=seconds,
            counters_before=totals(),implementation_sha256=sha(__file__),repository_commit=git('rev-parse','HEAD'))
        self.checkpoint('RUNNING')

    def checkpoint(self,status):
        self.record.update(status=status,wall_seconds=time.perf_counter()-self.start,next_operation=self.next_operation,
                           counters=totals(),at_utc=now())
        save(TASK/'sessions'/(self.id+'.json'),self.record)

    def before(self,operation,reserve_seconds=0):
        self.next_operation=operation
        if time.perf_counter()-self.start+reserve_seconds>=self.allowance:
            self.checkpoint('PAUSED_SESSION')
            raise PausedSession(operation)


def atomic_intent(kind,key,inputs):
    path=TASK/'operations'/(rank(kind,key)+'.json')
    require(not path.exists(), 'BLOCKED_ACCOUNTING: existing atomic intent needs audit')
    value=dict(kind=kind,key=key,status='started',inputs=inputs,at_utc=now(),
               implementation_sha256=sha(__file__),counters_before=totals())
    save(path,value)
    return path,value


def complete_intent(path,value,artifacts):
    value.update(status='complete',completed_at_utc=now(),sha256={str(p.relative_to(ROOT).as_posix()):sha(p) for p in artifacts},counters_after=totals())
    save(path,value)


class FittingEncoder(NativeEncoder):
    def forward(self,rgbs,phase,validate=False):
        require(phase=='PROBE_FIT','BLOCKED_STAGE_BOUNDARY: only PROBE_FIT allowed')
        identifier=uuid.uuid4().hex
        # These are durable per-forward journals. Original collect alone owns the
        # cumulative encoder counters, avoiding double charging.
        event('task1_forward_intent',forward_id=identifier,phase=phase,attempted_calls=1)
        result=super().forward(rgbs,phase,validate)
        event('task1_forward_completion',forward_id=identifier,phase=phase,completed_calls=1)
        return result


def load_fold(fold,fs,predictions,rows):
    p=TASK/'folds'/f'{fold}.json'
    if not p.exists():
        require(not list((OUT/'probes').glob(f'{fold}_*.json')), 'BLOCKED_ACCOUNTING: partial fold needs audit before reuse')
        return False
    r=read(p)
    require(r['fold']==fold and r['manifest_sha256']==sha(OUT/'val_manifest.json.gz') and r['folds']==fs.tolist(),
            'BLOCKED_IDENTITY: fold membership mismatch')
    for name,digest in r['sha256'].items():
        require(sha(ROOT/name)==digest,'BLOCKED_IDENTITY: completed fold artifact changed')
    hold=np.concatenate([np.arange(i*12,(i+1)*12) for i in np.flatnonzero(fs==fold)])
    with np.load(ROOT/r['predictions_path'],allow_pickle=False) as saved:
        require(np.array_equal(saved['heldout_rows'],hold),'BLOCKED_IDENTITY: held-out rows changed')
        for target,views in predictions.items():
            for view,values in views.items():
                a=saved[target+'_'+view]
                require(a.shape==hold.shape and np.isfinite(a).all(),'BLOCKED_INPUT: invalid held-out predictions')
                values[hold]=a
    return True


def fitting_work(session):
    boundary()
    if state()['stage']=='OOF_COMPLETE':
        require(read(TASK/'oof_verification.json')['passed'], 'BLOCKED_VERIFICATION: completed OOF audit missing')
        session.next_operation='Task 2 budget authorization required; do not run integrated resume'
        return 'OOF_COMPLETE'
    from verify_native_head_value_confirmation import verify_source_certificate
    session.before('independent fitting certificate audit',120)
    with metered('cpu','task1-independent-source-audit'):
        audit=verify_source_certificate(OUT,'fitting')
        save(TASK/'source_audit.json',audit)
        require(audit['passed'],'BLOCKED_INPUT: fitting certificate failed '+repr(audit['errors']))
        rows=read(OUT/'val_manifest.json.gz');validate_components(rows,8)
        require(sha(OUT/'val_manifest.json.gz')==read(OUT/'fitting_source_certificate.json')['sha256'][
            'results/native_head_value_confirmation_v1/certificates/fitting/val_manifest.json.gz'], 'BLOCKED_IDENTITY: active fitting manifest differs')
        verified=sum(authenticate_reference(r) for r in rows)
        profile=ordinary_profile();save(TASK/'ordinary_profile_verification.json',profile)
    with metered('cpu','task1-fitting-resource-projection'):
        projection=resource_projection(rows,profile);save(TASK/'fitting_resource_projection.json',projection)
    require(all(projection[k] for k in ('fits_collection_cap','fits_cpu_cap','fits_retained_cap','fits_free_disk')),
            'BLOCKED_RESOURCE: fitting-only workload fails cumulative resource projection')
    session.before('collect remaining fitting references',max(profile['seconds'])*2)
    encoder=None;data=[]
    if verified<len(rows):
        with metered('collection','task1-native-model-setup'):
            encoder=FittingEncoder()
    for i,row in enumerate(rows):
        session.before('source '+row['filename'],max(profile['seconds'])*2)
        if i%32==0:
            with metered('cpu','task1-progress-resource-check'):
                projection=resource_projection(rows,profile)
                save(TASK/'fitting_resource_projection.json',projection)
            require(all(projection[k] for k in ('fits_collection_cap','fits_cpu_cap','fits_retained_cap','fits_free_disk')),
                    'BLOCKED_RESOURCE: observed workload no longer fits cumulative caps')
        with metered('cpu','task1-source-resume-authentication'):
            exists=authenticate_reference(row)
        intent=None
        if not exists:
            intent=atomic_intent('source',row['component'],dict(encoded_sha256=row['encoded_sha256'],grid=row['grid'],planned_forwards=16))
        data.extend(collect([row],encoder,'PROBE_FIT'))
        if intent:
            p=feature_path(row)
            complete_intent(*intent,[p,p.with_suffix('.json')])
        session.next_operation='source '+rows[i+1]['filename'] if i+1<len(rows) else 'OOF fold 0'
        session.checkpoint('RUNNING')
        if (i+1)%32==0:print(json.dumps(dict(references_processed=i+1,native_completions=totals()['encoder_completions'],collection_seconds=totals()['collection_seconds'])),flush=True)
    fs=folds(rows)
    predictions={target:{v:np.full(len(rows)*12,np.nan) for v in VIEWS} for target in ('advantage','drift')}
    for f in range(4):
        session.before('OOF fold '+str(f),300)
        if load_fold(f,fs,predictions,rows):continue
        intent=atomic_intent('fold',str(f),dict(folds=fs.tolist(),manifest_sha256=sha(OUT/'val_manifest.json.gz')))
        before=time.perf_counter()
        with metered('cpu','task1-oof-fold',fold=f):
            fit_models(data,np.flatnonzero(fs!=f),np.flatnonzero(fs==f),predictions,f)
            held=np.concatenate([np.arange(i*12,(i+1)*12) for i in np.flatnonzero(fs==f)])
            p=TASK/'folds'/f'{f}.npz'
            atomic_npz(p,heldout_rows=held,**{t+'_'+v:a[held] for t,views in predictions.items() for v,a in views.items()})
            models=[OUT/'probes'/f'{f}_{t}_{v}.json' for t in predictions for v in VIEWS]
            record=dict(fold=f,folds=fs.tolist(),manifest_sha256=sha(OUT/'val_manifest.json.gz'),
                predictions_path=str(p.relative_to(ROOT).as_posix()),
                sha256={str(q.relative_to(ROOT).as_posix()):sha(q) for q in [p,*models]},
                model_identities={q.name:rank('model',json.dumps(read(q)['model'],sort_keys=True)) for q in models},
                scaler_identities={q.name:rank('scaler',json.dumps({k:read(q)['model'][k] for k in ('means','scales')},sort_keys=True)) for q in models})
            save(TASK/'folds'/f'{f}.json',record)
        complete_intent(*intent,[p,TASK/'folds'/f'{f}.json',*models])
        duration=time.perf_counter()-before
        require(totals()['cpu_seconds']+(3-f)*duration*1.5+projection['scalar_verification_projection_seconds']<=CAPS['cpu_seconds'],
                'BLOCKED_RESOURCE: measured OOF cost exceeds remaining CPU allowance')
        session.checkpoint('RUNNING')
    session.before('OOF diagnostics, precision, independent fitting verification',projection['scalar_verification_projection_seconds'])
    with metered('cpu','task1-oof-summary'):
        require(all(np.isfinite(a).all() for views in predictions.values() for a in views.values()), 'BLOCKED_VERIFICATION: incomplete OOF predictions')
        summaries=source_values(data,predictions)
        save(OUT/'fit_oof_source_summaries.json',summaries)
        atomic_npz(OUT/'fit_oof_state_predictions.npz',**{t+'_'+v:a for t,views in predictions.items() for v,a in views.items()})
        save(OUT/'fit_folds.json',fs.tolist())
        pp=precision_plan(rows,[r['primary'] for r in summaries])
        pp.update(source_summaries_sha256=sha(OUT/'fit_oof_source_summaries.json'),predictions_sha256=sha(OUT/'fit_oof_state_predictions.npz'))
        save(OUT/'precision_plan.json',pp)
        diagnostics=dict(scope='Fitting-only OOF development evidence; not fresh evaluation confirmation',targets={},fold_variation={})
        for target in predictions:
            yy=np.concatenate([d['targets'][target] for d in data])
            diagnostics['targets'][target]=dict(quantiles=np.quantile(yy,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),zero=int((yy==0).sum()),near_zero=int((abs(yy)<=1e-12).sum()))
            diagnostics['fold_variation'][target]={v:{k:[read(OUT/'probes'/f'{f}_{target}_{v}.json')[k] for f in range(4)] for k in ('train_mae','heldout_mae','constant_train_mae','constant_heldout_mae')} for v in VIEWS}
        diagnostics['source_distributions']={k:np.quantile([r[k] for r in summaries],[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist() for k in summaries[0] if k.startswith(('V_','advantage_mae_','drift_mae_')) or k in ('primary','drift_contrast')}
        diagnostics['class_means']={str(c):{k:float(np.mean([r[k] for r in summaries if r['y']==c])) for k in diagnostics['source_distributions']} for c in range(400)}
        save(OUT/'fit_diagnostics.json',diagnostics)
    from verify_native_head_task1 import verify_fitting
    with metered('cpu','task1-independent-oof-verification'):
        checked=verify_fitting();save(TASK/'oof_verification.json',checked)
    require(checked['passed'],'BLOCKED_VERIFICATION: independent fitting verification failed '+repr(checked['errors']))
    require(pp['selected_m'] is not None,'BLOCKED_PRECISION_OR_POPULATION: no m in 8..20 meets half-width 0.005')
    advance('OOF_COMPLETE',[OUT/'precision_plan.json',OUT/'fit_oof_state_predictions.npz',TASK/'oof_verification.json'])
    return 'OOF_COMPLETE'


def accounting():
    c=totals();net=network_ledger()
    c.update(network_cap_charged_bytes=sum(r['bytes'] for r in net['requests']),
        known_response_body_bytes=sum(r.get('known_response_body_bytes',r.get('bytes_received',r['bytes'])) for r in net['requests']),
        retained_bytes=retained_bytes(),scratch_bytes=sum(p.stat().st_size for p in (CACHE/'scratch').rglob('*') if p.is_file() and p.name.endswith(('.tar.gz','.part'))),
        free_disk_bytes=shutil.disk_usage(ROOT).free,free_disk_reserve_bytes=FREE_RESERVE,memory=memory_snapshot())
    c['remaining']={k:limit-c.get({'network_bytes':'network_cap_charged_bytes'}.get(k,k),0) for k,limit in CAPS.items()}
    return c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['fit-plan'])
    parser.add_argument('--session-wall-seconds',type=float,default=SESSION_LIMIT)
    args=parser.parse_args()
    with run_lock(),threadpool_limits(limits=1):
        session=Session(args.session_wall_seconds)
        status='BLOCKED_VERIFICATION';blocker=None
        before=totals()
        try:
            freeze_operational_controls()
            status=fitting_work(session)
        except PausedSession as exc:
            status='PAUSED_SESSION';blocker=str(exc)
        except Exception as exc:
            blocker=str(exc)
            status=blocker.split(':',1)[0] if blocker.startswith('BLOCKED_') else 'BLOCKED_VERIFICATION'
        finally:
            # Charge process wall residual (administration, hashing and checkpoint
            # writes outside measured phases) to CPU, never subtract earlier cost.
            after=totals();wall=time.perf_counter()-session.start
            charged=sum(after[k]-before[k] for k in ('collection_seconds','cpu_seconds','timing_seconds'))
            event('task1-session-residual',cpu_seconds=max(0,wall-charged),session_id=session.id)
            session.checkpoint(status)
            report=dict(technical=status,scientific='NOT_EVALUATED',blocker=blocker,completed_stage=state()['stage'],
                session=session.record,resources=accounting(),evaluation_exposure=False,
                next_command='python fit_plan_native_head_confirmation.py fit-plan --session-wall-seconds 21600')
            save(OUT/'task1_status.json',report)
            print(json.dumps(report,indent=2),flush=True)
    return 0 if status=='OOF_COMPLETE' else 2


if __name__=='__main__':
    raise SystemExit(main())
