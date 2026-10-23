"""Persistent E27 stages and exclusive, crash-visible resource accounting."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from native_head_value_confirmation import read, save, sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/native_head_value_confirmation_v1'
CACHE = ROOT/'data/source/native_head_value_confirmation_v1'
STAGES = ['PREPARED','FIT_CERTIFIED','OOF_COMPLETE','JOINT_CERTIFIED','SEALED',
          'EVAL_COLLECTED','ANALYZED','TIMED','VERIFIED_PENDING_VISUAL_REVIEW','COMPLETE']
FREE_RESERVE = 10*2**30
_active = None
_ledger_position=0
_ledger_sums=None
_pending={}


def initialize():
    if not (OUT/'stage_state.json').exists():
        if (OUT/'evaluation_exposure.json').exists():
            raise RuntimeError('Existing exposure requires stage migration, never zero initialization')
        save(OUT/'stage_state.json',dict(stage='PREPARED',receipts=[],schema=1))
    if not (OUT/'retained_baseline.json').exists():
        save(OUT/'retained_baseline.json',dict(
            reviewed_commit='a9e1942f226f341071e1a990ea446dc717a070bb',
            pre_E27_task_bytes=0,scope='All E27 cache/result files count, including preparation and snapshots.',
            historical_roots=['data/source/cache_drift_v2','data/source/temporal_refresh_generalization_v1'],
            reuse='Historical files authenticated individually before use; never deleted or modified.',
            free_disk_reserve_bytes=FREE_RESERVE,
            original_accounting_sha256=sha(OUT/'preparation_a9e1942/execution_accounting.json'),
            initial_failed_preparation_wall_time='unknown, preserved; complete historic end-to-end accounting unavailable'))
    recover_interrupted_operations()


def state():
    return read(OUT/'stage_state.json')


def advance(stage, paths=()):
    current = state()
    if STAGES.index(stage)<STAGES.index(current['stage']):
        receipt=next((r for r in reversed(current['receipts']) if r['stage']==stage),None)
        if receipt is None or any(sha(ROOT/p)!=v for p,v in receipt['sha256'].items()):
            raise RuntimeError('Earlier completed stage receipt is missing or changed')
        return
    if stage==current['stage']:
        latest=next((r for r in reversed(current['receipts']) if r['stage']==stage),None)
        if not paths or (latest and latest['sha256']=={str(Path(p).relative_to(ROOT).as_posix()):sha(p) for p in paths}):return
        if (OUT/'evaluation_exposure.json').exists():raise RuntimeError('Stage receipt replacement after EVAL exposure is forbidden')
    from run_native_head_value_confirmation import now
    current['receipts'].append(dict(stage=stage,at_utc=now(),
        sha256={str(Path(p).relative_to(ROOT).as_posix()):sha(p) for p in paths}))
    current['stage']=stage
    save(OUT/'stage_state.json',current)


def totals():
    global _ledger_position,_ledger_sums,_pending
    path=OUT/'resource_exposure_ledger.jsonl'
    keys=('encoder_starts','encoder_completions','decodes','decoded_frames','probe_fits','probe_completions','verification_probe_fits',
          'collection_seconds','cpu_seconds','timing_seconds','acquisition_seconds','acquisition_upper_bound_extra_seconds','uncertain_encoder_calls')
    if _ledger_sums is None:_ledger_sums={k:0 for k in keys}
    if path.exists():
        if path.stat().st_size<_ledger_position:raise RuntimeError('Append-only resource ledger was truncated')
        with path.open('rb') as f:
            f.seek(_ledger_position)
            while True:
                start=f.tell();line=f.readline()
                if not line or not line.endswith(b'\n'):_ledger_position=start;break
                r=json.loads(line)
                for k in keys:_ledger_sums[k]+=r.get(k,0)
                if r['kind']=='operation_start':_pending[r['operation_id']]=r
                elif r['kind']=='operation_finish':_pending.pop(r['operation_id'],None)
                _ledger_position=f.tell()
    result=dict(_ledger_sums)
    result['uncertain_operations']=list(_pending.values())
    return result


def retained_bytes():
    def size_of(directory):
        total=0
        scratch=Path(directory).is_relative_to(CACHE/'scratch')
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():raise RuntimeError('BLOCKED_INPUT: unexpected link in task-owned resource tree')
                if entry.is_dir(follow_symlinks=False):
                    total+=size_of(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    if not scratch or not entry.name.endswith(('.tar.gz','.part')):total+=entry.stat(follow_symlinks=False).st_size
        return total
    return sum(size_of(base) for base in (CACHE,OUT) if base.exists())


def guard(category,additional_retained=0,scratch=0):
    from run_native_head_value_confirmation import CAPS
    counters=totals()
    for pending in counters['uncertain_operations']:
        if pending['pid']!=os.getpid():
            raise RuntimeError('BLOCKED_RESOURCE: interrupted work remains attempted/uncertain: '+pending['operation_id'])
    key=category+'_seconds'
    active_elapsed=sum(max(0,time.time()-r['started_epoch']) for r in counters['uncertain_operations'] if r['category']==category)
    if key in CAPS and counters.get(key,0)+active_elapsed>=CAPS[key]:
        raise RuntimeError('BLOCKED_RESOURCE: cumulative '+category+' time exhausted')
    if retained_bytes()+additional_retained>CAPS['retained_bytes']:
        raise RuntimeError('BLOCKED_RESOURCE: additional retained-data cap')
    if shutil.disk_usage(ROOT).free<FREE_RESERVE+scratch+additional_retained:
        raise RuntimeError('BLOCKED_RESOURCE: peak working space would consume free-disk reserve')


@contextmanager
def metered(category,label,**fields):
    global _active
    if _active is not None:
        if _active!=category:
            raise RuntimeError('Overlapping resource categories: '+_active+'/'+category)
        yield
        return
    from run_native_head_value_confirmation import event
    operation_id=uuid.uuid4().hex
    start=time.perf_counter()
    event('operation_start',operation_id=operation_id,category=category,label=label,
          pid=os.getpid(),process_created=process_identity(os.getpid()),started_epoch=time.time(),**fields)
    _active=category
    success=False
    try:
        guard(category)
        yield
        success=True
    finally:
        _active=None
        event('operation_finish',operation_id=operation_id,category=category,label=label,
              success=success,**{category+'_seconds':time.perf_counter()-start})


def recover_lock(path):
    from run_native_head_value_confirmation import event
    old=read(path)
    created=process_identity(old['pid'])
    if created is not None and ('process_created' not in old or created==old['process_created']):
        raise RuntimeError('Task lock owner is alive; no recovery permitted')
    event('stale_lock_recovered',previous=old)
    path.unlink()


def process_identity(pid):
    """Windows process birth identity without installing or upgrading packages."""
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        if ctypes.get_last_error()==87:return None
        raise RuntimeError('Cannot authenticate lock owner process, error '+str(ctypes.get_last_error()))
    try:
        code=wintypes.DWORD()
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        if code.value!=259:return None
        times=[wintypes.FILETIME() for _ in range(4)]
        kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
        if not kernel.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):raise ctypes.WinError(ctypes.get_last_error())
        return (times[0].dwHighDateTime<<32)|times[0].dwLowDateTime
    finally:
        kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle(handle)


def recover_interrupted_operations():
    """Preserve unknown work with a conservative upper wall-time charge."""
    from run_native_head_value_confirmation import event
    for pending in totals()['uncertain_operations']:
        if pending['pid']==os.getpid():continue
        created=process_identity(pending['pid'])
        if created is not None and ('process_created' not in pending or created==pending['process_created']):
            raise RuntimeError('Another operation owner is live; no recovery permitted')
        upper=max(0,time.time()-pending['started_epoch'])
        event('operation_finish',operation_id=pending['operation_id'],category=pending['category'],
            label=pending['label'],success=False,recovered_interruption=True,
            elapsed_basis='conservative upper bound through demonstrated stale-process recovery; actual crash time unknown',
            uncertain_encoder_calls=pending.get('planned_encoder_calls',0),
            **{pending['category']+'_seconds':upper})


def memory_snapshot():
    import ctypes
    from ctypes import wintypes
    class ProcessMemory(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    class SystemMemory(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(name,ctypes.c_ulonglong) for name in
            ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    current=ProcessMemory();current.cb=ctypes.sizeof(current)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessMemory),wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(current),current.cb):raise ctypes.WinError(ctypes.get_last_error())
    system=SystemMemory();system.length=ctypes.sizeof(system)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(system)):raise ctypes.WinError(ctypes.get_last_error())
    return dict(process_working_set_bytes=current.WorkingSetSize,process_peak_working_set_bytes=current.PeakWorkingSetSize,
        system_ram_bytes=system.total,system_available_ram_bytes=system.available,
        scope='Current process lifetime; earlier unmetered processes are not inferred')


def snapshot_accounting():
    from run_native_head_value_confirmation import network_ledger, now
    result=totals()
    network=network_ledger()
    result.update(at_utc=now(),new_http_body_bytes=sum(r['bytes'] for r in network['requests']),
        inflight_network=network.get('inflight'),additional_retained_bytes=retained_bytes(),
        free_disk_bytes=shutil.disk_usage(ROOT).free,free_disk_reserve_bytes=FREE_RESERVE,
        historic_unknowns=read(OUT/'retained_baseline.json')['initial_failed_preparation_wall_time'])
    result['known_recorded_http_body_bytes']=sum(r.get('known_response_body_bytes',r.get('bytes_received',r['bytes'])) for r in network['requests'])
    result['network_cap_charged_bytes']=result['new_http_body_bytes']
    result['uncertain_network_upper_bound_extra_bytes']=result['network_cap_charged_bytes']-result['known_recorded_http_body_bytes']
    result['network_count_scope']='Known response bodies plus explicit conservative interrupted-read upper reservations; Git/browser control traffic separately unmetered.'
    result['known_timed_acquisition_seconds']=result['acquisition_seconds']-result['acquisition_upper_bound_extra_seconds']
    result['timer_boundaries']='Acquisition is a subinterval of collection, never added again. Interrupted acquisition time uses a separately disclosed conservative upper charge.'
    result['memory']=memory_snapshot()
    resumed=OUT/'resumed_checkout_audit.json'
    if resumed.exists() and read(resumed).get('goal_started_epoch'):
        result['resumed_end_to_end_wall_seconds']=time.time()-read(resumed)['goal_started_epoch']
    result['current_temporary_working_bytes']=sum(p.stat().st_size for p in (CACHE/'scratch').rglob('*') if p.is_file() and p.name.endswith(('.tar.gz','.part'))) if (CACHE/'scratch').exists() else 0
    save(OUT/'execution_accounting.json',result)
    plan_path=OUT/'resource_plan.json'
    if plan_path.exists():
        plan=read(plan_path)
        plan.update(no_bulk_acquisition=not bool(list((CACHE/'catalogs').glob('*.json'))),
            current_network_cap_charged_bytes=result['network_cap_charged_bytes'],
            known_recorded_http_body_bytes=result['known_recorded_http_body_bytes'],
            current_additional_retained_bytes=result['additional_retained_bytes'],
            current_temporary_working_bytes=result['current_temporary_working_bytes'],
            current_available_disk_bytes=result['free_disk_bytes'])
        save(plan_path,plan)
    return result
