"""Actual encoded-memory-to-prediction GPU trials, independent of saved features."""
from __future__ import annotations
import ctypes
from ctypes import wintypes
import gc
import io
import random
import time

import av
import numpy as np
import pandas as pd
import torch

from temporal_refresh_common import *
from research_utils import configure_decoder


def working_set():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
            *[(name,ctypes.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]]
    c=Counters(); c.cb=ctypes.sizeof(c)
    handle=ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle),ctypes.byref(c),c.cb)
    return int(c.WorkingSetSize),int(c.PeakWorkingSetSize)


def process(encoded,entry,rep,positions,encoders,head):
    # End-to-end offline clip path includes candidate planning. Buffer decoder
    # frame objects inside the timer, never predecoded input or RGB features.
    # This avoids treating the full-clip timestamp inspection as free setup.
    decoded_frames=[];timestamps=[]; values=[]; converted=0; calls=0
    with av.open(io.BytesIO(encoded),mode='r') as container:
        stream=container.streams.video[0]; configure_decoder(stream)
        for frame in container.decode(stream):
            decoded_frames.append(frame);timestamps.append(float(frame.pts*frame.time_base))
        # Same deterministic floor mathematics, without diagnostic assertions.
        grid=operational_grid(timestamps)
        needed=set(sum([grid[t] if rep!=REPS[0] else [grid[t][-1]] for t in positions],[]))
        frames={i:decoded_frames[i].to_ndarray(format='rgb24') for i in sorted(needed)};converted=len(frames)
        for t in positions:
            rgbs=[frames[i] for i in grid[t]] if rep!=REPS[0] else [frames[grid[t][-1]]]
            feature=encoders.features(rgbs,rep); calls+=4 if rep==REPS[1] else 1
            values.append(feature)
    durations=np.diff(positions+[16])/16
    pooled=sum(float(w)*f.to(torch.float64) for w,f in zip(durations,values))
    prediction=((pooled-head['means'])/head['scales'])@head['weights'].T+head['bias']
    return {'logits':prediction,'prediction':prediction.argmax(1),'features':values,'decoded_frames':len(decoded_frames),'computed_grid':grid,
        'encoder_calls':calls,'refreshed_candidates':len(positions),'input_images':calls if rep!=REPS[2] else 4*len(positions),
        'rgb_conversions':converted,'candidate_outputs':16}


def run_timing(output,entries,encoders):
    require_evaluation_seal(output)
    target=output/'raw_timing_trials.csv'
    if target.exists(): return pd.read_csv(target,float_precision='round_trip')
    manifest=read(output/'evaluation_manifest.json'); ids=manifest['timing_sources']
    entries=sorted([v for v in entries if v['source_video_id'] in ids],key=lambda v:v['source_video_id'])
    assert len(entries)==20 and len({v['label'] for v in entries})==20
    modes=[(r,m) for r in REPS for m in ['dense16','periodic_B4']]
    rows=[]; execution=[]; loads=[]; rng=random.Random(SEED); progress=output/'timing_progress.json'
    device_context=[]
    if progress.exists():
        saved=read(progress); rows=saved['rows']; execution=saved['execution']; loads=saved['loads']
    completed={r['source_video_id'] for r in rows if r['mode']=='COMPLETE'}
    rows=[r for r in rows if r['mode']!='COMPLETE']
    encoders.log_calls=False # In-memory counters retained; no logging in timed region.
    attempt_log=(output/'timing_attempt_events.jsonl').open('a',encoding='utf-8',buffering=1)
    heads={}
    for rep in REPS:
        params=read(output/f'heads/{rep}/parameters.json')['parameters']
        heads[rep]={k:torch.tensor(v,dtype=torch.float64,device='cuda') for k,v in params.items()}
    with torch.inference_mode():
        for vi,v in enumerate(entries):
            if v['source_video_id'] in completed: continue
            import subprocess
            device_context.append({'source_video_id':v['source_video_id'],'at_utc':now(),
                'nvidia_smi_before_video':subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader'],text=True).strip()})
            # Deterministic per-source ordering permits exact restart, independent
            # of how many earlier videos were completed.
            rng=random.Random(int(rank('timing-order',v['source_video_id'])[:16],16))
            load_start=time.perf_counter(); encoded=(ROOT/v['relative_path']).read_bytes(); loading=time.perf_counter()-load_start
            assert hashlib.sha256(encoded).hexdigest()==v['file_identity']['sha256']
            loads.append({'source_video_id':v['source_video_id'],'bytes':len(encoded),'file_loading_seconds':loading})
            local=[]
            for repetition in range(-2,5):
                order=modes.copy(); rng.shuffle(order)
                execution.append({'source_video_id':v['source_video_id'],'run_id':repetition,'warmup':repetition<0,'order':[list(x) for x in order]})
                for rep,mode in order:
                    positions=list(range(16)) if mode=='dense16' else [0,4,8,12]
                    encoders.phase='timing_warmup' if repetition<0 else 'timing_measured'
                    encoders.capture={}; encoders.native=None; gc.collect()
                    cpu_before,_=working_set(); torch.cuda.synchronize()
                    resident=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
                    token=str(time.time_ns())
                    attempt_log.write(json.dumps({'event':'start','token':token,'source_video_id':v['source_video_id'],
                        'representation':rep,'mode':mode,'run_id':repetition,'expected_encoder_calls':len(positions)*(4 if rep==REPS[1] else 1),
                        'at_utc':now()})+'\n')
                    begin=time.perf_counter()
                    result=process(encoded,v,rep,positions,encoders,heads[rep])
                    torch.cuda.synchronize()
                    elapsed=(time.perf_counter()-begin)*1000
                    peak=torch.cuda.max_memory_allocated(); cpu_after,cpu_lifetime_peak=working_set()
                    attempt_log.write(json.dumps({'event':'completed','token':token,'encoder_calls':result['encoder_calls'],
                        'decoded_frames':result['decoded_frames'],'at_utc':now()})+'\n')
                    assert result['decoded_frames']==v['decoded_frame_count']
                    assert result['computed_grid']==v['snippet_grid']['frame_indices']
                    assert result['encoder_calls']==len(positions)*(4 if rep==REPS[1] else 1)
                    # All correctness checks and CPU copies are outside timing.
                    logits=result['logits'].cpu().numpy()[0]
                    saved_features=np.load(ROOT/read(output/'evaluation_collection.json')['videos'][v['source_video_id']]['path'])[rep]
                    cache=np.maximum.accumulate(np.where(np.isin(np.arange(16),positions),np.arange(16),0))
                    params=read(output/f'heads/{rep}/parameters.json')['parameters']
                    mean=saved_features[cache].mean(0,dtype=np.float64)
                    expected=((mean-params['means'])/params['scales'])@np.asarray(params['weights']).T+params['bias']
                    np.testing.assert_allclose(logits,expected,atol=1e-4,rtol=1e-4)
                    if repetition>=0:
                        local.append({'source_video_id':v['source_video_id'],'label':v['label'],'representation':rep,'mode':mode,
                            'run_id':repetition,'total_ms':elapsed,**{k:result[k] for k in ['decoded_frames','encoder_calls','refreshed_candidates','input_images','rgb_conversions','candidate_outputs']},
                            'cuda_resident_allocated_bytes':resident,'cuda_peak_allocated_bytes':peak,'cuda_incremental_peak_bytes':max(0,peak-resident),
                            'cpu_working_set_before_bytes':cpu_before,'cpu_working_set_after_bytes':cpu_after,'cpu_process_lifetime_peak_bytes':cpu_lifetime_peak,
                            'clip_duration_seconds':v['duration_seconds'],'clip_seconds_per_wall_second':1000*v['duration_seconds']/elapsed})
                    del result
            rows.extend(local); completed.add(v['source_video_id'])
            save(progress,{'rows':rows+[{'source_video_id':s,'mode':'COMPLETE'} for s in sorted(completed)],'execution':execution,'loads':loads})
            print(f'Encoded-memory timing {vi+1}/20 complete',flush=True)
    encoders.log_calls=True
    attempt_log.close()
    raw=pd.DataFrame(rows); assert len(raw)==20*6*5
    save(output/'timing_device_context.json',device_context)
    raw.to_csv(target,index=False,float_format='%.17g'); save(output/'timing_execution.json',execution); save(output/'file_loading_times.json',loads)
    raw.groupby(['source_video_id','label','representation','mode']).agg(median_ms=('total_ms','median'),min_ms=('total_ms','min'),max_ms=('total_ms','max'),
        incremental_gpu_peak_bytes=('cuda_incremental_peak_bytes','max')).reset_index().to_csv(output/'timing_per_video.csv',index=False,float_format='%.17g')
    save(output/'timing_protocol.json',{'boundary':'encoded bytes in memory through full sequential decode and within-trial decoded-frame buffering, timestamp candidate planning, frame preparation, transfers, float32 encoders, float64 pooling/head and completed GPU prediction',
        'exclusions':'disk load/model construction/downloads/assertions/logging/CPU feature copies; setup and file loading separate',
        'warmups':2,'measured_repetitions':5,'thread_count':1,'interleave_seed':SEED,'batch_size':1,
        'B_operational_choice':'four independent one-frame ResNet calls per refreshed snippet, no cross-snippet feature caching',
        'C_operational_choice':'native400 projection/global-average still executes; read-only hook extracts pre-projection features; this unused native-head work is included in C latency',
        'GPU_memory':'both models plus three heads resident throughout; torch allocated baseline, reset per-trial peak, difference; allocator reserved not device-wide physical memory',
        'CPU_memory':'Windows GetProcessMemoryInfo working set before/after; lifetime peak is process-wide, not a per-trial incremental peak',
        'throughput':'encoded clip duration seconds divided by whole-trial wall seconds; offline throughput, not streaming response latency',
        'forward_counts_total_including_warmups':{r:20*7*(16+4)*(4 if r==REPS[1] else 1) for r in REPS}})
    return raw


def training_timing_smoke(output,entries,encoders):
    target=output/'training_timing_smoke.json'
    if target.exists():return
    v=min([v for v in entries if v['recognition_role']=='train'],key=lambda v:rank('timing-smoke',v['source_video_id']))
    encoded=(ROOT/v['relative_path']).read_bytes(); rows=[];encoders.phase='training_timing_smoke'
    ledger=read(output/'train_development_collection.json')['videos'][v['source_video_id']]
    with torch.inference_mode(),np.load(ROOT/ledger['path']) as saved:
        for rep in REPS:
            params=read(output/f'heads/{rep}/parameters.json')['parameters']
            head={k:torch.tensor(x,dtype=torch.float64,device='cuda') for k,x in params.items()}
            for positions in [list(range(16)),[0,4,8,12]]:
                result=process(encoded,v,rep,positions,encoders,head);torch.cuda.synchronize()
                assert result['computed_grid']==v['snippet_grid']['frame_indices']
                assert result['decoded_frames']==v['decoded_frame_count']
                cache=np.maximum.accumulate(np.where(np.isin(np.arange(16),positions),np.arange(16),0))
                mean=saved[rep][cache].mean(0,dtype=np.float64)
                expected=((mean-params['means'])/params['scales'])@np.array(params['weights']).T+params['bias']
                actual=result['logits'].cpu().numpy()[0]
                np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-4)
                rows.append({'representation':rep,'refreshes':len(positions),'decoded_frames':result['decoded_frames'],
                    'maximum_logit_difference':float(np.max(abs(actual-expected))),'encoder_calls':result['encoder_calls']})
    save(target,{'source_video_id':v['source_video_id'],'role':'training','checks':rows,'not_a_latency_measurement':True})
