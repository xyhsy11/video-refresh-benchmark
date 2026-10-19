"""Real UCF DEVELOPMENT policy learning/execution; no official-test access."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch
import csv
import gzip
import io
import json
import math
import os
import time
import numpy as np
from scipy.special import logsumexp
from ucf101_causal_development_common import *
from causal_observability_policy import Policy, Observation, affine, vector, cheap_signal, TrainingReplayProvider, execute


def decode(raw,entry):
    """Real full AVI decode with actual presentation identities; no RGB yet."""
    import av
    from research_utils import configure_decoder
    from temporal_refresh_common import snippet_grid
    from ucf101_proxy_data import presentation_order
    entry_allowed(entry);frames=[];pts=[];ids=[]
    with av.open(io.BytesIO(raw)) as container:
        stream=container.streams.video[0];configure_decoder(stream)
        for i,f in enumerate(container.decode(stream)):
            require(f.pts is not None and f.time_base is not None,'Missing actual PTS')
            frames.append(f);pts.append(float(f.pts*f.time_base));ids.append(dict(index=i,pts=int(f.pts),time_base=str(f.time_base)))
        frames,pts,ids,_=presentation_order(frames,pts,ids)
    grid=snippet_grid(pts);grid['frame_identities']=[[ids[j] for j in row] for row in grid['frame_indices']]
    require(grid==entry['snippet_grid'] and len(frames)==entry['decoded_frame_count'],'Actual timestamp/grid differs from screened original')
    return frames,grid


class Provider:
    """Only sequential current-thumbnail and actual current-X3D operations."""
    def __init__(self,frames,grid,encoder):
        self._frames,self._grid,self._encoder=frames,grid,encoder
        self.t=-1;self.used=False;self.forward_completions=0;self.features={}
        self.RGB_conversions=0;self.grayscale_conversions=0;self.thumbnail_seconds=0.
    def advance(self,t):
        require(t==self.t+1,'Nonchronological access');self.t=t;self.used=False
    def thumbnail(self):
        tick=time.perf_counter();frame=self._frames[self._grid['endpoint_indices'][self.t]]
        a=frame.reformat(width=32,height=32,format='gray').to_ndarray().copy()
        self.grayscale_conversions+=1;self.thumbnail_seconds+=time.perf_counter()-tick;return a
    def refresh(self):
        require(not self.used,'Repeated current forward');self.used=True
        rgb=[self._frames[j].to_ndarray(format='rgb24') for j in self._grid['frame_indices'][self.t]]
        self.RGB_conversions+=len(rgb)
        f=self._encoder.feature(rgb,self.t);self.forward_completions+=1;self.features[self.t]=np.asarray(f).copy();return f


class LivePolicy(Policy):
    """Identical fixed numpy learner; retain its actual imitation probability."""
    def decide(self,obs):
        self.last_probability=None
        if self.method not in ['history','history_motion']:return super().decide(obs)
        x=vector(obs,self.head,self.method=='history_motion');m=self.model
        x=(x-np.asarray(m['means']))/np.asarray(m['scales'])
        h=np.maximum(0,np.asarray(m['w1']) @ x+np.asarray(m['b1']))
        logit=float(np.asarray(m['w2']) @ h+float(m['b2']))
        self.last_probability=float(1/(1+np.exp(-np.clip(logit,-700,700))))
        return self.last_probability>=self.threshold


def causal_execute(provider,policy,head):
    history=[];positions=[];cached=[];trace=[];last=-1;previous=None
    statistics_seconds=decide_seconds=observation_seconds=0.
    for t in range(16):
        provider.advance(t);start=time.perf_counter();signal=()
        if policy.needs_cheap:
            current=provider.thumbnail();tick=time.perf_counter();signal=tuple(cheap_signal(current,previous))
            statistics_seconds+=time.perf_counter()-tick;previous=current
        remaining=4-len(positions)
        obs=Observation(t,t-last,remaining,tuple(tuple(row) for row in history),signal)
        forced=t==0 or remaining==16-t
        observation_seconds+=time.perf_counter()-start
        probability=None;tick=time.perf_counter()
        if not remaining:selected=False
        elif forced:selected=True
        else:
            selected=bool(policy.decide(obs));probability=getattr(policy,'last_probability',None)
        elapsed=time.perf_counter()-tick;decide_seconds+=elapsed
        trace.append(dict(t=t,age=t-last,remaining_before=remaining,available_refreshes=positions.copy(),
            selected=selected,forced=forced,cheap_signal=list(signal),imitation_probability=probability,
            decision_seconds=elapsed))
        if selected:
            f=np.asarray(provider.refresh(),dtype=float);history.append(f.copy());positions.append(t);last=t
        require(bool(history),'No mandatory initial cache');cached.append(history[-1])
    require(len(positions)==4 and positions[0]==0 and provider.forward_completions==4,'Exact real four-forward budget failed')
    z=affine(np.mean(cached,axis=0,dtype=np.float64),head)
    return dict(logits=z.tolist(),prediction=int(z.argmax()),positions=positions,trace=trace,
        cache_indices=np.maximum.accumulate([t if t in positions else 0 for t in range(16)]).tolist(),
        cheap_statistics_seconds=statistics_seconds,controller_decide_seconds=decide_seconds,
        observation_preparation_seconds=observation_seconds)


def encoder_factory():
    import causal_observability_runtime as old_runtime
    class DevelopmentEncoder(old_runtime.Encoder):
        def __init__(self):
            self.buffer=[]
            with patch.object(old_runtime,'OUT',OUT):super().__init__()
        def event(self,event,t):
            self.buffer.append(dict(event=event,endpoint=t,clock_seconds=time.perf_counter()))
    tick=time.perf_counter();encoder=DevelopmentEncoder()
    save(OUT/'model_setup.json',dict(seconds=time.perf_counter()-tick,excluded_from_inference=True,identity_sha256=sha(OUT/'encoder_identity.json')))
    return encoder


def live(raw,entry,encoder,policy,head,dense=False):
    from temporal_refresh_timing import working_set
    torch=encoder.torch;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    resident=torch.cuda.memory_allocated();cpu_before,_=working_set();starts=encoder.starts;completions=encoder.completions
    start=time.perf_counter()
    if not dense:policy=policy()  # per-video random/cache-policy initialization is inside the boundary
    frames,grid=decode(raw,entry);provider=Provider(frames,grid,encoder)
    if dense:
        values=[]
        for t in range(16):provider.advance(t);values.append(provider.refresh())
        z=affine(np.asarray(values).mean(0,dtype=np.float64),head)
        result=dict(logits=z.tolist(),prediction=int(z.argmax()),positions=list(range(16)),cache_indices=list(range(16)),trace=[],
                    cheap_statistics_seconds=0.,controller_decide_seconds=0.,observation_preparation_seconds=0.)
    else:result=causal_execute(provider,policy,head)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    result.update(seconds=seconds,encoder_starts=encoder.starts-starts,encoder_completions=encoder.completions-completions,
        decoded_frames=len(frames),RGB_conversions=provider.RGB_conversions,grayscale_conversions=provider.grayscale_conversions,
        cheap_thumbnail_seconds=provider.thumbnail_seconds,resident_cuda_bytes=resident,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        incremental_peak_cuda_bytes=max(0,torch.cuda.max_memory_allocated()-resident),process_working_set_before=cpu_before,
        process_working_set_after=working_set()[0])
    require(result['encoder_starts']==result['encoder_completions']==(16 if dense else 4),'Started/completed forward mismatch')
    actual=np.stack([provider.features[t] for t in result['positions']])
    replay=affine(np.stack([provider.features[t] for t in result['cache_indices']]).mean(0,dtype=np.float64),head)
    np.testing.assert_allclose(result['logits'],replay,rtol=1e-10,atol=1e-10)
    return result,actual


def policy_for(method,head,selection,clip):
    if method in ['history','history_motion']:
        item=selection[method];verify({item['path']:item['sha256']})
        model=read(ROOT/item['path']);return lambda:LivePolicy(method,head,model,item['threshold'],clip)
    threshold=selection.get('motion',{}).get('threshold',.5)
    return lambda:LivePolicy(method,head,threshold=threshold,clip=clip)


def trial(entry,method,phase,repetition,raw,encoder,head,selection):
    """Checkpoint actual attempts; all disk I/O stays outside the timed boundary."""
    entry_allowed(entry)
    if phase in ['execution','dense_diagnostic','timing']:seal_check()
    key=rank('trial',phase,entry['clip'],method,repetition)
    path=DATA/'trials'/f'{key}.json'
    identity=dict(clip=entry['clip'],input_sha256=entry['sha256'],method=method,phase=phase,repetition=repetition,
        code_sha256={n:sha(ROOT/n) for n in CODE},head_sha256=sha(ROOT/read(OUT/'recognition_head_reuse.json')['path']),
        selection_sha256=sha(OUT/'controller_selection.json') if (OUT/'controller_selection.json').exists() else None)
    if path.exists():
        r=checkpoint(path,identity,lambda:None);verify({r['features_path']:r['features_sha256']});return r
    encoder.phase=phase;encoder.clip=entry['clip'];encoder.buffer=[]
    event=dict(token=key,phase=phase,clip=entry['clip'],method=method,repetition=repetition,at_utc=now())
    with (OUT/'execution_attempts.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='start',**event))+'\n')
    try:
        policy=None if method=='dense16' else policy_for(method,head,selection,entry['clip'])
        result,features=live(raw,entry,encoder,policy,head,dense=method=='dense16')
        feature_path=DATA/'executed_features'/f'{key}.npz';feature_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(feature_path,features=features)
        result.update(clip=entry['clip'],label=entry['label'],group=entry['group'],study_role=entry['study_role'],method=method,phase=phase,
            repetition=repetition,warmup=phase=='timing' and repetition<0,
            features_path=rel(feature_path),features_sha256=sha(feature_path),token=key)
        for item in encoder.buffer:
            with (OUT/'encoder_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(**item,token=key,phase=phase,clip=entry['clip'],method=method))+'\n')
        with (OUT/'execution_attempts.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='complete',**event,starts=result['encoder_starts'],completions=result['encoder_completions']))+'\n')
        return checkpoint(path,identity,lambda:result)
    except Exception as e:
        save(OUT/f'failed_attempt_{key}.json',dict(**event,error=repr(e),actual_events=encoder.buffer))
        raise


def observations(manifest):
    ledger_path=OUT/'observation_provenance.json';ledger=read(ledger_path) if ledger_path.exists() else {}
    entries=[v for v in manifest['videos'] if v['study_role'] in ['controller_train','selection']]
    for i,v in enumerate(entries):
        if v['clip'] in ledger:verify({ledger[v['clip']]['path']:ledger[v['clip']]['sha256']});continue
        f,source=archived_features(v);frames,grid=decode(raw_video(v),v)
        thumbs=np.stack([frames[j].reformat(width=32,height=32,format='gray').to_ndarray().copy() for j in grid['endpoint_indices']])
        path=DATA/'observations'/f'{v["clip"]}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,features=f,thumbnails=thumbs)
        ledger[v['clip']]=dict(path=rel(path),sha256=sha(path),original_feature=source,original_input_sha256=v['sha256'],
            actual_decoded_frames=len(frames),grayscale_conversions=16,study_role=v['study_role'])
        save(ledger_path,ledger)
        if (i+1)%50==0:print('Genuine training/selection observations',i+1,'/',len(entries),flush=True)
    return ledger


def obs_features(v,ledger):
    entry_allowed(v);require(v['study_role'] in ['controller_train','selection'],'Execution dense data forbidden during training')
    if v['clip'] in _OBS_CACHE:return _OBS_CACHE[v['clip']]
    r=ledger[v['clip']];verify({r['path']:r['sha256']})
    with np.load(ROOT/r['path'],allow_pickle=False) as f:result=(f['features'],f['thumbnails'])
    _OBS_CACHE[v['clip']]=result;return result


_OBS_CACHE={}


def metrics(z,entries,classes):
    from ucf101_proxy_math import hierarchical_weights
    weights=hierarchical_weights(entries,classes);y=np.array([classes.index(v['label']) for v in entries]);z=np.asarray(z)
    return dict(CE=float(weights @ (logsumexp(z,axis=1)-z[np.arange(len(y)),y])),
                accuracy=float(weights @ (z.argmax(1)==y)))


def train(manifest,head):
    from causal_observability_learning import examples,train_network
    contract=dict(manifest_sha256=sha(OUT/'population_manifest.json'),protocol_sha256=sha(OUT/'protocol.md'),
        code_sha256={p:sha(ROOT/p) for p in CODE},head_sha256=sha(ROOT/read(OUT/'recognition_head_reuse.json')['path']))
    contract_path=OUT/'training_contract.json'
    if contract_path.exists():require(read(contract_path)==contract,'Training code/data/head contract changed')
    else:save(contract_path,contract)
    if (OUT/'training_identity.json').exists():verify(read(OUT/'training_identity.json')['files'])
    if (OUT/'controller_selection.json').exists():
        selection=read(OUT/'controller_selection.json')
        for method in ['history','history_motion']:verify({selection[method]['path']:selection[method]['sha256']})
        return selection
    ledger=observations(manifest);training=[v for v in manifest['videos'] if v['study_role']=='controller_train']
    development=[v for v in manifest['videos'] if v['study_role']=='selection'];base_path=DATA/'base_examples.json.gz'
    if base_path.exists():
        with gzip.open(base_path,'rt',encoding='utf-8') as f:base=json.load(f)
    else:
        base=[]
        for i,v in enumerate(training):
            f,thumbs=obs_features(v,ledger)
            policies=[Policy('periodic',head,clip=v['clip'])]+[Policy('RandomB4_seeded',head,clip=v['clip']+f'/rollout{j}') for j in range(3)]
            base.extend(examples(v,f,thumbs,head,policies))
            if (i+1)%50==0:print('Genuine training rollout sources',i+1,'/',len(training),flush=True)
        with gzip.open(base_path,'wt',encoding='utf-8') as f:json.dump(base,f)
    preliminary={}
    for method in ['history','history_motion']:
        path=OUT/f'controllers/preliminary_{method}.json'
        if path.exists():preliminary[method]=read(path)['model'];continue
        tick=time.perf_counter();models,log=train_network(base,method=='history_motion',(40,));preliminary[method]=models[40]
        save(path,dict(model=models[40],training=log,seconds=time.perf_counter()-tick))
        print('Trained fixed preliminary',method,flush=True)
    augmented_path=DATA/'augmented_examples.json.gz'
    if augmented_path.exists():
        with gzip.open(augmented_path,'rt',encoding='utf-8') as f:augmented=json.load(f)
    else:
        augmented=base.copy()
        for v in training:
            f,thumbs=obs_features(v,ledger)
            policies=[Policy(m,head,preliminary[m],.35,clip=v['clip']) for m in preliminary]
            augmented.extend(examples(v,f,thumbs,head,policies))
        with gzip.open(augmented_path,'wt',encoding='utf-8') as f:json.dump(augmented,f)
    save(OUT/'training_identity.json',dict(base_examples=len(base),augmented_examples=len(augmented),sources=len(training),
        source_groups=len({(v['label'],v['group']) for v in training}),
        files={rel(p):sha(p) for p in [base_path,augmented_path]},head_sha256=sha(ROOT/read(OUT/'recognition_head_reuse.json')['path']),
        common_sources_and_optimization=True,architecture='145 ->32 ReLU ->1 sigmoid; float64 Adam .003 decay .001 batch256;40+80 epochs',
        source='Genuine archived training features and fresh original AVI thumbnails; privileged conditional Z_pool teacher only offline'))
    import pandas as pd
    pd.DataFrame([dict(stage='base' if i<len(base) else 'on_policy_augmentation',age=r['age'],remaining=r['remaining']) for i,r in enumerate(augmented)]).groupby(['stage','age','remaining']).size().rename('states').to_csv(OUT/'training_state_distribution.csv')
    selection={};records=[]
    for method in ['history','history_motion']:
        final_path=OUT/f'controllers/{method}_training.json'
        if final_path.exists():
            log=read(final_path);models={epoch:read(OUT/f'controllers/{method}_epoch{epoch}.json') for epoch in [20,40,80]}
        else:
            tick=time.perf_counter();models,log=train_network(augmented,method=='history_motion');
            for epoch,model in models.items():save(OUT/f'controllers/{method}_epoch{epoch}.json',model)
            save(final_path,dict(log=log,seconds=time.perf_counter()-tick))
        candidates=[]
        for epoch,model in models.items():
            for threshold in [.2,.35,.5]:
                z=[]
                for v in development:
                    f,thumbs=obs_features(v,ledger);z.append(execute(TrainingReplayProvider(f,thumbs),Policy(method,head,model,threshold,v['clip']),head)['logits'])
                candidates.append(dict(epoch=epoch,threshold=threshold,grid_order=len(candidates),**metrics(z,development,manifest['classes'])))
        best=min(candidates,key=lambda x:(x['CE'],-x['accuracy'],x['grid_order']));p=OUT/f'controllers/{method}_epoch{best["epoch"]}.json'
        selection[method]=dict(**best,path=rel(p),sha256=sha(p));records.append(dict(method=method,candidates=candidates))
        print('Fixed development selection',method,best,flush=True)
    motion=[]
    for v in training:
        _,thumbs=obs_features(v,ledger);motion.extend(cheap_signal(thumbs[t],thumbs[t-1])[26] for t in range(1,16))
    candidates=[]
    for q in [.5,.75,.9]:
        threshold=float(np.quantile(motion,q));z=[]
        for v in development:
            f,thumbs=obs_features(v,ledger);z.append(execute(TrainingReplayProvider(f,thumbs),Policy('motion',head,threshold=threshold,clip=v['clip']),head)['logits'])
        candidates.append(dict(quantile=q,threshold=threshold,grid_order=len(candidates),**metrics(z,development,manifest['classes'])))
    selection['motion']=min(candidates,key=lambda x:(x['CE'],-x['accuracy'],x['grid_order']))
    a=read(ROOT/selection['history']['path']);b=read(ROOT/selection['history_motion']['path'])
    np.testing.assert_allclose(a['means'][:-32],b['means'][:-32],atol=0,rtol=0)
    save(OUT/'controller_training_selection.json',dict(learned=records,motion=candidates));save(OUT/'controller_selection.json',selection)
    return selection


def smoke(manifest,encoder,head):
    path=OUT/'real_training_smoke.json'
    if path.exists():return read(path)
    entries=sorted([v for v in manifest['videos'] if v['study_role']=='controller_train'],key=lambda v:rank('smoke',v['clip']))[:2]
    records=[]
    for v in entries:
        raw=raw_video(v);reference,_=archived_features(v)
        for method in ['periodic','dense16']:
            r=trial(v,method,'training_smoke',0,raw,encoder,head,{})
            with np.load(ROOT/r['features_path'],allow_pickle=False) as f:actual=f['features']
            np.testing.assert_allclose(actual,reference[r['positions']],atol=1e-6,rtol=1e-6)
            z=affine(reference[np.array(r['cache_indices'])].mean(0,dtype=float),head)
            np.testing.assert_allclose(r['logits'],z,atol=1e-6,rtol=1e-6)
            records.append(dict(clip=v['clip'],method=method,seconds=r['seconds'],actual_calls=r['encoder_completions'],
                maximum_feature_difference=float(np.max(abs(actual-reference[r['positions']]))),
                maximum_logit_difference=float(np.max(abs(np.asarray(r['logits'])-z)))))
    native=encoder.native_check()
    b4=np.mean([r['seconds'] for r in records if r['method']=='periodic']);dense=np.mean([r['seconds'] for r in records if r['method']=='dense16'])
    ne=sum(v['study_role']=='execution' for v in manifest['videos'])
    result=dict(at_utc=now(),real_training_only_clips=[v['clip'] for v in entries],records=records,native_projection_check=native,
        planned_quality_calls=ne*(5*4+16),planned_timing_calls=30*7*(6*4+16),
        projected_quality_seconds=ne*(5*b4+dense),projected_timing_seconds=30*7*(6*b4+dense),
        projected_local_bytes=2*2**30,free_disk_bytes=__import__('shutil').disk_usage(ROOT).free,
        projection_limit='Two real training sources only; controller/thumbnail costs and thermals uncertain; no coverage reduction based on projection')
    require(result['free_disk_bytes']>result['projected_local_bytes']*2,'Insufficient disk for real execution records')
    save(path,result);print('Actual training smoke / projection',json.dumps(result),flush=True);return result


def freeze(manifest):
    path=OUT/'development_execution_seal.json'
    if path.exists():return seal_check()
    files=CODE+['results/ucf101_causal_development_v1/'+n for n in ['protocol.md','population_manifest.json','timing_manifest.json',
        'recognition_head_reuse.json','controller_selection.json','controller_training_selection.json','training_contract.json','training_identity.json','encoder_identity.json']]
    selection=read(OUT/'controller_selection.json')
    files += [selection[m]['path'] for m in ['history','history_motion']]
    files += [read(OUT/'recognition_head_reuse.json')['path']]
    save(path,dict(frozen_at_utc=now(),sha256={p:sha(ROOT/p) for p in files},
        scope='DEVELOPMENT/EXPLORATORY control freeze, NOT an independent official-test seal',
        official_test_access_authorized=False,statistics='class/group/clip;10000 paired class-stratified group bootstrap seed20261019'))
    return seal_check()


def run_quality(manifest,encoder,head,selection):
    path=OUT/'executed_development_rollouts.json';ledger=read(path) if path.exists() else dict(records={},all_deployable_complete=False)
    entries=[v for v in manifest['videos'] if v['study_role']=='execution']
    if not (OUT/'development_execution_exposure.json').exists():
        save(OUT/'development_execution_exposure.json',dict(at_utc=now(),seal_sha256=sha(OUT/'development_execution_seal.json'),
            scope='First actual causal DEVELOPMENT rollout; data already used for historical recognition-head selection; no official-test access'))
    for i,v in enumerate(entries):
        tick=time.perf_counter();raw=raw_video(v);load_seconds=time.perf_counter()-tick
        for method in METHODS:
            r=trial(v,method,'execution',0,raw,encoder,head,selection)
            ledger['records'][v['clip']+'/'+method]=dict(path=rel(DATA/'trials'/f'{r["token"]}.json'),
                sha256=sha(DATA/'trials'/f'{r["token"]}.json'),input_loading_seconds_excluded=load_seconds)
        save(path,ledger)
        if (i+1)%10==0:print('Real causal DEVELOPMENT execution',i+1,'/',len(entries),'five B4 methods;20 genuine forwards/clip',flush=True)
    ledger['all_deployable_complete']=True;save(path,ledger)
    return ledger


def load_trial(record):
    verify({record['path']:record['sha256']});value=read(ROOT/record['path'])
    require(digest(value['payload'])==value['payload_sha256'],'Actual execution payload mutation')
    r=value['payload'];verify({r['features_path']:r['features_sha256']});return r


def privileged_diagnostics(manifest,encoder,head,selection):
    rollout=read(OUT/'executed_development_rollouts.json');require(rollout['all_deployable_complete'],'Dense diagnostics require ALL real causal rollouts first')
    GUARD.set_population(manifest,diagnostics=True)
    path=OUT/'dense_development_diagnostics.json';ledger=read(path) if path.exists() else {}
    entries=[v for v in manifest['videos'] if v['study_role']=='execution']
    for i,v in enumerate(entries):
        r=trial(v,'dense16','dense_diagnostic',0,raw_video(v),encoder,head,selection)
        original,_=archived_features(v)
        with np.load(ROOT/r['features_path'],allow_pickle=False) as f:actual=f['features']
        np.testing.assert_allclose(actual,original,atol=1e-6,rtol=1e-6)
        z=affine(actual.mean(0,dtype=float),head)
        np.testing.assert_allclose(z,r['logits'],atol=1e-10,rtol=1e-10)
        for method in METHODS:
            q=load_trial(rollout['records'][v['clip']+'/'+method])
            with np.load(ROOT/q['features_path'],allow_pickle=False) as f:chosen=f['features']
            np.testing.assert_allclose(chosen,actual[q['positions']],atol=1e-6,rtol=1e-6)
            replay=affine(actual[np.array(q['cache_indices'])].mean(0,dtype=float),head)
            np.testing.assert_allclose(q['logits'],replay,atol=1e-6,rtol=1e-6)
        ledger[v['clip']]=dict(path=rel(DATA/'trials'/f'{r["token"]}.json'),sha256=sha(DATA/'trials'/f'{r["token"]}.json'),
            maximum_saved_feature_difference=float(np.max(abs(actual-original))),native_projection=encoder.native_check())
        save(path,ledger)
        if (i+1)%20==0:print('Actual DEVELOPMENT Dense16 diagnostic',i+1,'/',len(entries),flush=True)
    return ledger


def timing(manifest,encoder,head,selection):
    entries={v['clip']:v for v in manifest['videos'] if v['study_role']=='execution'}
    cohort=read(OUT/'timing_manifest.json');path=OUT/'timing_records.json';ledger=read(path) if path.exists() else {}
    quality=read(OUT/'executed_development_rollouts.json')['records'];dense=read(OUT/'dense_development_diagnostics.json')
    for i,clip in enumerate(cohort['clips']):
        v=entries[clip];tick=time.perf_counter();raw=raw_video(v);load_seconds=time.perf_counter()-tick
        rng=np.random.Generator(np.random.PCG64(int(rank('timing_order',clip)[:16],16)))
        for repetition in range(-2,5):
            for method in rng.permutation(TIMING_METHODS):
                method=str(method);r=trial(v,method,'timing',repetition,raw,encoder,head,selection)
                reference=load_trial(dense[clip] if method=='dense16' else quality[clip+'/'+('periodic' if method=='periodic_unused_gate' else method)])
                discrepancy=float(np.max(abs(np.asarray(r['logits'])-reference['logits'])))
                np.testing.assert_allclose(r['logits'],reference['logits'],atol=1e-6,rtol=1e-6)
                ledger[r['token']]=dict(path=rel(DATA/'trials'/f'{r["token"]}.json'),sha256=sha(DATA/'trials'/f'{r["token"]}.json'),
                    reference_logit_max_difference=discrepancy,input_loading_seconds_excluded=load_seconds)
                save(path,ledger)
        print('Real matched DEVELOPMENT timing',i+1,'/30 groups;2 warmups+5 measured,7 modes',flush=True)
    import pandas as pd
    frame=pd.DataFrame([{k:x for k,x in load_trial(r).items() if k not in ['trace']}|dict(reference_logit_max_difference=r['reference_logit_max_difference']) for r in ledger.values()])
    frame.to_csv(OUT/'timing_trials.csv',index=False,float_format='%.17g')
    measured=frame[~frame['warmup']]
    columns=['seconds','cheap_thumbnail_seconds','cheap_statistics_seconds','controller_decide_seconds','observation_preparation_seconds','peak_cuda_bytes','incremental_peak_cuda_bytes']
    group=measured.groupby(['clip','label','group','method'])
    med=group[columns].median().rename(columns={c:'median_'+c for c in columns}).reset_index()
    variation=group['seconds'].agg(['min','max','std']).reset_index();med=med.merge(variation,on=['clip','label','group','method'])
    med.to_csv(OUT/'timing_per_video.csv',index=False,float_format='%.17g')
    pivot=med.pivot(index=['clip','label','group'],columns='method',values='median_seconds')
    ratios=[]
    for method in TIMING_METHODS:
        values=pivot['periodic']/pivot[method]
        ratios.append(dict(method=method,groups=len(values),median_seconds=float(pivot[method].median()),
            median_paired_periodic_over_method=float(values.median()),min_paired_ratio=float(values.min()),max_paired_ratio=float(values.max())))
    save(OUT/'timing_summary.json',dict(scope='REAL DEVELOPMENT timing; one median per video/group; repeated trials are not independent',methods=ratios,
        boundary='encoded-memory -> fresh full decode/PTS -> all cheap observations/decisions -> selected RGB/transfers -> native X3D -> pooling/head/prediction -> CUDA-completed output',
        trials=len(frame),measured_trials=len(measured),warmups=2,repetitions=5,groups=30,CPU_threads=1,batch=1,encoder_dtype='FP32',
        encoder_event_logging='in-memory real start/completion events, durable flush outside boundary',offline_Z_pool_latency_assigned=False))


def score(z,y,dense):
    z=np.asarray(z);d=np.asarray(dense);lp=z-logsumexp(z);ld=d-logsumexp(d)
    return dict(CE=float(-lp[y]),correct=int(z.argmax()==y),dense_agreement=int(z.argmax()==d.argmax()),
                KL_dense=float(np.exp(ld) @ (ld-lp)),prediction=int(z.argmax()))


def analysis(manifest,head):
    import pandas as pd
    import refresh_pooling_utils as pool
    from causal_observability_learning import teacher
    from scipy.stats import spearmanr
    entries=[v for v in manifest['videos'] if v['study_role']=='execution'];classes=manifest['classes']
    rollouts=read(OUT/'executed_development_rollouts.json')['records'];denses=read(OUT/'dense_development_diagnostics.json')
    rows=[];decisions=[];traces=[];prediction_rows=[];live_states=[]
    table=pool.schedules(16,4)
    training=read(OUT/'training_identity.json');selection=read(OUT/'controller_selection.json')
    model=read(ROOT/selection['history_motion']['path'])
    for v in entries:
        dense=load_trial(denses[v['clip']]);y=classes.index(v['label'])
        with np.load(ROOT/dense['features_path'],allow_pickle=False) as f:features=f['features']
        candidate_z=affine(features,head);centered=candidate_z-candidate_z.mean(1,keepdims=True)
        objective=np.square(table['cache_weights'] @ centered-centered.mean(0)).mean(1)
        oracle_index=pool.first_tie(objective)['index'];oracle_positions=table['positions'][oracle_index].tolist()
        zpool=table['cache_weights'][oracle_index] @ candidate_z
        methods={m:load_trial(rollouts[v['clip']+'/'+m]) for m in METHODS}
        methods['dense16']=dense
        for method,r in methods.items():
            rows.append(dict(clip=v['clip'],label=v['label'],group=v['group'],method=method,**score(r['logits'],y,dense['logits']),
                encoder_starts=r['encoder_starts'],encoder_completions=r['encoder_completions'],seconds=r['seconds'],
                positions=json.dumps(r['positions']),cost_source='actual single execution; matched timing reported separately'))
            prediction_rows.append(dict(clip=v['clip'],method=method,logits=r['logits'],positions=r['positions']))
            traces.extend(dict(clip=v['clip'],label=v['label'],group=v['group'],method=method,**t) for t in r['trace'])
            for t in r['trace']:
                i=t['t']
                if not t['available_refreshes']:continue
                last=t['available_refreshes'][-1]
                loss=lambda z:float(logsumexp(z)-z[y])
                gain=loss(candidate_z[last])-loss(candidate_z[i])
                discretionary=not t['forced'] and t['remaining_before']>0
                target=teacher(features,head,r['trace'],i,objective) if discretionary else None
                decisions.append(dict(clip=v['clip'],label=v['label'],group=v['group'],method=method,t=i,age=t['age'],remaining=t['remaining_before'],
                    selected=t['selected'],forced=t['forced'],discretionary=discretionary,teacher_refresh=target,
                    teacher_agreement=int(t['selected']==target) if discretionary else None,
                    local_CE_gain=gain,helpful=gain>1e-12,harmful=gain< -1e-12,
                    missed_helpful=discretionary and not t['selected'] and gain>1e-12,
                    chosen_harmful=discretionary and t['selected'] and gain< -1e-12,
                    teacher_positive_task_harmful=discretionary and target==1 and gain< -1e-12,
                    current_MAD=t['cheap_signal'][26] if t['cheap_signal'] else None,imitation_probability=t['imitation_probability']))
                if method=='history_motion' and discretionary:
                    history=tuple(tuple(features[j]) for j in t['available_refreshes'])
                    obs=Observation(i,t['age'],t['remaining_before'],history,tuple(t['cheap_signal']))
                    x=(vector(obs,head,True)-model['means'])/np.asarray(model['scales'])
                    live_states.append(dict(clip=v['clip'],age=t['age'],remaining=t['remaining_before'],standardized_abs_mean=float(np.mean(abs(x))),
                        fraction_abs_gt3=float(np.mean(abs(x)>3))))
        rows.append(dict(clip=v['clip'],label=v['label'],group=v['group'],method='Z_pool_OFFLINE_privileged',**score(zpool,y,dense['logits']),
            encoder_starts=16,encoder_completions=16,seconds=None,positions=json.dumps(oracle_positions),cost_source='privileged real dense16 features plus unassigned CPU search; NOT four-call execution'))
        prediction_rows.append(dict(clip=v['clip'],method='Z_pool_OFFLINE_privileged',logits=zpool.tolist(),positions=oracle_positions))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'development_per_clip_metrics.csv',index=False,float_format='%.17g')
    pd.DataFrame(decisions).to_csv(OUT/'retrospective_decisions.csv',index=False,float_format='%.17g')
    with gzip.open(OUT/'development_predictions.json.gz','wt',encoding='utf-8') as f:json.dump(prediction_rows,f)
    with gzip.open(OUT/'development_decision_traces.json.gz','wt',encoding='utf-8') as f:json.dump(traces,f)
    pd.DataFrame(live_states).to_csv(OUT/'live_state_distribution.csv',index=False,float_format='%.17g')
    metric_cols=['CE','correct','dense_agreement','KL_dense']
    grouped=frame.groupby(['label','group','method'])[metric_cols].mean().reset_index()
    grouped.to_csv(OUT/'development_per_group_metrics.csv',index=False,float_format='%.17g')
    perclass=grouped.groupby(['label','method'])[metric_cols].mean().reset_index();perclass.to_csv(OUT/'development_per_class_metrics.csv',index=False,float_format='%.17g')
    means=perclass.groupby('method')[metric_cols].mean().reset_index();means['ordinary_clip_accuracy']=[frame[frame['method']==m]['correct'].mean() for m in means['method']]
    means.to_csv(OUT/'development_method_metrics.csv',index=False,float_format='%.17g')
    # Fixed-seed paired GROUP resampling; classes with one group cannot estimate between-group variance.
    groups=grouped[['label','group']].drop_duplicates().sort_values(['label','group']).reset_index(drop=True)
    rng=np.random.Generator(np.random.PCG64(SEED));blocks=[];weights=[];singletons=[]
    for c in classes:
        ids=np.flatnonzero(groups['label'].to_numpy()==c);blocks.append(ids[rng.integers(len(ids),size=(10000,len(ids)))]);weights.extend([1/(len(classes)*len(ids))]*len(ids))
        if len(ids)==1:singletons.append(c)
    indices=np.concatenate(blocks,axis=1).astype(np.int32);weights=np.array(weights)
    np.save(OUT/'development_bootstrap_group_indices.npy',indices,allow_pickle=False)
    save(OUT/'development_bootstrap_groups.json',dict(groups=groups.to_dict('records'),weights=weights.tolist(),seed=SEED,repeats=10000,single_group_classes=singletons,
        scope='Exploratory marginal intervals conditional on reused selected head/controllers; not independent confirmation'))
    pivot=grouped.pivot(index=['label','group'],columns='method',values='CE').loc[list(map(tuple,groups.to_numpy()))]
    contrasts={**{f'periodic_minus_{m}':pivot['periodic']-pivot[m] for m in ['RandomB4_seeded','motion','history','history_motion','dense16','Z_pool_OFFLINE_privileged']},
               'history_minus_history_motion':pivot['history']-pivot['history_motion'],
               'observability_gap_HplusM_minus_Z_pool':pivot['history_motion']-pivot['Z_pool_OFFLINE_privileged']}
    intervals=[]
    for name,values in contrasts.items():
        values=values.to_numpy();draws=(values[indices]*weights).sum(1);ci=np.percentile(draws,[2.5,97.5])
        intervals.append(dict(contrast=name,mean=float(values @ weights),CI95_lower=float(ci[0]),CI95_upper=float(ci[1]),scope='DEVELOPMENT/EXPLORATORY marginal'))
    pd.DataFrame(intervals).to_csv(OUT/'development_CE_contrasts.csv',index=False,float_format='%.17g')
    # Accuracy intervals use exactly the same paired group draws, including method marginals.
    accuracy=[]
    for method in frame['method'].unique():
        vals=grouped[grouped['method']==method].set_index(['label','group']).loc[list(map(tuple,groups.to_numpy())),'correct'].to_numpy()
        draws=(vals[indices]*weights).sum(1);ci=np.percentile(draws,[2.5,97.5])
        accuracy.append(dict(method=method,mean=float(vals @ weights),CI95_lower=float(ci[0]),CI95_upper=float(ci[1])))
    pd.DataFrame(accuracy).to_csv(OUT/'development_accuracy_intervals.csv',index=False,float_format='%.17g')
    decisions=pd.DataFrame(decisions);failure=[]
    for method in METHODS:
        d=decisions[(decisions['method']==method)&decisions['discretionary']]
        joined=frame[frame['method']==method].set_index('clip')['CE']-frame[frame['method']=='periodic'].set_index('clip')['CE']
        failure.append(dict(method=method,clips_higher_CE_than_periodic=int((joined>1e-12).sum()),discretionary_decisions=len(d),
            teacher_agreement=float(d['teacher_agreement'].mean()),chosen_harmful=int(d['chosen_harmful'].sum()),missed_helpful=int(d['missed_helpful'].sum()),
            teacher_positive_task_harmful=int(d['teacher_positive_task_harmful'].sum()),
            MAD_local_task_gain_spearman=float(spearmanr(d['current_MAD'],d['local_CE_gain'],nan_policy='omit').statistic)
                if d['current_MAD'].notna().sum()>2 and d['current_MAD'].dropna().nunique()>1 else None))
    with gzip.open(DATA/'augmented_examples.json.gz','rt',encoding='utf-8') as f:augmented=json.load(f)
    source_distributions={};state_counts={}
    for name,subset in [('base_training',augmented[:training['base_examples']]),('on_policy_training',augmented[training['base_examples']:])]:
        a=np.array([r['input'] for r in subset]);scaled=(a-model['means'])/np.asarray(model['scales'])
        source_distributions[name]=dict(states=len(subset),standardized_abs_mean=float(abs(scaled).mean()),fraction_abs_gt3=float(np.mean(abs(scaled)>3)))
        count=Counter((r['age'],r['remaining']) for r in subset);state_counts[name]={k:v/len(subset) for k,v in count.items()}
    live_count=Counter((r['age'],r['remaining']) for r in live_states);live_distribution={k:v/len(live_states) for k,v in live_count.items()}
    for name,dist in state_counts.items():source_distributions[name]['age_budget_total_variation_to_actual_HplusM']=sum(abs(dist.get(k,0)-live_distribution.get(k,0)) for k in set(dist)|set(live_distribution))/2
    source_distributions['actual_live_HplusM']=dict(states=len(live_states),**pd.DataFrame(live_states)[['standardized_abs_mean','fraction_abs_gt3']].mean().to_dict())
    save(OUT/'failure_analysis.json',dict(methods=failure,scope='Retrospective real development diagnostics; local gains are not cumulative causal effects',
        imitation_probability='Conditional proxy-teacher imitation score, NOT probability of beneficial task update',training=training,
        state_shift=source_distributions))
    return means,intervals


def accounting():
    events=[json.loads(x) for x in (OUT/'encoder_events.jsonl').read_text().splitlines()] if (OUT/'encoder_events.jsonl').exists() else []
    phases=Counter((r['phase'],r['event']) for r in events)
    failures=[]
    for p in OUT.glob('failed_attempt_*.json'):
        r=read(p);failures.append(dict(path=rel(p),recorded_start_events=sum(v['event']=='start' for v in r['actual_events']),
                                      recorded_complete_events=sum(v['event']=='complete' for v in r['actual_events'])))
    attempts=[json.loads(x) for x in (OUT/'execution_attempts.jsonl').read_text().splitlines()] if (OUT/'execution_attempts.jsonl').exists() else []
    starts={r['token'] for r in attempts if r['event']=='start'};completed={r['token'] for r in attempts if r['event']=='complete'}
    return dict(actual_encoder_starts=sum(r['event']=='start' for r in events)+sum(f['recorded_start_events'] for f in failures),
        actual_encoder_completions=sum(r['event']=='complete' for r in events)+sum(f['recorded_complete_events'] for f in failures),
        phases=[dict(phase=k[0],event=k[1],count=n) for k,n in sorted(phases.items())],failed_attempts=failures,
        unresolved_attempt_tokens=sorted(starts-completed),official_test_calls=0,
        official_test_accounting='Enforced path/population guards; every actual encoder event names an approved official-training clip',
        new_dataset_network_bytes=0,installed_packages=[],scope='Real observed calls, including recorded failed attempts; no inferred timing')


def figures():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    q=pd.read_csv(OUT/'development_method_metrics.csv');cost=read(OUT/'timing_summary.json')
    timing={r['method']:r for r in cost['methods']};modes=METHODS+['dense16','Z_pool_OFFLINE_privileged']
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    ce=q.set_index('method')['CE'];ax[0].barh(modes,[ce[m] for m in modes],color=['#9d9d9d','#9d9d9d','#c19441','#4f88a3','#32846c','#6a73a2','#b383a3'])
    ax[0].invert_yaxis();ax[0].set_xlabel('Class / group / clip balanced CE, nats');ax[0].set_title('Real DEVELOPMENT quality; oracle privileged')
    for m in METHODS+['dense16']:
        ax[1].scatter(timing[m]['median_seconds']*1000,ce[m]);ax[1].annotate(m,(timing[m]['median_seconds']*1000,ce[m]),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax[1].set(xlabel='Median per-video measured latency, ms',ylabel='Balanced development CE, nats',title='Actual execution quality / cost;30 timing groups')
    fig.suptitle('DEVELOPMENT / EXPLORATORY — reused head, no independent test confirmation',fontsize=12)
    fig.savefig(folder/'development_quality_cost.png',dpi=180);fig.savefig(folder/'development_quality_cost.pdf');plt.close(fig)
    failure=read(OUT/'failure_analysis.json')['methods'];f=pd.DataFrame(failure).set_index('method').loc[METHODS]
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    f[['chosen_harmful','missed_helpful']].plot.bar(ax=ax[0],rot=20);ax[0].set(ylabel='Retrospective discretionary decisions',title='Local task gain diagnostics; not causal treatment effects',xlabel='')
    ax[1].bar(METHODS,f['teacher_agreement']);ax[1].tick_params(axis='x',rotation=20);ax[1].set(ylabel='Actual teacher agreement fraction',ylim=(0,1),title='Imitating Z_pool does not guarantee lower task CE')
    fig.suptitle('DEVELOPMENT failure analysis — all actual rollouts',fontsize=12)
    fig.savefig(folder/'development_failure_analysis.png',dpi=180);fig.savefig(folder/'development_failure_analysis.pdf');plt.close(fig)


def report(manifest):
    import pandas as pd
    q=pd.read_csv(OUT/'development_method_metrics.csv').set_index('method');contrasts=pd.read_csv(OUT/'development_CE_contrasts.csv')
    cost={r['method']:r for r in read(OUT/'timing_summary.json')['methods']};calls=accounting()
    primary=contrasts[contrasts['contrast']=='periodic_minus_history_motion'].iloc[0]
    failure=read(OUT/'failure_analysis.json');h=next(r for r in failure['methods'] if r['method']=='history_motion')
    point='H+M has lower mean development CE than periodic.' if primary['mean']>0 else 'H+M does not lower mean development CE relative to periodic.'
    uncertainty='The conditional descriptive interval crosses zero; inconclusive even within this reused development setting.' if primary['CI95_lower']<=0<=primary['CI95_upper'] else 'The conditional descriptive interval excludes zero, but prior development exposure and selection prevent independent confirmation.'
    lines=['# UCF101 causal execution: DEVELOPMENT / EXPLORATORY','',point+' '+uncertainty,'',
        'This is actual execution engineering and exploratory decision utility. It is not an independent evaluation, a TMLR-ready result, or proof that imitating an oracle improves recognition. No official-test outcomes, arrays or videos were opened.', '',
        f"Main base: {read(OUT/'repository_review.json')['actual_main']}. Actual achieved population: {json.dumps(manifest['counts'])}.", '',
        'The frozen101-class head was fitted on original training data and its alpha selected using all original development clips. Current controller training overlaps that head’s fitting data; both current development selection and execution have prior head-selection exposure. Controller selection groups and current execution groups are disjoint, but this does not make execution independent confirmation. The original head reached its LBFGS iteration limit; it was not refitted.', '',
        '|Method|Balanced CE|Balanced top1|Ordinary clip accuracy|Dense agreement|Median measured latency, ms|', '|---|---:|---:|---:|---:|---:|']
    for method in METHODS+['dense16','Z_pool_OFFLINE_privileged']:
        r=q.loc[method];latency=f"{cost[method]['median_seconds']*1000:.3f}" if method in cost else 'not assigned'
        lines.append(f"|{method}|{r['CE']:.6f}|{100*r['correct']:.2f}%|{100*r['ordinary_clip_accuracy']:.2f}%|{100*r['dense_agreement']:.2f}%|{latency}|")
    lines += ['', f"Fixed exploratory Delta_CE=CE(periodic)-CE(H+M): {primary['mean']:+.6f} nats, conditional descriptive95% group interval [{primary['CI95_lower']:+.6f},{primary['CI95_upper']:+.6f}]. Positive favors H+M. The saved class-stratified paired group draws condition on the reused selected head/controllers; single-group classes have no estimated within-class group variance, and selection/prior-exposure uncertainty is omitted.", '',
        f"Recorded actual new encoder starts/completions: {calls['actual_encoder_starts']} / {calls['actual_encoder_completions']}. Every completed deployable trial executes exactly4 forwards and16 cached outputs; Dense16 executes16. Failed attempts and unresolved tokens are disclosed in execution_accounting.json. Native400 projection executes. C pooling uses reuse counts/16.", '',
        'Timing includes fresh full decoding/actual-PTS planning, all encountered thumbnail/statistics/decision work, selected RGB/transfers, real X3D/native projection, pooling/head and CUDA-completed prediction. Disk loading/model setup are excluded. Thirty input-selected groups, two warmups and five interleaved measured repetitions per mode; repeats are not independent sources. Per-video variation and overhead/memory columns are retained. Offline Z_pool has genuine dense16 quality and no borrowed periodic latency.', '',
        f"H+M has higher CE than periodic on {h['clips_higher_CE_than_periodic']} execution clips. Actual discretionary teacher agreement is {h['teacher_agreement']:.4f}; chosen locally harmful refreshes {h['chosen_harmful']}, missed locally helpful opportunities {h['missed_helpful']}, teacher-positive but task-harmful opportunities {h['teacher_positive_task_harmful']}. MAD/local-gain Spearman: {h['MAD_local_task_gain_spearman']}. These local signed gains explain specific timing mistakes, not total causal treatment effects. The learner predicts proxy imitation, not task-benefit probability. See failure_analysis.json and retrospective_decisions.csv.", '',
        'Training examples came from genuine archived training features and freshly decoded thumbnails, periodic/random chronological histories, and one common on-policy augmentation round. H and H+M share architecture, data and optimization; only32 cheap dimensions differ. Training/live age/budget and standardized-state distributions are saved to diagnose shift, not to tune after execution.', '',
        '## Claim and evidence','', '|Claim|Evidence and limit|','|---|---|',
        '|Controllers trained on real observations|observation_provenance.json, training_identity.json, hashed real examples and checkpoints; head/controller training overlap disclosed|',
        '|Actually causal four-call execution|per-trial checkpoints, encoder_events.jsonl, decision traces, independent cached-logit replay; no complete future embedding matrix in live decisions|',
        '|Quality/cost|development metrics plus actual timing trials; exploratory reused development population only|',
        '|Failure mechanisms|Retrospective actual local gains, teacher disagreement, cheap-signal associations and state shift; descriptive, not novel causal mechanism|',
        '|Fresh-dataset readiness|Engineering contracts/integration can pass; fresh-source validity and independently frozen head/controller evaluation remain unestablished|',
        '|Historical preservation and forbidden test access|repository_review.json Git blob identities, explicit I/O guards, verification.json; old test outcomes not reread|', '',
        '## Limits and next experiment','',
        'Preserve this result regardless of direction. Four snippets may miss discriminative moments, cheap appearance change may not track action/task gain, teacher imitation may optimize the wrong target, histories can shift, and controller/thumbnail overhead can consume savings. Frozen-feature recognition, small group counts/class, training overlap, prior UCF development exposure, incomplete parent/pretraining mappings, and full-clip offline planning prevent publication-confirmation claims. No additional architecture or favorable subset is searched.', '',
        'Before a fresh evaluation, obtain authorized RGB data without accepting agreements automatically; verify original subject/source/take/camera family independence, all-class/group minima, actual temporal eligibility and duplicate/pretraining relationships; establish compute/storage using training-only real smoke. Review and freeze independently separated head/controller/selection roles, all model hashes and inference/statistical rules before any new evaluation predictions. Do not relabel UCF development as untouched test or reuse exposed official UCF test outcomes.', '',
        'The framework is an executable development prototype if verification passes. Whether it is suitable for a fresh scientific comparison depends on this study’s actual quality/cost and the new dataset gate. No publication-readiness claim follows from passing tests.']
    (OUT/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'README.md').write_text('# DEVELOPMENT / EXPLORATORY reproduction\n\nRun in PowerShell from repository root. Existing authorized UCF train/development bytes, genuine arrays, frozen head, pinned PyTorchVideo/checkpoint and .venv_temporal are required. No downloads or package upgrades.\n\n```powershell\n'
        r'& .\.venv_temporal\Scripts\python.exe -B -m unittest test_ucf101_causal_development -v'+'\n'+
        r'& .\.venv_temporal\Scripts\python.exe -B .\run_ucf101_causal_development.py --resume'+'\n'+
        r'& .\.venv_temporal\Scripts\python.exe -B .\run_ucf101_causal_development.py --audit-only'+'\n```\n\n'
        'Resume reuses identical per-trial checkpoints; failed/incomplete attempts remain. Completed resume performs a read-only audit. Audit never decodes, trains, initializes CUDA, retimes, downloads or regenerates evidence. It verifies approved source identities, policy/feature/checkpoint hashes, actual call counts, scalar logit replay, group weighting/bootstrap, control freeze and historical Git identities. Official-test outcome paths are prohibited.\n\n'
        'Large genuine observations/training examples/executed feature arrays stay under ignored data/source/ucf101_causal_development_v1 with public SHA inventories and original feature provenance. No vectors are reconstructed from aggregates. No video or backbone weights are redistributed. See protocol.md, findings.md, failure_analysis.md, fresh_dataset_gate.md, timing trials and verification.json.\n',encoding='utf-8')
    (OUT/'failure_analysis.md').write_text('# Real development failure analysis\n\nDEVELOPMENT / EXPLORATORY. Read failure_analysis.json for full actual counts and retrospective_decisions.csv for every decision.\n\n'
        f"H+M versus periodic: {h['clips_higher_CE_than_periodic']} higher-CE clips; teacher agreement {h['teacher_agreement']:.6f}; chosen harmful {h['chosen_harmful']}; missed helpful {h['missed_helpful']}; teacher-positive/task-harmful {h['teacher_positive_task_harmful']}; MAD/local-gain Spearman {h['MAD_local_task_gain_spearman']}.\n\n"
        'Incorrect timing is evaluated using actual earlier cache versus current real dense feature. Cheap changes are compared with task gain only after decisions. Teacher disagreement is conditional on the actual prefix, and proxy-positive/task-harmful cases expose objective mismatch. The sigmoid output imitates a proxy action, not beneficial-task probability. Base/on-policy/live state distributions and feature-standardization excursions expose distribution shift. All local diagnostics omit downstream counterfactual policy changes and therefore are not causal treatment-effect estimates.\n\n'
        'Controller overhead, grayscale conversion, cheap-statistics work, observation construction and peak CUDA allocations are measured in actual timing_trials.csv and timing_per_video.csv. Periodic+unused-gate isolates gating work without changing the refresh schedule. Only per-video medians and variation support execution-cost comparisons; no simulated or replay-derived latency is used. No new architecture or post-outcome threshold change is permitted.\n',encoding='utf-8')


def historical_integrity():
    review=read(OUT/'repository_review.json');verify(review['historical_code'])
    original={r.split('\t',1)[1]:r.split('\t',1)[0] for r in review['historical_result_index']}
    current={r.split('\t',1)[1]:r.split('\t',1)[0] for r in git('ls-files','-s','results').splitlines()}
    require(all(current.get(p)==blob for p,blob in original.items()),'Historical result index changed')
    modified=[p for p in git('diff','--name-only','HEAD','--','results').splitlines() if p in original]
    require(not modified,'Historical scientific outputs modified')
    return dict(preserved_historical_Git_blobs=len(original),old_frozen_code_unchanged=True,
        verification='Git index identities and clean historical working-tree paths; official-test scientific contents NOT opened')


def audit(final=True):
    """Read-only CPU verification; no decoder, models, training or timing."""
    import pandas as pd
    manifest=read(OUT/'population_manifest.json');validate_population(manifest);GUARD.set_population(manifest,diagnostics=True)
    if final:verify(read(OUT/'SHA256SUMS.json'))
    seal_check();exposure=read(OUT/'development_execution_exposure.json')
    require(exposure['seal_sha256']==sha(OUT/'development_execution_seal.json'),'Development exposure mismatch')
    head_info=read(OUT/'recognition_head_reuse.json');verify({head_info['path']:head_info['sha256']})
    head={k:np.asarray(v,dtype=float) for k,v in read(ROOT/head_info['path'])['parameters'].items()}
    verify(read(OUT/'training_identity.json')['files'])
    for item in read(OUT/'observation_provenance.json').values():
        verify({item['path']:item['sha256'],item['original_feature']['path']:item['original_feature']['sha256']})
    events=[json.loads(x) for x in (OUT/'encoder_events.jsonl').read_text().splitlines()];by_token=defaultdict(Counter)
    for e in events:
        require(e['clip'] in GUARD.entries,'Unapproved encoder source');by_token[e['token']][e['event']]+=1
    references={};all_trials={}
    for name in ['executed_development_rollouts.json','dense_development_diagnostics.json','timing_records.json']:
        ledger=read(OUT/name)
        if name=='executed_development_rollouts.json':require(ledger['all_deployable_complete'],'Causal execution incomplete');ledger=ledger['records']
        for record in ledger.values():
            r=load_trial(record);expected=16 if r['method']=='dense16' else 4
            require(r['encoder_starts']==r['encoder_completions']==expected,'Real forward budget mismatch')
            require(by_token[r['token']]['start']==by_token[r['token']]['complete']==expected,'Actual event/checkpoint count mismatch')
            require(r['positions'][0]==0 and len(r['positions'])==expected,'First refresh/budget invalid')
            with np.load(ROOT/r['features_path'],allow_pickle=False) as f:features=f['features']
            lookup={t:features[j] for j,t in enumerate(r['positions'])};last=0;cached=[]
            for t in range(16):
                if t in lookup:last=t
                cached.append(lookup[last])
            expected_z=np.asarray(head['weights']) @ ((np.mean(cached,axis=0,dtype=float)-head['means'])/head['scales'])+head['bias']
            np.testing.assert_allclose(r['logits'],expected_z,atol=1e-10,rtol=1e-10)
            if r['method']=='history':require(r['grayscale_conversions']==0 and all(not t['cheap_signal'] for t in r['trace']),'History-only accessed current thumbnails')
            for t in r['trace']:require(all(j<t['t'] for j in t['available_refreshes']),'Future/current feature leaked into decision')
            all_trials[r['token']]=r
            if r['phase'] in ['execution','dense_diagnostic']:references[(r['clip'],r['method'])]=r
    entries=[v for v in manifest['videos'] if v['study_role']=='execution']
    require(len(references)==len(entries)*6,'Missing actual policy/dense execution')
    frame=pd.read_csv(OUT/'development_per_clip_metrics.csv',float_precision='round_trip')
    for v in entries:
        for method in METHODS+['dense16']:
            r=references[(v['clip'],method)];row=frame[(frame['clip']==v['clip'])&(frame['method']==method)].iloc[0]
            z=np.asarray(r['logits']);ce=float(logsumexp(z)-z[manifest['classes'].index(v['label'])])
            np.testing.assert_allclose(row['CE'],ce,atol=1e-12,rtol=0)
    balanced=frame.groupby(['label','group','method'])[['CE','correct']].mean().groupby(['label','method']).mean().groupby('method').mean()
    saved=pd.read_csv(OUT/'development_method_metrics.csv',float_precision='round_trip').set_index('method')
    np.testing.assert_allclose(balanced.loc[saved.index,['CE','correct']],saved[['CE','correct']],atol=1e-12,rtol=0)
    plan=read(OUT/'development_bootstrap_groups.json');idx=np.load(OUT/'development_bootstrap_group_indices.npy',allow_pickle=False)
    groups=plan['groups'];pivot=frame.groupby(['label','group','method'])['CE'].mean().unstack('method')
    delta=np.array([pivot.loc[(g['label'],g['group']),'periodic']-pivot.loc[(g['label'],g['group']),'history_motion'] for g in groups]);w=np.array(plan['weights'])
    draws=np.sum(delta[idx]*w,axis=1);ci=np.percentile(draws,[2.5,97.5]);contrast=pd.read_csv(OUT/'development_CE_contrasts.csv',float_precision='round_trip')
    row=contrast[contrast['contrast']=='periodic_minus_history_motion'].iloc[0]
    np.testing.assert_allclose([delta @ w,*ci],[row['mean'],row['CI95_lower'],row['CI95_upper']],atol=1e-12,rtol=0)
    count=accounting();require(not count['unresolved_attempt_tokens'],'Unresolved actual execution attempts')
    require(len(read(OUT/'timing_records.json'))==30*7*7,'Timing cohort/repetitions incomplete')
    require('torch' not in __import__('sys').modules,'Read-only audit must not initialize/import CUDA/model runtime')
    return dict(passed=True,scope='DEVELOPMENT/EXPLORATORY',real_execution_clips=len(entries),real_execution_groups=len({(v['label'],v['group']) for v in entries}),
        actual_verified_quality_and_timing_trials=len(all_trials),actual_calls=count,history=historical_integrity(),
        primary_development_CE_contrast_independently_reproduced=True,official_test_outcomes_opened=False,
        read_only=True,decodes=0,model_calls=0,regenerated_results=0)


def unit_tests():
    import sys
    records=[]
    for module in ['test_ucf101_causal_development','test_causal_observability']:
        r=__import__('subprocess').run([sys.executable,'-B','-m','unittest',module,'-v'],capture_output=True,text=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1'),encoding='utf-8')
        records.append(dict(module=module,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_sha256=sha(ROOT/(module+'.py')),
            fixtures='SYNTHETIC TEST FIXTURE; no scientific measurements',GPU_disabled=True))
        print('Unit/regression tests',module,'PASS' if r.returncode==0 else 'FAIL',flush=True)
    save(OUT/'unit_tests.json',dict(passed=all(r['returncode']==0 for r in records),suites=records));require(all(r['returncode']==0 for r in records),'Tests failed')
