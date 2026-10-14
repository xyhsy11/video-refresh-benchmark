"""Frozen exclusions, bounded resumable CVDF acquisition and CUDA references."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import threading
import time
import urllib.request
from unittest.mock import patch

import av
import numpy as np
import torch

from benchmark import Encoder, configure_numerics
from collect_cache_drift_v2 import collect_one
from prepare_cache_drift_v2 import CACHE, BASE, download_shard, inspect_eligible, metadata_download
from motion_utils import current_environment, write_csv, EXPORT_CONFIG, DESCRIPTOR_CONFIG
from research_utils import ROOT, now, relative_path, sha256_file, write_json


def strict_json(path:Path,value:dict) -> None:
    json.dumps(value,allow_nan=False); write_json(path,value)


def check_frozen(frozen_run:Path,protocol:dict) -> tuple[dict,dict]:
    for filename,key in [('fitted_predictors.json','fixed_models_sha256'),('selected_thresholds.json','fixed_settings_sha256')]:
        if not (frozen_run/filename).exists(): raise FileNotFoundError(relative_path(frozen_run/filename))
        if sha256_file(frozen_run/filename)!=protocol[key]: raise ValueError(f'Frozen digest mismatch: {filename}')
    return json.loads((frozen_run/'fitted_predictors.json').read_text())['models'],json.loads((frozen_run/'selected_thresholds.json').read_text())['selected']


def prior_exclusions(output:Path,protocol:dict) -> dict:
    path=output/'exclusions.json'
    if path.exists():
        existing=json.loads(path.read_text())
        assert set(protocol['excluded_source_ids'])<=set(existing['source_ids'])
        assert set(protocol['excluded_file_SHA256'])<=set(existing['file_hashes'])
        return existing
    assert len(set(protocol['excluded_source_ids']))==len(set(protocol['excluded_file_SHA256']))==210
    ids=set(protocol['excluded_source_ids']); hashes=set(protocol['excluded_file_SHA256']); records=[]; inputs={}
    # Conservative: even prior inspected eligibility candidates count as used.
    for p in sorted((ROOT/'results').rglob('*.json')):
        if output.resolve() in p.resolve().parents: continue
        if not (p.name.endswith('manifest.json') or p.name in ['eligible_candidates.json','collection_progress.json','exclusions.json'] or p.parent.name=='features'): continue
        data=json.loads(p.read_text(encoding='utf-8')); rows=[]
        for key in ['videos','reserves','records']:
            value=data.get(key,[])
            rows.extend(value.values() if isinstance(value,dict) else value if isinstance(value,list) else [])
        if 'entry' in data: rows.append(data['entry'])
        found=False
        for row in rows:
            row=row.get('entry',row)
            identity=row.get('file_identity',{}); digest=identity.get('sha256')
            sid=row.get('source_video_id') or str(row.get('video_id',''))[:11]
            if not re.fullmatch(r'[A-Za-z0-9_-]{11}',sid): continue
            if not digest and not row.get('archive_url'): continue
            found=True
            if sid not in ids or digest and digest not in hashes:
                records.append({'source_id':sid,'sha256':digest,'evidence':relative_path(p),'rule':'prior persisted membership or eligibility inspection; performance not consulted'})
            ids.add(sid)
            if digest: hashes.add(digest)
        if found: inputs[relative_path(p)]=sha256_file(p)
    result={'frozen_at_utc':now(),'protocol_sources':210,'protocol_hashes':210,'source_ids':sorted(ids),'file_hashes':sorted(hashes),
        'additional_sources':sorted(ids-set(protocol['excluded_source_ids'])),'additional_hashes':sorted(hashes-set(protocol['excluded_file_SHA256'])),
        'evidence':records,'evidence_file_hashes':inputs,'rule':'Freeze once, before selection. Include prior inspected candidates conservatively. Never rescan during resume or add this run itself.'}
    strict_json(path,result); return result


def make_sampling_plan(output:Path,protocol:dict,exclusions:dict) -> dict:
    path=output/'sampling_plan.json'
    if path.exists(): return json.loads(path.read_text())
    metadata_download(BASE+'annotations/val.csv',CACHE/'val.csv')
    metadata_download(BASE+'val/k400_val_path.txt',CACHE/'k400_val_path.txt')
    previous=json.loads((ROOT/'results/cache_drift_v2/sampling_plan.json').read_text())
    assert sha256_file(CACHE/'val.csv')==previous['annotations_sha256']
    assert sha256_file(CACHE/'k400_val_path.txt')==previous['shard_list_sha256']
    rows=list(csv.DictReader((CACHE/'val.csv').open(encoding='utf-8',newline='')))
    seed=protocol['sampling_seed']
    for row in rows:
        row['filename']=f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}.mp4"
        row['rank']=hashlib.sha256(f"fresh_refresh_confirmation_v1|{seed}|{row['label']}|{row['filename']}".encode()).hexdigest()
    candidates=sorted([r for r in rows if r['label'] in protocol['classes'] and r['youtube_id'] not in exclusions['source_ids']],key=lambda r:(r['rank'],r['filename']))
    urls=(CACHE/'k400_val_path.txt').read_text().splitlines()
    cached=[]; uncached=[]
    for url in urls:
        destination=CACHE/'shards'/PurePosixPath(url).name
        info_path=destination.with_suffix('.download.json')
        if destination.exists() and info_path.exists():
            info=json.loads(info_path.read_text())
            if destination.stat().st_size!=info['size_bytes'] or info['url']!=url: raise ValueError(f'Cache identity mismatch: {destination.name}')
            print(f'Verifying cached archive {destination.name}',flush=True)
            if sha256_file(destination)!=info['sha256']: raise ValueError(f'Cached archive SHA mismatch: {destination.name}')
            cached.append({**info,'path':relative_path(destination)})
        else: uncached.append(url)
    rank=lambda url:hashlib.sha256(f'fresh_refresh_confirmation_v1|archive|{seed}|{url}'.encode()).hexdigest()
    cached=sorted(cached,key=lambda a:rank(a['url'])); uncached.sort(key=rank)
    plan={'frozen_at_utc':now(),'seed':seed,'classes':protocol['classes'],'annotations_sha256':sha256_file(CACHE/'val.csv'),
        'archive_list_sha256':sha256_file(CACHE/'k400_val_path.txt'),'annotation_url':BASE+'annotations/val.csv','archive_list_url':BASE+'val/k400_val_path.txt',
        'verified_cached_archives':cached,'ordered_archives':[a['url'] for a in cached]+uncached,'candidate_rows':candidates,
        'ranking_rule':'verified cached archives first, then uncached; SHA256(fresh_refresh_confirmation_v1|archive|seed|URL) each group; within archive candidate SHA256(fresh_refresh_confirmation_v1|seed|label|filename)',
        'replacement_rule':'next same-class candidate in fixed archive/rank order; only input eligibility, never policy error or motion magnitude',
        'stop_rule':'stop after five finite CUDA references per class; no fresh policy evaluation until full manifest frozen',
        'exclusions_sha256':sha256_file(output/'exclusions.json')}
    strict_json(path,plan); return plan


class ReferenceCollector:
    def __init__(self,output:Path,protocol:dict):
        self.output=output
        original=json.loads((ROOT/'results/cache_drift_v2/collection_metadata.json').read_text())
        environment=current_environment(); strict_json(output/'environment.json',environment)
        if not environment['passed'] or not torch.cuda.is_available(): raise RuntimeError('CUDA environment check failed; no CPU feature substitution')
        for package in ['torch','torchvision','av','numpy','pandas','matplotlib','pillow']:
            old={k.lower():v for k,v in original['environment']['installed_distributions'].items()}
            current={k.lower():v for k,v in environment['installed_distributions'].items()}
            assert old[package]==current[package],f'Collection dependency changed: {package}'
        checkpoint=ROOT/'data/source/torch_hub/checkpoints'/original['model']['weights_url'].rsplit('/',1)[-1]
        if not checkpoint.exists(): raise FileNotFoundError(f'Required original weights absent: {relative_path(checkpoint)}')
        if sha256_file(checkpoint)!=original['model']['weights_sha256']: raise ValueError('Original ResNet18 weight digest mismatch')
        numerics=configure_numerics(protocol['sampling_seed'])
        assert numerics==original['numerical_settings']
        with patch('torch.hub.download_url_to_file',side_effect=RuntimeError('Weight download prohibited; use original cache')):
            self.encoder=Encoder(torch.device('cuda:0'))
        assert str(self.encoder.preprocess)==original['model']['preprocessing']
        assert str(self.encoder.weights)=='ResNet18_Weights.IMAGENET1K_V1'
        self.progress_path=output/'collection_progress.json'
        self.progress=json.loads(self.progress_path.read_text()) if self.progress_path.exists() else {'videos':[],'attempts':[],'smoke_passed':False}
        self.metadata_path=output/'collection_metadata.json'
        self.metadata=json.loads(self.metadata_path.read_text()) if self.metadata_path.exists() else {
            'started_at_utc':now(),'status':'collecting','device':'cuda:0','model':original['model'],'numerical_settings':numerics,
            'decoder':original['decoder'],'motion_export':EXPORT_CONFIG,'motion_descriptor_definition':DESCRIPTOR_CONFIG,
            'actual_forward_calls_entered':0,'retained_reference_features':0,'extra_forward_calls':0,'reference_collection_is_not_latency':True}
        assert self.metadata['model']==original['model'] and self.metadata['numerical_settings']==numerics
        self.active_attempt=None
        def count_forward(module,args):
            self.metadata['actual_forward_calls_entered']+=1
            self.active_attempt['forward_calls_entered']+=1
            strict_json(self.metadata_path,self.metadata)
            strict_json(self.progress_path,self.progress)
        self.encoder.model.register_forward_pre_hook(count_forward)
        for name in ['features','motion_frames']: (output/name).mkdir(exist_ok=True)

    def execute(self,entry:dict,kind:str) -> tuple:
        attempt={'video_id':entry['video_id'],'kind':kind,'started_at_utc':now(),'forward_calls_entered':0,'status':'started'}
        self.progress['attempts'].append(attempt); self.active_attempt=attempt
        strict_json(self.progress_path,self.progress)
        try:
            with torch.inference_mode(): result=collect_one(entry,self.encoder)
        except Exception as exc:
            attempt.update(status='failed_or_interrupted',reason=f'{type(exc).__name__}: {exc}',completed_at_utc=now())
            strict_json(self.progress_path,self.progress); raise
        attempt.update(status='completed_and_CUDA_synchronized',completed_at_utc=now())
        strict_json(self.progress_path,self.progress)
        assert attempt['forward_calls_entered']==16
        return result

    def smoke(self) -> None:
        if self.progress['smoke_passed']: return
        old=json.loads((ROOT/'results/cache_drift_v2/split_manifest.json').read_text())
        entry=sorted([e for e in old['videos'] if e['experiment_split']=='train'],key=lambda e:e['source_video_id'])[0]
        self.execute(entry,'original_training_smoke')
        self.progress['smoke_passed']=True; strict_json(self.progress_path,self.progress)
        print('CUDA reference smoke passed on original training video (16 extra forward passes).',flush=True)

    def collect(self,entry:dict) -> None:
        features,candidates,frames,unused_drift_labels,info=self.execute(entry,'fresh_candidate')
        vid=entry['video_id']; entry={**entry,'eligibility':'full decode/timestamps and finite CUDA references passed'}
        np.savez_compressed(self.output/'features'/f'{vid}.npz',features=features,frame_indices=np.array(entry['candidate_indices'],dtype=np.int64),
                            timestamps_seconds=np.array([c['timestamp_seconds'] for c in candidates],dtype=np.float64))
        strict_json(self.output/'features'/f'{vid}.json',{'entry':entry,'candidates':candidates})
        write_csv(self.output/'motion_frames'/f'{vid}.csv',frames,list(frames[0]))
        names=[f'features/{vid}.npz',f'features/{vid}.json',f'motion_frames/{vid}.csv']
        self.progress['videos'].append({'entry':entry,'counts':info,'artifact_sha256':{n:sha256_file(self.output/n) for n in names}})
        self.metadata['retained_reference_features']=16*len(self.progress['videos'])
        self.metadata['extra_forward_calls']=self.metadata['actual_forward_calls_entered']-self.metadata['retained_reference_features']
        strict_json(self.progress_path,self.progress); strict_json(self.metadata_path,self.metadata)


def acquire_and_collect(output:Path,protocol:dict,plan:dict,exclusions:dict,collector:ReferenceCollector,limit_gib:float,workers:int) -> dict:
    path=output/'acquisition.json'
    state=json.loads(path.read_text()) if path.exists() else {'started_at_utc':now(),'status':'acquiring','archives':[],
        'events':[],'eligibility_outcomes':[],'cached_archive_bytes':0,'newly_reserved_archive_bytes':0,'transfer_response_body_bytes':0,
        'compressed_limit_bytes':int(limit_gib*1024**3),'accounting_note':'reserve full uncached compressed size once, including partial/resumed; verified cached bytes excluded; response bodies include retransmissions, metadata headers/traffic can differ'}
    assert state['compressed_limit_bytes']==int(limit_gib*1024**3)
    completed={v['entry']['video_id']:v for v in collector.progress['videos']}
    used_ids={v['entry']['source_video_id'] for v in completed.values()}; used_hashes={v['entry']['file_identity']['sha256'] for v in completed.values()}
    for v in completed.values():
        for n,digest in v['artifact_sha256'].items(): assert sha256_file(output/n)==digest,n
        assert sha256_file(ROOT/v['entry']['relative_path'])==v['entry']['file_identity']['sha256']
    counts=Counter(v['entry']['label'] for v in completed.values())
    lookup={r['filename']:r for r in plan['candidate_rows']}
    done={r['video_id'] for r in state['eligibility_outcomes'] if r['status'] in ['retained','ineligible','excluded_duplicate']}
    archive_records={a['url']:a for a in state['archives']}
    cached={a['url']:a for a in plan['verified_cached_archives']}
    videos_dir=ROOT/'data/source/fresh_refresh_confirmation_v1/videos'; videos_dir.mkdir(parents=True,exist_ok=True)
    for url in plan['ordered_archives']:
        if all(counts[c]==protocol['videos_per_class'] for c in protocol['classes']): break
        if url in archive_records and archive_records[url]['status']=='processed': continue
        destination=CACHE/'shards'/PurePosixPath(url).name
        if url not in archive_records:
            if url in cached:
                archive={**cached[url],'status':'verified_cached','new_reservation_bytes':0}
                state['cached_archive_bytes']+=archive['size_bytes']
            else:
                with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=45) as r: size=int(r.headers['Content-Length']); etag=r.headers.get('ETag')
                if state['newly_reserved_archive_bytes']+size>state['compressed_limit_bytes']:
                    state['events'].append({'url':url,'kind':'cap_blocked','required_full_size_bytes':size,'at_utc':now()}); break
                archive={'url':url,'size_bytes':size,'etag':etag,'status':'reserved','new_reservation_bytes':size}
                state['newly_reserved_archive_bytes']+=size
            state['archives'].append(archive); archive_records[url]=archive
            strict_json(path,state)
        archive=archive_records[url]
        if url not in cached and archive['status'] in ['reserved','interrupted']:
            archive.setdefault('partial_bytes_before_first_attempt',sum(p.stat().st_size for p in destination.with_name(destination.name+'.chunks').glob('*.part')))
            original_open=urllib.request.urlopen; lock=threading.Lock(); last_saved=[state['transfer_response_body_bytes']]
            class Response:
                def __init__(self,r): self.r=r
                def __getattr__(self,k): return getattr(self.r,k)
                def __enter__(self): self.r.__enter__(); return self
                def __exit__(self,*args): return self.r.__exit__(*args)
                def read(self,*args):
                    data=self.r.read(*args)
                    with lock:
                        state['transfer_response_body_bytes']+=len(data)
                        if state['transfer_response_body_bytes']-last_saved[0]>=64*1024**2:
                            strict_json(path,state); last_saved[0]=state['transfer_response_body_bytes']
                    return data
            def tracked_open(*args,**kwargs):
                request=args[0]; method=request.get_method() if hasattr(request,'get_method') else 'GET'
                event={'url':url,'kind':'http_attempt','method':method,'range':request.get_header('Range') if hasattr(request,'get_header') else None,'at_utc':now()}
                try: response=original_open(*args,**kwargs)
                except Exception as exc:
                    event.update(status='failed_resumable',reason=str(exc))
                    with lock: state['events'].append(event); strict_json(path,state)
                    raise
                event.update(status='opened',response_status=response.status)
                with lock: state['events'].append(event); strict_json(path,state)
                return Response(response)
            print(f"Downloading {destination.name}; newly reserved {state['newly_reserved_archive_bytes']/1024**3:.2f} GiB / {limit_gib:g} GiB",flush=True)
            try:
                with patch('urllib.request.urlopen',tracked_open): result=download_shard(url,destination,workers)
                if result['size_bytes']!=archive['size_bytes'] or result['etag']!=archive['etag']: raise ValueError('Reserved remote archive identity changed')
                archive.update(result); archive['status']='downloaded'
            except Exception as exc:
                archive['status']='interrupted'; state['events'].append({'url':url,'kind':'resumable_download_failure','reason':str(exc),'at_utc':now()})
                strict_json(path,state); raise
            strict_json(path,state)
        print(f'Indexing {destination.name} in frozen traversal order',flush=True)
        matches=[]
        with tarfile.open(destination,mode='r|gz') as tar:
            for member in tar:
                filename=PurePosixPath(member.name).name
                if not member.isfile() or filename not in lookup: continue
                row=lookup[filename]
                if counts[row['label']]>=protocol['videos_per_class'] or filename[:-4] in done: continue
                target=videos_dir/filename
                if not target.exists():
                    temporary=target.with_suffix('.mp4.part')
                    with tar.extractfile(member) as src,temporary.open('wb') as dst: shutil.copyfileobj(src,dst)
                    assert temporary.stat().st_size==member.size; temporary.replace(target)
                else: assert target.stat().st_size==member.size
                matches.append((row,target))
        for row,target in sorted(matches,key=lambda pair:(pair[0]['rank'],pair[0]['filename'])):
            if counts[row['label']]>=protocol['videos_per_class'] or target.stem in done: continue
            digest=sha256_file(target); event={'video_id':target.stem,'source_id':row['youtube_id'],'label':row['label'],'archive_url':url,
                'rank':row['rank'],'encoded_sha256':digest,'at_utc':now()}
            if row['youtube_id'] in used_ids or row['youtube_id'] in exclusions['source_ids'] or digest in used_hashes or digest in exclusions['file_hashes']:
                event.update(status='excluded_duplicate',reason='prior or within-cohort source/hash overlap')
            else:
                try: entry=inspect_eligible(target,row,archive)
                except (ValueError,av.error.FFmpegError) as exc:
                    event.update(status='ineligible',reason=f'{type(exc).__name__}: {exc}')
                else:
                    entry.update(experiment_split='fresh_test',sampling_rank=row['rank'])
                    try: collector.collect(entry)
                    except av.error.FFmpegError as exc: event.update(status='ineligible',reason=f'decode: {exc}')
                    except ValueError as exc:
                        if 'Nonfinite/unsafe reference features' in str(exc): event.update(status='ineligible',reason=str(exc))
                        else: raise  # model/configuration/motion errors are blockers, not replacements
                    else:
                        used_ids.add(row['youtube_id']); used_hashes.add(digest); counts[row['label']]+=1
                        event.update(status='retained',selection_ordinal=sum(counts.values()))
                        print(f"Collected {sum(counts.values())}/100: {row['label']} {counts[row['label']]}/5; finite 16 CUDA references",flush=True)
            state['eligibility_outcomes'].append(event); done.add(target.stem); strict_json(path,state)
        archive['status']='processed'; archive['matching_candidates']=len(matches); strict_json(path,state)
    shortfall={c:max(0,protocol['videos_per_class']-counts[c]) for c in protocol['classes']}
    state.update(status='complete' if not any(shortfall.values()) else 'incomplete',selected_videos=sum(counts.values()),class_counts=dict(counts),shortfall_per_class=shortfall,updated_at_utc=now())
    strict_json(path,state)
    if any(shortfall.values()): raise RuntimeError(f'Incomplete cohort: {sum(counts.values())}/100; exact shortfall {shortfall}')
    entries=[v['entry'] for v in collector.progress['videos']]
    assert len(entries)==100 and len(used_ids)==len(used_hashes)==100
    frozen={'status':'frozen','frozen_at_utc':now(),'sampling_plan_sha256':sha256_file(output/'sampling_plan.json'),
        'exclusions_sha256':sha256_file(output/'exclusions.json'),'videos':sorted(entries,key=lambda e:e['source_video_id']),
        'class_counts':{c:counts[c] for c in protocol['classes']},'reference_features':1600}
    strict_json(output/'frozen_test_manifest.json',frozen)
    collector.metadata.update(status='complete',completed_at_utc=now(),retained_reference_features=1600,
        extra_forward_calls=collector.metadata['actual_forward_calls_entered']-1600,
        accounting_note='Forward pre-hook counts every model-forward entry, including smoke/retries. Completed retained clips synchronized CUDA; interrupted calls may not complete.')
    strict_json(output/'collection_metadata.json',collector.metadata)
    return frozen
