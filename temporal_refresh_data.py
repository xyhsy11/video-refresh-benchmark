"""Identity-only cohort inventory and deterministic, technically gated acquisition."""
from __future__ import annotations
from collections import Counter
import csv
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from temporal_refresh_common import *


def canonical_frame_types(path):
    """Lossless pre-seal metadata spelling repair; never alter counts/members."""
    data=read(path);changed=False
    for v in data['videos']:
        names={'1':'I','2':'P','3':'B'}
        if any(k in names for k in v['frame_types']):
            v['frame_types']={names.get(k,k):n for k,n in v['frame_types'].items()};changed=True
        if 'resolution' not in v:
            import av
            with av.open(str(ROOT/v['relative_path'])) as container:
                stream=container.streams.video[0]
                v['resolution']={'width':stream.codec_context.width,'height':stream.codec_context.height}
                v['reported_fps']=float(stream.average_rate) if stream.average_rate else None
            changed=True
    if changed:
        assert not (path.parent/'evaluation_seal.json').exists()
        save(path,data)
    return data['videos']


def historical_exclusions(output):
    target=output/'source_exclusions.json'
    if target.exists(): return read(target)
    ids=set(); hashes=set(); evidence=[]
    # Membership/identity documents only. Never inspect protected outcomes,
    # reference arrays, motion tables, predictions or performance metadata.
    names=['dataset_manifest.json','split_manifest.json','development_split.json',
           'dataset_split.json','frozen_test_manifest.json','manifest.json','exclusions.json',
           'reused_membership.json']
    def walk(value):
        if isinstance(value,dict):
            source=value.get('source_video_id')
            if source: ids.add(source)
            elif 'video_id' in value and len(value['video_id'])==11: ids.add(value['video_id'])
            identity=value.get('file_identity',{})
            if identity.get('sha256'): hashes.add(identity['sha256'])
            for k in ['source_ids','excluded_source_ids','prior_source_ids']:
                if isinstance(value.get(k),list): ids.update(v for v in value[k] if isinstance(v,str))
            for k in ['file_hashes','excluded_file_hashes']:
                if isinstance(value.get(k),list): hashes.update(v for v in value[k] if isinstance(v,str))
            for v in value.values(): walk(v)
        elif isinstance(value,list):
            for v in value: walk(v)
    for directory in sorted((ROOT/'results').iterdir()):
        if not directory.is_dir() or directory.resolve()==output.resolve(): continue
        for name in names:
            path=directory/name
            if path.exists():
                walk(read(path)); evidence.append({'path':relative(path),'sha256':sha(path)})
    assert len(ids)>=690, f'Historical exclusions unexpectedly incomplete: {len(ids)}'
    result={'frozen_at_utc':now(),'source_ids':sorted(ids),'encoded_sha256':sorted(hashes),
        'evidence':evidence,'policy':'all historical/reserved/protected membership identities; no protected outcome access',
        'limitations':'parent IDs prevent adjacent same-source windows; exact hashes detect identical bytes, not reuploads/near duplicates'}
    save(target,result); return result


def freeze_data_plan(output):
    path=output/'data_plan.json'
    if path.exists(): return read(path)
    baseline=read(ROOT/'results/recognition_baseline_v2/dataset_manifest.json')
    prior=historical_exclusions(output)
    classes=baseline['classes']; existing=[v for v in baseline['videos'] if v['recognition_role'] in ['train','development']]
    assert Counter(v['recognition_role'] for v in existing)=={'train':400,'development':100}
    assert len({v['source_video_id'] for v in existing})==500
    official_train_path=ASSETS/'k400_train.csv'
    train_rows=list(csv.DictReader(official_train_path.open(encoding='utf-8')))
    train_ids={r['youtube_id'] for r in train_rows}
    # Pretraining overlap is excluded for new evaluation, reported for reused roles.
    overlap=[v['source_video_id'] for v in existing if v['source_video_id'] in train_ids]
    save(output/'pretraining_overlap_audit.json',{'official_training_annotations':relative(official_train_path),
        'sha256':sha(official_train_path),'rows':len(train_rows),'distinct_parent_ids':len(train_ids),
        'existing_train_development_parent_overlap':overlap,
        'new_evaluation_policy':'exclude all official training annotation parent IDs',
        'unresolved':'checkpoint full training provenance, reuploads, near duplicates and other pretraining sources cannot be universally verified'})
    annotation=ROOT/'data/source/cache_drift_v2/val.csv'
    rows=list(csv.DictReader(annotation.open(encoding='utf-8')))
    lookup={r['youtube_id']:r for r in rows}
    for v in existing:
        assert lookup[v['source_video_id']]['label']==v['label']
        p=ROOT/v['relative_path']
        if not p.exists(): raise FileNotFoundError(f'Frozen {v["recognition_role"]} source missing: {p}')
        assert sha(p)==v['file_identity']['sha256'],p
    candidates=[]; exclusions=set(prior['source_ids'])|train_ids
    for row in rows:
        if row['label'] not in classes or row['youtube_id'] in exclusions: continue
        filename=f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}.mp4"
        candidates.append({**row,'filename':filename,'rank':rank('candidate',filename)})
    urls=(ROOT/'data/source/cache_drift_v2/k400_val_path.txt').read_text().splitlines()
    cached=[]; uncached=[]
    for url in urls:
        p=ROOT/'data/source/cache_drift_v2/shards'/PurePosixPath(url).name
        info=p.with_suffix('.download.json')
        if p.exists() and info.exists():
            previous=read(info)
            print('Verify cached archive',p.name,flush=True)
            assert previous['url']==url and p.stat().st_size==previous['size_bytes'] and sha(p)==previous['sha256']
            cached.append({'url':url,'path':relative(p),'sha256':previous['sha256'],'size_bytes':p.stat().st_size})
        else: uncached.append({'url':url,'path':relative(ASSETS/'shards'/p.name)})
    cached.sort(key=lambda e:rank('archive',e['url'])); uncached.sort(key=lambda e:rank('archive',e['url']))
    result={'frozen_at_utc':now(),'classes':classes,'source_manifest_sha256':sha(ROOT/'results/recognition_baseline_v2/dataset_manifest.json'),
        'existing':existing,'candidate_rows':sorted(candidates,key=lambda e:(e['rank'],e['filename'])),
        'archive_order':cached+uncached,'cached_bytes':sum(e['size_bytes'] for e in cached),
        'annotation_sha256':sha(annotation),'archive_list_sha256':sha(ROOT/'data/source/cache_drift_v2/k400_val_path.txt'),
        'prior_exclusions_sha256':sha(output/'source_exclusions.json'),
        'sampling_rule':'verified cached archives first, SHA256(seed,archive URL) order; within each archive SHA256(seed,filename) rank; next eligible same-class candidate replaces only technical ineligibility',
        'roles':'exact existing400train/100development;100new evaluation5/class; no exploratory features or protected outcomes opened',
        'eligibility':'official class label/access; H264,5..15sec; complete finite increasing PTS;16 distinct supported trailing four-frame snippets; no outcome-based exclusions',
        'access_terms':'CVDF points to Kinetics encoded archive sources; dataset/video rights remain separate from MIT project code; no video redistribution',
        'resource_inventory':{'available_disk_bytes':shutil.disk_usage(ROOT).free,'new_video_target':100,'available_annotation_candidates_by_class':dict(Counter(r['label'] for r in candidates))}}
    save(path,result); return result


def prepare_existing(output,plan):
    target=output/'train_development_manifest.json'
    if target.exists():
        videos=canonical_frame_types(target); originals={v['source_video_id']:v for v in plan['existing']};changed=False
        for v in videos:
            for k in ['original_dataset_split','time_start','time_end','archive_url','archive_sha256']:
                if k not in v and k in originals[v['source_video_id']]:v[k]=originals[v['source_video_id']][k];changed=True
        if changed:
            assert not (output/'evaluation_seal.json').exists()
            data=read(target);data['videos']=videos;save(target,data)
        return videos
    videos=[]
    for i,v in enumerate(plan['existing']):
        inspection=inspect_encoded(ROOT/v['relative_path'])
        videos.append({**{k:v[k] for k in ['video_id','source_video_id','label','relative_path','file_identity','recognition_role',
            'original_dataset_split','time_start','time_end','archive_url','archive_sha256'] if k in v},**inspection})
        if i%25==0: print(f'Technical train/dev inspection {i+1}/500',flush=True)
    save(target,{'classes':plan['classes'],'videos':videos,'roles_unchanged':True,'created_at_utc':now()})
    return videos


def acquire_evaluation(output,plan,budget):
    import av
    frozen=output/'evaluation_manifest.json'
    if frozen.exists(): return canonical_frame_types(frozen)
    progress=output/'acquisition.json'
    state=read(progress) if progress.exists() else {'selected':[],'events':[],'traversed_archives':[],
        'sampling_plan_sha256':sha(output/'data_plan.json'),'started_at_utc':now()}
    assert state['sampling_plan_sha256']==sha(output/'data_plan.json')
    excluded_hashes=set(read(output/'source_exclusions.json')['encoded_sha256'])
    for v in state['selected']:
        assert sha(ROOT/v['relative_path'])==v['file_identity']['sha256']
    for archive in plan['archive_order']:
        if len(state['selected'])==100: break
        if archive['url'] in state['traversed_archives']: continue
        path=ROOT/archive['path']
        if not path.exists(): budget.download(archive['url'],path)
        digest=sha(path)
        if 'sha256' in archive: assert digest==archive['sha256']
        print('Search evaluation candidates',path.name,flush=True)
        # One compressed traversal, extract only prospective class members.
        candidates={r['filename']:r for r in plan['candidate_rows']}
        extracted={}
        with tarfile.open(path,mode='r|gz') as tar:
            for member in tar:
                name=PurePosixPath(member.name).name
                if not member.isfile() or name not in candidates: continue
                dest=ASSETS/'evaluation_candidates'/name
                if not dest.exists():
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(member) as source,dest.open('wb') as out: shutil.copyfileobj(source,out)
                extracted[name]=dest
        for row in plan['candidate_rows']:
            counts=Counter(v['label'] for v in state['selected'])
            if row['filename'] not in extracted or counts[row['label']]>=5: continue
            source=row['youtube_id']
            if source in {v['source_video_id'] for v in state['selected']}: continue
            p=extracted[row['filename']]; digest_file=sha(p)
            event={'source_video_id':source,'filename':p.name,'label':row['label'],'archive_url':archive['url'],'rank':row['rank']}
            try:
                if digest_file in excluded_hashes or digest_file in {v['file_identity']['sha256'] for v in state['selected']}:
                    raise ValueError('duplicate encoded bytes against historical/new cohort')
                inspection=inspect_encoded(p)
            except (ValueError,av.error.FFmpegError) as exc:
                # Inspection-only failures are recorded; CUDA/encoder never run here.
                event.update(eligible=False,reason=f'{type(exc).__name__}: {exc}')
                state['events'].append(event); save(progress,state); continue
            video={'video_id':p.stem,'source_video_id':source,'label':row['label'],'recognition_role':'evaluation',
                'relative_path':relative(p),'file_identity':{'sha256':digest_file,'size_bytes':p.stat().st_size},
                'original_dataset_split':'official Kinetics400 validation','time_start':row['time_start'],'time_end':row['time_end'],
                'archive_url':archive['url'],'archive_sha256':digest,**inspection}
            event.update(eligible=True,reason='technical eligibility passed; no predictions computed')
            state['events'].append(event); state['selected'].append(video); save(progress,state)
        state['traversed_archives'].append(archive['url']); save(progress,state)
    counts=Counter(v['label'] for v in state['selected'])
    missing={c:5-counts[c] for c in plan['classes'] if counts[c]!=5}
    state.update(shortfall=missing,complete=not missing); save(progress,state)
    if missing: raise RuntimeError(f'Evaluation acquisition incomplete: {missing}')
    assert len(state['selected'])==100
    save(frozen,{'classes':plan['classes'],'videos':state['selected'],'frozen_at_utc':now(),
        'no_evaluation_inference_performed':True,'timing_sources':[
            min([v['source_video_id'] for v in state['selected'] if v['label']==c],key=lambda s:rank('timing',s)) for c in plan['classes']]})
    return state['selected']
