"""Outcome-free provenance adapters, deterministic prefixes, official acquisition.

No model import is required by the audit or downloader. Publisher digests are
not asserted: archive/member SHA256 values are locally computed identities.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time
import urllib.request
import urllib.error

from native_head_value_confirmation import read, save, sha, rank, validate_components
from native_head_confirmation_state import guard, metered, advance, FREE_RESERVE

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/native_head_value_confirmation_v1'
CACHE=ROOT/'data/source/native_head_value_confirmation_v1'
ADAPTER_VERSION='explicit-input-schemas-3'
INPUT_CODE_SHA_AT_IMPORT=sha(__file__)
TRANSFER_IO_RECIPE='buffered_4MiB_reserved_v1'
YT=re.compile(r'^([A-Za-z0-9_-]{11})(?:_\d{6}_\d{6})?(?:\.mp4)?$')
HEX=re.compile(r'^[0-9a-f]{64}$')
RESERVED={'fresh_test','test','protected','reserved','holdout','protected_test','evaluation','confirmation','sealed_test','eval',
          'excluded_official_test','excluded_test_family','excluded_test_family_conflict','excluded_train_test_conflict','test_reserved'}
NONRESERVED={'train','training','development','dev','fit','calibration','probe_fit','unassigned_training_pool','train_development','head_train','validation','exploratory','exploratory_test'}
INVENTORY={'unassigned','excluded_official_unused','excluded_technical','unused','input_only','screening'}


class Union:
    def __init__(self): self.parent={}
    def root(self,node):
        self.parent.setdefault(node,node)
        p=node
        while self.parent[p]!=p: p=self.parent[p]
        while self.parent[node]!=node:
            nxt=self.parent[node];self.parent[node]=p;node=nxt
        return p
    def join(self,nodes):
        roots=sorted({self.root(n) for n in nodes},key=lambda n:(0 if n.startswith('yt:') else 1 if n.startswith('family:') else 2,n))
        for node in roots[1:]: self.parent[node]=roots[0]


def clip_nodes(value):
    if not isinstance(value,str): return []
    name=value.replace('\\','/').rsplit('/',1)[-1]
    match=YT.fullmatch(name)
    if match:
        return ['yt:'+match[1]]+(['clip:kinetics:'+name.removesuffix('.mp4')+'.mp4'] if '_' in name[11:] else [])
    if name.lower().endswith('.avi'):
        ucf=re.fullmatch(r'v_(.+)_g(\d+)_c\d+\.avi',name,re.I)
        return ['clip:ucf:'+name,'family:ucf:'+ucf[1].lower()+':g'+ucf[2]] if ucf else ['clip:hmdb:'+name]
    return []


def records(path):
    """Canonical JSON and JSONL, compressed or uncompressed, with pointers."""
    opener=gzip.open if path.name.endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8-sig') as f:
        if '.jsonl' in path.name:
            for i,line in enumerate(f):
                if line.strip(): yield json.loads(line), '/records/'+str(i)
        else: yield json.load(f),''


def evidence_paths():
    # Inventory file names only; never open arrays, state/prediction/results files.
    from run_native_head_value_confirmation import metadata_paths
    paths=set(metadata_paths())
    include=re.compile(r'(manifest|membership|population|screening|source|exclusion|exposure|duplicate|collection|split|allocation|eligib|protocol)',re.I)
    for base in (ROOT/'results',ROOT/'data/source/causal_observability_v1'):
        for p in base.rglob('*'):
            if not p.is_file() or 'native_head_value_confirmation_v1' in p.parts: continue
            if include.search(p.name) and (p.name.endswith(('.json.gz','.jsonl.gz')) or p.name=='protocol.md'):
                paths.add(p)
    return sorted(paths)


@dataclass
class Audit:
    graph: Union
    exposed: set
    forbidden: set
    evidence: list
    edges: list
    decisions: list
    fingerprints: list


def historical_audit(extra=(),write=True):
    graph=Union();exposed=set();reserved=set();unknown=set();affirmed=set();evidence=[];edges=[];decisions=[];fingerprints=[]
    def edge(nodes,path,pointer,kind):
        nodes=sorted(set(nodes))
        graph.join(nodes)
        if nodes: edges.append(dict(nodes=nodes,file=path,pointer=pointer,adapter=ADAPTER_VERSION,kind=kind))
    def walk(obj,path,pointer='',inherited=None):
        if isinstance(obj,list):
            for i,v in enumerate(obj): walk(v,path,pointer+'/'+str(i),inherited)
            return
        if not isinstance(obj,dict): return
        nodes=[]
        for key in ('youtube_id','source_video_id','original_parent_id','parent_youtube_id','video_id','clip','key','filename','relative_path','path'):
            nodes+=clip_nodes(obj.get(key))
        for key in ('original_url','source_url','parent_url','youtube_url'):
            if isinstance(obj.get(key),str):
                m=re.search(r'(?:youtu\.be/|[?&]v=)([A-Za-z0-9_-]{11})(?:[?&#/]|$)',obj[key])
                if m: nodes.append('yt:'+m[1])
        for key in ('encoded_sha256','video_sha256','file_sha256','input_sha256','sha256'):
            value=obj.get(key)
            if key=='sha256' and str(obj.get('path','')).lower().endswith(('.npz','.npy','.json','.gz')):
                continue  # Collection artifact hash is not an encoded-video hash.
            values=value if isinstance(value,list) else [value]
            # Flat exclusion hashes have no source association; do not join them.
            if isinstance(value,list):
                for i,v in enumerate(values):
                    if isinstance(v,str) and HEX.fullmatch(v):
                        n='bytes:'+v;graph.root(n);unknown.add(n)
                        decisions.append(dict(nodes=[n],decision='unresolved_union',file=path,pointer=pointer+'/'+key+'/'+str(i)))
            elif nodes and isinstance(value,str) and HEX.fullmatch(value): nodes.append('bytes:'+value)
        fi=obj.get('file_identity')
        if nodes and isinstance(fi,dict) and isinstance(fi.get('sha256'),str): nodes.append('bytes:'+fi['sha256'])
        dataset_nodes=nodes+[n for v in obj.get('members',[]) if isinstance(v,str) for n in clip_nodes(v)] if isinstance(obj.get('members',[]),list) else nodes
        dataset='hmdb' if any(n.startswith('clip:hmdb:') for n in dataset_nodes) else 'ucf' if any(n.startswith('clip:ucf:') for n in dataset_nodes) else 'kinetics'
        for key in ('source_family','family','global_component','duplicate_component'):
            if nodes and isinstance(obj.get(key),str): nodes.append('family:'+dataset+':'+obj[key])
        edge(nodes,path,pointer,'identity')
        role=inherited
        explicit=[str(obj[k]).lower() for k in ('role','recognition_role','experiment_split','data_role') if isinstance(obj.get(k),str)]
        for value in explicit:
            if value in RESERVED: role='reserved';break
            if value in NONRESERVED: role='nonreserved'
            elif value in INVENTORY: role=role or 'inventory'
            else: role='unresolved'
        if nodes:
            if role=='reserved': reserved.update(nodes)
            elif role=='nonreserved': affirmed.update(nodes)
            elif role=='unresolved': unknown.update(nodes)
            if role=='exposed' or ('collection' in Path(path).name and any(k in obj for k in ('completed_at_utc','artifact_sha256','feature_path','reference_features','counts','collected_at_utc','forward_starts','encoder_starts'))):
                exposed.update(nodes)
                decisions.append(dict(nodes=nodes,decision='model_exposed',file=path,pointer=pointer))
            decisions.append(dict(nodes=nodes,decision=role or 'inventory',file=path,pointer=pointer,explicit_roles=explicit))
            hashes=obj.get('fingerprints',obj.get('perceptual_fingerprint'))
            duration=obj.get('pts_span',obj.get('duration_pts_span',obj.get('duration_seconds',obj.get('decoded_pts_duration_seconds'))))
            if isinstance(hashes,list) and len(hashes)==3 and all(isinstance(h,str) and len(h)==63 for h in hashes) and duration:
                fingerprints.append(dict(nodes=nodes,fingerprints=hashes,duration=float(duration),file=path,pointer=pointer))
        # Actual duplicate schemas: {a,b}, {first,second}, pairs, and members.
        for a,b in [('a','b'),('first','second')]:
            if a in obj and b in obj: edge(clip_nodes(obj[a])+clip_nodes(obj[b]),path,pointer,'duplicate')
        members=obj.get('members')
        if isinstance(members,list):
            family=obj.get('source_family')
            mm=[n for v in members for n in clip_nodes(v)]
            if family: mm.append('family:'+dataset+':'+family)
            edge(mm,path,pointer+'/members','family_membership')
        for key in ('exact_edges','near_edges','edges'):
            for i,v in enumerate(obj.get(key,[]) if isinstance(obj.get(key),list) else []):
                if isinstance(v,list) and len(v)==2: edge(clip_nodes(v[0])+clip_nodes(v[1]),path,pointer+'/'+key+'/'+str(i),'duplicate_pair')
        for i,value in enumerate(obj.get('quarantined_groups',[])):
            if isinstance(value,list) and len(value)==2:
                n='family:ucf:'+str(value[0]).lower()+':'+str(value[1]);graph.root(n);reserved.add(n)
                decisions.append(dict(nodes=[n],decision='reserved',file=path,pointer=pointer+'/quarantined_groups/'+str(i)))
        for key in ('source_ids','excluded_source_ids','prior_source_ids','file_hashes','excluded_file_hashes','encoded_hashes'):
            values=obj.get(key,[])
            if isinstance(values,list):
                for i,value in enumerate(values):
                    ns=['bytes:'+value] if isinstance(value,str) and HEX.fullmatch(value) else clip_nodes(value)
                    for n in ns: graph.root(n);unknown.add(n)
                    if ns: decisions.append(dict(nodes=ns,decision='unresolved_union',file=path,pointer=pointer+'/'+key+'/'+str(i)))
        for key,value in obj.items():
            ptr=pointer+'/'+str(key).replace('~','~0').replace('/','~1')
            child_role=role
            if key in ('fresh_test','protected','reserved','holdout','frozen_test'): child_role='reserved'
            if key in ('train','development','training'): child_role='nonreserved'
            # Collection dictionaries key videos by parent; binding is explicit.
            if key=='videos' and isinstance(value,dict):
                for record_key,record in value.items():
                    if isinstance(record,dict) and clip_nodes(record_key):
                        walk(dict(record,video_id=record_key),path,ptr+'/'+record_key,child_role)
                    else: walk(record,path,ptr+'/'+record_key,child_role)
            elif isinstance(value,(dict,list)): walk(value,path,ptr,child_role)
    payloads={}
    for p in evidence_paths():
        name=p.relative_to(ROOT).as_posix();digest=sha(p)
        item=dict(path=name,sha256=digest,adapter=ADAPTER_VERSION)
        if p.name=='protocol.md':
            item['coverage']='original role protocol, no outcome content'; evidence.append(item);continue
        # Identity-resolve compressed/uncompressed copies, not double exposures.
        opener=gzip.open if p.name.endswith('.gz') else open
        h=hashlib.sha256()
        with opener(p,'rb') as f:
            for block in iter(lambda:f.read(2**20),b''):h.update(block)
        payload=h.hexdigest();item['payload_sha256']=payload
        if payload in payloads:
            item['duplicate_of']=payloads[payload];evidence.append(item);continue
        payloads[payload]=name
        policy=('reserved' if p.name in ('frozen_test_manifest.json','evaluation_manifest.json','confirmation_manifest.json') else
                'exposed' if p.name in ('train_development_collection.json','training_smoke_collection.json','evaluation_collection.json',
                                     'confirmation_collection.json','hmdb_eval_collection.json','hmdb_train_collection.json',
                                     'train_collection.json','test_collection.json','development_collection.json') else None)
        for obj,pointer in records(p):walk(obj,name,pointer,policy)
        evidence.append(item)
    # Bind all official annotations and all authenticated encoded identities
    # before final role decisions, including candidates outside the local prefix.
    for split in ('train','val','test'):
        import csv
        p=CACHE/'metadata'/f'{split}_annotations'
        with p.open(encoding='utf-8-sig',newline='') as f:
            for i,row in enumerate(csv.DictReader(f)):
                name=f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}.mp4"
                edge(clip_nodes(name),p.relative_to(ROOT).as_posix(),'/rows/'+str(i),'official_parent')
    for obj,path,pointer in extra:walk(obj,path,pointer,'inventory')
    reserved_roots={graph.root(n) for n in reserved};affirmed_roots={graph.root(n) for n in affirmed}
    unresolved_roots={graph.root(n) for n in unknown}-affirmed_roots
    forbidden_roots=reserved_roots|unresolved_roots
    forbidden={n for n in list(graph.parent) if graph.root(n) in forbidden_roots}
    result=Audit(graph,exposed,forbidden,evidence,edges,decisions,fingerprints)
    if write:
        def zipped(path,obj):
            tmp=path.with_name(path.name+'.writing')
            with gzip.open(tmp,'wt',encoding='utf-8',newline='\n',compresslevel=1) as f:json.dump(obj,f,sort_keys=True)
            tmp.replace(path)
        zipped(CACHE/'source_audit_graph.json.gz',dict(adapter=ADAPTER_VERSION,edges=edges,decisions=decisions,
            component_by_node={n:graph.root(n) for n in list(graph.parent)},
            exposed_components=sorted({graph.root(n) for n in exposed}),
            reserved_components=sorted(reserved_roots),unresolved_components=sorted(unresolved_roots),
            fingerprints=fingerprints))
        save(OUT/'historical_source_evidence.json',evidence)
        save(OUT/'source_audit_identity.json',dict(adapter=ADAPTER_VERSION,
            path=(CACHE/'source_audit_graph.json.gz').relative_to(ROOT).as_posix(),sha256=sha(CACHE/'source_audit_graph.json.gz'),
            edges=len(edges),evidence_files=len(evidence),inventory_nodes=len(graph.parent),
            model_exposed_components=len({graph.root(n) for n in exposed}),
            reserved_components=len(reserved_roots),unresolved_quarantined_components=len(unresolved_roots)))
    return result


def candidate_groups(split):
    return ordered_groups(read(CACHE/'candidate_manifest.json'),split)


def ordered_groups(candidates,split):
    from native_head_confirmation_availability import selectable
    grouped=defaultdict(list)
    for row in candidates:
        if row['split']==split and selectable(row):grouped[row['component']].append(row)
    return sorted(((c,sorted(v,key=lambda r:r['canonical_rank'])) for c,v in grouped.items()),key=lambda kv:rank('component',kv[0]))


def reserve_members(split,m):
    """Prospective quota plus two components/class; never arrival order."""
    counts=Counter();wanted=[]
    for component,members in candidate_groups(split):
        live=[]
        for row in members:
            receipt=CACHE/'screening'/(rank('screen',split+'|'+row['filename'])+'.json')
            if not receipt.exists() or read(receipt).get('eligible'):live.append(row)
        if not live:continue
        c=live[0]['y']
        if counts[c]<m+2:
            wanted.extend(members);counts[c]+=1
    return wanted


class InputMembershipChanged(RuntimeError):
    pass


class PlanningMembershipChanged(RuntimeError):
    """New input-only evidence invalidates an unexposed fitting generation."""
    pass


class MissingNeededInput(RuntimeError):
    def __init__(self,split,filename):
        self.split=split;self.filename=filename
        super().__init__(f'BLOCKED_INPUT: missing needed {split} prefix input {filename}')


def download_archive(record):
    from run_native_head_value_confirmation import network_ledger, CAPS, now, event
    split=record['split'];url=record['url'];size=record['content_length']
    if not size or not record.get('etag'):raise RuntimeError('BLOCKED_INPUT: remote archive representation not authenticated')
    if record.get('cached_path'):
        p=ROOT/record['cached_path']
        if sha(p)!=record['cached_sha256']:raise RuntimeError('Cached archive identity changed')
        return p,False
    target=CACHE/'scratch'/split/url.rsplit('/',1)[-1];target.parent.mkdir(parents=True,exist_ok=True)
    part=target.with_name(target.name+'.part');receipt=target.with_name(target.name+'.transfer.json')
    if target.exists():
        old=read(receipt)
        if old['sha256']!=sha(target) or old['etag']!=record['etag']:raise RuntimeError('Downloaded archive identity changed')
        return target,True
    previous=read(receipt) if receipt.exists() else None
    offset=part.stat().st_size if part.exists() else 0
    if offset and (not previous or previous.get('etag')!=record['etag'] or previous.get('url')!=url):
        raise RuntimeError('BLOCKED_INPUT: partial transfer has no matching representation receipt')
    ledger=network_ledger()
    if ledger.get('inflight'):
        # The upper read reservation was durably written before each read.
        inflight=ledger['inflight'];ledger['requests'].append(dict(inflight,status='interrupted_conservative',
            bytes=inflight['bytes_received']+inflight.get('read_reservation',0)))
        ledger['inflight']=None
        save(OUT/'network_ledger.json',ledger)
    used=sum(r['bytes'] for r in ledger['requests'])
    if used+size-offset>CAPS['network_bytes']:raise RuntimeError('BLOCKED_RESOURCE: complete archive exceeds remaining cumulative transfer')
    guard('collection',scratch=size-offset)
    headers={'User-Agent':'video-refresh-research/27','Accept-Encoding':'identity','If-Match':record['etag']}
    if offset:headers.update(Range=f'bytes={offset}-',**{'If-Range':record['etag']})
    count=0;started=time.perf_counter();success=False;response_identity=None
    ledger['inflight']=dict(url=url,etag=record['etag'],bytes_received=0,read_reservation=0,at_utc=now())
    save(OUT/'network_ledger.json',ledger)
    save(receipt,dict(url=url,etag=record['etag'],expected_bytes=size,status='partial'))
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
            response_identity=dict(http_status=response.status,etag=response.headers.get('ETag'),
                content_length=response.headers.get('Content-Length'),content_range=response.headers.get('Content-Range'),
                last_modified=response.headers.get('Last-Modified'))
            if response.headers.get('ETag')!=record['etag']:raise RuntimeError('Archive ETag changed')
            if offset:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{size-1}/{size}':
                    raise RuntimeError('Unverified range resume response')
            elif response.status!=200:raise RuntimeError('Unexpected archive response')
            if response.headers.get('Content-Encoding','identity')!='identity':raise RuntimeError('Unexpected content encoding')
            if int(response.headers['Content-Length'])!=size-offset:raise RuntimeError('Archive Content-Length changed')
            with part.open('ab' if offset else 'wb') as f:
                while offset+count<size:
                    chunk=min(4*2**20,size-offset-count,CAPS['network_bytes']-used-count)
                    if chunk<=0:raise RuntimeError('BLOCKED_RESOURCE: network cap exhausted')
                    ledger['inflight'].update(bytes_received=count,read_reservation=chunk)
                    save(OUT/'network_ledger.json',ledger)
                    block=response.read(chunk)
                    count+=len(block)
                    ledger['inflight'].update(bytes_received=count,read_reservation=0)
                    save(OUT/'network_ledger.json',ledger)
                    if not block:raise RuntimeError('Interrupted archive transfer')
                    f.write(block)
                f.flush()
                import os
                os.fsync(f.fileno())
        if part.stat().st_size!=size:raise RuntimeError('Archive byte count mismatch')
        part.replace(target)
        save(receipt,dict(url=url,etag=record['etag'],expected_bytes=size,sha256=sha(target),status='complete',
            response_identity=response_identity,range_offset=offset))
        success=True
        return target,True
    finally:
        uncertain=ledger['inflight'].get('read_reservation',0)
        ledger['requests'].append(dict(url=url,bytes=count+uncertain,known_response_body_bytes=count,
            uncertain_read_upper_bound_bytes=uncertain,status='complete' if success else 'failed',at_utc=now(),range_offset=offset,
            io_recipe=TRANSFER_IO_RECIPE,response_identity=response_identity))
        ledger['inflight']=None;save(OUT/'network_ledger.json',ledger)
        event('archive_transfer',acquisition_seconds=time.perf_counter()-started,bytes=count,url=url,complete=success,io_recipe=TRANSFER_IO_RECIPE)


def catalog_archive(path,record,wanted):
    """Bounded streaming unpack, no paths/links extracted from untrusted tar."""
    split=record['split'];url=record['url'];key=rank('archive',url)
    catalog=CACHE/'catalogs'/(key+'.json')
    from native_head_confirmation_availability import index
    if catalog.exists():
        old=read(catalog)
        if old['archive_sha256']!=sha(path):raise RuntimeError('Archive catalog identity mismatch')
        valid=next(r['complete'] for r in index(split)['coverage'] if r['url']==url)
        if valid and all((CACHE/'videos'/split/n).exists() for n in wanted if n in {m['filename'] for m in old['members']}):return old
        preserved=CACHE/'catalog_history'/sha(catalog)/catalog.name
        preserved.parent.mkdir(parents=True,exist_ok=True)
        if not preserved.exists():shutil.copyfile(catalog,preserved)
    members=[];archive_sha=sha(path);total=0
    with gzip.open(path,'rb') as raw:
        with tarfile.open(fileobj=raw,mode='r|',bufsize=2**20) as tar:
            for member in tar:
                name=PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or '\\' in member.name or member.issym() or member.islnk():
                    raise RuntimeError('Unsafe archive member')
                if member.isdir():continue
                if not member.isfile() or member.size<0 or member.size>64*2**20:raise RuntimeError('Unsupported/oversized archive member')
                total+=member.size
                if total>64*2**30:raise RuntimeError('Archive expansion bound exceeded')
                basename=name.name
                if not re.fullmatch(r'[A-Za-z0-9_-]{11}_\d{6}_\d{6}\.mp4',basename):raise RuntimeError('Unexpected official archive member: '+member.name)
                source=tar.extractfile(member);h=hashlib.sha256();saved=basename in wanted
                target=CACHE/'videos'/split/basename
                if saved:guard('collection',additional_retained=member.size);target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_name(target.name+'.writing') if saved else None
                f=temp.open('wb') if saved else None
                try:
                    for block in iter(lambda:source.read(2**20),b''):
                        h.update(block)
                        if f:f.write(block)
                finally:
                    if f:f.close()
                digest=h.hexdigest()
                if saved:
                    if target.exists():
                        if sha(target)!=digest:raise RuntimeError('Immutable member bytes changed')
                        temp.unlink()
                    else:temp.replace(target)
                members.append(dict(filename=basename,member=member.name,bytes=member.size,encoded_sha256=digest))
        # Consume gzip footer/CRC as well as all tar entries.
        while raw.read(2**20):pass
    result=dict(url=url,archive_sha256=archive_sha,publisher_digest_available=False,
        enumeration=dict(complete=True,gzip_crc_drained=True,etag=record['etag'],bytes=record['content_length'],
            list_sha256=sha(CACHE/'metadata'/f'{split}_archives')),
        local_integrity_identity=True,members=members,expanded_bytes=total)
    save(catalog,result)
    index(split,refresh=True)
    return result


def acquire(split='val'):
    from run_native_head_value_confirmation import event,network_ledger,CAPS
    from native_head_confirmation_availability import index,publish,membership,remaining_release_transfer
    index(split,refresh=True)
    if split=='test' and not (OUT/'precision_plan.json').exists():raise RuntimeError('BLOCKED_INPUT: fitting-only precision plan required before test acquisition')
    m=8 if split=='val' else read(OUT/'precision_plan.json')['selected_m']
    if m is None:raise RuntimeError('BLOCKED_PRECISION_OR_POPULATION: no allowed m')
    wanted_rows=reserve_members(split,m);wanted={r['filename'] for r in wanted_rows}
    inventory=read(OUT/'archive_inventory.json')
    records_for_split=[r for r in inventory if r['split']==split]
    all_cataloged=index(split)['complete']
    # Gate all remaining shards before a stage; retries are not assumed free.
    remaining=remaining_release_transfer(split)
    used=sum(r['bytes'] for r in network_ledger()['requests'])
    future_test=remaining_release_transfer('test') if split=='val' else 0
    if used+remaining+future_test>CAPS['network_bytes']:raise RuntimeError('BLOCKED_RESOURCE: staged route plus reserved test transfer exceeds remaining transfer')
    save(OUT/f'{split}_acquisition_plan.json',dict(split=split,m=m,reserve_components_per_class=2,
        wanted_rows=len(wanted_rows),selection='fixed canonical/component ranks, independent of archive arrival',
        manifest_sha256=sha(CACHE/'candidate_manifest.json'),new_archive_bytes=remaining,network_already_spent=used,
        reserved_future_test_transfer_bytes=future_test,remaining_retry_bytes=CAPS['network_bytes']-used-remaining-future_test,
        retained_cap_bytes=CAPS['retained_bytes'],free_disk_reserve_bytes=FREE_RESERVE,
        peak_scratch_bytes=max(r['content_length'] for r in records_for_split)+64*2**20,
        plan='Stream one whole gzip archive, catalog/hash every member, retain prospective prefix plus two reserves/class; never delete historical archives. Missing needed reserve triggers separately gated revisit.'))
    with metered('collection','acquire-'+split):
        for record in sorted(records_for_split,key=lambda r:(not bool(r.get('cached_path')),r['url'])):
            if all_cataloged:continue  # Resolve the needed prefix before fetching any new lower-ranked reserve.
            key=rank('archive',record['url']);catalog=CACHE/'catalogs'/(key+'.json')
            if catalog.exists():
                old=read(catalog);listed={r['filename'] for r in old['members']}
                valid=next(r['complete'] for r in index(split)['coverage'] if r['url']==record['url'])
                if valid and all((CACHE/'videos'/split/n).exists() for n in wanted&listed):continue
            print('Acquire/catalog',record['url'],flush=True)
            for attempt in range(3):
                try:
                    path,temporary=download_archive(record)
                    break
                except (urllib.error.URLError,TimeoutError,ConnectionError) as exc:
                    event('archive_acquisition_retry',url=record['url'],attempt=attempt+1,error=str(exc))
                    if attempt==2:raise RuntimeError('BLOCKED_INPUT: public archive remains inaccessible after metered resumable attempts: '+record['url']) from exc
                    time.sleep(2)
            result=catalog_archive(path,record,wanted)
            event('archive_catalog_complete',url=record['url'],archive_sha256=result['archive_sha256'],members=len(result['members']))
            if temporary:
                # Delete only verified task-owned scratch archive, never caches.
                if not path.resolve().is_relative_to((CACHE/'scratch').resolve()):raise RuntimeError('Scratch path escaped task root')
                path.unlink()
    save(OUT/f'{split}_acquisition_receipt.json',dict(catalogs=len(records_for_split),wanted=len(wanted),
        present=sum((CACHE/'videos'/split/n).exists() for n in wanted),no_model_calls=True))
    with metered('cpu','archive-membership-publication'):publish()
    # Newly cataloged encoded identities may change known exclusions/components
    # even when an absent higher-ranked filename itself has no local bytes.
    with metered('cpu','pre-prefix-encoded-identity-join'):rebuild_population()
    requested=set()
    while True:
        try:return certify(split)
        except MissingNeededInput as exc:
            if exc.filename in requested:raise
            requested.add(exc.filename)
            available=membership(split,exc.filename)
            matches=sorted({r['url'] for r in available['origins']})
            if available['state']!='PRESENT':
                save(OUT/f'{split}_unresolved_needed_input.json',dict(filename=exc.filename,split=split,
                    authenticated_complete_catalogs=len(records_for_split),catalog_matches=matches,
                    prerequisite='Resolve UNKNOWN pinned-release membership; verified absence alone excludes a clip.',availability=available,
                    no_replacement_or_model_exposure=True))
                raise
            record=next(r for r in records_for_split if r['url']==matches[0])
            used=sum(r['bytes'] for r in network_ledger()['requests'])
            revisit=0 if record.get('cached_path') else record['content_length']
            if used+revisit+future_test>CAPS['network_bytes']:
                raise RuntimeError('BLOCKED_RESOURCE: necessary deterministic-prefix archive revisit exceeds remaining transfer after reserved test route')
            event('deterministic_prefix_extension',split=split,filename=exc.filename,url=record['url'],
                reason='Higher-ranked technical/identity resolution requires this input; rank and quotas unchanged',new_archive_bytes=revisit)
            # Retain the current deterministic prefix/reserves for every class
            # on this revisit, including alternate clips in multi-class groups.
            wanted.update(r['filename'] for r in reserve_members(split,m))
            wanted.add(exc.filename)
            with metered('collection','needed-prefix-archive-revisit'):
                path,temporary=download_archive(record)
                catalog_archive(path,record,wanted)
                if temporary:
                    if not path.resolve().is_relative_to((CACHE/'scratch').resolve()):raise RuntimeError('Scratch path escaped task root')
                    path.unlink()


def certify(split='val'):
    from native_head_confirmation_runtime import screen_population
    from native_head_confirmation_availability import publish
    # A rank-preserving revisit may adopt the new enumeration schema for a
    # legacy catalog. Publish its actual new hash before copying certificates.
    with metered('cpu','pre-certificate-availability-publication'):publish()
    m=8 if split=='val' else read(OUT/'precision_plan.json')['selected_m']
    visited=set()
    while True:
        identity=sha(CACHE/'candidate_manifest.json')
        if identity in visited:raise RuntimeError('BLOCKED_INPUT: source identity/prefix audit did not converge; no outcomes accessed')
        visited.add(identity)
        with metered('collection','certify-'+split):rows=screen_population(split,m)
        try:return certify_rows(rows,split,m)
        except InputMembershipChanged:
            from run_native_head_value_confirmation import event
            event('input_only_membership_reconciliation',split=split,previous_candidates_sha256=identity,
                updated_candidates_sha256=sha(CACHE/'candidate_manifest.json'),no_model_calls=True)


def observed_identities():
    extra=[]
    for p in sorted((CACHE/'catalogs').glob('*.json')):
        for i,row in enumerate(read(p)['members']):extra.append((row,p.relative_to(ROOT).as_posix(),'/members/'+str(i)))
    for p in sorted((CACHE/'screening').glob('*.json')):
        r=read(p)
        if r.get('eligible'):extra.append((r['row'],p.relative_to(ROOT).as_posix(),'/row'))
    p=CACHE/'input_duplicate_edges.json'
    if p.exists():extra.append((read(p),p.relative_to(ROOT).as_posix(),''))
    return extra


def duplicate_screen():
    """Frozen historical <=4/63-bit rule on three actual 10/50/90% frames."""
    import numpy as np
    identity=read(OUT/'source_audit_identity.json')
    with gzip.open(ROOT/identity['path'],'rt',encoding='utf-8') as f:legacy=json.load(f)['fingerprints']
    unique={}
    for r in legacy:
        key=tuple(r['fingerprints']),r['duration'],tuple(r['nodes'])
        unique[key]=r
    historical=list(unique.values());new=[]
    for p in sorted((CACHE/'screening').glob('*.json')):
        receipt=read(p)
        if receipt.get('eligible'):
            row=receipt['row'];new.append(dict(row,file=p.relative_to(ROOT).as_posix()))
    comparisons=list(historical)
    for r in new:
        for variant,hashes in r['perceptual_fingerprints'].items():
            comparisons.append(dict(nodes=clip_nodes(r['filename']),fingerprints=hashes,duration=r['pts_span'],
                file=r['file'],pointer='/row/perceptual_fingerprints/'+variant))
    bits=np.array([[int(v,2) for v in r['fingerprints']] for r in comparisons],dtype=np.uint64)
    durations=np.array([r['duration'] for r in comparisons])
    lookup=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    edges=[];seen=set()
    for row in new:
        for variant,hashes in row['perceptual_fingerprints'].items():
            values=np.array([int(v,2) for v in hashes],dtype=np.uint64)
            distance=lookup[np.bitwise_xor(bits,values).view(np.uint8).reshape(-1,3,8)].sum(2)
            ratio=np.minimum(durations,row['pts_span'])/np.maximum(durations,row['pts_span'])
            for i in np.flatnonzero((distance<=4).all(1)&(ratio>=.9)):
                other=comparisons[int(i)]
                candidates=[n.split(':',2)[2] for n in other['nodes'] if n.startswith('clip:')]
                if not candidates:continue
                name=candidates[0]
                if name==row['filename']:continue
                pair=tuple(sorted((row['filename'],name)))
                if pair in seen:continue
                seen.add(pair);edges.append(dict(a=pair[0],b=pair[1],reason='frozen input-only three-frame pHash candidate',
                    hamming=distance[i].tolist(),duration_ratio=float(ratio[i]),variant=variant,
                    evidence=[row['file'],other['file']],historical_pointer=other['pointer']))
    save(CACHE/'input_duplicate_edges.json',dict(rule='all three 63-bit DCT hashes <=4 bits, duration ratio >=.9; both actual historical conversion conventions',
        new_sources=len(new),historical_fingerprints=len(historical),edges=edges,model_calls=0))
    save(OUT/'input_duplicate_screen_identity.json',dict(path=(CACHE/'input_duplicate_edges.json').relative_to(ROOT).as_posix(),
        sha256=sha(CACHE/'input_duplicate_edges.json'),new_sources=len(new),historical_fingerprints=len(historical),edges=len(edges),
        unavailable='No perceptual comparison against unavailable K400 training RGB; parent identities still exclude all known train parents.'))


def rebuild_population():
    """Metadata coverage includes every split; locality applies only to prefixes."""
    from run_native_head_value_confirmation import annotations
    from native_head_value_confirmation import class_mapping
    from native_head_confirmation_availability import membership,VERSION
    ann=annotations();mapping=class_mapping(read(CACHE/'metadata/native_classes'),{r['label'] for r in ann['val']})
    audit=historical_audit(observed_identities());g=audit.graph
    train={g.root('yt:'+r['youtube_id']) for r in ann['train']}
    prior={g.root(n) for n in audit.exposed};forbidden={g.root(n) for n in audit.forbidden}
    both={g.root('yt:'+r['youtube_id']) for r in ann['val']}&{g.root('yt:'+r['youtube_id']) for r in ann['test']}
    grouped=defaultdict(list);exclusions=[]
    for split in ('val','test'):
        for i,r in enumerate(ann[split]):
            component=g.root('yt:'+r['youtube_id'])
            reason=('known_train_component' if component in train else 'fit_test_collision' if component in both else
                    'reserved_or_unresolved_component' if component in forbidden else
                    'prior_model_exposure' if split=='test' and component in prior else None)
            filename=f"{r['youtube_id']}_{int(r['time_start']):06d}_{int(r['time_end']):06d}.mp4"
            if reason:
                exclusions.append(dict(split=split,component=component,filename=filename,reason=reason,annotation_row=i));continue
            availability=membership(split,filename)
            if availability['state']=='CONFIRMED_ABSENT':
                exclusions.append(dict(split=split,component=component,filename=filename,reason='archive_absent',annotation_row=i,
                    population_rule=VERSION,evidence=sha(OUT/f'{split}_archive_availability.json.gz')));continue
            grouped[split,component].append(dict(split=split,component=component,youtube_id=r['youtube_id'],label=r['label'],
                y=mapping[r['label']],filename=filename,previously_exposed=component in prior,canonical_rank=rank('canonical',filename)))
    candidates=sorted([r for v in grouped.values() for r in sorted(v,key=lambda r:r['canonical_rank'])],key=lambda r:(r['split'],r['y'],rank('component',r['component']),r['canonical_rank']))
    save(CACHE/'candidate_manifest.json',candidates)
    save(OUT/'candidate_manifest_identity.json',dict(path=(CACHE/'candidate_manifest.json').relative_to(ROOT).as_posix(),sha256=sha(CACHE/'candidate_manifest.json'),rows=len(candidates)))
    save(OUT/'source_exclusion_ledger.json',exclusions)
    counts={s:Counter(min(v,key=lambda r:r['canonical_rank'])['y'] for (sp,c),v in grouped.items() if sp==s) for s in ('val','test')}
    report=dict(metadata_only=True,technically_verified=False,annotation_rows={k:len(v) for k,v in ann.items()},
        candidate_components={s:sum(c.values()) for s,c in counts.items()},per_class={s:{str(y):c[y] for y in range(400)} for s,c in counts.items()},
        minimum_components={s:min(c[y] for y in range(400)) for s,c in counts.items()},
        exclusions=dict(Counter(r['split']+':'+r['reason'] for r in exclusions)),
        known_model_exposed_components=len(prior),reserved_or_unresolved_components=len(forbidden),
        isolation_limit='Known parent/bytes/available frozen duplicate evidence only; unknown reuploads and undocumented pretraining remain.')
    save(OUT/'population_feasibility.json',report)
    old=CACHE/'attempt_history'/ 'd97802601500101a94c91de2a546769e9fae29ab'/ 'data/source/native_head_value_confirmation_v1/candidate_manifest.json'
    if old.exists():
        before={(r['split'],r['component'],r['filename']) for r in read(old)}
        after={(r['split'],r['component'],r['filename']) for r in candidates}
        save(OUT/'population_selection_changes.json',dict(version=VERSION,old_candidates_sha256=sha(old),
            new_candidates_sha256=sha(CACHE/'candidate_manifest.json'),removed=sorted(before-after),added=sorted(after-before),
            selected_prefix_and_reserves='Recomputed by canonical clip rank before class quotas; availability is input-only.'))
    return report


def certify_rows(rows,split,m):
    validate_components(rows,m)
    # Source membership is recomputed after all observed byte and duplicate joins.
    with metered('collection','input-duplicate-screen'):duplicate_screen()
    with metered('cpu','join-input-identities'):rebuild_population()
    graph_identity=read(OUT/'source_audit_identity.json')
    with gzip.open(ROOT/graph_identity['path'],'rt',encoding='utf-8') as f:graph=json.load(f)
    nodes=graph['component_by_node'];forbidden=set(graph['reserved_components'])|set(graph['unresolved_components'])
    prior=set(graph['exposed_components'])
    manifest=[]
    for row in rows:
        component=nodes['yt:'+row['youtube_id']]
        if component in forbidden or (split=='test' and component in prior):
            raise InputMembershipChanged('Newly linked forbidden selected component; rebuild deterministic prefix before outcomes')
        if nodes.get('bytes:'+row['encoded_sha256'])!=component:raise RuntimeError('Selected byte identity not joined to source')
        manifest.append(dict(row,component=component))
    if len({r['component'] for r in manifest})!=len(manifest):
        raise InputMembershipChanged('New duplicate component merges selected fitting inputs; reselect by frozen rank')
    validate_components(manifest,m)
    save(OUT/f'{split}_manifest.json.gz',manifest)
    # Re-run rank/prefix check under the newly joined graph. Cached input receipts
    # are authenticated, and a changed membership invalidates any older plan.
    from native_head_confirmation_runtime import screen_population
    repeated=screen_population(split,m)
    if [(r['component'],r['filename']) for r in manifest]!=[(r['component'],r['filename']) for r in repeated]:
        raise InputMembershipChanged('Selected membership changed after byte graph join; recertify before fitting')
    files=[OUT/'source_audit_identity.json',OUT/'historical_source_evidence.json',OUT/'candidate_manifest_identity.json',
           OUT/'source_exclusion_ledger.json',OUT/'input_duplicate_screen_identity.json',OUT/f'{split}_manifest.json.gz',OUT/f'{split}_prefix_certificate.json']
    files.append(OUT/'archive_inventory.json')
    files += [OUT/'population_rule.json',OUT/'population_amendment.md',OUT/f'{split}_archive_availability.json.gz']
    if split=='test':files.append(OUT/'val_archive_availability.json.gz')
    files.extend([OUT/'class_mapping.json',OUT/'model_identity.json'])
    files.extend(sorted((CACHE/'metadata').glob('*')))
    files.extend(sorted((CACHE/'catalogs').glob('*.json')))
    files.extend(sorted((CACHE/'scratch').rglob('*.transfer.json')))
    scope='fitting' if split=='val' else 'joint'
    if split=='test':
        fit=read(OUT/'val_manifest.json.gz')
        old_prefix=read(OUT/'val_prefix_certificate.json')
        try:current_fit=screen_population('val',8)
        except MissingNeededInput as exc:
            raise PlanningMembershipChanged('New joint input evidence requires a deeper fitting prefix: '+exc.filename) from exc
        finally:
            save(OUT/'val_manifest.json.gz',fit)
            save(OUT/'val_prefix_certificate.json',old_prefix)
        if [(r['component'],r['filename']) for r in current_fit]!=[(r['component'],r['filename']) for r in fit]:
            raise PlanningMembershipChanged('New joint input evidence changed the canonical fitting membership')
        fit_components={nodes['yt:'+r['youtube_id']] for r in fit}
        if len(fit_components)!=len(fit) or fit_components&{r['component'] for r in manifest} or fit_components&forbidden:
            raise PlanningMembershipChanged('New joint input identities invalidate fitting independence or roles')
        files += [OUT/'val_manifest.json.gz',OUT/'val_prefix_certificate.json',OUT/'precision_plan.json']
    certificate=dict(scope=scope,adapter=ADAPTER_VERSION,classes=400,per_class=m,selected_sources=len(manifest),
        input_audit_implementation_sha256=INPUT_CODE_SHA_AT_IMPORT,
        sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        graph_sha256=graph_identity['sha256'],no_selected_unresolved_roles=True,
        residual_limits=['Unknown reuploads may remain.','Unrecorded historical exposure and undocumented checkpoint pretraining cannot be ruled out.',
                        'Perceptual screening is finite-resolution, outcome-independent candidate detection, not universal duplication proof.'])
    old_cert=OUT/f'{scope}_source_certificate.json'
    if old_cert.exists():
        previous=read(old_cert);history=CACHE/'certificate_history'/sha(old_cert)
        if not history.exists():
            history.mkdir(parents=True);mapping={}
            for i,(name,digest) in enumerate(previous['sha256'].items()):
                original=ROOT/name
                if sha(original)!=digest:raise RuntimeError('Prior certificate changed before preservation: '+name)
                target=history/(str(i)+'_'+original.name);shutil.copyfile(original,target)
                mapping[name]=dict(path=target.relative_to(ROOT).as_posix(),sha256=digest)
            shutil.copyfile(old_cert,history/old_cert.name)
            save(OUT/'certificate_history'/(sha(old_cert)+'.json'),dict(original_certificate_sha256=sha(old_cert),
                preserved=mapping,reason='Pre-EVAL input-only re-certification; prior evidence retained without scientific relabeling'))
    snapshot=OUT/'certificates'/scope;snapshot.mkdir(parents=True,exist_ok=True)
    copied={}
    for p in files:
        target=(snapshot/'metadata'/p.name if p.parent==CACHE/'metadata' else
                snapshot/'transfer_receipts'/p.parent.name/p.name if p.name.endswith('.transfer.json') else
                snapshot/'member_catalogs'/p.name if p.parent==CACHE/'catalogs' else snapshot/p.name)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target);copied[target.relative_to(ROOT).as_posix()]=sha(target)
    graph_copy=snapshot/'source_audit_graph.json.gz'
    shutil.copyfile(ROOT/graph_identity['path'],graph_copy)
    candidate_copy=snapshot/'candidate_manifest.json.gz';save(candidate_copy,read(CACHE/'candidate_manifest.json'))
    copied[graph_copy.relative_to(ROOT).as_posix()]=sha(graph_copy)
    copied[candidate_copy.relative_to(ROOT).as_posix()]=sha(candidate_copy)
    certificate.update(sha256=copied,graph_path=graph_copy.relative_to(ROOT).as_posix(),
        candidate_path=candidate_copy.relative_to(ROOT).as_posix(),candidate_uncompressed_sha256=sha(CACHE/'candidate_manifest.json'),
        artifact_directory=snapshot.relative_to(ROOT).as_posix())
    save(OUT/f'{scope}_source_certificate.json',certificate)
    from verify_native_head_value_confirmation import verify_source_certificate
    with metered('cpu','independent-'+scope+'-source-certificate'):
        checked=verify_source_certificate(OUT,scope)
    save(OUT/f'{scope}_source_certificate_verification.json',checked)
    if not checked['passed']:raise RuntimeError('BLOCKED_INPUT: independent source certificate check: '+repr(checked['errors']))
    advance('FIT_CERTIFIED' if split=='val' else 'JOINT_CERTIFIED',[OUT/f'{scope}_source_certificate.json'])
    return manifest
