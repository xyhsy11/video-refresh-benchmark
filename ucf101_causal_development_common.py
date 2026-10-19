"""Isolated exploratory UCF identities and an enforced no-official-test I/O guard."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json
import os
import subprocess
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/ucf101_causal_development_v1'
DATA=ROOT/'data/source/ucf101_causal_development_v1'
OLD=ROOT/'results/ucf101_proxy_replication_v1'
SOURCE=ROOT/'data/source/ucf101_proxy_replication_v1'
SEED=20261019
METHODS=['periodic','RandomB4_seeded','motion','history','history_motion']
TIMING_METHODS=METHODS+['periodic_unused_gate','dense16']
CODE=['ucf101_causal_development_common.py','ucf101_causal_development.py','run_ucf101_causal_development.py',
      'causal_observability_policy.py','causal_observability_learning.py','causal_observability_runtime.py',
      'causal_observability_common.py','ucf101_proxy_inference.py','ucf101_proxy_math.py','ucf101_proxy_common.py',
      'temporal_refresh_common.py','ucf101_proxy_data.py','research_utils.py','benchmark.py']


class Blocker(RuntimeError):pass
def now():return datetime.now(timezone.utc).isoformat()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    import time
    for i in range(100):
        try:tmp.replace(p);return
        except PermissionError:
            if i==99:raise
            time.sleep(.05)
def require(v,message):
    if not v:raise Blocker(message)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def rank(*parts):return hashlib.sha256('|'.join(map(str,[SEED,*parts])).encode()).hexdigest()
def verify(records):
    for p,h in records.items():require((ROOT/p).is_file() and sha(ROOT/p)==h,'Changed/missing identity: '+p)


class InputGuard:
    """Deny exposed outcome paths and all non-whitelisted original RGB/array reads."""
    metadata={'eligible_manifest.json','heads/parameters.json','head_selection.json','train_collection.json',
        'development_collection.json','encoder_identity.json','evaluation_exposure.json','protocol.md',
        'post_exposure_amendment.md','duplicate_and_pretraining_audit.json'}
    def __init__(self):self.entries={};self.feature_clips=set();self.installed=False
    def set_population(self,manifest,diagnostics=False):
        self.entries={v['clip']:v for v in manifest['videos']}
        self.feature_clips={v['clip'] for v in manifest['videos'] if diagnostics or v['study_role']!='execution'}
    def path_check(self,path):
        if isinstance(path,int):return
        q=Path(os.fsdecode(path)).resolve()
        if q.is_relative_to(OLD):
            name=q.relative_to(OLD).as_posix()
            require(name in self.metadata,'Official-test/outcome I/O prohibited: '+name)
        if q.is_relative_to(SOURCE/'features'):
            clip=q.name.removesuffix('.npz')
            require(clip in self.feature_clips,'Unapproved feature access (including execution before diagnostics): '+clip)
        if q.is_relative_to(SOURCE/'videos'):
            require(q.name in self.entries,'Nonstudy/official-test video access prohibited: '+q.name)
            require(q==ROOT/self.entries[q.name]['path'],'Original video path mismatch')
        if q.is_relative_to(ROOT/'data/source/causal_observability_v1/videos'):
            raise Blocker('HMDB video access prohibited in UCF development study')
    def hook(self,event,args):
        if event=='open':
            self.path_check(args[0])
            if not isinstance(args[0],int):
                q=Path(os.fsdecode(args[0])).resolve();mode=args[1] or '';flags=args[2] or 0
                writing=any(x in mode for x in ['w','a','+','x']) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
                if writing and q.is_relative_to(ROOT/'results') and not q.is_relative_to(OUT):
                    raise Blocker('Historical result write prohibited: '+str(q))
    def install(self):
        if not self.installed:sys.addaudithook(self.hook);self.installed=True


GUARD=InputGuard()


def original_train_members():
    with zipfile.ZipFile(SOURCE/'official_splits.zip') as z:
        return {line.split()[0] for line in z.read('ucfTrainTestlist/trainlist01.txt').decode().splitlines() if line.strip()}


def validate_population(manifest):
    members=original_train_members();seen=set();roles=defaultdict(set);hashroles=defaultdict(set)
    require(len(manifest['classes'])==101,'Full101 classes required')
    for v in manifest['videos']:
        require(v['official_role']=='official_train' and v['official_path'] in members,'Official test/foreign video rejected')
        require(v['role'] in ['train','development'],'Invalid inherited role')
        require(v['study_role'] in ['controller_train','selection','execution'],'Unknown study role')
        require(v['clip'] not in seen,'Duplicate clip');seen.add(v['clip'])
        require(v['role']==('train' if v['study_role']=='controller_train' else 'development'),'Role reassignment rejected')
        roles[(v['label'],v['group'])].add(v['study_role']);hashroles[v['sha256']].add(v['study_role'])
    require(all(len(s)==1 for s in roles.values()),'Source group crosses study roles')
    require(all(len(s)==1 for s in hashroles.values()),'Exact duplicate crosses study roles')
    for c in manifest['classes']:
        for r,ng in [('controller_train',2),('selection',1),('execution',None)]:
            items=[v for v in manifest['videos'] if v['label']==c and v['study_role']==r]
            groups=Counter(v['group'] for v in items)
            require(groups and all(n==2 for n in groups.values()),'Two genuine frozen clips/group required')
            require(len(groups)==ng if ng else len(groups)>=1,'Missing fixed class/role group coverage')
    # Existing duplicate graph is input evidence, not model outcomes.
    duplicate=read(OLD/'duplicate_and_pretraining_audit.json')
    parent={v['clip']:v['clip'] for v in manifest['videos']}
    def root(k):
        while parent[k]!=k:k=parent[k]
        return k
    for e in duplicate['edges']:
        if e['a'] in parent and e['b'] in parent:parent[root(e['a'])]=root(e['b'])
    component_roles=defaultdict(set)
    for v in manifest['videos']:component_roles[root(v['clip'])].add(v['study_role'])
    require(all(len(s)==1 for s in component_roles.values()),'Observed duplicate component crosses new roles')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    review=OUT/'repository_review.json'
    if not review.exists():
        save(review,dict(actual_main=git('rev-parse','origin/main'),initial_checkout='81b6cc570890415bf5846232acccab49f0f50706',
            branch=git('branch','--show-current'),historical_result_index=git('ls-files','-s','results').splitlines(),
            historical_code={p:sha(ROOT/p) for p in CODE if not p.startswith('ucf101_causal_') and not p.startswith('run_ucf101_causal_')},
            old_test_exposure_metadata=read(OLD/'evaluation_exposure.json'),
            reviewed_without_test_outcomes=['UCF extraction/head fitting and train/development ledgers, protocol/exposure/amendment',
                'Experiment19 policy/training/runtime/timing code; Experiment19.1 NO_GO findings'],
            forbidden_outcome_files_opened=[],at_utc=now()))
    manifest_path=OUT/'population_manifest.json'
    if manifest_path.exists():manifest=read(manifest_path)
    else:
        old=read(OLD/'eligible_manifest.json');rows=[]
        for c in old['classes']:
            training=[v for v in old['videos'] if v['label']==c and v['role']=='train']
            development=[v for v in old['videos'] if v['label']==c and v['role']=='development']
            tg=sorted({v['group'] for v in training},key=lambda g:(rank('controller_group',c,g),g))[:2]
            dg=sorted({v['group'] for v in development},key=lambda g:(rank('development_group',c,g),g))
            require(len(tg)==2 and len(dg)>=2,'Insufficient input-only fixed groups')
            rows.extend(dict(v,study_role='controller_train') for v in training if v['group'] in tg)
            rows.extend(dict(v,study_role='selection' if v['group']==dg[0] else 'execution') for v in development)
        manifest=dict(classes=old['classes'],videos=rows,seed=SEED,source_manifest_sha256=sha(OLD/'eligible_manifest.json'),
            scope='DEVELOPMENT/EXPLORATORY; all rows belong to official UCF training membership',
            counts={r:dict(clips=sum(v['study_role']==r for v in rows),groups=len({(v['label'],v['group']) for v in rows if v['study_role']==r})) for r in ['controller_train','selection','execution']})
        validate_population(manifest);save(manifest_path,manifest)
        timing=[]
        for c in sorted(manifest['classes'],key=lambda c:rank('timing_class',c))[:30]:
            choices=sorted([v for v in rows if v['label']==c and v['study_role']=='execution'],key=lambda v:rank('timing_clip',v['clip']))
            timing.append(choices[0]['clip'])
        save(OUT/'timing_manifest.json',dict(clips=timing,classes=30,selection='input-only hash; one clip from each of30 distinct classes/groups',warmups=2,repetitions=5))
    validate_population(manifest);GUARD.set_population(manifest)
    selection=read(OLD/'head_selection.json');verify({selection['path']:selection['sha256']})
    payload=read(ROOT/selection['path']);require(payload['classes']==manifest['classes'],'Head class order mismatch')
    save(OUT/'recognition_head_reuse.json',dict(path=selection['path'],sha256=selection['sha256'],classes=payload['classes'],
        fitted_on='Original3020 official-training clips only; normalization and empirical CE class/group/clip weighted',
        development_used_for_head_selection=True,development_rows_used_for_head_parameter_fit=False,
        controller_training_overlaps_head_training=True,independent_head_controller_protocol=False,
        head_iteration_limit=payload['fit']['at_iteration_limit'],exploratory_reuse=True))
    return manifest,{k:np.asarray(v,dtype=float) for k,v in payload['parameters'].items()}


def entry_allowed(v):
    require(v['clip'] in GUARD.entries and GUARD.entries[v['clip']]==v,'Unapproved or mutated source entry')


def raw_video(v):
    entry_allowed(v);raw=(ROOT/v['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==v['sha256'],'Video hash mismatch');return raw


def archived_features(v):
    entry_allowed(v);GUARD.path_check(SOURCE/'features'/f'{v["clip"]}.npz')
    ledger=read(OLD/f'{v["role"]}_collection.json');r=ledger[v['clip']]
    require(r['input_sha256']==v['sha256'] and r['grid_sha256']==digest(v['snippet_grid']),'Archived real feature provenance mismatch')
    require(r['encoder_identity_sha256']==sha(OLD/'encoder_identity.json'),'Original encoder identity mismatch')
    verify({r['path']:r['sha256']})
    with np.load(ROOT/r['path'],allow_pickle=False) as f:a=f['x3d_xs']
    require(a.shape==(16,2048) and a.dtype==np.float32 and np.isfinite(a).all(),'Missing/invalid genuine arrays; no reconstruction permitted')
    return a,r


def seal_check():
    p=OUT/'development_execution_seal.json';require(p.exists(),'Development execution seal missing')
    s=read(p);verify(s['sha256']);return s


def checkpoint(path,identity,compute):
    if Path(path).exists():
        r=read(path);require(r['identity']==identity and digest(r['payload'])==r['payload_sha256'],'Checkpoint changed')
        return r['payload']
    payload=compute();save(path,dict(identity=identity,payload=payload,payload_sha256=digest(payload)));return payload
