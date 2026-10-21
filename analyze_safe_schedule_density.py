"""CPU-only exact schedule-density diagnostic; never invokes historical workflows.

See results/safe_schedule_density_v1/protocol.md. Four public stage flags are
mutually exclusive. Preflight freezes inputs; pilot validates; full resumes
per-clip journals; audit checks sealed results without writing.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd
import refresh_pooling_utils as u

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/safe_schedule_density_v1'
PRIOR = 'results/ucf101_causal_development_v1/'
OBS = 'results/task_utility_observability_v1/'
ORIGINAL = 'results/ucf101_proxy_replication_v1/'
REVIEWED = '6a7c433b8ac9df4ca4fadf12b46d0bf30fcdae9d'
SEED, TOL, REPEATS = 20261021, 1e-10, 2000
EPSILONS = (0., .01, .05)
CODE = ['analyze_safe_schedule_density.py', 'refresh_pooling_utils.py',
        'test_safe_schedule_density.py']
META = [PRIOR+n for n in ['evaluation_manifest.json', 'population_manifest.json',
    'recognition_head_reuse.json', 'dense_development_diagnostics.json',
    'executed_development_rollouts.json']]+[OBS+n for n in ['findings.md',
    'observation_provenance.json', 'per_clip_offline_headroom.csv', 'offline_headroom.json']]+[
    'results/refresh_failure_evidence_v1/summary.md',
    'results/refresh_failure_evidence_v1/experiment_23b_preregistered_protocol.md',
    'results/refresh_failure_evidence_v1/b4_reproduction.json',
    ORIGINAL+'development_collection.json', ORIGINAL+'encoder_identity.json',
    ORIGINAL+'official_metadata_identities.json',
    ORIGINAL+'duplicate_and_pretraining_audit.json',
    'task_utility_observability.py', 'task_utility_observability_study.py']


class InputError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise InputError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, 'Refusing to replace existing artifact: '+str(path))
        return
    # Identical interrupted-stage reports can be resumed; replacements fail.
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_csv(path, rows):
    rows = list(rows)
    require(bool(rows), 'No rows for '+str(path))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=keys)
    writer.writeheader()
    writer.writerows({k: ('N/A' if v is None else v) for k, v in r.items()} for r in rows)
    content = buffer.getvalue()
    path = Path(path)
    if path.exists():
        with path.open(newline='',encoding='utf-8') as f:
            require(f.read() == content, 'Refusing to replace CSV: '+str(path))
        return
    with path.open('x',newline='',encoding='utf-8') as f:
        f.write(content)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


class Guard:
    """Deny nonallowlisted data/results, historical writes, GPU/decoder/network use."""
    def __init__(self, readonly=False):
        self.allowed = {(ROOT/p).resolve() for p in META}
        self.readonly = readonly

    def approve(self, path):
        p = (ROOT/path).resolve()
        require(p.is_relative_to(ROOT), 'Input escapes repository')
        self.allowed.add(p)
        return p

    def check(self, path, writing=False):
        if isinstance(path, int):
            return
        p = Path(os.fsdecode(path)).resolve()
        if writing and p.is_relative_to(ROOT):
            require(not self.readonly and p.is_relative_to(OUT), 'Write prohibited: '+str(p))
            for name in ('pilot','full'):
                require(not (p.is_relative_to(OUT/name) and (OUT/name/'completion.json').exists()),
                        'Completed outputs are read-only: '+str(p))
        if not writing and (p.is_relative_to(ROOT/'data') or p.is_relative_to(ROOT/'results')):
            require(p.is_relative_to(OUT) or p in self.allowed, 'Nonapproved input: '+str(p))

    def hook(self, event, args):
        if event == 'open':
            mode, flags = args[1] or '', args[2] or 0
            self.check(args[0], any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        elif event == 'import':
            require(args[0].split('.')[0] not in {'torch', 'torchvision', 'cupy', 'av', 'cv2', 'sklearn'},
                    'GPU/decoder/fitting import forbidden')
        elif event in {'socket.connect', 'socket.getaddrinfo'}:
            raise InputError('Network forbidden')
        elif event in {'os.remove', 'os.rmdir', 'os.mkdir'}:
            self.check(args[0], True)
        elif event in {'os.rename', 'os.replace'}:
            self.check(args[0], True)
            self.check(args[1], True)
        elif event == 'subprocess.Popen':
            # Windows reports executable=None when Popen resolves a bare name.
            argv = args[1] if isinstance(args[1],(list,tuple)) else args[1].split()
            executable = args[0] or argv[0].strip('"')
            require(Path(executable).stem.lower() == 'git' and len(argv) > 1 and argv[1] in
                {'ls-files','hash-object','diff','log','status','rev-parse'}, 'Only read-only Git subprocesses permitted')


def inventory_file(guard, inventory, path, expected=None):
    p = guard.approve(path)
    require(p.is_file(), 'Missing authenticated input: '+path)
    actual = sha(p)
    require(expected is None or actual == expected, 'Input hash mismatch: '+path)
    inventory[path] = dict(sha256=actual, bytes=p.stat().st_size,
                           authentication='recorded SHA256' if expected else 'tracked snapshot')
    return p


def ce(z, y):
    return np.logaddexp.reduce(np.asarray(z, dtype=np.float64), axis=-1)-z[..., y]


def weights(videos):
    groups, clips = defaultdict(set), Counter()
    classes = {v['label'] for v in videos}
    for v in videos:
        groups[v['label']].add(v['group'])
        clips[v['label'], v['group']] += 1
    w = np.array([1/(len(classes)*len(groups[v['label']])*clips[v['label'], v['group']]) for v in videos])
    require(abs(w.sum()-1) < 1e-12, 'Invalid hierarchical weights')
    return w


def pilot_order(videos, provenance):
    return sorted(videos, key=lambda v: (hashlib.sha256(
        f"{SEED}|{v['sha256']}|{provenance[v['clip']]['original_feature']['sha256']}".encode()).hexdigest(), v['clip']))


def historical_snapshot():
    entries = git('ls-files', '-s', 'results').splitlines()
    return {r.split('\t')[1]: r.split()[1] for r in entries
            if not r.split('\t')[1].startswith('results/safe_schedule_density_v1/')}


def verify_history(snapshot):
    result = subprocess.run(['git', 'hash-object', '--no-filters', '--stdin-paths'], cwd=ROOT,
        input='\n'.join(snapshot)+'\n', text=True, capture_output=True, check=True)
    hashes = result.stdout.splitlines()
    require(len(hashes) == len(snapshot), 'Incomplete historical identity check')
    require(all(a == b for a, b in zip(snapshot.values(), hashes)), 'Historical scientific content changed')
    return dict(passed=True, tracked_result_files=len(snapshot),
                method='Git blob hashes; protected scientific contents not parsed')


def load_inputs(guard):
    inv = {}
    for name in META:
        inventory_file(guard, inv, name)
    evaluation = read(ROOT/(PRIOR+'evaluation_manifest.json'))
    population = read(ROOT/(PRIOR+'population_manifest.json'))
    require(sha(ROOT/(PRIOR+'population_manifest.json')) == evaluation['population_manifest_sha256'], 'Population identity')
    videos = sorted(evaluation['videos'], key=lambda v:v['clip'])
    classes = population['classes']
    require(len(videos) == 400 and len(classes) == 101 and len({(v['label'],v['group']) for v in videos}) == 200,
            'Intended full cohort must be 400 clips/200 groups/101 classes')
    require({v['label'] for v in videos} == set(classes) and len({v['clip'] for v in videos}) == 400, 'Class coverage/duplicate clips')
    official_path = 'data/source/ucf101_proxy_replication_v1/official_splits.zip'
    # Only parse the official-training list, never official-test outcomes/features.
    identities = read(ROOT/(ORIGINAL+'official_metadata_identities.json'))
    split_file = inventory_file(guard, inv, official_path,identities['split_sha256'])
    with zipfile.ZipFile(split_file) as z:
        train_bytes = z.read('ucfTrainTestlist/trainlist01.txt')
        require(hashlib.sha256(train_bytes).hexdigest() == identities['members']['trainlist01.txt'], 'Official training-list hash')
        official_train = {line.split()[0] for line in train_bytes.decode().splitlines()}
    pop = {v['clip']:v for v in population['videos']}
    all_roles = defaultdict(set)
    hashes = defaultdict(set)
    for v in population['videos']:
        all_roles[v['label'],v['group']].add(v['study_role'])
        hashes[v['sha256']].add(v['study_role'])
    require(all(len(s) == 1 for s in [*all_roles.values(),*hashes.values()]), 'Cross-role group/byte overlap')
    duplicate = read(ROOT/(ORIGINAL+'duplicate_and_pretraining_audit.json'))
    parent = {v['clip']:v['clip'] for v in population['videos']}
    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for edge in duplicate['edges']:
        if edge['a'] in parent and edge['b'] in parent:
            parent[root(edge['a'])] = root(edge['b'])
    component_roles = defaultdict(set)
    for v in population['videos']:
        component_roles[root(v['clip'])].add(v['study_role'])
    require(all(len(s) == 1 for s in component_roles.values()), 'Known duplicate component role overlap')
    reuse = read(ROOT/(PRIOR+'recognition_head_reuse.json'))
    head_path = inventory_file(guard, inv, reuse['path'], reuse['sha256'])
    hp = read(head_path)
    require(hp['classes'] == classes == reuse['classes'], 'Head class order mismatch')
    head = {k:np.asarray(a,dtype=np.float64) for k,a in hp['parameters'].items()}
    for key, shape in [('means',(2048,)),('scales',(2048,)),('weights',(101,2048)),('bias',(101,))]:
        require(head[key].shape == shape and np.isfinite(head[key]).all(), 'Head parameters invalid: '+key)
    require((head['scales'] > 0).all(), 'Head scale must be positive')
    require(hp['development_collection_sha256'] == sha(ROOT/(ORIGINAL+'development_collection.json')), 'Head collection lineage')
    provenance = read(ROOT/(OBS+'observation_provenance.json'))
    collection = read(ROOT/(ORIGINAL+'development_collection.json'))
    encoder_hash = sha(ROOT/(ORIGINAL+'encoder_identity.json'))
    encoder = read(ROOT/(ORIGINAL+'encoder_identity.json'))
    require('4 RGB' in encoder['preprocessing'] and '2048' in encoder['feature'], 'Unexpected encoder semantics')
    dense = read(ROOT/(PRIOR+'dense_development_diagnostics.json'))
    rollout = read(ROOT/(PRIOR+'executed_development_rollouts.json'))['records']
    trials, inputs = {}, []
    for v in videos:
        p = pop[v['clip']]
        require(v['study_role'] == p['study_role'] == 'execution' and v['role'] == 'development' and
            v['official_role'] == p['official_role'] == 'official_train' and p['official_path'] in official_train,
            'Forbidden population membership')
        require(all(v[k] == p[k] for k in ['label','group','sha256','path']), 'Manifest row mismatch')
        record = provenance[v['clip']]['original_feature']
        require(record == collection[v['clip']] and record['input_sha256'] == v['sha256'] and
            record['grid_sha256'] == digest(p['snippet_grid']) and record['encoder_identity_sha256'] == encoder_hash,
            'Feature provenance mismatch')
        require(record['encoder_starts'] == record['encoder_completions'] == 16, 'Original candidate calls mismatch')
        grid = p['snippet_grid']
        require(len(grid['frame_indices']) == 16 and all(len(f) == len(set(f)) == 4 for f in grid['frame_indices']), 'Snippet grid invalid')
        require(np.all(np.diff(grid['endpoint_timestamps']) > 0), 'Snippet endpoints not increasing')
        inventory_file(guard, inv, v['path'], v['sha256'])  # Hash bytes only; never decode.
        fp = inventory_file(guard, inv, record['path'], record['sha256'])
        with np.load(fp, allow_pickle=False) as a:
            f = a['x3d_xs'].copy()
        require(f.shape == (16,2048) and f.dtype == np.float32 and np.isfinite(f).all(), 'Invalid genuine features')
        inputs.append(dict(**v, feature=record, snippet_grid_sha256=record['grid_sha256']))
        trials[v['clip']] = {}
        for method in ['dense16','periodic','history','history_motion']:
            tr = dense[v['clip']] if method == 'dense16' else rollout[v['clip']+'/'+method]
            tp = inventory_file(guard, inv, tr['path'], tr['sha256'])
            trial = read(tp)
            payload = trial['payload']
            require(digest(payload) == trial['payload_sha256'], 'Trial payload hash mismatch')
            require(trial['identity']['clip'] == v['clip'] and trial['identity']['input_sha256'] == v['sha256'] and
                    trial['identity']['head_sha256'] == reuse['sha256'], 'Trial identity mismatch')
            require(all(payload[k] == v[k] for k in ['clip','label','group','study_role']) and payload['method'] == method,
                    'Trial membership/method mismatch')
            positions = payload['positions']
            k = 16 if method == 'dense16' else 4
            require(len(positions) == len(set(positions)) == k and positions == sorted(positions) and positions[0] == 0 and
                    positions[-1] < 16 and payload['encoder_starts'] == payload['encoder_completions'] == k, 'Executed exact budget invalid')
            cache = [max(j for j in positions if j <= t) for t in range(16)]
            require(cache == payload['cache_indices'], 'Saved cache trace mismatch')
            ep = inventory_file(guard, inv, payload['features_path'], payload['features_sha256'])
            with np.load(ep, allow_pickle=False) as a:
                actual = a['features'].copy()
            require(np.array_equal(actual, f[positions]), 'Executed features do not match original genuine candidates')
            require(np.asarray(payload['logits']).shape == (101,) and np.isfinite(payload['logits']).all(), 'Saved logits invalid')
            if method == 'periodic':
                require(positions == [0,4,8,12], 'Archived Periodic mismatch')
            trials[v['clip']][method] = dict(positions=positions, logits=payload['logits'],
                prediction=payload['prediction'], trial_path=tr['path'])
    # Only execution records enter reference comparisons or subsequent analysis.
    with (ROOT/(OBS+'per_clip_offline_headroom.csv')).open(newline='', encoding='utf-8') as f:
        refs = {r['clip']:r for r in csv.DictReader(f) if r['role'] == 'execution'}
    require(set(refs) == {v['clip'] for v in videos}, 'Archived execution headroom coverage')
    return dict(videos=videos, inputs=inputs, classes=classes, head=head, head_payload=hp,
                reuse=reuse, provenance=provenance, trials=trials, refs=refs, inventory=inv)


def verify_schedules():
    tables = {k:u.schedules(16,k) for k in (4,8,16)}
    for k,n in [(4,455),(8,6435),(16,1)]:
        t = tables[k]
        require(t['positions'].shape == (n,k) and len(np.unique(t['positions'],axis=0)) == n, 'Enumeration count/duplicates')
        require((t['positions'][:,0] == 0).all() and (np.diff(t['positions'],axis=1) > 0).all(), 'Mandatory0/exact budget')
        for name in ['cache_weights','unique_weights']:
            require(np.allclose(t[name].sum(1),1,atol=1e-14,rtol=0), 'Pooling not normalized')
        periodic = list(range(0,16,16//k))
        idx = int(np.flatnonzero(np.all(t['positions'] == periodic,axis=1))[0])
        np.testing.assert_array_equal(t['cache_weights'][idx],t['unique_weights'][idx])
    return tables


def density_record(z, y, dense_z, epsilon, periodic_id):
    losses = ce(z,y)
    delta = losses-float(ce(dense_z,y))
    pred, dense_pred = z.argmax(1), int(dense_z.argmax())
    correct, preserve = pred == y, delta <= epsilon+TOL
    joint = correct & preserve
    n = len(z)
    periodic_correct = bool(correct[periodic_id])
    improve = (losses <= losses[periodic_id]-.01+TOL) & (correct | (not periodic_correct))
    counts = dict(correct=int(correct.sum()), CE_nondegrade=int(preserve.sum()),
                  correct_and_CE=int(joint.sum()), dense_agreement=int((pred == dense_pred).sum()))
    require(counts['correct_and_CE'] <= min(counts['correct'],counts['CE_nondegrade']), 'Joint exceeds components')
    row = dict(epsilon=epsilon,total=n,dense_correct=bool(dense_pred == y),
        dense_CE=float(ce(dense_z,y)),dense_prediction=dense_pred, RandomExact_CE=float(losses.mean()),
        delta_mean=float(delta.mean()),periodic_correct=periodic_correct,
        periodic_CE=float(losses[periodic_id]),periodic_joint=bool(joint[periodic_id]),
        periodic_delta_CE=float(delta[periodic_id]),any_joint=bool(joint.any()),
        P_safe=float(joint.mean()) if dense_pred == y else None,
        improve_periodic_001_count=int(improve.sum()),improve_periodic_001_fraction=float(improve.mean()))
    for name,count in counts.items():
        row[name+'_count'] = count
        row['p_'+name] = count/n
    for name,val in zip(['min','q05','q25','median','q75','q95','max'],np.quantile(delta,[0,.05,.25,.5,.75,.95,1])):
        row['delta_'+name] = float(val)
    for a in (False,True):
        for b in (False,True):
            row[f'transition_D{int(a)}_S{int(b)}_count'] = int(((dense_pred == y) == a)*np.count_nonzero(correct == b))
    return row, delta, pred


def analyze_clip(v, ctx, tables, scalar=False):
    record = ctx['provenance'][v['clip']]['original_feature']
    with np.load(ROOT/record['path'],allow_pickle=False) as a:
        f = a['x3d_xs'].copy()
    z = u.logits(f,ctx['head_payload']['parameters'])
    y = ctx['classes'].index(v['label'])
    dense_z = z.mean(axis=0)
    rows, methods, arrays = [], [], {}
    errors = dict(archive_logits=0.,archive_CE=0.,scalar_logits=0.,scalar_CE=0.,K16=0.)
    for name in ['cache_weights','unique_weights']:
        errors['K16'] = max(errors['K16'],float(abs(tables[16][name]@z-dense_z).max()))
    for method,trial in ctx['trials'][v['clip']].items():
        cache = [max(j for j in trial['positions'] if j <= t) for t in range(16)]
        actual_z = z[cache].mean(0)
        errors['archive_logits'] = max(errors['archive_logits'],float(abs(actual_z-trial['logits']).max()))
        require(int(actual_z.argmax()) == trial['prediction'], 'Saved prediction mismatch')
    ref = ctx['refs'][v['clip']]
    errors['archive_CE'] = abs(float(ce(dense_z,y))-float(ref['dense_CE']))
    for k in (4,8):
        table = tables[k]
        pid = int(np.flatnonzero(np.all(table['positions'] == list(range(0,16,16//k)),axis=1))[0])
        pooled = {pool:table[name]@z for pool,name in [('C','cache_weights'),('U','unique_weights')]}
        np.testing.assert_allclose(pooled['C'][pid],pooled['U'][pid],atol=TOL,rtol=0)
        for pool,values in pooled.items():
            previous = -1
            for eps in EPSILONS:
                row, delta, pred = density_record(values,y,dense_z,eps,pid)
                require(row['correct_and_CE_count'] >= previous, 'Epsilon monotonicity failed')
                previous = row['correct_and_CE_count']
                row = dict(clip=v['clip'],label=v['label'],group=v['group'],K=k,pool=pool,**row)
                rows.append(row)
                for method in ['periodic']+(['history','history_motion'] if k == 4 else []):
                    positions = list(range(0,16,16//k)) if method == 'periodic' else ctx['trials'][v['clip']][method]['positions']
                    mid = int(np.flatnonzero(np.all(table['positions'] == positions,axis=1))[0])
                    loss = float(ce(values[mid],y))
                    correct = bool(pred[mid] == y)
                    joint = correct and loss-row['dense_CE'] <= eps+TOL
                    methods.append(dict(clip=v['clip'],label=v['label'],group=v['group'],K=k,pool=pool,
                        epsilon=eps,method=method,positions=json.dumps(positions),CE=loss,correct=correct,
                        dense_correct=row['dense_correct'],CE_nondegrade=bool(loss-row['dense_CE'] <= eps+TOL),
                        joint=bool(joint),any_joint=row['any_joint'],missed_safe=bool(not joint and row['any_joint']),
                        dense_agreement_fidelity_only=bool(pred[mid] == row['dense_prediction']),
                        interpretation='archived executed B4+C replay' if k == 4 and pool == 'C' else 'offline re-pooling/enumeration'))
            arrays[f'delta_K{k}_{pool}'] = delta
            arrays[f'prediction_K{k}_{pool}'] = pred.astype(np.uint8)
            if scalar:
                for sid in sorted(set([0,len(values)//2,len(values)-1,pid])):
                    pos = table['positions'][sid].tolist()
                    selected, accum = 0,np.zeros(2048,dtype=np.float64)
                    if pool == 'C':
                        for t in range(16):
                            if t in pos:
                                selected = t
                            accum += f[selected].astype(np.float64)
                        raw_z = u.logits(accum/16,ctx['head_payload']['parameters'])
                    else:
                        for t in pos:
                            accum += f[t].astype(np.float64)
                        raw_z = u.logits(accum/k,ctx['head_payload']['parameters'])
                    errors['scalar_logits'] = max(errors['scalar_logits'],float(abs(raw_z-values[sid]).max()))
                    errors['scalar_CE'] = max(errors['scalar_CE'],abs(float(ce(raw_z,y))-float(ce(values[sid],y))))
            if k == 4 and pool == 'C':
                errors['archive_CE'] = max(errors['archive_CE'],abs(float(ce(values[pid],y))-float(ref['periodic_CE'])),
                    abs(float(ce(values,y).min())-float(ref['task_oracle_CE'])))
    require(max(errors.values()) <= TOL, 'Real feature/archive verification failed: '+str(errors))
    return dict(rows=rows,methods=methods,verification=dict(clip=v['clip'],scalar_raw_features_checked=scalar,**errors)),arrays


def bootstrap(videos):
    groups = sorted({(v['label'],v['group']) for v in videos})
    rng = np.random.Generator(np.random.PCG64(SEED))
    blocks = []
    for c in sorted({c for c,g in groups}):
        ids = np.array([i for i,(label,g) in enumerate(groups) if label == c])
        blocks.append(ids[rng.integers(len(ids),size=(REPEATS,len(ids)))])
    draws = np.concatenate(blocks,axis=1).astype(np.int32)
    ids = np.array([groups.index((v['label'],v['group'])) for v in videos])
    counts = np.stack([np.bincount(d,minlength=len(groups)) for d in draws])
    bw = weights(videos)[None,:]*counts[:,ids]
    require(np.allclose(bw.sum(1),1,atol=1e-12,rtol=0), 'Bootstrap not normalized')
    return groups,draws,bw


def weighted_quantiles(values,w):
    if w.sum() == 0:
        return {name:None for name in ['min','q05','q25','median','q75','q95','max']}
    supported = w > 0
    values,w = values[supported],w[supported]
    order = np.argsort(values,kind='stable')
    a,cdf = values[order],np.cumsum(w[order])/w.sum()
    return {name:float(a[min(int(np.searchsorted(cdf,q,side='left')),len(a)-1)])
        for name,q in zip(['min','q05','q25','median','q75','q95','max'],[0,.05,.25,.5,.75,.95,1])}


def estimate(values,mask,w,bw):
    mask = np.asarray(mask,dtype=bool)
    mass = float(w@mask)
    result = dict(estimate=None if mass == 0 else float(w@(values*mask)/mass),weighted_mass=mass,
                  CI95_lower=None,CI95_upper=None,valid_bootstrap_draws=0)
    if bw is not None:
        denominator = bw@mask.astype(float)
        valid = denominator > 0
        result['valid_bootstrap_draws'] = int(valid.sum())
        if valid.any():
            samples = (bw@(values*mask))[valid]/denominator[valid]
            result['CI95_lower'],result['CI95_upper'] = map(float,np.percentile(samples,[2.5,97.5],method='linear'))
    return result


def summarize(stage,videos,records,full):
    rows = [r for record in records for r in record['rows']]
    methods = [r for record in records for r in record['methods']]
    w = weights(videos)
    bw = None
    if full:
        groups,draws,bw = bootstrap(videos)
        save(stage/'bootstrap_groups.json',dict(groups=groups,seed=SEED,repeats=REPEATS,rng='PCG64',
            single_group_classes=[c for c in sorted({c for c,g in groups}) if sum(cc == c for cc,g in groups) == 1],
            interval_scope='marginal unadjusted; exposed cohort/reused selected head; no within-class variance for single-group classes'))
        bp = stage/'bootstrap_group_indices.npy'
        if bp.exists():
            np.testing.assert_array_equal(np.load(bp,allow_pickle=False),draws)
        else:
            with bp.open('xb') as f:
                np.save(f,draws,allow_pickle=False)
    summary,rescue,method_summary,transitions,paired = [],[],[],[],[]
    grouped = []
    for k in (4,8):
        for pool in ('C','U'):
            for eps in EPSILONS:
                a = [r for r in rows if (r['K'],r['pool'],r['epsilon']) == (k,pool,eps)]
                require([r['clip'] for r in a] == [v['clip'] for v in videos], 'Pairing/order mismatch')
                base = dict(K=k,pool=pool,epsilon=eps)
                dc = np.array([r['dense_correct'] for r in a])
                allmask = np.ones(len(a),dtype=bool)
                metrics = ['p_correct','p_CE_nondegrade','p_correct_and_CE','p_dense_agreement',
                    'RandomExact_CE','delta_mean','improve_periodic_001_fraction']
                strata = [('all',allmask),('Dense_correct',dc),('Dense_wrong',~dc)]
                for stratum,mask in strata:
                    for metric in metrics:
                        values = np.array([r[metric] for r in a],dtype=float)
                        summary.append(dict(**base,stratum=stratum,metric=metric,clips=int(mask.sum()),
                            groups=len({(v['label'],v['group']) for v,m in zip(videos,mask) if m}),
                            interpretation='fidelity only' if metric == 'p_dense_agreement' else
                            'correction / preservation; P_safe N/A' if stratum == 'Dense_wrong' else
                            'conditional P_safe' if stratum == 'Dense_correct' and metric == 'p_correct_and_CE' else 'task / CE diagnostic',
                            **estimate(values,mask,w,bw)))
                    if stratum == 'Dense_correct':
                        vals = np.array([r['p_correct_and_CE'] for r in a])
                        summary.append(dict(**base,stratum=stratum,metric='P_safe',clips=int(mask.sum()),
                            groups=len({(v['label'],v['group']) for v,m in zip(videos,mask) if m}),
                            interpretation='conditional P_safe',**estimate(vals,mask,w,bw)))
                # Four full-population correctness transitions, not conditioning away Dense errors.
                pc = np.array([r['p_correct'] for r in a])
                for d in (0,1):
                    for s in (0,1):
                        values = (dc == d)*(pc if s else 1-pc)
                        transitions.append(dict(**base,method='RandomExact',dense_correct=d,schedule_correct=s,
                            **estimate(values,allmask,w,bw)))
                fail = dc & ~np.array([r['periodic_joint'] for r in a])
                anysafe = np.array([r['any_joint'] for r in a])
                density = np.array([r['p_correct_and_CE'] for r in a])
                rb = dict(**base,clips=int(fail.sum()),groups=len({(v['label'],v['group']) for v,m in zip(videos,fail) if m}),
                    weighted_mass=float(w@fail),safe_alternative_clips=int((fail&anysafe).sum()),
                    no_safe_clips=int((fail&~anysafe).sum()),safe_schedule_success_total=int(sum(r['correct_and_CE_count'] for r,m in zip(a,fail) if m)),
                    schedule_total=int(fail.sum()*a[0]['total']),distribution=weighted_quantiles(density,w*fail))
                for name,values,mask in [('any_safe_fraction',anysafe,fail),('no_safe_fraction',~anysafe,fail),
                    ('safe_density',density,fail),('whole_cohort_rescuable_mass',fail&anysafe,allmask),
                    ('periodic_failure_mass',fail,allmask)]:
                    rb[name] = estimate(values.astype(float),mask,w,bw)
                rescue.append(rb)
                for level in ['group','class']:
                    keys = sorted({(v['label'],v['group'] if level == 'group' else '') for v in videos})
                    for c,g in keys:
                        mask = np.array([v['label'] == c and (level == 'class' or v['group'] == g) for v in videos])
                        for metric in metrics:
                            values = np.array([r[metric] for r in a],dtype=float)
                            grouped.append(dict(**base,level=level,label=c,group=g,metric=metric,clips=int(mask.sum()),
                                weighted_mass=float(w@mask),estimate=float(w@(values*mask)/(w@mask))))
                candidates = ['periodic']+(['history','history_motion'] if k == 4 else [])
                method_arrays = {}
                for method in candidates:
                    b = [r for r in methods if (r['K'],r['pool'],r['epsilon'],r['method']) == (k,pool,eps,method)]
                    require([r['clip'] for r in b] == [v['clip'] for v in videos], 'Method pairing mismatch')
                    method_arrays[method] = b
                    for stratum,mask in strata:
                        for metric in ['CE','correct','CE_nondegrade','joint','missed_safe','dense_agreement_fidelity_only']:
                            values = np.array([r[metric] for r in b],dtype=float)
                            method_summary.append(dict(**base,method=method,stratum=stratum,metric=metric,
                                clips=int(mask.sum()),groups=len({(v['label'],v['group']) for v,m in zip(videos,mask) if m}),
                                positive_clips=int(np.count_nonzero(values[mask])) if metric != 'CE' else None,
                                **estimate(values,mask,w,bw)))
                        failures = mask & ~np.array([r['joint'] for r in b])
                        method_summary.append(dict(**base,method=method,stratum=stratum,metric='missed_safe_given_failure',
                            clips=int(failures.sum()),groups=len({(v['label'],v['group']) for v,m in zip(videos,failures) if m}),
                            positive_clips=int(sum(r['missed_safe'] for r,m in zip(b,failures) if m)),
                            **estimate(np.array([r['missed_safe'] for r in b],dtype=float),failures,w,bw)))
                    for d in (0,1):
                        for s in (0,1):
                            vals = np.array([r['dense_correct'] == d and r['correct'] == s for r in b],dtype=float)
                            transitions.append(dict(**base,method=method,dense_correct=d,schedule_correct=s,
                                **estimate(vals,allmask,w,bw)))
                for candidate,reference in [('history','periodic'),('history_motion','periodic'),('history_motion','history')]:
                    if candidate not in method_arrays:
                        continue
                    for metric in ['CE','correct','joint','missed_safe']:
                        vals = np.array([float(x[metric])-float(y[metric]) for x,y in zip(method_arrays[candidate],method_arrays[reference])])
                        paired.append(dict(**base,candidate=candidate,reference=reference,metric=metric,
                            orientation='candidate minus reference',**estimate(vals,allmask,w,bw)))
    write_csv(stage/'per_video.csv',rows)
    write_csv(stage/'frozen_methods_per_video.csv',methods)
    write_csv(stage/'summaries.csv',summary)
    write_csv(stage/'grouped_summaries.csv',grouped)
    write_csv(stage/'frozen_method_comparisons.csv',method_summary)
    write_csv(stage/'paired_method_contrasts.csv',paired)
    write_csv(stage/'correctness_transitions.csv',transitions)
    save(stage/'periodic_rescue.json',dict(rows=rescue,conditioning='Dense correct AND Periodic fails joint',
        density_distribution='inverse weighted CDF across qualifying clips; includes zero-density clips',
        empty='null / N/A; not zero'))
    display_ids = read(OUT/'pilot_manifest.json')['clips']
    display = [r for clip in display_ids for r in rows if r['clip'] == clip and (r['K'],r['pool'],r['epsilon']) == (4,'C',0.)]
    write_csv(stage/'display_20_clips.csv',display)
    text = '|Clip|Dense correct|Periodic joint|Exact joint count|Joint density|\n|---|---:|---:|---:|---:|\n'
    for r in display:
        text += f"|{r['clip']}|{r['dense_correct']}|{r['periodic_joint']}|{r['correct_and_CE_count']}/455|{r['p_correct_and_CE']:.6f}|\n"
    (stage/'display_20_clips.md').write_text('# Input-hash-selected technical display (not a representative subset)\n\n'+text,encoding='utf-8')
    return rows,summary,rescue,method_summary


def make_figures(stage,rows,videos):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    w = weights(videos)
    fig,ax = plt.subplots(figsize=(8.4,5),layout='constrained')
    for k,pool,style in [(4,'C','-'),(8,'C','-'),(4,'U','--'),(8,'U','--')]:
        subset = [r for r in rows if (r['K'],r['pool'],r['epsilon']) == (k,pool,0.)]
        v = np.array([r['p_correct_and_CE'] for r in subset])
        order = np.argsort(v,kind='stable')
        ax.step(np.r_[0,v[order],1],np.r_[0,np.cumsum(w[order]),1],where='post',linestyle=style,label=f'K{k}, {pool}')
    ax.set(xlabel='Exact joint schedule fraction per clip (all clips)',ylabel='Hierarchically weighted clip CDF',
        xlim=(0,1),ylim=(0,1),title='EXPLORATORY / exposed UCF development\nExact schedule density, epsilon=0 (agreement is not safety)')
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(stage/'figure_1_joint_density.png',dpi=170);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(10,4.7),layout='constrained')
    a = [r for r in rows if (r['K'],r['pool'],r['epsilon']) == (4,'C',0.)]
    fail = np.array([r['dense_correct'] and not r['periodic_joint'] for r in a])
    dens = np.array([r['p_correct_and_CE'] for r in a])
    axes[0].scatter([r['periodic_delta_CE'] for r,m in zip(a,fail) if m],dens[fail],s=16,alpha=.7)
    axes[0].set(xlabel='Periodic CE minus Dense16 CE (nats)',ylabel='Exact safe-schedule fraction',ylim=(-.02,1.02),
        title=f'Dense-correct Periodic failures (n={fail.sum()})')
    bins = np.array([0,1e-15,.01,.1,.25,.5,.75,1.00000001])
    masses = np.histogram(dens[fail],bins=bins,weights=w[fail])[0]
    if fail.any():
        masses /= w[fail].sum()
    axes[1].bar(range(7),masses)
    axes[1].set(xticks=range(7),xticklabels=['0','(0,.01)','[.01,.1)','[.1,.25)','[.25,.5)','[.5,.75)','[.75,1]'],
        ylabel='Conditional weighted clip mass',title='Safe-alternative density distribution')
    axes[1].tick_params(axis='x',rotation=35)
    fig.suptitle('EXPLORATORY / exposed UCF development: B4+C, epsilon=0\nOffline alternatives; no causal controller or runtime claim')
    fig.savefig(stage/'figure_2_periodic_rescue.png',dpi=170);plt.close(fig)


def seal(directory):
    files = {p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'completion.json'}
    save(directory/'completion.json',dict(status='COMPLETE_EXPOSED_DEVELOPMENT' if directory.name == 'full' else 'PARTIAL_PILOT',
        completed_at_utc=now(),sha256=files,read_only=True))


def verify_seal(directory):
    c = read(directory/'completion.json')
    for p,h in c['sha256'].items():
        require(sha(directory/p) == h,'Completed output changed: '+p)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name != 'completion.json'}
    require(actual == set(c['sha256']), 'Completed output membership changed')
    return c


def preflight(guard,ctx):
    snapshot = historical_snapshot()
    history = verify_history(snapshot)
    require(git('diff','--name-only',REVIEWED,'HEAD') == '', 'Unexpected changes relative to reviewed main; inspect before freezing')
    ordered = pilot_order(ctx['videos'],ctx['provenance'])
    pilot,classes = [],set()
    for v in ordered:
        if v['label'] not in classes:
            pilot.append(v['clip']);classes.add(v['label'])
        if len(pilot) == 20:
            break
    require(len(pilot) == 20,'Cannot form 20-class technical pilot')
    save(OUT/'cohort_manifest.json',dict(clips=ctx['inputs'],classes=ctx['classes'],intended_clips=400,intended_groups=200,
        weights=weights(ctx['videos']).tolist(),scope='execution only / official training / exposed development'))
    save(OUT/'pilot_manifest.json',dict(clips=pilot,selection='input-only SHA256 ranking; greedily distinct classes',
        ranked_clips=[v['clip'] for v in ordered],seed=SEED,groups=20,classes=20))
    save(OUT/'input_inventory.json',ctx['inventory'])
    save(OUT/'head_inspection.json',dict(path=ctx['reuse']['path'],sha256=ctx['reuse']['sha256'],classes=ctx['classes'],
        parameter_shapes={k:list(v.shape) for k,v in ctx['head'].items()},fit_metadata=ctx['head_payload']['fit'],
        reused_head_limitations=ctx['reuse'],fitting_performed=False))
    save(OUT/'preflight.json',dict(status='PREFLIGHT_PASSED',at_utc=now(),actual_HEAD=git('rev-parse','HEAD'),reviewed_main=REVIEWED,
        origin_main_at_freeze=git('rev-parse','origin/main'),
        changes_reviewed_main_to_HEAD=git('diff','--stat',REVIEWED,'HEAD'),
        later_origin_main_commits=git('log','--oneline',REVIEWED+'..origin/main'),
        git_status_at_freeze=git('status','--porcelain=v1'),historical_snapshot=snapshot,historical_verification=history,
        intended_clips=400,intended_groups=200,classes=101,new_encoder_calls=0,fits=0,decodes=0,downloads=0,
        environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,CPU_threads=1),
        gates='23A NO_GO/BLOCKED_INPUT; 23B frozen/unexecuted'))
    paths = CODE+['results/safe_schedule_density_v1/'+n for n in ['protocol.md','cohort_manifest.json','pilot_manifest.json',
        'input_inventory.json','head_inspection.json','preflight.json']]
    save(OUT/'design_seal.json',dict(at_utc=now(),sha256={p:sha(ROOT/p) for p in paths},
        seed=SEED,epsilon=list(EPSILONS),numerical_tolerance=TOL,bootstrap_draws=REPEATS,
        primary='full-population p_correct_and_CE, C K4 epsilon0',outcomes_computed=False))


def check_design(ctx):
    sealdata = read(OUT/'design_seal.json')
    for p,h in sealdata['sha256'].items():
        require(sha(ROOT/p) == h,'Frozen design changed; new version required: '+p)
    require(ctx['inventory'] == read(OUT/'input_inventory.json'),'Frozen input inventory changed')
    require(ctx['inputs'] == read(OUT/'cohort_manifest.json')['clips'],'Frozen cohort changed')
    return verify_history(read(OUT/'preflight.json')['historical_snapshot'])


def run_stage(ctx,full):
    stage = OUT/('full' if full else 'pilot')
    if (stage/'completion.json').exists():
        verify_seal(stage)
        print('Completed outputs are read-only; use --audit-only.',flush=True)
        return
    if full:
        require((OUT/'pilot/completion.json').is_file(),'Run and validate --pilot first')
        verify_seal(OUT/'pilot')
    stage.mkdir(parents=True,exist_ok=True)
    (stage/'clips').mkdir(exist_ok=True)
    pilot = read(OUT/'pilot_manifest.json')['clips']
    scalar = set(pilot[:3])
    videos = [v for v in ctx['videos'] if full or v['clip'] in pilot]
    tables = verify_schedules()
    records = []
    start = time.perf_counter()
    for i,v in enumerate(videos):
        path = stage/'clips'/(v['clip']+'.json')
        array_path = path.with_suffix('.npz')
        if path.exists():
            journal = read(path)
            require(journal['design_sha256'] == sha(OUT/'design_seal.json') and journal['arrays_sha256'] == sha(array_path), 'Journal authentication failed')
            records.append(journal['record'])
        else:
            record,arrays = analyze_clip(v,ctx,tables,v['clip'] in scalar)
            # If interrupted between these two writes, retain the identical array
            # and verify it rather than replacing any prior data.
            if array_path.exists():
                with np.load(array_path,allow_pickle=False) as a:
                    require(set(a.files) == set(arrays) and all(np.array_equal(a[k],arrays[k]) for k in arrays),'Interrupted array mismatch')
            else:
                with array_path.open('xb') as f:
                    np.savez_compressed(f,**arrays)
            save(path,dict(record=record,arrays_sha256=sha(array_path),design_sha256=sha(OUT/'design_seal.json')))
            records.append(record)
        if (i+1)%20 == 0:
            print(f'{stage.name}: exact enumeration {i+1}/{len(videos)}',flush=True)
    # Summaries are generated only after all predeclared stage clips succeed.
    # A partial report interrupted mid-generation must be audited/recovered in a
    # new output version; completed clip journals remain exact resumable evidence.
    rows,summary,rescue,methods = summarize(stage,videos,records,full)
    make_figures(stage,rows,videos)
    errs = {k:max(r['verification'][k] for r in records) for k in ['archive_logits','archive_CE','scalar_logits','scalar_CE','K16']}
    archived = []
    if full:
        ref = next(r for r in read(ROOT/(OBS+'offline_headroom.json'))['roles'] if r['role'] == 'execution')
        w = weights(videos)
        a = [r for r in rows if (r['K'],r['pool'],r['epsilon']) == (4,'C',0.)]
        for key,field in [('periodic_CE','periodic_CE'),('dense_CE','dense_CE')]:
            value = float(w@np.array([r[field] for r in a]))
            require(abs(value-ref[key]) < TOL,'Full-precision archived aggregate mismatch')
            archived.append(dict(metric=key,replayed=value,archived=ref[key]))
        # Oracle values already reproduced clipwise; aggregate is verification only.
        value = float(w@np.array([float(ctx['refs'][v['clip']]['task_oracle_CE']) for v in videos]))
        require(abs(value-ref['task_oracle_CE']) < TOL,'Archived oracle aggregate mismatch')
        archived.append(dict(metric='task_oracle_CE_verification_only',replayed=value,archived=ref['task_oracle_CE']))
        old = read(ROOT/'results/refresh_failure_evidence_v1/b4_reproduction.json')['methods']
        for r in old:
            q = [x for x in methods if (x['K'],x['pool'],x['epsilon'],x['method'],x['stratum'],x['metric']) == (4,'C',0.,r['method'],'all','CE')][0]
            require(abs(q['estimate']-r['balanced_CE']) < TOL,'Frozen method archive CE mismatch')
    save(stage/'verification.json',dict(passed=True,clips=len(videos),enumerated_schedules={4:455,8:6435},
        maximum_errors=errs,raw_scalar_clips=[r['verification']['clip'] for r in records if r['verification']['scalar_raw_features_checked']],
        checks=['mandatory0/exact budgets/no duplicates','K16 reconstructs Dense16','Periodic C=U',
                'archived full-precision per-clip Dense16/Periodic/B4 oracle','saved B4 H/H+M logits/predictions/features',
                'independent scalar raw-feature last-cache and U pooling','joint <= correctness and CE densities',
                'epsilon monotonicity; no K8>=K4 assumption'],archived_aggregates=archived,
        historical_files=verify_history(read(OUT/'preflight.json')['historical_snapshot'])))
    accounting = dict(status='COMPLETE_EXPOSED_DEVELOPMENT' if full else 'PARTIAL_PILOT',
        clips=len(videos),groups=len({(v['label'],v['group']) for v in videos}),classes=len({v['label'] for v in videos}),
        new_encoder_calls=0,fits=0,decodes=0,downloads=0,GPU_initializations=0,CPU_analysis_seconds_this_invocation=time.perf_counter()-start,
        timing_boundary='CPU enumeration/reporting this invocation; excludes input preflight; NOT inference latency',
        bootstrap_draws=REPEATS if full else 0,synthetic_rows_in_empirical_tables=0,
        gates='23A NO_GO/BLOCKED_INPUT preserved; 23B frozen/unexecuted')
    if not (stage/'accounting.json').exists():
        save(stage/'accounting.json',accounting)
    write_findings(stage,summary,rescue,methods,full)
    seal(stage)
    print(f'{stage.name} sealed: '+str(stage),flush=True)


def write_findings(stage,summary,rescue,methods,full):
    def s(metric,stratum='all'):
        return next(r for r in summary if (r['K'],r['pool'],r['epsilon'],r['stratum'],r['metric']) == (4,'C',0.,stratum,metric))
    def m(method,metric):
        return next(r for r in methods if (r['K'],r['pool'],r['epsilon'],r['stratum'],r['method'],r['metric']) == (4,'C',0.,'all',method,metric))
    def fmt(r):
        if r['estimate'] is None:
            return 'N/A'
        text = f"{100*r['estimate']:.2f}%"
        if full:
            text += f" [95% marginal interval {100*r['CI95_lower']:.2f}%, {100*r['CI95_upper']:.2f}%]"
        return text
    r = next(r for r in rescue if (r['K'],r['pool'],r['epsilon']) == (4,'C',0.))
    text = f"""# {'Completed exposed-development diagnostic' if full else 'PARTIAL_PILOT — technical validation only'}

Experiment 23A **NO_GO / BLOCKED_INPUT** and frozen, unexecuted 23B are preserved.
{'All 400 predeclared execution clips / 200 groups / 101 classes were processed.' if full else 'Only the 20 input-selected technical clips were processed. These are not population conclusions and have no population confidence intervals.'}
All percentages below use equal-class, equal-group-within-class, equal-clip weights.
Primary endpoint is B4+C, epsilon0; tolerance 1e-10 is separate.

1. **Density:** full-population exact joint fraction averages {fmt(s('p_correct_and_CE'))}.
   Dense-correct conditional P_safe averages {fmt(s('P_safe','Dense_correct'))}.
   These are densities under uniform schedule enumeration, not best-subset existence.
2. **Periodic adequacy:** Periodic passes joint on {fmt(m('periodic','joint'))} of weighted clips;
   accuracy alone is {fmt(m('periodic','correct'))}. CE nondegradation and correctness differ.
3. **Periodic failures:** {r['clips']} Dense-correct clips in {r['groups']} groups
   ({100*r['weighted_mass']:.2f}% of whole-cohort weight) fail Periodic joint.
   Of this conditional mass, {fmt(r['any_safe_fraction'])} has a safe alternative;
   {fmt(r['no_safe_fraction'])} has none. Exact density averages {fmt(r['safe_density'])},
   with weighted clip-density median {r['distribution']['median']} and quartiles
   {r['distribution']['q25']}, {r['distribution']['q75']} (zeros included).
   Whole-cohort mass of such rescuable Periodic failures: {fmt(r['whole_cohort_rescuable_mass'])}.
4. **Frozen methods:** missed_safe whole-cohort mass is {fmt(m('history','missed_safe'))}
   for H and {fmt(m('history_motion','missed_safe'))} for H+M. This means method joint
   failure with a passing same-K/C/epsilon alternative; actual comparisons apply
   only to archived executed B4+C. U is offline re-pooling; K8 H/H+M is N/A.
5. **Unknowns:** label/future-informed density cannot establish a usable causal
   pre-decision signal, fresh-population generalization, or complete execution
   cost. The reused recognition head was development-selected and reached its
   historical iteration limit; controller/head training overlap remains. Unknown
   parent/reupload/pretraining relationships and single-group classes limit
   independence and uncertainty. CPU diagnostic time is not inference latency.
   Offline alternatives alone do not justify fitting another controller.

Dense-wrong clips remain in all-population summaries; correction and CE
preservation are separate rows, conditional P_safe is N/A. Dense agreement is
fidelity only. RandomExact averages individual schedule losses/correctness.
The fixed 0.01-nat Periodic-improvement fraction is {fmt(s('improve_periodic_001_fraction'))}.
Sensitivity rows for K8, U and epsilon .01/.05 are retained without selecting a
favorable result. No K8>=K4 safety assumption is imposed. All four correctness
transitions, paired contrasts, group/class tables and exact distributions are saved.

{'Intervals use 2000 paired PCG64 class-stratified group draws, seed20261021. They are marginal/unadjusted and conditional on reused inputs/head. Empty bootstrap conditioning sets are omitted with valid draw counts. Single-group classes have no within-class resampling variance.' if full else 'No bootstrap population intervals are produced for this technical pilot.'}
New encoder calls=0; fits=0; decodes=0; downloads=0; GPU initializations=0.
"""
    (stage/'findings.md').write_text(text,encoding='utf-8')


def audit(ctx):
    check_design(ctx)
    tables = verify_schedules()
    for stage in [OUT/'pilot',OUT/'full']:
        if not (stage/'completion.json').exists():
            continue
        verify_seal(stage)
        videos = [v for v in ctx['videos'] if stage.name == 'full' or v['clip'] in read(OUT/'pilot_manifest.json')['clips']]
        scalar = set(read(OUT/'pilot_manifest.json')['clips'][:3])
        for i,v in enumerate(videos):
            expected,arrays = analyze_clip(v,ctx,tables,v['clip'] in scalar)
            path = stage/'clips'/(v['clip']+'.json')
            journal = read(path)
            require(journal['record'] == expected,'Read-only independent replay differs: '+v['clip'])
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
                require(set(a.files) == set(arrays) and all(np.array_equal(a[k],arrays[k]) for k in arrays),'Exact arrays changed')
            if (i+1)%50 == 0:
                print(f'audit {stage.name}: {i+1}/{len(videos)}',flush=True)
        # Reproduce stored bootstrap pairing independently from the clip journals.
        if stage.name == 'full':
            groups,draws,bw = bootstrap(videos)
            require(read(stage/'bootstrap_groups.json')['groups'] == [list(g) for g in groups],'Bootstrap group ordering')
            np.testing.assert_array_equal(np.load(stage/'bootstrap_group_indices.npy',allow_pickle=False),draws)
            records = [read(stage/'clips'/(v['clip']+'.json'))['record'] for v in videos]
            a = [r for rec in records for r in rec['rows'] if (r['K'],r['pool'],r['epsilon']) == (4,'C',0.)]
            with (stage/'summaries.csv').open(newline='',encoding='utf-8') as f:
                stored = next(r for r in csv.DictReader(f) if (r['K'],r['pool'],r['epsilon'],r['stratum'],r['metric']) == ('4','C','0.0','all','p_correct_and_CE'))
            actual = estimate(np.array([r['p_correct_and_CE'] for r in a]),np.ones(len(a),bool),weights(videos),bw)
            for key in ['estimate','CI95_lower','CI95_upper']:
                require(abs(actual[key]-float(stored[key])) < 1e-14,'Bootstrap primary summary mismatch')
        print(stage.name+': sealed hashes and exact empirical replay PASSED',flush=True)
    print('Read-only audit PASSED; historical files unchanged.',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ['preflight-only','pilot','all-exposed','audit-only']:
        mode.add_argument('--'+name,action='store_true')
    args = parser.parse_args()
    guard = Guard(args.audit_only)
    sys.addaudithook(guard.hook)
    # Matplotlib's cache lives outside historical outputs; its files are neither
    # scientific measurements nor part of any inference boundary.
    try:
        ctx = load_inputs(guard)
        if args.preflight_only:
            if (OUT/'design_seal.json').exists():
                check_design(ctx)
                print('Existing frozen preflight verified; no writes.',flush=True)
            else:
                preflight(guard,ctx)
                print('PREFLIGHT_PASSED; design and input-only pilot sealed before outcomes.',flush=True)
        elif args.audit_only:
            audit(ctx)
        else:
            check_design(ctx)
            run_stage(ctx,args.all_exposed)
    except (InputError,FileNotFoundError,KeyError,ValueError) as e:
        print('BLOCKED_INPUT: '+str(e),file=sys.stderr,flush=True)
        # Never replace prior artifacts or relabel a completed run on failure.
        if not args.audit_only and not (OUT/'blocked_input.json').exists():
            save(OUT/'blocked_input.json',dict(status='BLOCKED_INPUT',reason=str(e),at_utc=now(),
                new_encoder_calls=0,fits=0,decodes=0,downloads=0))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
