"""Experiment 22 isolated CPU-only workflow, authenticated inputs and read-only audit."""
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import zipfile
import numpy as np
from scipy.special import logsumexp
import task_utility_observability as m

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/task_utility_observability_v1'
DATA = ROOT/'data/source/task_utility_observability_v1'
PRIOR = ROOT/'results/ucf101_causal_development_v1'
OLD = ROOT/'results/ucf101_proxy_replication_v1'
SOURCE = ROOT/'data/source/ucf101_proxy_replication_v1'
CODE = ['task_utility_observability.py', 'task_utility_observability_study.py', 'run_task_utility_observability.py',
        'test_task_utility_observability.py', 'causal_observability_policy.py', 'causal_observability_common.py']
REVIEWED = 'e65e3a1c4066e2360cec1a45dd558060f0532359'
SCOPE = 'EXPLORATORY / DEVELOPMENT: already exposed UCF101 official-training membership only'


def now(): return datetime.now(timezone.utc).isoformat()
def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(v): return hashlib.sha256(json.dumps(v, sort_keys=True, allow_nan=False).encode()).hexdigest()
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(8*2**20), b''): h.update(part)
    return h.hexdigest()


def save(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name+'.writing')
    temp.write_text(json.dumps(v, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    # Windows readers/indexers can briefly deny delete-sharing on the journal.
    # Retry the identical atomic replacement; never alter its scientific value.
    import time
    for attempt in range(100):
        try:
            temp.replace(p);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)


def write_csv(p, rows):
    rows = list(rows)
    m.require(bool(rows), 'Empty empirical CSV')
    with Path(p).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


class Guard:
    """No GPU/decoder/network imports; original outcomes denied; history write protected."""
    original_metadata = {'heads/parameters.json', 'train_collection.json', 'development_collection.json',
        'encoder_identity.json', 'duplicate_and_pretraining_audit.json'}
    def __init__(self):
        self.read_only = False; self.installed = False; self.approved = set(); self.video_paths = set()
    def check(self, path, writing=False):
        if isinstance(path, int): return
        p = Path(os.fsdecode(path)).resolve()
        if writing:
            m.require(not self.read_only, 'Read-only audit forbids writes')
            if p.is_relative_to(ROOT):
                m.require(p.is_relative_to(OUT) or p.is_relative_to(DATA), 'Historical/workspace mutation prohibited: '+str(p))
        elif p.is_relative_to(OLD):
            m.require(p.relative_to(OLD).as_posix() in self.original_metadata, 'Official-test/outcome access prohibited: '+str(p))
        elif p.is_relative_to(ROOT/'data/source'):
            m.require(p.is_relative_to(DATA) or p in self.approved or p in self.video_paths,
                      'Nonapproved source/official-test access prohibited: '+str(p))
        elif p.is_relative_to(ROOT/'results'):
            m.require(p.is_relative_to(OUT) or p.is_relative_to(PRIOR), 'Other historical scientific outcome access prohibited')
    def hook(self, event, args):
        if event == 'open':
            mode, flags = args[1] or '', args[2] or 0
            self.check(args[0], any(v in mode for v in 'wa+x') or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        elif event == 'import':
            m.require(args[0].split('.')[0] not in {'torch', 'cupy', 'av', 'torchvision'}, 'No GPU/model/decoder imports in Experiment22')
        elif event in {'socket.connect', 'socket.getaddrinfo'}:
            raise RuntimeError('Experiment22 does not use network')
        elif event in {'os.remove', 'os.rmdir', 'os.mkdir'}:
            self.check(args[0], True)
        elif event in {'os.rename', 'os.replace'}:
            self.check(args[0], True); self.check(args[1], True)
    def install(self):
        if not self.installed: sys.addaudithook(self.hook); self.installed = True


GUARD = Guard()


def verify(records):
    for path, expected in records.items():
        m.require((ROOT/path).is_file() and sha(ROOT/path) == expected, 'Changed/missing identity: '+path)


def historical_snapshot():
    return git('ls-files', '-s', 'results').splitlines()


def historical_check(snapshot):
    # Git hashes contents without opening their scientific values in Python.
    old = {line.split('\t', 1)[1]:line.split()[1] for line in snapshot if not line.split('\t', 1)[1].startswith(rel(OUT)+'/')}
    result=subprocess.run(['git','hash-object','--no-filters','--stdin-paths'],cwd=ROOT,
        input='\n'.join(old)+'\n',capture_output=True,text=True,check=True)
    actual_hashes=result.stdout.splitlines()
    m.require(len(actual_hashes)==len(old),'Incomplete historical hash inventory')
    for (p, expected),actual in zip(old.items(),actual_hashes):
        m.require(actual == expected, 'Historical output changed: '+p)
    return dict(passed=True, historical_result_blobs=len(old), method='Git blob identity only; official-test scientific contents not parsed')


def validate_roles(videos, classes, official_train):
    roles,hashes,seen=defaultdict(set),defaultdict(set),set()
    for v in videos:
        m.require(v['official_path'] in official_train and v['official_role']=='official_train','Forbidden official-test/foreign population')
        m.require(v['clip'] not in seen,'Duplicate clip');seen.add(v['clip'])
        roles[v['label'],v['group']].add(v['study_role']);hashes[v['sha256']].add(v['study_role'])
    m.require(all(len(s)==1 for s in [*roles.values(),*hashes.values()]),'Group/duplicate role leakage')
    expected={'controller_train','selection','execution'}
    m.require({v['study_role'] for v in videos}==expected,'Unexpected role')
    for label in classes:
        m.require({v['study_role'] for v in videos if v['label']==label}==expected,'Missing class/role')


def load_inputs():
    manifest = read(PRIOR/'population_manifest.json')
    GUARD.video_paths = {(ROOT/v['path']).resolve() for v in manifest['videos']}
    GUARD.approved.add((SOURCE/'official_splits.zip').resolve())
    with zipfile.ZipFile(SOURCE/'official_splits.zip') as f:
        official_train = {line.split()[0] for line in f.read('ucfTrainTestlist/trainlist01.txt').decode().splitlines()}
    validate_roles(manifest['videos'],manifest['classes'],official_train)
    roles, hashes, seen = defaultdict(set), defaultdict(set), set()
    for v in manifest['videos']:
        m.require(v['official_path'] in official_train and v['official_role'] == 'official_train', 'Forbidden official-test/foreign population')
        m.require(v['clip'] not in seen, 'Duplicate clip'); seen.add(v['clip'])
        roles[v['label'], v['group']].add(v['study_role']); hashes[v['sha256']].add(v['study_role'])
        GUARD.approved.add((SOURCE/'features'/f"{v['clip']}.npz").resolve())
    m.require(len(manifest['classes']) == 101 and all(len(x) == 1 for x in [*roles.values(), *hashes.values()]), 'Group/duplicate role leakage')
    expected_roles = {'controller_train', 'selection', 'execution'}
    m.require({v['study_role'] for v in manifest['videos']} == expected_roles, 'Unexpected role')
    for label in manifest['classes']:
        m.require({v['study_role'] for v in manifest['videos'] if v['label'] == label} == expected_roles, 'Missing class/role')
    duplicate = read(OLD/'duplicate_and_pretraining_audit.json')
    parent = {v['clip']:v['clip'] for v in manifest['videos']}
    def root(x):
        while parent[x] != x: x = parent[x]
        return x
    for edge in duplicate['edges']:
        if edge['a'] in parent and edge['b'] in parent: parent[root(edge['a'])] = root(edge['b'])
    component_roles = defaultdict(set)
    for v in manifest['videos']: component_roles[root(v['clip'])].add(v['study_role'])
    m.require(all(len(s) == 1 for s in component_roles.values()), 'Known duplicate component crosses roles')
    provenance = read(PRIOR/'observation_provenance.json')
    rollout = read(PRIOR/'executed_development_rollouts.json')['records']
    dense = read(PRIOR/'dense_development_diagnostics.json')
    for value in [*provenance.values(), *rollout.values(), *dense.values()]: GUARD.approved.add((ROOT/value['path']).resolve())
    head_id = read(PRIOR/'recognition_head_reuse.json'); verify({head_id['path']:head_id['sha256']})
    head = {k:np.asarray(v, dtype=float) for k, v in read(ROOT/head_id['path'])['parameters'].items()}
    return manifest, provenance, rollout, dense, head


def prior_trial(item, inputs):
    verify({item['path']:item['sha256']}); inputs[item['path']] = item['sha256']
    r = read(ROOT/item['path']); m.require(digest(r['payload']) == r['payload_sha256'], 'Historical trial payload mutation')
    payload = r['payload']; GUARD.approved.add((ROOT/payload['features_path']).resolve())
    verify({payload['features_path']:payload['features_sha256']}); inputs[payload['features_path']] = payload['features_sha256']
    return payload


def authenticate_cached_execution(clip, rollout, dense, inputs):
    """Reestablish the allowlist from authenticated historical parents on resume.

    Never whitelist paths merely because a new/possibly corrupted journal lists
    them. The old hashed trial payloads identify every retained feature file.
    """
    prior_trial(dense[clip],inputs)
    for method in ['periodic','RandomB4_seeded','history_motion']:
        prior_trial(rollout[clip+'/'+method],inputs)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    manifest, provenance, rollout, dense, head = load_inputs()
    if not (OUT/'review.json').exists():
        names = ['protocol.md', 'findings.md', 'completion_review.md', 'failure_analysis.md', 'failure_analysis.json',
            'teacher_imitation_context.json', 'timing_summary.json', 'metric_column_semantics.md', 'training_manifest.json',
            'selection_manifest.json', 'evaluation_manifest.json', 'observation_provenance.json', 'verification.json',
            'population_manifest.json', 'SHA256SUMS.json', 'recognition_head_reuse.json', 'controller_selection.json']
        save(OUT/'review.json', dict(actual_main=git('rev-parse', 'origin/main'), reviewed_main=REVIEWED,
            starting_head=git('rev-parse', 'HEAD'), reviewed_inputs={rel(PRIOR/n):sha(PRIOR/n) for n in names},
            reviewed_implementations={p:sha(ROOT/p) for p in ['ucf101_causal_development.py', 'ucf101_causal_development_common.py',
            'causal_observability_policy.py', 'causal_observability_runtime.py', 'causal_observability_learning.py', 'refresh_pooling_utils.py']},
            historical_index=historical_snapshot(), official_test_outcomes_opened=False, at_utc=now()))
    expected = {rel(ROOT/p):sha(ROOT/p) for p in CODE}
    expected.update({rel(OUT/n):sha(OUT/n) for n in ['protocol.md', 'task_target_definitions.md']})
    if (OUT/'analysis_design_seal.json').exists():
        verify(read(OUT/'analysis_design_seal.json')['sha256'])
    else:
        save(OUT/'analysis_design_seal.json', dict(sha256=expected, at_utc=now(),
            scope='Analysis specification before new targets/probes; prior development outcomes already exposed',
            seed=m.SEED, bootstrap_repeats=m.BOOTSTRAPS))
    save(OUT/'population_manifest.json', dict(scope=SCOPE, original_manifest_sha256=sha(PRIOR/'population_manifest.json'),
        classes=manifest['classes'], counts=manifest['counts'], videos=[{k:v[k] for k in
        ['clip','label','group','sha256','path','official_path','official_role','role','study_role']} for v in manifest['videos']],
        primary_state_population='Periodic B4 chronological prefixes only, all original roles',
        secondary_populations=['actual seeded random execution prefixes', 'actual H+M execution prefixes'],
        independence_unit='(class,gXX); unknown parent/reupload and pretraining relationships remain unresolved'))
    save(OUT/'environment.json', dict(python=sys.version, packages={k:importlib.metadata.version(k) for k in
        ['numpy','scipy','matplotlib']}, CPU_threads=1, CUDA_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        GPU_calls=0, new_network_bytes=0, package_installs=[], available_disk_bytes=__import__('shutil').disk_usage(ROOT).free))
    return manifest, provenance, rollout, dense, head


def extract():
    from causal_observability_policy import cheap_signal
    manifest, provenance, rollout, dense, head = prepare()
    led_path = OUT/'observation_provenance.json'
    ledger = read(led_path) if led_path.exists() else {}
    collections = {role:read(OLD/f'{role}_collection.json') for role in ['train','development']}
    encoder_hash = sha(OLD/'encoder_identity.json')
    for i, v in enumerate(manifest['videos']):
        clip = v['clip']
        if clip in ledger:
            if v['study_role']=='execution':authenticate_cached_execution(clip,rollout,dense,{})
            verify(ledger[clip]['outputs']); verify(ledger[clip]['inputs']); continue
        inputs = {v['path']:v['sha256']}; verify(inputs)
        original = collections[v['role']][clip]
        m.require(original['input_sha256'] == v['sha256'] and original['grid_sha256'] == digest(v['snippet_grid']) and
            original['encoder_identity_sha256'] == encoder_hash, 'Original feature lineage mismatch')
        verify({original['path']:original['sha256']}); inputs[original['path']] = original['sha256']
        with np.load(ROOT/original['path'], allow_pickle=False) as a: original_f = a['x3d_xs'].copy()
        m.require(original_f.shape == (16,2048) and original_f.dtype == np.float32 and np.isfinite(original_f).all(), 'Genuine features missing')
        populations = {}
        if v['study_role'] != 'execution':
            r = provenance[clip]; verify({r['path']:r['sha256']}); inputs[r['path']] = r['sha256']
            with np.load(ROOT/r['path'], allow_pickle=False) as a: f, thumbs = a['features'], a['thumbnails']
            np.testing.assert_array_equal(f, original_f)
            m.require(thumbs.shape == (16,32,32) and thumbs.dtype == np.uint8, 'Genuine thumbnails missing')
            signals = np.stack([cheap_signal(thumbs[t], thumbs[t-1] if t else None) for t in range(16)])
            positions = [0,4,8,12]
            trace = [dict(t=t, available_refreshes=[j for j in positions if j<t], selected=t in positions) for t in range(16)]
            populations['periodic'] = (trace, {j:f[j] for j in positions})
        else:
            d = prior_trial(dense[clip], inputs)
            with np.load(ROOT/d['features_path'], allow_pickle=False) as a: f = a['features']
            np.testing.assert_array_equal(f, original_f)
            hm = prior_trial(rollout[clip+'/history_motion'], inputs)
            signals = np.array([s['cheap_signal'] for s in hm['trace']])
            m.require(signals.shape == (16,32), 'Actual encountered cheap observations missing')
            for method, population in [('periodic','periodic'), ('RandomB4_seeded','random'), ('history_motion','HM_executed')]:
                r = hm if population == 'HM_executed' else prior_trial(rollout[clip+'/'+method], inputs)
                m.require(r['encoder_starts'] == r['encoder_completions'] == 4 and len(r['positions']) == 4 and r['positions'][0] == 0,
                          'Historical genuine four-call execution failed')
                with np.load(ROOT/r['features_path'], allow_pickle=False) as a: actual = a['features']
                np.testing.assert_array_equal(actual, f[r['positions']])
                np.testing.assert_allclose(r['logits'], m.affine(f[r['cache_indices']].mean(0,dtype=float), head), atol=1e-9, rtol=1e-9)
                populations[population] = (r['trace'], dict(zip(r['positions'], actual)))
        z = m.affine(f, head); losses = m.ce(m.WEIGHTS @ z, manifest['classes'].index(v['label']))
        centered = z-z.mean(1, keepdims=True); proxy = np.square(m.WEIGHTS @ centered-centered.mean(0)).mean(1)
        scalar_z = np.stack([sum((z[j] for j in cache), start=np.zeros(z.shape[1]))/16 for cache in m.CACHE])
        scalar_losses = np.logaddexp.reduce(scalar_z, axis=1)-scalar_z[:,manifest['classes'].index(v['label'])]
        scalar_loss_error = float(np.max(abs(scalar_losses-losses)))
        m.require(scalar_loss_error < 1e-10, 'Independent scalar C pooling/loss disagreement')
        states, vectors, max_target_error = [], [], 0.
        for population, (trace, actual_features) in populations.items():
            positions = [r['t'] for r in trace if r['selected']]
            for t in range(1,16):
                hist = [j for j in positions if j<t]
                row = trace[t]
                m.require(row['t'] == t and row['available_refreshes'] == hist, 'Recorded prefix differs')
                obs = m.causal_inputs(t, hist, {j:actual_features[j] for j in hist}, signals[t])
                target = m.task_targets(z, manifest['classes'].index(v['label']), t, hist, losses, proxy)
                ilocal, iq = m.independent_completion_targets(z, manifest['classes'].index(v['label']), t, hist, scalar_losses)
                errors = [abs(ilocal-target['gain_local'])]
                for branch, q in zip(['Q_hold','Q_refresh'], iq):
                    m.require((q is None) == (target[branch] is None), 'Independent branch feasibility mismatch')
                    if q is not None: errors.append(abs(q-target[branch]))
                max_target_error = max(max_target_error, *errors)
                m.require(max_target_error < 1e-10, 'Independent target enumeration disagreement')
                disc = m.discretionary(t, hist)
                m.require(disc == (target['gain_budget'] is not None), 'Discretionary feasibility mismatch')
                states.append(dict(clip=clip, label=v['label'], group=v['group'], role=v['study_role'], population=population,
                    t=t, candidate_seconds=v['snippet_grid']['endpoint_timestamps'][t],
                    endpoint_frame_index=v['snippet_grid']['endpoint_indices'][t], age=obs.age, remaining=obs.remaining,
                    discretionary=disc, selected=bool(row['selected']), history=hist, MAD=float(signals[t,26]), **target))
                vectors.append(m.observation_vector(obs, head))
        # Full-information headroom is an explicit retrospective diagnostic.
        periodic_id = int(np.flatnonzero(np.all(m.POSITIONS == [0,4,8,12], axis=1))[0])
        zp_id = int(np.flatnonzero(proxy <= proxy.min()+1e-12)[0]); task_id = int(np.argmin(losses))
        headroom = dict(clip=clip,label=v['label'],group=v['group'],role=v['study_role'], periodic_CE=float(losses[periodic_id]),
            task_oracle_CE=float(losses[task_id]), Z_pool_CE=float(losses[zp_id]), dense_CE=float(m.ce(z.mean(0),manifest['classes'].index(v['label']))),
            oracle_schedule=m.POSITIONS[task_id].tolist(), Z_pool_schedule=m.POSITIONS[zp_id].tolist())
        p = DATA/'clips'/f'{clip}.npz'; p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, observations=np.asarray(vectors,dtype=np.float64))
        q = p.with_suffix('.json'); save(q,dict(states=states,headroom=headroom))
        ledger[clip] = dict(inputs=inputs, outputs={rel(p):sha(p),rel(q):sha(q)}, original_input_sha256=v['sha256'],
            original_feature=original, sampling_grid_sha256=digest(v['snippet_grid']), scalar_loss_max_error=scalar_loss_error,
            independent_target_max_error=max_target_error, states=len(states), scope='Genuine archived embeddings/encountered observations; CPU retrospective targets only')
        save(led_path,ledger)
        if (i+1)%50 == 0: print('Authenticated CPU target extraction',i+1,'/',len(manifest['videos']),flush=True)
    return manifest, ledger


def dataset(ledger, role, population='periodic'):
    rows, x = [], []
    for clip, record in ledger.items():
        p = DATA/'clips'/f'{clip}.json'; a = read(p)
        choose = [i for i,r in enumerate(a['states']) if r['role']==role and r['population']==population and r['discretionary']]
        if choose:
            with np.load(p.with_suffix('.npz'),allow_pickle=False) as z: x.extend(z['observations'][choose])
            rows.extend(a['states'][i] for i in choose)
    m.require(bool(rows),'Missing genuine state population')
    return rows, np.asarray(x)


def train_probes():
    ledger = read(OUT/'observation_provenance.json')
    train, x = dataset(ledger,'controller_train'); selection, xs = dataset(ledger,'selection')
    wt, ws = m.hierarchical_weights(train), m.hierarchical_weights(selection)
    identity = dict(design_sha256=sha(OUT/'analysis_design_seal.json'), observations_sha256=sha(OUT/'observation_provenance.json'))
    path = OUT/'probe_training.json'
    if path.exists():
        old = read(path); m.require(old['identity']==identity,'Probe checkpoint identity changed');verify(old['model_files']);return
    records, model_files = [], {}
    for target in m.TARGETS:
        yt = np.array([r['gain_'+target] for r in train]); ys = np.array([r['gain_'+target] for r in selection])
        for view in m.VIEWS:
            ids = m.view_indices()[view]; xt, xv = x[:,ids], xs[:,ids]
            classification, regression = [], []
            for order, alpha in enumerate(m.GRID if view!='constant' else [0.]):
                classifier = m.fit_linear(xt,(yt>0).astype(float),wt,alpha,True)
                ridge = m.fit_linear(xt,yt,wt,alpha,False)
                pc, pr = m.predict(classifier,xv), m.predict(ridge,xv)
                met = m.metrics(ys,pc,pr,ws)
                classification.append(dict(grid_order=order,alpha=alpha,selection_Brier=met['Brier'],selection_PR_AUC=met['PR_AUC'],model=classifier))
                regression.append(dict(grid_order=order,alpha=alpha,selection_MAE=met['MAE'],selection_Spearman=met['Spearman'],model=ridge))
            bestc = min(classification,key=lambda r:(r['selection_Brier'],-(r['selection_PR_AUC'] or 0),r['grid_order']))
            bestr = min(regression,key=lambda r:(r['selection_MAE'],-(r['selection_Spearman'] or 0),r['grid_order']))
            cp = m.predict(bestc['model'],xv); cal = m.calibrate(cp,(ys>0).astype(float),ws)
            threshold, threshold_grid = m.threshold_choice(ys,m.calibrated(cp,cal),ws)
            payload = dict(target=target,view=view,feature_indices=ids.tolist(),classifier=bestc['model'],regression=bestr['model'],calibration=cal,
                threshold=threshold['threshold'],identity=identity)
            p = OUT/'probes'/f'{target}_{view}.json';save(p,payload);model_files[rel(p)]=sha(p)
            records.append(dict(target=target,view=view,classification_grid=classification,regression_grid=regression,
                selected_classifier_alpha=bestc['alpha'],selected_regression_alpha=bestr['alpha'],calibration=cal,threshold_grid=threshold_grid,
                selected_threshold=threshold['threshold']))
            print('CPU linear diagnostic probes',target,view,flush=True)
    save(path,dict(identity=identity,model_files=model_files,records=records,training_states=len(train),selection_states=len(selection),
        training_clips=len({r['clip'] for r in train}),selection_clips=len({r['clip'] for r in selection}),
        architecture='regularized linear logistic/ridge only; no adaptive controller trained',optimization='weighted objective; train-only standardization; SciPy L-BFGS-B max400; grid .001/.01/.1/1',
        calibration='nondecreasing Platt, selection only, fixed slope penalty .01; threshold fixed grid .25/.5/.75 maximizing selection F1',
        classification_selection='lowest raw selection Brier then highest AP then grid order',regression_selection='lowest selection MAE then highest Spearman then grid order'))


def weighted_quantiles(values, weights, quantiles=(.01,.1,.5,.9,.99)):
    order=np.argsort(values,kind='stable'); a=np.asarray(values)[order]; w=np.asarray(weights)[order]
    return {str(q):float(a[min(np.searchsorted(np.cumsum(w),q,side='left'),len(a)-1)]) for q in quantiles}


def evidence_rows(rows, gain, w):
    return dict(states=len(rows), clips=len({r['clip'] for r in rows}), groups=len({(r['label'],r['group']) for r in rows}),
        classes=len({r['label'] for r in rows}), positive=float(w @ (gain>0)), negative=float(w @ (gain<0)),
        exact_zero=float(w @ (gain==0)), near_zero_abs_le_1e_10=float(w @ (abs(gain)<=1e-10)),
        gain_quantiles=weighted_quantiles(gain,w), weighted_mean_gain=float(w @ gain))


def analyze_predictions():
    ledger=read(OUT/'observation_provenance.json'); selections=read(OUT/'probe_training.json')
    verify(selections['model_files']); metrics_rows=[]; calibration=[]; distributions=[]; predictions=[]
    populations=[('controller_train','periodic'),('selection','periodic'),('execution','periodic'),('execution','random'),('execution','HM_executed')]
    models={(t,v):read(OUT/'probes'/f'{t}_{v}.json') for t in m.TARGETS for v in m.VIEWS}
    all_states=[]
    for role,population in populations:
        rows,x=dataset(ledger,role,population); w=m.hierarchical_weights(rows)
        gains=np.array([[r['gain_'+t] for r in rows] for t in m.TARGETS])
        raw=np.empty((2,5,len(rows))); calibrated=np.empty_like(raw); regression=np.empty_like(raw)
        for ti,target in enumerate(m.TARGETS):
            distributions.append(dict(role=role,population=population,target=target,**evidence_rows(rows,gains[ti],w)))
            for vi,view in enumerate(m.VIEWS):
                model=models[target,view]; a=x[:,model['feature_indices']]
                raw[ti,vi]=m.predict(model['classifier'],a); calibrated[ti,vi]=m.calibrated(raw[ti,vi],model['calibration'])
                regression[ti,vi]=m.predict(model['regression'],a)
                for kind,prob in [('raw',raw[ti,vi]),('selection_calibrated',calibrated[ti,vi])]:
                    met=m.metrics(gains[ti],prob,regression[ti,vi],w,model['threshold'])
                    metrics_rows.append(dict(role=role,population=population,target=target,view=view,calibration=kind,stratum='all',
                        states=len(rows),groups=len({(r['label'],r['group']) for r in rows}),classes=len({r['label'] for r in rows}),
                        threshold=model['threshold'],**met))
                    calibration.append(dict(role=role,population=population,target=target,view=view,calibration=kind,
                        bins=m.calibration_bins(gains[ti],prob,w)))
                if role=='execution':
                    strata={**{f'position_{lo}_{hi}':np.array([lo<=r['t']<=hi for r in rows]) for lo,hi in [(1,4),(5,8),(9,15)]},
                            **{f'remaining_{b}':np.array([r['remaining']==b for r in rows]) for b in [1,2,3]}}
                    for name,mask in strata.items():
                        subset=[r for r,k in zip(rows,mask) if k]
                        if not subset:continue
                        ww=m.hierarchical_weights(subset)
                        metrics_rows.append(dict(role=role,population=population,target=target,view=view,calibration='selection_calibrated',
                            stratum=name,states=len(subset),groups=len({(r['label'],r['group']) for r in subset}),
                            classes=len({r['label'] for r in subset}),threshold=model['threshold'],
                            **m.metrics(gains[ti,mask],calibrated[ti,vi,mask],regression[ti,vi,mask],ww,model['threshold'])))
                error=regression[ti,vi]-gains[ti]
                distributions.append(dict(role=role,population=population,target=target,view=view,
                    signed_error_quantiles=weighted_quantiles(error,w),absolute_error_quantiles=weighted_quantiles(abs(error),w),
                    predicted_positive_true_negative=float(w @ ((regression[ti,vi]>0)&(gains[ti]<=0))),
                    predicted_nonpositive_true_positive=float(w @ ((regression[ti,vi]<=0)&(gains[ti]>0)))))
        p=OUT/f'predictions_{role}_{population}.npz'
        np.savez_compressed(p,gains=gains,raw=raw,calibrated=calibrated,regression=regression)
        meta=OUT/f'states_{role}_{population}.json.gz'
        with gzip.open(meta,'wt',encoding='utf-8') as f:json.dump(rows,f,separators=(',',':'))
        predictions.append(dict(role=role,population=population,path=rel(p),sha256=sha(p),states_path=rel(meta),states_sha256=sha(meta)))
        all_states.extend(rows)
    write_csv(OUT/'probe_metrics.csv',metrics_rows)
    save(OUT/'prediction_inventory.json',predictions);save(OUT/'calibration_summary.json',calibration)
    save(OUT/'gain_and_error_distributions.json',distributions)
    # Conditional distributions describe the actual dependent state population.
    conditional=[]
    for role,population in populations:
        rows=[r for r in all_states if r['role']==role and r['population']==population]
        w=m.hierarchical_weights(rows)
        for variable in ['t','age','remaining','teacher_refresh']:
            conditional.extend(dict(role=role,population=population,variable=variable,value=value,
                states=sum(r[variable]==value for r in rows),weight=float(w @ np.array([r[variable]==value for r in rows])))
                for value in sorted({r[variable] for r in rows}))
        conditional.append(dict(role=role,population=population,variable='MAD_quantiles',value=weighted_quantiles(np.array([r['MAD'] for r in rows]),w),states=len(rows),weight=1.))
    save(OUT/'conditional_state_distributions.json',conditional)


def read_prediction(item):
    verify({item['path']:item['sha256'],item['states_path']:item['states_sha256']})
    with gzip.open(ROOT/item['states_path'],'rt',encoding='utf-8') as f:rows=json.load(f)
    with np.load(ROOT/item['path'],allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    return rows,arrays


def bootstrap():
    items=[r for r in read(OUT/'prediction_inventory.json') if r['role']=='execution']
    rows,_=read_prediction(items[0]);groups,draws=m.group_plan(rows)
    np.save(OUT/'bootstrap_group_indices.npy',draws,allow_pickle=False)
    save(OUT/'bootstrap_groups.json',dict(groups=groups,seed=m.SEED,repeats=m.BOOTSTRAPS,
        independent_unit='(class,gXX)',resampling='paired class-stratified groups; carry all clips/states/methods together',
        single_group_classes=[c for c in sorted({g[0] for g in groups}) if sum(x[0]==c for x in groups)==1],
        limitations='Conditional on fitted/selected/calibrated probes and reused head; no fitting/selection/exposure uncertainty. One-group classes have no within-class variance.'))
    lookup={tuple(g):i for i,g in enumerate(groups)}
    counts=np.stack([np.bincount(d,minlength=len(groups)) for d in draws])
    comparisons=[('H','baseline'),('HM','H'),('M','baseline'),('M','constant')]
    names=['PR_AUC','ROC_AUC','Brier','MAE','Spearman','signed_bias','sign_error','precision','recall']
    intervals=[]
    for item in items:
        rows,a=read_prediction(item);m.require(sorted({(r['label'],r['group']) for r in rows})==groups,'Secondary population group membership differs')
        ids=np.array([lookup[r['label'],r['group']] for r in rows]);w0=m.hierarchical_weights(rows)
        model_thresholds=np.array([[read(OUT/'probes'/f'{t}_{v}.json')['threshold'] for v in m.VIEWS] for t in m.TARGETS])
        for ti,target in enumerate(m.TARGETS):
            values=np.full((m.BOOTSTRAPS,5,len(names)),np.nan)
            for b in range(m.BOOTSTRAPS):
                w=w0*counts[b,ids];m.require(abs(w.sum()-1)<1e-12,'Bootstrap estimand weights changed')
                for vi in range(5):
                    met=m.metrics(a['gains'][ti],a['calibrated'][ti,vi],a['regression'][ti,vi],w,model_thresholds[ti,vi])
                    values[b,vi]=[np.nan if met[n] is None else met[n] for n in names]
                if (b+1)%500==0:print('Paired source bootstrap',item['population'],target,b+1,'/',m.BOOTSTRAPS,flush=True)
            np.save(OUT/f'bootstrap_metrics_{item["population"]}_{target}.npy',values,allow_pickle=False)
            for candidate,reference in comparisons:
                ci,ri=m.VIEWS.index(candidate),m.VIEWS.index(reference)
                cm=m.metrics(a['gains'][ti],a['calibrated'][ti,ci],a['regression'][ti,ci],w0,model_thresholds[ti,ci])
                rm=m.metrics(a['gains'][ti],a['calibrated'][ti,ri],a['regression'][ti,ri],w0,model_thresholds[ti,ri])
                for ni,name in enumerate(names):
                    dif=values[:,ci,ni]-values[:,ri,ni];valid=dif[np.isfinite(dif)]
                    bounds=np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else [None,None]
                    intervals.append(dict(population=item['population'],target=target,candidate=candidate,reference=reference,metric=name,
                        difference=None if cm[name] is None or rm[name] is None else cm[name]-rm[name],CI95=bounds,valid_draws=len(valid),
                        orientation='candidate minus reference; lower better for Brier/MAE/sign_error, higher for AUC/Spearman; signed_bias not a utility',
                        scope=SCOPE+'; marginal/unadjusted conditional bootstrap; not confirmation'))
    save(OUT/'paired_group_intervals.json',dict(repeats=m.BOOTSTRAPS,seed=m.SEED,intervals=intervals))


def alignment_and_headroom():
    align=[];headrooms=[]
    for item in read(OUT/'prediction_inventory.json'):
        rows,a=read_prediction(item);w=m.hierarchical_weights(rows);teacher=np.array([r['teacher_refresh'] for r in rows],bool)
        for ti,target in enumerate(m.TARGETS):
            gain=a['gains'][ti];positive=gain>0;negative=gain<0
            agree=(teacher&positive)|(~teacher&~positive)
            align.append(dict(role=item['role'],population=item['population'],target=target,states=len(rows),
                teacher_refresh_fraction=float(w @ teacher),task_positive_fraction=float(w @ positive),agreement=float(w @ agree),
                TP=float(w @ (teacher&positive)),FP=float(w @ (teacher&~positive)),TN=float(w @ (~teacher&~positive)),FN=float(w @ (~teacher&positive)),
                teacher_refresh_task_harmful_mass=float(w @ (teacher&negative)),
                task_tied_mass=float(w @ (gain==0)),near_tie_mass=float(w @ (abs(gain)<=1e-10)),
                mean_wrong_branch_hindsight_regret=float(w @ (abs(gain)*~agree)),
                gain_if_teacher_refresh=float(w[teacher] @ gain[teacher]/w[teacher].sum()) if teacher.any() else None))
    ledger=read(OUT/'observation_provenance.json')
    headrooms=[read(DATA/'clips'/f'{clip}.json')['headroom'] for clip in ledger]
    summary=[]
    for role in ['controller_train','selection','execution']:
        r=[v for v in headrooms if v['role']==role];w=m.hierarchical_weights(r)
        q={k:float(w @ np.array([v[k] for v in r])) for k in ['periodic_CE','task_oracle_CE','Z_pool_CE','dense_CE']}
        m.require(all(v['task_oracle_CE']<=v['periodic_CE']+1e-12 for v in r),'Oracle schedule missing periodic')
        q.update(role=role,clips=len(r),groups=len({(v['label'],v['group']) for v in r}),
            perfect_hindsight_headroom=q['periodic_CE']-q['task_oracle_CE'],proxy_reduction=q['periodic_CE']-q['Z_pool_CE'])
        summary.append(q)
    write_csv(OUT/'per_clip_offline_headroom.csv',headrooms)
    save(OUT/'teacher_task_alignment.json',dict(rows=align,interpretation='Conditional full-information proxy teacher versus positive task-gain signs. Ties retained; no deployable oracle or causal treatment effect.'))
    save(OUT/'offline_headroom.json',dict(roles=summary,all_schedules=455,scope='Full-information true-label task oracle; not achievable causal policy evidence'))


def costs():
    import pandas as pd
    trials=pd.read_csv(PRIOR/'timing_trials.csv',float_precision='round_trip')
    keep=trials[~trials['warmup']].copy()
    manifest=read(OUT/'population_manifest.json'); approved={v['clip'] for v in manifest['videos'] if v['study_role']=='execution'}
    m.require(set(trials['clip']).issubset(approved),'Nonapproved timing source')
    expected=np.where(trials['method']=='dense16',16,4)
    m.require(np.array_equal(trials['encoder_starts'],expected) and np.array_equal(trials['encoder_completions'],expected),'Recorded timing forward-count disagreement')
    cols=['seconds','cheap_thumbnail_seconds','cheap_statistics_seconds','controller_decide_seconds','observation_preparation_seconds']
    med=keep.groupby(['clip','label','group','method'])[cols].median().reset_index()
    write_csv(OUT/'recorded_timing_per_video.csv',med.to_dict('records'))
    prior=read(PRIOR/'timing_summary.json');methods=[]
    for method,g in med.groupby('method'):
        values={c:float(g[c].median()) for c in cols}
        saved=next(v for v in prior['methods'] if v['method']==method)
        m.require(abs(values['seconds']-saved['median_seconds'])<1e-12,'Archived timing summary not independently reproduced')
        methods.append(dict(method=method,groups=len(g),**values))
    pivot=med.pivot(index='clip',columns='method',values='seconds')
    incremental=[]
    for method,reference in [('history_motion','periodic'),('history','periodic'),('history_motion','history'),('periodic_unused_gate','periodic')]:
        d=(pivot[method]-pivot[reference]).to_numpy();rng=np.random.Generator(np.random.PCG64(m.SEED+1))
        draws=rng.integers(len(d),size=(m.BOOTSTRAPS,len(d)))
        incremental.append(dict(method=method,reference=reference,paired_median_extra_seconds=float(np.median(d)),
            conditional_group_bootstrap_median_CI95=np.percentile(np.median(d[draws],axis=1),[2.5,97.5]).tolist(),
            groups_slower=int((d>0).sum()),groups=len(d)))
    quality=pd.read_csv(PRIOR/'development_per_clip_metrics.csv',float_precision='round_trip')
    method_order=['periodic','RandomB4_seeded','motion','history','history_motion','dense16']
    pareto=[]
    for population,selected in [('all_400_quality_vs_30_cost',quality),('matched_30_timing_clips',quality[quality['clip'].isin(pivot.index)])]:
        points=[]
        for method in method_order:
            q=selected[selected['method']==method];means=q.groupby(['label','group'])['CE'].mean().groupby('label').mean()
            points.append(dict(method=method,CE=float(means.mean()),latency=float(next(r['seconds'] for r in methods if r['method']==method))))
        for point in points:
            dominated=[p['method'] for p in points if p['CE']<=point['CE'] and p['latency']<=point['latency'] and (p['CE']<point['CE'] or p['latency']<point['latency'])]
            pareto.append(dict(population=population,**point,dominated_by=dominated))
    save(OUT/'cost_feasibility.json',dict(source_sha256={rel(PRIOR/n):sha(PRIOR/n) for n in ['timing_trials.csv','timing_summary.json','metric_column_semantics.md','development_per_clip_metrics.csv']},
        measured_trials=len(keep),all_trials=len(trials),groups=trials['clip'].nunique(),classes=trials['label'].nunique(),methods=methods,incremental=incremental,pareto=pareto,
        boundary=prior['boundary'],observation_preparation_includes_thumbnail_and_statistics=True,
        no_replay_timing=True,new_GPU_calls=0,scope='Previously measured real execution; no new timing or optimized deployment claim',
        uncertainty='Timing bootstrap resamples 30 source groups, not repetitions; conditional on this input-selected cohort. Across-class inference limited to observed classes. Mixed-cohort quality/cost points are separately labeled.'))


def target_verification():
    ledger=read(OUT/'observation_provenance.json');manifest,_,_,_,head=load_inputs()
    # Hash-selected real-feature replay independently pools raw 2048D arrays for
    # every schedule. No selected example is synthetic or outcome-selected.
    chosen=sorted(manifest['videos'],key=lambda v:hashlib.sha256((str(m.SEED)+'|scalar_replay|'+v['clip']).encode()).hexdigest())[:12]
    errors=[]
    for v in chosen:
        record=ledger[v['clip']]['original_feature']
        with np.load(ROOT/record['path'],allow_pickle=False) as a:f=a['x3d_xs']
        z=m.affine(f,head);y=manifest['classes'].index(v['label']);expected=m.ce(m.WEIGHTS @ z,y)
        actual=[]
        for p in m.POSITIONS:
            current=0;cache=[]
            for t in range(16):
                if t in p:current=t
                cache.append(f[current])
            pooled=sum((np.asarray(row,dtype=float) for row in cache),start=np.zeros(2048))/16
            logits=((pooled-head['means'])/head['scales']) @ head['weights'].T+head['bias']
            actual.append(float(np.logaddexp.reduce(logits)-logits[y]))
        error=float(np.max(abs(np.asarray(actual)-expected)));m.require(error<1e-10,'Independent genuine pooled-feature/head objective failed')
        errors.append(dict(clip=v['clip'],schedules=len(actual),maximum_loss_error=error,feature_sha256=record['sha256']))
    all_states=[r for clip in ledger for r in read(DATA/'clips'/f'{clip}.json')['states']]
    save(OUT/'target_replay_validation.json',dict(passed=True,clips=len(ledger),states=len(all_states),
        discretionary_states=sum(r['discretionary'] for r in all_states),enumerated_schedules_per_clip=455,
        all_clip_independent_scalar_C_pooling_loss_max_error=max(v['scalar_loss_max_error'] for v in ledger.values()),
        all_state_independent_prefix_combinations_target_max_error=max(v['independent_target_max_error'] for v in ledger.values()),
        raw_feature_affine_replay=errors,local_sign='CE(cache)-CE(current)',budget_sign='min hold-completion CE minus min refresh-completion CE',
        forced_and_exhausted_states_retained_but_excluded_from_probes=True,near_zero_exclusion=False,
        new_X3D_calls=0,new_video_decodes=0,labels_used_only_for_retrospective_targets=True))


def audit():
    """Hash/consistency audit only: never fit, predict with a model, or rebuild targets."""
    GUARD.read_only=True
    manifest,_,_,_,_=load_inputs()
    verify(read(OUT/'analysis_design_seal.json')['sha256'])
    verify(read(OUT/'SHA256SUMS.json'))
    ledger=read(OUT/'observation_provenance.json')
    for r in ledger.values():
        for p in r['inputs']: GUARD.approved.add((ROOT/p).resolve())
        verify(r['inputs']);verify(r['outputs'])
    history=historical_check(read(OUT/'review.json')['historical_index'])
    training=read(OUT/'probe_training.json');verify(training['model_files'])
    result_rows=[]
    with (OUT/'probe_metrics.csv').open(encoding='utf-8',newline='') as f:stored=list(csv.DictReader(f))
    groups=read(OUT/'bootstrap_groups.json')['groups'];draws=np.load(OUT/'bootstrap_group_indices.npy',allow_pickle=False)
    base_rows=None
    for item in read(OUT/'prediction_inventory.json'):
        rows,a=read_prediction(item);w=m.hierarchical_weights(rows)
        if item['role']=='execution' and item['population']=='periodic':base_rows=rows
        for ti,target in enumerate(m.TARGETS):
            for vi,view in enumerate(m.VIEWS):
                model=read(OUT/'probes'/f'{target}_{view}.json')
                # No prediction: recompute metrics solely from retained prediction vectors.
                met=m.metrics(a['gains'][ti],a['calibrated'][ti,vi],a['regression'][ti,vi],w,model['threshold'])
                row=next(r for r in stored if r['role']==item['role'] and r['population']==item['population'] and r['target']==target and r['view']==view and r['calibration']=='selection_calibrated' and r['stratum']=='all')
                for k,v in met.items():
                    m.require(row[k]=='' if v is None else abs(float(row[k])-v)<1e-11,'Retained metric consistency failure')
                result_rows.append((item['role'],item['population'],target,view))
    actual_groups,expected=m.group_plan(base_rows)
    m.require(actual_groups==list(map(tuple,groups)) and np.array_equal(expected,draws),'Bootstrap seed/group pairing mismatch')
    paired=read(OUT/'paired_group_intervals.json')['intervals']
    names=['PR_AUC','ROC_AUC','Brier','MAE','Spearman','signed_bias','sign_error','precision','recall']
    for row in paired:
        values=np.load(OUT/f"bootstrap_metrics_{row['population']}_{row['target']}.npy",allow_pickle=False)
        d=values[:,m.VIEWS.index(row['candidate']),names.index(row['metric'])]-values[:,m.VIEWS.index(row['reference']),names.index(row['metric'])]
        valid=d[np.isfinite(d)]
        if len(valid):np.testing.assert_allclose(np.percentile(valid,[2.5,97.5]),row['CI95'],atol=1e-12,rtol=0)
        m.require(len(valid)==row['valid_draws'],'Bootstrap valid-draw count changed')
    m.require(read(OUT/'target_replay_validation.json')['passed'],'Independent target replay missing')
    m.require(not any(n.split('.')[0] in {'torch','cupy','av'} for n in sys.modules),'Forbidden GPU/decoder runtime loaded')
    return dict(passed=True,scope=SCOPE,input_clips=len(manifest['videos']),metric_tables_reproduced=len(result_rows),
        history=history,bootstrap_pairing_and_intervals_verified=True,independent_target_replay_receipt_verified=True,
        original_source_and_feature_hashes_verified=True,official_test_outcomes_opened=False,new_GPU_calls=0,
        new_video_decodes=0,network_bytes=0,read_only=True,model_predictions=0,training_calls=0,results_regenerated=0)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    frame=pd.read_csv(OUT/'probe_metrics.csv')
    selected=frame[(frame.role=='execution')&(frame.population=='periodic')&(frame.stratum=='all')&(frame.calibration=='selection_calibrated')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,target in zip(axes,m.TARGETS):
        rows=selected[selected.target==target].set_index('view').loc[['constant','baseline','H','M','HM']]
        bars=ax.bar(['Constant','Time/age/\nbudget','H','M','H+M'],rows.PR_AUC,color=['#a6a6a6','#879baf','#426b9d','#5b9a8b','#d28c3a'])
        ax.axhline(rows.positive_prevalence.iloc[0],color='black',ls='--',lw=1,label='Positive prevalence')
        ax.set(title='A: immediate task gain' if target=='local' else 'B: budget-constrained hindsight gain',ylabel='Weighted PR AUC (average precision)',ylim=(0,1))
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9);ax.legend(fontsize=8)
    fig.suptitle('EXPLORATORY / exposed development: identical periodic states, linear probes',fontsize=12)
    fig.tight_layout();fig.savefig(figdir/'task_benefit_prediction.png',dpi=200);fig.savefig(figdir/'task_benefit_prediction.pdf');plt.close(fig)
    align=read(OUT/'teacher_task_alignment.json')['rows'];fig,axes=plt.subplots(2,2,figsize=(9,7.5))
    for row,target in enumerate(m.TARGETS):
        for col,population in enumerate(['periodic','HM_executed']):
            r=next(v for v in align if v['role']=='execution' and v['population']==population and v['target']==target)
            matrix=np.array([[r['TN'],r['FN']],[r['FP'],r['TP']]])
            ax=axes[row,col];ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues')
            for i in range(2):
                for j in range(2):ax.text(j,i,f'{100*matrix[i,j]:.1f}%',ha='center',va='center',color='white' if matrix[i,j]>.5 else 'black')
            ax.set(xticks=[0,1],xticklabels=['Nonpositive','Positive'],yticks=[0,1],yticklabels=['Hold','Refresh'],
                xlabel='Retrospective task-gain sign',ylabel='Privileged proxy teacher',
                title=f'{"A: local" if target=="local" else "B: budget"}; {population}')
    fig.suptitle('EXPLORATORY: conditional proxy/task alignment (weighted state mass)',fontsize=12)
    fig.tight_layout();fig.savefig(figdir/'teacher_task_alignment.png',dpi=200);fig.savefig(figdir/'teacher_task_alignment.pdf');plt.close(fig)
    cost=read(OUT/'cost_feasibility.json');intervals=read(OUT/'paired_group_intervals.json')['intervals']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    names=['periodic','history','history_motion','periodic_unused_gate']
    times=[next(r['seconds'] for r in cost['methods'] if r['method']==n)*1000 for n in names]
    bars=axes[0].bar(['Periodic','H','H+M','Periodic +\nunused gate'],times,color=['#879baf','#426b9d','#d28c3a','#b09c81'])
    axes[0].bar_label(bars,fmt='%.1f',padding=3);axes[0].set(ylabel='Previously measured end-to-end latency (ms)',ylim=(0,max(times)*1.2),title='30 actual timing groups; per-video medians')
    for i,target in enumerate(m.TARGETS):
        r=next(v for v in intervals if v['population']=='periodic' and v['target']==target and v['candidate']=='HM' and v['reference']=='H' and v['metric']=='PR_AUC')
        axes[1].errorbar(r['difference'],i,xerr=[[r['difference']-r['CI95'][0]],[r['CI95'][1]-r['difference']]],fmt='o',color='#d28c3a',capsize=4)
    axes[1].axvline(0,color='black',lw=1);axes[1].set(yticks=[0,1],yticklabels=['A: local','B: budget'],xlabel='H+M minus H PR AUC; 95% paired group interval',ylim=(-.7,1.7),title='Predictive information; 200 development groups')
    fig.suptitle('EXPLORATORY cost feasibility: prediction gains are not policy gains',fontsize=12)
    fig.tight_layout();fig.savefig(figdir/'information_and_observed_cost.png',dpi=200);fig.savefig(figdir/'information_and_observed_cost.pdf');plt.close(fig)


def finalize():
    m.require(read(OUT/'figure_review.json')['actually_inspected'],'Inspect rendered genuine figures before finalization')
    for name in ['findings.md','hypothesis_assessment.md','decision.md','next_experiment.md','unit_tests.json']:
        m.require((OUT/name).is_file(),'Missing final reviewed evidence: '+name)
    historical=historical_check(read(OUT/'review.json')['historical_index']);save(OUT/'historical_integrity.json',historical)
    inventory={rel(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in {'SHA256SUMS.json','verification.json','run.lock'}}
    inventory.update({p:sha(ROOT/p) for p in CODE});save(OUT/'SHA256SUMS.json',inventory)
    # A fresh process makes no use of a fitted model cache and cannot initialize GPU.
    r=subprocess.run([sys.executable,'-B','run_task_utility_observability.py','--audit-only'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    m.require(r.returncode==0,'Independent read-only audit failed: '+r.stderr)
    record=json.loads(r.stdout);m.require(record['passed'],'Audit did not pass');save(OUT/'verification.json',record)
    save(OUT/'status.json',dict(stage='completed',status='verified',scope=SCOPE,at_utc=now()))
    inventory[rel(OUT/'status.json')]=sha(OUT/'status.json');save(OUT/'SHA256SUMS.json',inventory)
    print(r.stdout)
