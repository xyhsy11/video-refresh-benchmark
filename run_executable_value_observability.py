"""Experiment 26: authenticated retrospective CPU diagnostic, explicit new paths.

Modes: preflight, run, resume, verify. Never calls an old runner or an encoder.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits, threadpool_info
from executable_value_observability import (PERIODIC, VIEWS, SIZES, stable, clip_weights,
    make_folds, hog, trajectory, observation, ridge_fit, score, actions, feature_names)

ROOT = Path(__file__).resolve().parent
OLD = ROOT/'results/continuation_value_gap_v1'
DEFAULT = ROOT/'results/executable_value_observability_v1'
IMPLEMENTATION = ['executable_value_observability.py', 'run_executable_value_observability.py',
                  'verify_executable_value_observability.py', 'render_executable_value_observability.py',
                  'test_executable_value_observability.py']
PURE = ['causal_observability_policy.py', 'causal_observability_common.py',
        'causal_observability_population_v2.py', 'task_utility_observability.py', 'research_utils.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.writing')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, p)


def event(out, kind, **kw):
    with (out/'resource_ledger.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(at_utc=now(), kind=kind, **kw))+'\n')


def csvwrite(p, rows):
    if not rows:
        return
    with (gzip.open(p, 'wt', newline='', encoding='utf-8') if str(p).endswith('.gz') else open(p, 'w', newline='', encoding='utf-8')) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf-8').strip()


def load_metadata():
    p = read(OLD/'hmdb_population.json')
    meta = [v for v in p['videos'] if v['role'] == 'EVAL']
    with gzip.open(ROOT/'results/causal_observability_population_v2/screening_evidence.jsonl.gz', 'rt') as f:
        screens = {r['key']: r for r in map(json.loads, f)}
    return p, meta, screens


def feature_record(m):
    name = hashlib.sha256(m['clip'].encode()).hexdigest()+'.npz'
    p = ROOT/'data/source/continuation_value_gap_v1/features'/name
    return p, p.with_suffix('.json')


def check_seal(path):
    for name, expected in read(path)['sha256'].items():
        assert sha(ROOT/name) == expected, f'Frozen identity changed: {name}'


def history_fingerprint():
    paths = git('ls-files', '-z').split('\0')
    return {p: sha(ROOT/p) for p in paths if p and (ROOT/p).is_file() and
            (p.startswith('results/') or p.endswith('.py') or p.endswith('AGENTS.md')) and
            not p.startswith('results/executable_value_observability_v1/') and p not in IMPLEMENTATION}


def preflight(out):
    if (out/'design_seal.json').exists():
        check_seal(out/'design_seal.json'); print('Existing sealed design authenticated; retained.', flush=True); return
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Refusing to overwrite an existing unsealed experiment directory')
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    save(out/'checkout_audit.json', dict(at_utc=now(), head=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
        working_tree=git('status', '--short'), experiments=sorted(p.name for p in (ROOT/'results').iterdir() if p.is_dir()),
        requested_reviewed_commit='5c43513d7d56719972dc2e9d9a733f65dd6075fb',
        reviewed_commit_present=subprocess.run(['git', 'cat-file', '-e', '5c43513d7d56719972dc2e9d9a733f65dd6075fb^{commit}'], cwd=ROOT, capture_output=True).returncode == 0,
        authority='Actual local checkout; no fetch/pull/switch/reset/stash; direct final user instruction authorizes commit/push',
        applicable_AGENTS=[str(p) for p in [Path(ROOT.anchor)/'AGENTS.md', ROOT/'AGENTS.md', ROOT/'results/AGENTS.md', out/'AGENTS.md'] if p.exists()],
        completion_conflict='No newer completed executable-value experiment found in existing result directories'))
    save(out/'historical_fingerprint.json', history_fingerprint())
    versions = {}
    for name in ['numpy', 'scipy', 'av', 'Pillow', 'scikit-learn', 'matplotlib', 'threadpoolctl']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'UNAVAILABLE'
    save(out/'environment.json', dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        libraries=versions, threads=1, threadpools=threadpool_info(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        decoder_threads=1, decoder_thread_type='NONE', no_package_changes=True))
    # Only permitted Experiment 25 outcomes and input-only metadata are inventoried.
    required = [OLD/n for n in ['hmdb_population.json', 'hmdb_head.json', 'hmdb_head_fit.json', 'hmdb_encoder.json',
        'hmdb_input_manifest.json', 'hmdb_eval_collection.json', 'hmdb_train_collection.json', 'hmdb_population_seal.json',
        'hmdb_evaluation_seal.json', 'hmdb_first_test_exposure.json', 'active_implementation_seal.json', 'design_seal.json',
        'hmdb/outcome_table.npz', 'hmdb/population.json', 'hmdb/outcome_seal.json', 'hmdb/per_state.csv.gz']]
    required += [ROOT/'results/causal_observability_population_v2'/n for n in
        ['screening_evidence.jsonl.gz', 'source_family_audit.json', 'allocation_manifest.json', 'screening_identity.json']]
    missing = [str(p.relative_to(ROOT)).replace('\\', '/') for p in required if not p.is_file()]
    save(out/'input_inventory.json', dict(status='INVENTORY_IN_PROGRESS', missing=missing))
    if missing:
        raise FileNotFoundError('BLOCKED_INPUT: required manifests missing; see input_inventory.json')
    pop, meta, screens = load_metadata()
    assert len(meta) == 1530 and len({m['family'] for m in meta}) == 478 and len(pop['classes']) == 51
    train = [v for v in pop['videos'] if v['role'] == 'HEAD_TRAIN']; assert len(train) == 2754
    assert not ({m['family'] for m in meta} & {m['family'] for m in train})
    assert read(OLD/'hmdb/population.json') == [{k: m[k] for k in ['clip', 'label', 'family', 'y']} for m in meta]
    from causal_observability_population_v2 import graph_replay
    audit = read(ROOT/'results/causal_observability_population_v2/source_family_audit.json')
    mapping = graph_replay(list(screens.values()), audit)
    assert all(mapping[m['clip']] == m['family'] for m in pop['videos'])
    # Metadata-only confirmation gate, saved independently before empirical execution.
    occupied = {m['family'] for m in pop['videos']}
    unused = [s for s in screens.values() if s['official_flag'] == 0 and s['new_eligible'] and mapping[s['key']] not in occupied]
    counts = Counter(s['class_name'] for s in unused)
    feasibility = dict(status='BLOCKED_INPUT_FOR_CONFIRMATION', metadata_only=True,
        isolated_official_unused_clips=len(unused), classes=len(counts), global_components=len({mapping[s['key']] for s in unused}),
        per_class=dict(sorted(counts.items())), missing_classes=sorted(set(pop['classes'])-set(counts)),
        blockers=['No retained untouched all-51-class population isolated from all head/probe/design/evaluation components',
                  'Original parent URLs and unknown duplicates/reuploads not resolved',
                  'Official-unused eligibility and known graph isolation do not certify untouched outcome/design history',
                  'No prospective adequate independent component/precision certificate'],
        retained_original_roles=True, cross_fitting_does_not_restore_untouched_test=True)
    save(out/'fresh_confirmation_feasibility.json', feasibility)
    (out/'fresh_confirmation_feasibility.md').write_text(
        '# Metadata-only fresh-confirmation gate\n\nBLOCKED_INPUT_FOR_CONFIRMATION.\n\n'+
        f'{len(unused)} eligible official-unused clips / {len(counts)} classes survive known component exclusion of all Experiment 25 fitting/evaluation sources. '+
        'This fails all-51-class coverage before any outcomes are opened. Original parent/reupload and prior-design exposure certification are also missing. '+
        'This gate does not block the authorized exposed-HMDB retrospective diagnostic.\n', encoding='utf-8')
    for name in ['hmdb_population_seal.json', 'hmdb_evaluation_seal.json', 'hmdb/outcome_seal.json', 'active_implementation_seal.json']:
        check_seal(OLD/name)
    for name, expected in read(OLD/'hmdb_input_manifest.json').items():
        p = ROOT/name
        if not p.is_file(): missing.append(name)
        else: assert sha(p) == expected, f'Input hash mismatch {name}'
    records = []; collection = read(OLD/'hmdb_eval_collection.json'); encsha = sha(OLD/'hmdb_encoder.json')
    for i, m in enumerate(meta):
        fp, ip = feature_record(m); vp = ROOT/m['path']; screen = screens[m['clip']]
        for p in (fp, ip, vp):
            if not p.is_file(): missing.append(p.relative_to(ROOT).as_posix())
        if not all(p.is_file() for p in (fp, ip, vp)): continue
        identity = read(ip)
        assert identity == collection[m['clip']]
        assert identity['path'] == fp.relative_to(ROOT).as_posix()
        assert identity['clip'] == m['clip'] and identity['role'] == 'EVAL'
        assert sha(fp) == identity['feature_sha256'] and sha(vp) == m['input_sha256'] == screen['sha256'] == identity['input_sha256']
        assert digest(screen['grid']) == m['grid_sha256'] == identity['grid_sha256']
        assert identity['encoder_sha256'] == encsha
        with np.load(fp, allow_pickle=False) as a:
            assert a['features'].shape == (16, 2048) and np.isfinite(a['features']).all()
        assert screen['grid']['endpoint_indices'] == np.floor(np.linspace(3, m['decoded_frames']-1, 16)).astype(int).tolist()
        records.append(dict(**m, feature_path=fp.relative_to(ROOT).as_posix(), feature_sha256=sha(fp),
                            feature_identity_path=ip.relative_to(ROOT).as_posix(), feature_identity_sha256=sha(ip),
                            endpoint_indices=screen['grid']['endpoint_indices']))
        if i % 300 == 0: print(f'Authenticated E26 inputs {i}/1530', flush=True)
    inventory = dict(at_utc=now(), status='BLOCKED_INPUT' if missing else 'AUTHENTICATED', missing=sorted(set(missing)),
        manifest_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in required}, clips=records,
        training_population_metadata_sha256=sha(OLD/'hmdb_population.json'), train_count=len(train),
        population_clips=len(meta), global_components=478, classes=51, head_train_components_disjoint=True)
    save(out/'input_inventory.json', inventory)
    if missing: raise FileNotFoundError('BLOCKED_INPUT: exact missing ignored files in input_inventory.json')
    head = read(OLD/'hmdb_head.json'); assert head['classes'] == pop['classes']
    save(out/'source_head_audit.json', dict(head_sha256=sha(OLD/'hmdb_head.json'), encoder_sha256=encsha,
        encoder=read(OLD/'hmdb_encoder.json'), scaler_shape=[2048], weight_shape=list(np.array(head['parameters']['weights']).shape),
        classes=head['classes'], train_clips=len(train), train_components=len({m['family'] for m in train}),
        eval_clips=1530, eval_components=478, overlap_components=[], graph_sha256=sha(ROOT/'results/causal_observability_population_v2/source_family_audit.json'),
        known_graph_replayed=True, unknown_parent_reupload_pretraining_overlap_unresolved=True))
    folds, mapping, diag = make_folds(meta)
    save(out/'fold_manifest.json', dict(seed=20261022, rule='Greedy metadata-only normalized squared class clip and component-load balance; stable hash ties',
        global_component_to_fold=mapping, clips=[dict(clip=m['clip'], family=m['family'], fold=int(f)) for m, f in zip(meta, folds)], diagnostics=diag))
    components = sorted(mapping, key=lambda g: stable('timing-component', g))[:60]
    timing = [min((m for m in meta if m['family'] == g), key=lambda m: stable('timing-clip', m['clip'])) for g in components]
    save(out/'timing_manifest.json', dict(seed=20261022, components=60, clips=[m['clip'] for m in timing],
        class_coverage=len({m['label'] for m in timing}), classes=sorted({m['label'] for m in timing}),
        warmups=2, measured_repetitions=5, interleaving='Stable hash of timing-order, clip, repetition, arm; repetition -2,-1 warmups'))
    donors = {}
    for i, m in enumerate(meta):
        candidates = [j for j, n in enumerate(meta) if folds[j] == folds[i] and n['family'] != m['family']]
        donors[m['clip']] = meta[min(candidates, key=lambda j: stable('donor', m['clip'], meta[j]['clip']))]['clip']
    save(out/'shuffled_hog_manifest.json', dict(rule='Whole held-out trajectory, same fold, different global component, stable hash; no target use', donors=donors))
    save(out/'exposure_manifest.json', dict(at_utc=now(), status='DEVELOPMENT / EXPLORATORY',
        original_manifest_sha256=sha(OLD/'hmdb_population.json'), historical_roles_unchanged=True,
        previous_role='EVAL', new_role='EXPOSED_EXPLORATORY_PROBE_CROSSFIT', clips=[m['clip'] for m in meta],
        prior_first_exposure=read(OLD/'hmdb_first_test_exposure.json'),
        statement='All 1530 former EVAL clips reused for diagnostic probe fitting and OOF evaluation; overlapping training folds; no untouched-test claim'))
    config = dict(study='executable_value_observability_v1', seed=20261022, N=16, K=4, periodic=list(PERIODIC),
        states=list(range(1, 13)), pooling='C', target='D_P=Q_P(HOLD)-Q_P(REFRESH)',
        continuation='Exact Experiment 25 periodic repair at ORIGINAL reset prefixes; alternative single interventions',
        ridge=dict(alpha=1., intercept='unpenalized', objective='(1/(2*n))*sum(mean-one weights*(y-intercept-z*beta)^2)+(alpha/2)*sum(beta^2)',
                   scaling='training-only weighted mean/population SD; SD<=1e-12 -> 1; mean-one weights', final_refit=False),
        tree=dict(max_depth=3, min_samples_leaf=120, random_state=20261022, criterion='squared_error'),
        decision='prediction>1e-12 REFRESH; prediction<-1e-12 HOLD; otherwise original Periodic action',
        views=SIZES, hog=dict(shape=[32,32], scale='uint8/255', gradient='central differences /2; replicate edge pixels',
            cells=[4,4], cell_size=[8,8], bins=9, orientation='unsigned [0,pi); linear circular bin interpolation; no spatial interpolation',
            normalization='per-cell h/sqrt(sum(h*h)+epsilon^2), epsilon=1e-6',
            initialization='zero gradient -> zero descriptor; candidate zero initializes both references; refresh reference updated after anchor decision',
            transitions='row-major 16 L2 from last actually refreshed thumbnail, then row-major 16 from previous candidate'),
        grayscale='Exact unmodified causal_observability_policy.cheap_signal on PyAV reformat(width=32,height=32,format=gray)',
        evaluation_weights='class -> class-family -> clip -> mean of 12 reset-state consequences; fold subsets restrict and renormalize full weights',
        training_weights='same hierarchy recomputed using only training partition; rescale to mean one',
        timing=dict(arms=dict(A='Fresh encoded-file read + CPU full decode + authenticated presentation/grid path',
            B='A + earlier refreshed history construction + held-out ridge H scoring',
            C='A + history + all16 grayscale conversion/statistics + held-out ridge HG scoring',
            D='A + history + all16 grayscale/HOG acquisition + held-out ridge HGH scoring'),
            threads=1, common_file_read_included=True, saved_expensive_history_and_models_loaded_before_timer=True,
            disk_cache='OS cache not flushed; fresh byte read and fresh decode each trial; no decoded-frame reuse',
            expensive_encoder_absent=True, score_states=list(range(1,13)), thumbnails=list(range(16)),
            repetitions_collapsed='Within-clip median of paired repetition differences; summarize median and p95 across 60 clips'),
        decision_rules=dict(practical_nats=.01, median_total_added_ms=10.,
            promising='ridge delta_HOG>=.01 AND V(HGH)>0 AND >=4 favorable folds AND paired median D-A<=10ms',
            deprioritize='V(HGH)<=0 OR ridge delta_HOG<0 OR paired median D-A>10ms', otherwise='SMALL_OR_UNSTABLE_SIGNAL'),
        caps=dict(elapsed_seconds=14400, additional_retained_bytes=2*2**30, expensive_encoder_forwards=0, gpu_execution=0,
                  recognition_head_fits=0, controller_fits=0, downloads=0),
        uncertainty='Finite cohort and fold stability; no bootstrap CI, no confirmatory inference; no equivalence or impossibility claims')
    save(out/'frozen_config.json', config)
    save(out/'feature_dictionary.json', dict(order=feature_names(pop['classes']), sizes=SIZES,
        deterministic_state='At original prefixes cache_age=t-last anchor; remaining_budget=4-ceil(t/4); remaining_candidates=16-t. All determined by position. Historical state fields repeat these deterministically.',
        forbidden='Labels, true class identity, source/filename, current/future expensive features/logits, future pixels, outcomes, actions'))
    protocol = '''# Experiment 26: executable_value_observability_v1

DEVELOPMENT / EXPLORATORY. Bounded retrospective diagnostic on all 1,530 exposed HMDB E25 clips, 478 known global source components, all 51 classes. No outcome-dependent exclusions, searches, controller fitting, head fitting or expensive encoder execution. The metadata-only fresh-confirmation gate is separate.

The machine-readable frozen_config.json is normative for estimator objective, scaling, observation order, HOG, timing and decision rules. The frozen fold, timing, donor, input, exposure and feature manifests accompany it. Five component-separated folds use seed 20261022 and metadata balance; every training fold retains all classes/components per class. The frozen 51-class head/scalers/order, 16x2048 features and exact E25 grids are unchanged.

D_P is final true-label CE(HOLD) minus CE(REFRESH) under E25's exact periodic-repair continuation, at twelve ORIGINAL Periodic prefixes. Independently hash-selected real clips replay all455 C schedules through the frozen head. Every predicted action selects ONE alternative complete exact-K4 schedule; prefixes reset between states. Average twelve consequences within clip, then clips within class/global-family, families within class, classes equally. Never combine actions or sum gains. Positive advantage favors REFRESH; numerical predictions within +/-1e-12 preserve Periodic.

S has position one-hots, age and remaining budget/candidates. H concatenates S and the unchanged historical K+12 vector plus discretionary flag with genuinely earlier refreshed features. HG adds the exact32 cheap_signal statistics; HGH adds exactly32 fixed HOG transitions. No descriptor/resolution search. Pixels are genuine retained-original CPU-decoded endpoints, accessed chronologically for observations; full-clip input-only grid planning and full decoding do not constitute a live-stream experiment. Current/future embeddings are offline target ingredients only. One held-out whole-trajectory shuffled-HOG donor per clip has a different component and the same fold; this is an input-reliance check, not a p-value.

Primary weighted ridge alpha1 has training-only standardization, unpenalized intercept, mean-one weights and normalized half-MSE plus alpha/2 L2. One fixed depth3/min-leaf120 tree is sensitivity only (UNAVAILABLE if absent). Both task advantage and cosine drift targets use identical rows/folds/weights/views. No final refit, threshold search, calibration, model choice or sequential rollout.

Primary contrast is RIDGE V(HGH)-V(HG). Report H-S, HG-H, HGH-H, HGH-Periodic, oracle regret/value, prediction MAE/rank/AP/AUROC, predicted-observed signed bins, complete-schedule top1 changes, benefits/harms, folds, classes, sources/concentration and leave-one-component influence. Fold effects restrict/renormalize full weights. All finite-cohort effects include adverse/small outcomes; no confirmatory confidence interval. .01 nats per mean single intervention is a practical reference, not a publication requirement.

Acquisition uses 60 input-hash-selected global components/one clip each: four fresh CPU-decoded arms A common path, B H+ridge, C HG+ridge, D HGH+ridge. Two warmups and five seeded interleaved measured repetitions, threads1. Candidate zero initializes references; actual chronological Periodic features arrive only after anchors. All16 thumbnail acquisitions follow historical cheap-path behavior, scoring only twelve discretionary states. Supplied expensive history and models are loaded before timing, encoder execution absent. Encoded file loading, decoding/order/grid checks, conversion, history, descriptors and scoring are logged. Paired differences per clip/repetition are collapsed within clip before reporting median/p95; nested timers are not summed. Median paired D-A <=10ms is provisional engineering reference. No claim about end-to-end adaptive latency or speedup.

Resource caps:4h additional elapsed,2GiB retained, zero expensive/GPU/head/controller/download work. Atomic checkpoints and hash-authenticated resume preserve exact design/population; cap pressure cannot shrink population. Missing inputs produce exact lists and BLOCKED_INPUT. Mechanical changes retain originals/seals and dated amendments. Historical 19/19.1/23A/23B gates and outputs remain unchanged. Direct final user instruction authorizes commit/push of new work after completion; no PR.
'''
    (out/'protocol.md').write_text(protocol, encoding='utf-8')
    (out/'prospective_experiment_26B_protocol.md').write_text('''# Prospective Experiment 26B — NOT EXECUTED

Independent confirmation remains BLOCKED_INPUT_FOR_CONFIRMATION. Before outcomes, authenticate an authorized, genuinely untouched all-51-class population, exact video/grid identity and real PTS. Evaluation global components must be isolated from the frozen head training, every E26 probe-fitting fold, all exposed design/evaluation cohorts and known duplicates/reuploads; preserve the conservative graph, resolve original-parent uncertainty, audit prior exposure, and document pretraining limitations. No favorable class subset or exposed-fold renaming. No acquisition/download/outcome execution is authorized here.

After the gate, freeze independent fitting/selection/evaluation roles with adequate per-class components and a source-based precision/resource design using metadata and fitting data only. Keep E26 fixed observations, HOG, estimator objective/alpha, head, class order, N16/K4/C, original Periodic prefixes, exact periodic repair and sign/tie rule. Fit any diagnostic probe only on separately declared fitting sources before confirmation. Retain all eligible clips and harms; no selection by correctness, gain or density. Freeze one primary ridge HGH-HG task-value contrast and report all supporting endpoints, source/fold/adverse-tail evidence. Establish a prospective independent-source uncertainty method accounting for fitting rather than reusing fixed-OOF intervals. .01 nats remains a practical reference.

Repeat authenticated CPU acquisition on a metadata-selected independent timing cohort, with fixed threads, two warmups/five interleaved repetitions, paired within-clip D-A/D-C,10ms provisional median reference. Measure actual observation-path costs; any later controller or full adaptive execution requires another protocol and the unchanged historical controller gates. Do not start a controller or fresh outcomes in Experiment26.
''', encoding='utf-8')
    files = [out/n for n in ['checkout_audit.json', 'historical_fingerprint.json', 'environment.json', 'input_inventory.json',
        'source_head_audit.json', 'fold_manifest.json', 'timing_manifest.json', 'shuffled_hog_manifest.json', 'exposure_manifest.json',
        'frozen_config.json', 'feature_dictionary.json', 'protocol.md', 'fresh_confirmation_feasibility.json', 'fresh_confirmation_feasibility.md',
        'prospective_experiment_26B_protocol.md']]+[ROOT/n for n in IMPLEMENTATION+PURE]
    save(out/'design_seal.json', dict(at_utc=now(), before_probe_fitting=True, before_new_decoding=True,
        sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    event(out, 'preflight_complete', seconds=time.perf_counter()-start, authenticated_clips=len(records), missing=0)
    print('PREFLIGHT COMPLETE; design sealed before fitting/decoding.', flush=True)


def check_budget(out):
    ledger = [json.loads(s) for s in (out/'resource_ledger.jsonl').read_text().splitlines()] if (out/'resource_ledger.jsonl').exists() else []
    starts = [e['at_utc'] for e in ledger if e['kind'] == 'work_started']
    if starts and (datetime.now(timezone.utc)-datetime.fromisoformat(starts[0])).total_seconds() > 14400:
        raise RuntimeError('PARTIAL: four-hour elapsed cap reached; population not reduced')
    total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if total > 2*2**30: raise RuntimeError('PARTIAL: retained-output cap reached')


def decode(m, screen, out, purpose):
    """Authenticated original bytes, no old collector, no synthetic frames."""
    import av
    tick = time.perf_counter(); event(out, 'decode_started', clip=m['clip'], purpose=purpose)
    times = dict(load_seconds=0., decode_seconds=0., grid_seconds=0.)
    try:
        tick1 = time.perf_counter(); encoded = (ROOT/m['path']).read_bytes()
        assert hashlib.sha256(encoded).hexdigest() == m['input_sha256']
        times['load_seconds'] = time.perf_counter()-tick1
        tick1 = time.perf_counter(); frames = []; identities = []; pts = []
        with av.open(io.BytesIO(encoded)) as container:
            stream = container.streams.video[0]; stream.codec_context.thread_count = 1
            stream.codec_context.thread_type = 'NONE'; stream.codec_context.skip_frame = 'NONE'
            for i, frame in enumerate(container.decode(stream)):
                assert frame.pts is not None and frame.time_base is not None
                frames.append(frame); pts.append(float(frame.pts*frame.time_base))
                identities.append(dict(index=i, pts=int(frame.pts), time_base=str(frame.time_base)))
            times['decode_seconds'] = time.perf_counter()-tick1; tick1 = time.perf_counter()
            order = np.argsort(pts, kind='stable'); ordered = [{**identities[int(j)], 'presentation_index': i} for i, j in enumerate(order)]
            assert len(frames) == m['decoded_frames'] and len(set(pts)) == len(pts)
            assert ordered == screen['presentation_identities']
            from causal_observability_population_v2 import adaptive_grid
            grid = adaptive_grid(np.asarray(pts)[order], ordered)
            assert grid == screen['grid'] and digest(grid) == m['grid_sha256']
            frames = [frames[int(i)] for i in order]; times['grid_seconds'] = time.perf_counter()-tick1
        event(out, 'decode_complete', clip=m['clip'], purpose=purpose, frames=len(frames), seconds=time.perf_counter()-tick, **times)
        return frames, grid, times
    except Exception as e:
        event(out, 'decode_failed', clip=m['clip'], purpose=purpose, seconds=time.perf_counter()-tick, error=repr(e)); raise


def load_head():
    return {k: np.asarray(v) for k, v in read(OLD/'hmdb_head.json')['parameters'].items()}


def extracted(out, meta, screens):
    dst = out/'observations'; dst.mkdir(exist_ok=True); head = load_head(); allx = []; drifts = []
    inventory = []
    for i, m in enumerate(meta):
        check_budget(out)
        p = dst/(hashlib.sha256(m['clip'].encode()).hexdigest()+'.npz'); identity = p.with_suffix('.json')
        fp, _ = feature_record(m)
        with np.load(fp, allow_pickle=False) as z: f = z['features'].astype(np.float64)
        if p.exists() and identity.exists():
            record = read(identity); assert record['sha256'] == sha(p) and record['design_sha256'] == sha(out/'design_seal.json')
            with np.load(p, allow_pickle=False) as a: x = a['x']; drift = a['drift']
        else:
            frames, grid, tm = decode(m, screens[m['clip']], out, 'observation_extraction')
            thumbs = np.stack([frames[j].reformat(width=32, height=32, format='gray').to_ndarray().copy() for j in grid['endpoint_indices']])
            # API receives only features at the four Periodic anchors; it gates past access at each state.
            x = trajectory(thumbs, {u: f[u] for u in PERIODIC}, head)
            drift = np.array([1-f[t]@f[max(u for u in PERIODIC if u < t)]/
                max(np.linalg.norm(f[t])*np.linalg.norm(f[max(u for u in PERIODIC if u < t)]), 1e-12) for t in range(1,13)])
            with p.with_suffix('.writing').open('wb') as dest: np.savez_compressed(dest, x=x, drift=drift, thumbnails=thumbs)
            os.replace(p.with_suffix('.writing'), p)
            record = dict(clip=m['clip'], sha256=sha(p), design_sha256=sha(out/'design_seal.json'),
                original_sha256=m['input_sha256'], grid_sha256=m['grid_sha256'], shape=list(x.shape), genuine_endpoint_thumbnails=16)
            save(identity, record); del frames
        assert x.shape == (12,147) and np.isfinite(x).all() and np.isfinite(drift).all()
        allx.append(x); drifts.append(drift); inventory.append(record)
        if i % 100 == 0: print(f'CPU decoded/extracted {i}/1530', flush=True)
    save(out/'observation_inventory.json', inventory)
    return np.array(allx), np.array(drifts)


def targets(meta, out):
    """Reuse sealed E25 per-state targets; independently compare schedule identities."""
    with np.load(OLD/'hmdb/outcome_table.npz', allow_pickle=False) as z: table = {k: z[k] for k in z.files}
    with gzip.open(OLD/'hmdb/per_state.csv.gz', 'rt', newline='') as f: states = list(csv.DictReader(f))
    q = np.array([[float(r['Q_P_H']), float(r['Q_P_R'])] for r in states]).reshape(1530,12,2)
    y = np.array([float(r['D_P']) for r in states]).reshape(1530,12)
    positions = table['positions']; lookup = {tuple(map(int, p)): i for i, p in enumerate(positions)}
    ids = np.array([[int(r['hold_schedule']), int(r['refresh_schedule'])] for r in states]).reshape(1530,12,2)
    for i, m in enumerate(meta):
        for t in range(1,13):
            r = states[i*12+t-1]; assert r['clip'] == m['clip'] and int(r['t']) == t
            assert [v for v in positions[ids[i,t-1,0]] if v < t] == [v for v in PERIODIC if v < t]
            assert [v for v in positions[ids[i,t-1,1]] if v < t] == [v for v in PERIODIC if v < t]
            assert np.max(abs(q[i,t-1]-table['loss'][i, ids[i,t-1]])) < 1e-12
    assert np.max(abs(y-(q[:,:,0]-q[:,:,1]))) < 1e-12
    pid = lookup[PERIODIC]; baseline = table['loss'][:,pid]
    predictions = np.array([table['prediction'][i, ids[i]] for i in range(len(meta))])
    return y, q, baseline, predictions, table['prediction'][:,pid]


def probe(out, x, y, drift, meta, folds):
    modeldir = out/'models'; modeldir.mkdir(exist_ok=True)
    preds = {target: {family: {view: np.full_like(y, np.nan) for view in VIEWS} for family in ['ridge', 'tree']} for target in ['task','drift']}
    try:
        from sklearn.tree import DecisionTreeRegressor
        tree_available = True
    except ImportError:
        tree_available = False
    save(out/'tree_availability.json', dict(status='AVAILABLE' if tree_available else 'UNAVAILABLE'))
    for f in range(5):
        a = folds != f; b = ~a; trainmeta = [m for m,k in zip(meta,a) if k]
        w = np.repeat(clip_weights(trainmeta)/12,12); w = w/w.mean()
        for target, values in [('task',y), ('drift',drift)]:
            for view in VIEWS:
                check_budget(out); size = SIZES[view]
                trainx = x[a,:,:size].reshape(-1,size); testx = x[b,:,:size].reshape(-1,size)
                yy = values[a].ravel(); p = modeldir/f'{target}_ridge_{view}_fold{f}.json'
                if p.exists():
                    model = read(p)
                    assert model['design_sha256'] == sha(out/'design_seal.json')
                else:
                    tick = time.perf_counter(); event(out, 'probe_fit_started', family='ridge', target=target, view=view, fold=f)
                    model = ridge_fit(trainx, yy, w); model['design_sha256'] = sha(out/'design_seal.json'); save(p, model)
                    event(out, 'probe_fit_complete', family='ridge', target=target, view=view, fold=f, seconds=time.perf_counter()-tick)
                preds[target]['ridge'][view][b] = score(model,testx).reshape(-1,12)
                if tree_available:
                    tp = modeldir/f'{target}_tree_{view}_fold{f}.npz'
                    # Persist only genuine held-out predictions; no deployment/final refit.
                    if tp.exists():
                        assert read(tp.with_suffix('.json'))['design_sha256'] == sha(out/'design_seal.json')
                        with np.load(tp) as z: prediction = z['prediction']
                    else:
                        tick = time.perf_counter(); event(out, 'probe_fit_started', family='tree', target=target, view=view, fold=f)
                        tree = DecisionTreeRegressor(max_depth=3,min_samples_leaf=120,random_state=20261022,criterion='squared_error')
                        tree.fit(trainx,yy,sample_weight=w); prediction = tree.predict(testx).reshape(-1,12)
                        np.savez_compressed(tp,prediction=prediction)
                        save(tp.with_suffix('.json'), dict(design_sha256=sha(out/'design_seal.json'), sha256=sha(tp),
                            nodes=int(tree.tree_.node_count), leaves=int(tree.get_n_leaves())))
                        event(out, 'probe_fit_complete', family='tree', target=target, view=view, fold=f, seconds=time.perf_counter()-tick)
                    preds[target]['tree'][view][b] = prediction
        print(f'OOF probes complete fold {f}', flush=True)
    if not tree_available:
        for target in preds: del preds[target]['tree']
    donors = read(out/'shuffled_hog_manifest.json')['donors']; lookup = {m['clip']:i for i,m in enumerate(meta)}
    shuffled = np.empty_like(y)
    for f in range(5):
        ids = np.flatnonzero(folds == f); xx = x[ids].copy()
        for j,i in enumerate(ids): xx[j,:,115:] = x[lookup[donors[meta[i]['clip']]],:,115:]
        shuffled[ids] = score(read(modeldir/f'task_ridge_HGH_fold{f}.json'),xx.reshape(-1,147)).reshape(-1,12)
    arrays = {f'{t}_{family}_{v}': a for t,p in preds.items() for family,p2 in p.items() for v,a in p2.items()}
    arrays.update(task_ridge_HGH_shuffled=shuffled, task_target=y, drift_target=drift, fold=folds)
    assert all(np.isfinite(a).all() for a in arrays.values())
    np.savez_compressed(out/'oof_predictions.npz', **arrays)
    save(out/'prediction_seal.json',dict(at_utc=now(),sha256={(out/'oof_predictions.npz').relative_to(ROOT).as_posix():sha(out/'oof_predictions.npz')}))
    return arrays


def timing(out, meta, screens, folds):
    head = load_head(); manifest = read(out/'timing_manifest.json'); allrows = []
    timingdir = out/'timing'; timingdir.mkdir(exist_ok=True)
    for ci,clip in enumerate(manifest['clips']):
        i = next(i for i,m in enumerate(meta) if m['clip'] == clip); m = meta[i]
        fp,_ = feature_record(m)
        with np.load(fp) as z: supplied = {u:z['features'][u].astype(float) for u in PERIODIC}
        models = {arm:read(out/'models'/f'task_ridge_{view}_fold{folds[i]}.json') for arm,view in [('B','H'),('C','HG'),('D','HGH')]}
        for rep in range(-2,5):
            for arm in sorted('ABCD',key=lambda a:stable('timing-order',clip,rep,a)):
                check_budget(out); p = timingdir/f'{hashlib.sha256(clip.encode()).hexdigest()}_{rep}_{arm}.json'
                if p.exists():
                    row = read(p); assert row['design_sha256'] == sha(out/'design_seal.json'); allrows.append(row); continue
                tick = time.perf_counter(); frames,grid,tm = decode(m,screens[clip],out,f'timing_{arm}_{rep}')
                conversion = graytime = hogtime = historytime = scoring = 0.; nscore = nthumb = 0
                previous = cached_hog = previous_hog = None; checksum = 0.; past = {}
                for t,j in enumerate(grid['endpoint_indices']):
                    if arm in 'CD':
                        st = time.perf_counter(); thumb = frames[j].reformat(width=32,height=32,format='gray').to_ndarray().copy()
                        conversion += time.perf_counter()-st; nthumb += 1
                        st = time.perf_counter()
                        from causal_observability_policy import cheap_signal
                        gray = cheap_signal(thumb,previous); graytime += time.perf_counter()-st
                        if arm == 'D':
                            st = time.perf_counter(); desc = hog(thumb)
                            if cached_hog is None: cached_hog = desc; previous_hog = desc
                            changes = np.r_[np.linalg.norm(desc-cached_hog,axis=1),np.linalg.norm(desc-previous_hog,axis=1)]
                            hogtime += time.perf_counter()-st
                    if arm != 'A' and 1 <= t <= 12:
                        st = time.perf_counter(); xx = observation(t,past,head,
                            gray if arm in 'CD' else None, changes if arm == 'D' else None)
                        historytime += time.perf_counter()-st
                        st = time.perf_counter(); checksum += float(score(models[arm],xx)); scoring += time.perf_counter()-st; nscore += 1
                    if arm in 'CD': previous = thumb
                    if arm == 'D':
                        previous_hog = desc
                        if t in PERIODIC: cached_hog = desc
                    if arm != 'A' and t in PERIODIC: past[t] = supplied[t]
                total = time.perf_counter()-tick
                row = dict(clip=clip,family=m['family'],label=m['label'],fold=int(folds[i]),repetition=rep,warmup=rep<0,arm=arm,
                    status='COMPLETE',total_seconds=total,**tm,conversion_seconds=conversion,gray_seconds=graytime,hog_seconds=hogtime,
                    history_seconds=historytime,scoring_seconds=scoring,frames=len(frames),thumbnails=nthumb,scores=nscore,
                    score_checksum=checksum,design_sha256=sha(out/'design_seal.json'))
                save(p,row); allrows.append(row); del frames
        if ci % 5 == 0: print(f'Fresh CPU timing complete {ci+1}/60 clips',flush=True)
    csvwrite(out/'timing_raw_trials.csv',allrows)
    measured = [r for r in allrows if not r['warmup']]; perclip = []; pairs = []
    for clip in manifest['clips']:
        r = [v for v in measured if v['clip'] == clip]
        table = {(v['repetition'],v['arm']):v for v in r}
        result = dict(clip=clip)
        for arm in 'ABCD': result[f'{arm}_median_ms'] = float(np.median([v['total_seconds']*1000 for v in r if v['arm'] == arm]))
        for name,a,b in [('D_minus_A','D','A'),('D_minus_C','D','C'),('C_minus_B','C','B'),('B_minus_A','B','A')]:
            ds = []
            for rep in range(5):
                value = 1000*(table[rep,a]['total_seconds']-table[rep,b]['total_seconds']); ds.append(value)
                pairs.append(dict(clip=clip,repetition=rep,contrast=name,paired_ms=value))
            result[name+'_median_ms'] = float(np.median(ds))
        perclip.append(result)
    csvwrite(out/'timing_per_clip.csv',perclip); csvwrite(out/'timing_paired_differences.csv',pairs)
    summary = {k:dict(median=float(np.median([r[k] for r in perclip])), p95=float(np.quantile([r[k] for r in perclip],.95))) for k in perclip[0] if k != 'clip'}
    summary.update(clips=60,components=60,class_coverage=manifest['class_coverage'],measured_trials=len(measured),warmup_trials=len(allrows)-len(measured),
        boundaries='Fresh encoded byte read/hash, decode and grid included; models/saved expensive anchor history preloaded, expensive encoder absent; ledger I/O included in outer timer; no decoded-frame reuse; warm OS file cache possible',
        interpretation='Observation-path CPU wall costs only. Within-clip median of paired repetition differences then cohort median/p95; repeats are not sources. Component timers nested in totals; never sum nested medians.')
    save(out/'timing_summary.json',summary); return summary


def summarize(out, meta, arrays, q, baseline, schedule_prediction, baseline_prediction):
    from task_utility_observability import rank_auc, weighted_spearman
    w = clip_weights(meta); sw = np.repeat(w/12,12); folds = arrays['fold']; y = arrays['task_target']
    positions = np.tile(np.arange(1,13),(len(meta),1)); labels = np.array([m['y'] for m in meta])
    periodic_correct = baseline_prediction == labels; rows = []; clips = []; bins = []; foldrows = []; sourcerows = []; classrows = []
    value_by_name = {}; regret_by_name = {}; accuracy_by_name = {}; tails = []
    def quantiles(values, weights):
        order=np.argsort(values,kind='stable'); cumulative=np.cumsum(weights[order]); cumulative/=cumulative[-1]
        return [float(values[order[min(np.searchsorted(cumulative,p),len(order)-1)]]) for p in [0,.01,.05,.5,.95,.99,1]]
    for name,pred in arrays.items():
        if not name.startswith(('task_ridge_','task_tree_','drift_ridge_','drift_tree_')): continue
        target = arrays['task_target'] if name.startswith('task_') else arrays['drift_target']
        roc,ap = rank_auc((target.ravel()>0).astype(float),pred.ravel(),sw)
        row = dict(model=name,MAE=float(sw@abs(pred.ravel()-target.ravel())),
            RMSE=float(np.sqrt(sw@np.square(pred.ravel()-target.ravel()))),
            Spearman=weighted_spearman(target.ravel(),pred.ravel(),sw),AUROC_positive=roc,AP_positive=ap,
            positive_prevalence=float(sw@(target.ravel()>0)),signed_bias=float(sw@(pred-target).ravel()))
        if name.startswith('task_'):
            act = actions(pred,positions); selected = np.take_along_axis(q,act[:,:,None],axis=2)[:,:,0]
            value = (baseline[:,None]-selected).mean(1); regret = (selected-q.min(2)).mean(1)
            correct = np.take_along_axis(schedule_prediction,act[:,:,None],axis=2)[:,:,0] == labels[:,None]
            acc = correct.mean(1); value_by_name[name] = value; regret_by_name[name] = regret; accuracy_by_name[name] = acc
            tails.append(dict(model=name,probabilities=[0,.01,.05,.5,.95,.99,1],
                clip_gain_quantiles=quantiles(value,w),state_gain_quantiles=quantiles((baseline[:,None]-selected).ravel(),sw),
                clip_regret_quantiles=quantiles(regret,w),
                harmful_clips=int((value < -1e-12).sum()),beneficial_clips=int((value > 1e-12).sum())))
            row.update(value=float(w@value),regret=float(w@regret),top1_change=float(w@(acc-periodic_correct)),
                correct_to_wrong=float(sw@(periodic_correct[:,None]&~correct).ravel()),
                wrong_to_correct=float(sw@(~periodic_correct[:,None]&correct).ravel()),
                benefit_state_mass=float(sw@((baseline[:,None]-selected).ravel()>1e-12)),
                harm_state_mass=float(sw@((baseline[:,None]-selected).ravel() < -1e-12)),
                clip_gain_ge_point01_mass=float(w@(value>=.01)),clip_gain_le_minus_point01_mass=float(w@(value<=-.01)))
            for i,m in enumerate(meta):
                clips.append(dict(model=name,clip=m['clip'],label=m['label'],family=m['family'],fold=int(folds[i]),weight=float(w[i]),
                    value=float(value[i]),regret=float(regret[i]),top1_change=float(acc[i]-periodic_correct[i]),
                    min_state_gain=float((baseline[i]-selected[i]).min()),max_state_gain=float((baseline[i]-selected[i]).max())))
            for f in range(5):
                mask = folds == f; ww = w[mask]/w[mask].sum()
                foldrows.append(dict(model=name,fold=f,classes=len({m['label'] for m,k in zip(meta,mask) if k}),
                    clips=int(mask.sum()),population_weight=float(w[mask].sum()),value=float(ww@value[mask]),regret=float(ww@regret[mask])))
            for field,destination in [('label',classrows),('family',sourcerows)]:
                for group in sorted({m[field] for m in meta}):
                    mask = np.array([m[field] == group for m in meta]); ww = w[mask]/w[mask].sum()
                    destination.append(dict(model=name,group=group,clips=int(mask.sum()),weight=float(w[mask].sum()),value=float(ww@value[mask]),regret=float(ww@regret[mask])))
        rows.append(row)
        # Fixed signed bins, not selected or calibrated from outcomes.
        edges = [-np.inf,-.1,-.03,-.01,0.,.01,.03,.1,np.inf]
        for low,high in zip(edges[:-1],edges[1:]):
            mask = (pred.ravel()>=low)&(pred.ravel()<high); mass = sw[mask].sum()
            bins.append(dict(model=name,lower=str(low),upper=str(high),states=int(mask.sum()),weight=float(mass),
                predicted=None if mass == 0 else float(sw[mask]@pred.ravel()[mask]/mass),
                observed=None if mass == 0 else float(sw[mask]@target.ravel()[mask]/mass)))
    contrasts = [('H_minus_S','H','S'),('HG_minus_H','HG','H'),('HGH_minus_HG','HGH','HG'),('HGH_minus_H','HGH','H')]
    comparisons = []; influence = []
    for family in ['ridge','tree']:
        if f'task_{family}_S' not in value_by_name: continue
        for title,a,b in contrasts:
            delta = value_by_name[f'task_{family}_{a}']-value_by_name[f'task_{family}_{b}']
            effects = []
            for f in range(5):
                mask = folds == f; effects.append(float(w[mask]@delta[mask]/w[mask].sum()))
            comparisons.append(dict(family=family,contrast=title,value_difference=float(w@delta),fold_effects=effects,favorable_folds=sum(d>0 for d in effects)))
            # Recompute class->family->clip weights after omission; mark any missing class undefined.
            for g in sorted({m['family'] for m in meta}):
                mask = np.array([m['family'] != g for m in meta]); mm = [m for m,k in zip(meta,mask) if k]
                effect = None if len({m['label'] for m in mm}) != 51 else float(clip_weights(mm)@delta[mask])
                influence.append(dict(family=family,contrast=title,omitted_component=g,effect=effect))
    groups = sorted({m['family'] for m in meta}); masses = [float(w[[m['family']==g for m in meta]].sum()) for g in groups]
    oracle_value = float(w@((baseline[:,None]-q.min(2)).mean(1)))
    summary = dict(status='DEVELOPMENT / EXPLORATORY',population_clips=1530,states=18360,classes=51,components=478,
        metrics=rows,comparisons=comparisons,periodic_value=0.,matched_tail_privileged_oracle_value=oracle_value,matched_tail_privileged_oracle_regret=0.,
        periodic_regret=oracle_value,component_concentration=dict(maximum_weight=max(masses),sum_squared_weights=sum(v*v for v in masses),effective_components=1/sum(v*v for v in masses)),
        influence_ranges=[dict(family=f,contrast=c,minimum=min(r['effect'] for r in influence if r['family']==f and r['contrast']==c and r['effect'] is not None),
            maximum=max(r['effect'] for r in influence if r['family']==f and r['contrast']==c and r['effect'] is not None)) for f in ['ridge','tree'] if f'task_{f}_S' in value_by_name for c,_,_ in contrasts],
        uncertainty='Finite-cohort OOF effects; folds share fitting data; no confirmatory intervals or refitting-uncertainty claim',
        drift_interpretation='Cosine drift target uses current expensive feature only offline; drift score is never an action rule')
    save(out/'summary.json',summary)
    save(out/'benefit_harm_distributions.json',tails)
    csvwrite(out/'prediction_metrics.csv',rows); csvwrite(out/'comparisons.csv',[{k:v for k,v in r.items() if k != 'fold_effects'} for r in comparisons])
    for name,data in [('per_clip_metrics.csv.gz',clips),('per_class_metrics.csv',classrows),('per_source_metrics.csv.gz',sourcerows),
        ('signed_prediction_bins.csv',bins),('fold_metrics.csv',foldrows),('leave_one_component_influence.csv.gz',influence)]: csvwrite(out/name,data)
    csvwrite(out/'component_weights.csv',[dict(family=g,weight=v) for g,v in zip(groups,masses)])
    # State-level OOF actions/consequences retain all harms, Dense-wrong and near-zero targets.
    def state_rows():
        for i,m in enumerate(meta):
            for t in range(12):
                row = dict(clip=m['clip'],label=m['label'],family=m['family'],fold=int(folds[i]),t=t+1,weight=float(w[i]/12),D_P=float(y[i,t]),
                    Q_hold=float(q[i,t,0]),Q_refresh=float(q[i,t,1]),periodic_loss=float(baseline[i]))
                for name,pred in arrays.items():
                    if name.startswith(('task_ridge_','task_tree_')):
                        action = int(actions(pred[i,t],t+1)); row[name]=float(pred[i,t]); row[name+'_action']=action
                        row[name+'_value']=float(baseline[i]-q[i,t,action])
                yield row
    csvwrite(out/'oof_state_predictions.csv.gz',list(state_rows()))
    return summary


def resources(out):
    ledger = [json.loads(s) for s in (out/'resource_ledger.jsonl').read_text().splitlines()]
    starts = [e for e in ledger if e['kind']=='work_started']; completed = [e for e in ledger if e['kind']=='decode_complete']
    info = dict(at_utc=now(),expensive_encoder_forwards=0,gpu_model_execution=0,recognition_head_fits=0,controller_fits=0,downloads=0,
        decode_starts=sum(e['kind']=='decode_started' for e in ledger),decode_completions=len(completed),
        decode_failures=sum(e['kind']=='decode_failed' for e in ledger),decoded_frames=sum(e['frames'] for e in completed),
        decode_seconds=sum(e['seconds'] for e in completed),probe_fit_starts=sum(e['kind']=='probe_fit_started' for e in ledger),
        probe_fit_completions=sum(e['kind']=='probe_fit_complete' for e in ledger),
        probe_fit_seconds=sum(e['seconds'] for e in ledger if e['kind']=='probe_fit_complete'),
        elapsed_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(starts[0]['at_utc'])).total_seconds() if starts else 0,
        retained_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        cpu_only_import_check='torch' not in sys.modules,ledger_entries=len(ledger))
    save(out/'resource_accounting.json',info); return info


def findings(out):
    s = read(out/'summary.json'); tm = read(out/'timing_summary.json'); verify = read(out/'verification.json'); r = resources(out)
    primary = next(v for v in s['comparisons'] if v['family']=='ridge' and v['contrast']=='HGH_minus_HG')
    metrics = {v['model']:v for v in s['metrics']}; v = metrics['task_ridge_HGH']['value']; cost = tm['D_minus_A_median_ms']['median']
    decision = ('DEPRIORITIZE_THIS_CONFIGURATION' if v<=0 or primary['value_difference']<0 or cost>10 else
        'PROMISING_FOR_INDEPENDENT_CONFIRMATION' if primary['value_difference']>=.01 and primary['favorable_folds']>=4 else 'SMALL_OR_UNSTABLE_SIGNAL')
    execution = 'COMPLETE' if verify['passed'] else 'VERIFICATION_FAILED'
    save(out/'completion.json',dict(at_utc=now(),execution=execution,decision=decision,primary=primary,value_HGH=v,
        cost_D_minus_A_ms=cost,confirmation='BLOCKED_INPUT_FOR_CONFIRMATION'))
    lines = ['# Experiment26 — DEVELOPMENT / EXPLORATORY','',f'Execution: **{execution}**. Research prioritization: **{decision}**.', '',
        f"Primary RIDGE Delta_HOG=V(HGH)-V(HG): **{primary['value_difference']:.9f} nats** per mean alternative single intervention. "+
        f"Favorable held-out folds: {primary['favorable_folds']}/5; effects {primary['fold_effects']}. Practical reference .01 nats.", '',
        '|View|Value, nats|Regret, nats|Top1 change, percentage points|D_P MAE|Spearman|AUROC|AP|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for view in VIEWS:
        a = metrics['task_ridge_'+view]; lines.append(f"|{view}|{a['value']:.9f}|{a['regret']:.9f}|{100*a['top1_change']:.5f}|{a['MAE']:.6f}|{a['Spearman']:.6f}|{a['AUROC_positive']:.6f}|{a['AP_positive']:.6f}|")
    lines += ['',f"Periodic value=0; regret={s['periodic_regret']:.9f}. Matched-tail label/future-privileged oracle value={s['matched_tail_privileged_oracle_value']:.9f}; regret=0 by construction.",'',
        'Necessary ridge contrasts: '+ '; '.join(f"{c['contrast']}={c['value_difference']:.9f}" for c in s['comparisons'] if c['family']=='ridge')+'.','',
        f"Shuffled held-out HOG value={metrics['task_ridge_HGH_shuffled']['value']:.9f}; intact HGH={v:.9f}. This is an input-reliance check, not an independence test or p-value.",'',
        '|View|Cosine-drift MAE|Cosine-drift Spearman|D_P Spearman|','|---|---:|---:|---:|']
    for view in VIEWS:
        a=metrics['drift_ridge_'+view]; lines.append(f"|{view}|{a['MAE']:.6f}|{a['Spearman']:.6f}|{metrics['task_ridge_'+view]['Spearman']:.6f}|")
    lines += ['', 'Better drift ranking or AUROC alone is not task success. Drift is a separate offline target and never determines actions.', '',
        '|Paired acquisition contrast|Median ms|p95 ms|','|---|---:|---:|']
    for key in ['D_minus_A_median_ms','D_minus_C_median_ms','C_minus_B_median_ms','B_minus_A_median_ms']:
        a=tm[key]; lines.append(f"|{key}|{a['median']:.3f}|{a['p95']:.3f}|")
    lines += ['',f"Costs use {tm['clips']} clips/components, {tm['class_coverage']} classes, {tm['warmup_trials']} warmup and {tm['measured_trials']} measured trials. "+
        'Paired repetitions collapsed within clip before median/p95. The10ms reference applies to median paired D-A. All trials freshly decode authentic bytes; OS file cache may be warm. Encoder absent; cached expensive anchor features/models supplied outside timers. Full decode/grid is common; observations access only endpoints available by each candidate. These are observation-path costs, not end-to-end adaptive runtime; nested components are not double counted.', '',
        f"Resource accounting: {r['decode_starts']} decode starts/{r['decode_completions']} completions, {r['decoded_frames']} frames; "+
        f"{r['probe_fit_completions']} completed CPU probe fits; {r['elapsed_seconds']:.2f}s elapsed; {r['retained_bytes']/2**20:.2f}MiB retained at reporting. Zero new encoder forwards/GPU/head fits/controller fits/downloads.", '',
        f"Verification passed={verify['passed']}; historical files checked={verify['historical_files_unchanged']}. Actual initial HEAD is in checkout_audit.json; requested reference commit was not locally present, so the actual checkout was authoritative. Original roles/seals/19/19.1/23A/23B and untracked presentation work are preserved.", '',
        'All1530 clips/478 global components/51 classes and18360 reset states are retained, including Dense-wrong clips and harms. Original head-training components are disjoint; unknown parent/reupload/pretraining links remain unresolved. Structural fields and much of historical H are deterministic at Periodic prefixes. Only cached probability/history summaries use frozen head arithmetic. Frozen ridge is primary regardless of the fixed tree sensitivity results in summary.json.', '',
        'Finite-cohort retrospective cross-fitting does not restore untouched status. Overlapping training folds, probe/head fitting uncertainty, fixed head/classes/continuation, unknown dependencies and capacity/feature limitations preclude confirmation, equivalence, impossibility, significance or controller-gate claims. No bootstrap population confidence interval is supplied. Adverse tails, all fold/class/source results, signed bins and leave-one-component influence are retained.', '',
        'Fresh confirmation: **BLOCKED_INPUT_FOR_CONFIRMATION**. Metadata-only gate finds74 isolated official-unused clips across24 classes, with unresolved original-parent and exposure certification. See fresh_confirmation_feasibility and prospective_experiment_26B_protocol.md. No fresh outcomes were opened.', '',
        'Next research decision: '+('Deprioritize this fixed HOG/CPU observation configuration; retain the adverse task/cost evidence and require a separately authenticated untouched population before independent confirmation.' if decision=='DEPRIORITIZE_THIS_CONFIGURATION' else
            'Do not deploy/refit a controller. Resolve the untouched source-isolated Experiment26B input gate before any independent confirmation; treat the present signal as exploratory.'),'',
        '![Incremental task value and regret](figure_1_task_value.png)','![Drift versus task prediction](figure_2_drift_task.png)','![Measured acquisition costs](figure_3_acquisition_cost.png)', '',
        'Exact PowerShell commands (repository root):','', '```powershell',
        '$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:CUDA_VISIBLE_DEVICES=""',
        '.\\.venv_temporal\\Scripts\\python.exe -m unittest test_executable_value_observability -v',
        '.\\.venv_temporal\\Scripts\\python.exe run_executable_value_observability.py preflight --output results/executable_value_observability_v1',
        '.\\.venv_temporal\\Scripts\\python.exe run_executable_value_observability.py run --output results/executable_value_observability_v1',
        '.\\.venv_temporal\\Scripts\\python.exe run_executable_value_observability.py resume --output results/executable_value_observability_v1',
        '.\\.venv_temporal\\Scripts\\python.exe run_executable_value_observability.py verify --output results/executable_value_observability_v1', '```','']
    (out/'findings.md').write_text('\n'.join(lines),encoding='utf-8')


def run(out):
    preflight(out); check_seal(out/'design_seal.json')
    if (out/'completion.json').exists() and read(out/'completion.json')['execution']=='COMPLETE':
        print('Completed experiment preserved; use verify.',flush=True); return
    event(out,'work_started',pid=os.getpid()); pop,meta,screens = load_metadata()
    x,drift = extracted(out,meta,screens); y,q,base,pred,bpred = targets(meta,out)
    folds = np.array([read(out/'fold_manifest.json')['global_component_to_fold'][m['family']] for m in meta])
    arrays = probe(out,x,y,drift,meta,folds); summarize(out,meta,arrays,q,base,pred,bpred); timing(out,meta,screens,folds)
    resources(out)
    from verify_executable_value_observability import verify
    verify(out)
    from render_executable_value_observability import render
    render(out); findings(out); resources(out)
    print(json.dumps(read(out/'completion.json'),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['preflight','run','resume','verify']); parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); out=args.output.resolve()
    if out != DEFAULT.resolve():
        raise RuntimeError('This sealed experiment uses only results/executable_value_observability_v1; choose a new study for another path')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    with threadpool_limits(limits=1):
        try:
            if args.mode=='preflight': preflight(out)
            elif args.mode in ['run','resume']: run(out)
            else:
                from verify_executable_value_observability import verify
                verify(out)
        except Exception as e:
            if out.exists():
                event(out,'failure',mode=args.mode,error=repr(e),traceback=traceback.format_exc()); resources(out)
                save(out/'execution_failure.json',dict(at_utc=now(),error=repr(e),mode=args.mode,status='BLOCKED_INPUT' if isinstance(e,FileNotFoundError) else 'PARTIAL'))
            raise


if __name__=='__main__':
    main()
