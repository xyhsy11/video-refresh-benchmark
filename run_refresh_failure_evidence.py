"""Experiment 23A: authenticated historical reconciliation and input-only gate."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import refresh_failure_evidence as m

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/refresh_failure_evidence_v1'
BASE = '4092d2a0fd8388347dbc69c486c73f35910933db'
HISTORICAL_BASE = 'c3ec3e67081f4dba89356efac9549194cceebb13'
HISTORICAL_WORKTREE = ROOT.parent/'video-refresh-evidence-23a-historical'
E21 = 'results/ucf101_causal_development_v1/'
E22 = 'results/task_utility_observability_v1/'
HISTORIES = ('causal_observability_v1', 'causal_observability_population_v2',
    'ucf101_causal_development_v1', 'task_utility_observability_v1',
    'task_signal_diagnostic_v1', 'task_signal_ablation_v2', 'refresh_target_decomposition_v1',
    'cached_history_signal_diagnostic_v1_rerun1', 'refresh_generalization_audit_v1_rerun1')
CODE = ('refresh_failure_evidence.py', 'run_refresh_failure_evidence.py', 'test_refresh_failure_evidence.py')
INPUT_HASHES = {}
LOCAL_ALLOWED = set()
READ_ONLY = False


def git(*args, **kwargs):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, **kwargs).strip()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*2**20), b''):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def check_path(path, writing=False):
    if isinstance(path, int):
        return
    p = Path(os.fsdecode(path)).resolve()
    if writing:
        m.require(not READ_ONLY and p.is_relative_to(OUT), '23A permits writes only to its new result directory')
    elif p.is_relative_to(ROOT/'results'):
        allowed = [OUT, *(ROOT/'results'/h for h in HISTORIES)]
        m.require(any(p.is_relative_to(a) for a in allowed), 'Forbidden historical/official-test outcome path: '+str(p))
    elif p.is_relative_to(ROOT/'data'):
        m.require(p in LOCAL_ALLOWED, 'Unapproved input or official-test source path: '+str(p))


def hook(event, args):
    if event == 'open':
        mode, flags = args[1] or '', args[2] or 0
        check_path(args[0], any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
    elif event == 'import':
        m.require(args[0].split('.')[0] not in {'torch', 'torchvision', 'cupy', 'av', 'requests', 'urllib'},
                  '23A forbids GPU, video-decoder and network runtime imports')
    elif event.startswith('socket.'):
        raise RuntimeError('23A has no network access')
    elif event in ('os.remove', 'os.rmdir', 'os.mkdir'):
        check_path(args[0], True)
    elif event in ('os.rename', 'os.replace'):
        check_path(args[0], True); check_path(args[1], True)


def read(path):
    p = ROOT/path
    INPUT_HASHES[rel(p)] = sha(p)
    if p.suffix == '.gz':
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(p.read_text(encoding='utf-8'))


def csv_read(path):
    p = ROOT/path; INPUT_HASHES[rel(p)] = sha(p)
    with p.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def arrays(path):
    p = ROOT/path; INPUT_HASHES[rel(p)] = sha(p)
    with np.load(p, allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_save(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def baseline_entries():
    return {line.split('\t', 1)[1]:line.split()[2]
        for line in git('ls-tree', '-r', BASE, 'results').splitlines()}


def historical_integrity():
    # Opaque identity comparison, never deserialize old official-test outcomes.
    entries = baseline_entries()
    p = subprocess.run(['git','hash-object','--no-filters','--stdin-paths'], cwd=ROOT,
        input='\n'.join(entries)+'\n', capture_output=True, text=True, check=True)
    actual = p.stdout.splitlines()
    m.require(len(actual) == len(entries), 'Historical inventory incomplete')
    for (name, expected), got in zip(entries.items(), actual):
        m.require(got == expected, 'Historical scientific result changed: '+name)
    return dict(passed=True, base_commit=BASE, preserved_result_blobs=len(entries),
        official_test_outcomes_deserialized=False, method='Opaque Git blob identity only')


def verify_public_inputs():
    entries = baseline_entries()
    names = [p for p in INPUT_HASHES if p in entries]
    got = subprocess.check_output(['git','hash-object','--no-filters','--stdin-paths'], cwd=ROOT,
        input='\n'.join(names)+'\n', text=True).splitlines()
    m.require(len(got) == len(names), 'Public authentication incomplete')
    for p, h in zip(names, got):
        m.require(h == entries[p], 'Input differs from reviewed Git commit: '+p)
    return len(names)


def observations():
    manifest = read(E22+'population_manifest.json')
    original_population = read(E21+'population_manifest.json')
    grids = {v['clip']:v['snippet_grid'] for v in original_population['videos']}
    ledger = read(E22+'observation_provenance.json')
    videos = {v['clip']:v for v in manifest['videos']}
    m.require(len(videos) == 1006 and set(videos) == set(ledger), 'Population identity mismatch')
    all_states, signals = {}, {}
    for i, (clip, v) in enumerate(videos.items()):
        m.require(v['official_role'] == 'official_train', 'Official-test membership forbidden')
        paths = ledger[clip]['outputs']
        for name, expected in paths.items():
            p = (ROOT/name).resolve()
            m.require(p.is_relative_to(ROOT/'data/source/task_utility_observability_v1/clips'), 'Bad observation provenance path')
            LOCAL_ALLOWED.add(p)
            m.require(sha(p) == expected, 'Missing/changed genuine observation: '+name)
            INPUT_HASHES[name] = expected
        states_path = next(p for p in paths if p.endswith('.json'))
        array_path = next(p for p in paths if p.endswith('.npz'))
        ss = read(states_path)['states']; aa = arrays(array_path)['observations']
        m.require(aa.shape == (len(ss), 146) and np.isfinite(aa).all(), 'Missing genuine observation vectors')
        for r, a in zip(ss, aa):
            m.require(r['clip'] == clip and r['label'] == v['label'] and r['group'] == v['group'] and
                r['role'] == v['study_role'], 'Source-aligned state identity failed')
            m.require(np.isfinite(r['candidate_seconds']) and r['endpoint_frame_index'] >= 0 and
                r['history'] == sorted(set(r['history'])) and all(t < r['t'] for t in r['history']),
                'Genuine timestamp or chronology missing')
            grid = grids[clip]
            m.require(r['candidate_seconds'] == grid['endpoint_timestamps'][r['t']] and
                r['endpoint_frame_index'] == grid['endpoint_indices'][r['t']], 'Source PTS/grid join failed')
            m.require(np.all(np.diff(grid['endpoint_timestamps']) > 0) and
                np.all(np.diff(grid['endpoint_indices']) > 0), 'Nonincreasing genuine candidate grid')
            m.require(abs(r['MAD']-a[114+26]) < 1e-14, 'Gray MAD column alignment failed')
            key = (clip, r['population'], r['t'])
            m.require(key not in all_states, 'Duplicate state identity')
            all_states[key], signals[key] = r, a[114:]
        if (i+1) % 250 == 0:
            print('Authenticated genuine clip observations', i+1, '/', len(videos), flush=True)
    associations = []
    for item in read(E22+'prediction_inventory.json'):
        rows = read(item['states_path']); a = arrays(item['path'])
        m.require(INPUT_HASHES[item['states_path']] == item['states_sha256'] and
                  INPUT_HASHES[item['path']] == item['sha256'], 'Prediction inventory identity failed')
        aligned = []
        for i, r in enumerate(rows):
            key = (r['clip'], r['population'], r['t'])
            m.require(r == all_states[key] and r['discretionary'], 'State/observation join mismatch')
            np.testing.assert_allclose(a['gains'][:, i], [r['gain_local'], r['gain_budget']], atol=0, rtol=0)
            aligned.append(signals[key])
        for r in m.proxy_associations(rows, np.array(aligned)):
            associations.append(dict(role=item['role'], population=item['population'], **r,
                source=item['states_path'], scope='EXPLORATORY / DEVELOPMENT; genuine aligned grayscale proxies'))
    csv_save('grayscale_signed_gain_associations.csv', associations)
    return manifest, dict(clips=len(videos), all_states=len(all_states), association_rows=len(associations),
        empirical_fixtures_used=False, semantic_event_annotations_invented=0,
        timestamp_scope='Previously recorded genuine PTS-aligned endpoint seconds and frame identities; no fresh decoding')


def quality():
    manifest = read(E22+'population_manifest.json')
    rows = csv_read(E21+'development_per_clip_metrics.csv')
    pp = read(E21+'development_predictions.json.gz')
    plan = read(E21+'development_bootstrap_groups.json')
    p = E21+'development_bootstrap_group_indices.npy'; INPUT_HASHES[p] = sha(ROOT/p)
    draws = np.load(ROOT/p, allow_pickle=False)
    result = m.reproduce_quality(rows, pp, manifest['classes'],
        [(g['label'], g['group']) for g in plan['groups']], draws)
    prior = csv_read(E21+'development_CE_contrasts.csv')
    for r in result['contrasts']:
        old = next(v for v in prior if v['contrast'] == r['contrast'])
        np.testing.assert_allclose([r['mean'], *r['CI95']],
            [float(old[k]) for k in ('mean', 'CI95_lower', 'CI95_upper')], atol=1e-12, rtol=0)
    save('b4_reproduction.json', result)
    return result


def costs():
    result = m.reconcile_cost(csv_read(E21+'timing_trials.csv'))
    previous = read(E22+'cost_feasibility.json')
    for r in result['methods']:
        old = next(v for v in previous['methods'] if v['method'] == r['method'])
        for key in ('seconds', 'cheap_thumbnail_seconds', 'cheap_statistics_seconds',
                    'controller_decide_seconds', 'observation_preparation_seconds'):
            m.require(abs(r[key]-old[key]) < 1e-12, 'Recorded timing reconciliation failed')
    for r in result['incremental']:
        old = next(v for v in previous['incremental'] if (v['method'],v['reference']) == (r['method'],r['reference']))
        m.require(abs(r['paired_median_extra_seconds']-old['paired_median_extra_seconds']) < 1e-12,
                  'Paired timing reconciliation failed')
    result['older_cost_evidence'] = [
        dict(study='cached_history_signal_diagnostic_v1_rerun1', type='CPU cached-component timing',
             history_us=33.116, baseline_RF_ms=83.250, augmented_RF_ms=88.042,
             boundary='Preloaded observations only; excludes acquisition/decode/encoder; not additive to E21'),
        dict(study='refresh_generalization_audit_v1_rerun1', type='CPU preloaded prediction timing',
             direct_RF_ms=41.348, two_target_ms=40.827, history_RF_ms=40.981, threshold_us=.327,
             boundary='Excludes history/observation construction; different run/models/boundary'),
        dict(study='causal_observability_v1', type='Native400 mechanical GPU smoke, historical only',
             note='16 completed successful smoke forwards plus 4 preserved failed-verification-attempt events; no final HMDB51 cost comparison'),
        dict(study='causal_observability_population_v2', type='Input screening CPU work',
             note='No policy latency measurement or model inference')]
    save('cost_reconciliation.json', result)


def population_gate():
    temporal = read('results/causal_observability_population_v2/temporal_sampling_audit.json')
    quota = read('results/causal_observability_population_v2/quota_feasibility.json')
    source = read('results/causal_observability_v1/source_metadata_identities.json')
    family = read('results/causal_observability_population_v2/source_family_audit.json')
    folders = sorted(p.name for p in (ROOT/'data/source').iterdir() if p.is_dir())
    # No recursive scan of UCF official-test videos/outcomes. Folder inventory is input existence only.
    result = dict(status='BLOCKED_INPUT', final_decision='NO_GO',
        scope='Existing authorized local inputs and retained authenticated screening only; no dataset acquisition',
        roles=list(m.ROLES), prospective_role_assignments=[], eligible_new_populations=0,
        local_source_folders=folders, available_disk_bytes=shutil.disk_usage(ROOT).free,
        candidates=[
            dict(population='HMDB51 existing authorized original inputs',
                labels='Official 51-class split labels authenticated by historical read-only audit',
                timestamps=dict(original=temporal['original_annotation_clips'], verified_decoded=temporal['actual_decoded_clips'],
                                eligible=temporal['new_eligible'], rule=temporal['rule']),
                family_components=len(family['components']),
                roles='A/B/C allocation infeasible; no four-role assignment',
                quota_status={k:v['status'] for k,v in quota.items()},
                duplicates='Conservative exact/pHash/title connected families; unresolved parent IDs and reuploads',
                licensing=source['license'],
                storage='Existing bytes retained; archive/video identities checked by historical audit',
                compute='No trained full-class pipeline cost certificate; no new smoke authorized or run',
                status='INELIGIBLE_POPULATION_DESIGN'),
            dict(population='UCF101 archived train/development', labels='101 classes; official-training membership only',
                timestamps='Genuine retained candidate PTS; hash-verified in prior audits',
                groups='Known class+gXX groups and duplicate isolation; unknown parent/pretraining overlap',
                duplicates='Known byte/components isolated; not proof of universally independent sources',
                licensing='Existing authorized local artifacts only; new-population rights not established by their presence',
                storage='1006 retained genuine development clips; no new storage requirement measured',
                compute='Historical actual E21 timing available; prospective full-role compute unknown',
                status='INELIGIBLE_EXPOSED_AND_HEAD_OVERLAP'),
            dict(population='Other cached Kinetics/cohort artifacts',
                status='NOT_CERTIFIED_AS_FRESH_FOUR_ROLE_POPULATION',
                reason='Previously used/reserved cohorts and incomplete parent identities; no authenticated prospective role certificate; protected outcomes not opened')],
        missing=['Authorized previously unexposed all-class RGB/labels with authenticated bytes and real PTS',
            'Verified original source/subject/take/camera families and duplicate components isolated globally across four roles',
            'Prospective full-class family counts, deterministic allocation certificate and group-based precision plan',
            'Verified licensing/terms and projected full-role storage/compute on intended execution hardware',
            'Task-relevant pre-decision observable with demonstrated incremental budget utility beyond cached history at acceptable acquisition cost'],
        future_minimum_independent_families_per_class=dict(head_train=10, controller_train=10, selection=5, evaluation=20),
        minima_note='Prospective engineering floors only, not power guarantees; no local candidate is declared feasible against these floors',
        actions=dict(downloaded_bytes=0, agreements_accepted=0, new_dataset_decodes=0,
                     new_model_calls=0, future_training_or_policy_execution=False))
    save('independent_population_gate.json', result)


def history_checks():
    modules = ['test_refresh_failure_evidence','test_task_utility_observability',
        'test_causal_observability','test_causal_observability_population_v2',
        'test_ucf101_causal_development','test_task_signals','test_task_signal_ablation',
        'test_refresh_target_decomposition','test_cached_history_signals','test_refresh_generalization_audit']
    commands = [*(['-m','unittest',module,'-v'] for module in modules),
        ['run_causal_observability.py','--audit-only'],
        ['run_causal_observability_population_v2.py','--audit-only'],
        ['run_ucf101_causal_development.py','--audit-only'],
        ['run_task_utility_observability.py','--audit-only'],
        ['results/task_utility_observability_v1/independent_replay_review.py'],
        ['results/task_utility_observability_v1/supplemental_review.py'],
        ['analyze_task_signals.py','--output','results/task_signal_diagnostic_v1','--audit-only'],
        ['analyze_task_signal_ablation.py','--output','results/task_signal_ablation_v2','--audit-only'],
        ['analyze_refresh_target_decomposition.py','--output','results/refresh_target_decomposition_v1','--audit-only'],
        ['analyze_cached_history_signals.py','--predictability-input','results/two_target_refresh_predictability_v1',
         '--output','results/cached_history_signal_diagnostic_v1_rerun1','--seed','20261021','--bootstrap','10000','--audit-only'],
        ['audit_refresh_generalization.py','--output','results/refresh_generalization_audit_v1_rerun1',
         '--seed','20261022','--bootstrap','10000','--audit-only']]
    # Original guards deliberately compare the entire historical index. A later
    # attribution-notice addition makes them reject today's tree, before replay.
    # Use an intact historical checkout; never monkeypatch or relax their guard.
    old_studies = ['task_signal_diagnostic_v1','task_signal_ablation_v2',
        'refresh_target_decomposition_v1','cached_history_signal_diagnostic_v1_rerun1',
        'refresh_generalization_audit_v1_rerun1']
    notice_checks = {}
    for study in old_studies:
        p = read('results/'+study+'/protocol.json')
        differences = set(git('diff','--name-only',p['starting_commit'],BASE).splitlines())
        changed = sorted(differences.intersection(p['previous_tracked_blob_identities']))
        m.require(changed == ['THIRD_PARTY_NOTICES.md'], 'Unexplained historical guard incompatibility')
        notice_checks[study] = changed
    if not HISTORICAL_WORKTREE.exists():
        subprocess.run(['git','worktree','add','--detach',str(HISTORICAL_WORKTREE),HISTORICAL_BASE],
                       cwd=ROOT, check=True, capture_output=True)
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=HISTORICAL_WORKTREE,text=True).strip()
    status = subprocess.check_output(['git','status','--porcelain'],cwd=HISTORICAL_WORKTREE,text=True).strip()
    m.require(revision == HISTORICAL_BASE and not status, 'Historical audit checkout differs; do not reset user work')
    scientific_differences = git('diff','--name-only',HISTORICAL_BASE,BASE,'--',
                                 *('results/'+s for s in old_studies))
    m.require(not scientific_differences, 'Historical scientific outputs differ from current evidence')
    save('historical_checkout_review.json', dict(commit=revision, path=str(HISTORICAL_WORKTREE), clean=True,
        current_tree_guard_blockers=notice_checks, scientific_result_differences=[],
        remedy='Unmodified historical checkout, original guards and real arrays; no replacement of current files'))
    prior = []
    if (OUT/'verification_commands.json').exists():
        prior = json.loads((OUT/'verification_commands.json').read_text(encoding='utf-8'))
    def run(item):
        i, args = item
        cwd = HISTORICAL_WORKTREE if i >= len(commands)-5 else ROOT
        name = f'check_{i:02d}.txt'
        # Resume only already successful current-tree real audits, whose public
        # evidence is checked unchanged at start/end. Unit modules always rerun.
        previous = next((r for r in prior if r['command'][2:] == args and r['returncode'] == 0), None)
        if previous and cwd == ROOT and args[0] != '-m':
            old_log = OUT/previous['log']
            text = old_log.read_text(encoding='utf-8')
            (OUT/name).write_text(text,encoding='utf-8')
            print('Retained successful CPU audit:', ' '.join(args),flush=True)
            return dict(command=[sys.executable,'-B',*args],returncode=0,log=name,cwd=str(cwd),
                reused_successful_receipt=previous['log'],log_sha256=sha(OUT/name),
                fixtures='Retained real-evidence audit; disclosed arithmetic fixture checks are not empirical results')
        print('CPU verification started:', ' '.join(args), flush=True)
        r = subprocess.run([sys.executable,'-B',*args], cwd=cwd, capture_output=True, text=True, encoding='utf-8',
                           env={**os.environ, 'PYTHONIOENCODING':'utf-8'})
        (OUT/name).write_text(r.stdout+r.stderr, encoding='utf-8')
        print('CPU verification finished:', i, 'return code', r.returncode, flush=True)
        return dict(command=[sys.executable,'-B',*args], returncode=r.returncode, log=name,cwd=str(cwd),
                    log_sha256=sha(OUT/name),
                    fixtures='Unit-test synthetic fixtures only' if args[0] == '-m' else
                        'Real retained evidence audit; disclosed arithmetic fixture checks are not empirical results')
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, enumerate(commands)))
    save('verification_commands.json', results)
    return results


def audit():
    global READ_ONLY
    READ_ONLY = True
    inventory = read('results/refresh_failure_evidence_v1/SHA256SUMS.json')
    actual_files = {rel(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.json'}
    expected_files = {p for p in inventory if p.startswith(rel(OUT)+'/')}
    m.require(actual_files == expected_files, 'Output inventory missing/unexpected artifacts')
    for p, h in inventory.items():
        m.require(sha(ROOT/p) == h, 'Changed output/code: '+p)
    run = read('results/refresh_failure_evidence_v1/run_manifest.json')
    for p, h in run['input_sha256'].items():
        if p.startswith('data/'):
            LOCAL_ALLOWED.add((ROOT/p).resolve())
        m.require(sha(ROOT/p) == h, 'Changed/missing authenticated input: '+p)
    history = historical_integrity()
    q = quality_read_only()
    m.require(read('results/refresh_failure_evidence_v1/independent_population_gate.json')['status'] == 'BLOCKED_INPUT',
              'Unreviewed population gate change')
    m.require(all(r['returncode'] == 0 for r in read('results/refresh_failure_evidence_v1/verification_commands.json')),
              'Recorded verification failure')
    m.require(not {'torch','cupy','av'} & set(sys.modules), 'Forbidden runtime loaded')
    return dict(passed=True, read_only=True, history=history, b4_intervals_reproduced=q,
                input_hashes_verified=len(run['input_sha256']), official_test_outcomes_opened=False,
                new_GPU_calls=0, downloads=0, decodes=0, model_fits=0, policy_execution=0,
                final_decision='NO_GO', population_status='BLOCKED_INPUT')


def quality_read_only():
    # Read-only arithmetic checks use independent saved logits, not fitting/prediction.
    manifest = read(E22+'population_manifest.json')
    plan = read(E21+'development_bootstrap_groups.json')
    p = E21+'development_bootstrap_group_indices.npy'
    result = m.reproduce_quality(csv_read(E21+'development_per_clip_metrics.csv'),
        read(E21+'development_predictions.json.gz'), manifest['classes'],
        [(g['label'],g['group']) for g in plan['groups']], np.load(ROOT/p,allow_pickle=False))
    m.require(result == read('results/refresh_failure_evidence_v1/b4_reproduction.json'), 'B4 reproduction changed')
    result_cost = m.reconcile_cost(csv_read(E21+'timing_trials.csv'))
    saved = read('results/refresh_failure_evidence_v1/cost_reconciliation.json')
    m.require(all(saved[k] == v for k,v in result_cost.items()), 'Timing reconciliation changed')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    sys.addaudithook(hook)
    if args.audit_only or (OUT/'SHA256SUMS.json').exists():
        print(json.dumps(audit(), indent=2)); return 0
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    history = historical_integrity()
    # Read and authenticate every narrative source used in the curated matrix.
    for h in HISTORIES:
        p = ROOT/'results'/h/'findings.md'
        INPUT_HASHES[rel(p)] = sha(p)
    for name in ('feature_information_audit.md',):
        p = ROOT/'results/cached_history_signal_diagnostic_v1_rerun1'/name
        INPUT_HASHES[rel(p)] = sha(p)
    manifest, obs = observations()
    quality(); costs(); population_gate()
    for p in ('paired_group_intervals.json', 'teacher_task_alignment.json', 'offline_headroom.json',
              'target_replay_validation.json', 'exposure_status.json', 'source_model_identity.json'):
        read(E22+p)
    checks = history_checks()
    m.require(all(r['returncode'] == 0 for r in checks), 'Verification incomplete: inspect verification_commands.json; no success seal written')
    public_count = verify_public_inputs()
    m.require(historical_integrity() == history, 'History changed during 23A')
    save('audit.json', dict(passed=True, historical_integrity=history, genuine_observations=obs,
        baseline_authenticated_public_inputs=public_count, verification_commands_passed=len(checks),
        no_semantic_events_invented=True, new_GPU_calls=0, new_policy_or_probe_fits=0,
        new_network_bytes=0, new_video_decodes=0, official_test_outcomes_opened=False,
        historical_test_outcome_blobs='Opaque identity only; never parsed',
        scientific_decision='NO_GO', independent_population='BLOCKED_INPUT'))
    save('run_manifest.json', dict(experiment='23A', started_utc=started,
        completed_utc=datetime.now(timezone.utc).isoformat(), reviewed_commit=BASE,
        starting_commit=git('rev-parse','HEAD'), branch=git('branch','--show-current'),
        python=sys.version, numpy=np.__version__, input_sha256=INPUT_HASHES,
        code_sha256={p:sha(ROOT/p) for p in CODE}, observations=obs,
        genuine_experiments='Archived real evidence only; no new empirical inference, cost timing or fitting',
        test_fixtures='Synthetic unit fixtures are excluded from every scientific result',
        constraints=dict(GPU_calls=0, decoder_calls=0, network_bytes=0, agreement_acceptances=0,
                         controllers_created=0, fits=0, official_test_outcomes_opened=False)))
    inventory = {rel(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.json'}
    inventory.update({p:sha(ROOT/p) for p in CODE})
    save('SHA256SUMS.json', inventory)
    print(json.dumps(audit(), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
