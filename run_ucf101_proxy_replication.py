"""Run/resume ucf101_proxy_replication_v1, or perform a genuinely read-only audit."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
for thread_variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[thread_variable] = '1'
from ucf101_proxy_common import *

COMMAND = r'& .\.venv_temporal\Scripts\python.exe .\run_ucf101_proxy_replication.py --resume'
AUDIT = r'& .\.venv_temporal\Scripts\python.exe -B .\run_ucf101_proxy_replication.py --audit-only'


def state(stage, status, **details):
    p = OUT/'status.json'
    previous = read(p) if p.exists() else dict(stages={})
    for name in ['baseline_verification', 'metadata', 'acquisition', 'population', 'training_smoke',
                 'train_features', 'development_features', 'head', 'development_analysis', 'precision',
                 'evaluation', 'test_features', 'quality', 'timing', 'figures', 'final']:
        previous['stages'].setdefault(name, dict(status='pending'))
    previous['stages'][stage] = dict(status=status, at_utc=now(), **details)
    previous.update(status=status, current_stage=stage, updated_at_utc=now(), resume_command=COMMAND, audit_command=AUDIT)
    save(p, previous)


def preflight():
    from ucf101_proxy_verify import baseline_verification
    OUT.mkdir(parents=True, exist_ok=True)
    config = OUT/'run_configuration.json'
    if not config.exists():
        before = git('status', '--porcelain=v1')
        if git('diff', '--name-only', '--', 'results').strip():
            raise Blocker('Historical tracked result modifications need preservation review')
        ancestors = []
        for p in [ROOT.parent/'AGENTS.md', ROOT/'AGENTS.md']:
            if p.exists():
                ancestors.append(dict(path=str(p), sha256=sha(p)))
        save(config, dict(actual_starting_HEAD=git('rev-parse', 'HEAD'), reviewed_HEAD=REVIEWED,
            reviewed_tree=git('rev-parse', REVIEWED+'^{tree}'), actual_tree=git('rev-parse', 'HEAD^{tree}'),
            relevant_changes=git('log', '--oneline', '-3'), first_inspection_user_status='clean', implementation_preflight_status=before, AGENTS=ancestors,
            reviewed_tree_equal=git('rev-parse', REVIEWED+'^{tree}') == git('rev-parse', 'HEAD^{tree}'),
            free_disk_bytes=shutil.disk_usage(ROOT).free, client_date='2026-10-17', client_timezone='Asia/Tokyo',
            protocol_seed=SEED, execution_started_at_utc=now(), environment='existing .venv_temporal, no in-place upgrades',
            historical_tracked_result_blobs=git('ls-files', '-s', 'results').splitlines()))
    if not (OUT/'baseline_verification.json').exists():
        result = baseline_verification(); save(OUT/'baseline_verification.json', result)
    if read(OUT/'baseline_verification.json')['status'] != 'passed':
        raise Blocker('Targeted baseline real-array verification incomplete; see baseline_verification.json')
    state('baseline_verification', 'verified')
    if (OUT/'evaluation_seal.json').exists():
        # Never rewrite a sealed unit-test record merely to resume an exposed run.
        seal_check((OUT/'evaluation_exposure.json').exists())
        return
    if shutil.disk_usage(ROOT).free < 20*1024**3:
        raise Blocker('Less than 20 GiB free before dataset acquisition')
    test = subprocess.run([sys.executable, '-m', 'unittest', 'test_ucf101_proxy_replication', '-v'],
                          cwd=ROOT, capture_output=True, text=True)
    save(OUT/'unit_tests.json', dict(returncode=test.returncode, output=test.stdout+test.stderr,
        code_sha256={rel(p): sha(p) for p in ROOT.glob('*ucf101*.py')}, at_utc=now(), fixtures='explicitly synthetic unit tests only'))
    if test.returncode:
        raise Blocker('Focused unit tests failed; see unit_tests.json')


def freeze(manifest):
    if (OUT/'evaluation_seal.json').exists():
        seal_check()
        return
    if (OUT/'evaluation_exposure.json').exists():
        raise Blocker('Exposure exists without seal; preserve evidence, no replacement seal')
    files = list(ROOT.glob('*ucf101*.py'))
    files += [ROOT/n for n in ['refresh_pooling_utils.py', 'temporal_refresh_common.py', 'research_utils.py',
                              'benchmark.py', 'temporal_refresh_timing.py']]
    files += [OUT/n for n in ['protocol.md', 'proposal_amendments.md', 'population_plan.json', 'eligible_manifest.json',
        'official_metadata_identities.json', 'duplicate_threshold.json', 'duplicate_and_pretraining_audit.json',
        'encoder_identity.json', 'heads/parameters.json', 'head_selection.json', 'training_grid.json',
        'precision_estimates.json', 'stratum_thresholds.json', 'unit_tests.json']]
    identity = read(OUT/'encoder_identity.json')
    files += [ROOT/identity['checkpoint_path'], *[ROOT/n for n in identity['upstream_code_hashes']]]
    files += [DATA/'official_splits.zip', DATA/'official_page.html']
    for budget in [2, 4, 8]:
        import refresh_pooling_utils as pool
        import numpy as np
        p = OUT/f'schedules_B{budget}.npz'; np.savez_compressed(p, **pool.schedules(16, budget)); files.append(p)
    counts = {r: sum(manifest['counts'][r].values()) for r in ['train', 'development', 'test']}
    seal = dict(sealed_at_utc=now(), code_HEAD=git('rev-parse', 'HEAD'), sha256={rel(p): sha(p) for p in files},
        group_counts=counts, clip_counts={r: n*2 for r, n in counts.items()},
        primary='test X3D-XS T16 B4 C g=CE(Z_pool)-CE(LegacyCosine)', secondary='p=Z_pool-periodic; d=LegacyCosine-periodic',
        test_size='all eligible official split-1 test groups fixed now, two eligible clips per group, no sign-based extension',
        bootstrap=dict(repeats=10000, rng='PCG64', seed=SEED, unit='class-stratified paired group', percentiles=[2.5, 97.5]),
        no_test_features_before_seal=not (OUT/'test_collection.json').exists())
    if not seal['no_test_features_before_seal']:
        raise Blocker('Test features already exist before seal')
    save(OUT/'evaluation_seal.json', seal)


def findings(blocker=None):
    config = read(OUT/'run_configuration.json') if (OUT/'run_configuration.json').exists() else {}
    status = read(OUT/'status.json') if (OUT/'status.json').exists() else {}
    baseline = read(OUT/'baseline_verification.json') if (OUT/'baseline_verification.json').exists() else {}
    primary = read(OUT/'primary_intervals.json') if (OUT/'primary_intervals.json').exists() else []
    ucf_g = ucf_p = 'Pending; no UCF test result'
    lines = ['# ucf101_proxy_replication_v1', '',
        f'Actual starting HEAD: `{config.get("actual_starting_HEAD")}`. Reviewed tree equal: {config.get("reviewed_tree_equal")}.',
        f'Execution state: **{status.get("status", "prepared")}**, stage `{status.get("current_stage")}`.', '']
    if blocker:
        lines += [f'**Blocker:** {blocker}', '', 'Completed work is retained. No missing measurement is filled with synthetic features or reconstructed individual observations.', '']
    if primary:
        lines += ['|Contrast|Mean CE difference, nats|95% paired GROUP CI|Groups / clips|', '|---|---:|---|---|']
        for r in primary:
            lines.append(f'|{r["contrast"]}|{r["mean"]:+.6f}|[{r["CI95_lower"]:+.6f}, {r["CI95_upper"]:+.6f}]|{r["groups"]} / {r["clips"]}|')
        g = next(r for r in primary if r['contrast'] == 'g'); p = next(r for r in primary if r['contrast'] == 'p')
        fmt = lambda r: f'{r["mean"]:+.6f} [{r["CI95_lower"]:+.6f},{r["CI95_upper"]:+.6f}]'
        ucf_g, ucf_p = fmt(g), fmt(p)
        lines += ['', f'Primary achieved interval width: {g["interval_width"]:.6f} nats.',
                  'Improvement over LegacyCosine established by this interval.' if g['CI95_upper'] < 0 else 'Improvement over LegacyCosine is not established; a CI crossing zero is inconclusive, not equivalence.',
                  'Periodic beaten in secondary CE interval.' if p['CI95_upper'] < 0 else 'Periodic was not established to be beaten in CE. Improvement in g alone is insufficient.', '']
        sec = __import__('pandas').read_csv(OUT/'secondary_intervals.csv')
        parts = sec[(sec.budget == 4) & (sec.pool == 'C')].set_index('contrast')
        lines += [f'Exact decomposition of g: directional difference {parts.loc["directional_g", "mean"]:+.6f} + KL difference {parts.loc["KL_g", "mean"]:+.6f}.',
            'This describes signed movement relative to the true label and dense prediction discrepancy. It is standard algebra, not a new theorem or causal mechanism. Labels enter only this retrospective explanation and the explicit task oracles.', '']
        lines += ['|Decomposition contribution|Mean|Secondary marginal 95% group CI|', '|---|---:|---|']
        for q in ['directional_g', 'KL_g']:
            r = parts.loc[q]
            lines.append(f'|{q}|{r["mean"]:+.6f}|[{r.CI95_lower:+.6f},{r.CI95_upper:+.6f}]|')
        lines += ['', '|Selector, B4 C (dense16 separate)|CE|Balanced top1|Ordinary clip accuracy|Dense agreement|KL|', '|---|---:|---:|---:|---:|---:|']
        for r in read(OUT/'aggregate_metrics.json'):
            if (r['budget'] == 4 and r['pool'] == 'C' and r['selector'] in ['periodic', 'LegacyCosine', 'Z_pool']) or r['selector'] == 'dense16':
                lines.append(f'|{r["selector"]}|{r["CE"]["mean"]:.6f}|{100*r["correct"]["mean"]:.2f}%|{100*r["ordinary_clip_accuracy"]:.2f}%|{100*r["dense_agreement"]["mean"]:.2f}%|{r["KL"]["mean"]:.6f}|')
        lines.append('')
    else:
        lines += ['UCF primary estimate, confidence interval, achieved precision, periodic comparison, decomposition contributions and measured dense/periodic costs are **pending**. No UCF recognition result is claimed.', '']
    if (OUT/'eligible_manifest.json').exists():
        m = read(OUT/'eligible_manifest.json')
        lines += ['Achieved eligible counts (all 101 official classes):']
        for role, counts in m['counts'].items():
            lines.append(f'- {role}: {sum(counts.values())} groups, {2*sum(counts.values())} clips; per-class groups {min(counts.values())} to {max(counts.values())}.')
        lines += ['', 'Conclusions apply to the technically eligible split-1 population under the frozen two-clips-per-group rule and duplicate quarantine.', '']
    elif (OUT/'population_feasibility.json').exists():
        feasibility = read(OUT/'population_feasibility.json')
        lines += ['Input-screened population failed or is awaiting the fixed all-class population gate:', '']
        for role, counts in feasibility['counts'].items():
            minimum = feasibility['thresholds'][role]
            failures = ', '.join(f'{c}={n}' for c, n in counts.items() if n < minimum)
            lines.append(f'- {role}: {sum(counts.values())} groups / {2*sum(counts.values())} clips; minimum {minimum}/class; below threshold: {failures or "none"}.')
        lines.append('')
    if (OUT/'precision_estimates.json').exists():
        precision = read(OUT/'precision_estimates.json')
        for q, r in precision['contrasts'].items():
            lines.append(f'Development-based predicted {q} halfwidth: {r["predicted_CI95_halfwidth"]:.6f} nats (planning target .015; noisy variance transfer, no power guarantee).')
        lines.append('')
    if (OUT/'timing_summary.json').exists():
        t = read(OUT/'timing_summary.json')
        lines += [f'Measured median across {t["groups"]} clip/group medians: dense16 {1000*t["dense_median_seconds"]:.1f} ms, periodicB4 {1000*t["periodic_median_seconds"]:.1f} ms; median paired dense/periodic ratio {t["median_paired_ratio"]:.3f}.',
            'Encoded bytes in memory through fresh full decode/planning, selected preparation, native encoder/head and CUDA-completed prediction. Two warmups/five interleaved trials per mode, one clip/group; repeats are not independent groups. Proxy selectors need dense16 plus separately logged exhaustive search.', '']
    decision = 'Resolve any recorded access/population/resource gate and resume this same fixed design; recognition results remain pending.'
    if primary:
        if g['CI95_upper'] < 0 and p['CI95_upper'] < 0:
            decision = 'The fixed UCF comparison favors Z_pool over LegacyCosine and the preplanned periodic comparator. A separately preregistered deployable-policy feasibility study may follow, with a new evaluation cohort and explicit cost limits; these dense-reference selectors are not such a policy. The Kinetics result remains inconclusive, so cross-dataset generality is unresolved.'
        elif g['CI95_upper'] < 0:
            decision = 'Retain only the supported relative improvement over LegacyCosine. Improvement over periodic is not established. Inspect the signed directional/KL contributions and dense-error strata before proposing another controller; preserving dense predictions alone is insufficient.'
        elif g['CI95_lower'] > 0:
            decision = 'The fixed primary comparison favors LegacyCosine over Z_pool. Do not proceed to a controller premised on Z_pool recognition improvement; retain this unfavorable replication and examine the fidelity/true-label mismatch using the saved decomposition.'
        else:
            decision = 'The fixed primary comparison is inconclusive, not equivalent. Do not proceed to a controller based on a claimed Z_pool improvement. Retain the achieved uncertainty and use the preplanned decomposition to investigate the fidelity/true-label mismatch; any new hypothesis needs a new, preregistered evaluation cohort.'
    decision += ' Do not extend exposed test groups, search budgets/backbones, or substitute another dataset to obtain significance. Passing validation does not establish publication readiness.'
    lines += ['## Within-dataset comparison and limits', '',
        '|Dataset / head / averaging unit|g|p|', '|---|---|---|',
        '|Kinetics fresh N=120 / 20 classes / source|-0.011773 [-0.034418,+0.008860]|+0.001131 [-0.021947,+0.023645]|',
        f'|UCF101 / NEW 101-class head / clips within group, groups within class|{ucf_g}|{ucf_p}|', '',
        'No raw cross-dataset CE average is a primary statistic. Different labels/heads and independent units prevent that interpretation. This is a second-dataset replication with disjoint experimental groups; it is not universally pretraining-clean. Unknown UCF-to-Kinetics/reupload mappings remain. This split-1 frozen-feature protocol is not the full three-split UCF101 benchmark or a state-of-the-art comparison. Splits 2/3 overlap sources and are not independent datasets.', '',
        'All proxy selectors require 16 dense snippet encodings plus search. True-label task_C/task_U are privileged hindsight. U cross-evaluation holds C-selected schedules fixed and is a target-mismatch sensitivity. Secondary intervals are marginal/unadjusted; sparse strata (<20 represented groups) support no population claim. Bootstrap intervals condition on the fitted head and selected classes, omitting training/model-selection/repeated-research uncertainty.', '',
        '## Claim / evidence', '', '|Claim|Evidence / limit|', '|---|---|',
        '|Archived Kinetics g/p reproduced|baseline_verification.json: genuine per-source tables and saved bootstrap indices; real feature replays listed separately|',
        '|All 101 classes and group isolation|population_plan.json and eligible_manifest.json; population gate blocks if minima fail|',
        '|Recognition improvement|Only real UCF primary_intervals.json can establish this; pending files imply no result|',
        '|Directional + KL attribution|Independent scalar identity checks and per-clip/group contributions; descriptive, not causal|',
        '|Practical cost|Only actual timing_trials.csv at encoded-memory boundary; no periodic latency assigned to an oracle|',
        '|Historical work preserved|Reviewed tree comparison, old seals, unchanged tracked historical result blobs|', '',
        '## Next research decision', '',
        decision, '',
        '## PowerShell', '', '```powershell', COMMAND, AUDIT, '```', '']
    (OUT/'findings.md').write_text('\n'.join(lines), encoding='utf-8')
    archive_note = ('\nThe publication snapshot stores eligibility_progress.json and screened_population.json as lossless .json.gz copies to avoid GitHub file-size limits. Original raw files remain in this workspace; metadata_archive_packaging.json records both byte hashes. Decompress those copies to their original names when using an archival clone. Large real embeddings and full prediction vectors remain in data/source with hash ledgers.\n'
                    if (OUT/'metadata_archive_packaging.json').exists() else '')
    (OUT/'README.md').write_text('# UCF101 fixed-objective replication\n\nSee findings.md, protocol.md and status.json for measured scope and pending stages.\n\n```powershell\n'+COMMAND+'\n'+AUDIT+'\n```\n'+archive_note, encoding='utf-8')


def accounting():
    events = []
    if (OUT/'encoder_events.jsonl').exists():
        from collections import Counter
        counts = Counter((r['phase'], r['event']) for r in map(json.loads, (OUT/'encoder_events.jsonl').read_text().splitlines()))
        events = [dict(phase=k[0], event=k[1], count=v) for k, v in sorted(counts.items())]
    trials = read(OUT/'timing_progress.json')['trials'] if (OUT/'timing_progress.json').exists() else []
    if (OUT/'network_accounting.json').exists():
        network = read(OUT/'network_accounting.json')
        if not network.get('other_preflight_network_reserve_bytes') and not any(r['status'] == 'inflight' for r in network['requests']):
            network['other_preflight_network_reserve_bytes'] = 8*1024**2
            network['charged_bytes'] += 8*1024**2
            save(OUT/'network_accounting.json', network)
    save(OUT/'execution_accounting.json', dict(at_utc=now(), encoder_event_counts=events,
        timing_encoder_starts=sum(t['encoder_starts'] for t in trials), timing_encoder_completions=sum(t['encoder_completions'] for t in trials),
        timing_trials=len(trials), timing_attempts_logged=(OUT/'timing_attempts.jsonl').exists(),
        collection_counts={r: len(read(OUT/f'{r}_collection.json')) if (OUT/f'{r}_collection.json').exists() else 0 for r in ['train', 'development', 'test']},
        input_screening_attempts=sum(1 for _ in (OUT/'technical_inspection_events.jsonl').open(encoding='utf-8')) if (OUT/'technical_inspection_events.jsonl').exists() else 0,
        early_input_work=read(OUT/'early_input_accounting.json') if (OUT/'early_input_accounting.json').exists() else None,
        nominal_calls=80800, nominal_calls_are_planning_only=True,
        actual_new_network=read(OUT/'network_accounting.json') if (OUT/'network_accounting.json').exists() else None,
        historical_outputs_modified=bool(git('diff', '--name-only', '--', 'results/refresh_objective_alignment_v1', 'results/temporal_refresh_generalization_v1'))))


def run():
    preflight()
    from ucf101_proxy_data import metadata, acquire, eligibility
    plan = metadata(); state('metadata', 'prepared', classes=len(plan['classes']), official_clips=len(plan['clips']))
    state('acquisition', 'running')
    directory = acquire(); state('acquisition', 'prepared', archive_sha256=sha(DATA/'UCF101.rar'))
    state('population', 'running')
    manifest = eligibility(plan, directory); state('population', 'prepared', clips=len(manifest['videos']))
    from ucf101_proxy_inference import X3D, smoke, collect, timing
    from ucf101_proxy_analysis import fit, evaluate, precision, summarize, figures
    encoder = X3D()
    result = smoke(manifest, encoder); state('training_smoke', 'pilot', projection=result)
    for role in ['train', 'development']:
        entries = [v for v in manifest['videos'] if v['role'] == role]
        state(role+'_features', 'running', planned_clips=len(entries))
        collect(entries, role, encoder); state(role+'_features', 'prepared', clips=len(entries))
    state('head', 'running'); head = fit(manifest); state('head', 'prepared')
    entries = [v for v in manifest['videos'] if v['role'] == 'development']
    state('development_analysis', 'running')
    dev = evaluate(entries, 'development', manifest['classes'], head)
    state('development_analysis', 'pilot')
    precision(dev, manifest); state('precision', 'pilot')
    freeze(manifest); state('evaluation', 'frozen')
    if not (OUT/'evaluation_exposure.json').exists():
        seal_check()
        save(OUT/'evaluation_exposure.json', dict(first_test_inference_authorized_at_utc=now(),
             seal_sha256=sha(OUT/'evaluation_seal.json'), scope='recorded immediately before any test model call, including features'))
    entries = [v for v in manifest['videos'] if v['role'] == 'test']
    state('test_features', 'running', planned_clips=len(entries))
    collect(entries, 'test', encoder)
    state('test_features', 'evaluated', clips=len(entries)); state('quality', 'running')
    raw = evaluate(entries, 'test', manifest['classes'], head); summarize(raw, manifest['classes'])
    state('quality', 'evaluated')
    state('timing', 'running'); timing(manifest, encoder, head); state('timing', 'evaluated')
    figures(); state('figures', 'awaiting_visual_review')
    from ucf101_proxy_verify import audit
    result = audit(); save(OUT/'verification.json', result)
    accounting(); findings()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Default stage checkpoints are identity checked')
    parser.add_argument('--audit-only', action='store_true', help='Read only; no output-file mutations')
    parser.add_argument('--finalize-reviewed', action='store_true', help='After actual visual review of all four rendered figures')
    args = parser.parse_args()
    if args.audit_only:
        from ucf101_proxy_verify import audit
        print(json.dumps(audit(), indent=2)); return
    existing_status = OUT/'status.json'
    if existing_status.exists():
        completed = read(existing_status)
        if completed.get('current_stage') == 'final' and completed.get('status') == 'verified':
            # A completed resume is a read-only audit, preserving its final checksums.
            from ucf101_proxy_verify import audit
            print(json.dumps(audit(), indent=2)); return
    if args.finalize_reviewed:
        if read(OUT/'status.json')['status'] != 'awaiting_visual_review':
            raise Blocker('Visual finalization requires completed evaluation/timing/audit')
        save(OUT/'figure_review.json', dict(at_utc=now(), rendered_figures_inspected=True,
            files=['primary_intervals', 'fidelity_recognition', 'decomposition', 'measured_cost']))
        state('final', 'verified'); accounting(); findings()
        save(OUT/'SHA256SUMS.json', {rel(p): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS.json', 'run.lock']})
        return
    # Atomic single-runner lock; stale lock may be removed only after confirming process exit.
    lock = OUT/'run.lock'; OUT.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open('x') as f:
            f.write(str(__import__('os').getpid()))
    except FileExistsError:
        raise Blocker('Runner lock exists; confirm recorded process has exited before removing stale run.lock')
    try:
        run()
    except Blocker as exc:
        stage = read(OUT/'status.json').get('current_stage', 'preflight') if (OUT/'status.json').exists() else 'preflight'
        state(stage, 'blocked', reason=str(exc)); accounting(); findings(str(exc)); print('BLOCKED:', exc, flush=True)
        sys.exit(2)
    except Exception as exc:
        stage = read(OUT/'status.json').get('current_stage', 'implementation') if (OUT/'status.json').exists() else 'implementation'
        state(stage, 'incomplete', error_type=type(exc).__name__, reason=str(exc)); accounting(); findings(str(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
