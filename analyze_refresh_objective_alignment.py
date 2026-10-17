"""Frozen-objective pilots, input-only feasibility, then sealed confirmation.

Run from the repository root. Completed invocations audit scientific outputs;
they never silently overwrite them. Large real arrays remain ignored assets.
"""
from __future__ import annotations

import os
for _variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_variable] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import refresh_pooling_utils as pool
from temporal_refresh_common import ROOT, REPS, now, read, save, sha, relative
import refresh_objective_alignment_utils as objective
import refresh_objective_alignment_data as data

CODE = ['analyze_refresh_objective_alignment.py', 'refresh_objective_alignment_utils.py',
        'refresh_objective_alignment_data.py', 'refresh_objective_alignment_report.py',
        'test_refresh_objective_alignment.py', 'refresh_pooling_utils.py', 'temporal_refresh_common.py',
        'temporal_refresh_data.py', 'temporal_refresh_encoders.py', 'temporal_refresh_quality.py',
        'research_utils.py', 'benchmark.py']


def code_identity() -> dict:
    return {name: sha(ROOT/name) for name in CODE}


def heads(old: Path, classes: list[str]) -> dict:
    selection = read(old/'head_selection.json')
    result = {}
    for rep in REPS:
        path = old/f'heads/{rep}/parameters.json'
        assert sha(path) == selection[rep]['sha256']
        wrapper = read(path)
        assert wrapper['classes'] == classes
        head = wrapper['parameters']
        d = 2048 if rep == 'x3d_xs' else 512
        assert np.asarray(head['weights']).shape == (20, d)
        assert np.asarray(head['means']).shape == (d,) and np.asarray(head['scales']).shape == (d,)
        assert (np.asarray(head['scales']) > 0).all()
        assert all(np.isfinite(np.asarray(value)).all() for value in head.values())
        result[rep] = head
    return result


def cohorts(old: Path) -> tuple[list[str], dict]:
    manifest = read(old/'dataset_manifest.json')
    groups = {'pilot_development': [v for v in manifest['videos'] if v['recognition_role'] == 'development'],
              'pilot_v1_evaluation_exposed': [v for v in manifest['videos'] if v['recognition_role'] == 'evaluation']}
    classes = manifest['classes']
    assert len(classes) == 20
    for entries in groups.values():
        assert len(entries) == 100 and set(Counter(v['label'] for v in entries).values()) == {5}
    assert not ({v['source_video_id'] for v in groups['pilot_development']} &
                {v['source_video_id'] for v in groups['pilot_v1_evaluation_exposed']})
    return classes, groups


def ledger(old: Path, output: Path, cohort: str) -> dict:
    if cohort == 'confirmation':
        return read(output/'confirmation_collection.json')['videos']
    phase = 'train_development' if cohort == 'pilot_development' else 'evaluation'
    return read(old/f'{phase}_collection.json')['videos']


def references(v: dict, record: dict, provenance_sha: str) -> dict:
    source = v['source_video_id']
    path = ROOT/record['path']
    if not path.exists():
        raise FileNotFoundError(f'Real reference missing for {source}: {record["path"]}; re-extract only verified original {v["relative_path"]} with frozen v1 recipe')
    assert sha(path) == record['sha256'], path
    assert record['input_sha256'] == v['file_identity']['sha256']
    assert record['encoder_provenance_sha256'] == provenance_sha
    assert record['snippet_grid_sha256'] == hashlib.sha256(json.dumps(v['snippet_grid'], sort_keys=True).encode()).hexdigest()
    grid = v['snippet_grid']
    endpoints = np.array(grid['endpoint_timestamps'])
    requested, actual = np.array(grid['requested_timestamps']), np.array(grid['actual_timestamps'])
    indices = np.array(grid['frame_indices'])
    assert endpoints.shape == (16,) and np.isfinite(endpoints).all() and (np.diff(endpoints) > 0).all()
    assert indices.shape == (16, 4) and all(len(set(row)) == 4 for row in indices)
    np.testing.assert_allclose(requested, endpoints[:, None]-np.array([47/30, 32/30, 16/30, 0]), atol=1e-9, rtol=0)
    assert np.isfinite(actual).all() and np.all(actual <= requested+1e-9)
    assert (np.diff(actual, axis=1) > 0).all()
    with np.load(path, allow_pickle=False) as archive:
        arrays = {rep: archive[rep] for rep in REPS}
    for rep, array in arrays.items():
        assert array.dtype == np.float32 and array.shape == (16, 2048 if rep == 'x3d_xs' else 512)
        assert np.isfinite(array).all() and (np.linalg.norm(array.astype(float), axis=1) > 1e-12).all()
    return arrays


def preflight(output: Path, old: Path, args) -> tuple[list[str], dict, dict]:
    objective.verify_seal(ROOT, read(old/'evaluation_seal.json'))
    assert read(old/'verification.json')['status'] == 'passed'
    classes, groups = cohorts(old)
    hs = heads(old, classes)
    provenance = sha(old/'encoder_provenance.json')
    identities = {}
    missing = []
    annotation = {r['youtube_id']: r['label'] for r in __import__('csv').DictReader((ROOT/'data/source/cache_drift_v2/val.csv').open(encoding='utf-8'))}
    for cohort, entries in groups.items():
        records = ledger(old, output, cohort)
        for v in entries:
            source = v['source_video_id']
            assert annotation[source] == v['label']
            record = records[source]
            try:
                references(v, record, provenance)
            except FileNotFoundError as exc:
                missing.append(str(exc))
            identities[source] = dict(cohort=cohort, label=v['label'], input_sha256=v['file_identity']['sha256'],
                                      grid_sha256=record['snippet_grid_sha256'], reference_path=record['path'], reference_sha256=record['sha256'])
    if missing:
        save(output/'missing_prerequisites.json', {'missing': missing, 'status': 'blocked_real_inputs'})
        raise FileNotFoundError('\n'.join(missing))
    if not (output/'protocol.md').exists():
        starting = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        difference = subprocess.check_output(['git', 'diff', '--stat', starting, args.reviewed_commit], cwd=ROOT, text=True)
        inputs = [old/name for name in ['dataset_manifest.json', 'evaluation_manifest.json', 'train_development_manifest.json',
            'evaluation_seal.json', 'encoder_provenance.json', 'head_selection.json', 'verification.json',
            'evaluation_collection.json', 'train_development_collection.json', 'bootstrap_intervals.json',
            'development_quality.csv', 'evaluation_quality.csv', 'evaluation_bootstrap_indices.npy', 'evaluation_bootstrap_sources.json']]
        inputs += [old/f'heads/{rep}/parameters.json' for rep in REPS]
        tracked = subprocess.check_output(['git', 'ls-files', 'results', '*.py', 'requirements*.txt'], cwd=ROOT, text=True).splitlines()
        # Protected confirmation outcomes are never opened, even for hashing.
        history = {name: sha(ROOT/name) for name in tracked if (ROOT/name).is_file()
                   and not name.startswith('results/fresh_refresh_confirmation_v1/')}
        save(output/'input_provenance.json', dict(starting_commit=starting, reviewed_commit=args.reviewed_commit,
            tracked_difference_from_reviewed=difference, inputs={relative(p): sha(p) for p in inputs},
            source_code=code_identity(), references=identities, prior_scientific_sha256=history,
            source_membership={c: [v['source_video_id'] for v in entries] for c, entries in groups.items()},
            training_arrays_loaded=False, protected_outcomes_accessed=False))
        from refresh_objective_alignment_report import protocol
        (output/'protocol.md').write_text(protocol(starting, args), encoding='utf-8')
        save(output/'run_configuration.json', dict(input=relative(old), seed=args.seed, bootstrap=args.bootstrap,
              max_download_gib=args.max_download_gib, device=args.device, code_sha256=code_identity(), created_at_utc=now()))
        save(output/'environment.json', dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
            packages={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'matplotlib', 'torch', 'torchvision', 'av', 'Pillow']},
            historical_environment_modified=False, analysis_dtype='float64', CPU_threads=1, new_package_installs=0))
    config = read(output/'run_configuration.json')
    assert config['seed'] == args.seed and config['bootstrap'] == args.bootstrap
    assert config['max_download_gib'] == args.max_download_gib and config['device'] == args.device
    assert config['code_sha256'] == code_identity(), 'Code changed: preserve run and use a new version or disclose an amendment'
    objective.verify_seal(ROOT, {'sha256': read(output/'input_provenance.json')['inputs']})
    return classes, groups, hs


def reproduce_v1(output: Path, old: Path, groups: dict, hs: dict, classes: list[str]) -> dict:
    target = output/'legacy_reproduction.json'
    if target.exists():
        return read(target)
    start = time.perf_counter()
    table = pool.schedules(16, 4)
    checks = 0
    largest = 0.
    evaluation_d = {}
    for cohort, entries in groups.items():
        historical_cohort = 'development' if cohort == 'pilot_development' else 'evaluation'
        archived = pool.frame(old/f'{historical_cohort}_quality.csv')
        archived = archived[archived.budget.isin([4, 16])].set_index(['source_video_id', 'representation', 'budget', 'policy', 'pool'])
        selected = { (r['source_video_id'], r['representation'], r['policy'], r['pool']): r
                    for r in read(old/f'{historical_cohort}_selected_schedules.json') if r['budget'] == 4 }
        records = ledger(old, output, cohort)
        for v in entries:
            source, y = v['source_video_id'], classes.index(v['label'])
            arrays = references(v, records[source], sha(old/'encoder_provenance.json'))
            for rep, f in arrays.items():
                values = pool.evaluate(f, hs[rep], y, table)
                pindex = int(np.flatnonzero((table['positions'] == [0, 4, 8, 12]).all(1))[0])
                choices = dict(periodic={'index': pindex}, feature_oracle=values['F'], cache_task_oracle=values['SC'], unique_task_oracle=values['SU'])
                dense_z = pool.logits(f, hs[rep]).mean(axis=0)
                old_dense = archived.loc[(source, rep, 16, 'dense16', 'C=U')]
                dense_ce = float(pool.loss(dense_z[None], y)[0])
                np.testing.assert_allclose(dense_ce, old_dense.CE, atol=1e-10, rtol=1e-10)
                assert int(dense_z.argmax() == y) == old_dense.correct
                for policy, choice in choices.items():
                    j = choice['index']
                    for pooling in ['C', 'U']:
                        old_row = archived.loc[(source, rep, 4, policy, pooling)]
                        assert json.loads(old_row.positions) == table['positions'][j].tolist()
                        diff = abs(float(values['loss_'+pooling][j])-old_row.CE)
                        largest = max(largest, diff)
                        np.testing.assert_allclose(values['loss_'+pooling][j], old_row.CE, atol=1e-10, rtol=1e-10)
                        np.testing.assert_allclose(values['logits_'+pooling][j], selected[(source, rep, policy, pooling)]['logits'], atol=1e-10, rtol=1e-10)
                        assert values['correct_'+pooling][j] == old_row.correct
                        checks += 1
                if historical_cohort == 'evaluation' and rep == 'x3d_xs':
                    evaluation_d[source] = float(values['loss_C'][values['F']['index']]-values['loss_C'][pindex])
    sources = read(old/'evaluation_bootstrap_sources.json')
    delta = np.array([evaluation_d[s] for s in sources])
    indices = np.load(old/'evaluation_bootstrap_indices.npy', allow_pickle=False)
    estimate = pool.describe(delta, indices)
    expected = [.009784449348300034, -.012442287572646777, .032961286928129406]
    np.testing.assert_allclose([estimate['mean'], estimate['CI95_lower'], estimate['CI95_upper']], expected, atol=1e-10, rtol=0)
    result = dict(status='passed', archived_selected_rows=checks, maximum_CE_difference=largest,
                  old_X3D_B4_mean_and_interval=estimate, archived_seed=20261024,
                  seconds=time.perf_counter()-start, no_new_encoder_or_decoder_calls=True)
    save(target, result)
    return result


def evaluate_cohort(output: Path, old: Path, entries: list, cohort: str, hs: dict, classes: list) -> pd.DataFrame:
    target = output/f'{cohort}/per_video_metrics.csv'
    if target.exists():
        return pool.frame(target)
    if cohort == 'confirmation':
        objective.verify_seal(ROOT, read(output/'evaluation_seal.json'))
        assert entries == read(output/'confirmation_manifest.json')['videos']
    target.parent.mkdir(parents=True, exist_ok=True)
    rows, selections, correlations, audit = [], [], [], []
    artifacts = []
    records = ledger(old, output, cohort)
    tables = {b: pool.schedules(16, b) for b in [2, 4, 8]}
    start = time.perf_counter()
    for number, v in enumerate(entries):
        source, y = v['source_video_id'], classes.index(v['label'])
        arrays = references(v, records[source], sha(old/'encoder_provenance.json'))
        for rep, f in arrays.items():
            head = hs[rep]
            dense_z = pool.logits(f, head).mean(axis=0)
            base = dict(cohort=cohort, representation=rep, source_video_id=source, label=v['label'], label_index=y)
            rows.append({**base, 'budget': 16, 'selector': 'dense16', 'pool': 'C=U', 'feature_error': 0.,
                         **objective.fidelity(dense_z, dense_z, y), 'positions': json.dumps(list(range(16))), 'search_seconds': 0.})
            for b in ([2, 4, 8] if rep == 'x3d_xs' else [4]):
                table = tables[b]
                values = objective.evaluate(f, head, y, table)
                archive = data.ASSETS/'outcomes'/cohort/rep/f'{source}_B{b}.npz'
                archive.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(archive, **{k: values[k] for k in [*objective.OBJECTIVES, 'LegacyCosine', 'loss_C', 'loss_U', 'correct_C', 'correct_U']})
                artifacts.append(dict(path=relative(archive), sha256=sha(archive)))
                for selector, choice in values['choices'].items():
                    j = choice['index']
                    positions = table['positions'][j].tolist()
                    checked = objective.scalar_check(f, head, y, positions, values, j)
                    audit.append({**base, 'budget': b, 'selector': selector, **checked})
                    for pooling in ['C', 'U']:
                        z = values['logits_'+pooling][j]
                        row = {**base, 'budget': b, 'selector': selector, 'pool': pooling,
                            **objective.fidelity(z, dense_z, y), 'feature_error': float(values['feature_error'][j]),
                            'positions': json.dumps(positions), 'schedule_index': j,
                            'search_seconds': values['search_seconds'],
                            'F_local': float(values['F_local'][j]), 'F_pool': float(values['F_pool'][j]),
                            'Z_local': float(values['Z_local'][j]), 'Z_pool': float(values['Z_pool'][j]),
                            'feature_cancellation_variance': float(values['F_local'][j]-values['F_pool'][j]),
                            'logit_cancellation_variance': float(values['Z_local'][j]-values['Z_pool'][j])}
                        rows.append(row)
                        selections.append({**row, **choice, 'weights': table['cache_weights' if pooling == 'C' else 'unique_weights'][j].tolist(), 'logits': z.tolist()})
                # Representative full-schedule checks independent of selected optima.
                for j in np.linspace(0, len(table['positions'])-1, 5, dtype=int):
                    objective.scalar_check(f, head, y, table['positions'][j].tolist(), values, int(j))
                for pooling in ['C', 'U']:
                    losses = values['loss_'+pooling]
                    rows.append({**base, 'budget': b, 'selector': 'RandomExact', 'pool': pooling,
                        'CE': float(losses.mean()), 'correct': float(values['correct_'+pooling].mean()),
                        'feature_error': float(values['feature_error'].mean()), 'positions': None,
                        'search_seconds': values['search_seconds']})
                    for proxy in ['LegacyCosine', *objective.OBJECTIVES]:
                        correlations.append({**base, 'budget': b, 'pool': pooling, 'proxy': proxy,
                            'schedules': len(losses), **objective.correlation(values[proxy], losses)})
        if (number+1) % 10 == 0:
            print(f'{cohort} fixed objectives {number+1}/{len(entries)}', flush=True)
    pool.csv(target, pd.DataFrame(rows))
    save(target.parent/'selected_schedules.json', selections)
    pool.csv(target.parent/'schedule_rank_correlations.csv', pd.DataFrame(correlations))
    pool.csv(target.parent/'independent_selected_checks.csv', pd.DataFrame(audit))
    save(target.parent/'schedule_artifacts.json', artifacts)
    save(target.parent/'execution.json', dict(analysis_seconds=time.perf_counter()-start,
         real_sources=len(entries), source_representation_budget_jobs=len(artifacts),
         independent_selected_checks=len(audit), maximum_scalar_difference=max(v['maximum_difference'] for v in audit),
         actual_encoder_calls=0, model_fits=0, note='CPU saved-reference analysis, not inference latency; collection separately counted'))
    return pd.DataFrame(rows)


def pairs(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, rep, source, b), group in raw[raw.budget != 16].groupby(['cohort', 'representation', 'source_video_id', 'budget']):
        cells = group.set_index(['selector', 'pool'])
        c = lambda s, p='C': float(cells.loc[(s, p)].CE)
        dense = raw[(raw.cohort == cohort) & (raw.representation == rep) & (raw.source_video_id == source) & (raw.budget == 16)].CE.iloc[0]
        contrast = dict(g=c('Z_pool')-c('LegacyCosine'), d=c('LegacyCosine')-c('periodic'),
            p=c('Z_pool')-c('periodic'), a=c('Z_pool')-dense,
            aggregation=((c('F_pool')-c('F_local'))+(c('Z_pool')-c('Z_local')))/2,
            metric=((c('Z_local')-c('F_local'))+(c('Z_pool')-c('F_pool')))/2,
            interaction=(c('Z_pool')-c('Z_local'))-(c('F_pool')-c('F_local')),
            H_C=c('periodic')-c('task_C'), H_U=c('periodic', 'U')-c('task_U', 'U'),
            W_C=c('task_C', 'U')-c('task_C'), W_U=c('task_U', 'U')-c('task_U'),
            g_U_cross_evaluation=c('Z_pool', 'U')-c('LegacyCosine', 'U'),
            p_U_cross_evaluation=c('Z_pool', 'U')-c('periodic', 'U'))
        for selector in ['LegacyCosine', *objective.OBJECTIVES]:
            contrast[selector+'_accuracy_minus_periodic_pp'] = 100*(float(cells.loc[(selector, 'C')].correct)-float(cells.loc[('periodic', 'C')].correct))
        assert abs(contrast['g']+contrast['d']-contrast['p']) < 1e-10
        assert min(contrast['H_C'], contrast['H_U']) >= -1e-12
        for name, value in contrast.items():
            rows.append(dict(cohort=cohort, representation=rep, source_video_id=source, label=group.label.iloc[0],
                budget=int(b), contrast=name, value=value,
                primary=cohort == 'confirmation' and rep == 'x3d_xs' and b == 4 and name == 'g'))
    return pd.DataFrame(rows)


def summarize(output: Path, cohort: str, raw: pd.DataFrame, classes: list, args) -> pd.DataFrame:
    path = output/cohort
    deltas = pairs(raw)
    pool.csv(path/'per_source_deltas.csv', deltas)
    sources = sorted(raw.source_video_id.unique())
    labels = raw.drop_duplicates('source_video_id').set_index('source_video_id').label.to_dict()
    idx = pool.stratified_indices(sources, labels, classes, args.bootstrap, args.seed)
    np.save(path/'bootstrap_indices.npy', idx)
    save(path/'bootstrap_sources.json', sources)
    comparisons = []
    for (rep, b, contrast), group in deltas.groupby(['representation', 'budget', 'contrast']):
        values = group.set_index('source_video_id').loc[sources].value.to_numpy()
        comparisons.append(dict(cohort=cohort, representation=rep, budget=int(b), contrast=contrast,
            **pool.describe(values, idx), sources=len(sources), primary=bool(group.primary.any()),
            positive=int((values > 1e-12).sum()), negative=int((values < -1e-12).sum()), tied=int((abs(values) <= 1e-12).sum())))
    save(path/'bootstrap_intervals.json', dict(rows=comparisons, seed=args.seed, resamples=args.bootstrap,
        RNG='PCG64', percentile='linear', unit='class-stratified whole source; shared saved indices',
        limits='fixed heads/classes/source pool and historical choices; no training uncertainty; secondaries marginal unadjusted'))
    pool.csv(path/'paired_comparisons.csv', pd.DataFrame(comparisons))
    aggregates, classes_rows, strata = [], [], []
    for (rep, b, selector, pooling), group in raw.groupby(['representation', 'budget', 'selector', 'pool']):
        ordered = group.set_index('source_video_id').loc[sources]
        identity = dict(cohort=cohort, representation=rep, budget=int(b), selector=selector, pool=pooling)
        aggregates.append({**identity, 'sources': len(sources), 'CE': pool.describe(ordered.CE.to_numpy(), idx),
            'accuracy_pp': pool.describe(100*ordered.correct.to_numpy(), idx),
            'macro_accuracy_pp': float(100*group.groupby('label').correct.mean().mean()),
            'expected_correctness': selector == 'RandomExact', 'correct_count_or_expectation': float(group.correct.sum()),
            'mean_feature_error': float(group.feature_error.mean())})
        for label, block in group.groupby('label'):
            classes_rows.append({**identity, 'label': label, 'sources': len(block), 'CE': float(block.CE.mean()), 'accuracy_pp': float(100*block.correct.mean())})
        if selector != 'RandomExact':
            for dense_correct, block in group.groupby('dense_correct'):
                strata.append({**identity, 'dense_correct': int(dense_correct), 'sources': len(block),
                    'CE': float(block.CE.mean()), 'delta_CE_dense': float(block.delta_CE_dense.mean()),
                    'helpful_CE': int((block.delta_CE_dense < -1e-12).sum()), 'harmful_CE': int((block.delta_CE_dense > 1e-12).sum()),
                    'prediction_agreement': float(block.dense_prediction_agreement.mean()),
                    'mean_KL_dense_to_selected': float(block.KL_dense_to_selected.mean())})
    save(path/'aggregate_metrics.json', {'rows': aggregates})
    pool.csv(path/'per_class_metrics.csv', pd.DataFrame(classes_rows))
    pool.csv(path/'dense_correctness_strata.csv', pd.DataFrame(strata))
    correlations = pool.frame(path/'schedule_rank_correlations.csv')
    corr_summary = []
    for (rep, b, proxy, pooling), group in correlations.groupby(['representation', 'budget', 'proxy', 'pool']):
        valid = group.spearman.dropna()
        corr_summary.append(dict(representation=rep, budget=int(b), proxy=proxy, pool=pooling,
            defined=len(valid), undefined=len(group)-len(valid), mean=float(valid.mean()) if len(valid) else None,
            median=float(valid.median()) if len(valid) else None, unit='per-source schedule-wise rank correlation, then equal-source summary'))
    save(path/'rank_correlation_summary.json', corr_summary)
    return deltas


def audit(output: Path, old: Path, args) -> dict:
    """Read-only independent scalar reconstruction and saved-CSV inference audit."""
    from refresh_objective_alignment_report import independent_audit
    return independent_audit(output, old, args)


def resource_smoke(output: Path, old: Path, groups: dict, hs: dict) -> None:
    """Small real cached-reference CPU smoke before bulk new-source inference."""
    target = output/'resource_smoke.json'
    if target.exists():
        return
    import tracemalloc
    chosen = sorted(groups['pilot_development'], key=lambda v: data.ranking(v['source_video_id'], 20261025))[:2]
    records = ledger(old, output, 'pilot_development')
    timings = []
    tracemalloc.start()
    classes = read(old/'dataset_manifest.json')['classes']
    for v in chosen:
        arrays = references(v, records[v['source_video_id']], sha(old/'encoder_provenance.json'))
        for rep in REPS:
            start = time.perf_counter()
            objective.evaluate(arrays[rep], hs[rep], classes.index(v['label']), pool.schedules(16, 4))
            timings.append(dict(source_video_id=v['source_video_id'], representation=rep, B4_search_seconds=time.perf_counter()-start))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    old_seconds = [records[v['source_video_id']]['seconds'] for v in chosen]
    save(target, dict(at_utc=now(), selection='two development parents with lowest fixed seed source-ID hashes; independent of outcomes',
        source_ids=[v['source_video_id'] for v in chosen], CPU_search_timings=timings,
        tracemalloc_peak_bytes=peak, memory_scope='Python/NumPy tracked allocation peak, not whole-process/GPU peak',
        historical_same_recipe_collection_seconds=old_seconds,
        estimated_collection_seconds_at400=float(np.mean(old_seconds)*400),
        historical_estimate_not_a_new_collection_measurement=True,
        extra_encoder_forwards=0, decodes=0, downloads=0,
        free_disk_bytes=__import__('shutil').disk_usage(ROOT).free,
        feasible_local_resources=True, note='Search smoke uses real saved arrays; no synthetic scientific data; collection estimate from exact frozen recipe, not adaptive latency'))
    for b in [2, 4, 8]:
        path = output/f'schedules_B{b}.npz'
        if not path.exists():
            np.savez_compressed(path, **pool.schedules(16, b))


def run(args) -> None:
    old = (ROOT/args.input).resolve()
    output = (ROOT/args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        print(json.dumps(audit(output, old, args), indent=2))
        return
    if (output/'status.json').exists() and read(output/'status.json')['status'] in ['complete', 'exploratory_complete_confirmation_blocked']:
        print('Completed scientific run: auditing without overwrite', flush=True)
        print(json.dumps(audit(output, old, args), indent=2))
        return
    started = time.perf_counter()
    classes, groups, hs = preflight(output, old, args)
    plan = data.inventory(output, old, classes, args.seed)
    print('Input inventory', plan['candidate_counts'], flush=True)
    if args.preflight_only:
        print('Preflight verified real references; no new outcomes computed', flush=True)
        return
    reproduce_v1(output, old, groups, hs, classes)
    pilot_pairs = []
    for cohort, entries in groups.items():
        raw = evaluate_cohort(output, old, entries, cohort, hs, classes)
        if (output/cohort/'per_source_deltas.csv').exists() and (output/cohort/'bootstrap_intervals.json').exists():
            pilot_pairs.append(pool.frame(output/cohort/'per_source_deltas.csv'))
        else:
            pilot_pairs.append(summarize(output, cohort, raw, classes, args))
    precision_path = output/'pilot_precision.json'
    if not precision_path.exists():
        save(precision_path, objective.precision_plan(pd.concat(pilot_pairs), classes))
    if args.pilot_only:
        print('Pilot stage complete; confirmation not attempted', flush=True)
        return
    resource_smoke(output, old, groups, hs)
    feasibility = data.screen(output, plan, read(precision_path), args.max_download_gib)
    if feasibility['confirmation_feasible']:
        seal_path = output/'evaluation_seal.json'
        if not seal_path.exists():
            identity = {**code_identity(), **read(output/'input_provenance.json')['inputs']}
            for name in ['protocol.md', 'confirmation_manifest.json', 'sampling_plan.json', 'source_exclusions.json', 'precision_and_feasibility.json', 'pilot_precision.json']:
                identity[relative(output/name)] = sha(output/name)
            save(seal_path, dict(sealed_at_utc=now(), sha256=identity, N_fixed=feasibility['N_frozen'],
                primary='confirmation x3d_xs B4 C: CE(Z_pool)-CE(LegacyCosine)',
                no_confirmation_inference_before_seal=True))
        objective.verify_seal(ROOT, read(seal_path))
        data.collect(output, old, read(output/'confirmation_manifest.json')['videos'], args.device)
        raw = evaluate_cohort(output, old, read(output/'confirmation_manifest.json')['videos'], 'confirmation', hs, classes)
        summarize(output, 'confirmation', raw, classes, args)
    starts = sum(1 for _ in (output/'encoder_execution_events.jsonl').open()) if (output/'encoder_execution_events.jsonl').exists() else 0
    retained = read(output/'confirmation_collection.json')['videos'] if (output/'confirmation_collection.json').exists() else {}
    completed = sum(sum(v['forward_calls'].values()) for v in retained.values())
    save(output/'execution_accounting.json', dict(recorded_at_utc=now(), invocation_elapsed_seconds=time.perf_counter()-started,
         actual_encoder_forward_starts=starts, completed_retained_forward_calls=completed,
         unmatched_started_calls=starts-completed, retained_reference_vectors=len(retained)*16*3,
         decoded_technical_candidates=feasibility['technically_screened'],
         decoded_collection_videos=len(read(output/'confirmation_collection.json')['videos']) if (output/'confirmation_collection.json').exists() else 0,
         model_fits=0, package_upgrades=0, dense_reference_calls_required_per_video={'resnet18_frame_calls': 80, 'x3d_snippet_calls': 16},
         costs='all16 dense references per representation plus CPU search; no four-forward deployable objective; no new end-to-end timing'))
    from refresh_objective_alignment_report import report
    report(output, old, args)
    verification = audit(output, old, args)
    save(output/'verification.json', verification)
    save(output/'status.json', dict(status='awaiting_figure_review', confirmation_feasible=feasibility['confirmation_feasible'], updated_at_utc=now()))
    print(json.dumps(verification, indent=2), flush=True)
    print('Scientific computation complete; inspect figures then --finalize-reviewed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/temporal_refresh_generalization_v1')
    parser.add_argument('--output', default='results/refresh_objective_alignment_v1')
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--device', default='cuda', choices=['cuda'])
    parser.add_argument('--max-download-gib', type=float, default=20.)
    parser.add_argument('--reviewed-commit', default='caf5e5921d9299305b805364a0e2b91082d73611')
    parser.add_argument('--resume', action='store_true', help='Identity-checked resume; also safe by default')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--finalize-reviewed', action='store_true', help='Record human/agent visual review after computation/audit')
    args = parser.parse_args()
    assert args.seed == 20261025 and args.bootstrap == 10000, 'This named protocol has fixed seed/bootstrap'
    assert 0 < args.max_download_gib <= 20
    if args.finalize_reviewed:
        output = ROOT/args.output
        assert read(output/'status.json')['status'] == 'awaiting_figure_review'
        assert read(output/'verification.json')['status'] == 'passed'
        feasible = read(output/'precision_and_feasibility.json')['confirmation_feasible']
        save(output/'figure_review.json', dict(reviewed_at_utc=now(), PNGs_visually_inspected=True,
            checks='readable labels, signs/units/cohorts, exploratory/confirmation labels, no misleading oracle latency'))
        save(output/'status.json', dict(status='complete' if feasible else 'exploratory_complete_confirmation_blocked',
                                       completed_at_utc=now(), confirmation_feasible=feasible))
        paths = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
        (output/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(output).as_posix()}\n' for p in paths), encoding='utf-8')
        print(read(output/'status.json'))
    else:
        try:
            run(args)
        except Exception as exc:
            # Preserve completed stages. Never convert an exception into success.
            if not args.audit_only:
                save(ROOT/args.output/'last_failure.json', dict(at_utc=now(), error_type=type(exc).__name__, error=str(exc),
                    status='incomplete', resume_command='See protocol.md; rerun same command with --resume'))
            raise


if __name__ == '__main__':
    main()
