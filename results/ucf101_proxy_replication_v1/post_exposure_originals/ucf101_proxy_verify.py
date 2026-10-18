"""Read-only scientific audits; new audit records are written only by the runner."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
import refresh_pooling_utils as pool
from ucf101_proxy_common import *
from ucf101_proxy_math import *


def baseline_verification():
    old = ROOT/'results/refresh_objective_alignment_v1'
    temporal = ROOT/'results/temporal_refresh_generalization_v1'
    verify_hashes(read(old/'evaluation_seal.json')['sha256'])
    verify_hashes(read(temporal/'evaluation_seal.json')['sha256'])
    raw = pd.read_csv(old/'confirmation/per_video_metrics.csv', float_precision='round_trip')
    rows = raw[(raw.representation == 'x3d_xs') & (raw.budget == 4) & (raw.pool == 'C')]
    wide = rows.pivot(index='source_video_id', columns='selector', values='CE').sort_index()
    sources = read(old/'confirmation/bootstrap_sources.json')
    indices = np.load(old/'confirmation/bootstrap_indices.npy', allow_pickle=False)
    recorded = pd.read_csv(old/'confirmation/paired_comparisons.csv', float_precision='round_trip')
    records = {}
    for name, a, b in [('g', 'Z_pool', 'LegacyCosine'), ('p', 'Z_pool', 'periodic')]:
        values = (wide[a]-wide[b]).loc[sources].to_numpy()
        ci = np.percentile(values[indices].mean(1), [2.5, 97.5], method='linear')
        r = recorded[(recorded.representation == 'x3d_xs') & (recorded.budget == 4) & (recorded.contrast == name)].iloc[0]
        actual = [float(values.mean()), *map(float, ci)]
        np.testing.assert_allclose(actual, r[['mean', 'CI95_lower', 'CI95_upper']].to_numpy(float), atol=1e-12, rtol=0)
        records[name] = dict(mean=actual[0], CI95_lower=actual[1], CI95_upper=actual[2], sources=len(values),
                             evidence='archived genuine per-source CE + saved source bootstrap indices; no feature claim')
    hs = read(temporal/'heads/x3d_xs/parameters.json')['parameters']
    ledger = read(old/'confirmation_collection.json')['videos']
    selected = sorted(sources, key=lambda s: (rank('baseline_real_audit', s), s))[:3]
    feature_checks, missing = [], []
    table = pool.schedules(16, 4)
    for source in selected:
        r = ledger[source]; p = ROOT/r['path']
        if not p.exists():
            missing.append(dict(source=source, path=r['path'], affected_check='real feature/schedule recomputation'))
            continue
        if sha(p) != r['sha256']:
            raise Blocker('Historical real feature hash mismatch: '+source)
        with np.load(p, allow_pickle=False) as saved:
            f = saved['x3d_xs']
        y = int(rows[rows.source_video_id == source].label_index.iloc[0])
        v = select_proxies(f, hs, table)
        maximum = 0.
        for method in [*PROXIES, 'periodic']:
            row = rows[(rows.source_video_id == source) & (rows.selector == method)].iloc[0]
            j = v['choices'][method]['index']
            np.testing.assert_array_equal(table['positions'][j], json.loads(row.positions))
            ce = float(pool.loss(v['logits_C'][j:j+1], y)[0])
            np.testing.assert_allclose(ce, row.CE, atol=1e-10, rtol=1e-10)
            maximum = max(maximum, scalar_replay(f, hs, y, table['positions'][j].tolist(), v, j), abs(ce-row.CE))
        feature_checks.append(dict(source=source, path=r['path'], sha256=sha(p), maximum_difference=maximum,
                                   real_shape=list(f.shape), schedules_enumerated=len(table['positions']), methods_checked=7))
    return dict(at_utc=now(), archived_primary=records, real_feature_checks=feature_checks,
                affected_blockers=missing, status='blocked_real_array_checks' if missing else 'passed',
                historical_seals_verified=True, new_encoder_calls=0,
                protocol_paths=[rel(old/'protocol.md'), rel(temporal/'protocol.md')],
                implementation_review='Historical floor-PTS tolerance 1e-9; full-video offline metadata; raw cosine tie1e-12; float64 pooling. Historical uniform clip fitting must be replaced by class/group/clip weighted fitting; H264/5..15s filter must NOT be reused.')


def audit():
    """No writes, acquisition, feature extraction, training or result regeneration."""
    result = dict(baseline=baseline_verification())
    for filename in ['eligible_manifest.json', 'encoder_identity.json', 'head_selection.json', 'primary_intervals.json']:
        path = OUT/filename
        result[filename] = 'present' if path.exists() else 'pending'
    if (OUT/'evaluation_seal.json').exists():
        seal_check((OUT/'evaluation_exposure.json').exists())
        result['seal'] = 'verified'
    if (OUT/'eligible_manifest.json').exists():
        from ucf101_proxy_data import require_population
        m = read(OUT/'eligible_manifest.json')
        require_population(m['videos'], m['classes'])
        missing = []
        for role in ['train', 'development', 'test']:
            p = OUT/f'{role}_collection.json'
            if not p.exists():
                continue
            ledger = read(p)
            for clip, r in ledger.items():
                if not (ROOT/r['path']).exists():
                    missing.append(r['path'])
                elif sha(ROOT/r['path']) != r['sha256']:
                    raise Blocker('Changed real feature '+clip)
        result['missing_real_features'] = missing
        if missing:
            raise Blocker('Real feature integrity audit blocked; missing arrays: '+', '.join(missing[:5]))
        if (OUT/'test_metrics.csv').exists():
            from ucf101_proxy_inference import read_features
            manifest_entries = [v for v in m['videos'] if v['role'] == 'test']
            head = read(OUT/'heads/parameters.json')['parameters']
            rows = pd.read_csv(OUT/'test_metrics.csv', float_precision='round_trip')
            replay = []
            for v in sorted(manifest_entries, key=lambda v: rank('final_real_audit', v['clip']))[:5]:
                features = read_features(v, 'test')
                label = m['classes'].index(v['label'])
                selected = rows[(rows.clip == v['clip']) & (rows.budget == 4) & (rows.pool == 'C')]
                table = pool.schedules(16, 4)
                values = select_proxies(features, head, table)
                maximum = 0.
                for method in [*PROXIES, 'periodic']:
                    r = selected[selected.selector == method].iloc[0]
                    j = values['choices'][method]['index']
                    np.testing.assert_array_equal(table['positions'][j], json.loads(r.positions))
                    ce = float(pool.loss(values['logits_C'][j:j+1], label)[0])
                    np.testing.assert_allclose(ce, r.CE, atol=1e-10, rtol=1e-10)
                    error = scalar_replay(features, head, label, table['positions'][j].tolist(), values, j)
                    terms = decomposition(values['candidate_logits'].mean(0), values['logits_C'][j], label)
                    np.testing.assert_allclose([r.directional, r.KL], [terms['directional'], terms['KL']], atol=1e-10)
                    maximum = max(maximum, error, abs(ce-r.CE))
                replay.append(dict(clip=v['clip'], maximum_difference=maximum, real_features=True))
            result['real_UCF_selected_checks'] = replay
        if (OUT/'test_metrics.csv').exists():
            # Independent table-only g/p/d replay and per-class group draw means.
            table = pd.read_csv(OUT/'test_metrics.csv', float_precision='round_trip')
            f = table[(table.budget == 4) & (table.pool == 'C')]
            pivot = f.pivot(index=['label', 'group', 'clip'], columns='selector', values='CE')
            canonical = read(OUT/'bootstrap_groups.json')
            indices = np.load(OUT/'bootstrap_group_indices.npy', allow_pickle=False)
            index = pd.MultiIndex.from_tuples([(v['label'], v['group']) for v in canonical], names=['label', 'group'])
            archived = read(OUT/'primary_intervals.json')
            for b in archived:
                first, second = {'g': ('Z_pool', 'LegacyCosine'), 'p': ('Z_pool', 'periodic'),
                                 'd': ('LegacyCosine', 'periodic')}[b['contrast']]
                group_values = (pivot[first]-pivot[second]).groupby(['label', 'group']).mean().reindex(index).to_numpy()
                draws = np.zeros(10000)
                class_means = []
                offset = 0
                for c in m['classes']:
                    n = sum(v['label'] == c for v in canonical)
                    class_means.append(float(group_values[offset:offset+n].mean()))
                    draws += group_values[indices[:, offset:offset+n]].mean(1)/101
                    offset += n
                ci = np.percentile(draws, [2.5, 97.5], method='linear')
                np.testing.assert_allclose([np.mean(class_means), *ci],
                    [b['mean'], b['CI95_lower'], b['CI95_upper']], atol=1e-12, rtol=0)
            result['grouped_intervals'] = 'independently recomputed from per-clip table'
    if (OUT/'SHA256SUMS.json').exists():
        verify_hashes(read(OUT/'SHA256SUMS.json'))
        result['published_artifact_hashes'] = 'verified'
    if (OUT/'run_configuration.json').exists():
        original = read(OUT/'run_configuration.json')['historical_tracked_result_blobs']
        # Ignore this study's newly committed files while comparing every old scientific blob.
        current = {line.split('\t', 1)[1]: line.split('\t', 1)[0] for line in git('ls-files', '-s', 'results').splitlines()}
        for line in original:
            metadata, path = line.split('\t', 1)
            if current.get(path) != metadata:
                raise Blocker('Historical tracked artifact changed: '+path)
        old_paths = {line.split('\t', 1)[1] for line in original}
        changed = set(git('diff', '--name-only', '--', 'results').splitlines())
        if changed & old_paths:
            raise Blocker('Historical working-tree scientific bytes changed')
        result['historical_artifacts'] = 'all original tracked result blobs preserved'
    result['state'] = read(OUT/'status.json') if (OUT/'status.json').exists() else 'not_started'
    return result
