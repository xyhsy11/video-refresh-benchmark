"""Pure CPU pooling/schedule arithmetic for a fixed affine recognition head."""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

TOL = 1e-12
ATOL = RTOL = 1e-10
POLICIES = ['periodic', 'feature_oracle', 'cache_task_oracle', 'unique_task_oracle', 'RandomExact']


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip')


def csv(path: Path, value: pd.DataFrame) -> None:
    value.to_csv(path, index=False, float_format='%.17g')


def compare_csv(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    assert expected.columns.tolist() == actual.columns.tolist() and len(expected) == len(actual)
    for c in expected:
        missing = expected[c].isna().to_numpy()
        np.testing.assert_array_equal(missing, actual[c].isna().to_numpy())
        a, b = expected.loc[~missing, c], actual.loc[~missing, c]
        if not len(a): continue
        if pd.api.types.is_numeric_dtype(a.dtype):
            np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), atol=TOL, rtol=0)
        else:
            np.testing.assert_array_equal(a.astype(str).to_numpy(), b.astype(str).to_numpy())


def schedules(n: int, budget: int) -> dict[str, np.ndarray]:
    positions = np.array([(0, *rest) for rest in combinations(range(1, n), budget-1)], dtype=np.int64)
    marks = np.zeros((len(positions), n), dtype=np.int64)
    marks[np.arange(len(positions))[:, None], positions] = positions
    cache = np.maximum.accumulate(marks, axis=1)
    duration = np.diff(np.c_[positions, np.full(len(positions), n)], axis=1)
    cache_weight, unique_weight = np.zeros((len(positions), n)), np.zeros((len(positions), n))
    cache_weight[np.arange(len(positions))[:, None], positions] = duration/n
    unique_weight[np.arange(len(positions))[:, None], positions] = 1/budget
    assert (duration > 0).all() and np.all(positions[:, 0] == 0)
    np.testing.assert_allclose(cache_weight.sum(1), 1., atol=TOL, rtol=0)
    np.testing.assert_allclose(unique_weight.sum(1), 1., atol=TOL, rtol=0)
    return dict(positions=positions, cache=cache, durations=duration,
                cache_weights=cache_weight, unique_weights=unique_weight)


def logits(features: np.ndarray, head: dict) -> np.ndarray:
    return ((np.asarray(features, dtype=np.float64)-head['means'])/head['scales'])@np.asarray(head['weights']).T+head['bias']


def loss(values: np.ndarray, label: int) -> np.ndarray:
    # Preserve historical logaddexp arithmetic and near-tie selection convention.
    return np.logaddexp.reduce(values, axis=1)-values[:, label]


def distances(features: np.ndarray) -> np.ndarray:
    result = np.empty((len(features), len(features)), dtype=np.float64)
    for t, a in enumerate(features.astype(np.float64)):
        for r, b in enumerate(features.astype(np.float64)):
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            assert np.isfinite([na, nb]).all() and min(na, nb) > TOL
            result[t, r] = 1.-float(np.clip(float(np.dot(a, b)/(na*nb)), -1., 1.))
    return result


def first_tie(values: np.ndarray) -> dict:
    minimum = float(values.min())
    ties = np.flatnonzero(values <= minimum+TOL)
    return {'index': int(ties[0]), 'true_minimum': minimum,
            'achieved_objective': float(values[ties[0]]), 'tie_count': len(ties), 'tolerance': TOL}


def evaluate(features: np.ndarray, head: dict, label: int, table: dict) -> dict:
    z = logits(features, head)
    cache_logits, unique_logits = table['cache_weights']@z, table['unique_weights']@z
    d = distances(features)
    error = d[np.arange(len(features))[None, :], table['cache']].mean(1)
    lc, lu = loss(cache_logits, label), loss(unique_logits, label)
    return {'loss_C': lc, 'loss_U': lu, 'feature_error': error,
        'prediction_C': cache_logits.argmax(1).astype(np.uint8), 'prediction_U': unique_logits.argmax(1).astype(np.uint8),
        'correct_C': cache_logits.argmax(1) == label, 'correct_U': unique_logits.argmax(1) == label,
        'logits_C': cache_logits, 'logits_U': unique_logits,
        'F': first_tie(error), 'SC': first_tie(lc), 'SU': first_tie(lu)}


def stratified_indices(sources: list[str], labels: dict[str, str], classes: list[str], repeats: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    blocks = []
    for label in classes:
        members = np.array([i for i, s in enumerate(sources) if labels[s] == label])
        assert len(members) > 0
        blocks.append(members[rng.integers(0, len(members), (repeats, len(members)))])
    return np.concatenate(blocks, axis=1)


def describe(values: np.ndarray, indices: np.ndarray) -> dict:
    bounds = np.percentile(values[indices].mean(1), [2.5, 97.5], method='linear')
    return {'mean': float(values.mean()), 'CI95_lower': float(bounds[0]), 'CI95_upper': float(bounds[1])}


def per_source_contrasts(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, source, budget), group in raw[raw.budget != 16].groupby(['cohort', 'source_video_id', 'budget'], sort=True):
        cells = group.set_index(['policy', 'pool'])
        pc, pu = cells.loc[('periodic', 'C')].CE, cells.loc[('periodic', 'U')].CE
        cc, cu = cells.loc[('cache_task_oracle', 'C')].CE, cells.loc[('cache_task_oracle', 'U')].CE
        uc, uu = cells.loc[('unique_task_oracle', 'C')].CE, cells.loc[('unique_task_oracle', 'U')].CE
        hc, hu, wc, wu = pc-cc, pu-uu, cu-cc, uu-uc
        np.testing.assert_allclose(pc, pu, atol=TOL, rtol=0)
        np.testing.assert_allclose(hc, (pu-cu)+wc, atol=TOL, rtol=0)
        assert min(hc, hu) >= -TOL
        base = {'cohort': cohort, 'source_video_id': source, 'label': group.label.iloc[0], 'budget': int(budget)}
        values = {'H_C': hc, 'H_U': hu, 'Delta': hc-hu, 'W_C': wc, 'W_U': wu,
                  'selection_gain_U_on_SC': pu-cu, 'L_C_SC': cc, 'L_U_SC': cu, 'L_C_SU': uc, 'L_U_SU': uu,
                  'feature_oracle_C_loss_reduction': pc-cells.loc[('feature_oracle', 'C')].CE,
                  'feature_oracle_U_loss_reduction': pu-cells.loc[('feature_oracle', 'U')].CE}
        for name, value in values.items():
            rows.append({**base, 'contrast': name, 'metric': 'CE_nats', 'paired_value': float(value),
                         'primary': cohort == 'development' and budget == 4 and name == 'Delta',
                         'orientation': 'H_C-H_U;positive favors larger cache headroom' if name == 'Delta' else
                             'U-C;positive means cache weighting lowers CE for the same selected set' if name in ['W_C', 'W_U'] else 'formula recorded in protocol'})
        for policy in ['periodic', 'feature_oracle', 'cache_task_oracle', 'unique_task_oracle']:
            a, b = cells.loc[(policy, 'C')], cells.loc[(policy, 'U')]
            rows.append({**base, 'contrast': policy+'_accuracy_U_minus_C', 'metric': 'accuracy_pp',
                         'paired_value': float(100*(b.correct-a.correct)), 'primary': False,
                         'orientation': 'positive means unique pooling more accurate at fixed frames'})
        for policy, pool in [('feature_oracle', 'C'), ('feature_oracle', 'U'), ('cache_task_oracle', 'C'),
                             ('cache_task_oracle', 'U'), ('unique_task_oracle', 'C'), ('unique_task_oracle', 'U')]:
            rows.append({**base, 'contrast': policy+'_'+pool+'_accuracy_vs_periodic', 'metric': 'accuracy_pp',
                         'paired_value': float(100*(cells.loc[(policy, pool)].correct-cells.loc[('periodic', pool)].correct)),
                         'primary': False, 'orientation': 'candidate minus same-pool periodic;positive means accuracy improvement'})
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame, pairs: pd.DataFrame, classes: list[str], seed: int, repeats: int):
    aggregates, intervals, matrices = [], [], {}
    for cohort, group in raw.groupby('cohort', sort=True):
        sources = sorted(group.source_video_id.unique())
        labels = group.drop_duplicates('source_video_id').set_index('source_video_id').label.to_dict()
        idx = stratified_indices(sources, labels, classes, repeats, seed)
        matrices[cohort] = idx
        for (budget, policy, pool), g in group.groupby(['budget', 'policy', 'pool'], sort=True):
            ordered = g.set_index('source_video_id').loc[sources]
            expected = policy == 'RandomExact'
            aggregates.append({'cohort': cohort, 'budget': int(budget), 'policy': policy, 'pool': pool,
                'sources': len(g), 'correct_count_or_expectation': float(g.correct.sum()) if expected else int(g.correct.sum()),
                'correctness_is_expectation': expected, 'top1_accuracy': float(g.correct.mean()),
                'macro_top1_accuracy': float(g.groupby('label').correct.mean().mean()),
                'CE': describe(ordered.CE.to_numpy(), idx),
                'accuracy_pp': describe(100*ordered.correct.to_numpy(), idx),
                'mean_last_cache_feature_error': float(g.feature_error.mean()),
                'mean_effective_weight_count': float(g.effective_weight_count.mean())})
        subset = pairs[pairs.cohort == cohort]
        for (budget, contrast, metric), g in subset.groupby(['budget', 'contrast', 'metric'], sort=True):
            ordered = g.set_index('source_video_id').loc[sources]
            values = ordered.paired_value.to_numpy()
            intervals.append({'cohort': cohort, 'budget': int(budget), 'contrast': contrast, 'metric': metric,
                'primary': bool(g.primary.iloc[0]), 'interval_role': 'primary' if g.primary.iloc[0] else 'secondary marginal unadjusted',
                **describe(values, idx), 'positive_sources': int((values > TOL).sum()),
                'tied_sources': int((abs(values) <= TOL).sum()), 'negative_sources': int((values < -TOL).sum()),
                'orientation': g.orientation.iloc[0], 'sources': len(g)})
    return {'rows': aggregates}, {'contrasts': intervals, 'seed': seed, 'resamples': repeats,
        'rng': 'PCG64', 'percentile': 'linear', 'resampling': 'class-stratified whole sources;shared indices within cohort;cohorts separate'}, matrices
