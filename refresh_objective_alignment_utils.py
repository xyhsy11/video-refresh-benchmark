"""Exact, float64 objectives for saved temporal embeddings; no model fitting.

Array convention: rows are candidates and W has shape [classes, dimensions].
Synthetic arrays belong in test_refresh_objective_alignment.py, never results.
"""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
from scipy.stats import spearmanr
import refresh_pooling_utils as pool

TOL = 1e-12
ATOL = RTOL = 1e-10
OBJECTIVES = ('F_local', 'F_pool', 'Z_local', 'Z_pool')
SELECTORS = ('periodic', 'LegacyCosine', *OBJECTIVES, 'task_C', 'task_U')


def evaluate(features: np.ndarray, head: dict, label: int, table: dict) -> dict:
    """One source/representation/budget, not a cross-video tensor.

    Local losses use pairwise residual norms. Pooled feature residuals are
    calculated in bounded batches rather than materializing S x T x D.
    """
    start = time.perf_counter()
    f = np.asarray(features, dtype=np.float64)
    assert f.shape[0] == 16 and np.isfinite(f).all()
    x = (f - np.asarray(head['means'])) / np.asarray(head['scales'])
    w = np.asarray(head['weights'], dtype=np.float64)
    z = x @ w.T + head['bias']
    qz = z - z.mean(axis=1, keepdims=True)
    result = pool.evaluate(f, head, label, table)  # unchanged historical cosine
    result['LegacyCosine'] = result['feature_error']
    cache = table['cache']
    for name, rows in [('F', x), ('Z', qz)]:
        # First axis is current dense position; second is cached position.
        pair = np.square(rows[:, None] - rows[None, :]).mean(axis=2)
        result[name + '_local'] = pair[np.arange(16)[None, :], cache].mean(axis=1)
        values = []
        for lo in range(0, len(cache), 256):
            residual = table['cache_weights'][lo:lo+256] @ rows - rows.mean(axis=0)
            values.append(np.square(residual).mean(axis=1))
        result[name + '_pool'] = np.concatenate(values)
        assert np.min(result[name+'_local']-result[name+'_pool']) >= -ATOL
    periodic = list(range(0, 16, 16 // table['positions'].shape[1]))
    p = int(np.flatnonzero((table['positions'] == periodic).all(axis=1))[0])
    result['choices'] = {key: pool.first_tie(result[key]) for key in (*OBJECTIVES, 'LegacyCosine')}
    result['choices'].update(periodic={'index': p}, task_C=result['SC'], task_U=result['SU'])
    result['dense_logits'] = z.mean(axis=0)
    result['dense_CE'] = float(pool.loss(z.mean(axis=0)[None], label)[0])
    result['candidate_logits'] = z
    result['search_seconds'] = time.perf_counter() - start
    return result


def log_probability(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return z - np.logaddexp.reduce(z, axis=-1, keepdims=True)


def fidelity(z: np.ndarray, dense_z: np.ndarray, label: int) -> dict:
    lp, ld = log_probability(z), log_probability(dense_z)
    pred, dense_pred = int(np.argmax(z)), int(np.argmax(dense_z))
    discrepancy = z - dense_z
    centered = discrepancy - discrepancy.mean()
    bound = float(np.sqrt(2.) * np.linalg.norm(centered))
    loss = float(pool.loss(z[None], label)[0])
    dense_loss = float(pool.loss(dense_z[None], label)[0])
    assert abs(loss-dense_loss) <= bound + ATOL
    kl = float(np.sum(np.exp(ld) * (ld-lp)))
    assert kl >= -ATOL
    return dict(CE=loss, correct=int(pred == label), prediction_index=pred,
                dense_correct=int(dense_pred == label), dense_prediction_index=dense_pred,
                dense_prediction_agreement=int(pred == dense_pred),
                KL_dense_to_selected=kl, centered_logit_discrepancy_norm=float(np.linalg.norm(centered)),
                CE_bound=bound, delta_CE_dense=loss-dense_loss)


def scalar_check(features: np.ndarray, head: dict, label: int, positions: list[int],
                 expected: dict, index: int) -> dict:
    """Independent explicit cache-copy computation; no vectorized pooling helper."""
    f = np.asarray(features, dtype=np.float64)
    assert len(positions) == len(set(positions)) and positions == sorted(positions) and positions[0] == 0
    x = (f - np.asarray(head['means'])) / np.asarray(head['scales'])
    w, bias = np.asarray(head['weights']), np.asarray(head['bias'])
    cached = []
    last = None
    for t in range(16):
        if t in positions:
            last = t
        cached.append(last)
    replay = np.array([f[r].copy() for r in cached])
    residual = np.array([x[r]-x[t] for t, r in enumerate(cached)])
    centered = []
    for e in residual:
        a = w @ e
        centered.append(a - a.mean())
    centered = np.array(centered)
    errors = []
    for name, values in [('F', residual), ('Z', centered)]:
        local = float(np.mean([np.dot(row, row)/len(row) for row in values]))
        mean = np.mean(values, axis=0)
        pooled = float(np.dot(mean, mean)/len(mean))
        variance = float(np.mean([np.dot(row-mean, row-mean)/len(row) for row in values]))
        np.testing.assert_allclose(local, pooled+variance, atol=ATOL, rtol=RTOL)
        for suffix, actual in [('local', local), ('pool', pooled)]:
            errors.append(abs(actual-float(expected[name+'_'+suffix][index])))
            np.testing.assert_allclose(actual, expected[name+'_'+suffix][index], atol=ATOL, rtol=RTOL)
    for name, selected in [('C', replay), ('U', f[positions])]:
        mean = selected.mean(axis=0)
        z = ((mean-np.asarray(head['means']))/head['scales']) @ w.T + bias
        # Also check average affine logits, with explicit per-row projection.
        affine_rows = [w @ ((row-head['means'])/head['scales']) + bias for row in selected]
        np.testing.assert_allclose(z, np.mean(affine_rows, axis=0), atol=ATOL, rtol=RTOL)
        m = float(z.max())
        ce = m + float(np.log(sum(np.exp(float(v)-m) for v in z))) - float(z[label])
        errors.append(abs(ce-float(expected['loss_'+name][index])))
        np.testing.assert_allclose(ce, expected['loss_'+name][index], atol=ATOL, rtol=RTOL)
    return {'maximum_difference': max(errors), 'outputs': len(replay)}


def correlation(a: np.ndarray, b: np.ndarray) -> dict:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {'spearman': None, 'reason': 'constant proxy or true-label CE across schedules'}
    value = float(spearmanr(a, b).statistic)
    assert np.isfinite(value)
    return {'spearman': value, 'reason': None}


def verify_seal(root: Path, seal: dict) -> None:
    for name, digest in seal['sha256'].items():
        if not (root/name).is_file() or pool.sha(root/name) != digest:
            raise RuntimeError(f'Frozen artifact changed or missing: {name}')


def precision_plan(pairs, classes: list[str], h: float = .015) -> dict:
    import math
    estimates = []
    for cohort in sorted(pairs.cohort.unique()):
        for contrast in ['g', 'd']:
            rows = pairs[(pairs.cohort == cohort) & (pairs.representation == 'x3d_xs') &
                         (pairs.budget == 4) & (pairs.contrast == contrast)]
            variances = {label: float(rows[rows.label == label].value.var(ddof=1)) for label in classes}
            assert all(np.isfinite(list(variances.values())))
            estimates.append(dict(cohort=cohort, contrast=contrast, class_variances=variances,
                                  average_class_variance=float(np.mean(list(variances.values())))))
    v = 1.25 * max(row['average_class_variance'] for row in estimates)
    requested = 20*math.ceil(max(100, 1.96**2*v/h**2)/20)
    return dict(h_target_nats=h, variance_inflation=1.25, V=v, estimates=estimates,
                N_requested=requested, resource_source_ceiling=400,
                predicted_halfwidth_at_400=1.96*math.sqrt(v/400),
                predicted_halfwidth_at_uncapped_requested=1.96*math.sqrt(v/requested),
                assumptions='normal approximation; pilot within-class variances transfer to unseen parents; fixed heads/classes; no guaranteed power')
