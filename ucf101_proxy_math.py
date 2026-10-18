"""K-generic weighted fitting, label-free selectors and paired group inference.

Synthetic inputs are permitted only in test_ucf101_proxy_replication.py.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr
import refresh_pooling_utils as pool
from ucf101_proxy_common import SEED

OBJECTIVES = ('F_local', 'F_pool', 'Z_local', 'Z_pool')
PROXIES = ('LegacyCosine', *OBJECTIVES)


def hierarchical_weights(entries, classes):
    """Each class, its groups, and a group's clips have equal total weight."""
    frame = pd.DataFrame(entries)
    if set(frame.label) != set(classes):
        raise ValueError('All classes must be represented')
    counts = frame.groupby(['label', 'group']).size()
    groups = frame.groupby('label')['group'].nunique()
    return np.array([1/(len(classes)*groups[v['label']]*counts.loc[(v['label'], v['group'])])
                     for v in entries], dtype=np.float64)


def balanced(frame, columns):
    return frame.groupby(['label', 'group'])[columns].mean().groupby('label').mean().mean()


def fit_head(features, labels, entries, classes, alpha):
    """Historical torch float64 LBFGS settings; weighted CE and scaler moments."""
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    f = np.asarray(features, dtype=np.float64)
    weights = hierarchical_weights(entries, classes)
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
    means = weights @ f
    std = np.sqrt(weights @ np.square(f-means))
    scales = np.where(std <= 1e-12, 1., std)
    x = torch.tensor((f-means)/scales, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.int64)
    sw = torch.tensor(weights, dtype=torch.float64)
    head = torch.nn.Linear(f.shape[1], len(classes), dtype=torch.float64)
    opt = torch.optim.LBFGS(head.parameters(), lr=1., max_iter=400, max_eval=800,
                          tolerance_grad=1e-9, tolerance_change=1e-12,
                          history_size=50, line_search_fn='strong_wolfe')
    calls = 0
    def closure():
        nonlocal calls
        opt.zero_grad()
        ce = torch.nn.functional.cross_entropy(head(x), y, reduction='none')
        loss = (ce*sw).sum()+alpha/2*head.weight.square().sum()
        loss.backward()
        calls += 1
        return loss
    opt.step(closure)
    objective = float(closure().detach())
    gradient = max(float(p.grad.abs().max()) for p in head.parameters())
    iterations = int(opt.state[head.weight]['n_iter'])
    return dict(means=means.tolist(), scales=scales.tolist(), weights=head.weight.detach().numpy().tolist(),
                bias=head.bias.detach().numpy().tolist()), dict(alpha=alpha, objective=objective,
                gradient_max_abs=gradient, closure_calls=calls, iterations=iterations,
                at_iteration_limit=iterations >= 400, convergence_assessment='gradient and objective tolerances logged; no silent retry',
                constant_columns=int((std <= 1e-12).sum()), solver='torch LBFGS historical settings',
                weight_sum=float(weights.sum()), empirical_CE_weighting='class/group/clip')


def select_proxies(features, head, table):
    """No labels or task losses enter this function or its schedule choices."""
    start = time.perf_counter()
    f = np.asarray(features, dtype=np.float64)
    if f.shape[0] != 16 or not np.isfinite(f).all():
        raise ValueError('Expected 16 finite real feature vectors')
    x = (f-head['means'])/head['scales']
    z = x @ np.asarray(head['weights']).T+head['bias']
    qz = z-z.mean(1, keepdims=True)
    objectives = {}
    d = pool.distances(f)  # historical RAW-feature cosine and clamping
    objectives['LegacyCosine'] = d[np.arange(16)[None], table['cache']].mean(1)
    for name, values in [('F', x), ('Z', qz)]:
        pair = np.square(values[:, None]-values[None]).mean(2)
        objectives[name+'_local'] = pair[np.arange(16)[None], table['cache']].mean(1)
        objectives[name+'_pool'] = np.concatenate([
            np.square(table['cache_weights'][lo:lo+256] @ values-values.mean(0)).mean(1)
            for lo in range(0, len(table['cache']), 256)])
        if np.min(objectives[name+'_local']-objectives[name+'_pool']) < -1e-10:
            raise ArithmeticError('Cancellation identity violated')
    choices = {name: pool.first_tie(v) for name, v in objectives.items()}
    b = table['positions'].shape[1]
    periodic = list(range(0, 16, 16//b))
    choices['periodic'] = dict(index=int(np.flatnonzero((table['positions'] == periodic).all(1))[0]))
    return dict(objectives=objectives, choices=choices, candidate_logits=z,
                logits_C=table['cache_weights'] @ z, logits_U=table['unique_weights'] @ z,
                search_seconds=time.perf_counter()-start)


def decomposition(dense, selected, label):
    """CE_S-CE_D = (p_D-y).delta + KL(p_D||p_S), stable float64.

    CE difference = LSE(z_S)-LSE(z_D)-delta_y. KL equals that
    LSE difference minus p_D.delta. Adding cancels p_D.delta.
    """
    d, s = np.asarray(dense, dtype=np.float64), np.asarray(selected, dtype=np.float64)
    # Remove harmless common shifts BEFORE log-softmax and directional dot product.
    # This avoids amplifying logsumexp rounding by a large common delta shift.
    d, s = d-d.max(), s-s.max()
    ld, ls = d-logsumexp(d), s-logsumexp(s)
    p = np.exp(ld)
    delta = s-d
    direction = float(p @ delta-delta[label])
    kl = float(p @ (ld-ls))
    ce = float(-ls[label])
    difference = float(ld[label]-ls[label])
    # Independent scalar max-shift implementation verifies the identity.
    import math
    lse = lambda z: max(z)+math.log(math.fsum(math.exp(float(v-max(z))) for v in z))
    independent = lse(s)-lse(d)-float(delta[label])
    np.testing.assert_allclose(direction+kl, difference, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(independent, difference, atol=1e-10, rtol=1e-10)
    if kl < -1e-10:
        raise ArithmeticError('Negative KL')
    return dict(CE=ce, correct=int(s.argmax() == label), dense_correct=int(d.argmax() == label),
                dense_agreement=int(s.argmax() == d.argmax()), KL=kl, directional=direction,
                delta_CE_dense=difference, decomposition_residual=difference-direction-kl,
                prediction=int(s.argmax()))


def scalar_replay(features, head, label, positions, values, index):
    """Explicit cache copies and per-frame projections, independent of table matmul."""
    f = np.asarray(features, dtype=np.float64)
    x = (f-head['means'])/head['scales']
    w = np.asarray(head['weights'])
    cached, last = [], 0
    for t in range(16):
        if t in positions:
            last = t
        cached.append(last)
    e = np.array([x[r]-x[t] for t, r in enumerate(cached)])
    a = np.array([w @ v for v in e]); a -= a.mean(1, keepdims=True)
    errors = []
    for name, residual in [('F', e), ('Z', a)]:
        local = np.mean([float(v @ v)/len(v) for v in residual])
        mean = residual.mean(0)
        pooled = float(mean @ mean)/len(mean)
        variance = np.mean([float((v-mean) @ (v-mean))/len(v) for v in residual])
        np.testing.assert_allclose(local, pooled+variance, atol=1e-10, rtol=1e-10)
        for suffix, actual in [('local', local), ('pool', pooled)]:
            expected = values['objectives'][name+'_'+suffix][index]
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
            errors.append(abs(actual-expected))
    for pooling, rows in [('C', cached), ('U', positions)]:
        z = np.mean([w @ x[r]+head['bias'] for r in rows], axis=0)
        pooled_z = pool.logits(f[rows].mean(0), head)
        np.testing.assert_allclose(z, pooled_z, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(z, values['logits_'+pooling][index], atol=1e-10, rtol=1e-10)
    return float(max(errors))


def bootstrap_plan(groups, classes, repeats=10000):
    """Indices address one canonical row per (class,group); columns retain class blocks."""
    g = groups[['label', 'group']].drop_duplicates().copy()
    g['_class_order'] = g.label.map({c: i for i, c in enumerate(classes)})
    g = g.sort_values(['_class_order', 'group']).drop(columns='_class_order').reset_index(drop=True)
    if set(g.label) != set(classes):
        raise ValueError('Bootstrap requires all official classes')
    rng = np.random.Generator(np.random.PCG64(SEED))
    blocks, weights = [], []
    for c in classes:
        members = np.flatnonzero(g.label.to_numpy() == c)
        blocks.append(members[rng.integers(len(members), size=(repeats, len(members)))])
        weights.extend([1/(len(classes)*len(members))]*len(members))
    return g, np.concatenate(blocks, axis=1).astype(np.int32), np.asarray(weights)


def interval(values, indices, weights):
    values = np.asarray(values, dtype=np.float64)
    draws = np.sum(values[indices]*weights[None], axis=1)
    ci = np.percentile(draws, [2.5, 97.5], method='linear')
    return dict(mean=float(values @ weights), CI95_lower=float(ci[0]), CI95_upper=float(ci[1]),
                interval_width=float(ci[1]-ci[0]))


def contrast_values(frame):
    """Rows already refer to one clip and budget/pool; all differences paired."""
    f = frame.set_index('selector')
    ce = f.CE
    values = dict(g=ce.Z_pool-ce.LegacyCosine, p=ce.Z_pool-ce.periodic,
                  d=ce.LegacyCosine-ce.periodic,
                  aggregation=((ce.F_pool-ce.F_local)+(ce.Z_pool-ce.Z_local))/2,
                  metric=((ce.Z_local-ce.F_local)+(ce.Z_pool-ce.F_pool))/2,
                  interaction=(ce.Z_pool-ce.Z_local)-(ce.F_pool-ce.F_local),
                  directional_g=f.loc['Z_pool', 'directional']-f.loc['LegacyCosine', 'directional'],
                  KL_g=f.loc['Z_pool', 'KL']-f.loc['LegacyCosine', 'KL'])
    np.testing.assert_allclose(values['g'], values['directional_g']+values['KL_g'], atol=1e-10)
    return values


def schedule_correlations(values, losses):
    return {name: None if np.ptp(v) == 0 or np.ptp(losses) == 0 else float(spearmanr(v, losses).statistic)
            for name, v in values['objectives'].items()}
