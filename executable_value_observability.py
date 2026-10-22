"""Experiment 26 pure CPU arithmetic. No encoder, head fit, or controller API."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import numpy as np
from causal_observability_policy import Observation, history_vector, cheap_signal

SEED = 20261022
PERIODIC = (0, 4, 8, 12)
VIEWS = ('S', 'H', 'HG', 'HGH')
SIZES = dict(S=19, H=83, HG=115, HGH=147)


def stable(*parts):
    return hashlib.sha256('|'.join(map(str, (SEED, *parts))).encode()).hexdigest()


def clip_weights(meta):
    classes = sorted({m['label'] for m in meta})
    families = {c: {m['family'] for m in meta if m['label'] == c} for c in classes}
    n = Counter((m['label'], m['family']) for m in meta)
    return np.array([1/(len(classes)*len(families[m['label']])*n[m['label'], m['family']]) for m in meta])


def make_folds(meta):
    """Input-only greedy squared imbalance of per-class clip and component loads."""
    classes = sorted({m['label'] for m in meta}); groups = defaultdict(list)
    for m in meta:
        groups[m['family']].append(m)
    vectors = {g: np.array([sum(m['label'] == c for m in mm) for c in classes], float) for g, mm in groups.items()}
    total = sum(vectors.values()); load = np.zeros((5, len(classes))); counts = np.zeros(5)
    mapping = {}
    order = sorted(groups, key=lambda g: (-float(np.sum(vectors[g]/total)), stable('fold-component', g)))
    for g in order:
        v = vectors[g]
        costs = [float(np.sum(((load[f]+v)/(total/5))**2-(load[f]/(total/5))**2)) +
                 .1*((counts[f]+1)**2-counts[f]**2)/(len(groups)/5)**2 for f in range(5)]
        f = min(range(5), key=lambda f: (costs[f], stable('fold-tie', g, f)))
        mapping[g] = f; load[f] += v; counts[f] += 1
    folds = np.array([mapping[m['family']] for m in meta], int)
    diagnostics = []
    for f in range(5):
        train = [m for m, k in zip(meta, folds) if k != f]
        test = [m for m, k in zip(meta, folds) if k == f]
        minima = min(len({m['family'] for m in train if m['label'] == c}) for c in classes)
        assert len({m['label'] for m in train}) == 51 and minima >= 1
        diagnostics.append(dict(fold=f, train_clips=len(train), test_clips=len(test),
            train_classes=len({m['label'] for m in train}), test_classes=len({m['label'] for m in test}),
            min_training_components_per_class=minima,
            min_training_clips_per_class=min(sum(m['label'] == c for m in train) for c in classes),
            test_components=len({m['family'] for m in test})))
    return folds, mapping, diagnostics


def hog(thumbnail):
    """32px, replicated-edge central difference /2, 4x4 cells, 9 circular bins."""
    a = np.asarray(thumbnail, dtype=np.float64)/255
    if a.shape != (32, 32):
        raise ValueError('Genuine uint8 32x32 thumbnail required')
    p = np.pad(a, 1, mode='edge')
    dx = (p[1:-1, 2:]-p[1:-1, :-2])/2
    dy = (p[2:, 1:-1]-p[:-2, 1:-1])/2
    magnitude = np.hypot(dx, dy)
    b = (np.arctan2(dy, dx) % np.pi)*9/np.pi
    lo = np.floor(b).astype(int) % 9; fraction = b-np.floor(b)
    cell = np.repeat(np.arange(4), 8)[:, None]*4+np.repeat(np.arange(4), 8)[None, :]
    out = np.zeros((16, 9))
    np.add.at(out, (cell.ravel(), lo.ravel()), (magnitude*(1-fraction)).ravel())
    np.add.at(out, (cell.ravel(), ((lo+1) % 9).ravel()), (magnitude*fraction).ravel())
    return out/np.sqrt(np.square(out).sum(1, keepdims=True)+1e-6**2)


def observation(t, past_features, head, gray=None, hog_transition=None):
    """Only explicitly supplied earlier Periodic refreshes can enter this API."""
    expected = [u for u in PERIODIC if u < t]
    if t not in range(1, 13) or sorted(past_features) != expected:
        raise ValueError('Only original Periodic earlier refreshed features are permitted')
    age = t-expected[-1]; remaining = 4-len(expected)
    s = np.r_[np.eye(16)[t], age, remaining, 16-t]
    h = history_vector(Observation(t, age, remaining, tuple(past_features[u] for u in expected), ()), head)
    h = np.r_[s, h, float(0 < remaining < 16-t)]
    if gray is None:
        return h
    if np.asarray(gray).shape != (32,):
        raise ValueError('32 historical cheap statistics required')
    hg = np.r_[h, gray]
    if hog_transition is None:
        return hg
    if np.asarray(hog_transition).shape != (32,):
        raise ValueError('32 HOG transitions required')
    return np.r_[hg, hog_transition]


def trajectory(thumbnails, supplied_history, head):
    """Endpoint chronology; references update AFTER current decisions at anchors."""
    if sorted(supplied_history) != list(PERIODIC):
        raise ValueError('History supply must contain only four Periodic anchors')
    descriptors = [hog(a) for a in thumbnails]
    out = []
    for t in range(1, 13):
        r = max(u for u in PERIODIC if u < t)
        changes = np.r_[np.linalg.norm(descriptors[t]-descriptors[r], axis=1),
                        np.linalg.norm(descriptors[t]-descriptors[t-1], axis=1)]
        earlier = {u: supplied_history[u] for u in PERIODIC if u < t}
        out.append(observation(t, earlier, head, cheap_signal(thumbnails[t], thumbnails[t-1]), changes))
    return np.array(out)


def ridge_fit(x, y, weights):
    """(1/2n)sum mean-one-w*(y-b-z beta)^2 + (1/2)||beta||^2."""
    w = np.asarray(weights, float)*len(weights)/np.sum(weights)
    mu = (w @ x)/len(w); sd = np.sqrt(w @ np.square(x-mu)/len(w))
    sd = np.where(sd <= 1e-12, 1., sd)
    z = np.c_[(x-mu)/sd, np.ones(len(x))]
    lhs = z.T @ (w[:, None]*z)/len(w)+np.diag(np.r_[np.ones(x.shape[1]), 0.])
    rhs = z.T @ (w*y)/len(w); coef = np.linalg.solve(lhs, rhs)
    return dict(means=mu.tolist(), scales=sd.tolist(), coefficients=coef[:-1].tolist(),
                intercept=float(coef[-1]), alpha=1., normal_residual=float(abs(lhs@coef-rhs).max()))


def score(model, x):
    return ((x-np.asarray(model['means']))/np.asarray(model['scales'])) @ np.asarray(model['coefficients'])+model['intercept']


def actions(prediction, positions):
    return np.where(prediction > 1e-12, 1, np.where(prediction < -1e-12, 0, np.isin(positions, PERIODIC).astype(int)))


def feature_names(classes):
    s = [f'position_is_{i:02d}' for i in range(16)]+['cache_age', 'remaining_budget', 'remaining_candidates']
    h = ['historical_t_over15', 'historical_age_over15', 'historical_remaining_over4']
    h += [f'cached_probability_{c}' for c in classes]
    h += ['cached_entropy', 'cached_max_probability', 'cached_logit_margin', 'cached_norm_over_sqrt2048',
          'cached_previous_refresh_cosine', 'cached_refresh_difference_over_sqrt2048', 'refresh_count_over4',
          'refresh_probability_margin_mean', 'refresh_probability_margin_std', 'discretionary']
    gray = [f'gray_histogram_{j}' for j in range(8)]+[f'gray_cell_mean_{j:02d}' for j in range(16)]
    gray += ['gray_mean', 'gray_std', 'gray_abs_difference_mean', 'gray_difference_rms', 'gray_difference_max',
             'gray_fraction_difference_gt_point1', 'gray_histogram_l1', 'gray_mean_difference_gt_point2']
    hh = [f'hog_l2_last_refresh_cell_{j:02d}' for j in range(16)]+[f'hog_l2_previous_candidate_cell_{j:02d}' for j in range(16)]
    assert len(s+h+gray+hh) == 147
    return s+h+gray+hh
