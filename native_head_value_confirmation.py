"""Experiment 27 contracts. Inputs are genuine observations or official metadata.

No data generators, recognition-head fitting, or adaptive-controller API.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import gzip
from itertools import combinations
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

STUDY = 'native_head_value_confirmation_v1'
SEED = 20261023
PERIODIC = (0, 4, 8, 12)
VIEWS = dict(S=19, H=432, HG=464, HGH=496)
SCHEDULES = tuple((0, *tail) for tail in combinations(range(1, 16), 3))
CHECKPOINT_SHA = '6dafc96144c88f800b190d2cb179ff69b8fd48c40afd4deca6c1e61f76fdfe86'
REVISION = 'f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0'
# Explicit annotation -> native-class spelling alias, never an index guess.
ALIASES = {}  # Current official annotation/native spellings agree after quote removal.


def rank(kind, value):
    return hashlib.sha256(f'{STUDY}|{SEED}|{kind}|{value}'.encode()).hexdigest()


def derived_seed(kind):
    return int(rank('seed', kind)[:16], 16)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path=Path(path)
    if path.name.endswith('.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.writing')
    if path.name.endswith('.gz'):
        payload=(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n').encode('utf-8')
        with tmp.open('wb') as f:
            with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as archive:archive.write(payload)
            f.flush();os.fsync(f.fileno())
        atomic_replace(tmp,path)
        return
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())
    atomic_replace(tmp,path)


def atomic_replace(source,target):
    """Readers/indexers can briefly deny Windows rename of an existing file."""
    for attempt in range(24):
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if attempt==23:raise
            time.sleep(min(.05*(attempt+1),.5))


def class_mapping(native, annotation_labels):
    """Official index values determine ordering; never sort names into indices."""
    if isinstance(native, list):
        pairs = [(name, i) for i, name in enumerate(native)]
    elif all(str(k).isdigit() for k in native):
        pairs = [(v, int(k)) for k, v in native.items()]
    else:
        pairs = [(k, int(v)) for k, v in native.items()]
    # Official PySlowFast metadata retains literal CSV quote delimiters on
    # multiword labels. Remove exactly the enclosing pair, preserving indices.
    pairs = [(name[1:-1] if name.startswith('"') and name.endswith('"') else name, index)
             for name, index in pairs]
    mapping = dict(pairs)
    if len(mapping) != 400 or set(mapping.values()) != set(range(400)):
        raise ValueError('Official native map must contain exactly indices 0..399')
    result = {}
    for label in sorted(annotation_labels):
        canonical = ALIASES.get(label, label)
        if canonical not in mapping:
            raise ValueError(f'Unresolved official/native spelling: {label!r}')
        result[label] = mapping[canonical]
    if len(result) != 400 or len(set(result.values())) != 400:
        raise ValueError('Annotation/native mapping is not a 400-class bijection')
    return result


def repair(t, action):
    if t not in range(1, 13) or action not in (0, 1):
        raise ValueError('Only the twelve reset states and binary actions are legal')
    prefix = [u for u in PERIODIC if u < t]
    remaining = 4-len(prefix)-action
    anchors = [u for u in PERIODIC[1:] if u > t]
    anchors = anchors[-remaining:] if remaining else []
    if len(anchors) < remaining:
        anchors += [u for u in range(t+1, 16) if u not in anchors][:remaining-len(anchors)]
    return tuple(prefix+([t] if action else [])+sorted(anchors))


def pool_weights():
    w = np.zeros((455, 16), dtype=np.float64)
    for j, schedule in enumerate(SCHEDULES):
        for k, u in enumerate(schedule):
            w[j, u] = ((schedule[k+1] if k < 3 else 16)-u)/16
    return w


def consequences(logits, features, y):
    z = np.asarray(logits, dtype=np.float64)
    f = np.asarray(features, dtype=np.float64)
    if z.shape != (16, 400) or f.shape != (16, 2048) or not np.isfinite(z).all() or not np.isfinite(f).all():
        raise ValueError('Genuine finite native logits/features required')
    pooled = pool_weights() @ z
    loss = logsumexp(pooled, axis=1)-pooled[:, y]
    prediction = pooled.argmax(1)
    ids = np.array([[SCHEDULES.index(repair(t, a)) for a in (0, 1)] for t in range(1, 13)])
    branch_loss = loss[ids]
    drift = []
    for t in range(1, 13):
        r = max(u for u in PERIODIC if u < t)
        drift.append(1-float(f[t] @ f[r])/max(np.linalg.norm(f[t])*np.linalg.norm(f[r]), 1e-12))
    dense = z.mean(0)
    p = SCHEDULES.index(PERIODIC)
    return dict(loss=loss, prediction=prediction, branches=ids, branch_loss=branch_loss,
                advantage=branch_loss[:, 0]-branch_loss[:, 1], drift=np.array(drift),
                periodic_loss=loss[p], periodic_correct=float(prediction[p] == y),
                dense_loss=logsumexp(dense)-dense[y], dense_correct=float(dense.argmax() == y))


def native_observation(t, history, gray=None, hog_transition=None):
    """History contains ONLY earlier refreshed (feature, genuine native logits)."""
    from causal_observability_policy import softmax
    expected = [u for u in PERIODIC if u < t]
    if t not in range(1, 13) or sorted(history) != expected:
        raise ValueError('Original Periodic earlier history required')
    f = np.array([history[u][0] for u in expected], dtype=float)
    z = np.array([history[u][1] for u in expected], dtype=float)
    if f.shape != (len(expected), 2048) or z.shape != (len(expected), 400):
        raise ValueError('Untruncated native history required')
    age = t-expected[-1]
    remaining = 4-len(expected)
    s = np.r_[np.eye(16)[t], age, remaining, 16-t]
    p = softmax(z[-1])
    probs = np.array([softmax(v) for v in z])
    top_probs = np.partition(probs, -2, axis=1)[:, -2:]
    margins = top_probs.max(1)-top_probs.min(1)
    prev = f[-2] if len(f) > 1 else f[-1]
    norm = np.linalg.norm(f[-1])
    cosine = float(f[-1] @ prev)/max(norm*np.linalg.norm(prev), 1e-12)
    top_z = np.partition(z[-1], -2)[-2:]
    h = np.r_[s, t/15, age/15, remaining/4, p,
              -np.sum(p*np.log(np.maximum(p, 1e-300))), p.max(), top_z.max()-top_z.min(),
              norm/np.sqrt(2048), cosine, np.linalg.norm(f[-1]-prev)/np.sqrt(2048),
              len(f)/4, margins.mean(), margins.std(), float(0 < remaining < 16-t)]
    if h.shape != (VIEWS['H'],):
        raise ValueError('Native history schema mismatch')
    if gray is None:
        return h
    if np.asarray(gray).shape != (32,):
        raise ValueError('Historical 32 grayscale statistics required')
    if hog_transition is None:
        return np.r_[h, gray]
    if np.asarray(hog_transition).shape != (32,):
        raise ValueError('Historical 32 HOG transitions required')
    return np.r_[h, gray, hog_transition]


def feature_schema(classes):
    if len(classes)!=400:
        raise ValueError('Native class order must retain 400 names')
    names = [f'position_is_{i:02d}' for i in range(16)]+['cache_age','remaining_budget','remaining_candidates']
    names += ['historical_t_over15','historical_age_over15','historical_remaining_over4']
    names += ['cached_probability_'+c for c in classes]
    names += ['cached_entropy','cached_max_probability','cached_logit_margin','cached_norm_over_sqrt2048',
              'cached_previous_refresh_cosine','cached_refresh_difference_over_sqrt2048','refresh_count_over4',
              'refresh_probability_margin_mean','refresh_probability_margin_std','discretionary']
    if len(names)!=VIEWS['H']:
        raise ValueError('History schema dimension mismatch')
    names += [f'gray_histogram_{j}' for j in range(8)]+[f'gray_cell_mean_{j:02d}' for j in range(16)]
    names += ['gray_mean','gray_std','gray_abs_difference_mean','gray_difference_rms','gray_difference_max',
              'gray_fraction_difference_gt_point1','gray_histogram_l1','gray_mean_difference_gt_point2']
    names += [f'hog_l2_last_refresh_cell_{j:02d}' for j in range(16)]+[f'hog_l2_previous_candidate_cell_{j:02d}' for j in range(16)]
    if len(names)!=VIEWS['HGH']:
        raise ValueError('HGH schema dimension mismatch')
    return names


def interval_claims(interval):
    lo,hi = interval
    claims = []
    if lo>.01:
        claims.append('MATERIAL_INCREMENTAL_BENEFIT')
    if lo>0 and lo<=.01:
        claims.append('POSITIVE_BUT_MATERIALITY_UNRESOLVED')
    if hi<.01:
        claims.append('BENEFIT_AT_LEAST_POINT01_RULED_OUT_AT_CONDITIONAL_SCOPE')
    if lo>=-.01 and hi<=.01:
        claims.append('BOUNDED_SMALL_EFFECT')
    if hi<-.01:
        claims.append('MATERIAL_HARM')
    return claims or ['UNRESOLVED']


def trajectory(thumbnails, features, logits):
    from executable_value_observability import hog
    from causal_observability_policy import cheap_signal
    if np.asarray(thumbnails).shape != (16, 32, 32):
        raise ValueError('Actual endpoint thumbnails required')
    rows = []
    # Chronological calculation; descriptors are never constructed ahead of t.
    descriptors = [hog(thumbnails[0])]
    for t in range(1, 13):
        descriptors.append(hog(thumbnails[t]))
        r = max(u for u in PERIODIC if u < t)
        hh = np.r_[np.linalg.norm(descriptors[t]-descriptors[r], axis=1),
                   np.linalg.norm(descriptors[t]-descriptors[t-1], axis=1)]
        history = {u: (features[u], logits[u]) for u in PERIODIC if u < t}
        rows.append(native_observation(t, history, cheap_signal(thumbnails[t], thumbnails[t-1]), hh))
    return np.asarray(rows)


def validate_components(meta, per_class=None):
    if len({m['component'] for m in meta}) != len(meta):
        raise ValueError('One canonical clip per global component, including across classes, required')
    counts = Counter(m['y'] for m in meta)
    if set(counts) != set(range(400)) or (per_class is not None and set(counts.values()) != {per_class}):
        raise ValueError('All 400 classes with fixed per-class independent component counts required')
    return counts


def folds(meta):
    validate_components(meta, 8)
    result = np.empty(len(meta), dtype=int)
    for c in range(400):
        ids = sorted([i for i, m in enumerate(meta) if m['y'] == c], key=lambda i: rank('fold', meta[i]['component']))
        for j, i in enumerate(ids):
            result[i] = j//2
    return result


def precision_plan(meta, primary):
    validate_components(meta, 8)
    primary = np.asarray(primary, dtype=float)
    if primary.shape != (len(meta),) or not np.isfinite(primary).all():
        raise ValueError('Complete genuine fitting-only OOF source consequences required')
    variances = [float(np.var([primary[i] for i, m in enumerate(meta) if m['y'] == c], ddof=1)) for c in range(400)]
    projections = {str(m): {str(a): 1.96*a*np.sqrt(sum(variances)/(400**2*m)) for a in (1., 1.5, 2.)} for m in range(8, 21)}
    selected = next((m for m in range(8, 21) if projections[str(m)]['1.5'] <= .005), None)
    return dict(selected_m=selected, per_class_variance=variances, projections=projections,
                zero_variance_classes=[c for c, v in enumerate(variances) if v == 0],
                transfer_warning='OOF variance transfer to new split/final fits is uncertain; no coverage/power guarantee')


def source_inference(meta, values, draw_path):
    counts = validate_components(meta)
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != len(meta) or not np.isfinite(x).all() or min(counts.values()) < 8 or max(counts.values())>20 or len(set(counts.values()))!=1:
        raise ValueError('Complete finite per-source consequences required')
    groups = [np.array([i for i, m in enumerate(meta) if m['y'] == c]) for c in range(400)]
    if Path(draw_path).exists():
        with np.load(draw_path, allow_pickle=False) as saved:
            draws = saved['indices']
        if draws.shape != (10000, len(meta)):
            raise ValueError('Saved draw shape mismatch')
    else:
        rng = np.random.Generator(np.random.PCG64(derived_seed('bootstrap')))
        draws = np.concatenate([rng.choice(g, size=(10000, len(g)), replace=True) for g in groups], axis=1).astype(np.int16)
        tmp = Path(str(draw_path)+'.writing')
        with tmp.open('wb') as f:
            np.savez_compressed(f, indices=draws)
        tmp.replace(draw_path)
    theta = np.mean([x[g].mean(0) for g in groups], axis=0)
    boot = np.zeros((10000, x.shape[1]))
    offset = 0
    vc = []
    for g in groups:
        d = draws[:, offset:offset+len(g)]
        if not np.isin(d, g).all():
            raise ValueError('Bootstrap draw crosses class/component boundaries')
        boot += x[d].mean(1)/400
        vc.append(x[g].var(0, ddof=1)/(400**2*len(g)))
        offset += len(g)
    vc = np.array(vc)
    variance = vc.sum(0)
    denominator = np.sum([v*v/(len(g)-1) for v, g in zip(vc, groups)], axis=0)
    df = np.divide(variance**2, denominator, out=np.full_like(variance, np.inf), where=denominator > 0)
    half = student_t.ppf(.975, df)*np.sqrt(variance)
    return dict(mean=theta.tolist(), bootstrap95=np.quantile(boot, [.025, .975], axis=0, method='linear').T.tolist(),
                analytic95=np.array([theta-half, theta+half]).T.tolist(),
                df=[float(v) if np.isfinite(v) else None for v in df],
                zero_variance=(variance == 0).tolist(), draws_sha256=sha(draw_path))
