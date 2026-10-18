"""Fixed dense/proxy evaluation, decomposition, grouped summaries and figures."""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import refresh_pooling_utils as pool
from ucf101_proxy_common import *
from ucf101_proxy_math import *
from ucf101_proxy_inference import read_features


def fit(manifest):
    path = OUT/'head_selection.json'
    if path.exists():
        result = read(path)
        verify_hashes({result['path']: result['sha256']})
        return read(ROOT/result['path'])['parameters']
    classes = manifest['classes']
    entries = {r: [v for v in manifest['videos'] if v['role'] == r] for r in ['train', 'development']}
    arrays = {r: np.stack([read_features(v, r).mean(0, dtype=np.float64) for v in entries[r]]) for r in entries}
    labels = {r: np.array([classes.index(v['label']) for v in entries[r]]) for r in entries}
    grid = []
    for order, alpha in enumerate([.001, .01, .1, 1.]):
        p = OUT/f'heads/grid_{order}.json'
        if p.exists():
            cached = read(p)
            for key, filename in [('training_manifest_sha256', 'eligible_manifest.json'),
                                  ('training_collection_sha256', 'train_collection.json'),
                                  ('development_collection_sha256', 'development_collection.json')]:
                if cached[key] != sha(OUT/filename):
                    raise Blocker('Training grid provenance changed: '+filename)
            record = cached['fit']
        else:
            head, record = fit_head(arrays['train'], labels['train'], entries['train'], classes, alpha)
            record.update(grid_order=order)
            for role in entries:
                z = pool.logits(arrays[role], head)
                ce = np.logaddexp.reduce(z, axis=1)-z[np.arange(len(z)), labels[role]]
                correct = z.argmax(1) == labels[role]
                sw = hierarchical_weights(entries[role], classes)
                record[role+'_CE'] = float(sw @ ce)
                record[role+'_accuracy'] = float(sw @ correct)
            save(p, dict(parameters=head, fit=record, classes=classes,
                         training_manifest_sha256=sha(OUT/'eligible_manifest.json'),
                         training_collection_sha256=sha(OUT/'train_collection.json'),
                         development_collection_sha256=sha(OUT/'development_collection.json')))
        grid.append(record)
        print(f'Head grid alpha={alpha}: dev dense CE={record["development_CE"]:.6f}, iterations={record["iterations"]}', flush=True)
    best = min(grid, key=lambda r: (r['development_CE'], -r['development_accuracy'], r['grid_order']))
    p = OUT/'heads/parameters.json'
    save(p, read(OUT/f'heads/grid_{best["grid_order"]}.json'))
    save(OUT/'training_grid.json', grid)
    save(path, dict(**best, path=rel(p), sha256=sha(p), selection='development DENSE CE, accuracy, grid order; no sparse selection'))
    return read(p)['parameters']


def evaluate(entries, role, classes, head):
    if role == 'test':
        seal_check(True)
    target = OUT/f'{role}_metrics.csv'
    if target.exists():
        return pd.read_csv(target, float_precision='round_trip')
    tables = {b: pool.schedules(16, b) for b in ([4] if role == 'development' else [2, 4, 8])}
    rows, correlations, checks, artifacts = [], [], [], []
    for number, v in enumerate(entries):
        p = DATA/f'analysis/{role}/{v["clip"]}.json'
        if p.exists():
            saved = read(p)
            if saved['feature_sha256'] != read(OUT/f'{role}_collection.json')[v['clip']]['sha256'] or saved['head_sha256'] != sha(OUT/'heads/parameters.json'):
                raise Blocker('Analysis checkpoint provenance changed')
        else:
            f = read_features(v, role)
            # Pin CPU BLAS thread count for separately measured exhaustive search.
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=1)
            y = classes.index(v['label'])
            base = dict(clip=v['clip'], label=v['label'], group=v['group'], role=role, label_index=y)
            dense = pool.logits(f, head).mean(0)
            margin = float(dense[y]-np.max(np.delete(dense, y)))
            local_rows = [dict(**base, budget=16, selector='dense16', pool='C=U',
                              **decomposition(dense, dense, y), dense_margin=margin, logits=dense.tolist())]
            local_correlations, local_checks, local_artifacts = [], [], []
            for budget, table in tables.items():
                values = select_proxies(f, head, table)
                losses = {p: pool.loss(values['logits_'+p], y) for p in ['C', 'U']}
                choices = dict(values['choices'])
                choices.update(task_C=pool.first_tie(losses['C']), task_U=pool.first_tie(losses['U']))
                archive = DATA/f'schedules/{role}/{v["clip"]}_B{budget}.npz'
                archive.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(archive, **values['objectives'], loss_C=losses['C'], loss_U=losses['U'])
                local_artifacts.append(dict(path=rel(archive), sha256=sha(archive)))
                for method, choice in choices.items():
                    j = choice['index']; positions = table['positions'][j].tolist()
                    error = scalar_replay(f, head, y, positions, values, j)
                    local_checks.append(dict(**base, budget=budget, selector=method, maximum_difference=error))
                    for pooling in ['C', 'U']:
                        z = values['logits_'+pooling][j]
                        terms = decomposition(dense, z, y)
                        local_rows.append(dict(**base, budget=budget, selector=method, pool=pooling,
                            **terms, dense_margin=margin, positions=positions, schedule_index=j, logits=z.tolist(),
                            choice=choice, search_seconds=values['search_seconds'],
                            **{name: float(a[j]) for name, a in values['objectives'].items()},
                            F_cancellation=float(values['objectives']['F_local'][j]-values['objectives']['F_pool'][j]),
                            Z_cancellation=float(values['objectives']['Z_local'][j]-values['objectives']['Z_pool'][j]),
                            oracle_regret=float(losses[pooling][j]-losses[pooling].min())))
                for pooling in ['C', 'U']:
                    # RandomExact is the expectation of each individual schedule's metrics.
                    from scipy.special import logsumexp
                    z = values['logits_'+pooling]
                    ld, ls = dense-logsumexp(dense), z-logsumexp(z, axis=1, keepdims=True)
                    delta = z-dense
                    directional = delta @ np.exp(ld)-delta[:, y]
                    kl = (ld-ls) @ np.exp(ld)
                    difference = -ls[:, y]+ld[y]
                    np.testing.assert_allclose(difference, directional+kl, atol=1e-10, rtol=1e-10)
                    averaged = dict(CE=float(losses[pooling].mean()), correct=float((z.argmax(1) == y).mean()),
                        dense_correct=int(dense.argmax() == y), dense_agreement=float((z.argmax(1) == dense.argmax()).mean()),
                        KL=float(kl.mean()), directional=float(directional.mean()), delta_CE_dense=float(difference.mean()),
                        decomposition_residual=float(np.max(np.abs(difference-directional-kl))))
                    local_rows.append(dict(**base, budget=budget, selector='RandomExact', pool=pooling,
                        **averaged, dense_margin=margin, search_seconds=values['search_seconds'],
                        oracle_regret=float(losses[pooling].mean()-losses[pooling].min())))
                    local_correlations.extend(dict(**base, budget=budget, pool=pooling, proxy=name, spearman=value,
                        schedules=len(losses[pooling]), undefined_reason='constant proxy/loss' if value is None else None)
                        for name, value in schedule_correlations(values, losses[pooling]).items())
            saved = dict(rows=local_rows, correlations=local_correlations, checks=local_checks, artifacts=local_artifacts,
                         feature_sha256=read(OUT/f'{role}_collection.json')[v['clip']]['sha256'], head_sha256=sha(OUT/'heads/parameters.json'))
            save(p, saved)
        rows.extend(saved['rows']); correlations.extend(saved['correlations']); checks.extend(saved['checks']); artifacts.extend(saved['artifacts'])
        if (number+1) % 25 == 0:
            print(f'Fixed real-feature {role} analyses {number+1}/{len(entries)}', flush=True)
    frame = pd.DataFrame(rows)
    # JSON vectors have a stable representation; no Python repr in scientific CSV.
    for column in ['positions', 'logits', 'choice']:
        if column in frame:
            frame[column] = frame[column].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else None)
    # Complete vectors remain in genuine per-clip analysis checkpoints with hashes.
    # Avoid one enormous public CSV; the metrics table retains predictions/losses.
    frame = frame.drop(columns=['logits'], errors='ignore')
    save(OUT/f'{role}_prediction_artifacts.json', [dict(path=rel(DATA/f'analysis/{role}/{v["clip"]}.json'),
        sha256=sha(DATA/f'analysis/{role}/{v["clip"]}.json')) for v in entries])
    frame.to_csv(target, index=False, float_format='%.17g')
    frame[frame.budget != 16][['clip', 'label', 'group', 'budget', 'search_seconds']].drop_duplicates().to_csv(
        OUT/f'{role}_search_timings.csv', index=False, float_format='%.17g')
    from threadpoolctl import threadpool_info
    save(OUT/f'{role}_search_scope.json', dict(boundary='already loaded real arrays -> float64 proxy objectives, exhaustive minima, schedule logits; excludes feature I/O and writing',
        CPU_threads=1, threadpools=threadpool_info(), encoding_calls=0,
        caveat='one combined search pass per clip/budget, not repeated latency trials or four-call inference; true-label oracle loss search outside this scoped proxy timer'))
    pd.DataFrame(correlations).to_csv(OUT/f'{role}_schedule_correlations.csv', index=False, float_format='%.17g')
    pd.DataFrame(checks).to_csv(OUT/f'{role}_scalar_checks.csv', index=False, float_format='%.17g')
    save(OUT/f'{role}_schedule_artifacts.json', artifacts)
    return frame


def pairs(raw):
    rows = []
    for (clip, budget, pooling), frame in raw[raw.budget != 16].groupby(['clip', 'budget', 'pool'], sort=True):
        first = frame.iloc[0]
        values = contrast_values(frame)
        rows.append(dict(clip=clip, label=first.label, group=first.group, budget=int(budget), pool=pooling,
                         dense_correct=int(first.dense_correct), dense_margin=float(first.dense_margin), **values))
    return pd.DataFrame(rows)


def precision(dev, manifest):
    paired = pairs(dev)
    groups = paired[(paired.budget == 4) & (paired.pool == 'C')].groupby(['label', 'group'])[['g', 'p']].mean().reset_index()
    estimates = {}
    for q in ['g', 'p']:
        rows = []
        for c in manifest['classes']:
            values = groups[groups.label == c][q]
            variance = float(values.var(ddof=1))
            n = manifest['counts']['test'][c]
            rows.append(dict(label=c, development_groups=len(values), development_variance=variance, test_groups=n))
        var = sum(r['development_variance']/r['test_groups'] for r in rows)/101**2
        halfwidth = float(1.96*np.sqrt(var))
        estimates[q] = dict(variance_of_mean=var, predicted_CI95_halfwidth=halfwidth,
                           predicted_CI95_width=2*halfwidth, by_class=rows,
                           target_halfwidth=.015, target_met=halfwidth <= .015)
    result = dict(contrasts=estimates, fixed_test_size=sum(manifest['counts']['test'].values()),
        rule='Var(mean) approximately sum_c(s_c^2/G_test,c)/101^2; development group means, two clips per group',
        caveats='Only 2-3 development groups/class; head selected on development; noisy variances and uncertain transfer. h=.015 nats is a planning precision target, not a practical-effect/equivalence margin; no 80% power claim. Execute all eligible fixed test groups regardless of sign/precision.')
    save(OUT/'precision_estimates.json', result)
    dense = dev[dev.selector == 'dense16']
    # Signed true-class margin strata are descriptive; thresholds set on development.
    sw = hierarchical_weights(dense.to_dict('records'), manifest['classes'])
    order = np.argsort(dense.dense_margin.to_numpy(), kind='stable')
    cumulative = np.cumsum(sw[order])
    thresholds = [float(dense.dense_margin.to_numpy()[order[min(np.searchsorted(cumulative, q, side='left'), len(order)-1)]]) for q in [1/3, 2/3]]
    save(OUT/'stratum_thresholds.json', dict(dense_margin_terciles=thresholds,
        definition='dense true-class logit minus largest other-class logit; weighted inverse empirical CDF at 1/3,2/3; <=lower, (lower,upper], >upper',
        weight='class/group/clip development weights', fixed_before_test_features=True))
    return result


def summarize(raw, classes, write=True):
    paired = pairs(raw)
    gframe, idx, weights = bootstrap_plan(paired, classes)
    if write:
        np.save(OUT/'bootstrap_group_indices.npy', idx, allow_pickle=False)
        save(OUT/'bootstrap_groups.json', gframe.to_dict('records'))
        np.save(OUT/'bootstrap_column_weights.npy', weights, allow_pickle=False)
        paired.to_csv(OUT/'per_clip_contrasts.csv', index=False, float_format='%.17g')
    contrasts = ['g', 'p', 'd', 'aggregation', 'metric', 'interaction', 'directional_g', 'KL_g']
    intervals, primary, group_records = [], [], []
    index = pd.MultiIndex.from_frame(gframe)
    for (budget, pooling), f in paired.groupby(['budget', 'pool'], sort=True):
        groups = f.groupby(['label', 'group'])[contrasts].mean().reindex(index)
        if groups.isna().any().any():
            raise Blocker('Incomplete groups in paired evaluation')
        for q in contrasts:
            r = dict(budget=int(budget), pool=pooling, contrast=q, **interval(groups[q].to_numpy(), idx, weights),
                     groups=len(groups), clips=len(f), primary=budget == 4 and pooling == 'C' and q == 'g',
                     inference='class-stratified paired group bootstrap; all secondary intervals marginal/unadjusted')
            intervals.append(r)
            if budget == 4 and pooling == 'C' and q in ['g', 'p', 'd']:
                primary.append(r)
        group_records.append(groups.reset_index().assign(budget=budget, pool=pooling))
    metrics = ['CE', 'correct', 'dense_agreement', 'KL', 'directional', 'delta_CE_dense', 'oracle_regret']
    aggregates, class_records = [], []
    for (budget, pooling, method), f in raw.groupby(['budget', 'pool', 'selector'], sort=True):
        columns = [m for m in metrics if m in f and f[m].notna().all()]
        groups = f.groupby(['label', 'group'])[columns].mean().reindex(index)
        by_class = groups.groupby('label').mean()
        for c in classes:
            class_records.append(dict(label=c, budget=budget, pool=pooling, selector=method, **by_class.loc[c].to_dict()))
        r = dict(budget=int(budget), pool=pooling, selector=method, groups=len(groups), clips=len(f), ordinary_clip_accuracy=float(f.correct.mean()))
        for m in columns:
            r[m] = interval(groups[m].to_numpy(), idx, weights)
        aggregates.append(r)
    if write:
        pd.concat(group_records, ignore_index=True).to_csv(OUT/'per_group_contrasts.csv', index=False, float_format='%.17g')
        raw.groupby(['label', 'group', 'budget', 'pool', 'selector'])[[m for m in metrics if m in raw]].mean().reset_index().to_csv(OUT/'per_group_metrics.csv', index=False, float_format='%.17g')
        pd.DataFrame(class_records).to_csv(OUT/'per_class_metrics.csv', index=False, float_format='%.17g')
        pd.DataFrame(intervals).to_csv(OUT/'secondary_intervals.csv', index=False, float_format='%.17g')
        save(OUT/'primary_intervals.json', primary)
        save(OUT/'aggregate_metrics.json', aggregates)
        strata(paired, classes)
    return primary


def strata(paired, classes):
    thresholds = read(OUT/'stratum_thresholds.json')['dense_margin_terciles']
    # Clip-defined strata, with bootstrap resampling ALL groups and carrying their clips.
    f = paired[(paired.budget == 4) & (paired.pool == 'C')].copy()
    gframe, idx, weights = bootstrap_plan(f, classes)
    index = pd.MultiIndex.from_frame(gframe)
    masks = {'dense_correct': f.dense_correct == 1, 'dense_wrong': f.dense_correct == 0,
             'margin_low': f.dense_margin <= thresholds[0],
             'margin_mid': (f.dense_margin > thresholds[0]) & (f.dense_margin <= thresholds[1]),
             'margin_high': f.dense_margin > thresholds[1]}
    rows = []
    for name, mask in masks.items():
        selected = f[mask]
        represented = len(selected[['label', 'group']].drop_duplicates())
        # Conditional weighted estimand via ratio of weighted numerator/stratum mass.
        for q in ['g', 'p', 'directional_g', 'KL_g']:
            values = f.assign(num=np.where(mask, f[q], 0.), mass=mask.astype(float)).groupby(['label', 'group'])[['num', 'mass']].mean().reindex(index)
            num, mass = values.num.to_numpy(), values.mass.to_numpy()
            denominator = mass[idx] @ weights
            draws = (num[idx] @ weights)[denominator > 0]/denominator[denominator > 0]
            bounds = np.percentile(draws, [2.5, 97.5]) if len(draws) else [None, None]
            rows.append(dict(stratum=name, contrast=q, mean=float(num @ weights/(mass @ weights)) if mass @ weights else None,
                CI95_lower=bounds[0], CI95_upper=bounds[1], represented_groups=represented, represented_classes=selected.label.nunique(),
                clips=len(selected), sparse=represented < 20, invalid_bootstrap_replicates=int(np.sum(denominator == 0)),
                inference='secondary weighted conditional clip stratum, whole-group paired resampling; sparse<20 groups: no population claim'))
    pd.DataFrame(rows).to_csv(OUT/'secondary_strata.csv', index=False, float_format='%.17g')


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 160})
    folder = OUT/'figures'; folder.mkdir(exist_ok=True)
    intervals = read(OUT/'primary_intervals.json')
    fig, ax = plt.subplots(figsize=(8.2, 3.8), layout='constrained')
    for i, r in enumerate(intervals):
        ax.errorbar(r['mean'], i, xerr=[[r['mean']-r['CI95_lower']], [r['CI95_upper']-r['mean']]], fmt='o', capsize=5)
    ax.set_yticks(range(3), [r['contrast'] for r in intervals]); ax.axvline(0, color='grey', lw=1)
    ax.set_xlabel('Paired CE difference (nats), negative favors first selector')
    ax.set_title('UCF101 split 1: g = Zpool−Legacy; p = Zpool−periodic; d = Legacy−periodic\n95% paired group intervals; g primary, p/d secondary and unadjusted', fontsize=11)
    fig.savefig(folder/'primary_intervals.png'); fig.savefig(folder/'primary_intervals.pdf'); plt.close(fig)
    raw = pd.read_csv(OUT/'test_metrics.csv', float_precision='round_trip')
    f = raw[(raw.budget == 4) & (raw.pool == 'C') & raw.selector.isin(['periodic', 'LegacyCosine', *OBJECTIVES])]
    means = f.groupby(['label', 'group', 'selector'])[['KL', 'CE']].mean().groupby(['label', 'selector']).mean().groupby('selector').mean()
    fig, ax = plt.subplots(figsize=(8.2, 4.5), layout='constrained')
    for (method, r), marker in zip(means.iterrows(), ['o', 's', '^', 'D', 'X', 'P']):
        ax.scatter(r.KL, r.CE, marker=marker, s=65, label=method, alpha=.85)
    ax.legend(loc='center left', bbox_to_anchor=(1, .5), frameon=False)
    ax.set(xlabel='KL(dense || selected), nats', ylabel='Group/class-balanced CE, nats', title='Dense prediction fidelity versus recognition\nDescriptive means; proxy selectors require 16 dense encodings + search')
    fig.savefig(folder/'fidelity_recognition.png'); fig.savefig(folder/'fidelity_recognition.pdf'); plt.close(fig)
    secondary = pd.read_csv(OUT/'secondary_intervals.csv', float_precision='round_trip')
    f = secondary[(secondary.budget == 4) & (secondary.pool == 'C')].set_index('contrast')
    fig, ax = plt.subplots(figsize=(7, 3.5), layout='constrained')
    names = ['directional_g', 'KL_g', 'g']
    for i, name in enumerate(names):
        r = f.loc[name]
        ax.errorbar(r['mean'], i, xerr=[[r['mean']-r.CI95_lower], [r.CI95_upper-r['mean']]], fmt='o', capsize=5)
    ax.axvline(0, color='grey', lw=1); ax.set_yticks(range(3), ['Signed directional difference', 'KL difference', 'Total g'])
    ax.set(xlabel='CE contribution difference, nats', title='Exact descriptive decomposition: g = directional difference + KL difference\nIntervals marginal/unadjusted; no causal attribution')
    fig.savefig(folder/'decomposition.png'); fig.savefig(folder/'decomposition.pdf'); plt.close(fig)
    if (OUT/'timing_per_clip.csv').exists():
        f = pd.read_csv(OUT/'timing_per_clip.csv').pivot(index='clip', columns='mode', values='median')
        fig, ax = plt.subplots(figsize=(6.4, 4.2), layout='constrained')
        for _, r in f.iterrows():
            ax.plot([0, 1], [1000*r.dense16, 1000*r.periodicB4], color='#287e8a', alpha=.35, marker='o', ms=3)
        ax.set_xticks([0, 1], ['Dense16', 'Periodic B4']); ax.set_ylabel('Per-clip median latency, ms')
        ax.set_title('Measured encoded-memory → completed prediction\n30 groups / 30 classes; five trials per mode, not 150 sources')
        fig.savefig(folder/'measured_cost.png'); fig.savefig(folder/'measured_cost.pdf'); plt.close(fig)
