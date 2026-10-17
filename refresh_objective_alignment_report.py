"""Protocol, independent read-only audit and human-readable scientific report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import refresh_pooling_utils as pool
from temporal_refresh_common import ROOT, REPS, read, save, sha, relative, now
import refresh_objective_alignment_utils as objective


def protocol(commit: str, args) -> str:
    return f'''# Frozen analysis protocol on reused data: refresh_objective_alignment_v1

Written {now()}, before calculating new real-data outcome tables. Starting commit
`{commit}`; reviewed PR16 `{args.reviewed_commit}`. The starting tracked tree equals
the reviewed merge tree; no reset, historical rewrite or head fitting. Exact source,
head, input and cohort hashes are in input_provenance.json. This is not independent
preregistration; historical development choices and exposed outcomes are acknowledged.

## Fixed question, representations and endpoints
Do aggregation and frozen classifier geometry improve full-information label-free
selection? Primary is fresh-confirmation X3D-XS B4 cache-pooling
g=CE_C(Z_pool)-CE_C(LegacyCosine), equal-source mean in nats. Negative favors repair.
d=CE_C(LegacyCosine)-CE_C(periodic); p=CE_C(Z_pool)-CE_C(periodic);
a=CE_C(Z_pool)-CE_C(dense16). Improvement in g alone need not improve periodic.
No outcome changes these definitions or chooses a different objective/primary budget.
All three representations use their exact v1 frozen heads, scales, class order,
float32 arrays and 16 trailing snippets [u-47/30,u-32/30,u-16/30,u], floor PTS
resolution +1e-9 seconds, four distinct frames, endpoints unchanged. No refitting,
new normalization/calibration, controller, new architecture or head selection.
Pilot cohorts: exactly100 reused/tuned development and100 exposed v1 evaluation,
reported separately. Neither is confirmation for this new hypothesis. No protected
confirmation outcomes, features or motion tables are opened. Training arrays are unnecessary.

## Column-vector definitions, implemented as row arrays
x_t=(f_t-mu)/scale; z_t=W x_t+b; W is K by D, K=20;
Q=I-11^T/K. e_t=x_r(t)-x_t; a_t=Q W e_t.
F_local=mean_t(||e_t||^2)/D; F_pool=||mean_t(e_t)||^2/D.
Z_local=mean_t(||a_t||^2)/K; Z_pool=||mean_t(a_t)||^2/K.
Use float64; all455 schedules include0, have4 distinct ordered positions, lex order.
LegacyCosine remains original raw-feature local cosine (reject norm<=1e-12;
clamp cosine roundoff), a bridge baseline outside the squared-error factorial.
Minimization chooses lex-first within1e-12 of the true minimum; retain both the
minimum and achieved value, index, positions, ties. Pool affine logits/features,
never probabilities. C weights are candidate reuse counts/16, not elapsed seconds;
U weights selected references equally. Cross-evaluate C-selected objectives in U,
explicitly mismatched to their optimization target. Dense16, periodic range(0,16,16//B),
RandomExact and true-label task_C/task_U included. RandomExact averages individual
schedule losses/correctness, not logits or an ensemble. X3D B2/B8 (15/6435 schedules)
are predetermined secondary sensitivities in each cohort. Other reps only B4.
All selectors require dense features; task oracles additionally require labels.

## Mechanism analyses fixed before outcomes
aggregation=((CE_F_pool-CE_F_local)+(CE_Z_pool-CE_Z_local))/2;
metric=((CE_Z_local-CE_F_local)+(CE_Z_pool-CE_F_pool))/2;
interaction=(CE_Z_pool-CE_Z_local)-(CE_F_pool-CE_F_local).
These are secondary selector changes, not uniquely causal percentages.
Verify mean||e||^2=||mean e||^2+mean||e-mean e||^2, likewise a.
Common logit shifts do not affect probabilities. The CE gradient is softmax(z)-e_y,
whose norm is at most sqrt(2). The fundamental theorem of calculus along the
centered logit segment gives |CE(z_S,y)-CE(z_dense,y)|<=sqrt(2)||Q(z_S-z_dense)||.
This standard explanatory bound is not novel and minimizing its discrepancy does
not minimize true-label CE. Dense predictions may be wrong or worse than periodic.
Report actual CE,top1/macro accuracy,dense agreement,KL(dense probabilities||selected
probabilities),helpful/harmful CE changes. Per-class,dense-correct/wrong are descriptive.
Per-source schedule-wise Spearman(proxy,true-label CE), then summarize sources;
constant proxy/outcome correlation is null with reason. No schedule independence claim.

## Precision, input-only feasibility, frozen confirmation
Pilot results are for fixed analysis/debugging/variance, never formula/model selection.
h=.015 nats for g and d; V=1.25 times the maximum of average20 within-class sample
variances, over both contrasts and both pilot cohorts. N_requested=20*ceil(max(100,
1.96^2*V/h^2)/20). Report uncapped N and precision at ceiling400.
Rebuild all historical/reserved parent and encoded-hash exclusions from membership
documents, include100 exposed v1 evaluation and official K400 training parents.
Stable ascending SHA256('refresh_objective_alignment_v1|20261025|source_id') order.
Technical only: verified class/access,H264,5..15sec,actual>=16 frames,finite increasing
PTS,16 complete distinct trailing snippets. Never use recognition during screening.
No video transcoding or source replacement for network/CUDA/policy outcomes.
Traverse verified cached archives then uncached URLs lexicographically, inspect all
remaining annotated candidates within20GiB new-response-body cap incl retries/deps.
Cache reuse does not consume new cap; conservative uncertain transfer charges recorded.
Default no dependency changes. Metadata/access does not prove individual video rights;
no videos redistributed. Original parents/adjacent windows and byte duplicates excluded;
unknown reuploads and universal pretraining independence unresolved.
m=smallest eligible count across20classes within resource limit;
n_per_class=min(N_requested/20,20,m). Require>=5/class, else confirmation blocked;
finish exploratory mechanism analysis without repurposing exposed data.
Freeze N,membership,formula/code/head/checkpoint/preprocessing/grid/statistics before
ANY confirmation inference. No interim outcomes,optional stopping or extension.
If below requested, prominently report precision-limited, no guarantee of conclusiveness.
Collect real embeddings all3; use same source/snippet indices. No invented embeddings.

## Statistics, numerical checks and execution accounting
10,000 class-stratified whole-source paired resamples, PCG64 seed20261025, linear
2.5/97.5percentiles. Save shared index matrix across objectives/representations/budgets
within cohort, separate cohorts. Only fresh X3D g B4 C is primary; secondaries marginal
unadjusted. Source, not schedule/frame, is unit. Condition fixed heads/classes/source
pool/historical choices, exclude training and repeated-research uncertainty.
g CI below0 supports repair vsLegacy; above0 rejects proposed improvement; crossing0
unresolved, not equivalence. g improvement without p improvement does not establish
useful selection over periodic. Dense fidelity without better CE is insufficient.
No post-hoc equivalence margin or meaningful-effect threshold.
ATOL=RTOL=1e-10 for independent numerical reproduction; sign/tie tolerance1e-12;
float64 calculations,CSV17g round-trip parsing,JSON no NaN/Infinity, null+reason.
Before extending reproduce v1 X3D B4 oldseed20261024 and old per-source logits/losses,
correctness/schedules. Independently replay all selected schedules and sample5 full
schedules per real source/representation/budget; test objective loops,Jensen,common
shift,affine equivalence,ties and seal enforcement on explicitly synthetic unit fixtures.
Audit aggregates/bootstrap from saved source outputs without main evaluator.
All fitting calls0. Pilot encoders0/decodes0; new technical decodes and real forward
starts/completions separately counted. New selectors require16 dense snippets per
representation plus CPU search; not four-call deployed systems. No new latency
experiment or old-periodic latency assignment. Existing six-system v1 timing has
its own encoded-memory boundary and is only cited with that scope.

## Outputs, completion and commands
Retain protocol,identity/seal/source exclusions/eligibility/precision logs,raw per-source
metrics,objective distributions,selected positions/weights/logits,paired intervals,
rank/factorial/cancellation diagnostics,accounting,verification,figures and findings.
Real large arrays live in ignored data/source/refresh_objective_alignment_v1 with
hashes; no synthetic research results. Audit-only is read-only and completed calls
audit rather than overwrite. Completion requires independent checks,release validator,
and rendered-figure review. If blocked, preserve exact stage and resume command.

```powershell
& .\\.venv_temporal\\Scripts\\python.exe -m unittest test_refresh_objective_alignment -v
& .\\.venv_temporal\\Scripts\\python.exe .\\analyze_refresh_objective_alignment.py --preflight-only
& .\\.venv_temporal\\Scripts\\python.exe .\\analyze_refresh_objective_alignment.py --input results/temporal_refresh_generalization_v1 --output results/refresh_objective_alignment_v1 --seed 20261025 --bootstrap 10000 --device cuda --max-download-gib 20
# Interrupted run: same exact arguments plus --resume
& .\\.venv_temporal\\Scripts\\python.exe .\\analyze_refresh_objective_alignment.py --resume
& .\\.venv_temporal\\Scripts\\python.exe .\\analyze_refresh_objective_alignment.py --audit-only
```
'''


def independent_audit(output: Path, old: Path, args) -> dict:
    """No model forward, fitting, download or result writing in this path."""
    start = time.perf_counter()
    from analyze_refresh_objective_alignment import heads, references, ledger, cohorts, code_identity
    provenance = read(output/'input_provenance.json')
    objective.verify_seal(ROOT, {'sha256': provenance['inputs']})
    objective.verify_seal(ROOT, {'sha256': provenance['prior_scientific_sha256']})
    assert read(output/'run_configuration.json')['code_sha256'] == code_identity()
    classes, groups = cohorts(old)
    hs = heads(old, classes)
    seal = output/'evaluation_seal.json'
    if seal.exists():
        objective.verify_seal(ROOT, read(seal))
        fresh = read(output/'confirmation_manifest.json')['videos']
        excluded = read(output/'source_exclusions.json')
        ids, hashes = [v['source_video_id'] for v in fresh], [v['file_identity']['sha256'] for v in fresh]
        assert len(ids) == len(set(ids)) == len(set(hashes))
        assert not set(ids) & set(excluded['source_ids'])
        assert not set(hashes) & set(excluded['encoded_sha256'])
        import csv
        training_ids = {row['youtube_id'] for row in csv.DictReader((ROOT/'data/source/temporal_refresh_generalization_v1/k400_train.csv').open(encoding='utf-8'))}
        assert not set(ids) & training_ids
        assert len(set(__import__('collections').Counter(v['label'] for v in fresh).values())) == 1
        groups['confirmation'] = fresh
    tables = {b: pool.schedules(16, b) for b in [2, 4, 8]}
    assert [len(tables[b]['positions']) for b in [2, 4, 8]] == [15, 455, 6435]
    endpoints = checks = 0
    maximum = 0.
    for cohort, entries in groups.items():
        directory = output/cohort
        if not (directory/'per_video_metrics.csv').exists():
            continue
        raw = pool.frame(directory/'per_video_metrics.csv')
        selected = read(directory/'selected_schedules.json')
        artifacts = read(directory/'schedule_artifacts.json')
        objective.verify_seal(ROOT, {'sha256': {a['path']: a['sha256'] for a in artifacts}})
        sources = read(directory/'bootstrap_sources.json')
        assert set(sources) == {v['source_video_id'] for v in entries}
        labels = {v['source_video_id']: v['label'] for v in entries}
        idx = np.load(directory/'bootstrap_indices.npy', allow_pickle=False)
        np.testing.assert_array_equal(idx, pool.stratified_indices(sources, labels, classes, args.bootstrap, args.seed))
        for label in classes:
            members = set(i for i, s in enumerate(sources) if labels[s] == label)
            assert np.all(np.isin(idx, list(members)).sum(axis=1) == len(members))
        records = ledger(old, output, cohort)
        by_source = {v['source_video_id']: v for v in entries}
        grouped = {}
        for row in selected:
            grouped.setdefault(row['source_video_id'], []).append(row)
        for source, chosen in grouped.items():
            v = by_source[source]
            arrays = references(v, records[source], sha(old/'encoder_provenance.json'))
            jobs = {}
            for row in chosen:
                key = row['representation'], row['budget']
                if key not in jobs:
                    path = next(ROOT/a['path'] for a in artifacts if a['path'].endswith(f'/{key[0]}/{source}_B{key[1]}.npz'))
                    with np.load(path, allow_pickle=False) as archive:
                        jobs[key] = {name: archive[name] for name in archive.files}
                j = row['schedule_index']
                np.testing.assert_allclose(row['CE'], jobs[key]['loss_'+row['pool']][j], atol=1e-10, rtol=1e-10)
                csv_row = raw[(raw.source_video_id == source) & (raw.representation == key[0]) &
                    (raw.budget == key[1]) & (raw.selector == row['selector']) & (raw.pool == row['pool'])].iloc[0]
                np.testing.assert_allclose(csv_row.CE, row['CE'], atol=1e-10, rtol=0)
                positions = json.loads(row['positions'])
                feature = np.asarray(arrays[key[0]], dtype=float)
                if row['pool'] == 'C':
                    last = 0
                    explicit = []
                    for t in range(16):
                        if t in positions:
                            last = t
                        explicit.append(feature[last].copy())
                else:
                    explicit = [feature[t].copy() for t in positions]
                h = hs[key[0]]
                mean = np.mean(explicit, axis=0)
                reference_logit = np.asarray(h['weights']) @ ((mean-h['means'])/h['scales'])+h['bias']
                np.testing.assert_allclose(row['logits'], reference_logit, atol=1e-10, rtol=1e-10)
                if row['pool'] != 'C':
                    continue
                rep, b = key
                j = row['schedule_index']
                np.testing.assert_array_equal(tables[b]['positions'][j], json.loads(row['positions']))
                checked = objective.scalar_check(arrays[rep], hs[rep], v['label_index'] if 'label_index' in v else classes.index(v['label']),
                    json.loads(row['positions']), jobs[key], j)
                maximum = max(maximum, checked['maximum_difference'])
                if row['selector'] in ['LegacyCosine', *objective.OBJECTIVES, 'task_C', 'task_U']:
                    name = {'task_C': 'loss_C', 'task_U': 'loss_U'}.get(row['selector'], row['selector'])
                    values = jobs[key][name]
                    best = np.min(values)
                    ties = [n for n, value in enumerate(values) if value <= best+1e-12]
                    assert j == ties[0] and row['tie_count'] == len(ties)
                    np.testing.assert_allclose([row['true_minimum'], row['achieved_objective']], [best, values[j]], atol=1e-10, rtol=1e-10)
                checks += 1
            # Dense independently pooled/projected from real source references.
            for rep in REPS:
                h = hs[rep]
                mean = np.asarray(arrays[rep], dtype=float).mean(axis=0)
                z = np.asarray(h['weights']) @ ((mean-h['means'])/h['scales'])+h['bias']
                y = classes.index(v['label'])
                m = float(z.max())
                ce = m+np.log(sum(np.exp(float(value)-m) for value in z))-z[y]
                row = raw[(raw.source_video_id == source) & (raw.representation == rep) & (raw.budget == 16)].iloc[0]
                np.testing.assert_allclose(ce, row.CE, atol=1e-10, rtol=1e-10)
        # Independent source contrast reconstruction, no main pairs()/summarize().
        paired = pool.frame(directory/'per_source_deltas.csv')
        for (rep, b, source), block in raw[raw.budget != 16].groupby(['representation', 'budget', 'source_video_id']):
            cells = {(row.selector, row.pool): row for row in block.itertuples()}
            ce = lambda s: cells[(s, 'C')].CE
            expected = dict(g=ce('Z_pool')-ce('LegacyCosine'), d=ce('LegacyCosine')-ce('periodic'),
                p=ce('Z_pool')-ce('periodic'),
                aggregation=.5*(ce('F_pool')-ce('F_local')+ce('Z_pool')-ce('Z_local')),
                metric=.5*(ce('Z_local')-ce('F_local')+ce('Z_pool')-ce('F_pool')),
                interaction=ce('Z_pool')-ce('Z_local')-ce('F_pool')+ce('F_local'))
            saved = paired[(paired.representation == rep) & (paired.budget == b) & (paired.source_video_id == source)].set_index('contrast')
            for name, value in expected.items():
                np.testing.assert_allclose(value, saved.loc[name].value, atol=1e-10, rtol=1e-10)
        for row in read(directory/'bootstrap_intervals.json')['rows']:
            subset = paired[(paired.representation == row['representation']) & (paired.budget == row['budget']) & (paired.contrast == row['contrast'])]
            values = subset.set_index('source_video_id').loc[sources].value.to_numpy()
            low, high = np.percentile(np.mean(values[idx], axis=1), [2.5, 97.5], method='linear')
            np.testing.assert_allclose([values.mean(), low, high], [row['mean'], row['CI95_lower'], row['CI95_upper']], atol=1e-10, rtol=0)
            endpoints += 1
        for row in read(directory/'aggregate_metrics.json')['rows']:
            block = raw[(raw.representation == row['representation']) & (raw.budget == row['budget']) & (raw.selector == row['selector']) & (raw.pool == row['pool'])]
            np.testing.assert_allclose([block.CE.mean(), 100*block.correct.mean()], [row['CE']['mean'], row['accuracy_pp']['mean']], atol=1e-10, rtol=0)
        extra = directory/'additional_accuracy_comparisons.csv'
        if extra.exists():
            for row in pool.frame(extra).itertuples():
                block = raw[(raw.representation == row.representation) & (raw.budget == row.budget) & (raw.pool == 'C')]
                wide = block.pivot(index='source_video_id', columns='selector', values='correct').loc[sources]
                values = 100*(wide['Z_pool']-wide[row.baseline]).to_numpy()
                low, high = np.percentile(values[idx].mean(axis=1), [2.5, 97.5], method='linear')
                np.testing.assert_allclose([values.mean(), low, high], [row.mean, row.CI95_lower, row.CI95_upper], atol=1e-10, rtol=0)
                endpoints += 1
    if (output/'SHA256SUMS.txt').exists():
        for line in (output/'SHA256SUMS.txt').read_text().splitlines():
            digest, name = line.split('  ', 1)
            assert sha(output/name) == digest, name
    return dict(status='passed', checked_at_utc=now(), independent_selected_checks=checks,
                maximum_independent_difference=maximum, bootstrap_intervals_independently_reproduced=endpoints,
                archived_v1_primary_reproduced=read(output/'legacy_reproduction.json'),
                old_scientific_hashes_unchanged=True, exact_schedule_counts=[15, 455, 6435],
                no_protected_outcomes_or_training_arrays_opened=True, head_or_scaler_fits=0,
                elapsed_audit_seconds=time.perf_counter()-start,
                audit_does_not_write_scientific_outputs=True)


def _description(row: dict) -> str:
    return f"{row['mean']:+.6f} [{row['CI95_lower']:+.6f}, {row['CI95_upper']:+.6f}]"


def report(output: Path, old: Path, args) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cohorts = [name for name in ['pilot_development', 'pilot_v1_evaluation_exposed', 'confirmation'] if (output/name/'per_video_metrics.csv').exists()]
    names = {'pilot_development': 'Pilot: reused development', 'pilot_v1_evaluation_exposed': 'Pilot: exposed v1 evaluation', 'confirmation': 'Fresh confirmation'}
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    allrows = {c: read(output/c/'bootstrap_intervals.json')['rows'] for c in cohorts}
    get = lambda c, name, b=4: next(r for r in allrows[c] if r['representation'] == 'x3d_xs' and r['budget'] == b and r['contrast'] == name)
    raw = {c: pool.frame(output/c/'per_video_metrics.csv') for c in cohorts}
    feasibility = read(output/'precision_and_feasibility.json')

    # Additional predeclared descriptive metrics for the exact random baseline.
    # These derive from each schedule, never from ensemble logits. Initial pilot
    # CE/correctness tables remain byte-unchanged.
    from analyze_refresh_objective_alignment import cohorts as get_cohorts, heads, references, ledger
    classes, entries_by_cohort = get_cohorts(old)
    hs = heads(old, classes)
    if 'confirmation' in cohorts:
        entries_by_cohort['confirmation'] = read(output/'confirmation_manifest.json')['videos']
    for c in cohorts:
        target = output/c/'RandomExact_dense_fidelity.csv'
        if target.exists():
            continue
        random_rows = []
        records = ledger(old, output, c)
        for entry in entries_by_cohort[c]:
            source = entry['source_video_id']
            arrays = references(entry, records[source], sha(old/'encoder_provenance.json'))
            for rep, features in arrays.items():
                z = pool.logits(features, hs[rep]); dense_z = z.mean(axis=0)
                ld = objective.log_probability(dense_z)
                for b in ([2, 4, 8] if rep == 'x3d_xs' else [4]):
                    table = pool.schedules(16, b)
                    for pooling in ['C', 'U']:
                        schedule_z = table['cache_weights' if pooling == 'C' else 'unique_weights'] @ z
                        lp = objective.log_probability(schedule_z)
                        random_rows.append(dict(cohort=c, representation=rep, source_video_id=source, budget=b, pool=pooling,
                            expected_dense_prediction_agreement=float((schedule_z.argmax(1) == dense_z.argmax()).mean()),
                            expected_KL_dense_to_selected=float(np.sum(np.exp(ld)*(ld-lp), axis=1).mean()),
                            scope='mean of individual schedule quantities; not ensemble'))
        pool.csv(target, pd.DataFrame(random_rows))
    for c in cohorts:
        target = output/c/'additional_accuracy_comparisons.csv'
        if target.exists():
            continue
        sources = read(output/c/'bootstrap_sources.json')
        idx = np.load(output/c/'bootstrap_indices.npy', allow_pickle=False)
        results = []
        for (rep, b), block in raw[c][(raw[c].budget != 16) & (raw[c].pool == 'C')].groupby(['representation', 'budget']):
            wide = block.pivot(index='source_video_id', columns='selector', values='correct').loc[sources]
            for baseline in ['LegacyCosine', 'periodic']:
                values = 100*(wide.Z_pool-wide[baseline]).to_numpy()
                results.append(dict(cohort=c, representation=rep, budget=int(b), baseline=baseline,
                    orientation='accuracy(Z_pool)-accuracy(baseline), positive favors Z_pool, percentage points',
                    **pool.describe(values, idx), improved=int((values > 0).sum()), worsened=int((values < 0).sum()), tied=int((values == 0).sum())))
        pool.csv(target, pd.DataFrame(results))

    def export(fig, name):
        fig.savefig(figures/f'{name}.png', dpi=160, bbox_inches='tight')
        fig.savefig(figures/f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, len(cohorts), figsize=(5.2*len(cohorts), 4), squeeze=False)
    for ax, c in zip(axes[0], cohorts):
        for y, name in enumerate(['g', 'd', 'p']):
            r = get(c, name)
            ax.errorbar(r['mean'], y, xerr=[[r['mean']-r['CI95_lower']], [r['CI95_upper']-r['mean']]], fmt='o', capsize=4)
        ax.axvline(0, color='0.4', lw=1)
        ax.set(yticks=range(3), yticklabels=['g: Zpool − legacy', 'd: legacy − periodic', 'p: Zpool − periodic'],
               xlabel='Paired CE difference (nats/video); negative favors first', title=names[c])
    fig.suptitle('X3D-XS B4 C pooling: full-information diagnostics, 95% source intervals')
    fig.tight_layout()
    export(fig, 'primary_g_d_p')
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.8*len(cohorts), 4), squeeze=False)
    for ax, c in zip(axes[0], cohorts):
        group = raw[c][(raw[c].representation == 'x3d_xs') & (raw[c].budget == 4) & (raw[c].pool == 'C')]
        values = np.array([[group[group.selector == key].CE.mean() for key in ['F_local', 'F_pool']],
                           [group[group.selector == key].CE.mean() for key in ['Z_local', 'Z_pool']]])
        image = ax.imshow(values, cmap='viridis')
        for i in range(2):
            for j in range(2):
                color = 'black' if values[i,j] > (values.max()+values.min())/2 else 'white'
                ax.text(j, i, f'{values[i,j]:.4f}', ha='center', va='center', color=color, fontsize=12)
        ax.set(xticks=[0, 1], xticklabels=['Local', 'Pooled'], yticks=[0, 1], yticklabels=['Standardized feature', 'Centered logit'], title=names[c])
        fig.colorbar(image, ax=ax, label='Actual true-label CE (nats/video)')
    fig.suptitle('2×2 selector objectives: achieved recognition loss, X3D B4 C')
    fig.tight_layout()
    export(fig, 'factorial_objectives')
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5*len(cohorts), 4), squeeze=False)
    for ax, c in zip(axes[0], cohorts):
        group = raw[c][(raw[c].representation == 'x3d_xs') & (raw[c].budget == 4) & (raw[c].pool == 'C')]
        for selector in ['LegacyCosine', 'Z_pool']:
            g = group[group.selector == selector]
            ax.scatter(g.centered_logit_discrepancy_norm, g.delta_CE_dense, alpha=.6, s=14, label=selector)
        ax.axhline(0, color='.5', lw=1)
        ax.set(xlabel='Centered pooled-logit discrepancy to dense (L2)', ylabel='CE(selected) − CE(dense), nats/video', title=names[c])
        ax.legend()
    fig.suptitle('Dense fidelity is not true-label recognition quality')
    fig.tight_layout()
    export(fig, 'dense_fidelity_recognition')
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5*len(cohorts), 4), squeeze=False)
    for ax, c in zip(axes[0], cohorts):
        d = pool.frame(output/c/'per_source_deltas.csv')
        values = d[(d.representation == 'x3d_xs') & (d.budget == 4) & (d.contrast == 'g')].sort_values('value').value.to_numpy()
        ax.bar(np.arange(len(values)), values, color=np.where(values < 0, '#247a75', '#b74b4b'))
        ax.axhline(0, color='.4', lw=1)
        ax.set(xlabel='Sources sorted by g (descriptive order)', ylabel='g = CE(Zpool) − CE(legacy), nats/video', title=names[c])
    fig.suptitle('X3D B4 paired source effects; no favorable-source exclusions')
    fig.tight_layout()
    export(fig, 'per_source_effects')

    table = ['|Cohort|N|g: Zpool−legacy|d: legacy−periodic|p: Zpool−periodic|', '|---|---:|---:|---:|---:|']
    for c in cohorts:
        table.append(f"|{names[c]}|{get(c,'g')['sources']}|{_description(get(c,'g'))}|{_description(get(c,'d'))}|{_description(get(c,'p'))}|")
    if 'confirmation' in cohorts:
        g, p = get('confirmation', 'g'), get('confirmation', 'p')
        verdict = ('improves relative to LegacyCosine' if g['CI95_upper'] < 0 else
                   'is worse than LegacyCosine' if g['CI95_lower'] > 0 else 'is unresolved at the achieved precision')
        primary = f"Fresh primary X3D B4 C: **g={_description(g)} nats/video**, N={g['sources']}. The proposed objective {verdict}. "
        primary += ('Periodic is also improved in CE.' if p['CI95_upper'] < 0 else 'Improvement over periodic is not established by its paired interval.')
    else:
        primary = '**Confirmation blocked: no fresh primary estimate.** The completed pilot/mechanism analysis is exploratory/partial, not completed confirmation.'
    status_note = '**Precision-limited confirmation**: available N is below the requested size.' if feasibility['precision_limited'] else 'Requested precision-based sample size was available; precision is still not guaranteed.'
    lines = ['# refresh_objective_alignment_v1 findings', '', primary, '', status_note, '', '\n'.join(table), '',
        'Intervals are 10,000 class-stratified paired whole-source resamples, PCG64 seed20261025, linear percentiles. Negative g favors Zpool; negative p favors Zpool over periodic. Only fresh X3D B4 C g is primary. Pilot cohorts are exposed, separate, and never pooled with confirmation. All other intervals are secondary marginal/unadjusted.', '',
        '## Frozen inputs and precision', '',
        f"Starting commit `{read(output/'input_provenance.json')['starting_commit']}` matches reviewed PR16 contents. All three heads/scalers/class orders/checkpoints/grids are unchanged. The archived X3D primary and selected logits/losses/schedules were reproduced before the extension. Actual float32 features are used; evaluation arithmetic is float64. Training arrays and protected confirmation outcomes/references/motion are not read.", '',
        f"h=.015; V={feasibility['V']:.8g}; uncapped requested N={feasibility['N_requested']}, predicted half-width at400={feasibility['predicted_halfwidth_at_400']:.6f}. Eligible pool={feasibility['eligible_total']}, minimum/class={feasibility['smallest_eligible_class_count']}; frozen N={feasibility['N_frozen']}. Predicted achieved half-width={feasibility['predicted_halfwidth_achieved']}. This approximation assumes pilot variance transfers and does not guarantee a conclusive interval. Per-class inventories, rejections and deterministic ranks are saved.", '',
        '## Controlled mechanism, not a unique causal attribution', '',
        'The four squared-error objectives change aggregation and metric separately. LegacyCosine remains a separate raw-feature bridge; differences to it also change standardization and squared versus cosine distance. Local error equals pooled error plus residual variance, identifying cancellation in the pooling calculation. Factorial contrasts concern selector changes, not percentages of semantic importance.', '',
        '|Cohort|Aggregation contrast|Metric contrast|Interaction|', '|---|---:|---:|---:|']
    for c in cohorts:
        lines.append(f"|{names[c]}|{_description(get(c,'aggregation'))}|{_description(get(c,'metric'))}|{_description(get(c,'interaction'))}|")
    lines += ['', 'Zpool reduces a centered-logit discrepancy associated with a generic CE perturbation bound. It does not optimize the unknown true label. Dense predictions can be wrong: preserving them does not guarantee lower CE or accuracy. Saved correctness strata, KL divergences, helpful/harmful counts, per-class tables and per-source schedule-wise rank correlations make this distinction auditable. Read those as descriptive secondary results, never independent schedules or source-selection criteria.', '',
        '## Pooling and budgets', '', 'All C-selected schedules are cross-evaluated under U without reselection; that is a deliberately mismatched sensitivity. Separately optimized true-label C/U oracles use labels and all dense references. Their positive headroom is guaranteed by including periodic, not evidence of a usable scheduler. X3D B2/B8 sensitivities are fixed in advance; other representations remain B4 descriptive controls.', '',
        '|Cohort|B|g|p|', '|---|---:|---:|---:|']
    for c in cohorts:
        for b in [2, 8]:
            lines.append(f"|{names[c]}|{b}|{_description(get(c,'g',b))}|{_description(get(c,'p',b))}|")
    lines += ['', '## Actual recognition and dense fidelity, X3D B4 C', '',
        '|Cohort|Selector|CE (nats)|Top1 / macro accuracy|Dense prediction agreement|KL(dense to selected)|',
        '|---|---|---:|---:|---:|---:|']
    for c in cohorts:
        for selector in ['dense16', 'periodic', 'LegacyCosine', 'F_local', 'F_pool', 'Z_local', 'Z_pool', 'task_C']:
            group = raw[c][(raw[c].representation == 'x3d_xs') & (raw[c].selector == selector) &
                ((raw[c].budget == 16) if selector == 'dense16' else ((raw[c].budget == 4) & (raw[c].pool == 'C')))]
            lines.append(f"|{names[c]}|{selector}{' (true-label hindsight)' if selector == 'task_C' else ''}|{group.CE.mean():.6f}|{100*group.correct.mean():.1f}% / {100*group.groupby('label').correct.mean().mean():.1f}%|{100*group.dense_prediction_agreement.mean():.1f}%|{group.KL_dense_to_selected.mean():.6f}|")
    lines += ['', 'Class balance makes macro and overall accuracy coincide. RandomExact expectations, all other representation results, dense-correct/wrong strata and per-class descriptions are retained separately. The new pilot d bootstrap intervals use seed20261025; the archived seed20261024 result was reproduced exactly in legacy_reproduction.json, so a slight endpoint change here is not a changed v1 measurement.', '',
        '|Cohort|g improved / tied / worsened|p improved / tied / worsened|', '|---|---|---|']
    for c in cohorts:
        g, p = get(c, 'g'), get(c, 'p')
        lines.append(f"|{names[c]}|{g['negative']} / {g['tied']} / {g['positive']}|{p['negative']} / {p['tied']} / {p['positive']}|")
    lines += ['', '## Schedule-wise diagnostic alignment', '',
        '|Cohort|Proxy|Mean defined per-video Spearman with CE_C|Defined / missing|', '|---|---|---:|---|']
    for c in cohorts:
        for r in read(output/c/'rank_correlation_summary.json'):
            if r['representation'] == 'x3d_xs' and r['budget'] == 4 and r['pool'] == 'C':
                lines.append(f"|{names[c]}|{r['proxy']}|{r['mean']}|{r['defined']} / {r['undefined']}|")
    lines += ['', '## Cost, access and limitations', '',
        'Every selector sees all16 dense candidate embeddings. Collecting those requires80 RN18 frame forwards (16 endpoint +64 four-frame-mean) and16 X3D snippet forwards per source, plus CPU search. Actual initiated and completed collection calls, retries, technical decodes, downloads and offline analysis runtime are separate accounting fields. There are zero new head/scaler/controller fits and no package upgrades. No four-call deployment cost or adaptive speedup is claimed.', '',
        'The unchanged v1 timing measured encoded bytes in memory through fresh full decoding, frame preparation, transfers, encoder, pooling/head and final prediction, with its own20 sources and 2warmups/5trials. Those measurements are not timings of these dense-reference selectors. No end-to-end timing was repeated or assigned to an oracle. Full-clip candidate construction is offline. Parent/hash exclusions do not prove absence of reuploads or universal checkpoint pretraining overlap. Confidence intervals omit training, model-selection and repeated-research uncertainty. The frozen dense-trained head may react differently to sparse aggregation. This diagnostic establishes no deployed controller, streaming causality, adaptive latency, recognition guarantee or journal acceptance.', '',
        '## Prior work and claim/evidence', '',
        '[Knowledge distillation](https://arxiv.org/abs/1503.02531) motivates distinguishing output geometry from raw features. [Task-Oriented Feature Distillation](https://papers.neurips.cc/paper_files/paper/2020/hash/a96b65a721e561e1e3de768ac819ffbb-Abstract.html) studies task-driven feature transformations; we neither train such a transformation nor reproduce its method. [AdaFrame](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html) learns frame-selection decisions, unlike these full-information enumerations. These precedents prevent a novelty claim based solely on this design. The softmax-gradient bound is standard calculus.', '',
        '|Claim|Evidence / limit|', '|---|---|',
        '|Inputs and legacy result preserved|Hashes, head selection identities, independent replay and legacy_reproduction.json|',
        '|Objective repair improves legacy CE|Only the fresh primary g interval can support this fixed comparison; exposed pilots are exploratory|',
        '|Useful selection beyond periodic|Requires p, not merely g or lower dense discrepancy; still full-information|',
        '|Mechanistic cancellation|Exact algebra, not a uniquely identified causal explanation|',
        '|Practical inference benefit|Not evaluated: dense references and search are required|', '',
        '## Next decision', '',
        ('If the fixed fresh g and p evidence is favorable, plan one second-dataset replication with the same objectives and frozen selection rules. Otherwise resolve the observed objective-versus-dense/label mismatch and precision/data limitations before another controller. A deployable policy remains a separate study.' if 'confirmation' in cohorts else
         f"Confirmation cannot be recovered by relabeling exposed pilots. Shortfall to five/class: {feasibility['minimum_five_per_class_shortfall']}. Next obtain a separately licensed/provenanced unseen population with these classes, freeze its membership and reuse these fixed objectives; do not silently use official-training parents or change dataset versions."), '',
        '## Reproduction', '', 'From the repository root, use the unchanged existing `.venv_temporal` environment. See README.md for run/resume/audit commands. Large real arrays are ignored, with paths and hashes in input/collection/schedule ledgers; reconstruct only from verified original encodings using the frozen recipe. No random embedding fallback exists.']
    (output/'findings.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (output/'README.md').write_text(r'''# refresh_objective_alignment_v1

Read findings.md first. Pilot cohorts and confirmation are separate directories.
`protocol.md` was written before new comparisons; `evaluation_seal.json` precedes
confirmation inference if feasible. Large real arrays stay in ignored data/source,
with identities in the ledgers. Model fitting and package upgrades are forbidden.

```powershell
& .\.venv_temporal\Scripts\python.exe -m unittest test_refresh_objective_alignment -v
& .\.venv_temporal\Scripts\python.exe .\analyze_refresh_objective_alignment.py --input results/temporal_refresh_generalization_v1 --output results/refresh_objective_alignment_v1 --seed 20261025 --bootstrap 10000 --device cuda --max-download-gib 20
& .\.venv_temporal\Scripts\python.exe .\analyze_refresh_objective_alignment.py --resume
& .\.venv_temporal\Scripts\python.exe .\analyze_refresh_objective_alignment.py --audit-only
```

Saved CSV floats round-trip; JSON undefined values are null, never NaN/Infinity.
CPU analysis/search runtime is not adaptive inference latency. See execution and
download accounting for actual collection work. Completed invocations audit only.
''', encoding='utf-8')
