# Completed DEVELOPMENT / EXPLORATORY evidence review

The engineering objective was achieved: the fixed H/H+M controllers were trained from genuine cached-feature observations and executed chronologically on real development AVI, with four actual X3D forwards and sixteen cached outputs per B4 trial. The proposed recognition benefit was **not established**. No policy, threshold, head or core evaluation implementation was changed after development execution began.

|Actual method|Balanced CE|Balanced top1|Median per-video measured latency|
|---|---:|---:|---:|
|Periodic B4|0.247427|94.31%|405.6 ms|
|H B4|0.257951|93.07%|409.2 ms|
|H+M B4|0.255400|92.82%|570.0 ms|
|Dense16|0.244324|93.56%|1385.8 ms|

The fixed Delta_CE=CE(periodic)-CE(H+M) is **-0.007972 nats**, with conditional descriptive95% paired group interval **[-0.017681,+0.002085]**. Its sign favors periodic, and the interval crosses zero. This is inconclusive about improvement, not equivalence. H+M-versus-H is +0.002551 nats in the favorable direction, with interval[-0.007809,+0.012923]; that also does not establish benefit. Other intervals are unadjusted secondary summaries. The privileged offline Z_pool-versus-periodic difference is itself small/inconclusive on this development cohort; do not assume a large deployable advantage was present here.

## What actually ran

Training used404 real clips/202 groups across all101 classes; selection used202 clips/101 separate groups; real execution used400 clips/200 remaining development groups. Both learners used19124 base chronological states and28499 states after one shared on-policy augmentation round, the same145→32→1 architecture and optimization budget, and the prespecified checkpoint/threshold grid. These states are computational examples, not independent videos. Head coefficients/scalers were fitted on original training data, but alpha was selected on ALL original development data. Controller training overlaps head training. Both prior exposure and reuse prevent independent confirmation.

Observed new X3D starts/completions are22840/22840:40 training-smoke,8000 causal-quality,6400 fresh Dense16 diagnostics and8400 timing. There were no failed attempts or unresolved tokens. Timing contains1470 actual trials,1050 measured after420 warmups, on30 input-selected groups/classes. Repeats are not independent sources. All actual starts name approved official-training clips. Official-test outcomes/videos/arrays and HMDB test videos were not used. Synthetic13 new unit tests and17 existing regression tests passed; passing them is distinct from completing this real experiment.

## Genuine failure evidence

H+M has higher CE than periodic on199 clips, lower CE on178, and numerically identical CE on23. These are post-hoc descriptive subsets, with overlapping group membership possible. Among higher-loss cases, recorded discretionary decisions contain302 selected locally harmful updates and1007 missed locally helpful opportunities. Such local opportunities are budget-constrained hindsight descriptions, not downstream policy treatment effects.

H+M's conditional teacher agreement is69.94% on4740 observed discretionary states. The teacher requests refresh on30.46% of those states, so the observed label-majority “hold” frequency is69.54%. That comparison is a label-frequency statistic on recorded states, **not** the performance of a newly executed or simulated always-hold policy. Different policies visit different conditional teacher-state populations. Teacher confusion counts are TP585, FP566, TN2730, FN859. The score imitates a dense-reference proxy, not beneficial true-label task updates.

Current MAD versus local signed task gain has Spearman correlation-0.0099 across actual H+M discretionary decisions; the higher-loss subset is-0.0219. This cheap appearance-change statistic shows little observed rank association with local task gain here. It does not establish a universal absence of useful cheap signals. Teacher-positive/task-harmful cases occur, so accurate imitation would not guarantee task improvement.

State shift is present but should not be overstated: the mean absolute standardized input is0.3651 in base training,0.3625 in augmentation, and0.3817 in actual H+M states; fractions beyond3 training standard deviations are0.985%,0.952%,1.331%. Age/budget total variation to actual states is0.104 for base and0.054 for augmentation. These are descriptive state-distribution diagnostics, not a proven causal attribution for the CE difference.

H+M's paired median latency increment over periodic is152.2 ms. Periodic+unused-gate adds135.3 ms without changing refresh positions, exposing substantial gate/preprocessing cost in this implementation. H+M median thumbnail time is74.7 ms, cheap-statistics time12.9 ms, policy-decide time20.3 ms. Observation preparation includes thumbnail/statistics work; do not double-count overlapping components. All modes share the generic causal loop, CPU feature readback and native400 projection; no optimized deployment superiority is claimed. See metric_column_semantics.md and failure_subset_review.json.

## Review and decision

Independent input review verified1006 original AVI hashes and head manifest/collection identities. Independent scalar class/group/clip averaging reproduced the primary contrast. All400 actual Dense16 runs agreed with saved genuine development features and validated the B4 logit replays. The group-weight order check passed for the actual population despite nonlexical official class order. BenchPress/JavelinThrow have one execution group each; bootstrap variance is not estimated within those strata. The two rendered PNGs were inspected; corresponding PDFs use the same figures.

The inference framework is an executable development prototype. **Do not advance the current H+M method as an established recognition improvement or periodic speedup.** Preserve the negative/inconclusive utility and positive execution evidence. A separately reviewed development objective/cost investigation may be warranted before claiming a beneficial controller; do not add architectures or tune on the current execution outcomes in this commit.

A fresh scientific evaluation still requires authorized RGB data, verified source/subject/take/camera separation, full-class/group coverage, genuine PTS eligibility, sufficient independent groups and compute/storage, independently separated head/controller fitting and development selection, and a fully frozen protocol before fresh exposure. Unknown pretraining/reupload mappings, prior UCF development exposure, head/controller training overlap, few groups/class, the old head's optimizer limit, and offline full-clip planning remain publication-quality blockers. No new dataset was downloaded and no agreement accepted.

CPU-only supplementary reviews are reproducible after the core run:

```powershell
& .\.venv_temporal\Scripts\python.exe -B .\results\ucf101_causal_development_v1\failure_review_program.py
& .\.venv_temporal\Scripts\python.exe -B .\results\ucf101_causal_development_v1\independent_integrity_program.py
```

These defaults are read-only. Their --write options retain the supplemental descriptive receipts. Core run/resume and read-only audit commands are in README.md. No official-test outcome is needed for any of them.
