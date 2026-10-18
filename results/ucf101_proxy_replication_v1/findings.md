# ucf101_proxy_replication_v1

Post-exposure audit/layout amendment: see [amendment](post_exposure_amendment.md) and [completed evidence review](completion_review.md). The original seal/exposure/code and 23 completed scientific outputs are preserved. A pandas audit column-access defect and figure tick overlap were corrected; scientific estimators and results are unchanged. The implementation is not described as untouched confirmation.

Actual starting HEAD: `340a1ad4f08d4e817f916ac2c3c159aa50200687`. Reviewed tree equal: True.
Execution state: **verified**, stage `final`.

|Contrast|Mean CE difference, nats|95% paired GROUP CI|Groups / clips|
|---|---:|---|---|
|g|-0.010285|[-0.016929, -0.003889]|704 / 1408|
|p|-0.008225|[-0.015486, -0.001301]|704 / 1408|
|d|+0.002060|[-0.004773, +0.009124]|704 / 1408|

Primary achieved interval width: 0.013040 nats.
Improvement over LegacyCosine established by this interval.
Periodic beaten in secondary CE interval.

Exact decomposition of g: directional difference -0.004468 + KL difference -0.005817.
This describes signed movement relative to the true label and dense prediction discrepancy. It is standard algebra, not a new theorem or causal mechanism. Labels enter only this retrospective explanation and the explicit task oracles.

|Decomposition contribution|Mean|Secondary marginal 95% group CI|
|---|---:|---|
|directional_g|-0.004468|[-0.010940,+0.001846]|
|KL_g|-0.005817|[-0.007024,-0.004701]|

|Selector, B4 C (dense16 separate)|CE|Balanced top1|Ordinary clip accuracy|Dense agreement|KL|
|---|---:|---:|---:|---:|---:|
|LegacyCosine|0.351342|90.15%|90.13%|98.23%|0.009517|
|Z_pool|0.341057|90.36%|90.34%|98.44%|0.003700|
|periodic|0.349282|90.15%|90.13%|98.51%|0.011219|
|dense16|0.335901|90.64%|90.62%|100.00%|0.000000|

Achieved eligible counts (all 101 official classes):
- train: 1510 groups, 3020 clips; per-class groups 13 to 15.
- development: 301 groups, 602 clips; per-class groups 2 to 3.
- test: 704 groups, 1408 clips; per-class groups 5 to 7.

Conclusions apply to the technically eligible split-1 population under the frozen two-clips-per-group rule and duplicate quarantine.

Development-based predicted g halfwidth: 0.006051 nats (planning target .015; noisy variance transfer, no power guarantee).
Development-based predicted p halfwidth: 0.007745 nats (planning target .015; noisy variance transfer, no power guarantee).

Measured median across 30 clip/group medians: dense16 1945.4 ms, periodicB4 497.7 ms; median paired dense/periodic ratio 3.945.
Encoded bytes in memory through fresh full decode/planning, selected preparation, native encoder/head and CUDA-completed prediction. Two warmups/five interleaved trials per mode, one clip/group; repeats are not independent groups. Proxy selectors need dense16 plus separately logged exhaustive search.

## Within-dataset comparison and limits

|Dataset / head / averaging unit|g|p|
|---|---|---|
|Kinetics fresh N=120 / 20 classes / source|-0.011773 [-0.034418,+0.008860]|+0.001131 [-0.021947,+0.023645]|
|UCF101 / NEW 101-class head / clips within group, groups within class|-0.010285 [-0.016929,-0.003889]|-0.008225 [-0.015486,-0.001301]|

No raw cross-dataset CE average is a primary statistic. Different labels/heads and independent units prevent that interpretation. This is a second-dataset replication with disjoint experimental groups; it is not universally pretraining-clean. Unknown UCF-to-Kinetics/reupload mappings remain. This split-1 frozen-feature protocol is not the full three-split UCF101 benchmark or a state-of-the-art comparison. Splits 2/3 overlap sources and are not independent datasets.

All proxy selectors require 16 dense snippet encodings plus search. True-label task_C/task_U are privileged hindsight. U cross-evaluation holds C-selected schedules fixed and is a target-mismatch sensitivity. Secondary intervals are marginal/unadjusted; sparse strata (<20 represented groups) support no population claim. Bootstrap intervals condition on the fitted head and selected classes, omitting training/model-selection/repeated-research uncertainty.

## Claim / evidence

|Claim|Evidence / limit|
|---|---|
|Archived Kinetics g/p reproduced|baseline_verification.json: genuine per-source tables and saved bootstrap indices; real feature replays listed separately|
|All 101 classes and group isolation|population_plan.json and eligible_manifest.json; population gate blocks if minima fail|
|Recognition improvement|Only real UCF primary_intervals.json can establish this; pending files imply no result|
|Directional + KL attribution|Independent scalar identity checks and per-clip/group contributions; descriptive, not causal|
|Practical cost|Only actual timing_trials.csv at encoded-memory boundary; no periodic latency assigned to an oracle|
|Historical work preserved|Reviewed tree comparison, old seals, unchanged tracked historical result blobs|

## Next research decision

The fixed UCF comparison favors Z_pool over LegacyCosine and the preplanned periodic comparator. A separately preregistered deployable-policy feasibility study may follow, with a new evaluation cohort and explicit cost limits; these dense-reference selectors are not such a policy. The Kinetics result remains inconclusive, so cross-dataset generality is unresolved. Do not extend exposed test groups, search budgets/backbones, or substitute another dataset to obtain significance. Passing validation does not establish publication readiness.

## PowerShell

```powershell
& .\.venv_temporal\Scripts\python.exe .\run_ucf101_proxy_replication.py --resume
& .\.venv_temporal\Scripts\python.exe -B .\run_ucf101_proxy_replication.py --audit-only
```
