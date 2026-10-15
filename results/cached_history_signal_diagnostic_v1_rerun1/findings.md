# cached_history_signal_diagnostic_v1 findings (Experiment13)

This completed offline study changes only observation representation. It preserves the frozen recognition head, signed net-gain target,100 development/40 exploratory sources,900/360 opportunities,original five folds,RF400 recipe/seed20261020 and decision threshold1e-12. The baseline's six fitted models are reused byte-for-byte; only six augmented forests are fitted. Both cohorts were previously inspected, and recognition regularization used development. Neither is independent confirmation.

## Answers

1. **Information added:** no new information at history length1; length2 is algebraically recoverable from old C; length3 can preserve individual/order information discarded by the weighted past mean. T also makes differences and nonlinear statistics explicit. The44 features are dependent derived quantities, not44 new independent observations. See the controlled information audit.
2. **Relative utility:** development augmented minus baseline is +0.013177 [-0.001302, +0.029815]; exploratory is +0.004578 [-0.006186, +0.019054] nats/opportunity.
3. **Absolute utility:** augmented development is -0.011949 [-0.028149, +0.003645]; exploratory is -0.059949 [-0.196221, +0.019307]. Keeping reference is zero. A less negative result does not establish useful decisions.
4. **History length:** all three predeclared subgroup comparisons are below, using the same full-data models. They are descriptive, marginal and unadjusted. In length1, T is entirely zero, so prediction changes can reflect the refitted forest's partitions rather than new information at that opportunity. No favorable subgroup is selected as a new primary result.
5. **Incremental work:** state stores three cached vectors, three margins/top1 indices/refresh ordinals,one cached44-vector and a counter. Each of3 distinct events/video computes one stable20-class softmax/top1 and rebuilds T;6 repeated observations/video reuse T. Actual memory and component timings are below.
6. **Next step:** Move to a separately designed temporal-recognition backbone/task benchmark before another cached-feature variant. Freeze its task and independent-source evaluation protocol first; do not automatically launch it here.

The predeclared positive-relative and positive-absolute utility requirements are not both established; useful positive utility remains unproven.

The point estimates improve relative to C in both cohorts, but the development primary interval [-0.001302, +0.029815] and exploratory interval [-0.006186, +0.019054] both include zero. Absolute augmented utility remains negative in both cohorts. Development MAE changes from 0.280248 to 0.278711, while exploratory MAE worsens from 0.309259 to 0.315159. Neither beats its training-median baseline (0.253471 and 0.295157).

The largest development subgroup gain is at history length two: +0.026357 [0.003145, 0.061823], a predeclared secondary, unadjusted comparison. At that length all T values are recoverable from C; it therefore does not demonstrate a benefit from newly preserved order information. History-length-three gains are +0.011393 [-0.018980, +0.047814] and +0.005874 [-0.000050, +0.015163], both inconclusive. No subgroup is promoted to the primary analysis and no separate subgroup model is fitted.

## Overall measurements

Utility averages all9 opportunities within source, including zero for unselected moves, then gives sources equal weight. Signed targets and predictions are unclipped. These independent moves are never combined into a sequential schedule.

|Cohort|Method|MAE|MSE|MAE improvement vs training median|Utility and95% interval|Selected/total|Helpful/harmful/neutral|Mean selected gain (denominator)|
|---|---|---:|---:|---|---|---:|---|---|
|development_oof|keep_reference|0.253456|0.218374|+0.000014 [-0.000132, +0.000162]|+0.000000 [+0.000000, +0.000000]|0/900|0/0/0|undefined:no selections|
|development_oof|training_mean|0.261289|0.218163|-0.007818 [-0.012135, -0.003437]|+0.000000 [+0.000000, +0.000000]|0/900|0/0/0|undefined:no selections|
|development_oof|training_median|0.253471|0.218284|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|0/900|0/0/0|undefined:no selections|
|development_oof|C_RF400|0.280248|0.231720|-0.026778 [-0.037808, -0.016193]|-0.025125 [-0.042013, -0.008677]|285/900|107/178/0|-0.079343 (285)|
|development_oof|C_plus_T_RF400|0.278711|0.226969|-0.025240 [-0.036286, -0.014739]|-0.011949 [-0.028149, +0.003645]|275/900|116/159/0|-0.039104 (275)|
|exploratory|keep_reference|0.295243|0.430683|-0.000087 [-0.000276, +0.000099]|+0.000000 [+0.000000, +0.000000]|0/360|0/0/0|undefined:no selections|
|exploratory|training_mean|0.301426|0.423503|-0.006269 [-0.012797, +0.000692]|+0.000000 [+0.000000, +0.000000]|0/360|0/0/0|undefined:no selections|
|exploratory|training_median|0.295157|0.430424|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|0/360|0/0/0|undefined:no selections|
|exploratory|C_RF400|0.309259|0.460252|-0.014102 [-0.028816, +0.000381]|-0.064527 [-0.202720, +0.017148]|121/360|54/67/0|-0.191980 (121)|
|exploratory|C_plus_T_RF400|0.315159|0.469302|-0.020003 [-0.036353, -0.004883]|-0.059949 [-0.196221, +0.019307]|110/360|44/66/0|-0.196197 (110)|

All MAE/utility units are nats; MSE is nats squared. Spearman and positive-gain AUROC are pooled descriptive statistics, with event counts and null/reason for undefined cases in aggregate_metrics.json. Candidate observations are correlated within source.

## History-length comparisons (secondary)

|Cohort|Distinct refreshes|Baseline utility|Augmented utility|Paired augmented-minus-baseline and95% interval|Improved/tied/worsened sources|
|---|---:|---:|---:|---|---|
|development_oof|1|-0.012902|-0.011123|+0.001780 [-0.009764, +0.016245]|7/80/13|
|development_oof|2|-0.039236|-0.012879|+0.026357 [+0.003145, +0.061823]|12/84/4|
|development_oof|3|-0.023237|-0.011844|+0.011393 [-0.018980, +0.047814]|13/77/10|
|exploratory|1|-0.075930|-0.081197|-0.005267 [-0.015627, +0.002162]|2/33/5|
|exploratory|2|-0.134127|-0.121001|+0.013126 [-0.012332, +0.049323]|2/36/2|
|exploratory|3|+0.016477|+0.022351|+0.005874 [-0.000050, +0.015163]|2/37/1|

## Component CPU work and model accounting

CPU: {'Name': 'Intel(R) Core(TM) i7-10870H CPU @ 2.20GHz', 'NumberOfCores': 8, 'NumberOfLogicalProcessors': 16}. Python3.12.10;RF n_jobs1;OMP/MKL/OpenBLAS threads1. Timings use10 development sources selected lexicographically before outcomes,90 opportunities/pass,2 warmup and5 measured passes. All observations/models are preloaded. Methods are interleaved with PCG64 seed20261021;actual orders and identities are recorded.

|Component|Median microseconds/opportunity|Min|max|Measured passes|
|---|---:|---:|---:|---:|
|augmented_history_scale_RF1|88042.214|82276.186|95725.259|5|
|baseline_scale_RF1|83250.157|82616.569|88802.034|5|
|history_T|33.116|26.541|53.241|5|

Retained history ndarray payload is 904bytes plus a Python integer counter (28bytes in this environment). The slots/array/counter sizeof estimate is 1588bytes, excluding allocator/transient/RF workspace. A100-value float64 augmented input is800bytes versus448bytes for C, an additional352bytes per transient input. Scaling and sklearn float32 buffers add transient workspace; forest memory is separate.
History_T includes per-video state allocation/reset,all30 distinct cached events and90 feature accesses/pass. Stable softmax is computed once per distinct event; T is cached for repetitions. Baseline includes56-column scaling plus90 batch-size1 RF calls; augmented includes history handling,100-column concatenation/scaling and90 batch-size1 RF calls. Library prediction validation is included; external correctness assertions,loading,fitting and logging are excluded.
These are cached-observation component CPU timings. They exclude video decoding,RGB extraction,codec-vector extraction and full recognition execution. They are not adaptive inference latency, cannot be subtracted from the old pilot, and establish no end-to-end speedup. Five passes describe local variability, not statistical significance.

Per video, the implementation performs three stable 20-class softmax/argmax calculations and three T assemblies. The second and third events compute 20 and 40 score differences respectively (60 scalar subtractions total), population standard deviations over two/three retained margins, and one/two consecutive top-class comparisons. Six repeated observations reuse the cached T vector. The microbenchmark also stages a separate 20-value cached-score copy for each of its 90 preloaded observations (14,400 numeric bytes); that input-staging copy and its allocation are outside the timing boundary and are not part of the 904-byte per-video history state.

Median component times are 33.116 microseconds per opportunity for history/T alone, 83.250 milliseconds for baseline scaling plus batch-size-one RF400, and 88.042 milliseconds for augmented history/scaling/RF400. The prediction timing ranges overlap (82.617-88.802 ms and 82.276-95.725 ms). The median gap is not an isolated measurement of history cost: the models have different partitions and there are only five measured passes. Both predictors are resident during interleaving; serialized sizes below are not runtime-memory measurements.

|Final-development model|Trees|Nodes|Compressed serialized bytes|
|---|---:|---:|---:|
|C_RF400|400|6244|169418|
|C_plus_T_RF400|400|5264|140611|

Equal400-tree counts do not imply equal partitions, effective capacity, memory or inference cost. Extra explicit features can change the shallow forest's partitions even where they contain no new information.

## Verification and limits

Baseline prediction discrepancy is 0.0;1260 independent T checks and282 input-isolation mutations pass. Max first56 scaler discrepancy is 0. All folds/targets/source memberships,training-only scaling,latest history membership,missing-history zeros,cache-event deduplication,saved metrics and whole-source bootstrap reproduction pass. Earlier scientific artifacts and the head remain unchanged.
Actual RGB encoder executions,video decodes,video downloads and recognition-head training calls are zero. No fresh-confirmation artifact is accessed. Full-clip candidates and future-reference codec vectors retain the offline limitation; this does not evaluate a temporal video backbone or validate streaming causality.
Intervals use10000 paired whole-source resamples,PCG64 seed20261021,linear percentiles. All9 opportunities stay together,shared indices cover every method/group within cohort,and cohorts remain separate. Intervals condition on fitted models/reused cohorts and omit full training/model-selection/repeated-research uncertainty. Secondary intervals are unadjusted. No accuracy improvement,successful adaptive controller or end-to-end speedup is established.

## Preserved verification correction

The original default directory contains a preserved incomplete run: its strict verifier stopped because CSV inference turned float64 0/1 history masks into int64. Only the verification CSV reader and a schema regression test changed. The corrected reader explicitly retains float64 for all 44 T columns; dtype/value checks and numerical tolerances remain strict. The original protocol, sources and measurements were not overwritten. This completed run uses `results/cached_history_signal_diagnostic_v1_rerun1/`. Each attempt fitted six augmented probes, for twelve augmented-probe fits across both attempts; neither attempt trained the recognition head.

Original and rerun prediction CSVs, aggregate metrics and bootstrap intervals are byte-identical. CPU timings were measured anew under the unchanged timing protocol. See [the correction record](verification_correction/correction.json) for source hashes, the archived original protocol and the reproduction checks. No feature, target, model, threshold, seed or cohort was selected in response to outcomes.

## Reproduce

```powershell
& .\.venv\Scripts\python.exe -m unittest test_cached_history_signals -v
& .\.venv\Scripts\python.exe .\analyze_cached_history_signals.py --predictability-input results/two_target_refresh_predictability_v1 --output results/cached_history_signal_diagnostic_v1_rerun1 --seed 20261021 --bootstrap 10000
& .\.venv\Scripts\python.exe .\analyze_cached_history_signals.py --predictability-input results/two_target_refresh_predictability_v1 --output results/cached_history_signal_diagnostic_v1_rerun1 --seed 20261021 --bootstrap 10000 --audit-only
```

Completed invocations audit without overwriting or retiming;changed protocols select a clearly versioned new output directory. Models,observations,predictions,timing rows and indices are retained for independent audit.
