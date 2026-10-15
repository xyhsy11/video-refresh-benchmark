# task_signal_diagnostic_v1 findings

This offline diagnostic uses the frozen 400-video-trained linear recognition head and existing saved ResNet18 references. It evaluates 100 development videos and 40 previously inspected exploratory videos separately. The head previously used development data for regularization selection. Neither cohort is a fresh independent confirmation. No completed fresh-confirmation arrays or results were parsed.

Actual RGB encoder executions, video decodes, downloads and recognition-classifier training calls are all zero. Dense uses sixteen candidate refreshes, not every decoded frame. Sparse replay uses four simulated refreshes and sixteen latest-cache outputs. End-to-end latency is unmeasured.

## Caching and classification changes

A policy mistake is not automatically caused by caching. Harmful transitions are restricted to dense-correct sources that become wrong; prediction disagreement is recorded separately.

|Cohort|Policy|Correct / videos|Mean CE (nats)|Mean feature error|Disagreements|Harmful / dense correct|
|---|---|---:|---:|---:|---:|---:|
|development|dense|72/100|1.096979|0.000000|0|0/72|
|development|periodic|70/100|1.150692|0.092942|12|6/72|
|development|value_age_motion_budget|68/100|1.206224|0.094471|15|7/72|
|exploratory|dense|26/40|1.112254|0.000000|0|0/26|
|exploratory|periodic|26/40|1.075162|0.095736|2|0/26|
|exploratory|value_age_motion_budget|24/40|1.315519|0.094659|4|3/26|

|Cohort|Policy|Spearman: feature error vs delta CE|Feature error vs policy wrong AUROC|
|---|---|---:|---:|
|development|periodic|0.0320|0.5319|
|development|value_age_motion_budget|0.2540|0.4881|
|exploratory|periodic|0.0096|0.6209|
|exploratory|value_age_motion_budget|0.0133|0.6641|

These correlations are calculated separately for each policy/cohort. Dense and oracle observations are not pooled into them. The AUROC event is a policy classification error, which includes errors already present under dense inference. Conditional harmful-transition AUROCs and event counts are saved separately; one-class outcomes produce null plus a reason.

## Equal-budget target and probes

Each source supplies nine replacements: move planned refresh 4, 8 or 12 earlier by one, two or three candidate positions. Gain is periodic CE minus early-schedule CE in nats. Histories are identical before the decision, and both schedules use exactly four refreshes. This is one prescribed counterfactual change, not a globally optimal action value.

A uses t/15, elapsed seconds since the cached refresh, remaining refresh count and ordinal distance to the next planned refresh. B adds the existing six motion summaries; C adds cached and past-output logit entropy/max-probability/top-two probability margin; D adds both. Interval descriptors use decoded frames strictly after the cached position through the current candidate. Availability flags/fractions distinguish missing vectors from observed zero displacement. Confidence is never conditioned on the true label.

Ridge alpha is fixed at 1 for the sum-squared-residual objective, with training-only population standardization and an unpenalized intercept. Predictions are signed and unclipped. Five source-grouped class-stratified folds provide development OOF predictions. Only after saving all OOF predictions are probes fitted on all development sources, frozen, and evaluated on exploratory sources. No grid or seed selection was performed.

|Cohort|Videos / examples|Beneficial / neutral / harmful moves|Privileged drift vs gain Spearman|
|---|---:|---:|---:|
|development|100/900|412/0/488|-0.0686|
|exploratory|40/360|154/0/206|-0.0601|

All signed gains enter regression; only values within 1e-12 of zero are called floating-point ties for sign/count reporting. True current drift requires the uncomputed RGB feature and is a privileged diagnostic, never a predictor input.

|Cohort|Probe|Equal-video MAE (nats)|Predicted vs gain Spearman|Three-way sign accuracy|Positive-gain AUROC|
|---|---|---:|---:|---:|---:|
|development_oof|Constant zero|0.253456|undefined|0.00%|0.5000|
|development_oof|Training mean|0.261289|-0.0043|54.22%|0.5188|
|development_oof|A: age/budget|0.264103|-0.0303|54.78%|0.5159|
|development_oof|B: + motion|0.284141|-0.0757|48.56%|0.4684|
|development_oof|C: + cached prediction|0.265663|0.0263|54.00%|0.5029|
|development_oof|D: both|0.284506|-0.0047|49.22%|0.4815|
|exploratory|Constant zero|0.295243|undefined|0.00%|0.5000|
|exploratory|Training mean|0.301426|undefined|57.22%|0.5000|
|exploratory|A: age/budget|0.303550|-0.0264|57.22%|0.4867|
|exploratory|B: + motion|0.323415|-0.1013|53.06%|0.4765|
|exploratory|C: + cached prediction|0.309126|-0.0962|48.06%|0.4338|
|exploratory|D: both|0.330833|-0.1453|48.89%|0.4235|

Spearman in this table is descriptive across repeated examples within each cohort. Per-source correlations, undefined counts, RMSE, sign macro recall and positive precision/recall are also saved. Undefined correlations are missing, not zero. Three-way sign accuracy distinguishes harmful, neutral and beneficial predictions. Constant zero always predicts neutral, so its three-way accuracy is zero when every target is nonzero; this does not conflict with its lower MAE. For a binary positive-versus-nonpositive trigger, always predicting nonpositive would match the harmful-move fractions (54.22% development, 57.22% exploratory). A constant score has AUROC 0.5 when both event classes exist; a one-class outcome is undefined instead.

## Primary paired uncertainty

|Cohort|MAE(A) minus MAE(D), nats|95% percentile interval|Videos improved / tied / worsened|
|---|---:|---|---:|
|development_oof|-0.020402|[-0.031925, -0.009675]|37/0/63|
|exploratory|-0.027283|[-0.043709, -0.013034]|14/0/26|

Intervals use 10,000 PCG64 whole-source-video resamples, seed 20261017, with linear percentiles and shared indices across probes within each cohort. All nine examples remain together. They condition on saved fitted probes; overlapping OOF training folds and full training/model-selection uncertainty are not modeled. The exploratory comparison is separate and previously inspected.

## Answers and next decision

1. Feature error is not a sufficient explanation of harmful recognition changes. The policy-specific drift/loss correlations and conditional transition counts above distinguish additional caching errors from dense errors; correlation does not establish causation.
2. The privileged drift/gain correlations above test association for the nine prescribed moves only. They do not make true drift a cheap trigger or establish a global action-value signal.
3–4. development_oof: cached signals change MAE by -0.001559 nats versus A and -0.000364 versus B. Motion changes MAE by -0.020038 versus A and -0.018843 versus C. Positive values favor the added signals. These component comparisons are descriptive, not independently tuned tests.
3–4. exploratory: cached signals change MAE by -0.005576 nats versus A and -0.007418 versus B. Motion changes MAE by -0.019865 versus A and -0.021707 versus C. Positive values favor the added signals. These component comparisons are descriptive, not independently tuned tests.
5. Development: higher MAE for D. Exploratory: higher MAE for D. Every predefined probe and constant baseline is reported, including negative outcomes. Improvement over A alone does not establish useful prediction if constants remain competitive.
6. These particular signals and linear probes have not demonstrated useful prediction: constant zero has the lowest MAE on both cohorts; adding cached confidence or motion worsens the predefined component comparisons. Do not build a sequential controller from these results. The next single hypothesis should be whether a low-cost task-trained temporal-change signal predicts equal-budget refresh gain beyond constant and age/budget baselines, in a separate frozen study. Test that signal before implementing a controller. Do not assemble the nine independent counterfactual predictions into a claimed sequential policy: later cache states would change. Failure here does not rule out every task-aware policy.

Candidate indices use the full clip and codec vectors can reference future frames. Prefix descriptor access therefore remains an offline diagnostic, not strict streaming causality. Ground-truth labels and full RGB references appear only in targets/evaluation, never the inexpensive feature builder. Appearance/recognition probes do not establish maintained recognition for an adaptive controller, actual latency, or speedup.

## Reproduction

```powershell
& .\.venv\Scripts\python.exe -m unittest test_task_signals -v
& .\.venv\Scripts\python.exe .\analyze_task_signals.py --input results/recognition_baseline_v2 --output results/task_signal_diagnostic_v1 --seed 20261017
& .\.venv\Scripts\python.exe .\analyze_task_signals.py --input results/recognition_baseline_v2 --output results/task_signal_diagnostic_v1 --seed 20261017 --audit-only
```

Completed outputs are audited rather than overwritten. The protocol freezes input/source identities and folds before outcome comparisons. No acquisition, reference collection, recognition fitting or fresh-confirmation evaluation is permitted.
