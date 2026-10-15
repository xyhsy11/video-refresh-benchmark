# task_signal_ablation_v2 findings

This fixed 3-input x 2-predictor experiment reuses exactly 100 development and 40 exploratory sources, 900/360 saved signed counterfactual targets, the original five source-grouped class-stratified folds, and the frozen primary recognition head. The new seed 20261018 affects bootstrap bookkeeping only; fold assignment remains the saved seed-20261017 assignment.

Both cohorts are developmental. Recognition regularization already used development labels, and exploratory outcomes were previously inspected. The completed fresh-confirmation cohort was not parsed. Actual RGB encoder executions, video decodes, downloads and recognition training calls are zero.

## Inputs, probes and cost

S is exactly the previous 16-dimensional group D. C adds 40 class-centered logits: cached feature logits and mean output logits for past candidates 0 through t-1 only. P adds 20 centered current-feature logits and is privileged. P is not a cheap gate: keeping the reference schedule after observing f_t can add an RGB call. Four-refresh counterfactual schedules do not establish matched deployment cost for P.

Linear ridge uses training-only population standardization, sum squared residuals plus coefficient norm squared, alpha=1, and an unpenalized intercept. RBF kernel ridge uses exp(-squared distance/p), lambda=1, training-kernel/target centering and training-only query centering. Kernel width, regularization, features and seeds were not searched. Signed predictions are unclipped. Saved models include raw/standardized training rows, scaler and kernel statistics and coefficients.

The target is one prescribed equal-budget schedule change: CE(periodic) minus CE(earlier single refresh), not an optimal action value. All compared schedules contain initial refresh 0, four refreshes and sixteen latest-cache outputs. The nine counterfactuals per source are not combined into a sequential policy.

## development_oof

|Configuration|MAE (nats)|MSE (nats squared)|Spearman|Positive-gain AUROC|Realized gain (nats/decision)|Selected helpful/harmful/neutral|
|---|---:|---:|---:|---:|---:|---:|
|S-linear|0.284506|0.244926|-0.0047|0.4815|-0.019913|123/168/0|
|S-nonlinear|0.279222|0.228657|0.0053|0.4837|-0.014840|137/178/0|
|C-linear|0.348287|0.293865|-0.0342|0.4700|-0.031328|171/230/0|
|C-nonlinear|0.304876|0.238098|-0.0423|0.4427|-0.014451|178/245/0|
|P-linear (privileged)|0.373602|0.331759|-0.0676|0.4709|-0.040703|172/224/0|
|P-nonlinear (privileged)|0.305551|0.243523|-0.0580|0.4476|-0.027592|175/240/0|
|zero|0.253456|0.218374|undefined|undefined|+0.000000|0/0/0|
|training_mean|0.261289|0.218163|-0.0043|0.5188|+0.000000|0/0/0|
|training_median|0.253471|0.218284|-0.0259|0.4722|+0.000000|0/0/0|
|A-linear|0.264103|0.219746|-0.0303|0.5159|-0.002988|29/24/0|
|keep_reference|not a regression probe|not applicable|undefined|undefined|+0.000000|0/0/0|
|hindsight_positive_gain_reference (privileged)|not a regression probe|not applicable|undefined|undefined|+0.106017|412/0/0|

|Configuration|Selected / opportunities|Selection fraction|Helpful / selected|Mean gain among selected (nats)|
|---|---:|---:|---:|---:|
|S-linear|291/900|0.3233|123/291 (0.4227)|-0.061586|
|S-nonlinear|315/900|0.3500|137/315 (0.4349)|-0.042399|
|C-linear|401/900|0.4456|171/401 (0.4264)|-0.070311|
|C-nonlinear|423/900|0.4700|178/423 (0.4208)|-0.030746|
|P-linear (privileged)|396/900|0.4400|172/396 (0.4343)|-0.092507|
|P-nonlinear (privileged)|415/900|0.4611|175/415 (0.4217)|-0.059839|
|zero|0/900|0.0000|0/0 (undefined)|undefined|
|training_mean|0/900|0.0000|0/0 (undefined)|undefined|
|training_median|0/900|0.0000|0/0 (undefined)|undefined|
|A-linear|53/900|0.0589|29/53 (0.5472)|-0.050732|
|keep_reference|0/900|0.0000|0/0 (undefined)|undefined|
|hindsight_positive_gain_reference (privileged)|412/900|0.4578|412/412 (1.0000)|0.231591|

## exploratory

|Configuration|MAE (nats)|MSE (nats squared)|Spearman|Positive-gain AUROC|Realized gain (nats/decision)|Selected helpful/harmful/neutral|
|---|---:|---:|---:|---:|---:|---:|
|S-linear|0.330833|0.457549|-0.1453|0.4235|-0.069211|55/85/0|
|S-nonlinear|0.325476|0.486963|-0.1005|0.4507|-0.096858|69/107/0|
|C-linear|0.368692|0.492374|-0.1129|0.4351|-0.058499|62/88/0|
|C-nonlinear|0.339828|0.478741|-0.0582|0.4750|-0.088699|67/84/0|
|P-linear (privileged)|0.381820|0.519432|-0.1429|0.4089|-0.092284|57/96/0|
|P-nonlinear (privileged)|0.341665|0.488625|-0.1259|0.4356|-0.098007|54/94/0|
|zero|0.295243|0.430683|undefined|undefined|+0.000000|0/0/0|
|training_mean|0.301426|0.423503|undefined|undefined|+0.000000|0/0/0|
|training_median|0.295157|0.430424|undefined|undefined|+0.000000|0/0/0|
|A-linear|0.303550|0.425915|-0.0264|0.4867|-0.000201|1/1/0|
|keep_reference|not a regression probe|not applicable|undefined|undefined|+0.000000|0/0/0|
|hindsight_positive_gain_reference (privileged)|not a regression probe|not applicable|undefined|undefined|+0.093932|154/0/0|

|Configuration|Selected / opportunities|Selection fraction|Helpful / selected|Mean gain among selected (nats)|
|---|---:|---:|---:|---:|
|S-linear|140/360|0.3889|55/140 (0.3929)|-0.177972|
|S-nonlinear|176/360|0.4889|69/176 (0.3920)|-0.198118|
|C-linear|150/360|0.4167|62/150 (0.4133)|-0.140397|
|C-nonlinear|151/360|0.4194|67/151 (0.4437)|-0.211469|
|P-linear (privileged)|153/360|0.4250|57/153 (0.3725)|-0.217140|
|P-nonlinear (privileged)|148/360|0.4111|54/148 (0.3649)|-0.238395|
|zero|0/360|0.0000|0/0 (undefined)|undefined|
|training_mean|0/360|0.0000|0/0 (undefined)|undefined|
|training_median|0/360|0.0000|0/0 (undefined)|undefined|
|A-linear|2/360|0.0056|1/2 (0.5000)|-0.036251|
|keep_reference|0/360|0.0000|0/0 (undefined)|undefined|
|hindsight_positive_gain_reference (privileged)|154/360|0.4278|154/154 (1.0000)|0.219581|

## Primary and secondary comparisons

Earlier refresh is chosen only when prediction exceeds 1e-12. Realized gain is actual signed gain times this indicator. Average nine decisions within a source first, then average sources. Always keeping the reference gives zero. The separate hindsight reference max(actual gain,0) knows the outcome and is not deployable.

|Cohort|Candidate minus baseline comparison|Metric (positive favors candidate)|Mean improvement|95% interval|
|---|---|---|---:|---|
|development_oof|C-nonlinear vs keep_reference|realized_gain|-0.014451|[-0.039733, +0.011739]|
|development_oof|S-nonlinear vs S-linear|MAE|+0.005284|[-0.006442, +0.017458]|
|development_oof|S-nonlinear vs S-linear|realized_gain|+0.005073|[-0.016107, +0.026137]|
|development_oof|C-linear vs S-linear|MAE|-0.063781|[-0.082355, -0.046055]|
|development_oof|C-linear vs S-linear|realized_gain|-0.011415|[-0.031919, +0.009296]|
|development_oof|C-nonlinear vs S-nonlinear|MAE|-0.025655|[-0.037987, -0.013650]|
|development_oof|C-nonlinear vs S-nonlinear|realized_gain|+0.000389|[-0.022344, +0.023980]|
|development_oof|P-nonlinear (privileged) vs C-nonlinear|MAE|-0.000674|[-0.007480, +0.006153]|
|development_oof|P-nonlinear (privileged) vs C-nonlinear|realized_gain|-0.013142|[-0.031112, +0.002011]|
|exploratory|C-nonlinear vs keep_reference|realized_gain|-0.088699|[-0.221550, -0.003094]|
|exploratory|S-nonlinear vs S-linear|MAE|+0.005356|[-0.015983, +0.027395]|
|exploratory|S-nonlinear vs S-linear|realized_gain|-0.027646|[-0.117637, +0.029194]|
|exploratory|C-linear vs S-linear|MAE|-0.037860|[-0.058294, -0.018342]|
|exploratory|C-linear vs S-linear|realized_gain|+0.010713|[-0.024835, +0.047335]|
|exploratory|C-nonlinear vs S-nonlinear|MAE|-0.014351|[-0.033989, +0.003937]|
|exploratory|C-nonlinear vs S-nonlinear|realized_gain|+0.008158|[-0.028172, +0.043872]|
|exploratory|P-nonlinear (privileged) vs C-nonlinear|MAE|-0.001838|[-0.012528, +0.008805]|
|exploratory|P-nonlinear (privileged) vs C-nonlinear|realized_gain|-0.009307|[-0.027271, +0.009102]|

For MAE, improvement is baseline minus candidate. For decision utility, it is candidate minus baseline. The fixed primary is development OOF C-nonlinear versus always keeping reference. Exploratory is reported separately; secondary intervals are exploratory, marginal and unadjusted.

All intervals use 10,000 paired whole-source-video resamples with PCG64 seed20261018, linear percentiles, and one shared index matrix across configurations within each cohort. Nine repeated moves stay together. Intervals condition on saved probes and omit full training, model-selection and repeated-research uncertainty. Spearman/AUROC are descriptive across correlated opportunities. One-class outcomes and constant scores have explicit null/reason, never fabricated discrimination.

## Interpretation and next action

The measured contrasts do not establish a sufficient explanation for v1's failure. S-nonlinear lowers MAE by about 0.0053 nats in each cohort, but both intervals span zero and its decision utility remains negative. C adds information without improving the tested models: linear MAE worsens in both cohorts; C-nonlinear versus S-nonlinear does not show a reliable utility gain. Privileged P-nonlinear likewise does not improve on C-nonlinear within the reported uncertainty. These statements apply to the fixed ridge/RBF probes and these data; they do not establish that the inputs contain no predictive information.

development_oof: inconclusive decision utility; mean -0.014451 nats/decision, interval [-0.039733,+0.011739].
- S-nonlinear versus S-linear: MAE improvement +0.005284; utility improvement +0.005073. These diagnose capacity, retained class-score information, or current appearance, not a newly selected winning method.
- C-linear versus S-linear: MAE improvement -0.063781; utility improvement -0.011415. These diagnose capacity, retained class-score information, or current appearance, not a newly selected winning method.
- C-nonlinear versus S-nonlinear: MAE improvement -0.025655; utility improvement +0.000389. These diagnose capacity, retained class-score information, or current appearance, not a newly selected winning method.
- P-nonlinear (privileged) versus C-nonlinear: MAE improvement -0.000674; utility improvement -0.013142. These diagnose capacity, retained class-score information, or current appearance, not a newly selected winning method.
exploratory: evidence of harmful decision utility; mean -0.088699 nats/decision, interval [-0.221550,-0.003094].
- S-nonlinear versus S-linear: MAE improvement +0.005356; utility improvement -0.027646. These diagnose capacity, retained class-score information, or current appearance, not a newly selected winning method.
- C-linear versus S-linear: MAE improvement -0.037860; utility improvement +0.010713. These diagnose capacity, retained class-score information, or current appearance, not a newly selected winning method.
- C-nonlinear versus S-nonlinear: MAE improvement -0.014351; utility improvement +0.008158. These diagnose capacity, retained class-score information, or current appearance, not a newly selected winning method.
- P-nonlinear (privileged) versus C-nonlinear: MAE improvement -0.001838; utility improvement -0.009307. These diagnose capacity, retained class-score information, or current appearance, not a newly selected winning method.

Do not build a sequential controller from these probes. The next action is one separately frozen target-decomposition diagnostic: test whether dependence on later content dominates early-refresh gain. A move changes both the immediate cached outputs and outputs after the displaced planned refresh, so even current appearance need not reveal its whole-clip effect. This is a hypothesis for a later study, not an explanation established here. Keep data support and the static appearance representation as limitations. P itself cannot be deployed as a cheap gate.

Improved S-nonlinear would suggest interactions missed by the linear summary probe. C-over-S gains would suggest summaries discarded useful class-score information. Gains limited to P would suggest information missing from cached observations. No improvement would motivate revisiting target/representation/data. A single fixed RBF kernel cannot establish that all nonlinear models fail, and P is not an information-theoretic upper bound.

Any promising result still needs new-source confirmation. Full-clip candidate selection and future-reference codec vectors keep this offline. No sequential recognition result, adaptive latency, speedup, or matched P deployment cost is measured.

## Reproduction

```powershell
& .\.venv\Scripts\python.exe -m unittest test_task_signal_ablation -v
& .\.venv\Scripts\python.exe .\analyze_task_signal_ablation.py --recognition-input results/recognition_baseline_v2 --diagnostic-input results/task_signal_diagnostic_v1 --output results/task_signal_ablation_v2 --seed 20261018
& .\.venv\Scripts\python.exe .\analyze_task_signal_ablation.py --recognition-input results/recognition_baseline_v2 --diagnostic-input results/task_signal_diagnostic_v1 --output results/task_signal_ablation_v2 --seed 20261018 --audit-only
```

Completed outputs are audited without overwriting. Incomplete runs resume only with matching frozen input/source/configuration identities. Previous experiments remain unchanged.
