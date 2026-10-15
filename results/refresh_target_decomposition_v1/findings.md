# refresh_target_decomposition_v1 findings

This retrospective diagnostic uses the unchanged frozen primary recognition head, saved candidate features, 100 development sources and 40 exploratory sources. There are nine prescribed opportunities per source (900/360), not 900/360 independent videos. Both cohorts are developmental; neither is an untouched confirmation. No predictor or recognition head was fitted, and actual encoder executions, video decodes and downloads are zero.

## Four controls and signed attribution

A is [0,4,8,12]. B adds t while retaining r (five refreshes); C replaces r with t (four); D removes r without adding t (three). All produce sixteen latest-cache feature outputs. B and D are unequal-budget diagnostic controls, never equal-cost deployment improvements.

Forward benefit is LA-LB and cost is LC-LB. Reverse benefit is LD-LC and cost is LD-LA. The two-order averages preserve net gain LA-LC. All components remain signed: a negative removal cost means removing that refresh lowers loss in that context. Cross entropy is nonlinear, so neither ordering is a unique causal explanation. Even the early-addition component depends on the whole-video prediction context and true label; it is not an observable online signal.

## Primary and conditional cancellation rates

Cancellation means benefit >1e-12 and net gain < -1e-12. The primary is the forward count divided by all opportunities, separately in each cohort. Reverse and averaged rows are secondary attribution checks.

|Cohort|Scope|Order|Cancellation / opportunities|95% interval (%)|Cancellation / harmful moves|Cancellation / beneficial additions|
|---|---|---|---:|---|---:|---:|
|development|C_nonlinear_selected|averaged|75/423 (17.73%)|[13.8095, 21.8597]%|75/245 (30.61%)|75/200 (37.50%)|
|development|C_nonlinear_selected|forward|78/423 (18.44%)|[14.2857, 22.8703]%|78/245 (31.84%)|78/202 (38.61%)|
|development|C_nonlinear_selected|reverse|74/423 (17.49%)|[13.6674, 21.5910]%|74/245 (30.20%)|74/200 (37.00%)|
|development|all|averaged|151/900 (16.78%)|[14.3333, 19.2222]%|151/488 (30.94%)|151/440 (34.32%)|
|development|all|forward|153/900 (17.00%)|[14.4444, 19.5556]%|153/488 (31.35%)|153/439 (34.85%)|
|development|all|reverse|151/900 (16.78%)|[14.3333, 19.3333]%|151/488 (30.94%)|151/442 (34.16%)|
|exploratory|C_nonlinear_selected|averaged|17/151 (11.26%)|[6.0606, 16.9233]%|17/84 (20.24%)|17/70 (24.29%)|
|exploratory|C_nonlinear_selected|forward|16/151 (10.60%)|[5.8824, 15.5556]%|16/84 (19.05%)|16/68 (23.53%)|
|exploratory|C_nonlinear_selected|reverse|17/151 (11.26%)|[6.0606, 16.9233]%|17/84 (20.24%)|17/70 (24.29%)|
|exploratory|all|averaged|57/360 (15.83%)|[11.9444, 19.7222]%|57/206 (27.67%)|57/177 (32.20%)|
|exploratory|all|forward|56/360 (15.56%)|[11.6667, 19.4444]%|56/206 (27.18%)|56/175 (32.00%)|
|exploratory|all|reverse|57/360 (15.83%)|[11.9444, 19.7222]%|57/206 (27.67%)|57/178 (32.02%)|

Intervals use 10,000 whole-source-video resamples with PCG64 seed 20261019 and linear 2.5th/97.5th percentiles. The same within-cohort resampling indices are shared across orders, scopes and metrics.

All counts/denominators and conditional intervals are saved. Bootstrap numerators and denominators are recomputed after sampling whole source videos; zero denominators are undefined, with invalid-resample counts and null intervals if all are invalid. No conditional rate is imputed as zero.

## Signed means and absolute component sizes

|Cohort|Scope|Order|Defined component sources / all sources|Mean benefit|Mean cost|Mean absolute benefit|Mean absolute cost|
|---|---|---|---:|---:|---:|---:|---:|
|development|C_nonlinear_selected|averaged|74/100|0.050728|0.087454|0.200053|0.208339|
|development|C_nonlinear_selected|forward|74/100|0.012926|0.049652|0.090340|0.192292|
|development|C_nonlinear_selected|reverse|74/100|0.088530|0.125256|0.310214|0.264807|
|development|all|averaged|100/100|0.028699|0.070121|0.195767|0.201004|
|development|all|forward|100/100|0.006933|0.048355|0.091356|0.204337|
|development|all|reverse|100/100|0.050466|0.091888|0.300614|0.241342|
|exploratory|C_nonlinear_selected|averaged|34/40|-0.048301|0.123483|0.189243|0.219758|
|exploratory|C_nonlinear_selected|forward|34/40|-0.033304|0.138480|0.086744|0.259699|
|exploratory|C_nonlinear_selected|reverse|34/40|-0.063298|0.108486|0.292600|0.220550|
|exploratory|all|averaged|40/40|-0.024840|0.082540|0.199011|0.207114|
|exploratory|all|forward|40/40|-0.018838|0.088542|0.093164|0.233623|
|exploratory|all|reverse|40/40|-0.030842|0.076538|0.305569|0.229869|

Continuous outcomes are averaged within source video first, in nats. Selected-subset component means then give equal weight to sources with at least one selected opportunity; empty sources are undefined and their counts are shown. Count ratios retain opportunity numerators/denominators. Selected realized gain over all nine positions is separately saved and reproduces the failed predictor's original utility.

## Connection to C-nonlinear and attribution sensitivity

development: forward cancellation occurs in 153/900 (17.00%) of all opportunities and 153/488 (31.35%) of harmful moves. Reverse gives 151/900 (16.78%); averaged gives 151/900 (16.78%).
Saved C-nonlinear selects 423 opportunities: 178 helpful, 245 harmful and 0 tied. Of harmful selections, beneficial early addition occurs in 78/245 (31.84%) forward, 74/245 (30.20%) reverse and 75/245 (30.61%) averaged.
Across all opportunities, benefit signs change in 35/900 (3.89%), cost signs in 258/900 (28.67%), and cancellation classification in 18/900 (2.00%). Forward/reverse agree on cancellation for 143/900 (15.89%). Among selections, cancellation classification changes in 8/423 (1.89%); robust overlap is 72/423 (17.02%).

exploratory: forward cancellation occurs in 56/360 (15.56%) of all opportunities and 56/206 (27.18%) of harmful moves. Reverse gives 57/360 (15.83%); averaged gives 57/360 (15.83%).
Saved C-nonlinear selects 151 opportunities: 67 helpful, 84 harmful and 0 tied. Of harmful selections, beneficial early addition occurs in 16/84 (19.05%) forward, 17/84 (20.24%) reverse and 17/84 (20.24%) averaged.
Across all opportunities, benefit signs change in 13/360 (3.61%), cost signs in 111/360 (30.83%), and cancellation classification in 5/360 (1.39%). Forward/reverse agree on cancellation for 54/360 (15.00%). Among selections, cancellation classification changes in 3/151 (1.99%); robust overlap is 15/151 (9.93%).

These measurements show exactly which interpretation survives the attribution order; a flag change is not an encoder or recognition accuracy measurement. Rates and intervals are descriptive, conditional on the saved features/head and sampled classes. They exclude training, model-selection and repeated-research uncertainty. The two cohorts are never pooled.

## Three measured examples

Selection was frozen: largest beneficial net gain, most negative net among forward cancellations, and largest absolute interaction among sign/cancellation-sensitive rows. Ties use cohort/source/r/t ordering and distinct examples are preferred. These are descriptive extremes, not representative samples.

### beneficial_move

development; source hoEPeeBg7G0; video hoEPeeBg7G0_000098_000108; class swinging on something; t=3, r=4. Saved C-nonlinear selected: False. Rule: largest positive net gain.

|Schedule|Candidate refresh ordinals|Simulated refreshes|Whole-video CE (nats)|
|---|---|---:|---:|
|A|[0, 4, 8, 12]|4|5.507188378517|
|B|[0, 3, 4, 8, 12]|5|4.995646864635|
|C|[0, 3, 8, 12]|4|2.575489930653|
|D|[0, 8, 12]|3|4.737805345852|

Net +2.931698448; forward benefit/cost +0.511541514/-2.420156934; reverse benefit/cost +2.162315415/-0.769383033; averaged +1.336928465/-1.594769983.

### harmful_despite_beneficial_addition

exploratory; source f9R8VWDeXU4; video f9R8VWDeXU4_000049_000059; class unloading truck; t=2, r=4. Saved C-nonlinear selected: False. Rule: most negative net among forward cancellations.

|Schedule|Candidate refresh ordinals|Simulated refreshes|Whole-video CE (nats)|
|---|---|---:|---:|
|A|[0, 4, 8, 12]|4|1.455717171416|
|B|[0, 2, 4, 8, 12]|5|1.415442311613|
|C|[0, 2, 8, 12]|4|3.726887692658|
|D|[0, 8, 12]|3|3.944273866534|

Net -2.271170521; forward benefit/cost +0.040274860/+2.311445381; reverse benefit/cost +0.217386174/+2.488556695; averaged +0.128830517/+2.400001038.

### attribution_sensitive

development; source QeG2HREr6m0; video QeG2HREr6m0_000003_000013; class jumpstyle dancing; t=11, r=12. Saved C-nonlinear selected: True. Rule: largest absolute interaction among sign/cancellation-sensitive opportunities.

|Schedule|Candidate refresh ordinals|Simulated refreshes|Whole-video CE (nats)|
|---|---|---:|---:|
|A|[0, 4, 8, 12]|4|1.831556720360|
|B|[0, 4, 8, 11, 12]|5|1.401264177033|
|C|[0, 4, 8, 11]|4|1.331664053274|
|D|[0, 4, 8]|3|3.905830507938|

Net +0.499892667; forward benefit/cost +0.430292543/-0.069600124; reverse benefit/cost +2.574166455/+2.074273788; averaged +1.502229499/+1.002336832.

## Verification and limits

Independent explicit feature-copy replay checks 5040 schedules, latest-cache equality and exact temporal difference boundaries. Maximum original-target absolute discrepancy: 0; maximum defined relative discrepancy: 0.000000000000; undefined relative denominators: 0. Maximum algebra residual: 8.881784197e-16; independent-loss discrepancy: 8.881784197e-15. Sign tolerance remains 1e-12; independent replay uses the previously documented 1e-10 absolute/relative tolerances. No tolerance was loosened.

Representation quality, limited source/class support, unavailable true labels, and the predictability of future content remain unresolved. Retrospective cancellation does not prove future information is impossible to predict. Full-clip candidates and future-reference codec vectors keep this offline. There is no accuracy improvement, adaptive speedup, successful online controller or publication-ready contribution demonstrated here.

Cancellation is a real but partial explanation for these failures. Forward-order cancellation accounts for 78/245 harmful development selections and 16/84 harmful exploratory selections; the other 167/245 and 68/84 harmful selections do not have a beneficial early addition. Of all forward cancellation events, 143/153 development and 54/56 exploratory events remain cancellation events in reverse order. The binary event is therefore fairly stable, while the cost sign changes for 28.67% and 30.83% of all opportunities. Neither attribution is a unique causal explanation, and this diagnosis does not establish cancellation as the main reason the predictor failed.

Recommended next experiment: freeze a two-target predictability diagnostic for early-addition benefit and later-removal cost separately, using only permitted pre-refresh observations and source-grouped evaluation. Reconsider the replace-later action and its supervision before training another controller. Both proposed labels would remain retrospective, evaluator-only whole-video targets; this recommendation does not assume either is observable online. Keep the existing predictor, threshold and cohorts unchanged in the present experiment.

## Reproduction

```powershell
& .\.venv\Scripts\python.exe -m unittest test_refresh_target_decomposition -v
& .\.venv\Scripts\python.exe .\analyze_refresh_target_decomposition.py --recognition-input results/recognition_baseline_v2 --ablation-input results/task_signal_ablation_v2 --output results/refresh_target_decomposition_v1 --seed 20261019 --bootstrap 10000
& .\.venv\Scripts\python.exe .\analyze_refresh_target_decomposition.py --recognition-input results/recognition_baseline_v2 --ablation-input results/task_signal_ablation_v2 --output results/refresh_target_decomposition_v1 --seed 20261019 --bootstrap 10000 --audit-only
```

Completed outputs are audited without overwriting. Missing prerequisites block the run; no synthetic measurements substitute for the cached dataset.
