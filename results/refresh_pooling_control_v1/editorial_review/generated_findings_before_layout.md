# refresh_pooling_control_v1 findings

**Primary development B=4 Delta = +0.016958 [-0.019657,+0.050852] nats/video.** The interval is inconclusive:the interval includes zero;this is not equivalence. This is a frozen analysis on reused data,not independent preregistration or fresh confirmation.

|Cohort|Budget|Cache headroom H_C|Unique headroom H_U|Delta H_C-H_U|
|---|---:|---|---|---|
|development|2|+0.609866 [+0.470028,+0.756041]|+0.566766 [+0.425248,+0.716579]|+0.043100 [-0.006363,+0.092674]|
|development|4|+0.585507 [+0.455330,+0.726900]|+0.568549 [+0.437188,+0.710637]|+0.016958 [-0.019657,+0.050852]|
|development|8|+0.520575 [+0.418473,+0.628999]|+0.464590 [+0.374125,+0.560541]|+0.055986 [+0.030585,+0.083225]|
|exploratory|2|+0.534663 [+0.330303,+0.744864]|+0.520130 [+0.354690,+0.691084]|+0.014533 [-0.034483,+0.064850]|
|exploratory|4|+0.582720 [+0.371505,+0.791876]|+0.540060 [+0.366147,+0.711373]|+0.042660 [-0.016317,+0.103938]|
|exploratory|8|+0.529411 [+0.338088,+0.717626]|+0.434661 [+0.286791,+0.578011]|+0.094749 [+0.031172,+0.160192]|

Intervals are95% class-stratified whole-source percentiles. Exploratory40 and B=2/8 are predefined secondary,marginal/unadjusted. No budget,class or subgroup replaces the primary result. Positive oracle headroom follows from periodic being feasible;it is not evidence of a deployable policy.

## Same frames versus reoptimized choices

|Cohort|L_C(S_C)|L_U(S_C)|L_C(S_U)|L_U(S_U)|W_C (U-C at S_C)|W_U (U-C at S_U)|
|---|---:|---:|---:|---:|---|---|
|development|0.565185|0.714335|0.776007|0.582143|+0.149151 [+0.103270,+0.198438]|-0.193865 [-0.305527,-0.101297]|

For development,S_C/S_U mean Jaccard is 0.5730;16/100 schedules are identical. The cache-optimal fixed set has W_C +0.149151 [+0.103270,+0.198438];after reoptimizing for U,the remaining unique headroom is +0.568549 [+0.437188,+0.710637]. These are different comparisons. Changing pooling at S_C never silently substitutes S_U.
|exploratory|0.492441|0.647227|0.705893|0.535102|+0.154785 [+0.078627,+0.231023]|-0.170792 [-0.275495,-0.064243]|

For exploratory,S_C/S_U mean Jaccard is 0.5205;5/40 schedules are identical. The cache-optimal fixed set has W_C +0.154785 [+0.078627,+0.231023];after reoptimizing for U,the remaining unique headroom is +0.540060 [+0.366147,+0.711373]. These are different comparisons. Changing pooling at S_C never silently substitutes S_U.

H_C=[L_U(P)-L_U(S_C)]+W_C is verified per source. The first term uses S_C under U;it is not H_U unless that schedule is also U-optimal. This is a path-dependent algebraic decomposition,not independently identified causal contributions or a percentage of semantic importance. Duration weights count candidate outputs,not seconds.

## B=4 recognition and feature-fidelity control

|Cohort|Selected schedule|Pool|CE nats/video|Top-1 / macro (%)|Last-cache cosine error|Effective weight count|
|---|---|---|---:|---:|---:|---:|
|development|Periodic|C|1.150692|70.00 / 70.00|0.092942|4.000|
|development|Periodic|U|1.150692|70.00 / 70.00|0.092942|4.000|
|development|Feature oracle (dense-feature hindsight)|C|1.168484|70.00 / 70.00|0.069075|3.125|
|development|Feature oracle (dense-feature hindsight)|U|1.200515|70.00 / 70.00|0.069075|4.000|
|development|Cache task oracle (true-label hindsight)|C|0.565185|86.00 / 86.00|0.108693|2.504|
|development|Cache task oracle (true-label hindsight)|U|0.714335|80.00 / 80.00|0.108693|4.000|
|development|Unique task oracle (true-label hindsight)|C|0.776007|81.00 / 81.00|0.110115|2.654|
|development|Unique task oracle (true-label hindsight)|U|0.582143|83.00 / 83.00|0.110115|4.000|
|development|RandomExact (expected)|C|1.266654|65.22 / 65.22|0.103289|2.872|
|development|RandomExact (expected)|U|1.215206|67.03 / 67.03|0.103289|4.000|
|development|Dense16 reference|C=U|1.096979|72.00 / 72.00|0.000000|16.000|

For development,feature oracle reduces last-cache cosine error by 0.023867,while its CE relative to periodic changes by +0.017792 under C and +0.049823 under U. Its accuracy is 70.00%/70.00% (C/U),versus 70.00% periodic. The original cache-pool feature-fidelity/task-quality mismatch reproduces;unique pooling is a separate frozen-head aggregation comparison.
|exploratory|Periodic|C|1.075162|65.00 / 65.00|0.095736|4.000|
|exploratory|Periodic|U|1.075162|65.00 / 65.00|0.095736|4.000|
|exploratory|Feature oracle (dense-feature hindsight)|C|1.271517|65.00 / 65.00|0.070790|3.137|
|exploratory|Feature oracle (dense-feature hindsight)|U|1.155459|60.00 / 60.00|0.070790|4.000|
|exploratory|Cache task oracle (true-label hindsight)|C|0.492441|87.50 / 87.50|0.119230|2.646|
|exploratory|Cache task oracle (true-label hindsight)|U|0.647227|80.00 / 80.00|0.119230|4.000|
|exploratory|Unique task oracle (true-label hindsight)|C|0.705893|77.50 / 77.50|0.117152|2.735|
|exploratory|Unique task oracle (true-label hindsight)|U|0.535102|87.50 / 87.50|0.117152|4.000|
|exploratory|RandomExact (expected)|C|1.277207|61.79 / 61.79|0.105960|2.872|
|exploratory|RandomExact (expected)|U|1.243675|63.19 / 63.19|0.105960|4.000|
|exploratory|Dense16 reference|C=U|1.112254|65.00 / 65.00|0.000000|16.000|

For exploratory,feature oracle reduces last-cache cosine error by 0.024946,while its CE relative to periodic changes by +0.196355 under C and +0.080297 under U. Its accuracy is 65.00%/60.00% (C/U),versus 65.00% periodic. The original cache-pool feature-fidelity/task-quality mismatch reproduces;unique pooling is a separate frozen-head aggregation comparison.

RandomExact rows are expectations over every feasible individual schedule,not an ensemble. Its expected correct counts and entire outcome distributions are saved;accuracy is not inferred from averaged logits or probabilities. Feature error always describes the hypothetical16-output latest-cache timeline;it is not a reconstruction metric for U. Effective count is1/sum(w^2),a descriptive weight-concentration statistic,not effective sample size or encoder cost. Unique pooling produces one final classification from B unique features,not16 time-indexed predictions.

## Verification and scope

The primary400-training-video head,regularization0.01,class order,means,scales,weights and bias stay frozen. Exactly100 development and40 previously inspected exploratory sources are used,with16 float32 references each,cast to float64. Training arrays,protected confirmation outcomes/reference arrays/motion tables are not opened. Archived primary-head B=4 dense,periodic,feature-oracle and cache-task-oracle schedules/logits/losses/accuracy reproduce. Every selected schedule is independently cache-copied and unique-feature averaged;all966,700 feasible source/schedule outcomes are independently checked. Synthetic scalar exhaustive tests cover both minima,near ties,RandomExact,affine identities and the prediction-flip fixture. Saved per-source statistics and bootstrap endpoints are independently reproduced. See verification.json.

The chosen oracle is lexicographically first within1e-12 of the numerical minimum;minimum,achieved objective and tie count are all retained. CE minimization is label-guided hindsight and does not guarantee higher top-1 accuracy. Any nonnegative headroom/own-objective regret or exact affine identity is a computation check,not controller evidence.

The head was trained on dense mean features. Sparse C/U aggregations can create different representation distributions. These results isolate pooling/selection under that fixed head,not separately retrained best-case heads. Ground-truth labels and future reference candidates make both task oracles privileged/noncausal;the feature oracle also has dense references. No temporal localization evidence establishes semantic frame importance.

Bootstrap uses10,000 PCG64 seed20261023 resamples,linear percentiles,within each fixed class5 sources for development and2 for exploratory,with shared indices across budgets/pools/contrasts. Sources—not schedules or candidates—are independent resampling units. Cohorts stay separate. Intervals condition on the frozen head and chosen classes,exclude training/model-selection/repeated-research uncertainty,and are particularly limited by two exploratory sources/class. No gap ratios,post-hoc effect threshold or publication gate are used.

## Supported next experiment

Run the separately frozen temporal-backbone benchmark specified in next_temporal_backbone_protocol.md before another cached-feature controller variant. It uses one efficient temporal representation,valid complete snippets and matched snippet-call budgets,with pooling controls and genuinely new train/development/evaluation sources. No acquisition,backbone execution or fitting is launched here. The present experiment establishes no fresh generalization,streaming causality,successful adaptive controller,end-to-end speedup or publication acceptance.

## Reproduction

From the repository root:

```powershell
& .\.venv\Scripts\python.exe -m unittest test_refresh_pooling_control -v
& .\.venv\Scripts\python.exe .\analyze_refresh_pooling_control.py --input results/recognition_baseline_v2 --output results/refresh_pooling_control_v1 --budgets 2 4 8 --bootstrap 10000 --seed 20261023 --preflight-only
& .\.venv\Scripts\python.exe .\analyze_refresh_pooling_control.py --input results/recognition_baseline_v2 --output results/refresh_pooling_control_v1 --budgets 2 4 8 --bootstrap 10000 --seed 20261023
# Inspect all four rendered figures,then explicitly finalize the visual review:
& .\.venv\Scripts\python.exe .\analyze_refresh_pooling_control.py --input results/recognition_baseline_v2 --output results/refresh_pooling_control_v1 --budgets 2 4 8 --bootstrap 10000 --seed 20261023 --finalize-reviewed
& .\.venv\Scripts\python.exe .\analyze_refresh_pooling_control.py --input results/recognition_baseline_v2 --output results/refresh_pooling_control_v1 --budgets 2 4 8 --bootstrap 10000 --seed 20261023 --audit-only
```

Completed invocations perform read-only audits;they do not rerun or overwrite scientific measurements. Changed inputs/protocol require a fresh versioned directory. Analysis runtime is an offline CPU analysis cost,not inference latency. Actual new downloads,decodes,encoder forwards,model/scaler/threshold fits and package upgrades are zero.
