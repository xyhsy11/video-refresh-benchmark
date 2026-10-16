# refresh_generalization_audit_v1 findings

This bounded independent audit retains the frozen head,models,targets,folds and1e-12 decision threshold. It evaluates nine independent single-refresh moves per source,not sequential controller rollouts. Both cohorts are developmental and previously inspected. No genuinely independent test was run.

## What passed

All540 reference-array hashes/source identities and training-only recognition normalization pass. Recognition regularization selection reproduces0.01 from development dense accuracy/loss. Dense accuracy reproduces72% development and65% exploratory;periodic70%/65%. All1260 loss/cache counterfactuals reproduce independently (maximum target discrepancy 1.07e-14). Saved forests,scalers,source-held-out membership and selections are independently checked;see verification.json and model_checks.json. No verified implementation leakage was found in the checked paths.

## Fair single-move decisions

|Cohort|Method|Utility and95% interval (nats/opportunity)|Selected / total|Helpful / harmful / neutral|Harmful rate (denominator)|Missed benefit (denominator)|MAE vs training-median|
|---|---|---|---:|---|---|---|---|
|development_oof|Keep periodic|+0.000000 [+0.000000,+0.000000]|0/900|0/0/0|undefined:no selections|100.000% (412/412)|not a regression method|
|development_oof|Random single move (expected)|-0.020711 [-0.043853,+0.002944]|450/900 expected|206/244/0|54.222% (244/450)|50.000% (206/412)|not a regression method|
|development_oof|Motion-median single move|-0.013943 [-0.047427,+0.024048]|449/900|188/261/0|58.129% (261/449)|54.369% (224/412)|not a regression method|
|development_oof|Direct RF400|-0.025125 [-0.042323,-0.009137]|285/900|107/178/0|62.456% (178/285)|74.029% (305/412)|0.280248 vs 0.253471|
|development_oof|Forward two-target RF200x2|-0.011775 [-0.029779,+0.005975]|284/900|122/162/0|57.042% (162/284)|70.388% (290/412)|0.281674 vs 0.253471|
|development_oof|Cached-history RF400|-0.011949 [-0.028049,+0.003995]|275/900|116/159/0|57.818% (159/275)|71.845% (296/412)|0.278711 vs 0.253471|
|exploratory|Keep periodic|+0.000000 [+0.000000,+0.000000]|0/360|0/0/0|undefined:no selections|100.000% (154/154)|not a regression method|
|exploratory|Random single move (expected)|-0.053690 [-0.135467,+0.005692]|180/360 expected|77/103/0|57.222% (103/180)|50.000% (77/154)|not a regression method|
|exploratory|Motion-median single move|-0.074634 [-0.233279,+0.028311]|170/360|79/91/0|53.529% (91/170)|48.701% (75/154)|not a regression method|
|exploratory|Direct RF400|-0.064527 [-0.202759,+0.018178]|121/360|54/67/0|55.372% (67/121)|64.935% (100/154)|0.309259 vs 0.295157|
|exploratory|Forward two-target RF200x2|-0.078571 [-0.223520,+0.007901]|108/360|48/60/0|55.556% (60/108)|68.831% (106/154)|0.315446 vs 0.295157|
|exploratory|Cached-history RF400|-0.059949 [-0.196049,+0.020383]|110/360|44/66/0|60.000% (66/110)|71.429% (110/154)|0.315159 vs 0.295157|

Random accepts independently with probability0.5. Counts and utility are analytical expectations;its interval resamples source heterogeneity of expected utility,not Bernoulli decision noise. The motion baseline is training-available p90 median,strictly exceeded;missing=>reject. Neither baseline is a complete Random Refresh/Motion Threshold controller. All A-versus-C moves retain4 simulated refreshes and16 outputs. Four encoder calls do not imply equal total runtime.

## Training and generalization

|Phase|Method|Unique sources|Utility|MAE|
|---|---|---:|---:|---:|
|development OOF|Direct RF400|100|-0.025125|0.280248|
|development OOF|Cached-history RF400|100|-0.011949|0.278711|
|development OOF|Forward two-target RF200x2|100|-0.011775|0.281674|
|exploratory held out|Direct RF400|40|-0.064527|0.309259|
|exploratory held out|Cached-history RF400|40|-0.059949|0.315159|
|exploratory held out|Forward two-target RF200x2|40|-0.078571|0.315446|
|final-model in-sample|Direct RF400|100|+0.055012|0.238753|
|final-model in-sample|Cached-history RF400|100|+0.057635|0.237123|
|final-model in-sample|Forward two-target RF200x2|100|+0.054796|0.235421|
|fold-training in-sample (four models/source averaged)|Direct RF400|100|+0.057262|0.234336|
|fold-training in-sample (four models/source averaged)|Cached-history RF400|100|+0.059941|0.235804|
|fold-training in-sample (four models/source averaged)|Forward two-target RF200x2|100|+0.058127|0.233331|

Every frozen forest is better in sample than on its held-out sources in utility and MAE. This supports an overfitting/generalization concern for these tested representations/models. In-sample performance is not a new method result. It does not identify distribution shift,prove sufficient input information,or establish inherent unpredictability.

No audited nontrivial method establishes positive utility over keeping periodic on these reused cohorts.
A relative improvement over another harmful method is not useful positive utility. No regression metrics are assigned to decision-only methods. AUROC is pooled descriptive discrimination with explicit event counts;constant scores/single-class outcomes are missing with reasons. Per-source distributions and per-class results are descriptive;no favorable subgroup is promoted to confirmation.

## What failed or remains unverifiable

Untouched full-pipeline independence is absent. Recognition-head regularization used all100 development videos;controller OOF does not undo that selection. Exploratory outcomes were already inspected. Near-duplicate/reuploaded content under different IDs,unrecorded human outcome access and annotation correctness remain unresolved. Exact source IDs,windows and encoded hashes rule out only their recorded forms of overlap. These are design/provenance limits,not evidence of an accidental code leak. No protected confirmation outcomes are accessed. See leakage_audit.md and cohort_roles.csv.

## Computation

Actual RGB encoder executions,decodes,downloads,recognition fits and new predictor fits are zero. Sparse calls are simulated. New cpu_component_timings.csv measures only preloaded scaling/saved sklearn batch1 prediction/decision or motion threshold work,two warmups/five interleaved passes,fixed threads. It excludes observation construction,history reconstruction,codec extraction,decoding,RGB and recognition. The two-target model requires two200-tree predictions;equal total trees do not imply equal cost. Model nodes/serialized sizes are in model_checks.json. Existing history state payload904bytes and history-construction timing are historical component evidence,not measurements to add to these timings. Full-pipeline latency and adaptive speedup remain unmeasured.

## One next experiment

Design a separately preregistered temporal-recognition backbone/task benchmark with stronger temporal representation and genuinely new source videos before another cached-feature controller variant. First establish an adequate recognition/task objective and freeze source exclusions and comparisons. Do not launch an architecture search or acquisition in this audit. A frozen-method fresh-test protocol is provided for later use,without claiming these methods earned confirmation.

All intervals use10,000 paired whole-source resamples,PCG64 seed20261022,linear percentiles,shared method indices within cohort. All nine opportunities stay together;cohorts are separate. Intervals condition on already fitted models and reused cohorts,excluding full training/model-selection/repeated-research uncertainty;paired secondary intervals are marginal/unadjusted. This audit establishes no recognition-accuracy improvement,streaming causality,successful adaptive controller or end-to-end speedup.

## Reproduce from the repository root

```powershell
& .\.venv\Scripts\python.exe -m unittest test_refresh_generalization_audit -v
& .\.venv\Scripts\python.exe .\audit_refresh_generalization.py --output results/refresh_generalization_audit_v1_rerun1 --seed 20261022 --bootstrap 10000
& .\.venv\Scripts\python.exe .\audit_refresh_generalization.py --output results/refresh_generalization_audit_v1_rerun1 --seed 20261022 --bootstrap 10000 --audit-only
```

Completed runs audit without changing results or retiming. Use a versioned fresh directory if inputs/protocol change.

## Additional independent review and preserved correction

The supplementary input reviewer independently reconstructs all 1,260 C rows from the frozen head, legitimate cached/past scores and 140 hash-checked motion CSVs. Maximum discrepancy is 7.1054e-15. It verifies 420 archived per-source utilities/counters and all 12 new summary rows with explicit scalar accumulation, including expected random counts and undefined denominators. Recorded motion data contain 21,248 B-frames, 30,297,428 future-reference vector entries and 341 frames without motion; these support the stated offline limitation. The 320 newly added recognition-training sources exclude all prior reserved identities/hashes; 80 previous exclusively-training sources were reused. See independent_review.json.

The first audit attempt's final CSV roundtrip assertion failed on None versus NaN for undefined-reason blanks. Its prematurely written complete marker did not represent a successful final command. That attempt, sources and measurements remain preserved locally. This version explicitly verifies identical null masks and all defined values without changing numerical tolerances, then marks completion only after the final audit. The six scientific CSV/JSON outputs listed in verification_correction/correction.json are byte-identical across attempts. No model or scientific setting changed; CPU timings were remeasured.

New component timing medians are 41.348 ms/opportunity for direct RF400, 40.827 ms for two-target (two 200-tree predictions), 40.981 ms for history RF400, and 0.327 microseconds for the motion threshold rule. All input rows are preloaded and observation/history construction is excluded. Five measured passes do not establish equal runtime; their ranges overlap, and no end-to-end latency or speedup is inferred. The prior pilot's GPU feature-extraction measurements and Experiment 13's history-construction/state accounting remain separate historical evidence.

Every learned-versus-random and learned-versus-motion paired utility interval includes zero in both cohorts. For cached history versus random, the development difference is +0.008762 [-0.012251,+0.030562] nats/opportunity and exploratory is -0.006259 [-0.079156,+0.047120]. Against the motion-median decision, those differences are +0.001994 [-0.028627,+0.029984] and +0.014685 [-0.025460,+0.061651]. These are predefined secondary, marginal/unadjusted comparisons; none establishes an advantage over the simple single-move baselines.
