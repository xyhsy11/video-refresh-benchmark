# Task-aware oracle feasibility v1

Decision: **CONTINUE**. This is a development / separately identified exploratory hindsight study, not an independent confirmation or deployable accuracy result.

## Classifier quality and data

Eighty original-training videos (four/class) fit the linear head and training-only population normalization. Forty development videos (two/class) select one of four predefined regularization values; forty previously used original-test videos are evaluated only after freezing. Original validation arrays and the 100-video confirmation cohort are unused. Official CVDF annotation identity and labels are checked. The class-index order is the original frozen twenty-class list.

Selected lambda=0.1; CPU float64 deterministic cross-entropy training; mean pooling; frozen ResNet18 references. Chance and majority-class accuracy are 5%. Training: 100.00% top1, CE 0.0933. Development: 50.00% top1, CE 1.5230. Exploratory test: 52.50% top1, CE 1.4583. Macro equals top1 here because classes have equal counts. Class-wise values and confusion counts are saved.

Predefined exploratory quality gate (dense >=50% on both evaluation cohorts): passed. Performance exceeds chance, but only two evaluation videos per class and eighty training clips limit reliability; this is not a validated action-recognition system. Training minus exploratory accuracy is 47.50 percentage points. This large gap and classes with no correct dense predictions call for a stronger recognition baseline before publication or claims about useful video understanding.

## B=4 recognition and feature fidelity

| Cohort | Method | Top1 / macro (%) | CE (nats) | Feature cosine error | Disagreement vs dense (%) |
|---|---|---:|---:|---:|---:|
|development|Dense|50.00|1.522989|0.000000|0.00|
|development|Periodic|45.00|1.491844|0.098383|7.50|
|development|Random (10 seeds)|46.25|1.618684|0.107923|19.00|
|development|Frozen age/budget|50.00|1.546837|0.095387|10.00|
|development|Frozen age/motion/budget|50.00|1.588144|0.096393|10.00|
|development|Feature oracle (hindsight)|45.00|1.524746|0.067008|12.50|
|development|Task oracle (label + hindsight)|70.00|1.024161|0.109831|25.00|
|exploratory_test|Dense|52.50|1.458322|0.000000|0.00|
|exploratory_test|Periodic|52.50|1.446087|0.095736|17.50|
|exploratory_test|Random (10 seeds)|51.25|1.549083|0.106250|16.50|
|exploratory_test|Frozen age/budget|52.50|1.486941|0.096839|17.50|
|exploratory_test|Frozen age/motion/budget|55.00|1.530569|0.094659|15.00|
|exploratory_test|Feature oracle (hindsight)|52.50|1.526141|0.070790|2.50|
|exploratory_test|Task oracle (label + hindsight)|67.50|0.994350|0.114641|25.00|

## Paired primary hindsight headroom

Positive gain favors the task oracle. Cross-entropy gains use periodic minus oracle loss; correctness gains use oracle minus periodic. Bootstrap uses 10,000 whole-source-video paired resamples, PCG64 seed20261015, linear percentiles. Models remain fixed. Frames, candidate schedules and random seeds are not independent sampling units. Intervals are conditional on the selected head/classes and do not account for model selection or small training-sample uncertainty.

|Cohort|Outcome|Mean gain|95% interval|Improved/tied/worsened|
|---|---|---:|---|---|
|development|cross_entropy|+0.467683 nats|[+0.369362,+0.572066]|40/0/0|
|development|correct|+25.000000 pp|[+12.500000,+40.000000]|10/30/0|
|exploratory_test|cross_entropy|+0.451737 nats|[+0.356726,+0.549307]|40/0/0|
|exploratory_test|correct|+15.000000 pp|[+5.000000,+27.500000]|6/34/0|

The oracle evaluates every valid schedule using the evaluation video's true label and future candidate features: 15/455/6,435 schedules for B2/4/8. It has a large selection advantage. Since periodic is among these schedules, its minimum CE cannot be worse by construction. Positive loss intervals quantify this privileged opportunity, not predictive generalization. CE optimization need not maximize top1 correctness. All budgets are exact. Dense is a separate B16 reference, not necessarily the task-loss optimum across sparse budgets.

## Research questions and recommendation

A. Task-aware versus periodic: development top1 difference +25.00pp; exploratory +15.00pp. Exploratory CE headroom +0.451737 nats. These are privileged results; see paired intervals above.

B. Feature oracle: exploratory B4 top1 52.50%, versus task oracle 67.50%; CE 1.526141 versus 0.994350. Feature fidelity and downstream loss are distinct objectives.

C. Exploratory periodic B4 feature error versus CE change from dense: Spearman -0.08968105065666042. All within-policy/budget correlations are saved, avoiding a pooled cross-budget association. This small descriptive association does not validate cosine error as a task surrogate. There are correct classifications with large feature errors and changed classifications with relatively small errors; exact counts/examples are in analysis_examples.json. Oracles choose identical B4 schedules for 0/40 exploratory videos; mean shared-refresh fraction 0.394. Candidate margins cannot establish action-relevant temporal localization without independent annotations.

D. Recommendation **CONTINUE** under the frozen gate. Investigate a causal task-aware controller in a separately frozen development study, then confirm on genuinely new sources and measure recognition quality and actual latency separately.

E. Classifier reliability remains limited: twenty classes, four training clips/class, mean-pooled ImageNet appearance features without a trained temporal backbone. Development hyperparameters were selected on reported development videos. The reused original test is exploratory, not pristine. The same existing controller coefficients are reused, not task-trained; absence of their gains would not test all possible causal controllers.

## Limits

Zero actual ResNet18 encoder executions and zero video decodes occurred. Head training and saved-feature replay are actual CPU work; simulated B refreshes are not measured GPU calls or latency/speedup. Every replay produces sixteen outputs using the latest cache. Motion controllers consume only current/past decoded positions, but codec vectors may reference future frames; full-clip candidate selection also makes this offline. Task oracle uses true labels/future RGB and is unavailable at inference. Neither improved feature fidelity nor this weak diagnostic establishes real action-recognition preservation in a deployed system. No 100-video confirmation feature/motion/accuracy artifact was used, and no historical experiment was rewritten.

## Reproduction

Run from the repository root using the existing environment:

```powershell
& .\.venv\Scripts\python.exe -m unittest test_task_aware_oracle -v
& .\.venv\Scripts\python.exe .\run_task_aware_oracle.py --input results/cache_drift_v2 --frozen-run results/budget_aware_refresh_v2 --output results/task_aware_oracle_v1 --seed 20261015
& .\.venv\Scripts\python.exe .\run_task_aware_oracle.py --input results/cache_drift_v2 --output results/task_aware_oracle_v1 --audit-only
& .\.venv\Scripts\python.exe .\validate_release.py
```

Recollection is unnecessary. Full refitting checks cached data/source/cache_drift_v2/val.csv against its original hash; --annotations selects another copy. Saved-result auditing needs only the published feature/motion artifacts and saved head, not encoded videos or the annotation download. Existing output directories are preserved and a fresh run directory is chosen. Parameters and training configuration are classifier_parameters.json and model_configuration.json. Exact source and input identities are in metadata.json; checksums cover this snapshot.
