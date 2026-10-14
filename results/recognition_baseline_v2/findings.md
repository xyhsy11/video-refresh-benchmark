# recognition_baseline_v2 findings

This is a custom CVDF Kinetics validation-source diagnostic, not an official full-dataset benchmark or independent confirmation. The official training archive list was available, but no training shards were cached; the cached validation route was fixed before training for bounded acquisition. No completed 100-video fresh-confirmation feature or policy data was used. Its source IDs/file hashes were excluded.

## Learning curve

The 400-video classifier is primary by protocol, not selected by oracle gains. All sizes use class-balanced nested subsets of one 400-video pool and the same 100-video development cohort. The new 80-video subset shares only11 sources with the old 80-video v1 cohort;69 are different. Original custom validation/test sources never enter training. Existing 40 development sources remain, with 60 new additions; existing 40 exploratory sources remain separate.

|Training videos|Cohort|Dense top1/macro (%)|CE (nats)|
|---:|---|---:|---:|
|80|training|100.00|0.600302|
|80|development|61.00|1.824665|
|200|training|98.00|0.870206|
|200|development|70.00|1.605690|
|400|training|100.00|0.044047|
|400|development|72.00|1.096979|
|80|exploratory|45.00|1.845393|
|200|exploratory|60.00|1.597041|
|400|exploratory|65.00|1.112254|

Normalization and every fit use training-subset data only. Four unchanged regularization values (.001,.01,.1,1) use the v1 deterministic float64 linear cross-entropy/LBFGS recipe. Development dense top1, then lower loss, then grid order select lambda. All grids and frozen scalers/weights are saved; no model is refitted on development. Every selected model freezes before exploratory evaluation.

## Primary 400-video classifier

|Cohort|Policy|Correct/videos|Top1/macro (%)|CE (nats)|Feature cosine error|
|---|---|---|---:|---:|---:|
|development|Dense B16|72/100|72.00|1.096979|0.000000|
|development|Feature oracle B4 (hindsight)|70/100|70.00|1.168484|0.069075|
|development|Periodic B4|70/100|70.00|1.150692|0.092942|
|development|Task oracle B4 (label + hindsight)|86/100|86.00|0.565185|0.108693|
|exploratory|Dense B16|26/40|65.00|1.112254|0.000000|
|exploratory|Feature oracle B4 (hindsight)|26/40|65.00|1.271517|0.070790|
|exploratory|Periodic B4|26/40|65.00|1.075162|0.095736|
|exploratory|Task oracle B4 (label + hindsight)|35/40|87.50|0.492441|0.119230|

## Paired whole-video uncertainty

Positive accuracy gains are candidate minus baseline; positive loss reductions are baseline minus candidate.10,000 whole-source-video paired resamples use PCG64 seed 20261016 and linear percentiles. One saved index identity is shared across all compared policies within each cohort. Classifier parameters remain fixed. Intervals condition on these fitted models and sampled classes; training/subset/regularization-selection uncertainty is excluded. Development is tuned; exploratory sources were previously evaluated. Secondary intervals are marginal/unadjusted.

|Cohort|Candidate vs baseline|Outcome|Mean gain|95% interval|Improved/tied/worsened|
|---|---|---|---:|---|---|
|development|task_oracle vs periodic|correct|+16.000000 pp|[+9.000000,+23.000000]|16/84/0|
|development|task_oracle vs periodic|cross_entropy|+0.585507 nats|[+0.440676,+0.747974]|100/0/0|
|development|feature_oracle vs periodic|correct|+0.000000 pp|[-7.000000,+7.000000]|6/88/6|
|development|feature_oracle vs periodic|cross_entropy|-0.017792 nats|[-0.112377,+0.079607]|45/1/54|
|development|task_oracle vs feature_oracle|correct|+16.000000 pp|[+9.000000,+24.000000]|16/84/0|
|development|task_oracle vs feature_oracle|cross_entropy|+0.603299 nats|[+0.455949,+0.763719]|99/1/0|
|development|periodic vs dense|correct|-2.000000 pp|[-8.000000,+4.000000]|4/90/6|
|development|periodic vs dense|cross_entropy|-0.053713 nats|[-0.132247,+0.019328]|58/0/42|
|exploratory|task_oracle vs periodic|correct|+22.500000 pp|[+10.000000,+35.000000]|9/31/0|
|exploratory|task_oracle vs periodic|cross_entropy|+0.582720 nats|[+0.331209,+0.885879]|39/1/0|
|exploratory|feature_oracle vs periodic|correct|+0.000000 pp|[+0.000000,+0.000000]|0/40/0|
|exploratory|feature_oracle vs periodic|cross_entropy|-0.196355 nats|[-0.464951,-0.000118]|14/0/26|
|exploratory|task_oracle vs feature_oracle|correct|+22.500000 pp|[+10.000000,+35.000000]|9/31/0|
|exploratory|task_oracle vs feature_oracle|cross_entropy|+0.779075 nats|[+0.459254,+1.158128]|40/0/0|
|exploratory|periodic vs dense|correct|+0.000000 pp|[+0.000000,+0.000000]|0/40/0|
|exploratory|periodic vs dense|cross_entropy|+0.037092 nats|[-0.139866,+0.261174]|22/0/18|

## Research interpretation

1. **Additional training improves dense recognition on these matched cohorts.** Development accuracy rises from 61% to 72%; exploratory accuracy rises from 45% to 65%. These comparisons use the new nested training subsets and the same evaluation sources. The new 80-video subset shares only 11 sources with v1, so an unmatched old/new comparison cannot isolate the effect of training size.

2. **The generalization gap narrows but remains substantial.** The training/development gap changes from 39 to 28 percentage points. The training/exploratory gap changes from 55 to 35 points. The primary model has 100% training accuracy, 72% development accuracy and 65% exploratory accuracy.

3. **Task-oracle headroom persists with the primary classifier.** Against periodic B=4, development accuracy improves by 16 percentage points [9, 23], with loss reduction 0.585507 nats. Exploratory accuracy improves by 22.5 points [10, 35], with loss reduction 0.582720 nats. This is a true-label, future-feature hindsight comparison. Its loss cannot exceed periodic loss because periodic is one of the 455 searched schedules. It optimizes cross entropy, not accuracy; it is neither deployable accuracy nor an accuracy-maximizing upper bound.

4. **Improved feature fidelity does not improve recognition here.** The feature oracle lowers mean cosine error relative to periodic by 0.023867 on development and 0.024946 on exploratory, while accuracy remains unchanged. Cross entropy increases by 0.017792 and 0.196355 nats, respectively. The task oracle has higher feature cosine error but lower classification loss. These are distinct objectives.

5. **Temporal reweighting can explain part of the oracle advantage.** Caching repeats a refreshed feature for several outputs. With a linear mean-pooled head, this exactly changes the temporal weights of candidate logits. A true-label oracle can overweight class-favorable appearance and suppress ambiguous frames. These results do not demonstrate causal action-segment selection or semantic importance; no temporal localization annotation supports that claim.

6. **The next step should test label-free temporal selection or reweighting.** Use development sources to test whether current/past signals can predict task value, with a simple temporal-selection/reweighting baseline. Freeze the method before a genuinely new confirmation. The measured privileged headroom motivates that feasibility study but does not establish that a causal controller can recover it. No arbitrary accuracy cutoff is used as a publication gate.

7. **Investigate the appearance representation before making stronger recognition claims.** Static ImageNet ResNet18 features with temporal mean pooling ignore action dynamics and may classify context rather than motion. Inspect class-wise errors and convergence; a stronger pretrained temporal action representation belongs in a separate study. No new representation or recognition model was downloaded here.

The original encodings, sixteen uniform decoded candidates, existing ResNet18 weights/preprocessing, and float32 batch-size-one CUDA reference extraction remain fixed. Every sparse replay includes candidate 0, exactly four refreshes, and sixteen latest-cache outputs. Replay executes zero encoders. Reference-collection elapsed time is not adaptive inference latency. Full-clip candidate selection and codec future references keep this offline; no streaming causality, adaptive speedup, or publication-level novelty is established.

## Reproduction

```powershell
& .\.venv\Scripts\python.exe -m unittest test_recognition_baseline_v2 -v
& .\.venv\Scripts\python.exe .\run_recognition_baseline_v2.py --preflight-only --max-download-gib 20
& .\.venv\Scripts\python.exe .\run_recognition_baseline_v2.py --resume --max-download-gib 20 --download-workers 8
& .\.venv\Scripts\python.exe .\run_recognition_baseline_v2.py --audit-only
& .\.venv\Scripts\python.exe .\validate_release.py
```

Protocol, source hashes, dataset identities, exclusions, all acquisition retries/body-byte accounting, frozen models and verification accompany this snapshot. Cached archives do not charge the new-download budget; all new HTTP body reads and conservative interrupted reservations do. Existing completed outputs are audited rather than overwritten.

## Execution accounting

Acquisition and collection completed with 400 training, 100 development and 40 exploratory sources. 380 new clips yielded 6,080 CUDA encoder forwards and retained references, with zero extra/failed forwards; 2,560 old references were reused. All 540 source IDs and encoded hashes are distinct. Five newly acquired validation archives plus the counted training-list probe transferred 7,521,016,370 HTTP body bytes (about 7.0045GiB), under 20 GiB, with zero conservative interruption charges. Saved-feature replay executed zero encoders. Independent cohort/label/byte-accounting verification and full prediction/oracle/bootstrap reproduction passed. Four figures were visually inspected.
