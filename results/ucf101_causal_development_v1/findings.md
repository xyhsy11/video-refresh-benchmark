# UCF101 causal execution: DEVELOPMENT / EXPLORATORY

H+M does not lower mean development CE relative to periodic. The conditional descriptive interval crosses zero; inconclusive even within this reused development setting.

This is actual execution engineering and exploratory decision utility. It is not an independent evaluation, a TMLR-ready result, or proof that imitating an oracle improves recognition. No official-test outcomes, arrays or videos were opened.

Main base: a0e658a1a27b976adca859724297ff65d42b9264. Actual achieved population: {"controller_train": {"clips": 404, "groups": 202}, "selection": {"clips": 202, "groups": 101}, "execution": {"clips": 400, "groups": 200}}.

The frozen101-class head was fitted on original training data and its alpha selected using all original development clips. Current controller training overlaps that head’s fitting data; both current development selection and execution have prior head-selection exposure. Controller selection groups and current execution groups are disjoint, but this does not make execution independent confirmation. The original head reached its LBFGS iteration limit; it was not refitted.

|Method|Balanced CE|Balanced top1|Ordinary clip accuracy|Dense agreement|Median measured latency, ms|
|---|---:|---:|---:|---:|---:|
|periodic|0.247427|94.31%|94.25%|97.52%|405.592|
|RandomB4_seeded|0.256159|93.56%|93.50%|97.28%|399.481|
|motion|0.275753|92.57%|92.50%|94.80%|506.304|
|history|0.257951|93.07%|93.00%|97.52%|409.217|
|history_motion|0.255400|92.82%|92.75%|96.78%|570.028|
|dense16|0.244324|93.56%|93.50%|100.00%|1385.836|
|Z_pool_OFFLINE_privileged|0.245851|93.07%|93.00%|98.27%|not assigned|

Fixed exploratory Delta_CE=CE(periodic)-CE(H+M): -0.007972 nats, conditional descriptive95% group interval [-0.017681,+0.002085]. Positive favors H+M. The saved class-stratified paired group draws condition on the reused selected head/controllers; single-group classes have no estimated within-class group variance, and selection/prior-exposure uncertainty is omitted.

Recorded actual new encoder starts/completions: 22840 / 22840. Every completed deployable trial executes exactly4 forwards and16 cached outputs; Dense16 executes16. Failed attempts and unresolved tokens are disclosed in execution_accounting.json. Native400 projection executes. C pooling uses reuse counts/16.

Timing includes fresh full decoding/actual-PTS planning, all encountered thumbnail/statistics/decision work, selected RGB/transfers, real X3D/native projection, pooling/head and CUDA-completed prediction. Disk loading/model setup are excluded. Thirty input-selected groups, two warmups and five interleaved measured repetitions per mode; repeats are not independent sources. Per-video variation and overhead/memory columns are retained. Offline Z_pool has genuine dense16 quality and no borrowed periodic latency.

H+M has higher CE than periodic on 199 execution clips. Actual discretionary teacher agreement is 0.6994; chosen locally harmful refreshes 590, missed locally helpful opportunities 1783, teacher-positive but task-harmful opportunities 735. MAD/local-gain Spearman: -0.009912613200823945. These local signed gains explain specific timing mistakes, not total causal treatment effects. The learner predicts proxy imitation, not task-benefit probability. See failure_analysis.json and retrospective_decisions.csv.

Training examples came from genuine archived training features and freshly decoded thumbnails, periodic/random chronological histories, and one common on-policy augmentation round. H and H+M share architecture, data and optimization; only32 cheap dimensions differ. Training/live age/budget and standardized-state distributions are saved to diagnose shift, not to tune after execution.

## Claim and evidence

|Claim|Evidence and limit|
|---|---|
|Controllers trained on real observations|observation_provenance.json, training_identity.json, hashed real examples and checkpoints; head/controller training overlap disclosed|
|Actually causal four-call execution|per-trial checkpoints, encoder_events.jsonl, decision traces, independent cached-logit replay; no complete future embedding matrix in live decisions|
|Quality/cost|development metrics plus actual timing trials; exploratory reused development population only|
|Failure mechanisms|Retrospective actual local gains, teacher disagreement, cheap-signal associations and state shift; descriptive, not novel causal mechanism|
|Fresh-dataset readiness|Engineering contracts/integration can pass; fresh-source validity and independently frozen head/controller evaluation remain unestablished|
|Historical preservation and forbidden test access|repository_review.json Git blob identities, explicit I/O guards, verification.json; old test outcomes not reread|

## Limits and next experiment

Preserve this result regardless of direction. Four snippets may miss discriminative moments, cheap appearance change may not track action/task gain, teacher imitation may optimize the wrong target, histories can shift, and controller/thumbnail overhead can consume savings. Frozen-feature recognition, small group counts/class, training overlap, prior UCF development exposure, incomplete parent/pretraining mappings, and full-clip offline planning prevent publication-confirmation claims. No additional architecture or favorable subset is searched.

Before a fresh evaluation, obtain authorized RGB data without accepting agreements automatically; verify original subject/source/take/camera family independence, all-class/group minima, actual temporal eligibility and duplicate/pretraining relationships; establish compute/storage using training-only real smoke. Review and freeze independently separated head/controller/selection roles, all model hashes and inference/statistical rules before any new evaluation predictions. Do not relabel UCF development as untouched test or reuse exposed official UCF test outcomes.

The framework is an executable development prototype if verification passes. Whether it is suitable for a fresh scientific comparison depends on this study’s actual quality/cost and the new dataset gate. No publication-readiness claim follows from passing tests.

Completed review: see completion_review.md for the adverse/inconclusive decision, teacher-label imbalance, measured overhead and remaining publication blockers. Supplemental failure subsets are post-hoc descriptive. Column accounting is clarified in metric_column_semantics.md.
