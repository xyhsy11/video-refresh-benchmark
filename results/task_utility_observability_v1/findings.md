# Experiment 22: reviewed exploratory findings

**Decision: SIGNAL_INFORMATION_NOT_ESTABLISHED.** Incremental task-benefit predictability from the existing 32 cheap statistics is not established beyond history. This preserves a modest positive finding for history on immediate gain, and does not claim information-theoretic impossibility.

This is DEVELOPMENT / EXPLORATORY on already exposed UCF101 data. No official-test outcomes were opened. No new X3D forwards, GPU inference, decoder calls, recognition-head fitting or adaptive-controller training were performed. Only fixed small CPU logistic/ridge probes were fitted; all timings below are genuine PREVIOUS Experiment 21 executions.

## Genuine population and targets

Inherited counts calculated from authenticated membership: {"controller_train": {"clips": 404, "groups": 202}, "selection": {"clips": 202, "groups": 101}, "execution": {"clips": 400, "groups": 200}}. All 101 classes are present. Every source belongs to original official split-1 TRAINING membership. Train/selection/execution groups and known byte/duplicate components are separated; unknown parent/pretraining relationships remain. Primary states are periodic B4 prefixes only. Training uses 4848 discretionary states, selection 2424; descriptive execution uses 4800 states from 400 clips/200 groups. Random 4683 and actual H+M4740 discretionary states are separately labeled sensitivity populations on the same 400 clips, not independent datasets. All 27090 t=1..15 states are retained; 21495 are discretionary across the complete study. Counts are computational states, not independent samples.

Target A = CE(last genuine cache feature)-CE(current genuine feature). Target B = min final C-pooled CE over feasible HOLD completions minus min CE over feasible REFRESH completions, preserving the complete recorded prefix and exact B4. Positive favors refresh. Full-information targets use labels/future dense arrays retrospectively; causal vectors never receive them. Target A is not final pooled utility; Target B is not a deployed policy or downstream treatment effect. Forced/exhausted states are retained but excluded from predictability; no gain-dependent exclusions.

All 455 schedules per clip and all conditional branches were checked against independent scalar pooling/combinations enumeration. Maximum scalar loss discrepancy 4.62e-14 nats, target discrepancy 4.97e-14. Raw 2048D pooling/head replay additionally checked all 455 schedules on 12 hash-selected real clips. Independent review reproduced 4848 original periodic training input vectors exactly and the original actual H+M teacher-confusion counts. No missing vector was reconstructed from means.

## Predictability on identical primary periodic states

Class-balanced, then group/clip/state-balanced metrics. PR AUC means weighted average precision. Classification columns below use selection-only calibration; raw scores, ROC, Brier, precision/recall, calibration bins, signed biases, error quantiles and strata are retained in machine-readable outputs.

|Target|View|PR AUC|ROC AUC|Brier|Signed-gain MAE|Gain Spearman|
|---|---|---:|---:|---:|---:|---:|
|local|constant|0.498762|0.500000|0.250152|0.195094|undefined|
|local|baseline|0.522939|0.530803|0.249466|0.195897|0.045913|
|local|H|0.568650|0.590471|0.245444|0.202170|0.137186|
|local|M|0.502062|0.506044|0.250148|0.195807|0.006065|
|local|HM|0.568831|0.590112|0.245407|0.202642|0.141514|
|budget|constant|0.407178|0.500000|0.241537|0.029014|undefined|
|budget|baseline|0.497707|0.621344|0.231170|0.028955|0.219654|
|budget|H|0.492871|0.609238|0.234462|0.028894|0.215343|
|budget|M|0.411075|0.493387|0.241718|0.029019|-0.007451|
|budget|HM|0.491409|0.608221|0.234531|0.028894|0.213349|

Positive prevalence is 49.8762% for A and 40.7178% for B. Primary execution has no exact/near-zero(abs<=1e-10) targets, but near-zero states in other roles/populations remain in all analyses. Selected threshold.25 labels all primary execution states positive for every view: recall 1, precision equal to prevalence. This threshold is a diagnostic selected by F1, not a feasible four-refresh policy.

H minus time/age/budget PR AUC: A +0.045711 [+0.026398, +0.067240]; B -0.004836 [-0.016178, +0.008498]. History therefore has modest immediate sign information, while richer history does not improve budget-sign classification over the structural baseline. A's H regression MAE is worse than the constant predictor (0.202170 versus 0.195094); classification/ranking and gain magnitudes are different questions.

H+M minus H PR AUC: A +0.000181 [-0.001292, +0.001687]; B -0.001462 [-0.005408, +0.002438]. Brier differences: A -0.000037 [-0.000143, +0.000071]; B +0.000068 [-0.000253, +0.000390]. Signed-gain MAE differences: A +0.000473 [+0.000027, +0.000903]; B +0.000001 [-0.000003, +0.000004]. Tiny/inconsistent score changes, no magnitude improvement, and M-only performance close to the constant baseline do not support added information of useful size. This conclusion does not rely only on CI crossings.

The same fixed probes transported to random states give H+M-H AP: A +0.000623 [-0.000970, +0.002121]; B +0.000486 [-0.004411, +0.006293]. On actual H+M states: A -0.000475 [-0.002217, +0.001002]; B -0.001031 [-0.005403, +0.003579]. The different populations are not pooled. Position/budget strata are descriptive; all retain 101 classes and at least 195 groups, so none meets the prespecified sparse(<20 groups) flag. Strata do not select the result.

Intervals use 2000 saved paired PCG64 class-stratified GROUP bootstrap draws. They are marginal/unadjusted, conditional on fitted/selected/calibrated probes and a reused head. They omit fitting and prior-exposure uncertainty. BenchPress/JavelinThrow have only one execution group; their within-class group variance is unestimated. No equivalence or power claim follows.

## Proxy alignment and task headroom

On primary states the Z_pool teacher agrees with A's beneficial-action sign on 51.11% and B on 56.60%. Its refresh action accompanies negative budget gain on 14.27% of weighted states. Mean budget gain conditional on teacher refresh is-0.008136 nats. The hold-label majority is 59.28%; that is a descriptive label frequency, NOT a newly executed hold policy. Conditional hindsight wrong-branch regret, confusion cells and group intervals are retained. Teacher imitation is not task-utility prediction.

Periodic final CE=0.247427; Z_pool=0.245851; true-label task oracle=0.135483; dense 16=0.244324. Perfect-hindsight task headroom is 0.111945 nats, whereas Z_pool's reduction is only 0.001576. The small fidelity-teacher advantage does NOT establish small task-scheduling headroom. The true-label optimum inspects 455 schedules and is an optimistic privileged bound, not a reachable causal-performance promise.

## Observed acquisition cost and competing explanations

Reproduced actual end-to-end medians: periodic 405.6 ms, H409.2 ms, H+M570.0 ms, periodic+unusedgate 544.7 ms. Paired median added latency is 152.2 ms versus periodic and 137.7 ms versus H. Unused gate alone adds 135.3 ms. H+M is slower on all 30 timing groups. These use 1050 measured trials from 1470 total trials, with repeats collapsed within video.

H+M median thumbnail conversion=74.7 ms, cheap statistics=12.9 ms, controller decide=20.3 ms. Observation preparation=93.5 ms ALREADY includes thumbnail/statistics; do not add nested fields twice. Medians of components do not sum to the median total. The current H+M point is dominated by periodic on the matched 30 quality/cost cohort and separately the 400-quality/30-cost summary. This is observed implementation feasibility, not an optimized deployment frontier or a replay-derived speedup.

H1 objective mismatch is supported by direct conditional proxy/task disagreement and the task-oracle versus Z_pool gap. H2 insufficient incremental information from these particular cheap statistics is consistent with all simple-probe comparisons, but information-theoretic absence is unproved. H3 failure to extract useful signal remains unresolved: history predicts local signs somewhat, but no missed cheap signal is established and the old neural learner optimized another target. H4 acquisition/controller overhead is directly observed; it cannot alone explain the original failure because recognition improvement was not established either. H5 a blanket lack of task-scheduling headroom is contradicted by the large true-label hindsight bound; limited fidelity-proxy headroom is observed. These explanations can coexist. See hypothesis_assessment.md.

## Integrity, exposure and next step

All 15 new synthetic tests plus 30 existing regression tests passed; genuine target replay and independent provenance/aggregation checks are separate evidence. Archived persistence/resume interruptions were resolved. A post-initial-analysis correction made constant Spearman undefined instead of a floating residual; all selected parameters, calibrators, thresholds and five complete prediction-vector files were independently verified bitwise unchanged. All original versions, errors/seals and partial evidence are retained. A supplemental exact-bit normalization check detected only 1.63e-14 BLAS roundoff; the disclosed 1e-12 check passes without changing observations or models. Passing tests is not scientific confirmation.

The recognition head fitted the probe-training population and selected alpha using all original development clips; its iteration cap and severe train/development loss-scale shift remain. Only one/two execution groups per class, unresolved parent/pretraining mappings, fixed periodic fitting and simple-model capacity constrain interpretation. No new independent confirmation or publication readiness is claimed.

Decision B: focus the manuscript on offline objective mismatch, conditional observability limitations and real overhead. Do not search another architecture or retune controllers on these outcomes. The next experiment is a fresh, authorized, input-only source-isolated population and separated-head/controller feasibility gate (Experiment 23A), specified in next_experiment.md. No fresh dataset acquisition, GPU evaluation or controller experiment begins in this commit.

|Claim|Evidence|Limit|
|---|---|---|
|Exact genuine task targets|target_replay_validation.json; original input/feature/trace hashes|Retrospective privileged targets, not treatment effects|
|History has modest local sign information|H-baseline paired metrics|Magnitude prediction and budget utility do not follow|
|Cheap incremental information unestablished|H+M-H and M-baseline across identical primary/secondary states|Limited linear probes and exposed population|
|Fidelity/task mismatch and nontrivial task headroom|teacher_task_alignment.json; exact 455 true-label oracle|Hindsight cannot be deployed|
|Cheap pathway currently costly|cost_feasibility.json; original actual timing rows|30 groups, previous implementation, no optimized speedup claim|
|Historical and official-test safety|historical_integrity.json; guards; verification.json|Unknown source/pretraining mappings remain|
