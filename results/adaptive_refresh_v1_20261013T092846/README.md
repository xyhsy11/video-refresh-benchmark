# adaptive_refresh_v1: reviewed offline replay snapshot

Actual replay: October 13, 2026. This snapshot uses saved reference features and per-frame codec motion descriptors from 120 v2 training videos and 40 validation videos. It contains 14,400 all-history training pairs, three fixed ridge predictors, globally validation-tuned thresholds, and 1,960 selected/control trajectories (31,360 outputs). No test videos, RGB encoder executions, downloads, decoding, or GPU latency measurements enter this experiment.

[Findings](findings.md) retain negative results: B=4 combined-policy average error only marginally improves over periodic, nineteen videos improve and twenty-one worsen, and B=8 is worse than periodic. Oracle headroom is substantial, while learned recovery is weak. These are developmental validation results, not unbiased test evidence, accuracy measurements, or speedup claims.

[Repository commands](../../README.md#adaptive_refresh_v1-offline-policy-replay-at-a-fixed-budget) reproduce the experiment from the project root using the existing environment. The requested base output and earlier fresh runs are preserved locally. [Run consistency](run_consistency.json) verifies that refinements to checks/reporting left all scientific CSVs, fitted parameters, selected thresholds, and results identical across the four executed runs.

- `protocol.json`: frozen pair/input/threshold/budget/seed/oracle/evaluation definitions, before validation results.
- `metadata.json`: actual timestamps, source/input hashes, environment, training/validation source IDs, preserved-file inventory, and original reference encoder settings.
- `train_pairs.csv`, `fitted_predictors.json`: 120 pairs per training video, training-only scalers/coefficients, alpha=1.0.
- `threshold_grid.json`, `threshold_search.csv`, `selected_thresholds.json`: training-derived grids, every threshold's forty validation results, one selected threshold per scored policy and budget.
- `policy_traces.csv`, `validation_trajectories.csv`: fresh per-trajectory cache state, actual simulated decisions, inputs/raw/clipped scores, forced refreshes, pre-decision drift and post-decision output errors.
- `validation_per_video_metrics.csv`, `validation_pairwise_comparisons.csv`, `validation_aggregate_metrics.json`: all three budgets, negative paired reductions/gap closure, and all random seeds averaged within video before aggregation.
- `oracle_schedules.json`: exact all-schedule offline bound with deterministic tie break.
- `example_selection.json`: predeclared largest-gain/largest-loss B=4 examples for description, including a failure case.
- `verification.json`, `independent_audit.json`, `final_checks.json`, `run_consistency.json`, `SHA256SUMS.txt`: executed software/real-data/saved-result/preservation/release/integrity checks.
- `error_vs_refresh_budget.png`, `per_video_B4.png`, `example_timelines.png`: simulated-budget comparisons, per-video outcomes, and success/failure trajectories.

Use round-trip float parsing when auditing saved CSVs. Missing denominators remain undefined, and missing motion remains distinct from observed zero displacement. Models and scalers never change during threshold selection or replay. Test data is not parsed; published test files are hashed only for preservation. Codec future-reference vectors and full-clip candidate selection make this offline. Thresholds were selected on these validation videos; no confidence intervals or statistical-significance claims are made.
