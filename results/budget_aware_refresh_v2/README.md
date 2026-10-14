# Budget-aware refresh v2: offline development snapshot

Executed October 14, 2026 using saved artifacts from 80 original training videos for fitting and 40 other original training videos for development. No original validation/test feature or motion artifacts enter this study. No encoder calls, decoding or GPU timing were executed.

[Findings](findings.md) preserve the mixed outcome: the motion-aware action-value controller has a small average B=4 advantage over periodic/pacing, improves twenty videos and worsens twenty, and trails age/budget alone. Both value controllers lose to periodic at B=2; pacing wins among deployable policies at B=8. Motion provides no consistent benefit across budgets.

[Repository commands](../../README.md#budget_aware_refresh_v2-oracle-action-value-supervision) reproduce the replay and focused/independent verification. The collector and all prior snapshots remain unchanged.

- `protocol.json`, `development_split.json`: frozen definitions and 80/40 class-balanced source split.
- `training_drift_pairs.csv`, `training_action_values.csv`: 9,600 drift pairs and 32,480 unique optional action states with finite oracle targets.
- `fitted_predictors.json`: two drift fits and six per-budget signed value fits, training-only population scaling and alpha=1.0.
- `threshold_grid.json`, `threshold_search.csv`, `selected_thresholds.json`: every predefined setting and per-video result; fixed zero threshold for value models.
- `development_policy_traces.csv`, `development_trajectories.csv`: all actual simulated actions, cache/budget state, privileged evaluator values and regrets, including all ten random seeds.
- `development_per_video_metrics.csv`, `development_pairwise_comparisons.csv`, `development_aggregate_metrics.json`: per-video and equal-video comparisons; negative differences retained.
- `oracle_verification.json`, `verification.json`, `independent_verification.json`: executed DP, regression, interval, information-isolation, saved-result and preservation checks.
- `metadata.json`, `SHA256SUMS.txt`: actual timestamps, source/input/environment identities and snapshot integrity.
- `error_vs_refresh_budget.png`, `per_video_B4_improvement.png`, `example_timelines.png`, `example_selection.json`: budget comparison, per-video gains/losses and predetermined descriptive success/failure examples.
- `fresh_test_protocol.json`: prospective 100-video fixed-model confirmation; not acquired/evaluated.

These are development results with settings selected on the evaluated videos. Features measure appearance fidelity, not recognition accuracy or latency. Codec future references and full-clip candidate selection keep this offline. Candidate observations and action states within a video are correlated; no significance or confidence intervals are claimed. CSV undefined values are blank, JSON missing values null; audit CSVs with round-trip float parsing.
