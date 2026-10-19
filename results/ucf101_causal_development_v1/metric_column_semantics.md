# Measurement and accounting semantics

This study is DEVELOPMENT / EXPLORATORY. No official-test outcomes are used.

For the five actual B4 methods and actual Dense16, per-clip encoder starts/completions refer to the corresponding executed trial. Every completed B4 trial has four actual X3D forwards and sixteen last-cache outputs. Dense16 has sixteen actual forwards.

The separate `Z_pool_OFFLINE_privileged` quality row reuses that clip's actual Dense16 features. Its sixteen-count fields describe the **required dense-information dependency**, not an additional set of executed calls and not a four-call deployment. There is no live Z_pool checkpoint or timing trial. Its latency is unassigned. Do not sum privileged-reference rows into execution totals; use `execution_accounting.json` and actual `encoder_events.jsonl` for observed starts/completions, including smoke and warmups.

Quality-execution seconds are single actual passes. The matched benchmark is `timing_trials.csv` and `timing_per_video.csv`: two warmups and five interleaved measurements per method/video, thirty independent operational groups from thirty classes. Repeats are not additional independent sources. Complete logits are checked against saved development results after timing. Disk loading, model loading and artifact export are outside the encoded-memory-to-completed-prediction boundary.

`observation_preparation_seconds` includes thumbnail conversion, cheap statistics and history packing. Its thumbnail/statistics components overlap and must not be summed with it. `controller_decide_seconds` is a separate measured decision block. The total includes all actual processing, including common trace bookkeeping. Periodic runs the common causal loop without cheap observations; it is not a separately hand-optimized minimal periodic implementation. CPU feature readback and per-forward CUDA completion are included in all modes.

CUDA memory is torch allocated resident/peak and incremental peak, not total driver-reserved GPU memory. CPU before/after working sets are process observations, not exact scoped peak memory. The process still holds development/training caches; these are not a deployment memory minimum.

The `at_utc` tag in both start/complete attempt records identifies the attempt's start. Actual encoder event `clock_seconds` and completed-prediction `seconds` record execution timing. The development exposure ledger records authorization immediately before the first real development rollout; it is not an independent test-exposure claim.

Failure subsets defined by higher/lower observed CE are post-hoc descriptive subsets of genuine development executions. Local true-label CE gains are retrospective opportunities under a fixed budget, not downstream policy treatment effects. The sigmoid score imitates a privileged proxy teacher; it is not a calibrated probability of task benefit.
