# Real development failure analysis

DEVELOPMENT / EXPLORATORY. Read failure_analysis.json for full actual counts and retrospective_decisions.csv for every decision.

H+M versus periodic: 199 higher-CE clips; teacher agreement 0.699367; chosen harmful 590; missed helpful 1783; teacher-positive/task-harmful 735; MAD/local-gain Spearman -0.009912613200823945.

Incorrect timing is evaluated using actual earlier cache versus current real dense feature. Cheap changes are compared with task gain only after decisions. Teacher disagreement is conditional on the actual prefix, and proxy-positive/task-harmful cases expose objective mismatch. The sigmoid output imitates a proxy action, not beneficial-task probability. Base/on-policy/live state distributions and feature-standardization excursions expose distribution shift. All local diagnostics omit downstream counterfactual policy changes and therefore are not causal treatment-effect estimates.

Controller overhead, grayscale conversion, cheap-statistics work, observation construction and peak CUDA allocations are measured in actual timing_trials.csv and timing_per_video.csv. Periodic+unused-gate isolates gating work without changing the refresh schedule. Only per-video medians and variation support execution-cost comparisons; no simulated or replay-derived latency is used. No new architecture or post-outcome threshold change is permitted.
