# Frozen fresh-video confirmation snapshot

Completed October14,2026:100 new sources,five per frozen existing class. Primary B4 confirmation is inconclusive: mean gain +0.000550,95% interval[-0.003083,+0.004406]. No model/scaler/threshold/lambda/feature tuning occurred. No original findings were rewritten.

[Findings](findings.md) separate fresh and development results and preserve negative comparisons. [Repository commands](../../README.md#fresh_refresh_confirmation_v1-frozen-100-video-confirmation) run or resume the workflow; completed runs are audited without acquiring another cohort.

- `fresh_test_protocol.json`: unchanged authoritative prospective protocol.
- `execution_plan.json`, `sampling_plan.json`, `exclusions.json`, `acquisition.json`: frozen implementation/sampling/exclusions and complete acquisition/eligibility/retry/accounting trail.
- `frozen_test_manifest.json`, `frozen_artifact_identity.json`:100 original encoded identities and300 immutable feature/motion artifact hashes, frozen before evaluation.
- `features/`, `motion_frames/`:1,600 retained float32 references, candidate metadata, per-frame descriptors and reference-direction information.
- `environment.json`, `collection_metadata.json`, `collection_progress.json`: actual CUDA collection and extra-forward accounting.
- `test_policy_traces.csv`, `test_trajectories.csv`, `test_per_video_metrics.csv`:88,000 traces,5,500 trajectories,2,800 per-video rows.
- `test_aggregate_metrics.json`, `paired_comparisons.csv`, `bootstrap_intervals.json`, `per_class_descriptive.csv`:equal-video estimates, shared whole-video bootstrap, explicit missing denominators and descriptive classes.
- `verification.json`, `software_verification.json`, `additional_software_verification.json`, `development_compatibility.json`, `independent_replay_reproduction.json`:executed correctness, compatibility, reproducibility and information-isolation checks.
- `implementation_corrections.json`, `implementation_versions/`:pre-evaluation Windows I/O correction and original source copies; model/method/cohort unchanged.
- `metadata.json`, `status.json`, `execution_status_at_completion.json`, `SHA256SUMS.txt`:actual provenance, completion and integrity. The original completion status is preserved; its historical I/O blocker is resolved.
- Four PNGs and `example_selection.json`:budget means, primary per-video gains, paired intervals, predetermined success/failure timelines.

All original media encodings were preserved. Archives, videos, weights and the environment are ignored by Git; this curated snapshot contains the reference features, motion descriptors and replay results needed for independent analysis. Replay invokes neither fitting nor the encoder. Primary uncertainty is conditional on fixed models and sampled existing classes; secondary intervals are marginal/unadjusted. Feature fidelity is neither action-recognition accuracy nor measured adaptive latency/speedup.

[Publication review](publication_review.json) records a post-evaluation Git inventory guard correction, exact collection-time source copies and one error-message path made relative. Scientific measurements, policies, models and settings are unchanged. The historical completion metadata records that the experiment itself performed no publication.
