# cache_drift_v2 results snapshot

This is a real CUDA diagnostic collected on October 12, 2026: 200 distinct new Kinetics-400 source videos, 20 classes, custom train/validation/test splits of 120/40/40. The seed is 20261013. This is not an official full-dataset recognition evaluation or a latency benchmark.

[Findings](findings.md) report all four fixed predictors on the forty unseen test videos, including negative per-video results and conditional whole-video bootstrap intervals. The primary comparison improves MAE in 33 videos and worsens it in seven. The difference between motion-only and age-plus-motion remains inconclusive.

The [repository README](../../README.md#cache_drift_v2-generalization-to-unseen-videos) contains reproduction commands, acquisition limits, and the experiment definitions. Run commands from the repository root in the existing project environment. Completed runs select fresh output directories. Keep the original encodings; never replace clips using prediction quality.

Artifacts:

- `sampling_plan.json`, `acquisition.json`, `exclusions.json`, `dataset_manifest.json`: source selection, shard URLs/hashes, exclusions, and pre-collection candidates.
- `protocol.json`, `split_manifest.json`: frozen analysis protocol and final source/file identities, labels, candidate positions/timestamps, and splits.
- `environment.json`, `collection_metadata.json`, `collection_progress.json`: actual CUDA/software settings, weights/source hashes, collection timestamps, file-loading/collection times, and motion-reference direction counts.
- `features/`: 200 float32 arrays of shape `[16,512]` plus candidate metadata; no model weights or videos.
- `motion_frames/`: one CPU codec-displacement descriptor per decoded frame, with explicit availability and past/future/other reference counts.
- `observations.csv`, `video_observations/`: 3,000 pre-refresh labels and audited inputs, with identifiers separated from predictor columns.
- `fitted_predictors.json`, `validation_selection.json`: training-only fitted parameters and the unchanged-model validation nomination recorded before test evaluation.
- `validation_predictions.csv`, `test_predictions.csv`: 2,400 predictions each (600 observations times four models), raw and clipped to `[0,2]`.
- `per_video_metrics.csv`, `paired_comparisons.csv`, `bootstrap_intervals.json`, `aggregate_metrics.json`: equal-weight per-video results and all five predefined comparisons.
- `collection_verification.json`, `setup_verification.json`, `verification.json`, `final_checks.json`: executed checks, with synthetic software fixtures kept separate from experimental evidence.
- `analysis_metadata.json`, `SHA256SUMS.txt`: analysis provenance and snapshot artifact integrity.
- `test_mae.png`, `test_paired_improvements.png`, `test_predicted_vs_observed.png`: numerical accuracy, signed paired improvements, and prediction quality.

Read prediction CSVs with `pd.read_csv(path, float_precision="round_trip")` when reproducing Spearman ranks. Default float parsing can alter floating-point near-ties. An initial analysis audit failed that round-trip check; it was fixed and the unchanged four-model protocol was rerun. No data, alpha, features, or fitting definitions were tuned in response to test results. The incomplete attempt is preserved locally outside this curated snapshot.

Motion descriptors summarize codec displacement on the CPU, not optical flow or an evaluated recognition branch. B-frames and future-reference vectors make this an offline diagnostic. Forty test videos limit generalization; bootstrap intervals condition on the fitted models and exclude training/class-selection uncertainty. Better drift prediction does not establish recognition accuracy or adaptive refresh speedup. Dataset and pretrained-weight terms remain separate from the project's MIT code license.
