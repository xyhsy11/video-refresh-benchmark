# Codec motion-vector overhead snapshot

Project: **video-refresh-benchmark**, Siyu Han. This is a preliminary offline overhead experiment with six conditions: `decode`, `dense`, `periodic`, `decode_mv`, `dense_mv`, and `periodic_mv`.

The directory's October 11 date is the requested experiment label. The actual CUDA measurements were collected on **2026-10-12**; precise UTC timestamps appear in the metadata. Original video encodings and the October 10 release snapshot are preserved.

- [Single-video trials](single_video.csv): 30 measured rows; [metadata](single_video.metadata.json) and [verification](single_video_verification.json).
- [Ten-video trials](pilot.csv): 300 measured rows; [metadata](pilot.metadata.json) and [verification](pilot_verification.json).
- [Per-video summary](summary.csv), [analysis validation](validation.json), and [findings](findings.md).
- [Six-condition latency plot](latency_comparison.png) and [caching-reduction comparison](caching_reduction_comparison.png); observed trial variability is displayed, not confidence intervals.
- [Reference-clip frame diagnostics](single_video_motion.csv) and [summary](single_video_motion.summary.json).
- [All-video frame diagnostics](all_video_motion.csv) and [summary](all_video_motion.summary.json).
- [Selected-video manifest](manifest.json), [current environment](environment.json), and [exact installed package pins](requirements-tested.txt).
- [Descriptor arithmetic/malformed-data checks](descriptor_validation.json), [release checks](release_checks.json), and [snapshot checksums](SHA256SUMS.txt).
- [Analysis rejection/negative-value checks](analysis_checks.json), including the actual repetitions where periodic motion processing was slower.

All three original controls were rerun in this session using the unchanged original `process_clip`. Every condition fully decodes the clip. Motion descriptors update on every decoded frame, including missing vectors and RGB cache reuses; they remain on the CPU until the trial ends. Appearance features remain on CUDA. The same encoded-byte boundary, float32/batch-size-1 ResNet18, preprocessing, deterministic settings, and single-thread decoder are used throughout.

Metadata records source-code hashes, the source base commit/branch, video identities/candidates, weight hash, settings, shuffled execution order, loading/verification/model-setup time exclusions, and GPU telemetry. The source base commit precedes these new files; the recorded source hashes identify the exact measured implementation. Relative paths are used; personal installation paths and unnecessary user/host information are omitted. No media, archives, weights, or credentials are distributed.

The vectors summarize codec displacement, not optical flow or physical velocity. B-frames can provide future-reference entries, so this does not establish strictly causal online inference. These are simple CPU descriptors, not an evaluated motion-recognition branch. Accuracy and usefulness for refresh decisions remain untested. Per-video statistics are computed before aggregation; configuration differences are not exact isolated component timings, and negative values are retained.

See the [project README and exact PowerShell commands](../../README.md) and [third-party terms](../../THIRD_PARTY_NOTICES.md). Original code uses the [MIT license](../../LICENSE); video and pretrained-weight terms remain separate. When repeating the experiment, write to a new directory to retain this frozen snapshot.
