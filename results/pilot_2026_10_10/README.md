# Public pilot snapshot — 2026-10-10

Project: **video-refresh-benchmark**. Author: **Siyu Han**.

This frozen snapshot records a preliminary feasibility experiment on ten Kinetics tiny validation videos. It contains 150 measured pilot trials, plus the preceding 15-trial single-video check. Headline statistics are calculated per video before aggregation: **34.74% median reduction**, **1.532× median speedup**, and **17.52–45.54% reduction range**.

## Contents

| File | Purpose |
| --- | --- |
| [pilot.csv](pilot.csv) | Original 150 measured trial rows; bytes unchanged |
| [summary.csv](summary.csv) | Original per-video medians, minima, maxima, speedups, and reductions; bytes unchanged |
| [latency_comparison.png](latency_comparison.png) | Original chart with trial variability; bytes unchanged |
| [findings.md](findings.md) | Original interpretation, variability, next step, and limits; bytes unchanged |
| [pilot.metadata.json](pilot.metadata.json) | Experiment settings, execution order, hashes, and environment; personal paths sanitized |
| [manifest.json](manifest.json) | Selected filenames, source URL, encoded-file hashes, inspection properties, and candidate indices |
| [dataset.json](dataset.json) | Source archive and selection provenance |
| [environment.json](environment.json) | Tested CPU/GPU/driver, Python/packages, PyAV-linked FFmpeg libraries, and GPU checks; personal paths sanitized |
| [requirements-tested.txt](requirements-tested.txt) | Full original tested package pins, including CUDA PyTorch |
| [aggregate_summary.json](aggregate_summary.json) | Explicitly labeled aggregates |
| [validation.json](validation.json) | Original artifact/counter validation results |
| [pilot_verification.json](pilot_verification.json) | Original two-trial-per-mode correctness verification for ten videos |
| [single_video.csv](single_video.csv) | Original 15 measured single-video trial rows; bytes unchanged |
| [single_video.metadata.json](single_video.metadata.json) | Settings for the initial single-video check; personal paths sanitized |
| [single_video_verification.json](single_video_verification.json) | Original single-video correctness results |
| [release_validation.json](release_validation.json) | Additional checks executed during public-release preparation |
| [SHA256SUMS.txt](SHA256SUMS.txt) | Checksums for all snapshot files except the checksum file itself |

`NVIDIA driver 610.62`, driver-reported CUDA support 13.3, and `torch.version.cuda == 13.2` are distinct records. Useful hardware/library details are preserved. Public environment records use `.venv/Scripts/python.exe` and `.venv`; the base Python installation path is omitted. No videos, pretrained weights, credentials, or hostnames are included.

The original measurement-source hashes remain in metadata. Benchmark processing, decoder settings, and timing source bytes are unchanged. For release, installation requirements were split into CUDA/non-CUDA files, the summarizer gained adjacent/explicit manifest selection, and release validation/documentation were added. These are installation or postprocessing changes; no new latency measurements were taken. The full original requirements are archived here so their original hash remains verifiable.

Recompute statistics from the repository root without downloading videos:

```powershell
& .\.venv\Scripts\python.exe .\summarize.py --input results/pilot_2026_10_10/pilot.csv --output results/recomputed_pilot_2026_10_10/summary.csv
& .\.venv\Scripts\python.exe .\validate_release.py
```

The snapshot manifest is historical; re-run inspection to generate current file timestamps/candidates before a new benchmark. Write new outputs outside this directory. See the [project README](../../README.md) for installation, measurement controls, limitations, roadmap, and [third-party terms](../../THIRD_PARTY_NOTICES.md).
