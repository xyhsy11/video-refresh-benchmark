# video-refresh-benchmark

**Author: Siyu Han**

A reproducible benchmark comparing sequential video decoding, dense RGB feature extraction, and periodic RGB feature caching.

**Research question:** If RGB features are computed at only one-quarter of the candidate positions and cached features are reused elsewhere, how much does encoded-clip-to-features latency decrease?

**Status: preliminary feasibility benchmark.** A CUDA pilot on ten low-resolution Kinetics validation clips has been completed and verified. Periodic processing was faster by median latency in every clip. Recognition accuracy, motion-processing overhead, adaptive refresh policies, and higher-resolution performance remain unevaluated.

## What the three modes do

Every mode sequentially decodes the **entire encoded video**, including frames between candidates and after the last refresh. There is no seeking, skipped-frame decoding, or reuse of previously decoded frames. Inspection selects 16 distinct approximately uniform zero-based decoded-frame indices, including the first and last frames.

| Mode | Work at candidate positions | Encoder calls | Cache reuses | Feature outputs |
| --- | --- | ---: | ---: | ---: |
| `decode` | Decode only; no RGB conversion or appearance encoder | 0 | 0 | 0 |
| `dense` | Convert to RGB, preprocess, transfer to CUDA, and encode every candidate | 16 | 0 | 16 |
| `periodic` | Refresh at candidate ordinals 0, 4, 8, and 12; reuse the latest feature elsewhere | 4 | 12 | 16 |

Candidate ordinals are positions in the saved candidate list, not raw video-frame indices. The periodic cache starts empty for every video and every trial. A reused output is the same most recently refreshed feature, not a newly computed feature for that frame.

The encoder is torchvision's pretrained **ResNet18_Weights.DEFAULT**, with its classification layer replaced by `torch.nn.Identity()`. Each encoder call produces a `[1, 512]` float32 feature. ResNet18 is an **appearance-feature timing prototype**, using batch size 1, CUDA, `model.eval()`, and `torch.inference_mode()`; it is not an evaluated action-recognition system. The weights' [documented preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) resizes the shorter edge to 256 with bilinear interpolation, center-crops to 224, rescales to [0,1], and applies ImageNet normalization. RGB conversion and preprocessing run only at refreshes.

## Published pilot: 2026-10-10

The curated [public results snapshot](results/pilot_2026_10_10/README.md) contains **10 videos and 150 measured trials**: ten videos × three modes × five repetitions. A preceding single-video check contains 15 trials. Videos are H.264, 7.600–10.022 seconds long, with 228–600 decoded frames and resolutions 256×452, 454×256, or 412×256.

The values below were verified directly against the saved trial CSV. They are descriptive pilot results for this dataset and machine.

| Video | Decode median (ms) | Dense median (ms) | Periodic median (ms) | Speedup | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0pVGiAU6XEA | 109.10 | 231.45 | 142.71 | 1.622x | 38.34% |
| AQrbRSnRt8M | 172.92 | 307.24 | 201.11 | 1.528x | 34.55% |
| IcLztCtvhb8 | 162.95 | 302.87 | 197.07 | 1.537x | 34.93% |
| SU_x2LQqSLs | 115.53 | 247.01 | 147.70 | 1.672x | 40.21% |
| b6Q_b7vgc7Q | 192.20 | 321.20 | 223.26 | 1.439x | 30.49% |
| ddvJ6-faICE | 84.42 | 210.36 | 114.57 | 1.836x | 45.54% |
| ik4BW3-SCts | 138.99 | 272.47 | 171.06 | 1.593x | 37.22% |
| jqRrH30V0k4 | 290.82 | 414.24 | 341.68 | 1.212x | 17.52% |
| u4Rm6srmIS8 | 160.84 | 293.10 | 195.14 | 1.502x | 33.42% |
| y5Iu7XkTqV0 | 169.15 | 276.36 | 211.02 | 1.310x | 23.64% |

For each video `i`, first calculate:

```text
speedup_i = median(dense trials_i) / median(periodic trials_i)
reduction_i (%) = 100 * (median(dense trials_i) - median(periodic trials_i)) / median(dense trials_i)
```

Then summarize those ten per-video statistics with equal weight per video:

- **Median per-video latency reduction: 34.74%.**
- **Median per-video speedup: 1.532×.**
- **Per-video reduction range: 17.52–45.54%.**
- Median latency was lower with periodic processing in **10/10 videos**.

The median across the ten video-level latency medians is 161.89 ms for decode only, 284.73 ms for dense, and 196.10 ms for periodic. Dividing the last two aggregate medians gives approximately **1.452×**, a different calculation from the **1.532× median of per-video speedups**. The headline statistics are calculated per video before aggregation.

![Per-video median latency with min–max trial variability](results/pilot_2026_10_10/latency_comparison.png)

Bars show per-video medians; whiskers show trial minimum–maximum; dots show individual trials. Nine videos had non-overlapping observed dense/periodic trial ranges. For `y5Iu7XkTqV0`, the ranges overlapped and the reduction differed between the initial single-video run and the ten-video pilot. Five repeats describe exploratory variability, not confidence intervals or statistical significance.

**Decode-only is a separate baseline with no feature outputs.** It is not assumed to be an exact additive component and is not subtracted from another mode to estimate isolated encoder time. Full decoding remains necessary even when encoder calls fall by 75%.

See the [raw trials](results/pilot_2026_10_10/pilot.csv), [summary](results/pilot_2026_10_10/summary.csv), and [findings](results/pilot_2026_10_10/findings.md).

## Measurement boundary and controls

The metric is **Encoded-clip-to-features latency (ms)**: encoded video bytes already in memory through completed feature outputs. For every timed trial:

1. Synchronize CUDA, then start `time.perf_counter()`.
2. Open a fresh PyAV container over a fresh `io.BytesIO(encoded_bytes)`.
3. Sequentially decode the complete video and execute the selected mode.
4. Close the container, synchronize CUDA, then stop the timer.

The boundary includes container opening/closing, demuxing/decoding, required RGB conversion and preprocessing, CPU-to-GPU transfer, encoder execution, and feature-output/cache handling. Feature outputs remain on the GPU until timing finishes. No decoded frames, preprocessed tensors, or features survive from an earlier trial.

Dataset downloading/extraction, file loading, inspection, model construction/weight download, correctness checks, logging, and CSV writing are excluded. Encoded-file loading is recorded separately for each video and its bytes are reused across that video's trials. Model weights are prepared before timing.

- **Warmups:** two complete trials per mode per video, excluded from CSV rows.
- **Measurements:** five repetitions per mode per video; shuffle the three modes within each repetition using seed **20261010**. Actual execution order is recorded in the metadata sidecar.
- **Decoder:** software decoding, `thread_count=1`, `thread_type=NONE`, `skip_frame=NONE`, identical across modes.
- **Numerics:** float32, batch size 1, deterministic algorithms enabled; TF32 and autocast disabled; cuDNN benchmark disabled. PyTorch CPU and interop thread counts are 1. No compiled model is used.
- **Execution:** videos and trials run sequentially in one process. CUDA synchronization occurs at both timer boundaries. Feature validation and counter checks occur outside timed regions.

The [run metadata](results/pilot_2026_10_10/pilot.metadata.json) records inputs, candidate indices, file-loading times, settings, execution order, weight hash, original measurement-source hashes, and environment. Reproducible inputs and settings do not guarantee identical wall-clock timings on a laptop with changing scheduling or power state.

## Tested hardware and software

| Component | Tested value |
| --- | --- |
| OS | Windows 11 Home Chinese, x64; build 22631 |
| CPU | Intel Core i7-10870H @ 2.20 GHz; 8 cores / 16 logical processors |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU; 16 GiB VRAM |
| NVIDIA driver | **610.62** |
| Driver-reported CUDA support | 13.3 (`nvidia-smi` CUDA UMD Version) |
| PyTorch CUDA build | **13.2**, from `torch.version.cuda` (`cu132`) |
| Python | 3.12.10, 64-bit |
| PyTorch | 2.14.1+cu132 |
| torchvision | 0.29.1+cu132 |
| PyAV | 19.0.1 |
| FFmpeg linked by PyAV | 9.0.2 |
| NumPy / pandas | 2.5.2 / 3.0.6 |
| matplotlib / Pillow | 3.11.2 / 12.3.0 |

The **NVIDIA driver version** and **PyTorch CUDA build version** are different quantities. The driver must support the selected runtime; their version numbers need not match. The tested driver supports the cu132 build. See [NVIDIA's compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html).

The sanitized [environment record](results/pilot_2026_10_10/environment.json) includes each linked FFmpeg library version and successful GPU checks. The [exact tested requirements](results/pilot_2026_10_10/requirements-tested.txt) retain all installed package versions. Personal installation paths and unnecessary user/host information are omitted from public copies.

## Installation

All PowerShell commands below run from the **repository root**, the folder containing `benchmark.py`. The checkout folder may have any name. The commands use `.venv` directly and do not require activation. Windows x64 with Python 3.12 and an NVIDIA GPU is the tested setup; other platforms and driver/build combinations require their own validation.

### 1. Create the project environment

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade --only-binary=:all: pip
```

### 2. Install CUDA-enabled PyTorch separately

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose a CUDA wheel family compatible with your NVIDIA driver, and install matching torch/torchvision builds together. For the exact tested cu132 setup:

```powershell
nvidia-smi
& .\.venv\Scripts\python.exe -m pip install --only-binary=:all: 'torch==2.14.1+cu132' 'torchvision==0.29.1+cu132' --index-url https://download.pytorch.org/whl/cu132
```

The same tested command is provided by `requirements-cuda.txt`:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r .\requirements-cuda.txt
```

Run either CUDA installation command. If your driver cannot support this build, choose a supported build from the selector and record that environment for new results. The prebuilt wheels provide the CUDA runtime libraries used by this project; a separate CUDA Toolkit is not needed for these checks.

### 3. Install the remaining dependencies and check CUDA

```powershell
& .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe .\check_env.py
if ($LASTEXITCODE -ne 0) { throw 'Resolve the environment diagnosis before GPU timing.' }
```

`requirements.txt` contains tested, index-resolvable non-PyTorch dependencies with no local `file://` paths. `check_env.py` prints and saves Python/package/FFmpeg versions, CPU/GPU/driver details, CUDA availability, and a small GPU calculation. CUDA must pass before reporting GPU timings. The benchmark can run CPU correctness checks if CUDA is unavailable, but it does not silently substitute CPU latency measurements.

## Reproduce a new experiment

New outputs below go under `results/repro/`, separate from the frozen public snapshot. Downloaded videos, archives, model weights, local diagnostics, and new results are ignored by Git.

### 1. Prepare the dataset

```powershell
& .\.venv\Scripts\python.exe .\prepare_data.py
if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed.' }
```

This downloads [OpenMMLab's kinetics400_tiny.zip](https://download.openmmlab.com/mmaction/kinetics400_tiny.zip) into `data/source/`, validates and reuses an existing valid archive, extracts it there, preserves source annotation files, and copies its ten validation MP4s into `data/videos/`. Original names and encoded bytes are preserved. The source URL, selected paths, and SHA-256 hashes are recorded in the manifest. No MMAction2 installation is required. Existing conflicting files are preserved and reported as errors.

### 2. Inspect the first clip, then the remaining nine

```powershell
& .\.venv\Scripts\python.exe .\inspect_video.py --input data/videos/y5Iu7XkTqV0.mp4
if ($LASTEXITCODE -ne 0) { throw 'First-video inspection failed.' }
& .\.venv\Scripts\python.exe .\inspect_video.py --input data/videos --skip-existing
if ($LASTEXITCODE -ne 0) { throw 'Remaining inspections failed.' }
```

Inspection fully decodes each clip, counts actual frames and picture types, determines duration, and writes 16 distinct candidate indices to `results/manifest.json`. It rejects fewer than 16 decoded frames or duration outside 5–15 seconds. Entries are keyed by relative path and retain file hashes, sizes, and timestamps to detect changes. Frames are discarded after inspection; benchmark trials decode again. Directory inputs use selected manifest paths, excluding unrelated MP4s. Reinspect after moving/replacing a video: the original machine's modification timestamps are not portable input identity for a new run.

### 3. Verify one video

```powershell
& .\.venv\Scripts\python.exe .\benchmark.py --input data/videos/y5Iu7XkTqV0.mp4 --modes decode dense periodic --candidates 16 --refresh-every 4 --device cuda --verify-only --output results/repro/single_video.csv
if ($LASTEXITCODE -ne 0) { throw 'Correctness failed; stop before timing.' }
```

Verification exercises two fresh trials per mode. It checks decoded-frame counts, encoder/cache/output counters, feature shape and finite values, refreshed-feature agreement with dense outputs, reused-feature equality, first-candidate refresh, and cache isolation across trials/videos. The initial clip decoded to 300 frames; its expected rows are `decode: 300/0/0/0`, `dense: 300/16/0/16`, and `periodic: 300/4/12/16` (frames/calls/reuses/outputs). `--verify-only` writes a verification JSON, not a measured CSV.

### 4. Run the single-video timing check

```powershell
& .\.venv\Scripts\python.exe .\benchmark.py --input data/videos/y5Iu7XkTqV0.mp4 --modes decode dense periodic --candidates 16 --refresh-every 4 --warmup 2 --repeats 5 --device cuda --output results/repro/single_video.csv
if ($LASTEXITCODE -ne 0) { throw 'Single-video timing failed.' }
if ((Import-Csv .\results\repro\single_video.csv).Count -ne 15) { throw 'Expected 15 measured rows.' }
```

### 5. Verify all ten, then measure all ten

```powershell
& .\.venv\Scripts\python.exe .\benchmark.py --input data/videos --modes decode dense periodic --candidates 16 --refresh-every 4 --device cuda --verify-only --output results/repro/pilot.csv
if ($LASTEXITCODE -ne 0) { throw 'Ten-video verification failed.' }
& .\.venv\Scripts\python.exe .\benchmark.py --input data/videos --modes decode dense periodic --candidates 16 --refresh-every 4 --warmup 2 --repeats 5 --device cuda --output results/repro/pilot.csv
if ($LASTEXITCODE -ne 0) { throw 'Ten-video timing failed.' }
if ((Import-Csv .\results\repro\pilot.csv).Count -ne 150) { throw 'Expected 150 measured rows.' }
```

Timing invocations repeat correctness verification before their warmups. Every trial's counters are checked outside timing. Add `--seed INTEGER` to change the default order seed and record it in the run metadata.

### 6. Summarize new measurements

```powershell
& .\.venv\Scripts\python.exe .\summarize.py --input results/repro/pilot.csv --output results/repro/summary.csv
if ($LASTEXITCODE -ne 0) { throw 'Result validation or summary failed.' }
```

The summary validates frame counts, counters, row counts, and recorded execution order. It writes per-video median/min/max latencies, speedups, reductions, a variability plot, aggregate statistics, and findings. Negative reductions are preserved. A manifest beside the input CSV is used when present; otherwise the live `results/manifest.json` is used. `--manifest FILE` selects one explicitly.

## Recompute the published summary without downloading videos

After installing the packages, the frozen snapshot can be analyzed without the local dataset, CUDA availability, or downloaded model weights:

```powershell
& .\.venv\Scripts\python.exe .\summarize.py --input results/pilot_2026_10_10/pilot.csv --output results/recomputed_pilot_2026_10_10/summary.csv
& .\.venv\Scripts\python.exe .\validate_release.py
```

The adjacent snapshot manifest and metadata provide the saved input identity/settings. The release validator checks raw counters, reported statistics, snapshot checksums, source hashes for the unchanged benchmark, and relative documentation links. It does not run a new GPU timing experiment. Do not write recomputed output into the frozen snapshot.

## Limitations and roadmap

This is a preliminary feasibility benchmark, not evidence of maintained recognition accuracy or a production streaming system.

- Recognition accuracy and the effect of stale appearance features remain unevaluated.
- Motion-feature extraction, a motion branch, motion-processing overhead, a learned/adaptive refresh controller, and an action-classification head are not included or evaluated.
- The clips are low-resolution; higher-resolution and 1080p performance remain unevaluated.
- File loading is measured separately and excluded from the reported metric.
- Offline full-clip processing time is not live-stream response latency.
- Ten clips and five repetitions per mode are a small pilot on one laptop; observed trial ranges and cross-run differences limit generalization.

The consistent pilot savings support adding motion-vector extraction and measuring its overhead with the same boundary. Next, evaluate recognition accuracy, compare adaptive refresh policies with the fixed periodic rule, and test larger datasets, longer repetitions, and higher resolutions. If savings disappear, profile decoding/conversion/preprocessing/transfers; if periodic becomes slower, inspect overhead and variability. Fix incorrect counters before interpreting performance.

## Repository layout

- `prepare_data.py`: archive validation, extraction, and selected-video copies.
- `inspect_video.py`: full sequential inspection and saved candidates.
- `benchmark.py`: shared verification/processing logic and CUDA timing.
- `summarize.py`: validation, per-video summaries, chart, and findings.
- `check_env.py`: environment and GPU correctness checks.
- `research_utils.py`: project-relative paths, manifest helpers, and decoder settings.
- `validate_release.py`: public snapshot and documentation validation.
- `requirements.txt` / `requirements-cuda.txt`: tested dependency installation, with CUDA separated.
- `results/pilot_2026_10_10/`: curated public results and exact tested environment.
- `data/source/`, `data/videos/`, `.venv/`, and other `results/` subfolders: local, ignored assets and outputs.

## Credits and license

Original project code is released under the [MIT License](LICENSE), **Copyright (c) 2026 Siyu Han**. Dataset and pretrained-weight terms remain separate.

This project uses **PyTorch/torchvision** for tensors and ResNet18, **PyAV/FFmpeg** for video decoding, and the **OpenMMLab/MMAction2** Kinetics tiny archive derived from the **Kinetics dataset**. NumPy, pandas, matplotlib, and Pillow support analysis and preprocessing. See [third-party notices and upstream terms](THIRD_PARTY_NOTICES.md). No downloaded videos, pretrained weights, or dependency source code are distributed in this repository.
