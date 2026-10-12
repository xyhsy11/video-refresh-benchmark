# video-refresh-benchmark

**Author: Siyu Han**

A reproducible benchmark comparing sequential video decoding, dense RGB feature extraction, and periodic RGB feature caching.

**Research question:** If RGB features are computed at only one-quarter of the candidate positions and cached features are reused elsewhere, how much does encoded-clip-to-features latency decrease?

**Status: preliminary feasibility benchmark.** The October 10 RGB pilot and an October 12 codec-vector CPU overhead follow-up have completed CUDA measurements on ten low-resolution Kinetics validation clips. Periodic processing was faster by median latency in every clip in both experiments. Drift-prediction diagnostics are also complete for the ten-video pilot and a separate 200-video subset with forty unseen test videos. Recognition accuracy, a learned motion branch, adaptive refresh policies, and higher-resolution performance remain unevaluated.

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

## Follow-up: codec motion-vector overhead

**Research question:** How much time remains saved by periodic RGB refresh when dense and periodic pipelines also export codec motion vectors and compute simple CPU motion descriptors?

The [motion experiment snapshot](results/motion_overhead_2026_10_11/README.md) uses the same ten encoded files, candidates, ResNet18 weights/preprocessing, numerical settings, and decoder threads. The folder name is the requested experiment label; actual collection timestamps are recorded in metadata. The three RGB controls are **rerun in this session** by calling the unchanged `benchmark.process_clip`; October 10 timings are not substitutes. The October 10 snapshot and its measurement sources remain unchanged.

| Condition | RGB calls / reuses / outputs | CPU motion descriptor updates |
| --- | --- | --- |
| `decode` | 0 / 0 / 0 | 0 |
| `dense` | 16 / 0 / 16 | 0 |
| `periodic` | 4 / 12 / 16 | 0 |
| `decode_mv` | 0 / 0 / 0 | One per decoded frame |
| `dense_mv` | 16 / 0 / 16 | One per decoded frame |
| `periodic_mv` | 4 / 12 / 16 | One per decoded frame, including RGB cache reuses |

All conditions fully decode every frame. `_mv` conditions set `stream.codec_context.options['flags2'] = '+export_mvs'` before the decoder opens and access `frame.side_data.get('MOTION_VECTORS').to_ndarray()`. This API was checked against the installed PyAV version and real decoder output. See [PyAV side data](https://pyav.basswood.io/docs/develop/api/sidedata.html), the [FFmpeg extraction example](https://www.ffmpeg.org/doxygen/9.0/extract__mvs_8c_source.html), and [AVMotionVector fields](https://www.ffmpeg.org/doxygen/9.0/structAVMotionVector.html).

For each exported entry, convert `motion_x`, `motion_y`, and `motion_scale` to float64 before division, then calculate `dx = motion_x / motion_scale`, `dy = motion_y / motion_scale`, and `numpy.hypot(dx, dy)`. Each frame's descriptor contains vector count, unweighted mean magnitude, and `numpy.percentile(magnitude, 90, method='linear')`, in codec-displacement pixels. Missing vectors give zeros with `mv_available=false`; observed zero displacement retains `mv_available=true`. Invalid scales or malformed fields raise a video/frame-specific error. Reference directions use `source<0` for past, `source>0` for future, and zero for other.

Each `_mv` mode retains the same fresh frozen CPU descriptor objects in a list until timing ends. There is no motion GPU transfer or learned motion encoder. Appearance features stay on CUDA until timing ends. Every trial uses a fresh container and resets all caches; neither frames nor raw vector arrays survive into later trials.

The boundary remains **encoded bytes already in memory through completed outputs**, with CUDA synchronization at both timer boundaries. Export configuration, export/access/conversion, CPU summaries, and descriptor retention are included. Inspection, file loading, verification, model preparation, CSV writing, and plotting are excluded. Each video gets two complete warmups per mode and five measured repeats per mode, with all six modes shuffled within each repeat using seed **20261011**, sequentially in one process. Metadata records actual order, loading times, source/video/weight hashes, environment, and GPU telemetry. Packages are recorded from the existing environment without upgrading them.

From the repository root, after dataset preparation and ordinary video inspection, run these PowerShell commands. To keep a published snapshot frozen when repeating the experiment, change all output paths to a fresh results directory.

```powershell
& .\.venv\Scripts\python.exe .\inspect_motion.py --input data/videos/y5Iu7XkTqV0.mp4 --output results/motion_overhead_2026_10_11/single_video_motion.csv
if ($LASTEXITCODE -ne 0) { throw 'Motion export inspection failed.' }
& .\.venv\Scripts\python.exe .\benchmark_motion.py --input data/videos/y5Iu7XkTqV0.mp4 --modes decode dense periodic decode_mv dense_mv periodic_mv --candidates 16 --refresh-every 4 --device cuda --verify-only
if ($LASTEXITCODE -ne 0) { throw 'Single-video correctness failed.' }
& .\.venv\Scripts\python.exe .\benchmark_motion.py --input data/videos/y5Iu7XkTqV0.mp4 --modes decode dense periodic decode_mv dense_mv periodic_mv --candidates 16 --refresh-every 4 --warmup 2 --repeats 5 --device cuda --output results/motion_overhead_2026_10_11/single_video.csv
if ($LASTEXITCODE -ne 0) { throw 'Single-video timing failed.' }
if ((Import-Csv .\results\motion_overhead_2026_10_11\single_video.csv).Count -ne 30) { throw 'Expected 30 measured rows.' }
```

```powershell
& .\.venv\Scripts\python.exe .\inspect_motion.py --input data/videos --output results/motion_overhead_2026_10_11/all_video_motion.csv
if ($LASTEXITCODE -ne 0) { throw 'Dataset motion inspection failed.' }
& .\.venv\Scripts\python.exe .\benchmark_motion.py --input data/videos --modes decode dense periodic decode_mv dense_mv periodic_mv --candidates 16 --refresh-every 4 --device cuda --verify-only
if ($LASTEXITCODE -ne 0) { throw 'Ten-video correctness failed.' }
& .\.venv\Scripts\python.exe .\benchmark_motion.py --input data/videos --modes decode dense periodic decode_mv dense_mv periodic_mv --candidates 16 --refresh-every 4 --warmup 2 --repeats 5 --device cuda --output results/motion_overhead_2026_10_11/pilot.csv
if ($LASTEXITCODE -ne 0) { throw 'Ten-video timing failed.' }
if ((Import-Csv .\results\motion_overhead_2026_10_11\pilot.csv).Count -ne 300) { throw 'Expected 300 measured rows.' }
& .\.venv\Scripts\python.exe .\summarize_motion.py --input results/motion_overhead_2026_10_11/pilot.csv --output results/motion_overhead_2026_10_11/summary.csv
if ($LASTEXITCODE -ne 0) { throw 'Motion analysis validation failed.' }
& .\.venv\Scripts\python.exe .\validate_release.py
```

Verification checks two fresh trials per condition, exact cross-mode motion descriptor equality, frame/vector/output counters, finite descriptors and features, refreshed features against the original dense control (`rtol=1e-5`, `atol=1e-6`), and appearance/motion state isolation across trials and videos. CUDA failure triggers CPU correctness and an environment diagnosis, with no CPU timing substitution.

`--verify-only` without `--output` writes into the labeled snapshot directory. For a fresh verification, add `--output results/motion_repro/single_video.csv` or `--output results/motion_repro/pilot.csv` as appropriate, and use that fresh directory consistently for subsequent diagnostics, timing, and analysis.

Analysis computes per-video statistics before aggregation and preserves negative differences. `_mv` minus control differences are **measured configuration overheads**, not exact isolated component times. Plots show trial variability; the reduction plot's whiskers combine observed dense/periodic extrema and are not confidence intervals. Five repeats do not establish statistical significance.

These are CPU codec displacement summaries, not optical flow, physical velocity, or an evaluated motion-recognition branch. B-frames and future-reference vectors make this an **offline** experiment; it does not establish strictly causal online inference. The encodings are preserved. Controlled I/P-only encoding is a separate experiment, and recognition accuracy and usefulness for refresh decisions remain untested.

### Measured motion-overhead result

The CUDA follow-up collected **30 single-video trials and 300 ten-video trials on October 12, 2026**. Every frame/count/state check passed. One pass over the ten clips yielded 3,096 frames, 3,074 with vectors and 22 without; 2,281,756 vector entries included 589,960 future-reference entries.

- Median per-video motion-enabled caching reduction: **25.86%**, range **17.72–31.97%**.
- Median per-video motion-enabled speedup: **1.349×**; periodic_mv was faster by median in **10/10 videos**.
- Same-session controls: **35.96% median per-video reduction**. The median of paired per-video reduction changes was **−11.48 percentage points**; the saving decreased in nine videos and increased by 1.75 points in `u4Rm6srmIS8`.
- Median per-video configuration overheads: **+84.14 ms** for decode_mv versus decode, **+71.47 ms** for dense_mv versus dense, and **+85.90 ms** for periodic_mv versus periodic.

This is a substantial observed CPU workload, while RGB caching retains a saving. Trial variability limits interpretation: only six videos had fully separated dense_mv/periodic_mv trial ranges; configuration differences were small relative to the larger trial range in seven dense comparisons and six periodic comparisons. These are descriptive comparisons, not statistical significance claims or exact isolated component costs. See the [per-video summary](results/motion_overhead_2026_10_11/summary.csv) and [full findings](results/motion_overhead_2026_10_11/findings.md).

Negative outcomes are retained: **2/50 same-repetition comparisons** had periodic_mv slower than dense_mv, with reductions of **−29.28%** (`b6Q_b7vgc7Q`, repetition 1) and **−6.78%** (`ddvJ6-faICE`, repetition 2). These sequential comparisons describe variability; they do not replace the per-video median statistics.

![Six-condition motion-overhead latency](results/motion_overhead_2026_10_11/latency_comparison.png)

![Caching reductions with and without CPU motion](results/motion_overhead_2026_10_11/caching_reduction_comparison.png)

## Diagnostic: predicting cached appearance-feature drift

This offline diagnostic asks whether codec motion adds predictive information beyond cache age. `collect_cache_drift.py` imports the unchanged encoder, preprocessing, numerical settings, input validation, decoder, and motion-descriptor helpers. CUDA collection fully decodes the same ten videos, records motion at every frame, and computes fresh features at all 16 saved candidates. Saving these reference targets is diagnostic work, not a latency benchmark.

For candidate `t=1..15`, the previous cached ordinal is `r=4*floor((t-1)/4)`. The target is **appearance-feature drift**, the cosine distance between fresh features at `t` and `r`. Labels at `t=4,8,12` are measured against `r=0,4,8` **before** refreshing. Candidate 0 has no prediction label. Four cache-age inputs and six motion inputs are listed in metadata. Interval motion covers decoded frames strictly after `r` through `t`, with explicit missing-vector availability. RGB features, target-derived inputs, identifiers, future decoded-frame positions, and whole-video summaries never enter the predictor matrix.

`analyze_cache_drift.py` evaluates constant, age-only, motion-only, and age-plus-motion predictors using leave-one-source-video-out folds. The three regressions use training-fold-only standardization and fixed ridge `alpha=1.0`, with no tuning or new dependencies. Raw predictions are saved; primary MAE, RMSE, and Spearman use predictions clipped to `[0,2]`. Constant-vector correlations are missing, not zero. Metrics and paired improvements are calculated per video before equal-weight aggregation.

Run from the repository root using the existing `.venv`:

```powershell
& .\.venv\Scripts\python.exe .\collect_cache_drift.py --input data/videos/y5Iu7XkTqV0.mp4 --candidates 16 --refresh-every 4 --device cuda --output results/cache_drift_smoke
if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke collection failed.' }
& .\.venv\Scripts\python.exe .\collect_cache_drift.py --input data/videos --candidates 16 --refresh-every 4 --device cuda --output results/cache_drift_v1
if ($LASTEXITCODE -ne 0) { throw 'Ten-video reference collection failed.' }
& .\.venv\Scripts\python.exe .\analyze_cache_drift.py --input results/cache_drift_v1 --output results/cache_drift_v1 --seed 20261012
if ($LASTEXITCODE -ne 0) { throw 'Drift verification or evaluation failed.' }
```

Collection preserves existing output directories by selecting and printing a fresh suffix. Use that reported directory for subsequent analysis. Completed analyses are also preserved: choose a fresh `--output` to recompute saved features without CUDA or videos. If CUDA or inputs are unavailable, resolve the environment check or run dataset preparation/inspection above; no CPU collection is substituted.

**Actual result, collected October 12, 2026:** 10 videos, 160 reference features, 150 observations, ten held-out folds, and 600 predictions. All focused checks and the original release validator passed. Adding motion reduced MAE in **6/10** videos and worsened it in **4/10**. Mean paired absolute MAE improvement was **0.018955**, median **0.009576**; mean per-video percentage improvement was **9.79%**, median **8.24%**, range **-34.82% to +61.73%**. These paired percentages differ from dividing aggregate MAE values.

Equal-weight mean per-video MAE was **0.100206** for age only, **0.081251** for age plus motion, **0.074165** for motion only, and **0.083268** for the constant baseline. Two clips account for about **80.92%** of the net absolute improvement. Age-plus-motion mean Spearman was **0.2850**, below age-only **0.3738**. Motion has useful but inconsistent predictive information under these settings; improved average magnitude prediction does not establish reliable ranking for refresh decisions. Validate fixed predictors on a larger untouched dataset before implementing a controller.

See [findings](results/cache_drift_v1/findings.md), [observations](results/cache_drift_v1/observations.csv), [held-out predictions](results/cache_drift_v1/predictions_oof.csv), [per-video metrics](results/cache_drift_v1/per_video_metrics.csv), [folds](results/cache_drift_v1/folds.json), [metadata](results/cache_drift_v1/metadata.json), and [verification](results/cache_drift_v1/verification.json). Saved features and per-frame motion enable independent label and interval audits. The previous scripts and published snapshots are unchanged.

![Per-video held-out MAE](results/cache_drift_v1/mae_age_vs_motion.png)

![Held-out predicted versus observed appearance-feature drift](results/cache_drift_v1/predicted_vs_observed.png)

Only ten independent videos are evaluated; their 150 observations are correlated. There is no statistical significance, action-recognition accuracy, adaptive refresh benefit, or new latency claim. Simple CPU codec descriptors are not a learned motion-recognition branch. B-frames can supply future-reference vectors, making this an offline diagnostic even though aggregation uses only current/preceding decoded positions. Original encodings are preserved; strictly causal I/P-only evaluation remains separate.

## cache_drift_v2: generalization to unseen videos

The v2 workflow evaluates the same four fixed drift predictors on a custom subset of **200 new Kinetics-400 clips from 20 classes**, excluding the ten original pilot source videos. Within each class, six videos train the models, two validate the pipeline, and two form the untouched test set: **120/40/40 videos**. Source IDs and identical-file hashes cannot overlap. This is a diagnostic subset, not an official full-dataset action-recognition evaluation.

`prepare_cache_drift_v2.py` uses [CVDF's hosted validation archive shards and annotations](https://github.com/cvdfoundation/kinetics-dataset). Seed **20261013** fixes class selection, shard order, within-class ranking, and custom splits before examining drift. Downloads use resumable HTTP ranges and an explicit compressed-archive limit; annotations, source hashes, acquisition events, malformed inputs, and deterministic replacement decisions are recorded. Original encodings are preserved. The limit includes reused/reserved archives, but excludes small annotation/HTTP metadata transfers and possible retransmissions. It does not download the complete dataset.

V2 uses its own manifest; the original collector's published-ten-video checks remain intact. Eligibility requires H.264, 5–15 seconds, at least 16 decoded frames, finite increasing candidate timestamps, and successful finite CUDA feature extraction. Missing motion vectors never disqualify an otherwise valid clip. Sixteen fresh ResNet18 features and fifteen **pre-refresh** cosine-drift labels are collected per video, with unchanged encoder/weights, preprocessing, float32 settings, decoder, age/motion inputs, and interval definitions. B-frames and future-reference vectors keep this an offline diagnostic.

Run from the repository root in the existing environment:

```powershell
& .\.venv\Scripts\python.exe .\verify_cache_drift_v2.py --output results/cache_drift_v2_checks/software
if ($LASTEXITCODE -ne 0) { throw 'Software checks failed.' }
& .\.venv\Scripts\python.exe .\prepare_cache_drift_v2.py --output results/cache_drift_v2 --max-download-gib 12 --download-workers 8
if ($LASTEXITCODE -ne 0) { throw 'Dataset incomplete; inspect acquisition and shortfall records, then resume.' }
& .\.venv\Scripts\python.exe .\collect_cache_drift_v2.py --manifest results/cache_drift_v2/dataset_manifest.json --device cuda --smoke --output results/cache_drift_v2_smoke
if ($LASTEXITCODE -ne 0) { throw 'Training-video smoke collection failed.' }
& .\.venv\Scripts\python.exe .\collect_cache_drift_v2.py --manifest results/cache_drift_v2/dataset_manifest.json --device cuda --output results/cache_drift_v2
if ($LASTEXITCODE -ne 0) { throw 'Reference collection incomplete; inspect collection metadata.' }
& .\.venv\Scripts\python.exe .\analyze_cache_drift_v2.py --input results/cache_drift_v2 --output results/cache_drift_v2_checks/real_verification --verify-only --check-video-files
if ($LASTEXITCODE -ne 0) { throw 'Real-data audit failed before evaluation.' }
& .\.venv\Scripts\python.exe .\analyze_cache_drift_v2.py --input results/cache_drift_v2 --output results/cache_drift_v2 --check-video-files
if ($LASTEXITCODE -ne 0) { throw 'Fixed-model evaluation failed.' }
```

Preparation or collection can resume with the same arguments plus `--resume`. Repeating a completed run selects and prints a fresh directory; use that directory consistently in subsequent commands. Cached downloads are reused with remote identity and local hash checks. The smoke run uses only training clips. GPU failure is explicitly incomplete and never substitutes CPU feature collection. Expected complete counts are **3,200 features and 3,000 observations**: 1,800 training, 600 validation, 600 testing.

The protocol is saved before acquisition/evaluation; the final split manifest is frozen after feature eligibility, before model evaluation. Scalers, ridge coefficients (`alpha=1.0`), and the constant mean use training rows only. Validation nominates the lowest-mean-MAE candidate for later research, without changing any model or evaluating an adaptive policy. All four fixed methods are evaluated on the same forty unseen test videos. Raw predictions and predictions clipped to `[0,2]` are saved separately; primary metrics consistently use clipped predictions.

MAE, RMSE, and defined Spearman correlations are calculated per video and aggregated with equal weight. Five predefined paired comparisons include age-plus-motion versus age-only and both motion models versus the constant baseline. Percentile bootstrap intervals use **10,000 paired whole-source-video resamples**, seed **20261013**, and fixed fitted models. They describe held-out-video uncertainty conditional on the fitted models. Undefined correlations/zero-denominator percentages remain missing. Negative outcomes and gain concentration are retained; candidate positions are not treated as independent videos.

The workflow saves acquisition/exclusions, manifests, environment/source/weight identities, reference features, per-frame motion, observations, fitted models, separate validation/test predictions, metrics, comparisons, bootstrap intervals, verification, plots, and findings. No recognition accuracy, adaptive-refresh benefit, or new speedup is measured. When auditing saved prediction CSVs, use `pd.read_csv(..., float_precision="round_trip")` to preserve float64 near-ties for Spearman ranks.

### Executed v2 results

Collection completed on **2026-10-12** using the existing RTX 3080 Laptop CUDA environment without package upgrades. Six of twenty validation shards supplied the complete subset, acquiring **8.81 GiB** of compressed archives within the 12 GiB limit. Seven clips failed the predefined duration eligibility check; no feature-collection replacements were needed. The frozen split contains **200 distinct videos**, **3,200 reference features**, and **3,000 observations** (1,800/600/600). The seed is 20261013; it is not the collection date.

All figures below give each of the **40 unseen test videos** equal weight. Lower MAE/RMSE means more accurate drift prediction; higher Spearman indicates better within-video ranking. The constant model has undefined correlation because its prediction is constant. No predictions required clipping.

| Predictor | Mean per-video MAE | Median per-video MAE | Mean per-video RMSE | Mean defined Spearman |
| --- | ---: | ---: | ---: | ---: |
| Constant | 0.082098 | 0.066609 | 0.093629 | undefined (40 videos) |
| Age only | 0.080349 | 0.071062 | 0.092088 | 0.4254 |
| Motion only | 0.068331 | 0.057542 | 0.080524 | 0.4179 |
| Age plus motion | 0.066963 | 0.058671 | 0.078713 | 0.4822 |

Age plus motion improves MAE over age only in **33/40 videos**, worsens it in seven, and ties in none. Mean paired absolute improvement is **0.013386**, with conditional 95% bootstrap interval **[0.007743, 0.018699]**. Mean and median per-video percentage improvements are **11.76% and 16.30%**. The two largest positive gains account for **17.18% of net improvement**, so the result is less concentrated than the ten-video pilot. Ranking correlation rises in 26 videos and falls in 14.

| Candidate versus reference | Mean paired MAE improvement | Conditional 95% interval | Improved / tied / worsened |
| --- | ---: | --- | --- |
| Motion only versus constant | +0.013767 | [+0.008163, +0.019157] | 34 / 0 / 6 |
| Motion only versus age only | +0.012018 | [+0.006721, +0.017169] | 33 / 0 / 7 |
| Age plus motion versus constant | +0.015135 | [+0.008274, +0.021785] | 32 / 0 / 8 |
| Age plus motion versus motion only | +0.001368 | [-0.000709, +0.003418] | 23 / 0 / 17 |

Motion adds predictive information beyond age and the constant baseline on this custom subset. The MAE difference between motion-only and age-plus-motion is **inconclusive**. Motion-only improves magnitude prediction but does not improve mean ranking over age-only. Validation nominated age plus motion using the predefined rule; test outcomes did not change any models or settings. The results support a separate adaptive-policy study on fresh held-out videos, with recognition accuracy and actual refresh/latency measurements. Prediction improvement alone establishes neither preserved accuracy nor adaptive speedup.

![V2 test MAE](results/cache_drift_v2/test_mae.png)

See the [complete findings](results/cache_drift_v2/findings.md), [paired improvements plot](results/cache_drift_v2/test_paired_improvements.png), [prediction plot](results/cache_drift_v2/test_predicted_vs_observed.png), [per-video metrics](results/cache_drift_v2/per_video_metrics.csv), [all bootstrap comparisons](results/cache_drift_v2/bootstrap_intervals.json), [frozen split](results/cache_drift_v2/split_manifest.json), and [verification](results/cache_drift_v2/verification.json). Eighteen software checks, the training-only smoke collection, independent real-data label/interval audits, saved-prediction metric reproduction, and the existing release validator passed. All 90 protected original sources and snapshot files remain unchanged. Archives, videos, and weights remain local and ignored by Git.

## Limitations and roadmap

This is a preliminary feasibility benchmark, not evidence of maintained recognition accuracy or a production streaming system.

- Recognition accuracy and the effect of stale appearance features remain unevaluated.
- The October 10 pilot excludes motion processing. The follow-up measures codec-vector export and simple CPU descriptor overhead; a learned motion branch, adaptive refresh controller, and action-classification head remain unevaluated.
- The clips are low-resolution; higher-resolution and 1080p performance remain unevaluated.
- File loading is measured separately and excluded from the reported metric.
- Offline full-clip processing time is not live-stream response latency.
- Ten clips and five repetitions per mode are a small pilot on one laptop; observed trial ranges and cross-run differences limit generalization.

The follow-up confirms savings remain with a substantial CPU codec-vector workload. Next, profile export/access/percentile computation, use longer controlled runs, evaluate recognition accuracy and motion usefulness for refresh decisions, and test controlled I/P-only encoding for a separate causal-inference study. Compare adaptive refresh policies with the periodic rule and test higher resolutions. If savings disappear, profile decoding/conversion/preprocessing/transfers; if periodic becomes slower, inspect overhead and variability. Fix incorrect counters before interpreting performance.

## Repository layout

- `prepare_data.py`: archive validation, extraction, and selected-video copies.
- `inspect_video.py`: full sequential inspection and saved candidates.
- `benchmark.py`: shared verification/processing logic and CUDA timing.
- `summarize.py`: validation, per-video summaries, chart, and findings.
- `check_env.py`: environment and GPU correctness checks.
- `research_utils.py`: project-relative paths, manifest helpers, and decoder settings.
- `validate_release.py`: public snapshot and documentation validation.
- `motion_utils.py` / `inspect_motion.py`: codec-vector export, CPU descriptors, and per-frame diagnostics.
- `benchmark_motion.py` / `summarize_motion.py`: six-condition correctness/timing and motion-overhead analysis.
- `prepare_cache_drift_v2.py` / `collect_cache_drift_v2.py` / `analyze_cache_drift_v2.py`: bounded dataset acquisition, CUDA reference collection, and fixed train/validation/test drift evaluation.
- `cache_drift_v2_utils.py` / `verify_cache_drift_v2.py`: frozen protocol, identity/preservation checks, and isolated software fixtures.
- `requirements.txt` / `requirements-cuda.txt`: tested dependency installation, with CUDA separated.
- `results/pilot_2026_10_10/`: curated public results and exact tested environment.
- `results/motion_overhead_2026_10_11/`: curated codec-vector overhead experiment, with actual run timestamps.
- `results/cache_drift_v1/` / `results/cache_drift_v2/`: curated drift-prediction diagnostics and their audit artifacts.
- `data/source/`, `data/videos/`, `.venv/`, and other `results/` subfolders: local, ignored assets and outputs.

## Credits and license

Original project code is released under the [MIT License](LICENSE), **Copyright (c) 2026 Siyu Han**. Dataset and pretrained-weight terms remain separate.

This project uses **PyTorch/torchvision** for tensors and ResNet18, **PyAV/FFmpeg** for video decoding, and the **OpenMMLab/MMAction2** Kinetics tiny archive derived from the **Kinetics dataset**. V2 uses **CVDF's hosted Kinetics-400 archives and annotations**, credited in its dataset manifest. NumPy, pandas, matplotlib, and Pillow support analysis and preprocessing. See [third-party notices and upstream terms](THIRD_PARTY_NOTICES.md). No downloaded videos, pretrained weights, or dependency source code are distributed in this repository.
