# Video latency pilot findings

Refreshing one-quarter of the candidates reduced median latency consistently across all selected videos.

Encoder calls fell from **16 to 4**, while dense and periodic both produced 16 feature outputs and decoded every frame. The **median of the 10 per-video latency reductions was 34.74%**, and the **median of the per-video speedups was 1.532×**. Positive reductions occurred in **10/10 videos**. These are equal-weight aggregates of statistics calculated separately for each video, not ratios of pooled mode medians.

Per-video reductions ranged from **17.52% to 45.54%**; speedups ranged from **1.212× to 1.836×**. Negative reductions, if present, are preserved.

## Actual per-video measurements

Latencies below are medians, with minimum–maximum measured trial latency in parentheses, in milliseconds.

| Video | Decode only | Dense | Periodic | Speedup | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0pVGiAU6XEA | 109.10 (107.01–114.06) | 231.45 (229.57–241.17) | 142.71 (137.16–149.09) | 1.622× | 38.34% |
| AQrbRSnRt8M | 172.92 (165.05–177.09) | 307.24 (295.31–322.01) | 201.11 (191.19–208.65) | 1.528× | 34.55% |
| IcLztCtvhb8 | 162.95 (158.67–210.39) | 302.87 (287.76–339.80) | 197.07 (190.39–275.83) | 1.537× | 34.93% |
| SU_x2LQqSLs | 115.53 (113.88–116.52) | 247.01 (245.56–250.47) | 147.70 (144.16–151.91) | 1.672× | 40.21% |
| b6Q_b7vgc7Q | 192.20 (188.69–204.51) | 321.20 (319.58–338.42) | 223.26 (218.18–228.95) | 1.439× | 30.49% |
| ddvJ6-faICE | 84.42 (82.36–84.98) | 210.36 (204.99–213.01) | 114.57 (112.24–116.18) | 1.836× | 45.54% |
| ik4BW3-SCts | 138.99 (137.51–142.57) | 272.47 (262.61–276.63) | 171.06 (168.11–179.12) | 1.593× | 37.22% |
| jqRrH30V0k4 | 290.82 (257.28–328.56) | 414.24 (387.79–439.27) | 341.68 (293.75–381.83) | 1.212× | 17.52% |
| u4Rm6srmIS8 | 160.84 (158.12–184.01) | 293.10 (285.47–336.65) | 195.14 (192.86–227.56) | 1.502× | 33.42% |
| y5Iu7XkTqV0 | 169.15 (133.05–235.25) | 276.36 (267.41–432.18) | 211.02 (157.04–275.19) | 1.310× | 23.64% |

## Decode baseline and trial variability

The median across per-video median latencies was 161.89 ms for decode only, 284.73 ms for dense, and 196.10 ms for periodic. Decode-only median latency was 54.3% of dense latency as a descriptive median of per-video ratios. Decode still processes the full clip, so a 75% reduction in encoder calls need not yield a 75% reduction in total latency.

Decode-only is a separate baseline that emits no appearance features. It is not an exact additive component: CPU/GPU work, transfers, scheduling, and power state can interact. Its timing was not subtracted from dense or periodic to estimate isolated encoder time.

For **9/10 videos**, the slowest measured periodic trial was faster than the fastest measured dense trial. The median per-video saving was 97.95 ms; the median within-video min–max range was 22.77 ms for dense and 14.69 ms for periodic. These observed ranges describe whether the difference is large relative to the five-trial variability; they are not confidence intervals or a statistical significance test.

## Reproducibility and verification

Metric: **Encoded-clip-to-features latency (ms)**. The measurement starts after encoded bytes are in memory, before a fresh BytesIO/container opens, and stops after container closure and CUDA synchronization. It includes full sequential decoding, required RGB conversion/preprocessing/transfers, encoder calls, and output/cache handling. File loading, inspection, model/weight preparation, correctness checks, and result writing are excluded.

All 150 measured rows passed frame-count and encoder/cache/output counter validation. Each mode had 2 full warmups and 5 measured trials per video. Warmups are excluded from CSV rows. Mode order was shuffled per measured repetition with seed 20261010, with the actual order saved in the metadata sidecar. Verification exercised two fresh trials per mode and checked cross-trial and cross-video cache isolation.

The model was pretrained ResNet18 (ResNet18_Weights.IMAGENET1K_V1) with the classification layer replaced by Identity, producing [1,512] float32 features. It used eval and inference mode on NVIDIA GeForce RTX 3080 Laptop GPU. Documented preprocessing: [torchvision ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). Numerical settings were identical across modes; TF32 and autocast were disabled and deterministic algorithms enabled. Software decoding used 1 thread, thread type NONE; PyTorch CPU and interop thread counts were 1.

File-loading time is separately recorded for every video in the metadata and summary; observed values here ranged from 0.179 to 2.617 ms. Input SHA-256 hashes, candidate indices, weight hash, environment, and source-code hashes are recorded. Reproducible inputs/settings/order do not guarantee identical wall-clock latency on a busy laptop.

## Next experiment

Add motion-vector extraction and measure its overhead using the same encoded-byte boundary, fresh full-clip decoding, fixed settings, and shuffled mode order. Compare appearance-only periodic processing with appearance plus motion extraction.

If counters become incorrect, fix correctness before interpreting performance. If later periodic runs are slower, inspect implementation overhead and variability; if fewer calls produce little saving, profile decoding, conversion, preprocessing, and transfers.

## Limits

- ResNet18 is an appearance-feature timing prototype.
- This experiment does not measure action-recognition accuracy; reusing a feature may change its usefulness at later positions.
- It does not yet include motion-feature extraction, a motion branch, a learned refresh controller, or an action-classification head.
- File loading is measured separately and excluded from this metric.
- These low-resolution clips do not establish performance for 1080p video.
- Offline clip processing time is not a measurement of live-stream response latency.

Artifacts: measured trial CSV and its `.metadata.json` sidecar, `summary.csv`, `aggregate_summary.json`, `latency_comparison.png`, `manifest.json`, and `environment.json`.

Observed dense/periodic trial ranges overlapped for: y5Iu7XkTqV0. Interpret those clip-level differences with the visible variability.

The earlier single-video timing check for y5Iu7XkTqV0 showed 38.83% reduction (257.85 → 157.72 ms), versus 23.64% in the ten-video run. The CSVs retain both actual runs. This change is evidence that wall-clock results vary between runs; these measurements alone do not identify the cause.
