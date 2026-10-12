# Motion-vector extraction overhead alongside periodic RGB caching

Actual measurement session: 2026-10-12T06:22:12.425813+00:00 to 2026-10-12T06:24:14.001612+00:00 (UTC). The October 11 directory name is an experiment label, not the collection date.

Validated 10 original videos and 300 CUDA trials, with two warmups and five measured repetitions per condition. All six controls/workloads were measured in the same session; October 10 numbers were not substituted.

Periodic_mv had a lower median than dense_mv in **10/10 videos**. The median of per-video motion-enabled reductions was **25.86%**, with range **17.72% to 31.97%**. The median of per-video motion-enabled speedups was **1.349x**.

The corresponding same-session RGB-control median per-video reduction was **35.96%**. The median of per-video motion-minus-control reduction changes was **-11.48 percentage points**, with range -16.73 to +1.75 points. These aggregate statistics weight videos equally; a ratio of aggregate latency medians is a different calculation.

## Measured configuration overheads

Each difference below is the median latency of an export-and-descriptor configuration minus its corresponding control. It includes decoder interactions, vector access, CPU summaries and descriptor retention, plus scheduling variability. It is not an exact isolated component timing. Negative values are preserved.

| Video | decode_mv - decode (ms) | dense_mv - dense (ms) | periodic_mv - periodic (ms) | Control reduction | Motion reduction | Motion speedup | Change (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0pVGiAU6XEA | +79.60 | +78.53 | +86.10 | 40.16% | 29.59% | 1.420x | -10.57 |
| AQrbRSnRt8M | +82.42 | +91.63 | +103.57 | 33.57% | 23.81% | 1.312x | -9.77 |
| IcLztCtvhb8 | +89.34 | +34.28 | +85.69 | 41.86% | 25.86% | 1.349x | -16.00 |
| SU_x2LQqSLs | +61.50 | +58.31 | +75.18 | 40.32% | 27.92% | 1.387x | -12.40 |
| b6Q_b7vgc7Q | +85.86 | +45.16 | +88.70 | 31.44% | 17.72% | 1.215x | -13.71 |
| ddvJ6-faICE | +81.48 | +51.64 | +74.90 | 48.70% | 31.97% | 1.470x | -16.73 |
| ik4BW3-SCts | +65.93 | +81.00 | +76.32 | 31.40% | 25.86% | 1.349x | -5.54 |
| jqRrH30V0k4 | +142.32 | +148.81 | +151.27 | 29.15% | 21.09% | 1.267x | -8.06 |
| u4Rm6srmIS8 | +86.77 | +115.11 | +78.84 | 24.87% | 26.62% | 1.363x | +1.75 |
| y5Iu7XkTqV0 | +93.41 | +64.41 | +91.86 | 38.35% | 23.68% | 1.310x | -14.67 |

Across videos, the median configuration overheads were:

- decode: +84.14 ms, range +61.50 to +142.32 ms; median per-video relative difference +46.49%.
- dense: +71.47 ms, range +34.28 to +148.81 ms; median per-video relative difference +24.32%.
- periodic: +85.90 ms, range +74.90 to +151.27 ms; median per-video relative difference +46.13%.

## Trial variability and consistency

Observed periodic_mv and dense_mv ranges were fully separated, with all periodic trials faster, in 6/10 videos. Five repeats do not establish statistical significance.

The table below flags a configuration difference as small relative to variability when its absolute median difference is no larger than the larger of the two observed latency ranges. Range overlap is reported independently; this descriptive rule is not a statistical test.

| Video | Small vs trial range: decode / dense / periodic | Ranges overlap: decode / dense / periodic |
| --- | --- | --- |
| 0pVGiAU6XEA | no / yes / yes | no / no / no |
| AQrbRSnRt8M | no / no / no | no / no / no |
| IcLztCtvhb8 | yes / yes / yes | yes / yes / yes |
| SU_x2LQqSLs | no / no / no | no / no / no |
| b6Q_b7vgc7Q | no / yes / yes | no / yes / no |
| ddvJ6-faICE | yes / yes / yes | yes / no / yes |
| ik4BW3-SCts | no / yes / yes | no / no / no |
| jqRrH30V0k4 | no / yes / no | no / no / no |
| u4Rm6srmIS8 | no / yes / yes | no / yes / no |
| y5Iu7XkTqV0 | no / no / no | no / no / no |

In **2/50 same-repetition comparisons**, periodic_mv was slower than dense_mv despite the lower per-video medians. Conditions within a shuffled repetition run sequentially, not simultaneously; this is a descriptive variability check, not a causal/statistical paired test.

| Video | Repetition | Dense_mv (ms) | Periodic_mv (ms) | Repetition reduction |
| --- | ---: | ---: | ---: | ---: |
| b6Q_b7vgc7Q | 1 | 400.49 | 517.77 | -29.28% |
| ddvJ6-faICE | 2 | 283.40 | 302.61 | -6.78% |

## Export and descriptor correctness

A single pass over the selected videos decoded 3096 frames: 3074 with vectors and 22 without. It exported 2281756 entries: 1691796 past-reference, 589960 future-reference and 0 other entries. Counts and descriptors matched exactly across all three motion modes and both verification trials; appearance features matched the original dense control at refreshed positions (rtol=1e-5, atol=1e-6).

Every decoded frame gets a fresh retained CPU descriptor, including missing-vector frames. Missing data uses zero descriptors with mv_available=false; observed zero displacement retains mv_available=true. dx and dy are computed after float64 conversion; magnitudes use numpy.hypot, an unweighted mean, and numpy.percentile(..., 90, method='linear'). Invalid scales/fields stop processing explicitly.

## Interpretation and next experiment

Median per-video relative configuration differences were +46.49% of decode, +24.32% of dense, and +46.13% of periodic control latency. Of 30 measured median configuration differences, 30 were positive and 0 were negative. Negative differences are retained.

The percentage caching saving decreased in 9/10 videos. The median paired change was -11.48 percentage points; it is calculated from each video's change, not by subtracting the two aggregate medians.

The saving increased for u4Rm6srmIS8; this mixed direction is reported rather than forced into the expected trend. Interpret that exception with its observed trial ranges. Motion configuration overhead was small relative to the larger trial range in 7/10 dense comparisons and 6/10 periodic comparisons.

RGB caching still saves processing time in every video after adding the CPU motion workload. Assess motion usefulness and recognition accuracy next, while profiling vector export and percentile computation to understand the added cost. Use longer runs and control laptop power/background load before attributing small changes to a component.

## Scope and limits

- These are simple CPU codec displacement descriptors, not optical flow, physical velocity, measured object motion, or an evaluated motion-recognition branch. No motion encoder or descriptor GPU transfer is included.
- The original encodings contain B-frames; exported vectors can reference future frames. This is offline full-clip overhead measurement and does not establish strictly causal online inference. Preserve these encodings; test controlled I/P-only encoding separately.
- ResNet18 remains an appearance-feature timing prototype. Recognition accuracy, stale-feature effects and motion usefulness for adaptive refresh decisions remain untested; there is no action head or learned refresh controller.
- File loading and diagnostic inspection are measured separately and excluded. All modes decode the full clip anew. Decode-only is a separate baseline, not an exact additive component.
- These low-resolution clips do not establish 1080p performance. Ten videos, five repeats and one laptop are preliminary; offline clip latency is not live-stream response latency.
- GPU telemetry before and after the measurement session is in metadata. Concurrent graphics activity and power/scheduling state can affect observed trials.

![Six-condition per-video latency](latency_comparison.png)

![Caching reductions with and without motion](caching_reduction_comparison.png)
