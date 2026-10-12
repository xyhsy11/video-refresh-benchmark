# Cache-drift prediction diagnostic

Actual CUDA collection: 2026-10-12T11:05:39.010781+00:00 through 2026-10-12T11:05:46.246165+00:00. 10 videos, 160 fresh float32 ResNet18 features, 150 pre-decision drift observations. This is reference collection and offline prediction evaluation, not a latency measurement.

## What was predicted

Appearance-feature drift is cosine distance between each current fresh feature and the previous periodic cache. Candidate 0 is excluded. At t=4/8/12 the label compares against candidate 0/4/8 BEFORE refreshing, so scheduled refreshes do not create artificial zero labels. Predictors receive four cache-age inputs, six motion inputs, or both; never RGB features, target-derived inputs, video identity, or future decoded-frame positions. Interval descriptors cover frames strictly after the cached candidate through the current candidate inclusive. Missing vectors retain availability flags.

## Held-out results

Whole source videos are held out. Each fold trains on the other nine videos, including fitting standardization there only. Ridge alpha is fixed at 1.0 with an unpenalized intercept and no tuning. Raw and [0,2]-clipped predictions are saved; primary metrics consistently use clipped predictions. Smaller MAE/RMSE means more accurate drift prediction. Spearman measures whether larger predictions tend to accompany larger observed drift; constant-vector correlations are undefined and saved as missing.

| Predictor | Mean per-video MAE | Median per-video MAE | Mean per-video RMSE | Mean defined per-video Spearman |
| --- | ---: | ---: | ---: | ---: |
| constant | 0.083268 | 0.062143 | 0.094466 | undefined |
| age_only | 0.100206 | 0.068788 | 0.112470 | 0.3738 |
| motion_only | 0.074165 | 0.051249 | 0.087422 | 0.1807 |
| age_plus_motion | 0.081251 | 0.051454 | 0.096673 | 0.2850 |

Age plus motion versus age only, calculated per video before equal-weight aggregation:

- Mean paired absolute MAE improvement: **+0.018955**; median: **+0.009576**.
- Mean paired percentage improvement: **+9.79%**; median: **+8.24%**; range **-34.82% to +61.73%**. Percentage improvement is undefined when age-only MAE is zero.
- Improved **6**, tied **0**, worsened **4** videos. Positive improvement means smaller MAE; negative results are retained.
- Benefited: IcLztCtvhb8, b6Q_b7vgc7Q, ddvJ6-faICE, jqRrH30V0k4, u4Rm6srmIS8, y5Iu7XkTqV0. Worsened: 0pVGiAU6XEA, AQrbRSnRt8M, SU_x2LQqSLs, ik4BW3-SCts.
- Motion only is **stronger** than age only by equal-weight mean per-video MAE (0.074165 versus 0.100206). The lowest mean MAE among the four models is **motion_only**.

| Video | Age-only MAE | Age-plus-motion MAE | Absolute improvement | Percentage improvement | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| 0pVGiAU6XEA | 0.039022 | 0.052609 | -0.013587 | -34.82% | worsened |
| AQrbRSnRt8M | 0.045002 | 0.050044 | -0.005042 | -11.20% | worsened |
| IcLztCtvhb8 | 0.061457 | 0.043583 | +0.017874 | +29.08% | improved |
| SU_x2LQqSLs | 0.046080 | 0.050299 | -0.004219 | -9.16% | worsened |
| b6Q_b7vgc7Q | 0.092580 | 0.078502 | +0.014078 | +15.21% | improved |
| ddvJ6-faICE | 0.076120 | 0.071046 | +0.005075 | +6.67% | improved |
| ik4BW3-SCts | 0.042151 | 0.047587 | -0.005436 | -12.90% | worsened |
| jqRrH30V0k4 | 0.241542 | 0.136588 | +0.104954 | +43.45% | improved |
| u4Rm6srmIS8 | 0.078461 | 0.030026 | +0.048435 | +61.73% | improved |
| y5Iu7XkTqV0 | 0.279642 | 0.252223 | +0.027419 | +9.81% | improved |

Consistency is assessed from these ten paired outcomes, not from treating 150 within-video observations as independent samples. Large relative changes for videos with small age-only MAE can influence the mean percentage improvement; the median and absolute improvements are shown alongside it. These are descriptive results, with no statistical significance claim. Shared training folds also make out-of-fold errors dependent.

The two largest absolute improvements (jqRrH30V0k4, u4Rm6srmIS8) account for 80.92% of the net summed improvement. The other videos' mean absolute improvement is +0.004520. This describes concentration after evaluation; no observations were removed from primary metrics and no model was retuned.

The constant training-mean baseline has mean per-video MAE 0.083268, compared with 0.100206 for age only. The richer models should therefore be judged against the constant baseline as well as age alone. Age-plus-motion mean per-video Spearman is 0.2850, versus 0.3738 for age only. Lower average MAE does not establish better within-video ranking, which matters for selecting refresh positions.

![Per-video MAE](mae_age_vs_motion.png)

![Held-out predictions](predicted_vs_observed.png)

## Recommendation and limits

The average improvement is mixed across videos. Keep cache age as the baseline and validate motion on a larger untouched set before relying on it for refresh decisions.

A drift predictor could eventually inform when to recompute appearance features, but this diagnostic does not evaluate refresh decisions, measured refresh benefit, latency savings, or action-recognition accuracy. ResNet18 is an appearance-feature prototype, with no evaluated action-classification head. Low drift need not preserve semantic decisions; high drift need not indicate an action error.

Motion inputs are simple CPU summaries of codec displacement, not optical flow, physical velocity, or an evaluated motion-recognition branch. The original encodings include B-frames and future-reference vectors. Aggregation uses only current/preceding decoded-frame positions, but the underlying vectors can refer to future frames: this is an offline diagnostic, not strictly causal streaming inference. Original encodings are preserved; controlled I/P-only encoding is a separate experiment.

Only ten independent source videos and one encoder are evaluated. Generalization to new actions, compression settings, resolutions, and hardware, the usefulness of motion for refresh decisions, and recognition accuracy remain uncertain. No alpha or descriptor tuning was performed on these held-out videos. Use a new, larger evaluation set for subsequent model choices rather than repeatedly optimizing this pilot.

## Reproduction and audit

See [observations](observations.csv), [predictions](predictions_oof.csv), [per-video metrics](per_video_metrics.csv), [paired comparisons](paired_improvements.csv), [aggregates](aggregate_metrics.json), [folds and fitted parameters](folds.json), [collection metadata](metadata.json), and [verification](verification.json). [Candidate features](features/) and [per-frame motion](motion_frames.csv) permit independent interval audits. Metadata records real timestamps, video/weight/source hashes, software, decoder/export settings, and all definitions. Original release validation and preservation checks passed. No videos or downloaded model weights are included.
