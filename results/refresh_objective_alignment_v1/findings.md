# refresh_objective_alignment_v1 findings

Fresh primary X3D B4 C: **g=-0.011773 [-0.034418, +0.008860] nats/video**, N=120. The proposed objective is unresolved at the achieved precision. Improvement over periodic is not established by its paired interval.

**Precision-limited confirmation**: available N is below the requested size.

|Cohort|N|g: Zpool−legacy|d: legacy−periodic|p: Zpool−periodic|
|---|---:|---:|---:|---:|
|Pilot: reused development|100|-0.019378 [-0.054048, +0.012660]|+0.007651 [-0.024134, +0.040273]|-0.011727 [-0.032006, +0.008076]|
|Pilot: exposed v1 evaluation|100|-0.008956 [-0.041085, +0.021685]|+0.009784 [-0.012039, +0.033162]|+0.000828 [-0.021043, +0.022817]|
|Fresh confirmation|120|-0.011773 [-0.034418, +0.008860]|+0.012905 [-0.014900, +0.043415]|+0.001131 [-0.021947, +0.023645]|

Intervals are 10,000 class-stratified paired whole-source resamples, PCG64 seed20261025, linear percentiles. Negative g favors Zpool; negative p favors Zpool over periodic. Only fresh X3D B4 C g is primary. Pilot cohorts are exposed, separate, and never pooled with confirmation. All other intervals are secondary marginal/unadjusted.

## Frozen inputs and precision

Starting commit `9d2ad854044ce2684ed6907063b61837b243b0cf` matches reviewed PR16 contents. All three heads/scalers/class orders/checkpoints/grids are unchanged. The archived X3D primary and selected logits/losses/schedules were reproduced before the extension. Actual float32 features are used; evaluation arithmetic is float64. Training arrays and protected confirmation outcomes/references/motion are not read.

h=.015; V=0.045411745; uncapped requested N=780, predicted half-width at400=0.020884. Eligible pool=196, minimum/class=6; frozen N=120. Predicted achieved half-width=0.03812848436633331. This approximation assumes pilot variance transfers and does not guarantee a conclusive interval. Per-class inventories, rejections and deterministic ranks are saved.

## Controlled mechanism, not a unique causal attribution

The four squared-error objectives change aggregation and metric separately. LegacyCosine remains a separate raw-feature bridge; differences to it also change standardization and squared versus cosine distance. Local error equals pooled error plus residual variance, identifying cancellation in the pooling calculation. Factorial contrasts concern selector changes, not percentages of semantic importance.

|Cohort|Aggregation contrast|Metric contrast|Interaction|
|---|---:|---:|---:|
|Pilot: reused development|+0.001619 [-0.015776, +0.020392]|-0.010695 [-0.028932, +0.006322]|+0.007617 [-0.020889, +0.036430]|
|Pilot: exposed v1 evaluation|+0.007875 [-0.014703, +0.030304]|-0.008742 [-0.024040, +0.005235]|+0.005315 [-0.015648, +0.027756]|
|Fresh confirmation|-0.007343 [-0.024113, +0.008682]|-0.003759 [-0.024962, +0.016108]|+0.020144 [-0.021504, +0.060768]|

Zpool reduces a centered-logit discrepancy associated with a generic CE perturbation bound. It does not optimize the unknown true label. Dense predictions can be wrong: preserving them does not guarantee lower CE or accuracy. Saved correctness strata, KL divergences, helpful/harmful counts, per-class tables and per-source schedule-wise rank correlations make this distinction auditable. Read those as descriptive secondary results, never independent schedules or source-selection criteria.

## Pooling and budgets

All C-selected schedules are cross-evaluated under U without reselection; that is a deliberately mismatched sensitivity. Separately optimized true-label C/U oracles use labels and all dense references. Their positive headroom is guaranteed by including periodic, not evidence of a usable scheduler. X3D B2/B8 sensitivities are fixed in advance; other representations remain B4 descriptive controls.

|Cohort|B|g|p|
|---|---:|---:|---:|
|Pilot: reused development|2|-0.014907 [-0.082939, +0.056022]|-0.035388 [-0.120296, +0.043962]|
|Pilot: reused development|8|+0.002975 [-0.010357, +0.017586]|+0.002353 [-0.008266, +0.012236]|
|Pilot: exposed v1 evaluation|2|-0.023292 [-0.077360, +0.022141]|+0.008970 [-0.031047, +0.047727]|
|Pilot: exposed v1 evaluation|8|+0.002222 [-0.008951, +0.013757]|-0.001394 [-0.013065, +0.010879]|
|Fresh confirmation|2|+0.001660 [-0.034348, +0.040136]|-0.004371 [-0.089971, +0.078729]|
|Fresh confirmation|8|-0.009452 [-0.022903, +0.004102]|+0.001435 [-0.009857, +0.012776]|

## Actual recognition and dense fidelity, X3D B4 C

|Cohort|Selector|CE (nats)|Top1 / macro accuracy|Dense prediction agreement|KL(dense to selected)|
|---|---|---:|---:|---:|---:|
|Pilot: reused development|dense16|0.444315|90.0% / 90.0%|100.0%|0.000000|
|Pilot: reused development|periodic|0.471060|88.0% / 88.0%|98.0%|0.012995|
|Pilot: reused development|LegacyCosine|0.478711|88.0% / 88.0%|96.0%|0.023764|
|Pilot: reused development|F_local|0.468409|88.0% / 88.0%|98.0%|0.015716|
|Pilot: reused development|F_pool|0.466219|88.0% / 88.0%|98.0%|0.007550|
|Pilot: reused development|Z_local|0.453906|90.0% / 90.0%|98.0%|0.008962|
|Pilot: reused development|Z_pool|0.459333|89.0% / 89.0%|99.0%|0.003452|
|Pilot: reused development|task_C (true-label hindsight)|0.234980|95.0% / 95.0%|94.0%|0.182363|
|Pilot: exposed v1 evaluation|dense16|0.428836|87.0% / 87.0%|100.0%|0.000000|
|Pilot: exposed v1 evaluation|periodic|0.425848|88.0% / 88.0%|98.0%|0.010100|
|Pilot: exposed v1 evaluation|LegacyCosine|0.435633|86.0% / 86.0%|96.0%|0.010656|
|Pilot: exposed v1 evaluation|F_local|0.427543|86.0% / 86.0%|96.0%|0.010289|
|Pilot: exposed v1 evaluation|F_pool|0.432761|88.0% / 88.0%|98.0%|0.006244|
|Pilot: exposed v1 evaluation|Z_local|0.416144|86.0% / 86.0%|96.0%|0.009509|
|Pilot: exposed v1 evaluation|Z_pool|0.426677|86.0% / 86.0%|96.0%|0.003194|
|Pilot: exposed v1 evaluation|task_C (true-label hindsight)|0.240840|95.0% / 95.0%|92.0%|0.118084|
|Fresh confirmation|dense16|0.484038|84.2% / 84.2%|100.0%|0.000000|
|Fresh confirmation|periodic|0.481129|84.2% / 84.2%|97.5%|0.013712|
|Fresh confirmation|LegacyCosine|0.494034|85.0% / 85.0%|96.7%|0.012647|
|Fresh confirmation|F_local|0.493362|84.2% / 84.2%|97.5%|0.009930|
|Fresh confirmation|F_pool|0.475947|85.0% / 85.0%|97.5%|0.008077|
|Fresh confirmation|Z_local|0.479531|84.2% / 84.2%|95.0%|0.011370|
|Fresh confirmation|Z_pool|0.482260|85.0% / 85.0%|98.3%|0.003724|
|Fresh confirmation|task_C (true-label hindsight)|0.258879|93.3% / 93.3%|90.0%|0.214522|

Class balance makes macro and overall accuracy coincide. RandomExact expectations, all other representation results, dense-correct/wrong strata and per-class descriptions are retained separately. The new pilot d bootstrap intervals use seed20261025; the archived seed20261024 result was reproduced exactly in legacy_reproduction.json, so a slight endpoint change here is not a changed v1 measurement.

|Cohort|g improved / tied / worsened|p improved / tied / worsened|
|---|---|---|
|Pilot: reused development|48 / 2 / 50|48 / 4 / 48|
|Pilot: exposed v1 evaluation|47 / 5 / 48|45 / 8 / 47|
|Fresh confirmation|50 / 7 / 63|58 / 3 / 59|

## Schedule-wise diagnostic alignment

|Cohort|Proxy|Mean defined per-video Spearman with CE_C|Defined / missing|
|---|---|---:|---|
|Pilot: reused development|F_local|0.012562802720117814|100 / 0|
|Pilot: reused development|F_pool|0.026943793743932853|100 / 0|
|Pilot: reused development|LegacyCosine|0.04613260064767137|100 / 0|
|Pilot: reused development|Z_local|-0.006139851917153108|100 / 0|
|Pilot: reused development|Z_pool|0.0019320913262485372|100 / 0|
|Pilot: exposed v1 evaluation|F_local|0.101772593972223|100 / 0|
|Pilot: exposed v1 evaluation|F_pool|0.10357048203361673|100 / 0|
|Pilot: exposed v1 evaluation|LegacyCosine|0.16437683542981485|100 / 0|
|Pilot: exposed v1 evaluation|Z_local|0.09379762562964508|100 / 0|
|Pilot: exposed v1 evaluation|Z_pool|0.1073297760667341|100 / 0|
|Fresh confirmation|F_local|0.024157840382046276|120 / 0|
|Fresh confirmation|F_pool|0.03360638762273356|120 / 0|
|Fresh confirmation|LegacyCosine|0.08553420466618937|120 / 0|
|Fresh confirmation|Z_local|0.03231710778980664|120 / 0|
|Fresh confirmation|Z_pool|0.047042933494359396|120 / 0|

## Cost, access and limitations

Every selector sees all16 dense candidate embeddings. Collecting those requires80 RN18 frame forwards (16 endpoint +64 four-frame-mean) and16 X3D snippet forwards per source, plus CPU search. Actual initiated and completed collection calls, retries, technical decodes, downloads and offline analysis runtime are separate accounting fields. There are zero new head/scaler/controller fits and no package upgrades. No four-call deployment cost or adaptive speedup is claimed.

The unchanged v1 timing measured encoded bytes in memory through fresh full decoding, frame preparation, transfers, encoder, pooling/head and final prediction, with its own20 sources and 2warmups/5trials. Those measurements are not timings of these dense-reference selectors. No end-to-end timing was repeated or assigned to an oracle. Full-clip candidate construction is offline. Parent/hash exclusions do not prove absence of reuploads or universal checkpoint pretraining overlap. Confidence intervals omit training, model-selection and repeated-research uncertainty. The frozen dense-trained head may react differently to sparse aggregation. This diagnostic establishes no deployed controller, streaming causality, adaptive latency, recognition guarantee or journal acceptance.

## Prior work and claim/evidence

[Knowledge distillation](https://arxiv.org/abs/1503.02531) motivates distinguishing output geometry from raw features. [Task-Oriented Feature Distillation](https://papers.neurips.cc/paper_files/paper/2020/hash/a96b65a721e561e1e3de768ac819ffbb-Abstract.html) studies task-driven feature transformations; we neither train such a transformation nor reproduce its method. [AdaFrame](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html) learns frame-selection decisions, unlike these full-information enumerations. These precedents prevent a novelty claim based solely on this design. The softmax-gradient bound is standard calculus.

|Claim|Evidence / limit|
|---|---|
|Inputs and legacy result preserved|Hashes, head selection identities, independent replay and legacy_reproduction.json|
|Objective repair improves legacy CE|Only the fresh primary g interval can support this fixed comparison; exposed pilots are exploratory|
|Useful selection beyond periodic|Requires p, not merely g or lower dense discrepancy; still full-information|
|Mechanistic cancellation|Exact algebra, not a uniquely identified causal explanation|
|Practical inference benefit|Not evaluated: dense references and search are required|

## Next decision

If the fixed fresh g and p evidence is favorable, plan one second-dataset replication with the same objectives and frozen selection rules. Otherwise resolve the observed objective-versus-dense/label mismatch and precision/data limitations before another controller. A deployable policy remains a separate study.

## Reproduction

From the repository root, use the unchanged existing `.venv_temporal` environment. See README.md for run/resume/audit commands. Large real arrays are ignored, with paths and hashes in input/collection/schedule ledgers; reconstruct only from verified original encodings using the frozen recipe. No random embedding fallback exists.

## Secondary representation results and measured decision

Only fresh X3D B4 C g is primary. The following B4 intervals are secondary, marginal and unadjusted; they do not replace the inconclusive primary result.

|Representation|g: Z_pool minus LegacyCosine|p: Z_pool minus periodic|
|---|---:|---:|
|resnet18_endpoint|-0.036778 [-0.073733, -0.002048]|-0.050240 [-0.097745, -0.004690]|
|resnet18_four_frame_mean|-0.023049 [-0.040281, -0.007366]|+0.004002 [-0.019344, +0.025689]|
|x3d_xs|-0.011773 [-0.034418, +0.008860]|+0.001131 [-0.021947, +0.023645]|

The endpoint RN18 secondary comparison favors Z_pool over both LegacyCosine and periodic; the four-frame RN18 comparison favors Z_pool over LegacyCosine but does not establish improvement over periodic. This is representation-dependent descriptive evidence, not an omnibus success or an isolated causal effect of temporal modeling.

For the primary X3D comparison, Z_pool lowers CE in 50 sources, raises it in 63 and ties in 7. The equal-source mean is negative because magnitudes differ. Its dense agreement rises from96.7% to98.3% and KL(dense||selected) falls from0.012647 to0.003724, yet primary g and periodic p intervals include zero. Both LegacyCosine and Z_pool have85.0% top1 accuracy; periodic and dense have84.17%.

|Actual claim|Evidence|
|---|---|
|Reliable X3D CE improvement over LegacyCosine|Not established: g=-0.011773,95%CI[-0.034418,+0.008860]|
|Reliable X3D improvement over periodic|Not established: p=+0.001131,95%CI[-0.021947,+0.023645]|
|Closer dense prediction preservation|Observed agreement/KL improve, without established true-label CE improvement|
|A deployable four-forward selector or speedup|Not evaluated; all16 dense references plus search are required|

**One next action:** plan a precision-powered replication of these fixed objectives, periodic and dense on a second, source-disjoint recognition dataset, with independently selected/frozen heads and membership. Do not launch another controller variant from this inconclusive result. Freeze the next population and sample size before outcomes; do not extend this120-source run until an interval becomes favorable.

This cohort is now exposed and cannot be relabeled untouched confirmation for a method designed after these results. No second-dataset acquisition is launched.

## Execution and correction record

There were210 unique technical inspections:209 inputs decoded a full video and one had no video stream. Collection added120 full-video decodes, giving329 complete video decode passes. The initial no-stream inspector failure was corrected before the confirmation seal; no frozen v1 source or scientific output was edited. Details and initial source diffs are preserved. There were11,520 initiated and11,520 completed retained encoder calls,5,760 reference vectors,zero unmatched starts,zero additional smoke encoder calls,zero model fits/upgrades andzero new end-to-end timing trials. The raw decoded_technical_candidates field counts screening attempts, as clarified in execution_accounting_addendum.json.

dataset_manifest.json is a byte-identical publication alias of the sealed confirmation_manifest.json, enabling future historical-source exclusions. Packaging command: Copy-Item -LiteralPath results/refresh_objective_alignment_v1/confirmation_manifest.json -Destination results/refresh_objective_alignment_v1/dataset_manifest.json. This adds no predictions or changed membership.

## Measured offline search cost

One float64 CPU search/analysis pass per source with one CPU thread, using already loaded real embeddings; excludes encoding, file loading and output writing inside the recorded search window. No warm-up/repetition inference-latency claim is made.

|Representation, fresh B4|Median CPU search ms|Mean CPU search ms|Min / max ms|
|---|---:|---:|---:|
|resnet18_endpoint|10.727|11.184|9.921 / 19.615|
|resnet18_four_frame_mean|10.750|11.178|10.111 / 23.604|
|x3d_xs|25.423|26.882|23.276 / 47.164|

## U cross-evaluation sensitivity and privileged cross-cells

These U scores use the unchanged C-selected schedules. The C-aligned objectives are not reoptimized for U; these are secondary target-mismatch sensitivities.

|Representation|g_U on C-selected schedules|p_U on C-selected schedules|
|---|---:|---:|
|resnet18_endpoint|-0.054139 [-0.096750, -0.012970]|-0.036014 [-0.078820, +0.005745]|
|resnet18_four_frame_mean|-0.020351 [-0.055657, +0.017573]|-0.002658 [-0.036396, +0.027982]|
|x3d_xs|+0.014817 [-0.019593, +0.050758]|-0.012938 [-0.053489, +0.021932]|

Task_C and task_U below separately optimize true-label CE under their own pooling. The other cells hold their selected positions fixed. These privileged diagnostics do not establish achievable policy performance.

|Representation|L_C(S_C)|L_U(S_C)|L_C(S_U)|L_U(S_U)|
|---|---:|---:|---:|---:|
|resnet18_endpoint|0.593610|0.751798|0.710128|0.615893|
|resnet18_four_frame_mean|0.729788|0.818318|0.871897|0.660817|
|x3d_xs|0.258879|0.286542|0.325204|0.224531|
