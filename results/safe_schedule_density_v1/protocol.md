# safe_schedule_density_v1 — frozen exploratory diagnostic

Frozen before new schedule outcomes. Seed **20261021**, PCG64; float64 arithmetic.
This is exposed UCF101 DEVELOPMENT, not an independent evaluation. Experiment
23A remains **NO_GO / BLOCKED_INPUT**; Experiment 23B remains frozen and unexecuted.
This diagnostic cannot clear either gate and does not recommend controller fitting.

## Inputs, cohort, and sequencing

Use exactly the 400 execution clips, 200 (class, group) units and all 101 classes
in Experiment 21's evaluation manifest, all original official split-1 training
members. Never add controller_train, selection, official-test, or reserved clips.
Authenticate the reused head from recognition_head_reuse.json, its class order,
parameters and hashes; authenticate original feature collection/provenance, input
byte hashes, snippet-grid hashes, encoder identity, dense execution and frozen
Periodic/H/H+M trial payloads/features. Missing or inconsistent inputs produce
BLOCKED_INPUT; no legacy cohort substitution. Do not run historical workflows.

Preflight saves actual HEAD, reviewed main
6a7c433b8ac9df4ca4fadf12b46d0bf30fcdae9d, subsequent-change inspection, all input
hashes, full intended membership and input-only pilot membership before computing
new outcomes. Pilot ranking is SHA256 of `20261021|input_sha256|feature_sha256`,
then clip identifier. Greedily take 20 different classes (therefore different
class/group units). If unavailable, stop; do not select by outcomes. Validate the
pilot before the full predeclared cohort. Pilot status is PARTIAL_PILOT and has
no population intervals. Full enumeration is attempted regardless of pilot signs
if technical verification passes and inputs/time permit. Interrupted work retains
per-clip journals; exact resume commands are in README. Completed stage outputs
are immutable and audits read-only. Protocol changes require a new version.

## Fixed model and schedules

Sixteen genuine X3D-XS trailing four-frame snippets give [16,2048] embeddings.
K counts snippet encoder calls; internal frames and preprocessing never change.
No GPU initialization, inference, decoding, fitting/tuning, downloads, upgrades,
or acquisition. Use pure CPU refresh_pooling_utils.schedules and logits:
Z=logits(F, parameters); Dense16=mean(Z); C=cache_weights@Z;
U=unique_weights@Z. Never average probabilities. Stable CE is
logaddexp.reduce(logits)-true-class-logit. Argmax follows NumPy first-index ties.

Enumerate all exact-K schedules containing 0: K4=455; K8=6435. Primary C pooling
averages 16 latest-cache outputs; U averages selected candidates, a distinct
offline sensitivity. Periodic4=[0,4,8,12]; Periodic8=[0,2,4,6,8,10,12,14].
Primary endpoint: **C / K4 / epsilon=0**, full-population joint density.
Epsilon sensitivities: 0.01, 0.05 nats. Numerical tolerance **1e-10** is separate.
delta_CE=CE(schedule,true_label)-CE(Dense16,true_label).
joint=correct(schedule) AND delta_CE<=epsilon+1e-10.

## Exact measurements and conditioning

For every clip/K/pool/epsilon save exact counts/denominator and fractions:
p_correct, p_CE_nondegrade, p_correct_and_CE, p_dense_agreement (fidelity only).
Save every schedule's float64 delta and prediction, plus min/5/25/50/75/95/max
quantiles (linear) and mean delta. Counts are exact, not Monte Carlo. All clips
remain in the primary full-population summary. On Dense-correct clips joint
density is conditional P_safe. On Dense-wrong clips P_safe is N/A; report
correction, CE preservation and their intersection separately. Save all four
Dense-correct/wrong -> schedule-correct/wrong transitions, for RandomExact and
frozen methods. Agreement with an incorrect Dense prediction is not safety.

RandomExact CE and correctness are means of individual schedule CE/correctness,
never CE of ensemble logits. A fixed descriptive threshold is the fraction with
CE at least 0.01 nats lower than Periodic (with numerical tolerance) without
losing Periodic correctness: if Periodic is correct, the schedule must be correct;
if Periodic is wrong, no additional correctness restriction is imposed.

Periodic rescue conditions on Dense correct AND Periodic failing joint. Report
clip/group counts, original weighted mass, conditional any-safe/no-safe fraction,
exact safe-density distribution (including zero), mean density, and whole-cohort
mass of Periodic failure with any safe alternative. Empty sets yield N/A.
No safety monotonicity across K is assumed. No new best-subset search endpoint:
the archived B4 minimum is reproduced solely as a verification check.

Use archived H/H+M positions only at B4. C replay is authenticated against actual
executed B4+C logits/features; U is offline re-pooling of those positions. K8
H/H+M are N/A (never invented). For every method, same-K/pool/epsilon:
missed_safe = method fails joint AND any admissible schedule passes.
Report this on the whole cohort and Dense-correct / Dense-wrong strata, and
conditional missed-safe rate among failures. Preserve paired comparisons.

## Aggregation and uncertainty

Equal class -> equal group within class -> equal clip within group weights.
Conditional estimates divide by qualifying weighted mass, not clip count.
Report raw counts alongside weighted estimates; schedule instances are not
independent observations. Weighted quantiles of clip densities use inverse
weighted CDF, with denominator explicitly stated. Group/class tables retain
the same hierarchy. For a complete full cohort only, use 2000 paired,
class-stratified whole-group bootstrap draws, PCG64 seed20261021. Sorted
(class,group) order, sorted classes; sample the original number of groups per
class with replacement. Save indices and ordering; carry both clips and all
methods/K/pools/epsilons together. Percentile 95% linear intervals are marginal,
unadjusted, conditional on this exposed cohort and reused selected head.
For bootstrap replicates with empty conditioning sets omit undefined ratios
and disclose valid-draw counts. Single-group classes have no within-class
resampling variance. Unknown parent/reupload/pretraining relationships remain.

## Required verification and reporting

Check schedule counts, mandatory0, unique exact budgets, no duplicate rows,
K16=Dense16, C=U for uniform Periodic, archived full-precision Dense/B4 Periodic
and B4 oracle, and saved B4 H/H+M logits. Independently scalar-pool raw last-cache
features for the first three input-ranked pilot clips on fixed schedule indices
[0, middle, last, Periodic] for K4/K8 (U scalar pooling also checked). Joint count
cannot exceed either component; density is nondecreasing with epsilon. Synthetic
tests never enter empirical tables. Hash historical tracked result blobs without
parsing protected contents; all historical scientific files must remain unchanged.

Deliver accounting, inventory/seals, per-video CSV, group/class/full summaries,
Periodic rescue, frozen-method comparisons and paired contrasts, verification,
20-clip input-selected display, two figures clearly marked EXPLORATORY/exposed
development, commands and findings answering the five user questions. Report
new encoder calls=0, fits=0, decodes=0, downloads=0. CPU diagnostic duration is
not inference latency; causal information, fresh generalization and complete
runtime remain unresolved. Preserve every adverse or null result.
