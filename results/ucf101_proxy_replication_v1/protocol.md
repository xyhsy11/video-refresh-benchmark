# Fixed protocol: ucf101_proxy_replication_v1

Protocol specified by the user, implementing the second-dataset replication motivated
by Experiments 2, 3 and 5 of “When Does Feature Fidelity Predict Task Performance?”.
No Word attachment is needed to define this run. Seed 20261026 is an experimental
seed, not an assertion about the calendar date. Client date is 2026-10-17 (Asia/Tokyo);
all execution events use explicit UTC timestamps. Starting and reviewed revisions,
tree equality, environment and preserved historical blobs are in run_configuration.json.
Do not reset the checkout or alter historical scientific outputs.

## Baseline gate

Read temporal/objective-alignment findings, protocols, seals, verification records,
head solver, timestamp/decoder, feature hooks and objective implementations. Reproduce
archived fresh X3D B4 C g and p from genuine per-source CE and the saved bootstrap.
Independently recompute 3 SHA256-selected actual feature files and selected schedules,
including explicit cache copies and affine projections. Verify both historical seals.
Missing arrays block their affected real-feature checks; table-only verification is
identified separately. No historical experiment is rerun and no vectors are invented.

## Official population and input-only eligibility

Use all 101 classes in official classInd order and official train/test split 1.
Official public URLs, archive/member/page hashes, downloaded versions and terms are
recorded. Never accept a new account or click-through license. Preserve native AVI
codecs and complete clips: no Kinetics H264 or 5–15 second filter, trimming, repetition
or padding. No video bytes are redistributed with this repository.

Group identity is (official canonical class, gXX), not gXX alone. The official folder
class is canonical; filename class spelling may differ only in letter case (e.g.
HandStandPushups versus official HandstandPushups), with original names retained.
Verify no official group occurs in both train and test before any model call.
For each class, choose exactly three official training groups by ascending
SHA256(UTF8(str(20261026)+class+group)); these are development, remaining official
training groups are training. Preserve all official test groups. Metadata membership
is fixed before any outcomes. Clip ranking uses a separate namespace:
SHA256(UTF8(str(20261026)+'clip'+class+group+filename)), lexical filename breaks ties.

Within every group, advance through ranked clips until two pass input-only screening.
Readable video, valid finite strictly increasing actual PTS, adequate complete temporal
support, 16 distinct increasing endpoints and four distinct frames/snippet are required.
If fewer than two are eligible, exclude the whole group, retain its role, and log it.
Do not replace development groups or choose clips using model outcomes/action clarity.
Require all classes and >=10 train, >=2 development, >=3 test groups/class after all
quarantines; otherwise stop with a population-design blocker. Expected planning counts
1515/303/707 groups and 3030/606/1414 clips are not measurements until verified.

Audit exact encoded hashes and available identifiers against repository cohorts and
checkpoint pretraining annotation identities. UCF filenames lack YouTube parent IDs;
unknown mappings and reuploads must be reported. Input-only perceptual screening uses
three 63-bit DCT hashes at 10/50/90% decoded frame positions. Calibrate on 256 deterministic
training clips, using nearest different-group maximum Hamming distances. Threshold is
min(4,max(0,floor(first training percentile)-1)), all three distances must pass, and
duration ratio must be >=.9. This threshold is fixed from training inputs before comparisons
to development/test inputs. Quarantine all flagged connected groups in full, never retain
cross-role copies or move roles. Log edges, all exclusions and any manual decisions before
test inference. No manual retain overrides are planned. Report screen scope and unresolved
historical/pretraining perceptual overlap. Say “second-dataset replication with disjoint
experimental groups”, never “universally pretraining-clean”.

## One frozen representation and a new head

Use only X3D-XS, PyTorchVideo revision f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0,
official checkpoint SHA256 6dafc96144c88f800b190d2cb179ff69b8fd48c40afd4deca6c1e61f76fdfe86.
Check imported revision/source hashes, actual checkpoint bytes and strict state loading.
Factory architecture unchanged (nominal 160); official inference shortside182/crop182,
uint8 RGB /255, mean .45, std .225, four frames, batch1 FP32. Read-only native projection
input hook, residual-axis mean gives 2048 pre-projection features. Native 400-class
projection still executes; include that overhead in timing and verify native logits.
No backbone fine-tuning, controller, second backbone or crop selection.

Use historical complete-clip snippet_grid: uniformly spaced valid trailing endpoints;
for each actual endpoint u request [u−47/30,u−32/30,u−16/30,u], greatest actual PTS
<=request+1e-9 seconds. Four distinct decoded frame identities; 16 strictly increasing
endpoints. Retain desired/requested/actual times, integer PTS, timebase and indices.
Full-clip candidate construction is offline. Recheck sampling on fresh real decodes.
Native MPEG4 AVI can emit decoded frames in nonmonotone PTS order. Sort REAL finite
unique PTS into presentation order before grid resolution, preserving each original
decode ordinal and presentation index. Reject duplicate/missing PTS; never substitute
frame-number/FPS timestamps. The same ordering is included inside timing boundaries.

NEW affine 101-class head trained on dense16 feature means. Fit training-only weighted
population standardization; std<=1e-12 becomes 1. Weight classes equally, groups equally
within class and their two clips equally. Apply normalized weights both to moments and
empirical training CE. Objective is weighted CE + alpha/2 ||W||², unpenalized bias.
Historical deterministic float64 torch LBFGS: lr1, max_iter400, max_eval800,
gradient tolerance1e-9, change1e-12, history50, strong_wolfe, seed20261026, one CPU thread.
Grid [.001,.01,.1,1], select lowest class/group/clip-balanced DEVELOPMENT DENSE CE,
then highest balanced accuracy, then grid order. Save every fit and convergence record;
no favorable-accuracy gate, sparse tuning or test-based model selection.

## Fixed selectors and estimands

Primary T16 B4 C. Enumerate all 455 lexicographic schedules containing candidate0.
B2 (15) and B8 (6435) are fixed secondary sensitivities. C weights are reuse counts/16,
not elapsed-time weights. U weights selected candidates equally. Pool affine features/logits,
never probabilities. Float64, historical lex-first within1e-12 of exact minimum; retain
true minimum, achieved objective and tie count. LegacyCosine is raw-feature historical
1−clamped cosine averaged on last-cache timeline, rejecting norm<=1e-12, a separate comparator.

x_t=(f_t−mu)/scale, e_t=x_r(t)−x_t, Q=I−11ᵀ/101, a_t=QW e_t.
F_local=mean ||e_t||²/D; F_pool=||mean e_t||²/D;
Z_local=mean ||a_t||²/101; Z_pool=||mean a_t||²/101.
Proxy selection functions receive no labels. Include dense16, periodic, RandomExact,
LegacyCosine, F_local, F_pool, Z_local, Z_pool and explicit true-label task_C/task_U.
RandomExact averages individual schedule CE/correctness/fidelity, not ensemble logits.
All selectors see dense references; task oracles additionally see labels. U cross-evaluates
the SAME C-selected proxy schedules and is explicitly a target mismatch.

g_clip=CE_C(Z_pool)−CE_C(LegacyCosine) is primary;
p_clip=CE_C(Z_pool)−CE_C(periodic), d_clip=CE_C(LegacyCosine)−CE_C(periodic) secondary.
Average clips within group, groups within class, then all101 classes equally.
Negative favors first selector. g alone never establishes improvement over periodic.

## Precision and exposure freeze

From development group-mean g/p variances, approximate Var(mean)=
sum_c(s_c²/G_test,c)/101². Report 1.96*sqrt(variance) halfwidth and full width.
h=.015 nats is a planning precision target, not a practical-effect or equivalence margin.
Only 2–3 development groups/class, development head selection and uncertain variance
transfer limit this approximation. No 80% power promise. Execute all eligible predetermined
test groups regardless of pilot sign or predicted precision; no additional within-group
clips masquerading as independent groups, no outcome-driven test extension/substitution.

Before ANY test feature/prediction call, seal protocol, amendments, code/transitive helpers,
checkpoint/source/head/scaler/class-order identities, split/clip membership, counts, grids,
preprocessing, precision, strata and statistical rules. Record UTC exposure time immediately
before first test inference. Enforce hashes on resume. Corrections before exposure may be
frozen and documented; changes after exposure require explicit amendment, never an untouched
confirmation claim. All real arrays have source, grid and encoder hashes in collection ledgers.

## Exact descriptive loss decomposition and secondaries

For z_D,z_S, delta=z_S−z_D, p_D,p_S:
CE(z_S,y)−CE(z_D,y) = (p_D−onehot(y))ᵀdelta + KL(p_D||p_S).
Derivation: CE difference=LSE(z_S)−LSE(z_D)−delta_y;
KL=LSE(z_S)−LSE(z_D)−p_Dᵀdelta. Addition cancels p_Dᵀdelta.
Use stable float64 max-shift logsoftmax/logsumexp and an independent scalar math.fsum
calculation. Remove common logit shifts before numerical evaluation. The Zpool−Legacy
directional and KL differences sum to g per clip/group and globally. Save contributions
and paired intervals. Labels enter this retrospective attribution only after proxy selection.
This is standard algebra and descriptive attribution, not a new theorem or causal biology.

Retain factorial aggregation=((Fpool−Flocal)+(Zpool−Zlocal))/2,
metric=((Zlocal−Flocal)+(Zpool−Fpool))/2, interaction=(Zpool−Zlocal)−(Fpool−Flocal),
all defined on CE of selected schedules. Verify local=pooled+residual variance for F and Z.
Save schedule-wise Spearman, undefined reasons, true-label oracle regret, dense agreement/KL,
per-class distributions. Dense-correct/wrong and signed true-class dense-margin terciles
are preplanned secondary strata; weighted development empirical 1/3,2/3 thresholds frozen
before test features. Clip-defined strata retain group resampling and conditional weighted
ratio estimands. Report represented classes/groups/clips, null replicates, and sparse flag
for <20 represented groups; no population claims for sparse strata. Never use strata for
eligibility or choosing the primary result.

## Inference and interpretation

10,000 class-stratified paired GROUP bootstrap replicates, numpy PCG64 seed20261026,
linear2.5/97.5 percentiles. Resample groups within each class, carry their two clips and
all methods together, recompute equal class/group/clip estimand. Save canonical group
order, indices and column weights. Report primary g mean/CI/groups/clips/width, plus
secondary marginal/unadjusted g/p/d, CE, balanced top1, ordinary clip accuracy, fidelity,
factorial/decomposition and per-class distributions. No selection of a significant secondary.
CI crossing zero is inconclusive, not equivalence; no post-hoc equivalence margin.
Intervals condition on fitted head/classes and omit training/model-selection/research history.
Present Kinetics and UCF within-dataset contrasts side by side with different heads/classes
and averaging units disclosed, never one primary raw CE average. Split1/two-clips/group
frozen features is not full official three-split benchmarking or state of the art; splits2/3
overlap sources. A negative g with unresolved/nonnegative p supports relative Legacy repair only.

## Actual system cost and resources

Before outcomes choose 30 distinct test classes by hash, one hashed group/class and one
hashed member of its frozen two clips. Batch1 FP32 encoder, CPU threads1, CUDA synchronization,
two warmups and five deterministically interleaved measured repetitions for dense16/periodicB4.
Boundary: encoded bytes in memory -> fresh FULL native decode + actual timestamp plan ->
selected RGB prep/transfers -> encoder (including native400) -> pooling/new affine head ->
CUDA-completed logits/prediction. No predecoded frames or cross-trial features. Disk load/model
setup excluded and recorded separately. Validate actual logits against saved reference features.
Record decoded frames, RGB conversions, encoder starts/completions, scoped allocated GPU
resident/incremental peaks, process working set/lifetime peak, per-clip medians/min/max/std
and paired ratios. Do not pool repeated trials as independent sources. Search CPU seconds
are measured separately on loaded real arrays with one CPU thread and exclude file I/O.
All proxy selectors need16 encodings plus search; never assign periodic latency to oracle quality.

Use existing compatible .venv_temporal; do not upgrade in place. NEW network cap20GiB
includes retries, failed/interrupted transfers and metadata. Windows native verified TLS curl;
body byte accounting plus conservative1MiB transport overhead/request. Interrupted unknown
consumption reserves full allowed response size; range responses validate Content-Range/length.
Dataset segmentation may resume verified received bytes, with official prefix and archive CRC
checks; preserve failed attempts. Disk gate20GiB free. Actual training-only 2-clip inference
smoke projects collection runtime and storage before bulk extraction. Nominal80800 snippet
calls are planning only; actual start/completion logs govern accounting. Checkpoint/resume all
long stages. Genuine blockers preserve completed artifacts and pending stages, no fake completion.

## Validation and deliverables

Focused explicitly synthetic unit tests for group roles/class-scoped gXX, weighted moments/
aggregation, PTS floor sampling, affine pooling, K101, minima/ties, scalar replay, exact loss
identity, unequal-count grouped bootstrap and seal mutation rejection. Real training smoke,
native-head check, runtime timestamp agreement and timed reference-logit agreement are
distinct empirical integration checks. Read-only audit must never regenerate scientific files.
Retain protocol/amendments, metadata/eligibility/exclusions/overlap, grids/convergence,
precision/seal/exposure/provenance, per-clip/group/class metrics, group bootstrap indices,
decomposition/secondary/strata, timing, execution accounting, verification/findings and four
rendered figures (g/p/d, fidelity/recognition, exact contributions, real measured cost).
Large real embeddings and complete logits live in ignored data/source with public hash ledgers.
Inspect rendered figures before final verified status. Preserve all old outputs, commit/push
completed implementation/evidence with actual execution state. Passing tests is not publication readiness.

```powershell
& .\.venv_temporal\Scripts\python.exe .\run_ucf101_proxy_replication.py --resume
& .\.venv_temporal\Scripts\python.exe -B .\run_ucf101_proxy_replication.py --audit-only
```
