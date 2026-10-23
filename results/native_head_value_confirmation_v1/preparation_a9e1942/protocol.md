# Experiment 27: native-head value confirmation

Study `native_head_value_confirmation_v1`, prospective seed 20261023. This is a
native-classifier transfer/robustness replication of the fixed E26 diagnostic.
Dataset, domain and head change together, so this cannot identify a head-only
causal effect. It neither fits a recognition head nor evaluates a sequential
adaptive controller. Technical feasibility and scientific success are separate.

The direct request authorizes commit and push after the work; it supersedes the
attachment's no-push sentence. No PR, merge, paid service, terms acceptance,
environment upgrade, population substitution or historical overwrite is authorized.
The direct prohibition on simulated data supersedes the attachment's request for
synthetic fixtures. Verification uses exact mathematical contracts, official
metadata, and explicitly identified replay of genuine stored fitting inputs.

## Population and exposure sequence

Retain the fixed official native400 X3D-XS checkpoint. PROBE_FIT comprises eight
independent eligible original validation components per native class, one clip
per global component: 3,200 sources. EVAL comprises m independent original test
components per class, m=8..20 chosen only by fitting-only OOF precision. Preserve
all 400 classes; never use correctness, confidence, loss, motion, gain or density
for eligibility or replacement. Official labels alone do not establish freshness.

Authenticate official train/validation/test CSVs, archive lists, native index map,
checkpoint and pinned PyTorchVideo source. Decode only authorized original RGB;
screen real presentation timestamps with no repeated-frame rescue. Exclude known
K400 train parents from both roles, protected/reserved components from fitting,
and every prior/reserved component from evaluation. Union original parents,
clip identities, encoded hashes, known global duplicate/family links and known
cross-dataset aliases. Unknown reuploads and incomplete pretraining/parent links
must remain explicit limitations rather than universal-independence claims.

Choose a single technically eligible canonical clip across all classes in a
component using SHA256 ranks before quota allocation. The rank bytes are UTF-8
`study|20261023|purpose|identifier`, documented in code. Unavailable higher-ranked
bytes have unknown eligibility; they cannot be quietly skipped for convenience.
Known fit/test or train/test collisions are quarantined. Legacy historical
exclusion unions mix exposure and reservation: prior validation exposure is
permitted for fitting when reservation status is resolved. An unresolved union
does not certify reservation of every prior source and does not certify fitting
permission. A source-role certificate and actual byte/component checks are
required before any fitting extraction. No new EVAL model calls occur during
screening, planning, parameter fitting or precision selection.

The complete technically verified evaluation manifest, frozen replacement rules
(no post-seal substitutions), input/grid identities, code/configuration, models,
scalers, inference and timing membership must be sealed before the first EVAL
model attempt. An interrupted cohort is exposed and resumes identical identities.
Failed/partial attempts remain in cumulative ledgers. A scientific correction
after exposure requires preserved originals and an explicit amendment.

## Native logits, grid and observations

Expected checkpoint SHA256 is
`6dafc96144c88f800b190d2cb179ff69b8fd48c40afd4deca6c1e61f76fdfe86`;
PyTorchVideo revision is `f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0`.
Keep the factory X3D-XS architecture and native 400-way pre-softmax projection.
Verify genuine inference against the captured pre-projection feature and actual
fixed affine head on eight preselected fitting smoke clips. No learned affine
adapter, class pruning, subset renormalization or temperature selection is allowed.
Historical real-input native verification is provenance, not the new smoke.

Use 16 uniformly spaced supported actual endpoints and trailing requests
`[u-47/30,u-32/30,u-16/30,u]`. Resolve greatest actual PTS <= request+1e-9 seconds,
requiring four distinct real frames and 16 distinct endpoints. Save integer PTS,
timebases, frame identities, requests and resolved timestamps. RGB /255, mean .45,
SD .225, short side182 and center crop182 remain unchanged. Model-zoo 4x12 is the
nominal convention; the retained implementation uses uniform4 on48 virtual frames
at30Hz, positions [0,15,31,47]. Full-clip grid construction is offline. Decoder
B-frame dependencies can entail physical lookahead even though decision inputs
use only current/past endpoints and earlier actually refreshed expensive features.

N16, mandatory0, exact K4, Periodic=[0,4,8,12], C pooling and all455 legal schedules
are fixed. C averages last-cache **logits** over 16 candidate outputs using reuse
counts/16, never probabilities or elapsed-time weights. The unchanged affine
native projection commutes with averaging; real inference must verify that
relationship including factory spatial averaging. Model-zoo multicrop accuracy
is not a promised single-grid result.

S is structural time/age/budget. H adds the unchanged historical semantics using
all400 cached probabilities; HG adds the existing32 grayscale statistics; HGH
adds the existing32 HOG-transition statistics. Dimensions are derived and frozen
in observation_schema.json: S19, H432, HG464, HGH496. HOG remains32px, replicated
central differences/2, unsigned9 bins, 4x4 cells of8px, circular interpolation,
per-cell L2 with epsilon1e-6, then16 distances from last refresh and16 from the
previous candidate. No feature, resolution, learner, architecture or budget search.

## Targets, fitting and precision

At each ORIGINAL Periodic prefix t1..12, reset independently. Force HOLD or
REFRESH, then use the exact deterministic E25 periodic-repair tail. Target
D_P=Q_P(HOLD)-Q_P(REFRESH), where Q is terminal true-label CE. Independently check
all455 schedules and both branch consequences. Each predicted action selects one
whole complete schedule. Average its twelve CE reductions within source. Never
compose twelve actions, sum gains or ensemble logits across interventions.
Prediction>1e-12 refreshes; <-1e-12 holds; otherwise preserve Periodic's action.

For S/H/HG/HGH and separately D_P/cosine drift, fit weighted ridge alpha1 with
unpenalized intercept, training-only weighted mean/population SD, SD<=1e-12 ->1,
and mean-one weights. Objective is normalized weighted half-MSE plus alpha/2 L2.
Weights are equal class, equal component within class, equal twelve states. Four
fixed component folds have exactly two sources/class/fold; no hyperparameter
selection. Report fitting/heldout errors, constants, fold variation and complete
targets, including harms, Dense-wrong clips, zeros and ties.

Using genuine fitting-only OOF source Delta_HOG effects, calculate unbiased
within-class s_c^2. Project h(m)=1.96*1.5*sqrt(sum(s_c^2)/(400^2*m)). Select the
smallest m>=8 reaching h<=.005, capped20 and subject to all-class input/resources.
Show SD inflation1 and2 as sensitivity; neither changes the primary1.5 rule.
Zero or unstable variance is disclosed and never removes the m>=8 floor.
OOF-to-new-split/final-fit variance transfer is uncertain, with no power or
coverage guarantee. If m>20 or a class lacks m, stop before EVAL exposure with
BLOCKED_PRECISION_OR_POPULATION. Fit final probes once on all PROBE_FIT and seal
their scaler/coefficient hashes. Do not enlarge EVAL after inspecting outcomes.

## Estimands and inference

Primary Delta_HOG=V(HGH)-V(HG), where V is source-averaged CE reduction relative to
original Periodic=0. Practical margin .01 nats and planning half-width .005 are
research conventions, not journal thresholds. Support V(S/H/HG/HGH), H-S, HG-H,
HGH-H, matched-tail privileged opportunity, each method's regret, Dense16,
PeriodicB4 and full455 true-label oracle quality as distinct references. Top1 is
calculated on each complete intervention before averaging correctness. Preserve
source/class distributions and adverse quantiles. Report task/drift MAE,
Spearman, AUROC/AP where defined, signed bins, constants and heldout errors.
The prespecified HOG-specific drift contrast is MAE_drift(HG)-MAE_drift(HGH).

Aggregate sources within native class and all400 classes equally. Primary95%
interval is a paired within-class whole-component percentile bootstrap with
10,000 saved PCG64 draws, named derived seed and NumPy linear quantiles. Never
resample frames, states, schedules, folds or repetitions as independent data.
Before EVAL, block shared cross-class components rather than pretending they
are independent. Independently verify weights, stored source consequences and
draw endpoints. Analytic sensitivity uses v_c=s_c^2/(400^2*m_c), V=sum(v_c),
df=V^2/sum(v_c^2/(m_c-1)), and theta +/- t(.975,df)*sqrt(V). Disclose zero variance.

These are approximate conditional95% intervals, with few-source bootstrap
limitations. They condition on fixed checkpoint, final probes/scalers, classes
and known-source assumptions; they omit fitting variation, checkpoint selection
and unknown duplicates. Supporting intervals are marginal/unadjusted. Report
both interval methods. If a practical boundary classification differs, flag
sensitivity and avoid a categorical bounded-effect conclusion.

Literal primary claims: lower>.01 material benefit; lower>0 but <=.01 positive
effect with materiality unresolved; upper<.01 rules out a >=.01 benefit at the
stated conditional scope; entire interval within[-.01,.01] bounded small mean;
upper<-.01 material harm; otherwise unresolved. Claims may coexist. Crossing zero
does not establish equivalence. A drift-decision gap additionally needs improved
matched HG-to-HGH drift prediction and appreciable privileged task opportunity;
no automatically multiplicity-controlled joint conclusion is implied. Weak
ridge adequacy or little target opportunity narrows any conclusion.

## Complete timing controls and resource gates

Keep dense-reference extraction/enumeration/fitting separate from inference
latency. Before EVAL outcomes, hash-select60 classes and one frozen EVAL source
per class. Measure Dense16, PeriodicB4, PeriodicB4+unused HG and PeriodicB4+unused
HGH. The unused arms log and ignore decisions and must preserve exactly the same
Periodic prediction and four encoder calls. They do not measure adaptive speedup.

Two warmups and five interleaved measured repetitions/system, CPU threads1,
fixed numerical settings, batch1 FP32 and CUDA synchronization. Boundary:
encoded bytes already in memory -> full fresh decode/grid -> selected RGB
preparation/transfers -> encoder/native head/pooling -> CUDA-completed prediction.
File loading is separate. Save decode/encoder counts, raw completed predictions,
scope-specific memory and non-overlapping/nested component boundaries. Collapse
repetitions within source. Report paired HG-Periodic, HGH-Periodic, HGH-HG median
and p95 source differences; do not sum nested medians or give oracle quality
Periodic latency. E26's10ms is only a descriptive reference with a different boundary.

Caps:20GiB new HTTP response-body bytes including full archives/retries,20GiB new
retained data plus separately checked peak scratch,8h collection,4h CPU work,2h
timing, no paid services. Metadata preflight prices complete gzip archives before
bulk extraction/fitting; individual range extraction is not assumed. Genuine
eight-clip fitting smoke must project throughput before bulk work. Saved-feature
replay uses zero encoder calls and is not measured compute savings. Every actual
download, decode, forward attempt/completion, fit and saved byte must be logged.
Git control traffic is separately disclosed, not represented as metered dataset
transfer. No new archives are automatically downloaded by this implementation.

## Verification and completion

Protect historical files by SHA256 inventory, without parsing protected outcome
arrays. Independent empirical verification uses explicit scalar C pooling/CE on
all455 schedules for the preselected real fitting smoke sources, then independently
reconstructs every EVAL primary consequence from stored schedule tables. Check
membership, hashes, PTS, finite outputs, bootstrap endpoints, native logits and
timing-arm prediction equality. Do not loosen tolerances. No fresh native smoke,
precision result, final fit, evaluation seal, outcome figure or scientific claim
is manufactured when prerequisites fail.

Only an actually executed, independently verified full cohort, complete real
timings and visually inspected four PDF/PNG figures can become COMPLETE.
Unexecuted stages remain explicit. See findings.md for the actual state,
resource_amendment.md for the measured route estimate, and README.md for commands.
Preserve HMDB26B BLOCKED_INPUT and all19/19.1/23A/23B gates.
