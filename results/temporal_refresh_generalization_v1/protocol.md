# Frozen analysis protocol: temporal_refresh_generalization_v1

Written 2026-10-16T11:11:09.523124+00:00, before evaluation inference. This protocol and subsequent hash seal
apply to new-source evaluation, conditional on reused development choices; not a
universal pretraining-independence guarantee. Actual starting commit: 0c5396f80adc9bdc047bf3809826fb680a75767b.
Design revision: ed59684e77bc481d627eb6942a52a4bfeae9dbc3; tracked difference recorded in run_configuration.json.

## Question and primary endpoint
X3D-XS, budget4, cache-duration pooling: d_i = CE(S_F) - CE(periodic).
The primary endpoint is equal-source mean d over exactly100 new official K400
validation parents,5 per each existing20 classes. Positive means reconstruction-
optimal selection harms classification; negative means helps. Above-zero95%CI
supports mismatch here; below-zero supports feature selection and contradicts a
broad mismatch claim; crossing zero is inconclusive, never equivalence.

## Roles, eligibility and amendment
Reuse exactly400 existing training and100 existing development parents from the
recognition baseline manifest. Re-extract every representation on a new snippet
grid. This resource-conscious amendment replaces the earlier600-entirely-new-
source proposal. Development previously selected recognition regularization and
is reused deliberately. Do not load old exploratory or protected confirmation
features/outcomes. Regenerate historical source/hash exclusions from membership
documents, including protected/reserved cohorts; exclude official K400 training
annotation parent IDs from new evaluation. Original encoded bytes preserved.
Deterministic cached-first archive SHA order and per-archive candidate SHA order,
seed20261024. Technical failures only: H264,5..15sec, valid increasing PTS,
16 distinct complete snippets, labels/access. No recognition-driven replacement.
Freeze100 membership before inference. No extending N after results. Parent IDs
exclude adjacent windows; unknown reuploads/near-duplicates remain unresolved.

## Inputs and encoders
T=16 desired uniformly spaced endpoints from earliest complete snippet through
last decoded PTS. For endpoint u request [u-47/30,u-32/30,u-16/30,u]. Resolve
greatest actual decoded PTS<=target+1e-9sec. Require four distinct frames and
strictly increasing endpoints; no nearest rounding, repeats or padding.
A: original ImageNet RN18 endpoint512. B: separate RN18 forwards for the same
four RGB frames then mean512. C: pinned X3D-XS revisionf3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0, official
X3D_XS.pyth, shortside/crop182, /255, mean.45/std.225, factorycrop160 unchanged.
Use pre-projection/pre-classification-activation features, mean residual axes.
Verify native400 path, strict checkpoint load and affine mean-logit equivalence.
Batch1,float32 encoders,eval/inference_mode, deterministic settings; no state
between snippets/videos. B-v-C matches observed images, not architecture,
resolution,pretraining or dimensionality; descriptive, not causal temporal effect.

## Recognition heads
One affine20-class head per representation, trained on400 dense feature means.
Training-only population standardization; std<=1e-12 replaced by1. Reuse float64
CPU multinomial meanCE + alpha/2||W||², unpenalized bias; deterministic LBFGS
max_iter400,max_eval800,grad1e-9/change1e-12,strong_wolfe,history50. Grid
[.001,.01,.1,1]. Select lowest DEVELOPMENT DENSE CE, highest accuracy, grid order.
This differs from historical accuracy-first selection. No sparse/evaluation
model selection. Freeze heads/scalers and code before evaluation inference.
Record solver convergence and dense train/development sanity, no favorable
accuracy-based source or study filtering. Estimate N100 precision using reused
development within-class variance; state approximations, no power guarantee.

## Exact schedules and pooling
Require0 and exactlyB selected candidates; B4primary,B2/8secondary; lex enumeration
15/455/6435. Periodic range(0,16,16//B). Dense16 separate.
C weights each selected feature by subsequent candidate reuse count/16.
U weights selected features equally1/B. Pool features/logits, not probabilities.
Cosine error averages raw-feature distance1-clamp(cos,-1,1) on16 last-cache
outputs. Historical convention REJECTS norms<=1e-12 rather than imputing distance.
SF minimizes this error without labels (still privileged full dense references).
SC/SU minimize true-label CE under their own pools; both privileged hindsight.
Lex first within1e-12 of true numerical minimum; retain minimum and achieved
objective/tie counts. ExactRandom averages individual schedule CE/correctness,
not ensemble logits. Cross-evaluate every chosen schedule under both pools.
Record LC(SC),LU(SC),LC(SU),LU(SU), W_C=LU(SC)-LC(SC), W_U=LU(SU)-LC(SU),
H_C=LC(P)-LC(SC),H_U=LU(P)-LU(SU), schedule agreement/Jaccard. Headroom is
nonnegative by construction, not evidence that a practical policy recovers it.
CE optimization does not guarantee top1 improvement.

## Statistics and secondary analyses
10000 class-stratified whole-source paired resamples,PCG64 seed20261024,
5/class in evaluation and reused development. Share saved indices across all
representations/budgets/pools/comparisons within cohort; separate cohort files.
Linear2.5/97.5 percentiles. Report raw source deltas, mean CE,top1/macro accuracy,
per-class descriptions, oracle headroom, selection/pooling cross-evaluation,
other representations and B2/8 as secondary marginal/unadjusted analyses.
No unstable per-video oracle-gap ratios. Intervals condition on heads/classes,
exclude training,selection,dataset/repeated-research uncertainty. No pooling
of development and fresh evaluation. No new endpoint after exposure.

## Timing and resources
LocalCUDA only;20GiB hard cap all new HTTP body bytes incl deps/weights/retries;
verified cache reuse free. Separate add-on environment preserves historicalenv.
Training-only smoke estimates runtime,disk,memory before bulk collection.
After quality seal, time20 evaluation sources1/class by frozen stablehash,
six actual systems A/B/C dense16/periodicB4. Two warmups,five measurements,
seeded interleaving,CPUthreads1,batch1,CUDA sync only at boundaries. Fresh
BytesIO,full sequential decode each trial. Encoded-memory→decode/preprocessing/
transfer/encoder/pooling/head/completed prediction. Disk load/setup/downloads
excluded and recorded. B makes4 independent frame forwards/refreshed snippet,
no cross-snippet feature caching. Equal B is not equal compute. Record counts,
per-video medians/min/max,offline clip throughput and scoped GPU/CPU memory.
Never place free-oracle quality at measured deployable latency.

## Verification, outputs, limits
Timestamp floor/prefix and pixel isolation; distinct frames; independent scalar
oracle/pooling fixtures; checkpoint/native/affine checks; source/hash isolation;
independent selected-schedule cache/unique replays and saved-output/bootstrap
reproduction; historical release validator once; immutable old scientificfiles.
Save cohort/exclusion/eligibility/acquisition records, identities, environment,
head grids,seal,features outsideGit with hashes, per-video quality/cross-pooling,
bootstrap arrays,raw timing,verification,figures,findings and reproductioncommands.
Offline candidates use whole clips; no streaming claim. Cross-backbone changes
can involve accuracy/calibration/pretraining/aggregation. Does not establish old
predictors fail on X3D. Newly exposed evaluation cannot later confirm a controller
designed after reading it. No architecture search/new refresh predictor, model
redesign,recognition accuracy guarantee,adaptive speedup or publication promise.


## Pre-exposure execution clarification

The actual technical cohort is100 new distinct parents,5/class,with697 prior/reserved parent IDs and690 recorded encoded hashes excluded. Source membership and all SHA256 values are in the paired JSON manifests/seal. No fresh recognition inference has occurred at this clarification.

Actual X3D checkpoint SHA256: 6dafc96144c88f800b190d2cb179ff69b8fd48c40afd4deca6c1e61f76fdfe86. Strict loading/native checks measured2048 features and projection input [1,1,2,2,2048]; activation=None. Factory crop160 remains unchanged while inference crop182 is used. Native400 projection still executes during feature extraction; this unused native-head work is included in C timing.

End-to-end timing includes within-trial sequential decode into temporary frame objects,full timestamp-grid construction,selected RGB conversion and prediction. No predecoded frames or free candidate inspection are supplied. All temporary buffers are trial-local. Float64 independent algebra checks use1e-10 absolute/relative; native float32 affine check2e-5; timing-versus-reference logits1e-4 absolute/relative,declared before evaluation. Sign/tie selection tolerance remains1e-12; timestamp floor tolerance1e-9sec.

Before exposure, Windows atomic JSON replacement gained bounded retry; collection batches only output transfers (encoder batch remains1); numeric frame-type names were losslessly converted to I/P/B. These engineering corrections did not select data/models by evaluation outcomes. Interrupted forward starts remain recorded separately. The initial2-source smoke estimate was conservative and includes initialization effects; it did not guarantee total runtime.

Class ordering (index0 onward):
["drinking shots", "playing flute", "playing violin", "unloading truck", "crossing river", "blasting sand", "cooking egg", "swinging on something", "tai chi", "eating hotdog", "disc golfing", "shoveling snow", "skipping rope", "washing hair", "javelin throw", "opening bottle", "petting cat", "waxing chest", "jumpstyle dancing", "trimming trees"]
