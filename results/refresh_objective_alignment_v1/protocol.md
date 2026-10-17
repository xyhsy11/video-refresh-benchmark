# Frozen analysis protocol on reused data: refresh_objective_alignment_v1

Written 2026-10-17T03:10:12.571367+00:00, before calculating new real-data outcome tables. Starting commit
`9d2ad854044ce2684ed6907063b61837b243b0cf`; reviewed PR16 `caf5e5921d9299305b805364a0e2b91082d73611`. The starting tracked tree equals
the reviewed merge tree; no reset, historical rewrite or head fitting. Exact source,
head, input and cohort hashes are in input_provenance.json. This is not independent
preregistration; historical development choices and exposed outcomes are acknowledged.

## Fixed question, representations and endpoints
Do aggregation and frozen classifier geometry improve full-information label-free
selection? Primary is fresh-confirmation X3D-XS B4 cache-pooling
g=CE_C(Z_pool)-CE_C(LegacyCosine), equal-source mean in nats. Negative favors repair.
d=CE_C(LegacyCosine)-CE_C(periodic); p=CE_C(Z_pool)-CE_C(periodic);
a=CE_C(Z_pool)-CE_C(dense16). Improvement in g alone need not improve periodic.
No outcome changes these definitions or chooses a different objective/primary budget.
All three representations use their exact v1 frozen heads, scales, class order,
float32 arrays and 16 trailing snippets [u-47/30,u-32/30,u-16/30,u], floor PTS
resolution +1e-9 seconds, four distinct frames, endpoints unchanged. No refitting,
new normalization/calibration, controller, new architecture or head selection.
Pilot cohorts: exactly100 reused/tuned development and100 exposed v1 evaluation,
reported separately. Neither is confirmation for this new hypothesis. No protected
confirmation outcomes, features or motion tables are opened. Training arrays are unnecessary.

## Column-vector definitions, implemented as row arrays
x_t=(f_t-mu)/scale; z_t=W x_t+b; W is K by D, K=20;
Q=I-11^T/K. e_t=x_r(t)-x_t; a_t=Q W e_t.
F_local=mean_t(||e_t||^2)/D; F_pool=||mean_t(e_t)||^2/D.
Z_local=mean_t(||a_t||^2)/K; Z_pool=||mean_t(a_t)||^2/K.
Use float64; all455 schedules include0, have4 distinct ordered positions, lex order.
LegacyCosine remains original raw-feature local cosine (reject norm<=1e-12;
clamp cosine roundoff), a bridge baseline outside the squared-error factorial.
Minimization chooses lex-first within1e-12 of the true minimum; retain both the
minimum and achieved value, index, positions, ties. Pool affine logits/features,
never probabilities. C weights are candidate reuse counts/16, not elapsed seconds;
U weights selected references equally. Cross-evaluate C-selected objectives in U,
explicitly mismatched to their optimization target. Dense16, periodic range(0,16,16//B),
RandomExact and true-label task_C/task_U included. RandomExact averages individual
schedule losses/correctness, not logits or an ensemble. X3D B2/B8 (15/6435 schedules)
are predetermined secondary sensitivities in each cohort. Other reps only B4.
All selectors require dense features; task oracles additionally require labels.

## Mechanism analyses fixed before outcomes
aggregation=((CE_F_pool-CE_F_local)+(CE_Z_pool-CE_Z_local))/2;
metric=((CE_Z_local-CE_F_local)+(CE_Z_pool-CE_F_pool))/2;
interaction=(CE_Z_pool-CE_Z_local)-(CE_F_pool-CE_F_local).
These are secondary selector changes, not uniquely causal percentages.
Verify mean||e||^2=||mean e||^2+mean||e-mean e||^2, likewise a.
Common logit shifts do not affect probabilities. The CE gradient is softmax(z)-e_y,
whose norm is at most sqrt(2). The fundamental theorem of calculus along the
centered logit segment gives |CE(z_S,y)-CE(z_dense,y)|<=sqrt(2)||Q(z_S-z_dense)||.
This standard explanatory bound is not novel and minimizing its discrepancy does
not minimize true-label CE. Dense predictions may be wrong or worse than periodic.
Report actual CE,top1/macro accuracy,dense agreement,KL(dense probabilities||selected
probabilities),helpful/harmful CE changes. Per-class,dense-correct/wrong are descriptive.
Per-source schedule-wise Spearman(proxy,true-label CE), then summarize sources;
constant proxy/outcome correlation is null with reason. No schedule independence claim.

## Precision, input-only feasibility, frozen confirmation
Pilot results are for fixed analysis/debugging/variance, never formula/model selection.
h=.015 nats for g and d; V=1.25 times the maximum of average20 within-class sample
variances, over both contrasts and both pilot cohorts. N_requested=20*ceil(max(100,
1.96^2*V/h^2)/20). Report uncapped N and precision at ceiling400.
Rebuild all historical/reserved parent and encoded-hash exclusions from membership
documents, include100 exposed v1 evaluation and official K400 training parents.
Stable ascending SHA256('refresh_objective_alignment_v1|20261025|source_id') order.
Technical only: verified class/access,H264,5..15sec,actual>=16 frames,finite increasing
PTS,16 complete distinct trailing snippets. Never use recognition during screening.
No video transcoding or source replacement for network/CUDA/policy outcomes.
Traverse verified cached archives then uncached URLs lexicographically, inspect all
remaining annotated candidates within20GiB new-response-body cap incl retries/deps.
Cache reuse does not consume new cap; conservative uncertain transfer charges recorded.
Default no dependency changes. Metadata/access does not prove individual video rights;
no videos redistributed. Original parents/adjacent windows and byte duplicates excluded;
unknown reuploads and universal pretraining independence unresolved.
m=smallest eligible count across20classes within resource limit;
n_per_class=min(N_requested/20,20,m). Require>=5/class, else confirmation blocked;
finish exploratory mechanism analysis without repurposing exposed data.
Freeze N,membership,formula/code/head/checkpoint/preprocessing/grid/statistics before
ANY confirmation inference. No interim outcomes,optional stopping or extension.
If below requested, prominently report precision-limited, no guarantee of conclusiveness.
Collect real embeddings all3; use same source/snippet indices. No invented embeddings.

## Statistics, numerical checks and execution accounting
10,000 class-stratified whole-source paired resamples, PCG64 seed20261025, linear
2.5/97.5percentiles. Save shared index matrix across objectives/representations/budgets
within cohort, separate cohorts. Only fresh X3D g B4 C is primary; secondaries marginal
unadjusted. Source, not schedule/frame, is unit. Condition fixed heads/classes/source
pool/historical choices, exclude training and repeated-research uncertainty.
g CI below0 supports repair vsLegacy; above0 rejects proposed improvement; crossing0
unresolved, not equivalence. g improvement without p improvement does not establish
useful selection over periodic. Dense fidelity without better CE is insufficient.
No post-hoc equivalence margin or meaningful-effect threshold.
ATOL=RTOL=1e-10 for independent numerical reproduction; sign/tie tolerance1e-12;
float64 calculations,CSV17g round-trip parsing,JSON no NaN/Infinity, null+reason.
Before extending reproduce v1 X3D B4 oldseed20261024 and old per-source logits/losses,
correctness/schedules. Independently replay all selected schedules and sample5 full
schedules per real source/representation/budget; test objective loops,Jensen,common
shift,affine equivalence,ties and seal enforcement on explicitly synthetic unit fixtures.
Audit aggregates/bootstrap from saved source outputs without main evaluator.
All fitting calls0. Pilot encoders0/decodes0; new technical decodes and real forward
starts/completions separately counted. New selectors require16 dense snippets per
representation plus CPU search; not four-call deployed systems. No new latency
experiment or old-periodic latency assignment. Existing six-system v1 timing has
its own encoded-memory boundary and is only cited with that scope.

## Outputs, completion and commands
Retain protocol,identity/seal/source exclusions/eligibility/precision logs,raw per-source
metrics,objective distributions,selected positions/weights/logits,paired intervals,
rank/factorial/cancellation diagnostics,accounting,verification,figures and findings.
Real large arrays live in ignored data/source/refresh_objective_alignment_v1 with
hashes; no synthetic research results. Audit-only is read-only and completed calls
audit rather than overwrite. Completion requires independent checks,release validator,
and rendered-figure review. If blocked, preserve exact stage and resume command.

```powershell
& .\.venv_temporal\Scripts\python.exe -m unittest test_refresh_objective_alignment -v
& .\.venv_temporal\Scripts\python.exe .\analyze_refresh_objective_alignment.py --preflight-only
& .\.venv_temporal\Scripts\python.exe .\analyze_refresh_objective_alignment.py --input results/temporal_refresh_generalization_v1 --output results/refresh_objective_alignment_v1 --seed 20261025 --bootstrap 10000 --device cuda --max-download-gib 20
# Interrupted run: same exact arguments plus --resume
& .\.venv_temporal\Scripts\python.exe .\analyze_refresh_objective_alignment.py --resume
& .\.venv_temporal\Scripts\python.exe .\analyze_refresh_objective_alignment.py --audit-only
```
