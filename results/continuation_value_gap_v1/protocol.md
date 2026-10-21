# Frozen Experiment 25 protocol

The complete user-supplied scientific specification follows verbatim. The direct user request additionally authorizes commit and push; it supersedes the attachment closing no-push restriction. No publication or merge is authorized.

Implementation choices fixed before endpoints: matched action ties also preserve Periodic; exact full-schedule bound uses the true minimum (first argmin for its descriptive accuracy); quantiles use inverse weighted empirical CDF; tie sensitivity uses 0,1e-12,1e-10,1e-8 nats; raw replay selects three clips per dataset by SHA256(20261026|clip), checks all455 schedules. All states enter the scatter, marker area proportional to hierarchical state weight; equal x/y limits within dataset show the complete range. No full UCF interval is substituted. Uniform tail branch means use individual losses. Exact state reset is t1..12; zero is mandatory, t13..15 have exhausted Periodic budget. Synthetic fixtures run before the seal and before new empirical endpoints.

The verifier synthetic-only command imports the implementation to test its contracts; the empirical verification path never imports or calls the main endpoint implementation. All empirical suffixes and state outcomes are reconstructed independently.

EXPERIMENT 25 — CONTINUATION VALUE GAP
Repository: xyhsy11/video-refresh-benchmark
Study identifier: continuation_value_gap_v1
Reference commit reviewed: aa147a1f98b1f82ca1bc3866bbceacf4633553af
Prepared: 2026-10-21

You are working in my local research repository. Implement, validate, and run the experiment below using the existing environments and retained real data. Produce a reviewable scientific result, including adverse or inconclusive outcomes. Do not stop after proposing a plan when the next authorized stage is feasible.

Research question:
Does a refresh action preferred by an oracle that optimizes every later decision still improve final recognition when the remaining refresh budget follows a fixed, executable continuation?

This is an offline diagnostic with a previously outcome-unexposed validation population. It does not train a new controller or establish a deployed speedup.

The study measures dependence on the assumed continuation. It does not test whether permitted observations predict either advantage or whether retraining on a different target improves a controller.

1. START WITH THE LOCAL REPOSITORY

Read applicable AGENTS.md instructions, current experiment documentation, and git status. Preserve my unrelated changes. Inspect commits newer than the reference above and identify changes affecting this design; do not reset the repository to the reference. If this experiment already exists, verify and resume it instead of duplicating it. Use an isolated branch or appropriate existing worktree.

Use the existing Windows Python environments, preferring .venv_temporal for temporal inference. Preserve installed Torch, torchvision, PyTorchVideo, and CUDA versions. Record the actual executable and dependencies. Use portable pathlib-based code and provide commands that work in PowerShell. Do not replace environments, upgrade dependencies, download datasets or checkpoints, or accept new external terms for this task.

Implement new output paths and explicit adapters. Several historical inference classes write through module-global OUT paths. Do not invoke old runners, monkeypatch historical guards, or overwrite earlier evidence.

Keep experiments 19/19.1/23A/23B and their statuses unchanged. This instruction authorizes the distinct two-role HMDB offline diagnostic described below; it does not turn the previously blocked four-role controller experiment into a successful one. Leave protected UCF official-test artifacts outside this study.

2. SCIENTIFIC SCOPE AND FROZEN ORDER

Stage 25A: exact CPU analysis of the existing 400-clip exposed UCF execution population. This is development evidence, regardless of its result.

Stage 25B: the same diagnostic on all eligible official HMDB51 test clips, using one fixed-recipe recognition head fitted on all eligible official-training clips remaining after the existing conservative test-family quarantine.

The primary validation endpoint is HMDB's signed mean final-CE gain G_star under the deterministic continuation defined below. UCF supplies development evidence and a separately reported estimate. Never pool raw CE across the two datasets.

Freeze the common scientific definitions, human-readable protocol, machine-readable configuration, implementation hashes, authenticated UCF identities, estimands, numerical tolerances, statistics, and decision rules before computing the new UCF endpoints. Missing HMDB assets must not block the UCF seal or Stage 25A. Separately freeze HMDB membership and input identities before HMDB training. Before any new HMDB test model call, additionally seal the trained head, scalers, encoder identity, input hashes, temporal grids, and evaluation manifest. Record timestamps and actual exposure events. Run 25B regardless of the sign of the UCF result if the technical and resource gates pass.

Changes after a seal require a dated amendment explaining whether they were mechanical bug fixes or scientific design changes. Do not quietly rewrite a seal or call a revised analysis untouched confirmation. Mechanical fixes must preserve all original evidence.

3. REQUIRED READS AND REUSABLE INPUTS

Inspect:
- results/task_utility_observability_v1/protocol.md
- results/task_utility_observability_v1/findings.md
- results/task_utility_observability_v1/prediction_inventory.json
- results/task_utility_observability_v1/states_execution_periodic.json.gz
- task_utility_observability.py
- results/safe_schedule_density_v1/protocol.md
- results/safe_schedule_density_v1/findings.md
- results/safe_schedule_density_v1/full/clips/*.npz
- results/safe_schedule_density_v1/full/per_video.csv
- results/safe_schedule_density_v1/cohort_manifest.json
- results/safe_schedule_density_v1/input_inventory.json
- results/safe_schedule_density_v1/design_seal.json
- refresh_pooling_utils.py
- analyze_safe_schedule_density.py
- results/causal_observability_population_v2/findings.md
- results/causal_observability_population_v2/per_class_eligibility.csv
- results/causal_observability_population_v2/source_family_audit.json
- results/causal_observability_population_v2/screening_evidence.jsonl.gz
- results/causal_observability_population_v2/allocation_manifest.json
- causal_observability_population_v2.py
- ucf101_proxy_math.py
- ucf101_proxy_common.py
- ucf101_proxy_inference.py
- temporal_refresh_common.py

Existing useful pure APIs include schedules, feasible_ids, task_targets, independent_completion_targets, adaptive_grid, graph_replay, and availability. Inspect their actual contracts and side effects before reuse.

Authenticate the 400 UCF clips, class order, exact schedule ordering, head identity, and archived loss arrays. Recover each schedule's absolute CE from its stored CE delta plus the authenticated per-clip Dense CE when necessary. Reproduce archived Periodic, Dense, and task-oracle means before computing new quantities.

Record the original loss-array dtype and numerical provenance before computing new endpoints. Casting a coarse archived array to float64 does not restore precision. If the stored precision cannot support the 1e-10 tie rule below, document a prospective precision/tie amendment or mark affected comparisons numerically indeterminate in a declared sensitivity analysis. Do not silently change the rule after seeing the outcomes; preserve the useful descriptive loss-array analysis and its precision limitation.

Stage 25A's primary analysis must run from the authenticated exact schedule outcomes. Do not require raw videos or new encoder calls merely to perform loss-array arithmetic. If genuine candidate logits/features are unavailable, record that limitation for independent raw-feature checks and the Z_pool secondary; do not invent these inputs or block an otherwise verified primary analysis.

4. FIXED SCHEDULE PROBLEM

Use X3D-XS, 16 genuine trailing four-frame snippet candidates, mandatory refresh at position 0, exactly K=4 refreshes, and the repository's C pooling convention.

There are exactly choose(15,3)=455 legal schedules. Enumerate all of them, preserving the correct held-cache contribution at every candidate position. Compute stable float64 terminal cross-entropy from the final pooled logits and the true class.

The baseline schedule is P=[0,4,8,12].

Use all 12 discretionary states t=1,...,12 on the ORIGINAL Periodic trajectory for every included clip. The prefix consists of all baseline refreshes strictly before t, together with the implied previous holds. At each state independently reset to that original prefix. Do not roll previously selected interventions into the next state. Positions 0 and 13,...,15 are outside the primary discretionary population; document why.

Retain all clips, including Dense-wrong and Periodic-wrong clips. Do not select states by head confidence, oracle gain, agreement, correctness, or safe-schedule density.

For state s and forced action a in {HOLD=0, REFRESH=1}, let S_a(s) be all complete legal schedules consistent with the entire prefix and the forced current action.

Define:
Q_star(s,a) = min over S in S_a(s) of L(S)
D_star(s) = Q_star(s,HOLD) - Q_star(s,REFRESH)

Positive advantage favors REFRESH. Choose a_star by minimum Q_star; for an action-value difference within 1e-10 nats, take the original Periodic action. Report numerical-tie frequency and a tie-sensitivity range without changing the primary tie rule.

This reproduces Experiment 22's budget-aware oracle target. The old H/H+M controllers were trained using Z_pool, not this target. Do not claim that training on D_star caused their historical failures.

5. THE DETERMINISTIC EXECUTABLE CONTINUATION

After forcing action a at t, set b=4-len(prefix)-a and consider future positions t+1,...,15.

- Begin with the original Periodic anchors {4,8,12} that remain in those future positions.
- If there are more than b anchors, discard the EARLIEST anchors until b remain.
- If there are fewer than b anchors, add the EARLIEST available future non-anchor positions until b positions remain.
- Execute those positions in chronological order.

This rule depends only on time and remaining budget. It reads no true labels, current uncomputed features, future features, confidence values, or oracle outcomes.

Required examples:
- At t=2, an extra refresh yields [0,2,8,12].
- At t=4, holding yields [0,5,8,12].
- Taking the original Periodic action at every original Periodic prefix must reproduce P exactly.

Implement an explicit sequential continuation as well as the direct schedule constructor, and verify their equivalence. Both branches must remain legal and use exactly four refreshes.

Let Q_P(s,a) be terminal CE of this complete schedule and D_P=Q_P(HOLD)-Q_P(REFRESH). Call the rule a periodic-repair continuation, with its exact definition attached; do not imply it is the unchanged original Periodic policy after an intervention.

6. PRIMARY AND SECONDARY ENDPOINTS

For each clip i, calculate:
g_i = mean over its 12 states of [L_i(P) - Q_P(s,a_star)].

G_star is the class/family/clip-weighted mean of g_i, as defined in Section 9. Positive means a privileged oracle-selected single intervention helps under this fixed tail; negative means it harms.

This is equivalent to choosing ONE intervention time uniformly among the 12 states and then returning to the frozen continuation. It is not a policy that uses oracle actions at every time. Do not sum state gains or claim this is a learned/deployable decision rule. Although the continuation is executable, a_star uses privileged labels and future outcomes.

Prespecified secondary endpoints:
- G_matched = mean [L(P)-min_a Q_P(s,a)]. This is the privileged best SINGLE intervention available under the same continuation.
- R_star = mean [Q_P(s,a_star)-min_a Q_P(s,a)].
- Verify R_star = G_matched-G_star within numerical tolerance.
- Numerical-tie frequency, action disagreement between D_star and D_P, and disagreement on the predeclared subset |D_P| >= 0.01 nats. Give all denominators and paired loss consequences.
- Signed gain and regret distributions, per-class estimates, tail quantiles, and the fraction of clips with material benefit or harm.
- Final accuracy and correct-to-wrong/wrong-to-correct transitions for the same single-intervention experiment. CE changes are not automatically classification errors.
- Compute that accuracy as the weighted mean of correctness indicators for the individual complete schedules at the 12 alternative intervention states. Do not average logits across the alternative interventions and classify the resulting ensemble.
- Dense, Periodic, full-schedule task oracle, and the matched single-intervention bound, with clearly different meanings.
- If genuine candidate logits are authenticated, compute the existing conditional Z_pool teacher's one-action gain using the SAME prefix and tail. Reproduce its original objective and tie rule exactly. Missing UCF logits make this secondary unavailable, not the primary invalid. HMDB extraction should retain the inputs needed to compute it.

R_star and G_matched are nonnegative by construction. Their positivity is not a discovery or success criterion. Their magnitude matters.

Verify the accounting identity:
r_a = Q_P(s,a)-Q_star(s,a) >= 0
D_P-D_star = r_HOLD-r_REFRESH.
Present this as verification/explanation, not a new theorem.

7. UNIFORM-CONTINUATION SENSITIVITY

For the same state and forced action, define:
Q_U(s,a) = arithmetic mean of the INDIVIDUAL terminal schedule losses over S_a(s).

Do not apply CE to the mean logits across schedules.

A uniform legal suffix has a causal implementation: refresh with probability remaining_budget/remaining_positions, forcing actions at feasibility boundaries. Verify this distribution analytically or by exact probability recursion on small fixtures; do not estimate the study's branch means using Monte Carlo.

Before the current action let q=(4-len(prefix))/(16-t).
V_U(s)=(1-q)Q_U(HOLD)+q Q_U(REFRESH).
Report:
G_star_U = mean [V_U-Q_U(s,a_star)]
G_matched_U = mean [V_U-min_a Q_U(s,a)]
R_star_U = mean [Q_U(s,a_star)-min_a Q_U(s,a)]
D_U = Q_U(HOLD)-Q_U(REFRESH), with disagreement and margin diagnostics.
Verify R_star_U=G_matched_U-G_star_U.

The correct baseline for G_star_U is V_U. Comparing Q_U(a_star) directly with Periodic CE would confound changing the action with changing the entire tail.

Report exact branch sizes. Different branch cardinalities are genuine feasible-space differences, not noisy minimum estimation or Monte Carlo bias.

Keep K=4 and C pooling throughout this study. Additional budgets, pooling rules, new controllers, extra datasets, optional trajectory-regret studies, and broad architecture searches are outside this task.

8. HMDB TWO-ROLE VALIDATION

Authenticate the current metadata, originals, exposure ledger, and source graph. The reviewed audit reports:
- 2,754 eligible official-training clips after test-priority known-family quarantine;
- 1,530 eligible official-test clips, 30 per class;
- all 51 classes;
- 5-20 test families per class, but some global families span classes;
- no HMDB official-test model outputs in 19/19.1.

These are expectations to verify, not numbers to force. If the current records differ, reconcile objective technical/provenance differences before test exposure and preserve a clear amendment. Do not select a convenient subset to recover the expected counts.

Construct exactly two roles:
HEAD_TRAIN: all eligible official-training clips whose global conservative component does not intersect official test.
EVAL: all eligible official-test clips.

Exclude official-unused split entries. Preserve all exact-duplicate, perceptual-hash, and conservative title-family graph connections. Do not split or weaken components to increase training counts. Do not drop difficult classes. Report global component counts, cross-class components, maximum component size, and effective weight concentration. The sum of class-specific family counts is not the number of independent components.

Reuse retained originals under data/source/causal_observability_v1/videos where verified. Preserve adaptive_history_index_uniform_v2:
- at least 19 genuine presentation frames;
- 16 endpoints floor(linspace(3,N-1,16));
- four distinct genuine trailing frames per endpoint;
- no duplicated-frame padding, interpolation, or invented timestamps.

Replay actual presentation ordering, identities and PTS, and match the authenticated grids. Disclose that this adaptive frame-index recipe differs from UCF's temporal recipe. Report candidate/snippet physical-duration distributions; dataset differences cannot be attributed solely to semantic domain.

The previous two HMDB technical smoke clips were official training, including failed/successful native400 checks. Preserve that exposure history. Select new smoke inputs by a fixed hash from the new clean HEAD_TRAIN manifest only.

Use the exact retained X3D-XS checkpoint and pinned upstream source; strictly verify their hashes. The reviewed provenance uses PyTorchVideo revision f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0 and checkpoint SHA256 6dafc96144c88f800b190d2cb179ff69b8fd48c40afd4deca6c1e61f76fdfe86. Reconcile any discrepancy before proceeding. Preserve the existing four-frame preprocessing and native projection computation from the authenticated extractor, including its actual crop convention rather than a nominal default. Use float32 eval/inference mode, batch size one, real [16,2048] embeddings, and retain input/grid/encoder hashes per clip. Use new explicit output paths. Freeze study/head seed 20261026, matching the reviewed UCF common module; record encoder-numerics seeds explicitly rather than relying on whichever SEED happens to be imported.

Train one 51-class affine head on each HEAD_TRAIN clip's Dense mean feature:
- equal class -> within-class global family -> clip weights;
- weighted train-only feature means and standard deviations;
- replace scale <=1e-12 with 1, as in the existing recipe;
- weighted CE + alpha/2 * squared Frobenius norm of W, unpenalized bias;
- alpha=0.01 fixed; no search, selection split, temperature tuning, or encoder fitting;
- literal seed 20261026 and deterministic float64 Torch LBFGS initialization;
- lr=1, max_iter=2000, max_eval=4000, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50, strong_wolfe line search.

This is one predeclared optimization attempt with a larger fixed iteration allowance than the historical 400. Independently recompute the training objective and final gradient. Accept only finite data/parameters/objective and maximum absolute gradient <=1e-6. Reaching the iteration cap is recorded but is not itself failure if that numerical criterion passes. If it fails, stop before EVAL model calls and report the failure; do not tune after looking at test outcomes.

Seal the final head/scalers and all design inputs before the first EVAL feature or prediction. Log that first exposure and every extraction/resume. Evaluate the complete frozen cohort once, with exact 455-schedule calculations per clip. Persist genuine logits and per-schedule results sufficient for verification.

Recognition accuracy is an outcome, not an eligibility rule. Report weak Dense performance, class imbalance, small training families, and all adverse diagnostics. Do not improve the head in response to EVAL results.

Describe this as previously task-outcome-unexposed HMDB evaluation under isolation of known conservative source components. Original parent identities, unknown reuploads and pretraining overlap remain uncertain; do not claim universal independence.

9. AGGREGATION, UNCERTAINTY, AND DECISION RULES

Average the 12 states within each clip. Average clips within each (class, global-family) cell. Average family cells within each class. Average the fixed classes equally.

Use dataset-appropriate source identities: UCF group codes such as g19 are class-scoped, so use (class, group) unless authenticated evidence joins sources across classes. HMDB uses the actual global source-component ID, shared across classes. Never merge unrelated UCF g19 entries merely because their group strings match.

Use paired outcomes throughout. Clips, states, and the 455 schedules are not independent observations. No class resampling is required for the fixed 51-class primary estimand.

For the primary interval use the following specialized CR2/Satterthwaite calculation, equivalent to unweighted class-indicator OLS on class-family means with global-family clustering and identity working covariance:

Let y_gc be a class-family mean, n_c the number of families in class c, C the number of classes, ybar_c their mean, and theta=mean_c ybar_c.
q_c=1/(C*n_c); a_c=1/sqrt(1-1/n_c).
For each GLOBAL family g:
u_g = sum over its classes c of q_c*a_c*(y_gc-ybar_c).
V = sum_g u_g^2.

Build a matrix K indexed by unique global families. For each class c with family-index set I_c, add
(q_c*a_c)^2 * [I - ones(n_c,n_c)/n_c]
to K[I_c,I_c].
df = trace(K)^2 / sum(K*K).
Report theta +/- t_quantile(0.975,df)*sqrt(V).

This shares the same global family across all classes. Validate the formula against an independently coded full class-dummy projection-matrix construction. In a one-class independent-family fixture it must equal the usual sample-mean standard error and df=n-1. Include a cross-class shared-family fixture, label-permutation invariance, and exact agreement of the estimate with the hierarchical mean. Handle rank/numerical failures explicitly. If any n_c<2 or the interval cannot be validated, retain all classes and point estimates but mark primary uncertainty inconclusive; never silently substitute a narrower interval.

The reviewed UCF400 manifest has two singleton-group classes, BenchPress and JavelinThrow, and 99 classes with two groups. Therefore retain the complete UCF point estimates/distributions but do not present this CR2 interval for its full 101-class mean. This does not block Stage 25A or HMDB validation. HMDB's audited minimum is five test families per class and must be rechecked. A zero empirical variance must be disclosed and cannot be treated as proof of population equivalence.

Report this as an approximate small-sample cluster-robust interval, conditional on the fixed head, fixed classes, and the assumed independent global components. It does not include head-training uncertainty or unknown dependencies. Cite Pustejovsky and Tipton's CR2/Satterthwaite methodology and its current correction/documentation. Do not claim exact finite-sample coverage.

Use delta=0.01 nats as a prospective practical margin, not a TMLR acceptance threshold:
- upper 95% limit < -delta: evidence of material mean harm under this continuation;
- lower limit > delta: evidence of material mean benefit;
- entire interval inside [-delta,+delta]: bounded small mean effect;
- otherwise: unresolved at this precision.

A small mean can coexist with substantial individual harms/benefits; report the distribution. Secondary intervals are descriptive and not multiplicity-adjusted. Do not choose the best endpoint, tail, class, or tolerance as the main result. Report primary degrees of freedom, family concentration, and leave-one-global-family-out estimates to expose influential components.

Before HMDB test exposure, use variability among the 99 two-group UCF classes and HMDB metadata for a clearly labeled exploratory precision projection, explicitly acknowledging the two singleton UCF classes and the assumption of variance transfer between datasets. It is not a full-cohort variance estimate or power guarantee. Use the full frozen HMDB cohort regardless of projected or observed significance; no outcome-driven sample expansion or stopping.

10. RESOURCES AND RECOVERY

Stage 25A needs no new encoder calls.

Stage 25B's expected dense collection is (2754+1530)*16=68,544 new snippet forwards before extra checks; retain completed smoke features so they count toward collection. Count every additional/failed verification forward and any reuse explicitly.

For this task use prospective caps:
- 8 cumulative hours of new feature-collection wall time, including decode, preprocessing, inference, synchronization and writes;
- 20 GiB of additional disk use;
- existing originals/checkpoints only; no data downloads.

These are caps, not promised runtime estimates. Measure throughput and memory on a deterministic training-only smoke of 8 clean training clips. Produce a conservative completion projection and available-disk check before launching bulk collection. If the full cohort does not fit the cap, complete Stage 25A, save the frozen Stage 25B plan and exact blocker, and stop before test extraction. Do not shrink the population based on outcomes or substitute an invented cache.

Use atomic per-clip checkpoints and hash-verified resume. Count cumulative wall time across restarts. A partial EVAL extraction is an exposure event, not a fresh unexposed population on restart. Resume only the same frozen model/design/membership. If interrupted, report exactly what ran and provide an exact resume command.

Do not label CPU analysis time or dense cache extraction time as deployed adaptive-inference latency. No deployment speedup is estimated here.

11. MEANINGFUL VERIFICATION

Before the first new outcome computation, use small clearly labeled synthetic fixtures ONLY to verify mathematical and software invariants, never as empirical video evidence.

Verify:
- exact 455 schedules, mandatory zero, prefix matching including past holds;
- both actions legal at all primary states;
- deterministic repair examples, sequential/direct equivalence, and baseline preservation;
- uniform suffix probabilities and loss averaging;
- branch minima/means and the continuation-gap identities;
- state-reset semantics and all aggregation/statistical invariants above.

Then independently verify real outputs:
- reproduce archived UCF baseline/oracle summaries and overlapping E22 D_star values;
- independently enumerate suffixes with itertools.combinations rather than calling the main branch routine;
- scalar-recompute all primary state values from sealed schedule-loss tables;
- on a fixed hash-selected set of genuine clips with available features, replay pooling and float64 CE through a separate scalar implementation;
- check all model/data/grid hashes, output sizes, missingness, counters, finite values and exact cohort membership.

For HMDB, the real-feature checks are required because the new collection is produced here. UCF loss-table analysis may complete without locally retained raw video/features, with the scope of its verification stated.

Use a separate verifier that reads sealed artifacts and does not import the main endpoint implementation. Do not manufacture summaries or silently relax tolerances to pass. Stop optional testing once these concrete risks are resolved.

12. DELIVERABLES AND COMPLETION

Use new code files such as:
- continuation_value_gap.py
- verify_continuation_value_gap.py
- a separate HMDB collection/head adapter only where necessary.

Write under results/continuation_value_gap_v1/ and a separate ignored data directory. Provide:
- protocol.md and frozen_config.json;
- input/population/model manifests, seals, amendments and exposure/resource ledger;
- per-state and per-clip results, class-family summaries, primary/secondary estimates and intervals;
- independent_verification.json;
- findings.md with actual numbers, interpretation, limitations and next decision;
- exact PowerShell run/verify/resume commands.

Keep the artifact structure compact. Large original videos and feature caches stay outside git. Preserve small authenticated outcome tables needed to recompute the findings.

Create three publication-quality figures as PDF and PNG:
1. D_star versus D_P with numerical/material-margin distinctions, fixed scales per comparison, and declared weighting/subsampling if display reduction is necessary.
2. G_star, G_matched, R_star and the uniform-tail counterparts, separately for UCF and HMDB with valid intervals where available and clear baseline labels. Label the full UCF cohort's unidentifiable cluster interval explicitly; do not invent an error bar.
3. Signed per-clip gain and regret distributions, including adverse tails.

Prepare a concise claim-to-evidence table. Separate technical completion from scientific outcome:
- COMPLETE / INCOMPLETE_RESOURCE / BLOCKED_INPUT / VERIFICATION_FAILED for execution;
- material harm / material benefit / bounded small mean / unresolved for an evaluated primary estimate; NOT_EVALUATED when HMDB evaluation did not run. Unavailable/invalid uncertainty cannot receive an inferential outcome label.

Interpretation must be conditional:
- Material oracle-action harm plus appreciable matched-tail benefit supports continuation mismatch in this tested setting.
- Material oracle-action benefit with little regret weakens that explanation.
- Little matched single-intervention opportunity means this local intervention family offers little room; it does not prove adaptive scheduling impossible.
- Effects confined to one continuation or tiny margins support a narrow finding.
- Replication on HMDB strengthens external evidence, while a different effect is informative heterogeneity.
- Missing fresh evaluation leaves the independent-evidence gap open, even if 25A is complete.

Discuss the prior sequential-learning literature, especially Ross and Bagnell's AggreVaTe discussion of optimistic expert cost-to-go. The contribution would be an exact, empirically supported diagnosis in budgeted video inference, not discovery of that general issue.

In your final report, state what you changed, what actually ran, the primary result and interval, real resource use, verification evidence, all blockers, the defensible paper claim, and the next scientific action. Leave a reviewable local diff/branch. Do not publish, push, merge, or claim TMLR readiness.

References:
- Current reviewed repository: https://github.com/xyhsy11/video-refresh-benchmark/tree/aa147a1f98b1f82ca1bc3866bbceacf4633553af
- Ross and Bagnell, Reinforcement and Imitation Learning via Interactive No-Regret Learning, section 2.5: https://arxiv.org/html/1406.5979v1
- Pustejovsky and Tipton, Small-Sample Methods for Cluster-Robust Variance Estimation and Hypothesis Testing in Fixed Effects Models: https://doi.org/10.1080/07350015.2016.1247004
- CR2 corrigendum: https://doi.org/10.1080/07350015.2023.2174123
- Official CR2 implementation documentation: https://jepusto.github.io/clubSandwich/reference/vcovCR.html
- Official Satterthwaite contrast documentation: https://jepusto.github.io/clubSandwich/reference/linear_contrast.html
- TMLR acceptance criteria: https://jmlr.org/tmlr/acceptance-criteria.html

