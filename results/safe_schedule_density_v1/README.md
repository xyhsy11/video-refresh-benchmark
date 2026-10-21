# safe_schedule_density_v1

CPU-only exact schedule-density diagnostic on the **already exposed UCF101
execution/development cohort**: 400 clips, 200 class/group units, 101 classes,
original official-training membership. No controller_train or selection clips
enter any empirical table. This is not an independent test or a controller
training proposal. Experiment 23A remains **NO_GO / BLOCKED_INPUT** and 23B
remains frozen and unexecuted.

## Commands

From the repository root, using the existing CPU numerical environment (no
installation, package upgrade, inference, decode, or download):

```powershell
.venv/Scripts/python.exe -m unittest -v test_safe_schedule_density
.venv/Scripts/python.exe analyze_safe_schedule_density.py --preflight-only
.venv/Scripts/python.exe analyze_safe_schedule_density.py --pilot
.venv/Scripts/python.exe analyze_safe_schedule_density.py --all-exposed
.venv/Scripts/python.exe analyze_safe_schedule_density.py --audit-only
.venv/Scripts/python.exe verify_safe_schedule_density.py
```

The first command uses synthetic/software fixtures only. The four stage flags
are mutually exclusive. `--preflight-only` authenticates original inputs and
freezes protocol/code/input hashes and both cohorts before new outcomes.
`--pilot` processes the 20 input-hash-selected clips from different classes and
groups, validates real-feature replay, and seals a **PARTIAL_PILOT** report without
population intervals. `--all-exposed` requires that validated pilot and processes
the entire predeclared cohort irrespective of the pilot's signs. `--audit-only`
performs read-only hash checks, exact empirical recomputation, historical-blob
verification, and bootstrap-order/primary-interval reproduction.
The separate read-only verifier independently reconstructs all per-video counts
and transitions from the saved exact arrays, all full summary/method/paired/rescue
estimates and intervals, and the archived B4 oracle mean from actual schedule
minima. It does not call the analyzer's aggregation or metric functions. Its
stdout is retained as `independent_verification.json` after completion.

Raw feature/trial/video bytes are local authenticated historical inputs and are
not redistributed here. Without them these commands report **BLOCKED_INPUT**;
committed CSVs never fabricate or substitute missing raw data. No legacy RN18
analysis substitutes for UCF.

## Outputs and interpretation

- `protocol.md`, `design_seal.json`: frozen estimands, thresholds, selection,
  aggregation, bootstrap plan and code hashes, timestamped before new outcomes.
- `preflight.json`, `input_inventory.json`, `head_inspection.json`: actual starting
  HEAD, reviewed main comparison, original hashes, inspected class order and
  parameter shapes, reused-head limitations, historical-result blob snapshot.
- `cohort_manifest.json`, `pilot_manifest.json`: exact full membership/weights and
  input-only pilot ranking. Neither can be expanded based on outcomes.
- `pilot/`, `full/`: separate stage reports; only `full/` supports statements about
  the complete exposed cohort. Start with `full/findings.md` and the figures.
- `per_video.csv`: one row per clip/K/pool/epsilon, exact count/denominator,
  probabilities, RandomExact CE, delta distribution quantiles, Periodic and Dense
  results, and all four correctness transitions. `P_safe` is N/A on Dense-wrong
  clips; their correction and CE-preservation measurements remain in the table.
- `clips/*.npz`: every enumerated schedule's float64 CE delta and uint8 prediction,
  ordered exactly as `refresh_pooling_utils.schedules(16,K)`. No synthetic rows.
  Each JSON journal binds these arrays and clip results to the design seal.
- `summaries.csv`: full-population and Dense-correct/wrong hierarchical estimates,
  weighted conditioning masses and marginal bootstrap intervals. Dense agreement
  is explicitly fidelity only. RandomExact averages losses and correctness of
  individual schedules, never an ensemble's loss.
- `grouped_summaries.csv`: class and class/group summaries; weights preserve the
  declared equal-class/equal-group/equal-clip hierarchy.
- `periodic_rescue.json`: Dense-correct Periodic failures, counts, masses,
  any/no-safe fractions, exact-density distributions, and whole-cohort rescuable
  mass. Null means N/A for empty conditioning sets.
- `frozen_methods_per_video.csv`, `frozen_method_comparisons.csv`,
  `paired_method_contrasts.csv`: archived Periodic/H/H+M position comparisons,
  joint success and missed_safe. Actual execution interpretation applies only to
  B4+C; U is offline re-pooling; K8 H/H+M is N/A, with no invented executions.
- `correctness_transitions.csv`: all four Dense correct/wrong -> method/schedule
  correct/wrong cells as whole-cohort weighted masses.
- `display_20_clips.csv/.md`: the preselected technical display, not an
  outcome-selected or representative subset.
- `figure_1_joint_density.png`: all-clip exact joint-density weighted CDFs.
  `figure_2_periodic_rescue.png`: primary Periodic-failure safe-density distribution.
  Both explicitly label exposed-development, exploratory interpretation.
- `bootstrap_groups.json`, `bootstrap_group_indices.npy`: 2000 paired PCG64
  class-stratified group draws, seed20261021, with group order and single-group
  classes. Intervals are marginal/unadjusted, conditional on the reused inputs
  and head; they are not schedule-level sampling intervals.
- `verification.json`, `accounting.json`, `completion.json`: numerical/history
  checks, zero new encoder calls/fits/decodes/downloads, CPU duration boundary,
  and immutable-stage hashes. CPU duration is not inference latency.
- Root `findings.md` answers the five scientific questions with sensitivity
  context; `checkout_review.md` records checkout preservation. Root
  `independent_verification.json` and `software_and_audit_verification.json`
  retain the successful independent/software/raw-input audit results.
  `release_manifest.json` seals the completed deliverable's code and file hashes.

Primary endpoint: full-population `p_correct_and_CE`, K4, C, epsilon0. Epsilon
.01/.05, K8, and U are predeclared sensitivities. Numerical tolerance 1e-10 is
separate. Exact single-clip fractions need no sampling interval. Weighted clip
quantiles use inverse weighted CDF; within-clip delta quantiles use linear
interpolation. Schedules are not independent samples.

## Resume and immutability

If enumeration is interrupted, rerun the identical stage command:

```powershell
.venv/Scripts/python.exe analyze_safe_schedule_density.py --pilot
# or, after a sealed pilot:
.venv/Scripts/python.exe analyze_safe_schedule_density.py --all-exposed
```

The analyzer authenticates the entire frozen input inventory again, reuses
design-bound per-clip journals, and continues only missing clips. If interrupted
between NPZ and JSON writes it independently reproduces and compares the retained
array before continuing. Identical partial reports can be resumed; conflicting
files stop rather than being replaced. Accounting retains the invocation that
first completed that report, with its stated timing boundary.

A sealed completed stage is read-only, including under repeated analysis flags.
Use `--audit-only` to verify it. A changed protocol or conflicting artifact needs
a new output version, not edits to completed v1 outputs. A failure report is
retained rather than replacing prior results. No historical prepare/run workflow
is called. Historical tracked scientific files are checked by Git blob identity;
protected outcome contents are not parsed for that check.

The initial preflight attempt encountered a Windows subprocess-audit argument
format error before writing a design seal or computing new outcomes. The guard
was corrected before the successful freeze; no scientific protocol changed.
