# E27 Task 1 handoff

**Stage status: OOF_COMPLETE. Scientific status: NOT_EVALUATED.**

Formal completed stage: OOF_COMPLETE. Blocker: none.

Verified original fitting sources: 3,200. Verified complete reference sets: 3,200/3,200. Native forward completions: 51,328 (51,072 new in Task 1; total includes 128 extra historical ordinary-profile forwards). Completed OOF fits: 32; complete fold checkpoints: 4/4.
Independent source/reference/checkpoint audit: PASS. Complete fitting-stage independent verification: True.

## OOF development evidence

| Target | View | Held-out MAE | Training-fold constant MAE | Spearman |
|---|---|---:|---:|---:|
| advantage | S | 0.15874676 | 0.15899190 | 0.044153598429066174 |
| advantage | H | 0.16048192 | 0.15899190 | 0.048283244194021094 |
| advantage | HG | 0.16069750 | 0.15899190 | 0.04633745209224667 |
| advantage | HGH | 0.16087115 | 0.15899190 | 0.047695981108318755 |
| drift | S | 0.06757521 | 0.07480372 | 0.43459332109136967 |
| drift | H | 0.05708354 | 0.07480372 | 0.6683715582178168 |
| drift | HG | 0.05452041 | 0.07480372 | 0.7139254748204452 |
| drift | HGH | 0.05144233 | 0.07480372 | 0.7470128661836782 |

Source-level effects (all negative/zero/positive effects retained): `{"drift_contrast": {"exact_zero": 0, "mean": 0.0030780784713296903, "negative": 1189, "positive": 2011}, "primary": {"exact_zero": 1964, "mean": 0.00016122304518243432, "negative": 616, "positive": 620}}`.
Per-fold training/held-out errors and class/source distributions are in fit_diagnostics.json, fit_oof_source_summaries.json and task1/oof_prediction_diagnostics.json. Lower drift error does not establish better refresh utility.

## Precision selection

Selected m: 8. Sum of 400 unbiased within-class variances: 0.134315330199.
Rule: h(m) = 1.96 × 1.5 × sqrt(sum(s_c²) / (400² × m)); choose the smallest integer 8–20 with h(m) ≤ 0.005.

| m | h, inflation 1 | h, inflation 1.5 | h, inflation 2 |
|---:|---:|---:|---:|
| 8 | 0.00063491 | 0.00095237 | 0.00126983 |
| 9 | 0.00059860 | 0.00089790 | 0.00119720 |
| 10 | 0.00056788 | 0.00085182 | 0.00113577 |
| 11 | 0.00054146 | 0.00081218 | 0.00108291 |
| 12 | 0.00051840 | 0.00077761 | 0.00103681 |
| 13 | 0.00049807 | 0.00074710 | 0.00099613 |
| 14 | 0.00047995 | 0.00071992 | 0.00095990 |
| 15 | 0.00046367 | 0.00069551 | 0.00092735 |
| 16 | 0.00044895 | 0.00067343 | 0.00089790 |
| 17 | 0.00043555 | 0.00065332 | 0.00087109 |
| 18 | 0.00042328 | 0.00063491 | 0.00084655 |
| 19 | 0.00041199 | 0.00061798 | 0.00082397 |
| 20 | 0.00040155 | 0.00060233 | 0.00080311 |

The OOF variance transfer limitation is recorded below. precision_plan.json contains all 400 variance inputs, the full sensitivity table and input hashes.

## Resources

Compute-session wall: 16924.735 seconds (allowance 21,600). Additional read-only audit/publication work is charged separately to the cumulative ledger. Resource snapshot UTC: 2026-10-26T12:35:48.203296+00:00.

| Resource | Cumulative actual/charged | Authorized cap | Remaining |
|---|---:|---:|---:|
| Collection seconds | 18,963.531 | 28,800 | 9,836.469 |
| CPU-budget seconds | 7,561.762 | 14,400 | 6,838.238 |
| Timing seconds | 0.000 | 7,200 | 7,200.000 |
| HTTP response bytes charged | 2,995,163,817.000 | 65,498,251,264 | 62,503,087,447.000 |
| Additional retained bytes | 6,993,060,238.000 | 21,474,836,480 | 14,481,776,242.000 |

Known HTTP bodies: 2,990,969,513 bytes; conservative excess charge: 4,194,304 bytes. Scratch archive bytes: 0. Free disk: 146.839 GiB; mandatory reserve: 10 GiB.
Production process startup wall charged separately: 2.837406 seconds. Session-end-to-first-stopped-observation upper bound charged to CPU: 13.277871 seconds; this deliberately includes observation delay and is not claimed as exact teardown time.
Unresolved operations: []; uncertain native calls: 0. Acquisition wall upper-bound extra already included in collection: 593.074 seconds. CPU counters are exclusive phase wall-budget seconds, not operating-system CPU-cycle measurements.

## Tasks 2–4 proposal

See task1_resource_proposal.md and task1_resource_proposal.json for measured rates, workload calculations, minimum estimates, conservative headroom and four-working-day feasibility. No budget increase is activated.

- m=8: future Tasks 2–4 projected wall 20.86 hours; recommended cumulative limits `{"collection_hours": 27, "cpu_hours": 7, "free_disk_reserve_gib": 10, "network_gib": 70, "retained_gib": 20, "timing_hours": 10}`.

## Provenance and validation

Execution base commit: `2f42e5315600b3e53ccfcc0a60bce576821bf6c0`. Branch: `codex/native-head-value-confirmation-v1`. Resolve the delivery commit with `git log -1 --format=%H -- results/native_head_value_confirmation_v1/task1_handoff.json`.
Original frozen scientific code/configuration hashes remain unchanged. Added code hashes, exact model/class identities and environment comparison are in task1_handoff.json. task1/initial_state.json and task1/baseline preserve the initial checkpoint. task1/artifact_sha256_inventory.json binds the delivered evidence; original videos/reference arrays are separately bound by authenticated manifests and the reference audit.
23/24 passed. Exact float-equality ledger assertion differed by 9.1e-13 seconds under Python 3.12 summation. Frozen test and counters unchanged; independent exact discrete and 1e-8-second floating reconciliation passed.
The focused Task 1 suite passed all nine tests. Initial failed focused-test output is preserved separately from its corrected pre-exposure pass. validate_release.py and final publication checks are reported in the machine-readable validation section. Implementation tests are not empirical confirmation.

## Execution boundary and resumption

Evaluation source certification has not started; no joint certificate, test acquisition, final probe, evaluation seal, evaluation model call, outcome access, or final timing run occurred.

Resume or verify Task 1 with:

```powershell
.\.venv_temporal\Scripts\python.exe -B fit_plan_native_head_confirmation.py fit-plan --session-wall-seconds 21600
```

After explicit revised-budget authorization and a passing complete-study gate, the recommended Task 2 acquisition entrypoint is:

```powershell
.\.venv_temporal\Scripts\python.exe -B run_native_head_value_confirmation.py acquire --split test
```

This command was not executed. Task 2 first needs a reviewed operational budget amendment that preserves existing exposure identities.

## Limitations

- OOF is fitting-stage development evidence, not fresh evaluation confirmation.
- OOF within-class variances use eight fitting components/class and may not transfer to the official test population. Shared training folds can induce dependence; fixed 1.5 inflation is a prespecified allowance, not a coverage guarantee.
- Ordinary profile component times were not separately instrumented. Decode/inference, I/O and collection overhead are measured jointly; offline/authentication work stays separately charged.
- Source/fold checkpoints are atomic. A hard interruption with incomplete intent blocks for audit instead of silent reexecution.
- All-source independent scalar verification is one read-only operation; the runner may pause before it if its conservative reserve does not fit the remaining session.
- Historical failed preparation wall time remains unknown. Previously charged interrupted transfer upper bounds are retained. Git/control network traffic is separate from dataset HTTP-body accounting.
