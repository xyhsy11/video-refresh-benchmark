# E27 Task 1 operational boundary

This is an operational implementation record. It does not amend the population,
scientific model, features, targets, precision rule, or cumulative resource caps.
It does not authorize Task 2 or claim scientific confirmation.

## Exact command

From the repository root, using the existing environment:

```powershell
.\.venv_temporal\Scripts\python.exe -B fit_plan_native_head_confirmation.py fit-plan --session-wall-seconds 21600
```

The dedicated entrypoint stops at `OOF_COMPLETE`. It never calls acquisition,
joint certification, final probe fitting, sealing, evaluation, final timing, or
finalization. Do not use the old integrated `plan`, `run`, or `resume` commands
to continue Task 1.

## Identity preservation

All twelve scientific/control files listed in the original frozen configuration
remain byte-identical. The original configuration, model identity, native class
mapping, population rule, and existing reference receipts retain their original
hashes. No historical computation is relabeled as using new code.

The added orchestration, independent fitting checker, and focused tests are
separately frozen in `task1/operational_controls.json`. Each newly attempted
source/fold has a durable operation receipt recording the orchestrator hash.
`task1/initial_state.json` and `task1/baseline/` preserve the starting ledger and
checkpoint. The old cumulative ledger is extended, never reset or replaced.

## Resource and throughput treatment

The eight-source ordinary profile is accepted only after authenticating original
video hashes, frozen grid/implementation identities, checkpoint, output hashes,
exact smoke/profile array parity, and 128 fresh-forward completion receipts.
Its timer includes hashing/read, decode, preprocessing, synchronized inference,
artifact writes, and collection guards. It excludes smoke-only affine assertions.
Component-specific timings were not captured; no retrospective decomposition is
claimed. Offline features/targets, cache authentication, comparisons and other
CPU work remain separately charged. Cached replay never measures model throughput.

The reference safety factor remains 1.5. Every 32 sources the fitting-only gate
uses the greater of the authenticated historical ordinary mean and the measured
new ordinary mean. Source variation and limited sample representativeness remain
explicit. Added durable forward journaling is included in new observed timings.
The original integrated complete-study gate remains unchanged for future stages.

CPU planning includes offline processing, all-source scalar verification, and an
explicit 1,800-second unmeasured reserve for OOF/audits. The first actual fold
must validate the cost of remaining folds. A passing projection is not a promise
that the full workload will fit. Disk projections retain the 10 GiB free reserve.

## Safe checkpoints

The session allowance is independent of cumulative experiment counters. The
runner checks it before sources, folds, and verification. It finishes a started
source/fold, persists artifacts, records the next operation and cumulative
counters, and returns `PAUSED_SESSION` at the next safe boundary. It does not
mislabel a session pause as completion or exhaustion.

Existing sources are authenticated against original encoded bytes, frozen grid,
checkpoint, original implementation and output hashes before reuse. Native calls
are journaled before and after each forward; source counters remain owned by the
original collector to avoid double charging. Incomplete operation records block
for an explicit audit rather than silently recomputing. In a hard crash, an
attempt may remain uncertain; the journal preserves that uncertainty and prevents
automatic repetition. No uncertain attempt is claimed as a successful forward.

Completed folds bind membership, training identities, models/scalers, and saved
held-out prediction hashes. Resumption verifies those hashes and reloads completed
fold predictions. Incomplete folds block for audit rather than refitting completed
expensive models without review. Current execution state is in `task1_status.json`
and `task1/sessions/`; historical status/accounting snapshots remain preserved.

## Verification limits

The new tests cover boundary isolation, real ordinary-profile authentication,
unchanged cumulative caps, session checkpoints, authentic cache reuse, incomplete
operation rejection, fold membership/integrity, and precision arithmetic. Metadata
arithmetic fixtures use actual encoded-file lengths solely as a unit contract;
they are never saved as OOF effects or used to choose E27's evaluation size.

The frozen suite's exact float-equality ledger assertion can differ from an
independent sum at machine rounding precision under Python 3.12. The independent
reconciliation records the differences, requires exact discrete counters and a
1e-8-second arithmetic bound, and changes no ledger entry. This is separate from
any unresolved operation, which remains a hard blocker.

The added OOF checker independently reconstructs folds, scalar schedule losses,
source consequences, training standardization, ridge stationarity, held-out
predictions, and within-class precision calculations. Implementation tests do
not establish that full empirical fitting or final evaluation occurred. The
handoff reports which paths were genuinely exercised and independently verified.
