# Experiment 27 commands and resume

Use the existing PowerShell environment from the repository root. No package
installation or global upgrade is required. These commands use real original
video inputs; no simulated observations or outcomes are generated.

```powershell
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py status
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py dry-run
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py preflight --online
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py acquire --split val
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py certify --split val
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py plan
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py acquire --split test
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py certify --split test
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py run
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py resume
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py verify
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py audit
.\.venv_temporal\Scripts\python.exe -m unittest test_native_head_value_confirmation -v
.\.venv_temporal\Scripts\python.exe validate_release.py
```

`resume` is the integrated entrypoint: it acquires/certifies the fitting prefix,
collects genuine fitting smoke and references, performs fixed OOF precision
planning, acquires/certifies the exact selected test cohort, fits final probes,
seals identities, and evaluates/times/verifies only after each gate passes.
Test acquisition requires the real fitting-only precision plan. `plan` stops
after sealing and does not expose EVAL. No user-created success certificate or
hand-edited ledger is required. A required PRESENT/UNKNOWN source blocks its prefix;
missing irrelevant lower-ranked files do not block it.

`status`, `dry-run` and `audit` are read-only. `dry-run` reports actual missing
prospective prefix/reserve inputs with zero decode/model/network calls. `audit`
checks identities and existing evidence without fitting or regenerating results.
`verify` writes a verification receipt. Completed invocations audit existing
artifacts rather than fitting, extracting or timing again. Exit2 denotes a
blocked/incomplete scientific stage, not a successful empirical experiment.

After a complete genuine execution, inspect the four PNG/PDF figures and record
their hashes with `actually_inspected: true` in visual_review.json. Only then run:

```powershell
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py finalize
```

The active network allowance is 61 GiB cumulative, including the preparation's
13,062,569 metered bytes, metadata, full gzip shards, failures and retries. The
additional retained cap remains 20 GiB. Temporary working space is separately
gated with a 10 GiB free-disk reserve. Collection/CPU/timing caps remain 8h/4h/2h.
Actual phase attempts and incomplete work persist in resource_exposure_ledger.jsonl.
A crash never resets counters. Unknown failed preparation wall times remain an
explicit accounting limitation. Historical caches are authenticated before reuse
and never removed; only verified task-owned scratch archives may be deleted.

[findings.md](findings.md), [status.json](status.json), [stage_state.json](stage_state.json)
and [execution_accounting.json](execution_accounting.json) report the actual state.
The immutable blocked preparation and original code are in
[preparation_a9e1942](preparation_a9e1942/snapshot_identity.json), associated with
commit a9e1942f226f341071e1a990ea446dc717a070bb. Historical result hashes remain in
[historical_fingerprint.json](historical_fingerprint.json). The
[protocol](protocol.md), [amendments](amendments.md), [resource plan](resource_plan.json)
and [related work](related_work.md) state the unchanged scientific design and its scope.

Large original RGB, member catalogs, audit graphs and feature arrays remain under
ignored data/source/native_head_value_confirmation_v1. All selected byte identities,
source role decisions and certificate hashes are sealed before EVAL calls. Known
parent/byte/perceptual evidence cannot exclude unknown reuploads or undocumented
pretraining. Implementation verification alone is not empirical completion.


The former d978026 attempt was BLOCKED_INPUT / NOT_EVALUATED. Its
[snapshot](blocked_d978026/snapshot_identity.json), old missing-input reports and
public-source checks remain historical evidence. The prospective
[population amendment](population_amendment.md) supersedes release-absent blockers
uniformly for both splits. [population_rule.json](population_rule.json) pins the
release; all-annotation availability ledgers distinguish PRESENT,
CONFIRMED_ABSENT and UNKNOWN. Required PRESENT uncached bytes still need acquisition.
Current status and actual executed stages are reported in status.json and findings.md.
The public [member catalogs](archive_member_catalogs.json.gz) and
[input-only graph](input_only_source_audit.json.gz) contain real metadata/byte
identities. They do not certify a complete fitting/evaluation cohort. Dependent
fitting/EVAL/timing paths have not been exercised on a complete E27 population;
passing contract and blocked-state checks does not validate those empirical stages.

The amended attempt reached `FIT_CERTIFIED`: 3,200 fitting sources, eight per
native class. Genuine native reference data exist for eight sources, with 128
smoke forwards plus 128 additional ordinary-reference diagnostic forwards on
the same sources; all arrays matched exactly. The runner stopped at
`BLOCKED_RESOURCE / NOT_EVALUATED`. No probes, OOF-selected m, EVAL outcomes,
final timing or figures exist. See
[resource diagnosis](minimum_design_resource_diagnosis.json) and
[actual scope](implementation_validation_scope.json). The smallest permitted
full-study workload projects about 16.9 cumulative collection hours under the
measured gates against the unchanged 8-hour cap. This is a projection, not an
observed future duration. A resource amendment or independently validated
adequate implementation is needed before further integrated execution.
