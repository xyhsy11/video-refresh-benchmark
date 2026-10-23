# Experiment 27 commands and state

Run from the repository root in the existing PowerShell environment. No package
installation is required. Public metadata retrieval is authorized; original video
archives are not downloaded automatically. Exit2 means a documented blocked or
incomplete stage, not a successful scientific experiment.

```powershell
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py preflight --online
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py dry-run
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py plan
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py run
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py resume
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py verify
.\.venv_temporal\Scripts\python.exe run_native_head_value_confirmation.py audit
.\.venv_temporal\Scripts\python.exe -m unittest test_native_head_value_confirmation -v
```

`audit` is read-only and does not decode, fit, infer, regenerate metrics or write
verification. `verify` writes a new verification receipt. A completed execution
audits rather than overwrites its sealed results. Before EVAL exposure, code or
metadata corrections must be reflected by rerunning metadata-only preflight and
retaining the amendment record. After exposure, changes invalidate the seal.

After a genuine full execution, inspect all four PNG/PDF figures and record
their hashes and `actually_inspected: true` in `visual_review.json`. Only then
use `run_native_head_value_confirmation.py finalize`; it rechecks the seal and
independent verification before assigning COMPLETE. No such record exists for
the current blocked study, and no visual inspection is inferred automatically.

Current scientific status is **NOT_EVALUATED**. See [findings](findings.md) and
[resource amendment](resource_amendment.md). The prospective [protocol](protocol.md),
[configuration](frozen_config.json), [source feasibility](population_feasibility.json),
[archive inventory](archive_inventory.json), [resource plan](resource_plan.json),
[class mapping](class_mapping.json), [schema](observation_schema.json),
[metadata source evidence](historical_source_evidence.json),
[verification](independent_verification.json), and [related work](related_work.md)
make the blocked state reviewable without inventing empirical artifacts.

Original bytes and large caches stay under ignored `data/source/` paths. Candidate
metadata is not a sealed eligible population. No evaluation manifest, OOF precision
result, probe hash, exposure seal, source outcome or empirical figure is supplied
while prerequisites are absent. `plan`, `run` and `resume` return the existing gate
before any model call. Future original RGB can be placed under the study's ignored
`videos/val` and `videos/test` directories using official clip filenames; this by
itself does not certify source independence or authorize a resource-cap change.

A source-isolation certificate must bind the current historical-source evidence
SHA256, resolve legacy reservation status with original input/exposure records,
and document known global input-byte/duplicate links before fitting. It must not
assert universal pretraining independence. Acquisition, source certification and
real eight-source throughput remain prerequisites. Dependent code paths are
implemented but have not been validated on this unavailable E27 population; the
blocked-state audit is deliberately narrower than empirical verification.
