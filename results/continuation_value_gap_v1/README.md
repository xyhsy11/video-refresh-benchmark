# Experiment 25 commands and evidence

Run in PowerShell at the repository root using the existing environment. No install/download command is needed.

```powershell
& .\.venv_temporal\Scripts\python.exe .\verify_continuation_value_gap.py --synthetic
& .\.venv_temporal\Scripts\python.exe .\continuation_value_gap.py seal
& .\.venv_temporal\Scripts\python.exe .\continuation_value_gap.py ucf
& .\.venv_temporal\Scripts\python.exe .\continuation_value_gap_hmdb.py --resume
& .\.venv_temporal\Scripts\python.exe .\verify_continuation_value_gap.py
& .\.venv_temporal\Scripts\python.exe .\continuation_value_gap.py report
& .\.venv_temporal\Scripts\python.exe .\render_continuation_value_gap.py
& .\.venv_temporal\Scripts\python.exe .\verify_continuation_value_gap_inputs.py
```

The first three steps are already completed and must not overwrite their seals/results. `seal` refuses an existing seal; `ucf` refuses a completed result. For this completed checkout, use the independent verifier to reproduce the arithmetic. The exact recovery command for an interrupted HMDB collection is:

```powershell
& .\.venv_temporal\Scripts\python.exe .\continuation_value_gap_hmdb.py --resume
```

Resume checks frozen design, population, model, input, grid and feature hashes; completed smoke features count toward collection. Partial EVAL exposure is retained. A failed fixed head fit may not be retrained. A resource gate cannot be bypassed by shrinking the cohort or resetting the cumulative ledger.

`outcome_table.npz` holds compact exact455 float64 schedule losses, predictions and genuine candidate logits (when available); `population.json` provides their exact row order. `per_state.csv.gz` contains the 12 reset-prefix alternatives per clip. `per_clip.csv`, `class_family.csv`, `per_class.csv`, `leave_one_family_out.csv` and `estimates.json` give the fixed hierarchy and paired consequences. The uniform branch baseline is V_U, not Periodic CE. All figures are available as PDF and PNG.

Large feature caches, original videos/checkpoints and installed environments remain ignored under `data/source/` and existing environment directories. No old runner or module-global output mutation is used. The original experiments, scientific statuses and UCF official-test artifacts are preserved.

For the same independent verification with batched historical content hashes (much faster on Windows), use:

```powershell
& .\.venv_temporal\Scripts\python.exe .\results\continuation_value_gap_v1\independent_audit.py
```

Use this guarded recovery entry point for an interrupted run; it refuses a second optimization attempt if head fitting started without an accepted retained head:

```powershell
& .\.venv_temporal\Scripts\python.exe .\resume_continuation_value_gap.py
```

After the complete artifact audit and figure render, regenerate the resource/decision appendix with:

```powershell
& .\.venv_temporal\Scripts\python.exe .\results\continuation_value_gap_v1\finalize_report.py
```

The input-only temporal support description is in `temporal_support.json`; exact12-state branch sizes and repair schedules are in `branch_sizes.csv`. The guarded recovery script adds no scientific definition or optimizer attempt.

Completed evidence status: UCF COMPLETE (point estimate only; no full-cohort CR2 interval), HMDB COMPLETE / material benefit. Independent checks cover both complete datasets, all4,284 genuine feature caches, all23,160 state values, exact68,544 model starts/completions, and unchanged4,652 historical result blobs. The fixed head hit iteration2000 but passed the declared gradient acceptance criterion; no retry occurred.

Additional identity verification:

```powershell
& .\.venv_temporal\Scripts\python.exe .\results\continuation_value_gap_v1\class_order_audit.py
```

`execution_summary.json`, `resource_summary.json`, `independent_verification.json`, `supplemental_input_verification.json`, `class_order_verification.json`, `figure_review.json`, and `amendments/` provide the completion, resource, numeric, identity, visual and change evidence. The findings prose includes an editorial completion note with capped-head numerical acceptance and the precise next research decision; regenerating the report scripts reproduces the frozen numbers, and the retained final Markdown preserves this interpretation.
