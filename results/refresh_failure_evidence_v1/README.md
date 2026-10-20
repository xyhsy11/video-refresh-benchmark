# Experiment 23A: refresh failure evidence and prospective population gate

Decision: **NO_GO for further predictive refresh development or independent evaluation now.** Population status: **BLOCKED_INPUT**. Phase A is a CPU-only reconciliation of genuine retained observations; Phase B reuses authenticated input screenings. This audit fits no probe, recognition head or controller and executes no policy. Historical outputs are immutable.

Read [summary.md](summary.md), [evidence_matrix.csv](evidence_matrix.csv), [failure_taxonomy.json](failure_taxonomy.json), [independent_population_gate.json](independent_population_gate.json) and the [conditional 23B preregistration](experiment_23b_preregistered_protocol.md). Grayscale association rows are exploratory numerical proxies, never semantic event annotations or causal performance.

From the repository root in PowerShell, with the existing local environment and retained genuine arrays:

```powershell
& .\.venv_temporal\Scripts\python.exe -B -m unittest test_refresh_failure_evidence -v
& .\.venv_temporal\Scripts\python.exe -B .\run_refresh_failure_evidence.py
& .\.venv_temporal\Scripts\python.exe -B .\run_refresh_failure_evidence.py --audit-only
```

The run command builds this new study once, runs the available historical read-only audits and regression tests, checks every earlier tracked result blob, and seals outputs/code with SHA256. After completion it performs read-only verification. It never launches a historical resume/training/screening command. Missing original arrays stop verification; they are not reconstructed from means. No packages, datasets or agreements are acquired. Local raw observations remain ignored and are inventoried in run_manifest.json; no video or model weight redistribution.

The audit command validates the complete output/code inventory and consumed input hashes and independently recomputes B4 CE/paired intervals and historical timing aggregation. It writes no receipts, retimes nothing, and performs no model predictions. Final stdout/stderr and return codes are in check_*.txt and verification_commands.json. Initial verification_*.txt logs and execution_history/ preserve failed attempts, including five older guards blocked by a later attribution-notice change. Those five audits run without guard changes in a clean local c3ec3e6 historical worktree; historical_checkout_review.json verifies identical scientific result directories. No dataset download is involved. Test modules run in separate CPU processes to avoid an existing HiGHS global thread-state conflict. Synthetic fixtures are excluded from scientific tables. A passed software audit does not mean scientific GO.
