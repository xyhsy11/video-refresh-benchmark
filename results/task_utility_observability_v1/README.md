# Experiment 22: CPU-only task-utility observability

EXPLORATORY / DEVELOPMENT. All videos belong to UCF101 official split-1 training membership; development outcomes and recognition-head selection were already exposed. No independent test claim, new X3D inference, video-recognition training, controller retuning or new timing. All historical outputs remain immutable.

From repository root in PowerShell, using the existing environment and authenticated local artifacts:

```powershell
& .\.venv_temporal\Scripts\python.exe -B -m unittest test_task_utility_observability -v
& .\.venv_temporal\Scripts\python.exe -B .\run_task_utility_observability.py --resume
& .\.venv_temporal\Scripts\python.exe -B .\run_task_utility_observability.py --audit-only
```

Resume checks the sealed code/protocol and retained per-clip checkpoints before reuse. On a completed study it runs the read-only audit. If original bytes/arrays are missing, stop with the recorded resource limitation; do not synthesize replacements or launch the historical runners. No packages or datasets are downloaded. GPU/model/decoder imports are blocked in the scientific runner. The separate historical regression unit tests use explicitly synthetic CPU fixtures only.

Audit-only verifies hashes, source membership, historical blobs, saved prediction/metric consistency and paired bootstrap indices/intervals. It never fits, performs model predictions, decodes, downloads, recomputes targets, retimes or regenerates scientific outputs. Independent actual target replay is performed and retained during the analysis, before read-only auditing; its authenticated receipt is checked afterward.

Large genuine per-state observation matrices and forced/exhausted state details are retained under ignored `data/source/task_utility_observability_v1/clips/`. The public observation_provenance.json inventories every local output and original source/feature/trace input by SHA256. Public predictions, targets, state identities, probe checkpoints and bootstrap draws permit review without distributing raw video or backbone weights. Original 32x32 thumbnails are reused for training/selection; actual previously encountered 32-value cheap statistics are reused for execution. No current/future dense array is passed to a causal observation.

Read protocol.md and task_target_definitions.md before interpreting the two targets. Local usefulness, hindsight continuation utility, predictive information and actually deployed policy quality are different quantities. Figures describe genuine observations and previously measured timing only.

Supplemental CPU reviews default to read-only. `--write` saves their receipts only in this study:

```powershell
& .\.venv_temporal\Scripts\python.exe -B .\results\task_utility_observability_v1\independent_replay_review.py
& .\.venv_temporal\Scripts\python.exe -B .\results\task_utility_observability_v1\supplemental_review.py
& .\.venv_temporal\Scripts\python.exe -B .\results\task_utility_observability_v1\write_reviewed_findings.py --write
```

The final command renders the already reviewed interpretation from retained genuine tables; it does not fit or infer. The supplemental review retains descriptive alignment/headroom intervals using the SAME group draws and verifies the disclosed constant-correlation amendment changed no model parameters/predictions. `execution_history/` and `implementation_amendments.json` preserve initial versions/errors; `incomplete_execution.json` is historical failure evidence, not the final state. Consult status.json and verification.json for completion.
