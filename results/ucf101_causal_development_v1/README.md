# DEVELOPMENT / EXPLORATORY reproduction

Run in PowerShell from repository root. Existing authorized UCF train/development bytes, genuine arrays, frozen head, pinned PyTorchVideo/checkpoint and .venv_temporal are required. No downloads or package upgrades.

```powershell
& .\.venv_temporal\Scripts\python.exe -B -m unittest test_ucf101_causal_development -v
& .\.venv_temporal\Scripts\python.exe -B .\run_ucf101_causal_development.py --resume
& .\.venv_temporal\Scripts\python.exe -B .\run_ucf101_causal_development.py --audit-only
```

Resume reuses identical per-trial checkpoints; failed/incomplete attempts remain. Completed resume performs a read-only audit. Audit never decodes, trains, initializes CUDA, retimes, downloads or regenerates evidence. It verifies approved source identities, policy/feature/checkpoint hashes, actual call counts, scalar logit replay, group weighting/bootstrap, control freeze and historical Git identities. Official-test outcome paths are prohibited.

Large genuine observations/training examples/executed feature arrays stay under ignored data/source/ucf101_causal_development_v1 with public SHA inventories and original feature provenance. No vectors are reconstructed from aggregates. No video or backbone weights are redistributed. See protocol.md, findings.md, failure_analysis.md, fresh_dataset_gate.md, timing trials and verification.json.

See completion_review.md, failure_case_comparison.csv, failure_subset_review.json, independent_integrity_review.json and metric_column_semantics.md. Supplemental CPU-only review commands are in completion_review.md.
