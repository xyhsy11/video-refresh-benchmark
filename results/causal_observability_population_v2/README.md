# Experiment19.1 reproduction

NO_GO; input-only, no GPU.

Run from repository root in PowerShell. Original authenticated data/source/causal_observability_v1 bytes are required for screening. No download or package installation is implemented. Existing extraction is read-only.

```powershell
& .\.venv_temporal\Scripts\python.exe -B -m unittest test_causal_observability_population_v2 -v
& .\.venv_temporal\Scripts\python.exe -B .\run_causal_observability_population_v2.py --screen-only --resume
& .\.venv_temporal\Scripts\python.exe -B .\run_causal_observability_population_v2.py --analyze-only --resume
& .\.venv_temporal\Scripts\python.exe -B .\run_causal_observability_population_v2.py --resume
& .\.venv_temporal\Scripts\python.exe -B .\run_causal_observability_population_v2.py --audit-only
```

The full screening command resumes completed per-clip checkpoints and rejects changed code/protocol/input identities. --analyze-only reads retained actual metadata without decoding; --resume runs feasible pending stages. After finalization, resume performs the same read-only audit. --audit-only reads/hashes evidence and replays saved frame algebra, graph links and allocation constraints; it never downloads, decodes, solves, trains, predicts or regenerates results. It requires the retained original local archives, annotations and AVI bytes and verifies their hashes.

screening_evidence.jsonl.gz publishes lossless actual frame metadata (no pixels): integer PTS/time bases, presentation/decode order, full candidate grids, offsets, spans, eligibility and clip hashes. The append-only operational checkpoint remains ignored locally, inventoried in screening_identity.json. Public archive_member_inventory.json binds every AVI to original RAR members. SHA256SUMS.json locks all retained public evidence and execution code. No raw videos/checkpoints/archives are added to Git.

See protocol.md for scientific amendments and stop rules, source_family_audit.json for every component/edge and title uncertainty, quota_feasibility.json for both deterministic solver executions per configuration, and findings.md for actual evidence and limitations.

The independent all-pairs scalar pHash/count review can also be rerun read-only:

```powershell
& .\.venv_temporal\Scripts\python.exe -B .\results\causal_observability_population_v2\independent_review_program.py
```

See source_component_review.md for large-component interpretation and next_dataset_decision.md for the prospective replacement framework.
