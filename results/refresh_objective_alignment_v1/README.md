# refresh_objective_alignment_v1

Read findings.md first. Pilot cohorts and confirmation are separate directories.
`protocol.md` was written before new comparisons; `evaluation_seal.json` precedes
confirmation inference if feasible. Large real arrays stay in ignored data/source,
with identities in the ledgers. Model fitting and package upgrades are forbidden.

```powershell
& .\.venv_temporal\Scripts\python.exe -m unittest test_refresh_objective_alignment -v
& .\.venv_temporal\Scripts\python.exe .\analyze_refresh_objective_alignment.py --input results/temporal_refresh_generalization_v1 --output results/refresh_objective_alignment_v1 --seed 20261025 --bootstrap 10000 --device cuda --max-download-gib 20
& .\.venv_temporal\Scripts\python.exe .\analyze_refresh_objective_alignment.py --resume
& .\.venv_temporal\Scripts\python.exe .\analyze_refresh_objective_alignment.py --audit-only
```

Saved CSV floats round-trip; JSON undefined values are null, never NaN/Infinity.
CPU analysis/search runtime is not adaptive inference latency. See execution and
download accounting for actual collection work. Completed invocations audit only.

On a newly computed run, inspect all four PNGs, then record completion:

```powershell
& .\.venv_temporal\Scripts\python.exe .\analyze_refresh_objective_alignment.py --finalize-reviewed
```

The strict input audit needs the real ignored reference arrays. Aggregate primary statistics can also be independently reconstructed from confirmation/per_source_deltas.csv and its saved bootstrap indices without loading encoders. Missing real references are a blocker to full input/replay auditing, never a reason to generate replacement embeddings.
