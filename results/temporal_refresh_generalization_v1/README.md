# Temporal refresh generalization v1

Status: complete and verified; new-source offline diagnostic. See [findings](findings.md), [protocol](protocol.md), [frozen identities](evaluation_seal.json), [source roles](dataset_manifest.json), [quality](per_video_quality.csv), [paired intervals](bootstrap_intervals.json), [timing](raw_timing_trials.csv), [verification](verification.json) and [computation accounting](metadata.json).

The primary effect is CE(feature oracle) minus CE(periodic), X3D-XS, B4, cache pooling: +0.009784 nats/video, 95% interval [-0.012442,+0.032961], inconclusive. Positive means worse recognition loss. New evaluation is100 sources,five/class;development is reused100 sources. Neither privileged oracle is a deployable policy.

Large encoded inputs,checkpoints,dependencies and reference arrays remain under ignored data/source/temporal_refresh_generalization_v1/,with relative paths and SHA256 identities in manifests/collection ledgers. The raw unsanitized download-error ledger is preserved there;public copies remove only project-absolute error paths and retain byte counts/measurements. The500 fixed encoded train/development sources are prerequisites;missing sources block without substitution. Old candidate fields in data_plan.json are provenance;extraction uses new snippet_grid entries.

From the repository root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe .\run_temporal_refresh_generalization.py --device cuda --max-download-gib 20 --resume
& .\.venv_temporal\Scripts\python.exe -m unittest test_temporal_refresh_generalization -v
& .\.venv_temporal\Scripts\python.exe .\run_temporal_refresh_generalization.py --audit-only
```

A new incomplete run pauses for rendered-figure review before `--finalize-reviewed`. Completed invocations audit without overwriting or collecting another cohort. Future work must exclude these now-exposed100 evaluation parents. No protected confirmation outcomes or arrays were opened. No refresh-gain predictor was trained. No streaming-causality or adaptive-speedup claim is established.
