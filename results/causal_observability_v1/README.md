# Experiment 19 execution evidence

See [findings](findings.md), [protocol](protocol.md), [population design](population_design.json), [source-family audit](metadata_source_family_audit.json), [technical exclusions](eligibility_exclusions.json), [native GPU integration](GPU_smoke.json) and [verification](verification.json).

The fixed scientific study is population-blocked. No51-class recognition head/controller or official-test outcomes are claimed. The only current figure is the inspected input-only population figure. The four task-quality/cost figures remain pending genuine outcomes. Original source arrays/videos are never reconstructed from aggregate statistics.

The lossless `eligibility_attempts.jsonl.gz` archives every real PyAV screening attempt. Its decompressed digest is recorded in `eligibility_archive.json`. `failed_input_byte_identities.json` supplies independently hashed encoded-file identities for short-support clips whose original exception record did not contain those fields. The original screening journal, original video/class archives, failed acquisition attempts, complete smoke vectors/logits and real model bytes remain under ignored `data/source/causal_observability_v1/`; public identities point to them. Missing local genuine arrays block the corresponding read-only replay.

Original annotations are redistributed as their byte-identical small archive under `official_annotations/`; no video bytes are redistributed. The source metadata records official/public URLs, the pinned distribution revision, hashes and advertised CC-BY4.0 attribution. Unknown external parent IDs and pretraining/reupload overlap remain explicit.

Two pretest corrections are preserved under `pretest_corrections/`: an FP32-accumulating replay reference was corrected to float64 without relaxing tolerances; later dense-checkpoint provenance hardening did not alter live inference arithmetic. These preceded any official test call; no evaluation seal or test-exposure ledger has been fabricated.

```powershell
& .\.venv_temporal\Scripts\python.exe .\run_causal_observability.py --resume
& .\.venv_temporal\Scripts\python.exe -B .\run_causal_observability.py --audit-only
```

Resume validates and retains the same blocked attempt; it does not authorize revised counts, dropping classes, replacing clips or changing temporal support. A new population/candidate protocol needs explicit direction before recognition outcomes.

The [technical-exposure and amendment limits](exposure_and_amendment_limits.md) disclose the native400 checks before any prospective redesign. The [input-only population figure](figures/input_population.png) was visually inspected.
