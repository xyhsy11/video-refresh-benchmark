# Experiment 19: Causal Observability Gap

Execution state: **blocked**, stage `eligibility`.

Population/resource blocker: Exact45/15/10 and all51-class evaluation are infeasible under the fixed source-family and historical temporal-support rules; no revised protocol was approved.

**Primary estimate and interval: not available. No HMDB test recognition result, trained-controller benefit, observability-gap estimate or final latency comparison is claimed.**

Original archive/split metadata: 6766 clips, 51 classes. Input screening: 4843 eligible and 1923 failures under the historical 1.57-second trailing-snippet rule. All failures are inadequate support for16 complete distinct-frame snippets.
After official-test-family priority and duplicate joins: 2145 eligible isolated official training clips and 1108 eligible official test clips. 48 classes fail the exact70-training requirement. Classes meeting only the clip-count prerequisite (family-role packing not yet proven): brush_hair, pour, pushup.
Classes with no eligible official test clip under this temporal rule: catch.

The official lists contain70 training clips/class. Exact45/15/10 leaves no spare clips for source-family quarantine or technical exclusions. Input metadata alone found137 title-based families crossing official train/test, involving813 training clips across42 classes. These source conflicts remain even if temporal eligibility is later redesigned. No flag0/test replacements or outcome-based substitutions were used.

## What ran versus what remains

Synthetic safety tests: return code 0. The test log states the actual count. These are unit fixtures, not scientific observations.
Real GPU integration: native400_integration_passed_new51_head_integration_pending.
Completed native400 smoke calls retained: 16. The preserved earlier failed verification attempt has four additional genuine start/completion events. The failure was float32 accumulation in an independent reference; correction uses float64 without relaxing tolerances.
The real native400 checks validate decode/PTS/chronology,16 cached outputs,four calls per execution,current gray observations and pooled head replay. They are not a newly fitted51-class recognition head, trained H/H+M policy evaluation or the final timing study.

Pending when population-blocked: new51-class head fitting; controller training/development selection; new51-class GPU integration; pretest seal and first-test exposure; all official test rollouts; privileged dense/oracle diagnostics; paired task intervals; final interleaved timing; four task-quality/cost figures. Those figures are intentionally absent rather than filled with synthetic data. The input-only population figure is separate.

## Claim / evidence

|Claim|Evidence and limit|
|---|---|
|Fixed population is feasible|Not established; population_design.json records the failed exact quotas and empty test support for catch|
|Causal runtime respects interface/budget|Synthetic contracts plus actual training-only four-forward native400 executions; no trained-policy quality claim|
|Causal learned policy beats periodic|Unmeasured; no primary interval exists before actual test execution|
|Offline advantage is deployable|Unmeasured; past UCF Z_pool used all16 features; its 3.945x figure is dense/periodic latency, not deployed Z_pool|
|Source independence|Conservative title-family/exact/near-duplicate constraints; external parent IDs and pretraining mappings remain missing|

## Research decision

Preserve this blocked fixed-design attempt. An all51-class empirical study needs an explicitly approved input-only redesign of role counts and short-clip temporal support, followed by a new frozen protocol before recognition outcomes. Restricting to three eligible classes would be a different, strongly selected task and would not answer the requested51-class question. Do not bypass source isolation, manufacture outcomes or optimize for significance.

The implementation starts from reviewed merge017fe79; the actual initial checkout was a50d130 with the same tree. No historical scientific outputs were reset or overwritten. Any future post-exposure correction must archive and disclose original evidence.

```powershell
& .\.venv_temporal\Scripts\python.exe .\run_causal_observability.py --resume
& .\.venv_temporal\Scripts\python.exe -B .\run_causal_observability.py --audit-only
```

See exposure_and_amendment_limits.md for the precise native400 training-only technical exposure chronology. All four requested task-quality/cost figures are pending, not simulated. Read-only audit and run/resume contracts were exercised; resume returned2 with no additional GPU calls.
