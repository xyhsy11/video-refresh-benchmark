# E27 Task 1 resource proposal

Proposal only; caps unchanged. Endpoint scenarios are not selected evaluation sizes.

Selected m: 8. No evaluation data were acquired or executed for this proposal.

Current cumulative collection / CPU / timing: 18963.531 / 7540.369 / 0.000 seconds.
Network charged: 2,995,163,817 bytes. Additional retained: 6,993,057,280 bytes. All prior costs remain charged.

## Scenario projections

| m | Evaluation sources | Future Tasks 2–4 wall hours | Unfinished Task 1 reference hours | Proposed cumulative network / retained GiB | Collection / CPU / timing hours |
|---:|---:|---:|---:|---|---|
| 8 | 3200 | 20.86 | 0.00 | 70 / 20 | 27 / 7 / 10 |

### m=8 workload

| Operation | Additional projected hours |
|---|---:|
| test transfer | 3.213 |
| future native model setup | 0.009 |
| test catalog | 0.333 |
| test screening | 1.798 |
| test duplicate | 0.035 |
| evaluation reference | 5.261 |
| final probe fit | 0.004 |
| independent verification | 1.905 |
| evaluation offline cpu | 0.368 |
| complete system timing | 7.433 |
| future audit analysis cpu reserve | 0.500 |

| Future task | Additional projected hours |
|---|---:|
| task2 preparation | 5.507 |
| task3 evaluation execution | 5.634 |
| task4 analysis timing verification | 9.718 |

Technical screening projection uses complete measured certify-val wall, including authentication, failed attempts and certificate work. Historical source-audit allowance is allocated to Task 2; remaining independent verification to Task 4. No cost is removed or counted twice.
The split of setup time equally between Tasks 3 and 4 is a planning allocation, not a future measurement.

Minimum projected cumulative collection / CPU / timing: 15.916 / 4.871 / 7.433 hours. These are conditional estimates, not guaranteed minimum execution times.
Minimum projected network / retained: 55.032 / 14.512 GiB.
The unchanged integrated gate still uses smoke timing for future references and requires a projected cumulative collection floor of 21.274 hours. The recommendation covers the greater of this operational gate and the ordinary-reference forecast, plus headroom. Task 1 does not weaken or bypass that future gate.

## Measurements and uncertainties

Ordinary eight-source mean: 3.654883 s/source. Additional genuine ordinary sources now measured: 3192. Projection rate: 3.945757 s/source; reference safety factor stays 1.5.
Measured new ordinary-reference timing distribution: {"mean_seconds": 3.945757428728079, "n": 3192, "quantiles_seconds": {"maximum": 9.53705120000086, "median": 3.179325350001818, "minimum": 0.9770596999987902, "p05": 1.7316820650004956, "p25": 2.342010299999856, "p75": 5.519296100000247, "p95": 7.284649089998854}, "sample_sd_seconds": 1.8914401021342913}. Includes actual decode, inference, durable journaling and collection overhead; cached replay is excluded.
Authenticated remaining test archive route: 56,094,802,765 response-body bytes. Transfer factor 1.2, screening/catalog factor 1.5, original/feature retention factor 1.3. Recommended cumulative limits add 25% time and 20% byte headroom over these already inflated estimates.
Keep 10 GiB free disk plus the largest compressed archive (about 1.51 GiB), decoded working buffers, and the projected retained increment. No paid compute or environment upgrade is proposed.

Measured source-level offline processing and authentication: 0.276131 CPU-budget seconds/source. These costs are projected separately for every evaluation source.
Fitting-run process memory receipt: {"process_peak_working_set_bytes": 2569760768, "process_working_set_bytes": 1243471872, "scope": "Current process lifetime; earlier unmetered processes are not inferred", "system_available_ram_bytes": 9021489152, "system_ram_bytes": 34110488576}. CUDA peak allocated/reserved receipts: {"cuda_peak_allocated_bytes": 68930048, "cuda_peak_reserved_bytes": 90177536}.
These memory peaks describe their recorded process scopes, not complete-system RAM. Future all-source data assembly may scale with 400m; reserve and recheck scratch and available RAM before that operation.

- Transfer and screening rates may change across test archives.
- Final probe time is extrapolated from genuine fixed folds when available; otherwise 1800 seconds is unmeasured planning reserve.
- Timing retains the existing conservative full-system smoke gate; no final timing experiment was run.
- Future analysis/audit reserve is an unmeasured 1800 CPU seconds.
- Retained original size uses measured fitting bytes with 1.3 inflation; scratch peak still needs future-stage validation.
- Unknown reuploads, test membership/technical failures and rank-preserving revisits may block the fixed population.
- Reference profile has eight sources; subsequent genuine fitting measurements are reported separately.

## Four working days

Using eight hours/day as a planning convention gives 32 hours. Recorded prior collection/CPU/timing are included in the totals below, along with unfinished Task 1 reserves and future work. These phase-wall sums are a conservative serial machine-work accounting convention, not a measurement of human working hours. Historical unknown preparation time, review, transfer stalls, safe-session pauses and input revisits remain additional uncertainty.
For m=8, recorded-plus-projected total machine work is 28.22 hours; within 32 hours before unknown costs and human review, so four days remains conditional.

## Authorization

This document does not activate any cap amendment. Task 2 must wait for explicit user authorization of revised cumulative limits, completion and verification of Task 1, and a fresh full-study resource check. The old integrated plan/run/resume paths must not be used to continue Task 1.
