# Experiment 19.1 findings

Population gate: **NO_GO**. No model was trained or executed in this study.

All 6,766 original authenticated clips across 51 classes were freshly decoded. Historical technical eligibility is 4,843; redesigned eligibility is 6,765. The old rule has 1,923 exclusions; the new rule has 1. These counts are calculated from retained real frame evidence, not assumed planning totals.

Eligible official train/test/unused: 3,569 / 1,530 / 1,666. Conservative test-priority isolation excludes 815 eligible training clips in 138 train/test-conflicting components, leaving 2,754 available training clips. No eligible official test member is removed or moved.

For catch, old/new eligible test clips are 0 / 30; new eligible training clips are 70, with 67 remaining after family quarantine and 16 training / 7 test components.

|Configuration|Status|Classes with necessary-condition failures|Failed necessary constraints|
|---|---|---:|---:|
|A|PROVEN_INFEASIBLE|15|17|
|B|PROVEN_INFEASIBLE|13|14|
|C|PROVEN_INFEASIBLE|11|12|

Selected configuration: none. Full solver options, repeated termination states and independent count certificates are in quota_feasibility.json. A timeout is never counted as infeasibility. The minima are engineering thresholds, not power guarantees.

## Class-level evidence

|Class|Old eligible|New eligible|Available train|Test clips|Train / test families|
|---|---:|---:|---:|---:|---:|
|brush_hair|107|107|70|30|26 / 10|
|cartwheel|76|107|56|30|26 / 15|
|catch|3|102|67|30|16 / 7|
|chew|107|109|66|30|24 / 12|
|clap|93|130|63|30|21 / 11|
|climb_stairs|97|112|58|30|19 / 11|
|climb|105|108|67|30|25 / 12|
|dive|57|127|60|30|27 / 15|
|draw_sword|81|103|66|30|26 / 12|
|dribble|141|145|70|30|20 / 8|
|drink|127|164|13|30|9 / 18|
|eat|82|108|16|30|12 / 15|
|fall_floor|37|136|26|30|4 / 11|
|fencing|45|116|67|30|21 / 9|
|flic_flac|72|107|64|30|29 / 14|
|golf|104|105|70|30|51 / 20|
|handstand|112|113|57|30|35 / 19|
|hit|6|127|55|30|19 / 12|
|hug|87|118|63|30|11 / 6|
|jump|32|151|50|30|14 / 7|
|kick_ball|29|128|63|30|22 / 11|
|kick|12|130|48|30|7 / 7|
|kiss|102|102|66|30|8 / 5|
|laugh|125|128|65|30|17 / 9|
|pick|94|106|48|30|26 / 15|
|pour|105|106|70|30|30 / 13|
|pullup|102|104|70|30|22 / 9|
|punch|52|126|38|30|9 / 10|
|push|113|116|69|30|21 / 9|
|pushup|103|103|70|30|21 / 8|
|ride_bike|97|103|70|30|16 / 6|
|ride_horse|113|116|61|30|14 / 8|
|run|119|232|22|30|10 / 16|
|shake_hands|153|162|67|30|33 / 17|
|shoot_ball|117|131|50|30|14 / 5|
|shoot_bow|98|112|64|30|22 / 9|
|shoot_gun|75|103|51|30|15 / 9|
|sit|108|142|24|30|15 / 14|
|situp|104|105|70|30|21 / 9|
|smile|65|102|62|30|16 / 11|
|smoke|101|109|57|30|26 / 12|
|somersault|106|139|61|30|32 / 17|
|stand|92|154|24|30|17 / 18|
|swing_baseball|104|143|69|30|15 / 5|
|sword_exercise|82|127|50|30|19 / 9|
|sword|84|127|58|30|21 / 11|
|talk|117|120|22|30|9 / 12|
|throw|78|102|48|30|16 / 10|
|turn|124|240|29|30|15 / 17|
|walk|430|548|29|30|20 / 18|
|wave|68|104|35|30|20 / 17|

## Interpretation and threats

The temporal amendment recovers input coverage, not demonstrated recognition fidelity. Endpoints are uniform in presentation-frame index and may be nonuniform in time. Windows in very short clips can contain little action evolution. This is not equivalent to the inherited offset recipe and requires a separately reviewed recognition protocol.

The graph contains 1385 components and 48 components with >=20 clips. Shared titles are uncertain proxies, sometimes broad titles/compilations. All original parent URLs/identifiers remain unavailable; different titles do not establish independence. No uncertain large family was split to rescue quotas. pHash is conservative finite-resolution detection, not an exhaustive duplicate guarantee. Exact-only / exact+pHash / title-only counts are labeled sensitivities and cannot justify GO.

The historical blocked experiment stopped before hypothesis testing; it is not evidence that learning failed. Its two official-training videos had prior native400 mechanical outputs, disclosed in historical_integrity.json and the original exposure notes. No official HMDB test model output exists in either study. UCF offline Z_pool quality required dense features; its periodic timing does not establish a deployable causal advantage. No new claim about that advantage follows from this audit.

## Claim and evidence

|Claim|Evidence and limit|
|---|---|
|Genuine population and frame contract|archive_member_inventory.json, screening_evidence.jsonl.gz and independent scalar replay; no fabricated frames|
|Technical eligibility change|per_clip_eligibility.csv, per_class_eligibility.csv, temporal_sampling_audit.json; no recognition outcomes|
|Known leakage protection|source_family_audit.json and leakage_validation.json; external parent identity remains uncertain|
|Allocation gate|quota_feasibility.json, independent necessary-count certificates / verified assignments; no power claim|
|Historical preservation|historical_baseline.json and historical_integrity.json; all previous tracked result hashes checked|
|Method accuracy or latency|Unmeasured in Experiment19.1; tests and population audit do not establish either|

## Next decision

Do not run the declared full51-class HMDB learning experiment. Sampling recovery cannot resolve the verified conservative source-allocation deficits. Do not split broad families, lower minima further or drop classes after this audit. Select a fresh independent dataset prospectively using the protocol’s declared source independence, full-class availability, genuine video/label access, temporal suitability, compute and licensing framework. No substitute dataset is acquired or selected using recognition outcomes here.
