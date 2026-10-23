# Active E27 resource amendment

The resumed 2026-10-23 brief authorizes **61 GiB cumulative new response-body
transfer**, including the preparation's metered 13,062,569 bytes, metadata,
complete shards, failures, interrupted reads and retries. The old 20 GiB network
block is superseded. Its original record remains in preparation_a9e1942/.

The additional retained-data cap remains 20 GiB; collection/CPU/timing caps remain
8h/4h/2h. The task separately enforces a 10 GiB free-disk reserve and bounded
single-shard scratch. Historical caches are never deleted. Retained originals,
metadata and transfer receipts count as retained; only task-owned temporary
compressed archive/partial bytes are separately reported as working space.

The initial full route priced 59 archives (20 validation, 39 test), with 18 authenticated
historical validation archives reusable. Its total new-body estimate including
then-metered metadata was 59,084,915,946 bytes, below 61*2^30. Actual execution has
now traversed all 20 validation archives and downloaded the two missing shards.
The final actual counters are in execution_accounting.json and network_ledger.json.
Known bodies and conservative interrupted-read upper charges are separate. The
4 MiB interrupted-read reserve is not presented as exact measured transfer.

The active blocker is **input access**, not the superseded network allowance:
the deterministic fitting prefix needs 7tTouR10Qro_000092_000102.mp4, absent from
every authenticated validation archive and local RGB inventory. Its public
YouTube parent reports that it cannot be played. A second prospective ice-fishing
clip, ieIssRi8iXU_000030_000040.mp4, is absent from the archives/local inventory
although public oEmbed metadata exists. No lower-ranked substitution, class
reduction or model exposure was used to bypass the first missing source.

No new resource amendment is requested. Authentic original inputs and provenance
must resolve this gate before fitting-only smoke, OOF planning or EVAL can begin.
No empirical precision, inference latency or scientific effect is estimated.
