# UCF101 fixed-objective replication

See findings.md, protocol.md and status.json for measured scope and pending stages.

```powershell
& .\.venv_temporal\Scripts\python.exe .\run_ucf101_proxy_replication.py --resume
& .\.venv_temporal\Scripts\python.exe -B .\run_ucf101_proxy_replication.py --audit-only
```

The publication snapshot stores eligibility_progress.json and screened_population.json as lossless .json.gz copies to avoid GitHub file-size limits. Original raw files remain in this workspace; metadata_archive_packaging.json records both byte hashes. Decompress those copies to their original names when using an archival clone. Large real embeddings and full prediction vectors remain in data/source with hash ledgers.

See [completion review](completion_review.md), [post-exposure audit/reporting amendment](post_exposure_amendment.md), and [figure captions](figures/captions.md). All scientific outputs remain locked to their pre-correction hashes.
