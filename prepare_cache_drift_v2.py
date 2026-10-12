"""Resumable bounded CVDF shard acquisition and deterministic v2 eligibility/splits."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import shutil
import tarfile
import time
import urllib.request

import av
import numpy as np

from cache_drift_v2_utils import (BASE, SOURCE, SEED, freeze_protocol, pilot_sources,
                                 preservation_inventory, read_json, validate_split, verify_preservation)
from collect_cache_drift import fresh_directory
from inspect_video import inspect
from research_utils import ROOT, configure_decoder, now, relative_path, sha256_file, write_json

CACHE = ROOT / "data/source/cache_drift_v2"
ELIGIBILITY_ERRORS = (ValueError, RuntimeError, av.error.FFmpegError)


def rank(row: dict) -> str:
    return hashlib.sha256(f"{SEED}|{row['label']}|{row['filename']}".encode()).hexdigest()


def metadata_download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read()
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def download_shard(url: str, destination: Path, workers: int = 8) -> dict:
    """Range chunks persist across interruptions; exact response boundaries checked."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=45) as r:
        size, etag = int(r.headers["Content-Length"]), r.headers.get("ETag")
    info_path = destination.with_suffix(".download.json")
    if destination.exists() and info_path.exists():
        old = read_json(info_path)
        if destination.stat().st_size == size and old["etag"] == etag and sha256_file(destination) == old["sha256"]:
            return {**old, "reused": True}
        raise ValueError("Existing archive identity differs; preserve it and choose another cache")
    chunks = destination.with_name(destination.name + ".chunks")
    chunks.mkdir(parents=True, exist_ok=True)
    remote_path = chunks / "remote.json"
    identity = {"url": url, "size_bytes": size, "etag": etag}
    if remote_path.exists() and read_json(remote_path) != identity:
        raise ValueError("Remote archive changed since partial download")
    write_json(remote_path, identity)
    block = 64*1024*1024
    ranges = [(i, start, min(start+block, size)-1) for i, start in enumerate(range(0, size, block))]

    def transfer(spec):
        index, start, end = spec
        part = chunks / f"{index:04d}.part"
        have = part.stat().st_size if part.exists() else 0
        if have > end-start+1:
            raise ValueError("Oversized range chunk")
        for attempt in range(4):
            if have == end-start+1:
                return
            request = urllib.request.Request(url, headers={"Range": f"bytes={start+have}-{end}", "If-Match": etag})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start+have}-{end}/{size}":
                        raise ValueError("Server did not honor the requested resumable byte range")
                    with part.open("ab") as handle:
                        while data := response.read(min(1024*1024, end-start+1-have)):
                            handle.write(data)
                            have += len(data)
                            if have == end-start+1:
                                break
                if have != end-start+1:
                    raise OSError("Truncated range response")
            except (OSError, TimeoutError):
                have = part.stat().st_size if part.exists() else 0
                if attempt == 3:
                    raise
                time.sleep(min(2**attempt, 8))
        print(f"  {destination.name}: chunk {index+1}/{len(ranges)} complete", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(transfer, ranges))
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("wb") as handle:
        for index, _, _ in ranges:
            with (chunks / f"{index:04d}.part").open("rb") as part:
                shutil.copyfileobj(part, handle, 1024*1024)
    assert temporary.stat().st_size == size
    temporary.replace(destination)
    result = {**identity, "sha256": sha256_file(destination), "completed_at_utc": now(), "reused": False}
    write_json(info_path, result)
    for index, _, _ in ranges:
        (chunks / f"{index:04d}.part").unlink()  # only this download's known chunks
    return result


def inspect_eligible(path: Path, annotation: dict, archive: dict) -> dict:
    entry = inspect(path, 16)
    if entry["codec"] != "h264":
        raise ValueError(f"Not H.264: {entry['codec']}")
    pts = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        configure_decoder(stream)
        for frame in container.decode(stream):
            pts.append(float(frame.pts*frame.time_base) if frame.pts is not None and frame.time_base is not None else None)
    times = [pts[i] for i in entry["candidate_indices"]]
    if any(t is None for t in times) or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("Invalid or nonincreasing candidate timestamps")
    return {**entry, "source_video_id": annotation["youtube_id"], "label": annotation["label"],
            "original_dataset_split": annotation["split"], "time_start": int(annotation["time_start"]),
            "time_end": int(annotation["time_end"]), "candidate_timestamps_seconds": times,
            "archive_url": archive["url"], "archive_sha256": archive["sha256"],
            "sampling_rank": rank(annotation), "eligibility": "decode/timestamps passed; finite CUDA features pending"}


def prepare(output: Path, limit_gib: float, workers: int) -> int:
    output.mkdir(parents=True, exist_ok=True)
    protocol_hash = freeze_protocol(output)
    acquisition_path = output / "acquisition.json"
    state = read_json(acquisition_path) if acquisition_path.exists() else {
        "created_at_utc": now(), "source": SOURCE, "compressed_download_limit_gib": limit_gib,
        "archives": [], "events": [], "protected_original_sha256": preservation_inventory()}
    verify_preservation(state["protected_original_sha256"])
    annotations_path = CACHE / "val.csv"
    paths_path = CACHE / "k400_val_path.txt"
    metadata_download(BASE+"annotations/val.csv", annotations_path)
    metadata_download(BASE+"val/k400_val_path.txt", paths_path)
    rows = list(csv.DictReader(annotations_path.open(encoding="utf-8", newline="")))
    for row in rows:
        row["filename"] = f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}.mp4"
    urls = paths_path.read_text(encoding="utf-8").splitlines()
    rng = random.Random(SEED)
    rng.shuffle(urls)
    classes = rng.sample(sorted(set(row["label"] for row in rows)), 20)
    pilots = pilot_sources()
    lookup = {row["filename"]: row for row in rows if row["label"] in classes and row["youtube_id"] not in pilots}
    plan_path = output / "sampling_plan.json"
    plan = {"sampling_seed": SEED, "source": SOURCE, "annotation_url": BASE+"annotations/val.csv",
            "annotations_sha256": sha256_file(annotations_path), "shard_list_sha256": sha256_file(paths_path),
            "ordered_shards": urls, "classes": classes, "excluded_pilot_sources": sorted(pilots),
            "candidate_priority": "SHA256(seed|label|filename), within acquired seeded archive subset",
            "stop_rule": "stop after the first shard prefix with >=10 eligible distinct source/file clips per class",
            "replacement_rule": "next eligible sampling-rank clip of the same class, no drift-based exclusions",
            "eligibility": "protocol.json; no exclusion for absent motion vectors",
            "protocol_sha256": protocol_hash}
    if plan_path.exists():
        assert read_json(plan_path) == plan
    else:
        write_json(plan_path, plan)
    candidates_path = output / "eligible_candidates.json"
    eligible = read_json(candidates_path).get("videos", []) if candidates_path.exists() else []
    exclusions_path = output / "exclusions.json"
    exclusions = read_json(exclusions_path).get("records", []) if exclusions_path.exists() else []
    known = {e["video_id"] for e in eligible} | {e.get("video_id") for e in exclusions}
    by_archive = {a["url"]: a for a in state["archives"]}
    for url in urls:
        if all(sum(e["label"] == c for e in eligible) >= 10 for c in classes):
            break
        destination = CACHE / "shards" / PurePosixPath(url).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if url not in by_archive:
            try:
                with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=45) as response:
                    size = int(response.headers["Content-Length"])
                reserved = sum(a["size_bytes"] for a in state["archives"])
                if reserved+size > int(limit_gib*1024**3):
                    state["events"].append({"url": url, "status": "download_limit_reached", "required_bytes": size,
                                             "reserved_bytes": reserved, "limit_bytes": int(limit_gib*1024**3), "at_utc": now()})
                    break
                # Reserve before download: failed/resumable chunks also count against the cap.
                archive = {"url": url, "size_bytes": size, "status": "downloading"}
                state["archives"].append(archive)
                by_archive[url] = archive
                write_json(acquisition_path, state)
            except OSError as exc:
                state["events"].append({"url": url, "status": "unavailable", "reason": str(exc), "at_utc": now()})
                write_json(acquisition_path, state)
                continue
        archive = by_archive[url]
        if archive.get("status") == "extracted":
            continue
        print(f"Acquiring {url}; reserved {sum(a['size_bytes'] for a in state['archives'])/1024**3:.2f} GiB", flush=True)
        try:
            archive.update(download_shard(url, destination, workers))
            archive["status"] = "downloaded"
            write_json(acquisition_path, state)
            extracted = []
            with tarfile.open(destination, mode="r|gz") as tar:
                for member in tar:
                    filename = PurePosixPath(member.name).name
                    if not member.isfile() or filename not in lookup:
                        continue
                    path = CACHE / "videos" / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        temporary = path.with_suffix(".mp4.part")
                        with tar.extractfile(member) as src, temporary.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        assert temporary.stat().st_size == member.size
                        temporary.replace(path)
                    extracted.append(path)
            # Fully reading tar validates gzip/tar without arbitrary path extraction.
            for path in sorted(extracted, key=lambda p: rank(lookup[p.name])):
                if path.stem in known:
                    continue
                known.add(path.stem)
                try:
                    entry = inspect_eligible(path, lookup[path.name], archive)
                    if any(e["source_video_id"] == entry["source_video_id"] or e["file_identity"]["sha256"] == entry["file_identity"]["sha256"] for e in eligible):
                        raise ValueError("Duplicate original source or identical encoded file")
                    eligible.append(entry)
                except ELIGIBILITY_ERRORS as exc:
                    exclusions.append({"video_id": path.stem, "label": lookup[path.name]["label"],
                                       "reason": f"{type(exc).__name__}: {exc}", "stage": "eligibility", "at_utc": now(), "archive_url": url})
            archive["status"] = "extracted"
            archive["matching_members"] = len(extracted)
        except (OSError, ValueError, tarfile.TarError) as exc:
            state["events"].append({"url": url, "status": "acquisition_failed_resumable", "reason": f"{type(exc).__name__}: {exc}", "at_utc": now()})
            write_json(acquisition_path, state)
            # Preserve this deterministic shard prefix; retry on resume rather than silently substituting a later shard.
            break
        write_json(candidates_path, {"videos": eligible})
        write_json(exclusions_path, {"records": exclusions})
        write_json(acquisition_path, state)
        print("Eligible per class: " + ", ".join(f"{c}={sum(e['label']==c for e in eligible)}" for c in classes), flush=True)
    selected, reserves = [], []
    for label in classes:
        available = sorted([e for e in eligible if e["label"] == label], key=lambda e: e["sampling_rank"])
        choice = available[:10]
        reserves.extend(available[10:])
        random.Random(f"{SEED}:split:{label}").shuffle(choice)
        for i, entry in enumerate(choice):
            selected.append({**entry, "experiment_split": "train" if i < 6 else "validation" if i < 8 else "test"})
    complete = len(selected) == 200
    manifest = {"schema_version": 2, "status": "prepared" if complete else "incomplete", "prepared_at_utc": now(),
                "sampling_seed": SEED, "source": SOURCE, "classes": classes, "videos": selected,
                "reserves": reserves, "excluded_pilot_sources": sorted(pilots), "protocol_sha256": protocol_hash,
                "shortfall_per_class": {c: max(0, 10-sum(e["label"] == c for e in selected)) for c in classes},
                "feature_eligibility_pending": True}
    if complete:
        validate_split(manifest)
    write_json(output / "dataset_manifest.json", manifest)
    state.update({"updated_at_utc": now(), "compressed_download_limit_gib": limit_gib,
                  "downloaded_compressed_bytes": sum(a["size_bytes"] for a in state["archives"] if a["status"] in ["downloaded", "extracted"]),
                  "eligible_candidates": len(eligible), "selected_count": len(selected), "status": manifest["status"]})
    write_json(acquisition_path, state)
    verify_preservation(state["protected_original_sha256"])
    print(f"Preparation {manifest['status']}: {len(selected)}/200; shortfall {200-len(selected)}. {relative_path(output)}", flush=True)
    return 0 if complete else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/cache_drift_v2"))
    parser.add_argument("--max-download-gib", type=float, default=12.0)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_download_gib <= 0 or not 1 <= args.download_workers <= 16:
        parser.error("Use positive download limit and 1..16 download workers")
    output = args.output
    if output.exists() and any(output.iterdir()) and (not args.resume or (output / "split_manifest.json").exists()):
        output = fresh_directory(output)
    return prepare(output, args.max_download_gib, args.download_workers)


if __name__ == "__main__":
    raise SystemExit(main())
