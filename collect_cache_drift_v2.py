"""CUDA reference collection using a separate v2 manifest; no latency measurement."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import time

from benchmark import Encoder, configure_numerics
import av
import numpy as np
import torch

from cache_drift_v2_utils import (SEED, freeze_protocol, preservation_inventory, read_json,
                                 validate_split, verify_preservation)
from collect_cache_drift import (OBSERVATION_COLUMNS, TARGET_CONFIG, INPUT_CONFIG,
                                fresh_directory, observations_for_video)
from motion_utils import (DESCRIPTOR_CONFIG, EXPORT_CONFIG, configure_motion_decoder,
                          current_environment, frame_descriptor, write_csv)
from research_utils import ROOT, now, relative_path, sha256_file, write_json

SOURCE_FILES = ["collect_cache_drift_v2.py", "cache_drift_v2_utils.py", "prepare_cache_drift_v2.py",
                "collect_cache_drift.py", "benchmark.py", "motion_utils.py", "research_utils.py", "inspect_video.py"]


def collect_one(entry: dict, encoder: Encoder) -> tuple[np.ndarray, list[dict], list[dict], list[dict], dict]:
    path = ROOT / entry["relative_path"]
    start = time.perf_counter()
    encoded = path.read_bytes()
    load_ms = (time.perf_counter()-start)*1000
    identity = entry["file_identity"]
    if len(encoded) != identity["size_bytes"] or hashlib.sha256(encoded).hexdigest() != identity["sha256"]:
        raise ValueError("Encoded input differs from v2 manifest")
    frames, candidates, features_gpu = [], [], []
    types, ordinal = Counter({"I": 0, "P": 0, "B": 0}), 0
    with av.open(io.BytesIO(encoded)) as container:
        stream = container.streams.video[0]
        if stream.codec_context.name != "h264":
            raise ValueError("V2 requires original H.264 encoding")
        configure_motion_decoder(stream)
        for index, frame in enumerate(container.decode(stream)):
            kind = av.video.frame.PictureType(frame.pict_type).name
            types[kind] += 1
            stamp = float(frame.pts*frame.time_base) if frame.pts is not None and frame.time_base is not None else None
            descriptor = frame_descriptor(frame, entry["video_id"], index)
            frames.append({"video_id": entry["video_id"], "decoded_frame_index": index, "pts": frame.pts,
                           "timestamp_seconds": stamp, "frame_type": kind, **asdict(descriptor)})
            if ordinal < 16 and index == entry["candidate_indices"][ordinal]:
                feature = encoder(frame)
                if feature.shape != (1, 512) or feature.dtype != torch.float32 or feature.device != encoder.device:
                    raise ValueError("Incorrect reference feature shape/dtype/device")
                features_gpu.append(feature)
                candidates.append({"candidate_ordinal": ordinal, "decoded_frame_index": index,
                                   "pts": frame.pts, "timestamp_seconds": stamp, "frame_type": kind})
                ordinal += 1
        flags2 = int(stream.codec_context.flags2)
    torch.cuda.synchronize(encoder.device)
    if len(frames) != entry["decoded_frame_count"] or ordinal != 16 or dict(types) != entry["frame_type_counts"]:
        raise ValueError("Actual decoded counts/types/candidates differ from v2 manifest")
    if not flags2 & int(av.codec.context.Flags2.export_mvs):
        raise ValueError("Motion export configuration not applied")
    stamps = [c["timestamp_seconds"] for c in candidates]
    if any(s is None for s in stamps) or not np.isfinite(stamps).all() or not (np.diff(stamps) > 0).all():
        raise ValueError("Invalid increasing candidate timestamps")
    np.testing.assert_allclose(stamps, entry["candidate_timestamps_seconds"], rtol=0, atol=1e-12)
    features = torch.cat(features_gpu).cpu().numpy()
    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    if not np.isfinite(features).all() or not np.isfinite(norms).all() or min(norms) <= 1e-12:
        raise ValueError("Nonfinite/unsafe reference features")
    observations = observations_for_video(entry, features, candidates, frames, 4)
    for row in observations:
        row.update({"label": entry["label"], "experiment_split": entry["experiment_split"]})
    motion = {key: sum(f[key] for f in frames) for key in ["vector_count", "past_reference_vectors", "future_reference_vectors", "other_reference_vectors"]}
    info = {"decoded_frames": len(frames), "reference_features": 16, "observations": 15,
            "file_loading_ms_separate": load_ms, "frame_type_counts": dict(types), "applied_flags2": flags2,
            "mv_frames_available": sum(f["mv_available"] for f in frames),
            "mv_frames_missing": sum(not f["mv_available"] for f in frames),
            "entire_clip_motion_missing": not any(f["mv_available"] for f in frames), **motion}
    return features, candidates, frames, observations, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Collect two training clips, without freezing an evaluation split")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    output = args.output
    if (output / "split_manifest.json").exists() or ((output / "collection_metadata.json").exists() and not args.resume):
        output = fresh_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol_hash = freeze_protocol(output)
    environment = current_environment()
    write_json(output / "environment.json", environment)
    if not environment["passed"] or not torch.cuda.is_available():
        write_json(output / "collection_metadata.json", {"status": "incomplete", "blocker": "CUDA/environment checks failed", "at_utc": now()})
        print("INCOMPLETE: CUDA environment failed; no CPU reference collection substituted.", flush=True)
        return 2
    if manifest["status"] != "prepared":
        write_json(output / "collection_metadata.json", {"status": "incomplete", "blocker": "dataset shortfall",
                   "selected": len(manifest["videos"]), "missing": 200-len(manifest["videos"]), "shortfall_per_class": manifest["shortfall_per_class"]})
        print(f"INCOMPLETE: dataset {len(manifest['videos'])}/200. Resume preparation.", flush=True)
        return 2
    validate_split(manifest)
    protected = preservation_inventory()
    numerics = configure_numerics(SEED)
    encoder = Encoder(torch.device("cuda:0"))
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / encoder.weights.url.rsplit("/", 1)[-1]
    original = read_json(ROOT / "results/cache_drift_v1/metadata.json")
    assert numerics == original["numerical_settings"] and sha256_file(checkpoint) == original["model"]["weights_sha256"]
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    progress_path = output / "collection_progress.json"
    progress = read_json(progress_path) if args.resume and progress_path.exists() else {
        "source_code_sha256": sources, "input_manifest_sha256": sha256_file(args.manifest), "videos": [], "replacement_records": []}
    assert progress["source_code_sha256"] == sources and progress["input_manifest_sha256"] == sha256_file(args.manifest)
    metadata = {"experiment": "cache_drift_v2", "status": "collecting", "started_at_utc": now(), "device": "cuda:0",
                "smoke": args.smoke, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_code_sha256": sources, "protected_original_sha256": protected,
                "input_manifest": relative_path(args.manifest), "input_manifest_sha256": sha256_file(args.manifest),
                "protocol_sha256": protocol_hash, "model": original["model"], "numerical_settings": numerics,
                "environment": environment, "decoder": original["decoder"], "motion_export": EXPORT_CONFIG,
                "motion_descriptor_definition": DESCRIPTOR_CONFIG, "target": TARGET_CONFIG, "predictor_inputs": INPUT_CONFIG,
                "offline_not_latency_benchmark": True}
    write_json(output / "collection_metadata.json", metadata)
    for directory in ["features", "motion_frames", "video_observations"]:
        (output / directory).mkdir(exist_ok=True)
    selected = sorted(manifest["videos"], key=lambda e: (e["experiment_split"] != "train", e["label"], e["video_id"]))
    if args.smoke:
        selected = [e for e in selected if e["experiment_split"] == "train"][:2]
    completed = {item["slot_video_id"]: item for item in progress["videos"]}
    used_ids = {e["source_video_id"] for e in selected} | {e["entry"]["source_video_id"] for e in progress["videos"]}
    used_hashes = {e["file_identity"]["sha256"] for e in selected} | {e["entry"]["file_identity"]["sha256"] for e in progress["videos"]}
    failed = False
    with torch.inference_mode():
        for slot in selected:
            if slot["video_id"] in completed:
                for path, digest in completed[slot["video_id"]]["artifact_sha256"].items():
                    assert sha256_file(output / path) == digest
                continue
            queue = [slot] + [{**e, "experiment_split": slot["experiment_split"]} for e in sorted(manifest["reserves"], key=lambda e: e["sampling_rank"])
                              if e["label"] == slot["label"] and e["source_video_id"] not in used_ids and e["file_identity"]["sha256"] not in used_hashes]
            success = None
            for entry in queue:
                try:
                    start = now()
                    features, candidates, frames, rows, info = collect_one(entry, encoder)
                except (ValueError, RuntimeError, av.error.FFmpegError) as exc:
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        failed = True
                        progress["blocker"] = "CUDA out of memory; no drift-based replacement performed"
                        break
                    progress["replacement_records"].append({"slot_video_id": slot["video_id"], "rejected_video_id": entry["video_id"],
                                                             "reason": f"{type(exc).__name__}: {exc}", "at_utc": now(), "rule": "next same-class eligible sampling rank"})
                    continue
                entry = {**entry, "eligibility": "decoding/timestamps and finite CUDA feature extraction passed"}
                video_id = entry["video_id"]
                np.savez_compressed(output / "features" / f"{video_id}.npz", features=features,
                                    frame_indices=np.array(entry["candidate_indices"], dtype=np.int64),
                                    timestamps_seconds=np.array([c["timestamp_seconds"] for c in candidates], dtype=np.float64))
                write_json(output / "features" / f"{video_id}.json", {"entry": entry, "candidates": candidates})
                write_csv(output / "motion_frames" / f"{video_id}.csv", frames, list(frames[0]))
                write_csv(output / "video_observations" / f"{video_id}.csv", rows, OBSERVATION_COLUMNS+["label", "experiment_split"])
                paths = [f"features/{video_id}.npz", f"features/{video_id}.json", f"motion_frames/{video_id}.csv", f"video_observations/{video_id}.csv"]
                success = {"slot_video_id": slot["video_id"], "entry": entry, "started_at_utc": start, "completed_at_utc": now(),
                           "counts": info, "artifact_sha256": {p: sha256_file(output/p) for p in paths}}
                progress["videos"].append(success)
                completed[slot["video_id"]] = success
                used_ids.add(entry["source_video_id"])
                used_hashes.add(entry["file_identity"]["sha256"])
                print(f"Collected {len(progress['videos'])}/{len(selected)} {video_id}: {info['decoded_frames']} frames; 16 features, 15 labels; missing MV {info['mv_frames_missing']}", flush=True)
                break
            write_json(progress_path, progress)
            if not success:
                failed = True
                progress.setdefault("blocker", f"No eligible replacement for {slot['video_id']}")
                break
    entries = [v["entry"] for v in progress["videos"]]
    rows = []
    for item in progress["videos"]:
        video_id = item["entry"]["video_id"]
        import csv
        with (output / "video_observations" / f"{video_id}.csv").open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if rows:
        write_csv(output / "observations.csv", rows, OBSERVATION_COLUMNS+["label", "experiment_split"])
    metadata.update({"status": "incomplete" if failed else "smoke_passed" if args.smoke else "collected",
                     "completed_at_utc": now(), "video_count": len(entries), "reference_feature_count": 16*len(entries),
                     "observation_count": len(rows), "counts_by_split": {s: sum(e["experiment_split"] == s for e in entries) for s in ["train", "validation", "test"]},
                     "blocker": progress.get("blocker"), "replacement_records": progress["replacement_records"],
                     "observations_sha256": sha256_file(output / "observations.csv") if rows else None})
    if not failed and not args.smoke:
        frozen = {**manifest, "videos": entries, "reserves": [], "feature_eligibility_pending": False,
                  "status": "frozen", "frozen_at_utc": now(), "replacement_records": progress["replacement_records"]}
        metadata["split_validation"] = validate_split(frozen)
        write_json(output / "split_manifest.json", frozen)
        metadata["split_manifest_sha256"] = sha256_file(output / "split_manifest.json")
    verify_preservation(protected)
    write_json(output / "collection_metadata.json", metadata)
    write_json(output / "collection_verification.json", {"passed": not failed, "smoke": args.smoke,
               "originals_unchanged": True, "reference_features_per_video": 16, "labels_per_video": 15,
               "missing_vectors_retained": True, "device": "cuda:0", "counts_by_split": metadata["counts_by_split"]})
    print(f"Collection {metadata['status']}: {len(entries)} videos, {16*len(entries)} features, {len(rows)} labels; {relative_path(output)}", flush=True)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
