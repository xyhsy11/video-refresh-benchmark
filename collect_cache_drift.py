"""Collect CUDA reference features and pre-decision cache drift; not a latency test."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time

# Preserve the original benchmark's CUDA initialization and numerical settings.
from benchmark import Encoder, configure_numerics, read_inputs
import av
import numpy as np
import torch

from motion_utils import (DESCRIPTOR_CONFIG, EXPORT_CONFIG, configure_motion_decoder,
                          current_environment, frame_descriptor, write_csv)
from research_utils import (ROOT, DECODER_THREAD_COUNT, DECODER_THREAD_TYPE, load_manifest,
                            now, relative_path, selected_input_paths, sha256_file, write_json)

AGE_INPUTS = ["age_seconds", "age_decoded_frames", "age_candidate_positions", "age_seconds_squared"]
MOTION_INPUTS = ["current_mean_magnitude_px", "current_p90_magnitude_px", "current_mv_available",
                 "interval_mean_available_magnitude_px", "interval_max_available_p90_px",
                 "interval_mv_available_fraction"]
AUDIT_COLUMNS = ["observation_id", "video_id", "source_video_id", "relative_path", "candidate_ordinal",
                 "cached_candidate_ordinal", "current_frame_index", "cached_frame_index",
                 "current_timestamp_seconds", "cached_timestamp_seconds", "scheduled_refresh_after_label",
                 "interval_start_frame", "interval_end_frame", "interval_frame_count",
                 "interval_available_frame_count"]
OBSERVATION_COLUMNS = AUDIT_COLUMNS + AGE_INPUTS + MOTION_INPUTS + ["appearance_feature_drift"]
COLLECTION_SOURCES = ["collect_cache_drift.py", "benchmark.py", "research_utils.py", "motion_utils.py", "check_env.py"]
ORIGINAL_SOURCES = ["benchmark.py", "research_utils.py", "inspect_video.py", "summarize.py",
                    "check_env.py", "prepare_data.py", "validate_release.py", "motion_utils.py",
                    "inspect_motion.py", "benchmark_motion.py", "summarize_motion.py",
                    "requirements.txt", "requirements-cuda.txt"]
TARGET_CONFIG = {
    "name": "appearance-feature drift", "range": [0, 2],
    "definition": "1 - clip(dot(f_t,f_r)/(norm(f_t)*norm(f_r)), -1, 1), computed in float64",
    "minimum_safe_norm": 1e-12, "excluded_candidate_ordinal": 0,
    "cached_ordinal": "refresh_every * floor((t - 1) / refresh_every)",
    "decision_order": "compute label against previous cache BEFORE a scheduled refresh at t",
    "not_measured": ["action-recognition error", "refresh benefit", "latency", "recognition accuracy"],
}
INPUT_CONFIG = {
    "age": AGE_INPUTS, "motion": MOTION_INPUTS,
    "interval": "decoded frames strictly after cached candidate r through current candidate t inclusive",
    "interval_mean": "equal weight per frame with available vectors, not per vector entry",
    "interval_max": "maximum per-frame p90 among frames with available vectors",
    "interval_missing": "zero magnitude summaries and zero fraction if no available frames",
    "availability": "current flag and interval fraction distinguish missing from observed zero motion",
    "forbidden_predictors": ["RGB features", "target-derived columns", "video identifiers", "future frames", "whole-video statistics"],
}


def protected_inventory() -> dict:
    paths = [ROOT / name for name in ORIGINAL_SOURCES]
    for directory in ["pilot_2026_10_10", "motion_overhead_2026_10_11"]:
        paths.extend(p for p in (ROOT / "results" / directory).rglob("*") if p.is_file())
    return {relative_path(p): sha256_file(p) for p in sorted(paths)}


def check_protected(expected: dict) -> None:
    actual = protected_inventory()
    if actual != expected:
        changed = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
        raise AssertionError(f"Existing experiment sources/snapshots changed: {changed}")


def fresh_directory(requested: Path) -> Path:
    """Never overwrite a previous collection, including partially completed work."""
    if not requested.exists() or not any(requested.iterdir()):
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    stamp = now().replace(":", "").split(".")[0].replace("-", "")
    candidate = requested.with_name(requested.name + "_" + stamp)
    suffix = 1
    while candidate.exists():
        candidate = requested.with_name(requested.name + "_" + stamp + f"_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    print(f"Preserved existing output; using {relative_path(candidate)}", flush=True)
    return candidate


def cache_ordinal(t: int, refresh_every: int) -> int:
    if t < 1 or refresh_every < 1:
        raise ValueError("A drift label needs a previous cache and a positive refresh interval")
    return refresh_every * ((t - 1) // refresh_every)


def cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != (512,) or b.shape != (512,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Expected finite 512-dimensional features")
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not np.isfinite([norm_a, norm_b]).all() or min(norm_a, norm_b) <= TARGET_CONFIG["minimum_safe_norm"]:
        raise ValueError("Unsafe/nonfinite feature norm")
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    if not np.isfinite(cosine):
        raise ValueError("Nonfinite cosine")
    return 1.0 - float(np.clip(cosine, -1.0, 1.0))


def interval_inputs(frames: list[dict], cached_index: int, current_index: int) -> dict:
    if not 0 <= cached_index < current_index < len(frames):
        raise ValueError("Invalid interval endpoints")
    interval = frames[cached_index + 1:current_index + 1]
    if [row["decoded_frame_index"] for row in interval] != list(range(cached_index + 1, current_index + 1)):
        raise AssertionError("Frame descriptors are not in sequential decoded order")
    available = [row for row in interval if row["mv_available"]]
    current = frames[current_index]
    return {
        "current_mean_magnitude_px": current["mean_magnitude_px"],
        "current_p90_magnitude_px": current["p90_magnitude_px"],
        "current_mv_available": int(current["mv_available"]),
        "interval_mean_available_magnitude_px": float(np.mean([row["mean_magnitude_px"] for row in available])) if available else 0.0,
        "interval_max_available_p90_px": max(row["p90_magnitude_px"] for row in available) if available else 0.0,
        "interval_mv_available_fraction": len(available) / len(interval),
        "interval_start_frame": cached_index + 1, "interval_end_frame": current_index,
        "interval_frame_count": len(interval), "interval_available_frame_count": len(available),
    }


def observations_for_video(entry: dict, features: np.ndarray, candidates: list[dict], frames: list[dict], refresh_every: int) -> list[dict]:
    rows = []
    if features.shape != (len(candidates), 512) or features.dtype != np.float32:
        raise AssertionError("Reference feature shape/dtype mismatch")
    for t in range(1, len(candidates)):
        r = cache_ordinal(t, refresh_every)
        current, cached = candidates[t], candidates[r]
        elapsed = current["timestamp_seconds"] - cached["timestamp_seconds"]
        if elapsed <= 0:
            raise AssertionError("Nonpositive timestamp age")
        rows.append({
            "observation_id": f"{entry['video_id']}:{t:02d}", "video_id": entry["video_id"],
            "source_video_id": entry["source_video_id"], "relative_path": entry["relative_path"],
            "candidate_ordinal": t, "cached_candidate_ordinal": r,
            "current_frame_index": current["decoded_frame_index"], "cached_frame_index": cached["decoded_frame_index"],
            "current_timestamp_seconds": current["timestamp_seconds"], "cached_timestamp_seconds": cached["timestamp_seconds"],
            "scheduled_refresh_after_label": int(t % refresh_every == 0),
            "age_seconds": elapsed, "age_decoded_frames": current["decoded_frame_index"] - cached["decoded_frame_index"],
            "age_candidate_positions": t - r, "age_seconds_squared": elapsed * elapsed,
            **interval_inputs(frames, cached["decoded_frame_index"], current["decoded_frame_index"]),
            "appearance_feature_drift": cosine_distance(features[t], features[r]),
        })
    if not np.isfinite([[row[name] for name in AGE_INPUTS + MOTION_INPUTS + ["appearance_feature_drift"]] for row in rows]).all():
        raise AssertionError("Nonfinite observation")
    return rows


def source_group(entry: dict) -> str:
    # The original dataset names are eleven-character YouTube source IDs. If a
    # later selection adds temporal suffixes, those files still share a group.
    if entry.get("source_video_id"):
        return entry["source_video_id"]
    stem = Path(entry["relative_path"]).stem
    match = re.fullmatch(r"([A-Za-z0-9_-]{11})(?:_.*)?", stem)
    if not match:
        raise ValueError("Cannot infer a source-video group; add source_video_id to the manifest")
    return match.group(1)


def check_published_inputs(entry: dict) -> None:
    for directory in ["pilot_2026_10_10", "motion_overhead_2026_10_11"]:
        published = json.loads((ROOT / "results" / directory / "manifest.json").read_text(encoding="utf-8"))
        old = published["videos"][entry["relative_path"]]
        for name in ["video_id", "candidate_indices", "decoded_frame_count"]:
            if old[name] != entry[name]:
                raise AssertionError(f"Input differs from published experiment: {name}")
        if old["file_identity"]["sha256"] != entry["file_identity"]["sha256"]:
            raise AssertionError("Video hash differs from published experiment")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--refresh-every", type=int, default=4)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.candidates, args.refresh_every) != (16, 4):
        parser.error("This fixed diagnostic uses the original 16 candidates and refresh interval 4")
    protected = protected_inventory()
    output = fresh_directory(args.output)
    environment = current_environment()
    write_json(output / "environment.json", environment)
    if not torch.cuda.is_available():
        write_json(output / "verification.json", {"passed": False, "blocker": "CUDA unavailable", "checked_at_utc": now()})
        print("BLOCKER: CUDA unavailable; see environment.json. No CPU feature collection substituted.", flush=True)
        return 2
    manifest = load_manifest()
    inputs = read_inputs(selected_input_paths(args.input, manifest), manifest, args.candidates)
    for entry, _, _ in inputs:
        check_published_inputs(entry)
    numerics = configure_numerics(20261012)
    device = torch.device("cuda:0")
    encoder = Encoder(device)
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / encoder.weights.url.rsplit("/", 1)[-1]
    old_metadata = json.loads((ROOT / "results/motion_overhead_2026_10_11/pilot.metadata.json").read_text(encoding="utf-8"))
    if numerics != old_metadata["numerical_settings"] or sha256_file(checkpoint) != old_metadata["model"]["weights_sha256"]:
        raise AssertionError("Numerics/weights differ from the previous experiment")
    metadata = {
        "experiment": "Predicting cached appearance-feature drift from motion and cache age",
        "started_at_utc": now(), "status": "collecting", "device": str(device), "diagnostic_not_latency_benchmark": True,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "source_code_sha256": {name: sha256_file(ROOT / name) for name in COLLECTION_SOURCES},
        "protected_original_sha256": protected, "input_manifest_sha256": sha256_file(ROOT / "results/manifest.json"),
        "dataset": manifest.get("dataset", {}), "environment": environment,
        "decoder": {"thread_count": DECODER_THREAD_COUNT, "thread_type": DECODER_THREAD_TYPE, "skip_frame": "NONE", "hardware_acceleration": False},
        "motion_export": EXPORT_CONFIG, "motion_descriptor_definition": DESCRIPTOR_CONFIG,
        "numerical_settings": numerics, "target": TARGET_CONFIG, "predictor_inputs": INPUT_CONFIG,
        "candidates": args.candidates, "refresh_every": args.refresh_every,
        "model": {"name": "resnet18", "weights": str(encoder.weights), "weights_url": encoder.weights.url,
                  "weights_sha256": sha256_file(checkpoint), "fc": "torch.nn.Identity", "preprocessing": repr(encoder.preprocess),
                  "feature_shape_per_call": [1, 512], "saved_array_shape_per_video": [16, 512], "saved_dtype": "float32"},
        "offline": "Existing B-frames and future-reference codec vectors prevent strictly causal streaming claims",
        "timestamp_policy": "decoded PTS * time_base; reject missing/nonfinite/nonincreasing candidate timestamps",
        "source_group_policy": "manifest source_video_id if set; otherwise eleven-character YouTube filename prefix",
        "videos": [],
    }
    write_json(output / "metadata.json", metadata)
    (output / "features").mkdir()
    observations, all_frames, checks = [], [], []
    with torch.inference_mode():
        for original, encoded, loading_ms in inputs:
            entry = {**original, "source_video_id": source_group(original)}
            started = now()
            start = time.perf_counter()
            frames, candidates, gpu_features = [], [], []
            types = Counter()
            ordinal = 0
            with av.open(io.BytesIO(encoded), mode="r") as container:
                stream = container.streams.video[0]
                configure_motion_decoder(stream)
                for index, frame in enumerate(container.decode(stream)):
                    descriptor = frame_descriptor(frame, entry["video_id"], index)
                    stamp = float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base is not None else None
                    kind = av.video.frame.PictureType(frame.pict_type).name
                    types[kind] += 1
                    frames.append({"video_id": entry["video_id"], "decoded_frame_index": index, "pts": frame.pts,
                                   "timestamp_seconds": stamp, "frame_type": kind, **asdict(descriptor)})
                    if ordinal < args.candidates and index == entry["candidate_indices"][ordinal]:
                        feature = encoder(frame)
                        if feature.shape != (1, 512) or feature.dtype != torch.float32 or feature.device != device:
                            raise AssertionError("Incorrect encoder feature shape/dtype/device")
                        gpu_features.append(feature)
                        candidates.append({"candidate_ordinal": ordinal, "decoded_frame_index": index, "pts": frame.pts,
                                           "timestamp_seconds": stamp, "frame_type": kind})
                        ordinal += 1
                flags2 = int(stream.codec_context.flags2)
            torch.cuda.synchronize(device)
            features = torch.cat(gpu_features, dim=0).cpu().numpy()
            del gpu_features
            if len(frames) != entry["decoded_frame_count"] or ordinal != 16 or dict(types) != entry["frame_type_counts"]:
                raise AssertionError("Decoded frames/types/candidate counts differ from the saved inspection")
            if not flags2 & int(av.codec.context.Flags2.export_mvs) or sum(f["vector_count"] for f in frames) == 0:
                raise AssertionError("Motion export produced no vectors or was not applied")
            stamps = [c["timestamp_seconds"] for c in candidates]
            if any(s is None for s in stamps) or not np.isfinite(stamps).all() or not (np.diff(stamps) > 0).all():
                raise AssertionError("Candidate timestamps missing/nonfinite/nonincreasing")
            norms = np.linalg.norm(features.astype(np.float64), axis=1)
            if not np.isfinite(features).all() or not np.isfinite(norms).all() or min(norms) <= 1e-12:
                raise AssertionError("Reference features/norms are unsafe")
            rows = observations_for_video(entry, features, candidates, frames, args.refresh_every)
            if len(rows) != 15 or any(cache_ordinal(t, 4) != r for t, r in [(4, 0), (8, 4), (12, 8)]):
                raise AssertionError("Wrong pre-decision labels/counts")
            np.savez_compressed(output / "features" / f"{entry['video_id']}.npz", features=features,
                                frame_indices=np.array(entry["candidate_indices"], dtype=np.int64),
                                timestamps_seconds=np.array(stamps, dtype=np.float64))
            write_json(output / "features" / f"{entry['video_id']}.json", {
                "video_id": entry["video_id"], "source_video_id": entry["source_video_id"],
                "relative_path": entry["relative_path"], "file_identity": entry["file_identity"],
                "candidates": candidates, "feature_norm_min": float(min(norms)), "feature_norm_max": float(max(norms))})
            observations.extend(rows)
            all_frames.extend(frames)
            checks.append({"video_id": entry["video_id"], "decoded_frames": len(frames), "reference_features": len(features),
                           "observations": len(rows), "candidate_indices_match": True, "pre_decision_schedule": True,
                           "finite_features_inputs_labels": True, "safe_feature_norms": True, "passed": True})
            metadata["videos"].append({"video_id": entry["video_id"], "source_video_id": entry["source_video_id"],
                                      "relative_path": entry["relative_path"], "file_identity": entry["file_identity"],
                                      "candidate_indices": entry["candidate_indices"], "decoded_frame_count": len(frames),
                                      "frame_type_counts": dict(types), "applied_flags2": flags2,
                                      "collection_started_at_utc": started, "collection_completed_at_utc": now(),
                                      "file_loading_ms_separate": loading_ms, "diagnostic_collection_ms_not_benchmark": (time.perf_counter()-start)*1000,
                                      **{key: sum(f[key] for f in frames) for key in ["vector_count", "past_reference_vectors", "future_reference_vectors", "other_reference_vectors"]},
                                      "mv_frames_available": sum(f["mv_available"] for f in frames)})
            print(f"{entry['video_id']}: {len(frames)} frames, 16 CUDA reference features, 15 pre-decision drift observations PASS", flush=True)
    write_csv(output / "observations.csv", observations, OBSERVATION_COLUMNS)
    write_csv(output / "motion_frames.csv", all_frames, list(all_frames[0]))
    check_protected(protected)
    metadata.update({"status": "collected", "completed_at_utc": now(), "video_count": len(inputs),
                     "reference_feature_count": 16*len(inputs), "observation_count": len(observations)})
    metadata["collection_artifact_sha256"] = {relative_path(p).removeprefix(relative_path(output)+"/"): sha256_file(p)
                                              for p in output.rglob("*") if p.is_file() and p.name not in ["metadata.json", "verification.json"]}
    write_json(output / "metadata.json", metadata)
    write_json(output / "verification.json", {"collection": {"passed": True, "checked_at_utc": now(), "rows": checks,
                "original_sources_and_snapshots_unchanged": True, "protected_file_count": len(protected)},
                "analysis": {"status": "not_run"}})
    print(f"Saved {len(inputs)} videos, {16*len(inputs)} features and {len(observations)} observations in {relative_path(output)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
