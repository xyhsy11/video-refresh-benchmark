"""Six-condition CUDA benchmark of codec-vector CPU descriptor overhead."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
import hashlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

# Import the original benchmark before initializing torch/CUDA.
from benchmark import (Encoder, TrialResult, configure_numerics, process_clip,
                       read_inputs, validate_counts, validate_features)
import av
import torch

from motion_utils import (DESCRIPTOR_CONFIG, EXPORT_CONFIG, MODES, RESULT_DIR,
                          MotionDescriptor, configure_motion_decoder, current_environment,
                          frame_descriptor, gpu_telemetry, sanitize_public, write_csv)
from research_utils import (ROOT, DECODER_THREAD_COUNT, DECODER_THREAD_TYPE, load_manifest,
                            now, relative_path, selected_input_paths, sha256_file, write_json)

COLUMNS = ["video_id", "mode", "run_id", "decoded_frames", "candidate_count",
           "encoder_calls", "cache_reuses", "feature_outputs", "motion_updates",
           "mv_frames_available", "mv_frames_missing", "total_vectors", "total_ms"]
SOURCE_FILES = ["motion_utils.py", "inspect_motion.py", "benchmark_motion.py",
                "benchmark.py", "research_utils.py", "inspect_video.py", "check_env.py"]
MOTION_FIELDS = ["motion_updates", "mv_frames_available", "mv_frames_missing", "total_vectors"]


@dataclass
class MotionTrialResult(TrialResult):
    descriptors: list[MotionDescriptor] = field(default_factory=list)
    motion_updates: int = 0
    mv_frames_available: int = 0
    mv_frames_missing: int = 0
    total_vectors: int = 0
    past_reference_vectors: int = 0
    future_reference_vectors: int = 0
    other_reference_vectors: int = 0
    applied_flags2: int = 0


def process_motion_clip(encoded: bytes, entry: dict, mode: str,
                        encoder: Encoder, refresh_every: int) -> TrialResult:
    if not mode.endswith("_mv"):
        # The controls execute the original implementation, byte-for-byte unchanged.
        return process_clip(encoded, entry, mode, encoder, refresh_every)
    base = mode.removesuffix("_mv")
    result = MotionTrialResult()
    candidates = entry["candidate_indices"]
    ordinal, cached = 0, None
    with av.open(io.BytesIO(encoded), mode="r") as container:
        stream = container.streams.video[0]
        configure_motion_decoder(stream)
        for index, frame in enumerate(container.decode(stream)):
            result.decoded_frames += 1
            descriptor = frame_descriptor(frame, entry["video_id"], index)
            result.descriptors.append(descriptor)
            result.motion_updates += 1
            result.mv_frames_available += int(descriptor.mv_available)
            result.mv_frames_missing += int(not descriptor.mv_available)
            result.total_vectors += descriptor.vector_count
            result.past_reference_vectors += descriptor.past_reference_vectors
            result.future_reference_vectors += descriptor.future_reference_vectors
            result.other_reference_vectors += descriptor.other_reference_vectors
            if base == "decode":
                continue
            if ordinal < len(candidates) and index == candidates[ordinal]:
                if base == "dense" or ordinal % refresh_every == 0:
                    cached = encoder(frame)
                    result.encoder_calls += 1
                    result.refreshed_ordinals.append(ordinal)
                else:
                    result.cache_reuses += 1
                result.features.append(cached)
                ordinal += 1
        result.applied_flags2 = int(stream.codec_context.flags2)
    return result


def counters(result: TrialResult) -> dict:
    return {"decoded_frames": result.decoded_frames, "encoder_calls": result.encoder_calls,
            "cache_reuses": result.cache_reuses, "feature_outputs": len(result.features),
            **{name: getattr(result, name, 0) for name in MOTION_FIELDS}}


def validate_motion_counts(result: TrialResult, entry: dict, mode: str, refresh_every: int,
                           expected_motion: dict | None = None) -> None:
    validate_counts(result, entry, mode.removesuffix("_mv"), refresh_every)
    actual = counters(result)
    if not mode.endswith("_mv"):
        if any(actual[key] for key in MOTION_FIELDS):
            raise AssertionError("Motion counters must be zero when motion is disabled")
        return
    if result.motion_updates != result.decoded_frames or len(result.descriptors) != result.decoded_frames:
        raise AssertionError("One motion descriptor is required per decoded frame")
    if result.mv_frames_available + result.mv_frames_missing != result.decoded_frames or result.total_vectors <= 0:
        raise AssertionError("Invalid motion availability/counts; export may be unsupported")
    if result.past_reference_vectors + result.future_reference_vectors + result.other_reference_vectors != result.total_vectors:
        raise AssertionError("Reference-direction counts disagree")
    if not result.applied_flags2 & int(av.codec.context.Flags2.export_mvs):
        raise AssertionError("Decoder export_mvs flag was not applied")
    if expected_motion and any(actual[name] != expected_motion[name] for name in MOTION_FIELDS):
        raise AssertionError("Timed motion counters disagree with prior correctness verification")


def validate_descriptors(result: MotionTrialResult) -> None:
    for descriptor in result.descriptors:
        if not all(math.isfinite(value) and value >= 0 for value in
                   (descriptor.mean_magnitude_px, descriptor.p90_magnitude_px)):
            raise AssertionError("Nonfinite or negative motion descriptor")
        if descriptor.mv_available != (descriptor.vector_count > 0):
            raise AssertionError("Availability must distinguish missing from observed zero displacement")
        if descriptor.past_reference_vectors + descriptor.future_reference_vectors + descriptor.other_reference_vectors != descriptor.vector_count:
            raise AssertionError("Descriptor reference-direction counters disagree")
        if not descriptor.mv_available and descriptor != MotionDescriptor():
            raise AssertionError("Missing motion data must have zero-valued descriptors")
    if sum(d.vector_count for d in result.descriptors) != result.total_vectors:
        raise AssertionError("Retained descriptor vector counts disagree with total")
    if sum(d.mv_available for d in result.descriptors) != result.mv_frames_available:
        raise AssertionError("Retained descriptor availability counters disagree")


def verify(inputs: list, modes: list[str], encoder: Encoder, refresh_every: int) -> dict:
    rows, prior_features, prior_motion = [], [], []
    descriptor_hashes, motion_by_video = {}, {}
    for entry, encoded, _ in inputs:
        results = {}
        print(f"\n{entry['video_id']} | mode | frames | calls | reuses | features | motion | available | missing | vectors", flush=True)
        for mode in modes:
            for trial in range(2):
                result = process_motion_clip(encoded, entry, mode, encoder, refresh_every)
                validate_motion_counts(result, entry, mode, refresh_every)
                validate_features(result, mode.removesuffix("_mv"), refresh_every, encoder.device)
                if result.features:
                    first = result.features[0]
                    if 0 not in result.refreshed_ordinals or any(first is old or first.data_ptr() == old.data_ptr() for old in prior_features):
                        raise AssertionError("Appearance cache survived a trial/video, or first candidate did not refresh")
                    prior_features.append(first)
                if mode.endswith("_mv"):
                    validate_descriptors(result)
                    if any(result.descriptors is old or result.descriptors[0] is old[0] for old in prior_motion):
                        raise AssertionError("Motion state survived a trial/video")
                    prior_motion.append(result.descriptors)
                results[(mode, trial)] = result
                rows.append({"video_id": entry["video_id"], "mode": mode,
                             "verification_trial": trial + 1, **counters(result), "passed": True})
            c = counters(results[(mode, 0)])
            print(f"{mode:12s} | {c['decoded_frames']:4d} | {c['encoder_calls']:2d} | {c['cache_reuses']:2d} | {c['feature_outputs']:2d} | {c['motion_updates']:4d} | {c['mv_frames_available']:4d} | {c['mv_frames_missing']:3d} | {c['total_vectors']}", flush=True)
        motion_results = [result for (mode, _), result in results.items() if mode.endswith("_mv")]
        if motion_results:
            reference = motion_results[0]
            for result in motion_results[1:]:
                if reference.descriptors != result.descriptors or counters(reference)["total_vectors"] != counters(result)["total_vectors"]:
                    raise AssertionError("Motion descriptors differ across modes or repeated trials")
            descriptor_hashes[entry["video_id"]] = hashlib.sha256(json.dumps(
                [asdict(d) for d in reference.descriptors], sort_keys=True).encode()).hexdigest()
            motion_by_video[entry["video_id"]] = {
                **{key: getattr(reference, key) for key in MOTION_FIELDS},
                **{key: getattr(reference, key) for key in ["past_reference_vectors", "future_reference_vectors", "other_reference_vectors", "applied_flags2"]}}
        if "dense" in modes:
            dense = results[("dense", 0)]
            for (mode, _), result in results.items():
                for ordinal in result.refreshed_ordinals:
                    if not torch.allclose(dense.features[ordinal], result.features[ordinal], rtol=1e-5, atol=1e-6):
                        raise AssertionError(f"{mode}: refreshed appearance feature disagrees with original dense control")
    return {"passed": True, "checked_at_utc": now(), "device": str(encoder.device),
            "verification_trials_per_mode": 2, "rows": rows, "descriptor_sha256_by_video": descriptor_hashes,
            "motion_by_video": motion_by_video,
            "cross_video_state_isolation_checked": len(inputs) > 1,
            "checks": ["manifest frame counts", "appearance counts/shape/finiteness/cache equality",
                       "refreshed features vs original dense rtol=1e-5 atol=1e-6",
                       "first candidate refresh", "two fresh trials per mode",
                       "one finite CPU descriptor per decoded frame including missing data",
                       "identical descriptors and vector counters across motion modes/trials",
                       "fresh feature storage and descriptor objects across all videos/trials"]}


def timed_trial(encoded, entry, mode, encoder, refresh_every, expected_motion):
    torch.cuda.synchronize(encoder.device)
    start = time.perf_counter()
    result = process_motion_clip(encoded, entry, mode, encoder, refresh_every)
    torch.cuda.synchronize(encoder.device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    validate_motion_counts(result, entry, mode, refresh_every, expected_motion)
    return result, elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--refresh-every", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes) or args.candidates < 2 or args.refresh_every < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("Use unique modes, at least two candidates, positive refresh/repeats and nonnegative warmups")
    if args.device == "cpu" and not args.verify_only:
        parser.error("CPU is supported for correctness only; GPU timings require CUDA")
    manifest = load_manifest()
    inputs = read_inputs(selected_input_paths(args.input, manifest), manifest, args.candidates)
    blocked = args.device == "cuda" and not torch.cuda.is_available()
    device = torch.device("cpu" if blocked or args.device == "cpu" else "cuda:0")
    numerics = configure_numerics(args.seed)
    environment = current_environment()
    output = args.output or RESULT_DIR / ("single_video.csv" if len(inputs) == 1 else "pilot.csv")
    write_json(output.parent / "environment.json", environment)
    write_json(output.parent / "manifest.json", sanitize_public(manifest))
    if blocked:
        print("BLOCKER: CUDA unavailable; CPU correctness only. See environment.json diagnosis.", flush=True)
    elif args.device == "cuda" and not environment["passed"]:
        raise RuntimeError("Current environment/GPU operation failed; see environment.json")
    preparation_start = time.perf_counter()
    encoder = Encoder(device)
    preparation_ms = (time.perf_counter() - preparation_start) * 1000
    with torch.inference_mode():
        verification_start = time.perf_counter()
        verification = verify(inputs, args.modes, encoder, args.refresh_every)
        verification.update({"requested_device": args.device, "cuda_blocked": blocked,
                             "elapsed_ms_excluded": (time.perf_counter() - verification_start) * 1000})
        verification_path = output.with_name(output.stem + "_verification.json")
        write_json(verification_path, verification)
        print(f"Correctness PASS: {len(inputs)} videos, two trials per mode.", flush=True)
        if args.verify_only or blocked:
            return 2 if blocked else 0
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / encoder.weights.url.rsplit("/", 1)[-1]
        metadata = {
            "started_at_utc": now(), "status": "running", "experiment_label": "motion_overhead_2026_10_11",
            "directory_date_is_label_not_measurement_date": True,
            "metric": "Encoded-clip-to-features latency (ms)", "device": str(device),
            "boundary": "encoded bytes in memory -> fresh BytesIO/container -> configure decoder -> full sequential decode/workload -> close container -> CUDA synchronize",
            "inclusions": ["container open/close", "demux/decode", "optional vector export/access/conversion",
                           "CPU descriptors and retention", "required RGB conversion/preprocessing/transfer/encoder", "GPU appearance cache/output retention"],
            "exclusions": ["file loading", "inspection", "model/weight preparation", "verification", "logging/CSV/plotting"],
            "input": relative_path(args.input), "seed": args.seed, "modes": args.modes,
            "candidates": args.candidates, "refresh_every": args.refresh_every,
            "warmup_trials_per_mode": args.warmup, "measured_repetitions": args.repeats,
            "decoder": {"thread_count": DECODER_THREAD_COUNT, "thread_type": DECODER_THREAD_TYPE, "skip_frame": "NONE", "hardware_acceleration": False},
            "motion_export": EXPORT_CONFIG, "motion_descriptors": DESCRIPTOR_CONFIG,
            "numerical_settings": numerics, "environment": environment,
            "model": {"name": "resnet18", "weights": str(encoder.weights), "weights_url": encoder.weights.url,
                      "weights_sha256": sha256_file(checkpoint), "fc": "torch.nn.Identity", "feature_shape": [1, 512],
                      "preprocessing": repr(encoder.preprocess), "preparation_elapsed_ms_excluded": preparation_ms},
            "videos": [{**entry, "file_loading_ms_excluded": ms} for entry, _, ms in inputs],
            "source_code_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
            "verification_file": relative_path(verification_path), "verification_sha256": sha256_file(verification_path),
            "warmup_order": [], "execution_order": [], "correctness_checks_inside_timing": False,
            "control_implementation": "benchmark.process_clip (unchanged)",
            "gpu_telemetry_before": gpu_telemetry(),
        }
        expected_motion = {row["video_id"]: {key: row[key] for key in MOTION_FIELDS}
                           for row in verification["rows"] if row["mode"].endswith("_mv")}
        rows, rng = [], random.Random(args.seed)
        metadata_path = output.with_suffix(".metadata.json")
        try:
            for entry, encoded, _ in inputs:
                expected = expected_motion.get(entry["video_id"])
                for mode in args.modes:
                    for warmup_id in range(1, args.warmup + 1):
                        result, _ = timed_trial(encoded, entry, mode, encoder, args.refresh_every, expected)
                        del result
                        metadata["warmup_order"].append({"video_id": entry["video_id"], "mode": mode, "warmup_id": warmup_id})
                for run_id in range(1, args.repeats + 1):
                    order = list(args.modes)
                    rng.shuffle(order)
                    metadata["execution_order"].append({"video_id": entry["video_id"], "run_id": run_id, "modes": order})
                    for mode in order:
                        result, elapsed_ms = timed_trial(encoded, entry, mode, encoder, args.refresh_every, expected)
                        rows.append({"video_id": entry["video_id"], "mode": mode, "run_id": run_id,
                                     "candidate_count": len(entry["candidate_indices"]), **counters(result), "total_ms": elapsed_ms})
                        del result
                print(f"Measured {entry['video_id']}: {len(args.modes)*args.repeats} rows; counters PASS", flush=True)
            if len(rows) != len(inputs) * len(args.modes) * args.repeats:
                raise AssertionError("Incorrect measured row count")
            write_csv(output, rows, COLUMNS)
            metadata.update({"status": "passed", "completed_at_utc": now(), "measured_rows": len(rows),
                             "csv_sha256": sha256_file(output), "gpu_telemetry_after": gpu_telemetry()})
            write_json(metadata_path, metadata)
        except Exception as exc:
            metadata.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}", "completed_rows_before_failure": len(rows)})
            write_json(metadata_path, metadata)
            raise
    print(f"Saved {len(rows)} measured rows: {relative_path(output)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
