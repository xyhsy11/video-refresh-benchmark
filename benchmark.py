"""Sequential encoded-clip-to-features benchmark; correctness is outside timing."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

# Set before CUDA initialization, for deterministic float32 matrix operations.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import av
from PIL import Image
import torch
from torchvision.models import ResNet18_Weights, resnet18

from research_utils import (ROOT, DECODER_THREAD_COUNT, DECODER_THREAD_TYPE,
                            configure_decoder, load_manifest, now, relative_path,
                            selected_input_paths, sha256_file, write_json)

COLUMNS = ["video_id", "mode", "run_id", "decoded_frames", "candidate_count",
           "encoder_calls", "cache_reuses", "feature_outputs", "total_ms"]


@dataclass
class TrialResult:
    decoded_frames: int = 0
    encoder_calls: int = 0
    cache_reuses: int = 0
    features: list = field(default_factory=list)
    refreshed_ordinals: list[int] = field(default_factory=list)


class Encoder:
    def __init__(self, device: torch.device):
        self.device = device
        self.weights = ResNet18_Weights.DEFAULT
        torch.hub.set_dir(str(ROOT / "data" / "source" / "torch_hub"))
        self.preprocess = self.weights.transforms()
        self.model = resnet18(weights=self.weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device=device, dtype=torch.float32)

    def __call__(self, frame):
        # These operations occur only at refreshes, identically in both modes.
        rgb = frame.to_ndarray(format="rgb24")
        image = Image.fromarray(rgb)
        tensor = self.preprocess(image).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=torch.float32, non_blocking=True)
        return self.model(tensor)


def process_clip(encoded_bytes: bytes, entry: dict, mode: str,
                 encoder: Encoder, refresh_every: int) -> TrialResult:
    """Shared untimed/timed logic. No correctness checks or synchronization here."""
    result = TrialResult()
    candidates = entry["candidate_indices"]
    ordinal = 0
    cached = None  # Local to this invocation: no state survives a clip or trial.
    with av.open(io.BytesIO(encoded_bytes), mode="r") as container:
        stream = container.streams.video[0]
        configure_decoder(stream)
        for index, frame in enumerate(container.decode(stream)):
            result.decoded_frames += 1
            if mode == "decode":
                continue
            if ordinal < len(candidates) and index == candidates[ordinal]:
                if mode == "dense" or ordinal % refresh_every == 0:
                    cached = encoder(frame)
                    result.encoder_calls += 1
                    result.refreshed_ordinals.append(ordinal)
                else:
                    result.cache_reuses += 1
                result.features.append(cached)
                ordinal += 1
    return result


def validate_counts(result: TrialResult, entry: dict, mode: str, refresh_every: int) -> None:
    count = len(entry["candidate_indices"])
    refreshes = list(range(0, count, refresh_every))
    expected = {
        "decode": (0, 0, 0, []),
        "dense": (count, 0, count, list(range(count))),
        "periodic": (len(refreshes), count - len(refreshes), count, refreshes),
    }[mode]
    actual = (result.encoder_calls, result.cache_reuses, len(result.features), result.refreshed_ordinals)
    if result.decoded_frames != entry["decoded_frame_count"] or actual != expected:
        raise AssertionError(f"{entry['video_id']} {mode}: frames={result.decoded_frames} expected={entry['decoded_frame_count']}; counters={actual} expected={expected}")


def validate_features(result: TrialResult, mode: str, refresh_every: int, device: torch.device) -> None:
    for ordinal, feature in enumerate(result.features):
        if feature.shape != (1, 512) or feature.dtype != torch.float32 or feature.device != device:
            raise AssertionError(f"Invalid feature at candidate {ordinal}: shape/dtype/device")
        if not torch.isfinite(feature).all().item():
            raise AssertionError(f"Nonfinite feature at candidate {ordinal}")
        if mode == "periodic" and ordinal % refresh_every:
            refreshed = result.features[(ordinal // refresh_every) * refresh_every]
            if feature is not refreshed or not torch.equal(feature, refreshed):
                raise AssertionError("A reused feature differs from its most recent refresh")


def verify(inputs: list, modes: list[str], encoder: Encoder, refresh_every: int) -> dict:
    rows = []
    # Hold prior outputs while verifying, so allocator reuse cannot hide a stale cache.
    prior_first_features = []
    for entry, encoded, _load_ms in inputs:
        results = {}
        for mode in modes:
            for trial in range(2):
                result = process_clip(encoded, entry, mode, encoder, refresh_every)
                validate_counts(result, entry, mode, refresh_every)
                validate_features(result, mode, refresh_every, encoder.device)
                if result.features:
                    first = result.features[0]
                    if 0 not in result.refreshed_ordinals:
                        raise AssertionError("First candidate did not refresh")
                    if any(first is old or first.data_ptr() == old.data_ptr() for old in prior_first_features):
                        raise AssertionError("First feature/cache survived across videos or trials")
                    prior_first_features.append(first)
                results[(mode, trial)] = result
                rows.append({"video_id": entry["video_id"], "mode": mode, "verification_trial": trial + 1,
                             "decoded_frames": result.decoded_frames, "encoder_calls": result.encoder_calls,
                             "cache_reuses": result.cache_reuses, "feature_outputs": len(result.features), "passed": True})
            first_trial = results[(mode, 0)]
            print(f"{entry['video_id']:12s} | {mode:8s} | {first_trial.decoded_frames:5d} | {first_trial.encoder_calls:2d} | {first_trial.cache_reuses:2d} | {len(first_trial.features):2d}", flush=True)
        if "dense" in modes and "periodic" in modes:
            for ordinal in range(0, len(entry["candidate_indices"]), refresh_every):
                if not torch.allclose(results[("dense", 0)].features[ordinal], results[("periodic", 0)].features[ordinal], rtol=1e-5, atol=1e-6):
                    raise AssertionError("Refreshed periodic features disagree with dense features")
    return {"passed": True, "checked_at_utc": now(), "device": str(encoder.device),
            "verification_trials_per_mode": 2, "rows": rows,
            "checks": ["decoded frame counts", "encoder calls", "cache reuses", "feature output counts",
                       "shape [1,512]", "float32 and correct device", "finite values", "cache equality",
                       "candidate zero refresh", "fresh outputs across videos/trials", "dense vs refreshed periodic features"]}


def configure_numerics(seed: int) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "dtype": "float32", "batch_size": 1, "eval": True, "inference_mode": True,
        "autocast": False, "compiled": False, "deterministic_algorithms": True,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "cudnn_allow_tf32": False, "matmul_allow_tf32": False,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_cpu_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "transfer_non_blocking": True, "input_cpu_memory_pinned": False,
        "note": "Pageable CPU transfers; non_blocking does not promise transfer/compute overlap.",
    }


def read_inputs(paths: list[Path], manifest: dict, candidate_count: int) -> list:
    inputs = []
    for path in paths:
        key = relative_path(path)
        if key not in manifest["videos"]:
            raise ValueError(f"No inspection for {key}; run inspect_video.py first")
        entry = manifest["videos"][key]
        indices = entry["candidate_indices"]
        if len(indices) != candidate_count or len(set(indices)) != candidate_count or indices != sorted(indices):
            raise ValueError(f"Invalid or wrong candidate count: {key}")
        if indices[0] != 0 or indices[-1] != entry["decoded_frame_count"] - 1:
            raise ValueError(f"Missing candidate endpoints: {key}")
        if not 5 <= entry["duration_seconds"] <= 15:
            raise ValueError(f"Duration outside 5–15s: {key}")
        start = time.perf_counter()
        encoded = path.read_bytes()
        loading_ms = (time.perf_counter() - start) * 1000
        identity = entry["file_identity"]
        if len(encoded) != identity["size_bytes"] or hashlib.sha256(encoded).hexdigest() != identity["sha256"] or path.stat().st_mtime_ns != identity["mtime_ns"]:
            raise ValueError(f"Input changed since inspection: {key}; inspect again")
        inputs.append((entry, encoded, loading_ms))
    ids = [entry["video_id"] for entry, _, _ in inputs]
    if len(ids) != len(set(ids)):
        raise ValueError("Video IDs must be unique for this CSV")
    return inputs


def timed_trial(encoded: bytes, entry: dict, mode: str, encoder: Encoder, refresh_every: int):
    torch.cuda.synchronize(encoder.device)
    start = time.perf_counter()
    result = process_clip(encoded, entry, mode, encoder, refresh_every)
    torch.cuda.synchronize(encoder.device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    # Validation uses counts only and occurs after the stop time.
    validate_counts(result, entry, mode, refresh_every)
    return result, elapsed_ms


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", newline="", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=["decode", "dense", "periodic"], default=["decode", "dense", "periodic"])
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--refresh-every", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes) or args.candidates < 2 or args.refresh_every < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("Use unique modes, at least two candidates, positive refresh/repeats, and nonnegative warmup")
    if args.device == "cpu" and not args.verify_only:
        parser.error("CPU is supported only for correctness checks; no CPU latency substitution is permitted")
    manifest = load_manifest()
    paths = selected_input_paths(args.input, manifest)
    inputs = read_inputs(paths, manifest, args.candidates)
    blocked = args.device == "cuda" and not torch.cuda.is_available()
    if blocked:
        from check_env import collect_report
        write_json(ROOT / "results" / "environment.json", collect_report())
        print("BLOCKER: CUDA unavailable. Running CPU correctness only; no GPU timing CSV will be written.", flush=True)
    device = torch.device("cpu" if blocked or args.device == "cpu" else "cuda:0")
    numerical = configure_numerics(args.seed)
    prepare_start = time.perf_counter()
    encoder = Encoder(device)
    preparation_ms = (time.perf_counter() - prepare_start) * 1000
    output = args.output or ROOT / "results" / "benchmark.csv"
    verification_path = output.with_name(output.stem + "_verification.json") if args.output else ROOT / "results" / ("single_video_verification.json" if len(inputs) == 1 else "pilot_verification.json")
    print("video_id     | mode     | decoded_frames | encoder_calls | cache_reuses | feature_outputs", flush=True)
    with torch.inference_mode():
        verification = verify(inputs, args.modes, encoder, args.refresh_every)
        verification.update({"requested_device": args.device, "cuda_blocked": blocked})
        write_json(verification_path, verification)
        print(f"Correctness PASS on {len(inputs)} video(s), two fresh trials per mode. Saved {verification_path}", flush=True)
        if args.verify_only or blocked:
            return 2 if blocked else 0
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / encoder.weights.url.rsplit("/", 1)[-1]
        environment_path = ROOT / "results" / "environment.json"
        metadata = {
            "started_at_utc": now(), "status": "running", "metric": "Encoded-clip-to-features latency (ms)",
            "boundary": "encoded bytes already in memory -> fresh BytesIO/container open -> complete sequential decode/processing -> container close -> CUDA synchronization",
            "input": relative_path(args.input), "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
            "seed": args.seed, "modes": args.modes, "candidates": args.candidates, "refresh_every": args.refresh_every,
            "warmup_trials_per_mode": args.warmup, "measured_repetitions": args.repeats,
            "decoder": {"thread_count": DECODER_THREAD_COUNT, "thread_type": DECODER_THREAD_TYPE, "skip_frame": "NONE", "hardware_acceleration": False},
            "numerical_settings": numerical,
            "model": {"name": "resnet18", "weights": str(encoder.weights), "weights_url": encoder.weights.url,
                      "weights_sha256": sha256_file(checkpoint), "fc": "torch.nn.Identity", "feature_shape": [1, 512],
                      "preprocessing": repr(encoder.preprocess), "preparation_elapsed_ms_excluded": preparation_ms},
            "environment": json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else None,
            "videos": [{"video_id": entry["video_id"], "relative_path": entry["relative_path"], "file_identity": entry["file_identity"],
                        "candidate_indices": entry["candidate_indices"], "decoded_frame_count": entry["decoded_frame_count"],
                        "file_loading_ms_excluded": loading_ms} for entry, _, loading_ms in inputs],
            "source_code_sha256": {name: sha256_file(ROOT / name) for name in ["benchmark.py", "research_utils.py", "inspect_video.py", "requirements.txt"]},
            "warmup_order": [], "execution_order": [], "verification_file": relative_path(verification_path),
            "correctness_checks_inside_timing": False, "features_retained_on_gpu_until_timer_stops": True,
        }
        metadata_path = output.with_suffix(".metadata.json")
        rows = []
        rng = random.Random(args.seed)
        try:
            for entry, encoded, _loading_ms in inputs:
                for mode in args.modes:
                    for warmup_id in range(1, args.warmup + 1):
                        result, _ = timed_trial(encoded, entry, mode, encoder, args.refresh_every)
                        del result
                        metadata["warmup_order"].append({"video_id": entry["video_id"], "mode": mode, "warmup_id": warmup_id})
                for run_id in range(1, args.repeats + 1):
                    order = list(args.modes)
                    rng.shuffle(order)
                    metadata["execution_order"].append({"video_id": entry["video_id"], "run_id": run_id, "modes": order})
                    for mode in order:
                        result, elapsed_ms = timed_trial(encoded, entry, mode, encoder, args.refresh_every)
                        rows.append({"video_id": entry["video_id"], "mode": mode, "run_id": run_id,
                                     "decoded_frames": result.decoded_frames, "candidate_count": len(entry["candidate_indices"]),
                                     "encoder_calls": result.encoder_calls, "cache_reuses": result.cache_reuses,
                                     "feature_outputs": len(result.features), "total_ms": elapsed_ms})
                        del result
                print(f"Measured {entry['video_id']}: {len(args.modes) * args.repeats} rows; all counters PASS", flush=True)
            expected_rows = len(inputs) * len(args.modes) * args.repeats
            if len(rows) != expected_rows:
                raise AssertionError(f"Wrong measured CSV row count: {len(rows)} != {expected_rows}")
            write_csv(output, rows)
            metadata.update({"status": "passed", "completed_at_utc": now(), "measured_rows": len(rows), "csv_sha256": sha256_file(output)})
            write_json(metadata_path, metadata)
        except Exception as exc:
            metadata.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}", "completed_rows_before_failure": len(rows)})
            write_json(metadata_path, metadata)
            raise
    print(f"Saved {len(rows)} measured rows: {output}\nSettings and execution order: {metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
