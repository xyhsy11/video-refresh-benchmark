"""Codec-vector export and CPU descriptors shared by inspection and timing."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib.metadata
from pathlib import Path, PureWindowsPath, PurePosixPath
import re
import subprocess
import tempfile

import av
import numpy as np

from research_utils import ROOT, configure_decoder

RESULT_DIR = ROOT / "results" / "motion_overhead_2026_10_11"
MODES = ["decode", "dense", "periodic", "decode_mv", "dense_mv", "periodic_mv"]
EXPORT_CONFIG = {
    "api": "stream.codec_context.options before the first decode",
    "options": {"flags2": "+export_mvs"},
    "side_data": "frame.side_data.get('MOTION_VECTORS').to_ndarray()",
    "software_decoder": True,
}
DESCRIPTOR_CONFIG = {
    "displacement": "dx=float64(motion_x)/float64(motion_scale); dy likewise",
    "magnitude": "numpy.hypot(dx, dy)", "units": "pixels of codec displacement",
    "mean": "unweighted mean over all exported vector entries",
    "p90": "numpy.percentile(magnitudes, 90, method='linear')",
    "missing": "count=0, mean=0, p90=0, mv_available=false",
    "reference_direction": "source<0: past; source>0: future; source==0: other",
    "retention": "one fresh frozen slotted MotionDescriptor per decoded frame in a CPU list until timer stops",
    "invalid_data": "raise ValueError with video/frame context; do not discard or invent vectors",
    "scope": "no optical flow, physical velocity, learned motion encoder, or descriptor GPU transfer",
}


@dataclass(frozen=True, slots=True)
class MotionDescriptor:
    vector_count: int = 0
    mean_magnitude_px: float = 0.0
    p90_magnitude_px: float = 0.0
    mv_available: bool = False
    past_reference_vectors: int = 0
    future_reference_vectors: int = 0
    other_reference_vectors: int = 0


def configure_motion_decoder(stream) -> None:
    configure_decoder(stream)
    if stream.codec_context.is_open:
        raise ValueError("Motion export must be configured before opening the decoder")
    stream.codec_context.options = {**stream.codec_context.options, **EXPORT_CONFIG["options"]}


def descriptor_from_vectors(vectors: np.ndarray | None) -> MotionDescriptor:
    """No raw arrays escape this function. All arithmetic uses float64."""
    if vectors is None:
        return MotionDescriptor()
    required = ("motion_x", "motion_y", "motion_scale", "source")
    if vectors.ndim != 1 or not vectors.dtype.names or any(name not in vectors.dtype.names for name in required):
        raise ValueError("Malformed motion-vector structured array or missing required fields")
    if any(not np.issubdtype(vectors[name].dtype, np.integer) for name in required):
        raise ValueError("Malformed motion-vector fields: integer fields required")
    count = len(vectors)
    if not count:
        return MotionDescriptor()
    scale = vectors["motion_scale"].astype(np.float64)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid motion_scale: every exported entry must have a finite positive scale")
    dx = vectors["motion_x"].astype(np.float64) / scale
    dy = vectors["motion_y"].astype(np.float64) / scale
    magnitude = np.hypot(dx, dy)
    if not np.isfinite(magnitude).all():
        raise ValueError("Nonfinite codec displacement")
    source = vectors["source"]
    past = int(np.count_nonzero(source < 0))
    future = int(np.count_nonzero(source > 0))
    return MotionDescriptor(count, float(magnitude.mean()),
                            float(np.percentile(magnitude, 90, method="linear")),
                            True, past, future, count - past - future)


def frame_descriptor(frame, video_id: str, index: int) -> MotionDescriptor:
    try:
        side = frame.side_data.get("MOTION_VECTORS")
        return descriptor_from_vectors(None if side is None else side.to_ndarray())
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{video_id} decoded frame {index}: {exc}") from exc


def library_info() -> dict:
    return {"pyav": av.__version__, "ffmpeg_release": av.ffmpeg_version_info,
            "ffmpeg_libraries": {name: list(version) for name, version in av.library_versions.items()}}


def sanitize_public(value):
    """Keep useful environment information, omit installation/user/host details."""
    if isinstance(value, dict):
        omitted = {"base_prefix", "hostname", "host_name", "username", "user_name", "computer_name"}
        return {key: sanitize_public(item) for key, item in value.items() if key not in omitted}
    if isinstance(value, list):
        return [sanitize_public(item) for item in value]
    if isinstance(value, str) and (PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()):
        try:
            return Path(value).resolve().relative_to(ROOT).as_posix() or "."
        except ValueError:
            return "[local installation path omitted]"
    return value


def current_environment() -> dict:
    from check_env import collect_report
    report = sanitize_public(collect_report())
    report["installed_distributions"] = dict(sorted(
        (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()))
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        support = re.search(r"(?:CUDA UMD Version|CUDA Version):\s*([0-9.]+)", smi.stdout)
    except OSError:
        support = None
    report["nvidia_driver_cuda_support"] = support.group(1) if support else None
    # CPU/GPU/package checks in collect_report happen before model construction/timing.
    return report


def gpu_telemetry() -> dict:
    from research_utils import now
    fields = "name,driver_version,temperature.gpu,pstate,power.draw,power.limit,utilization.gpu,memory.used"
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=False)
    return {"captured_at_utc": now(), "fields": fields.split(","), "values": result.stdout.strip(),
            "units": "C, W, percent, MiB for respective telemetry fields", "returncode": result.returncode}


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", newline="", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
