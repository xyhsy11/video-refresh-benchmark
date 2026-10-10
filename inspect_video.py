"""Decode complete clips, count real frames, and save uniform frame candidates."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

import av

from research_utils import (DECODER_THREAD_COUNT, DECODER_THREAD_TYPE, MANIFEST,
                            configure_decoder, file_identity, load_manifest, now,
                            relative_path, selected_input_paths, write_json)


def candidate_indices(frame_count: int, count: int) -> list[int]:
    if count < 2 or frame_count < count:
        raise ValueError(f"Need at least {count} decoded frames; found {frame_count}")
    # Nearest integer to uniformly spaced positions; endpoints are exact.
    indices = [(ordinal * (frame_count - 1) + (count - 1) // 2) // (count - 1) for ordinal in range(count)]
    if len(set(indices)) != count or indices[0] != 0 or indices[-1] != frame_count - 1:
        raise ValueError("Candidate construction failed")
    return indices


def inspect(path: Path, count: int) -> dict:
    identity = file_identity(path)
    frame_types = Counter({"I": 0, "P": 0, "B": 0})
    decoded = 0
    first_time = None
    last_time = None
    last_duration = None
    start = time.perf_counter()
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"No video stream: {path}")
        stream = container.streams.video[0]
        configure_decoder(stream)
        rate = float(stream.average_rate) if stream.average_rate else None
        stream_duration = float(stream.duration * stream.time_base) if stream.duration is not None and stream.time_base else None
        container_duration = container.duration / av.time_base if container.duration is not None else None
        metadata = {
            "codec": stream.codec_context.name,
            "resolution": {"width": stream.codec_context.width, "height": stream.codec_context.height},
            "reported_frame_rate": {"rational": str(stream.average_rate), "fps": rate},
            "reported_metadata_frame_count": stream.frames,
            "stream_index": stream.index,
            "stream_time_base": str(stream.time_base),
            "stream_duration_seconds": stream_duration,
            "container_duration_seconds": container_duration,
        }
        for frame in container.decode(stream):
            decoded += 1
            try:
                name = av.video.frame.PictureType(frame.pict_type).name
            except (ValueError, AttributeError):
                name = f"UNKNOWN_{frame.pict_type}"
            frame_types[name] += 1
            if frame.pts is not None and frame.time_base is not None:
                frame_time = float(frame.pts * frame.time_base)
                if first_time is None:
                    first_time = frame_time
                last_time = frame_time
                last_duration = float(frame.duration * frame.time_base) if frame.duration else (1 / rate if rate else None)
        # No decoded frames are retained beyond this loop.
    elapsed_ms = (time.perf_counter() - start) * 1000
    decoded_duration = last_time - first_time + last_duration if first_time is not None and last_time is not None and last_duration is not None else None
    if decoded_duration is not None:
        duration, method = decoded_duration, "decoded last PTS - first PTS + final frame duration (reported FPS fallback if final duration missing)"
    elif stream_duration is not None:
        duration, method = stream_duration, "stream.duration * stream.time_base"
    elif container_duration is not None:
        duration, method = container_duration, "container.duration / av.time_base"
    elif rate:
        duration, method = decoded / rate, "decoded frame count / reported frame rate"
    else:
        raise ValueError(f"Cannot determine duration: {path}")
    if not 5 <= duration <= 15:
        raise ValueError(f"Clip outside experiment's 5–15 second range: {path.name}: {duration:.6f}s")
    return {
        "video_id": path.stem, "relative_path": relative_path(path),
        "file_identity": identity, "inspected_at_utc": now(), **metadata,
        "duration_seconds": duration, "duration_method": method,
        "decoded_pts_duration_seconds": decoded_duration,
        "decoded_frame_count": decoded, "frame_type_counts": dict(frame_types),
        "candidate_count": count, "candidate_indices": candidate_indices(decoded, count),
        "candidate_index_basis": "zero-based sequential decoded-frame indices",
        "decoder": {"thread_count": DECODER_THREAD_COUNT, "thread_type": DECODER_THREAD_TYPE, "skip_frame": "NONE", "hardware_acceleration": False},
        "inspection_elapsed_ms": elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    for path in selected_input_paths(args.input, manifest):
        key = relative_path(path)
        existing = manifest["videos"].get(key)
        if args.skip_existing and existing and existing["file_identity"] == file_identity(path) and existing["candidate_count"] == args.candidates:
            print(f"Already inspected, unchanged: {key}")
            continue
        entry = inspect(path, args.candidates)
        manifest["videos"][key] = entry
        write_json(MANIFEST, manifest)
        size = entry["resolution"]
        print(f"{entry['video_id']} | {key}")
        print(f"  {entry['codec']} | {size['width']}x{size['height']} | reported FPS {entry['reported_frame_rate']['rational']} | duration {entry['duration_seconds']:.6f}s")
        print(f"  Duration method: {entry['duration_method']}")
        print(f"  Decoded frames: {entry['decoded_frame_count']} | frame types: {entry['frame_type_counts']} | inspection {entry['inspection_elapsed_ms']:.3f}ms")
        print(f"  {len(entry['candidate_indices'])} distinct candidates: {entry['candidate_indices']}")


if __name__ == "__main__":
    main()
