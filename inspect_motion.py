"""Sequential per-frame motion-vector diagnostics; never a benchmark timing."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import io
import time

import av

from benchmark import read_inputs
from motion_utils import (DESCRIPTOR_CONFIG, EXPORT_CONFIG, configure_motion_decoder,
                          frame_descriptor, library_info, write_csv)
from research_utils import (DECODER_THREAD_COUNT, DECODER_THREAD_TYPE, load_manifest,
                            now, relative_path, selected_input_paths, write_json)

COLUMNS = ["video_id", "decoded_frame_index", "pts", "timestamp_seconds", "frame_type",
           "vector_count", "mean_magnitude_px", "p90_magnitude_px", "mv_available",
           "past_reference_vectors", "future_reference_vectors", "other_reference_vectors"]


def main() -> None:
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest()
    inputs = read_inputs(selected_input_paths(args.input, manifest), manifest, 16)
    rows, summaries = [], []
    for entry, encoded, loading_ms in inputs:
        start = time.perf_counter()
        descriptors, types = [], Counter()
        with av.open(io.BytesIO(encoded)) as container:
            stream = container.streams.video[0]
            configure_motion_decoder(stream)
            for index, frame in enumerate(container.decode(stream)):
                descriptor = frame_descriptor(frame, entry["video_id"], index)
                descriptors.append(descriptor)
                kind = av.video.frame.PictureType(frame.pict_type).name
                types[kind] += 1
                timestamp = float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base is not None else None
                rows.append({"video_id": entry["video_id"], "decoded_frame_index": index,
                             "pts": frame.pts, "timestamp_seconds": timestamp, "frame_type": kind,
                             **asdict(descriptor)})
            flags2 = int(stream.codec_context.flags2)
        inspection_ms = (time.perf_counter() - start) * 1000
        if len(descriptors) != entry["decoded_frame_count"]:
            raise AssertionError(f"{entry['video_id']}: decoded frame count differs from manifest")
        total = sum(d.vector_count for d in descriptors)
        if not total:
            raise RuntimeError(f"{entry['video_id']}: no vectors; investigate export/decoder support before benchmarking")
        available = sum(d.mv_available for d in descriptors)
        summary = {"video_id": entry["video_id"], "relative_path": entry["relative_path"],
                   "file_identity": entry["file_identity"], "decoded_frames": len(descriptors),
                   "frame_type_counts": dict(types), "mv_frames_available": available,
                   "mv_frames_missing": len(descriptors) - available, "total_vectors": total,
                   "past_reference_vectors": sum(d.past_reference_vectors for d in descriptors),
                   "future_reference_vectors": sum(d.future_reference_vectors for d in descriptors),
                   "other_reference_vectors": sum(d.other_reference_vectors for d in descriptors),
                   "inspection_elapsed_ms_excluded": inspection_ms, "file_loading_ms_excluded": loading_ms,
                   "applied_flags2": flags2, "invalid_vectors": 0, "frame_count_matches_manifest": True}
        summaries.append(summary)
        print(f"{entry['video_id']}: frames={len(descriptors)}, available={available}, missing={len(descriptors)-available}, vectors={total}; manifest PASS", flush=True)
    writing_start = time.perf_counter()
    write_csv(args.output, rows, COLUMNS)
    write_json(args.output.with_suffix(".summary.json"), {
        "checked_at_utc": now(), "input": relative_path(args.input), "passed": True,
        "export_configuration": EXPORT_CONFIG, "descriptors": DESCRIPTOR_CONFIG,
        "decoder": {"thread_count": DECODER_THREAD_COUNT, "thread_type": DECODER_THREAD_TYPE, "skip_frame": "NONE"},
        "libraries": library_info(), "videos": summaries, "diagnostic_rows": len(rows),
        "csv_writing_ms_excluded": (time.perf_counter() - writing_start) * 1000})


if __name__ == "__main__":
    main()
