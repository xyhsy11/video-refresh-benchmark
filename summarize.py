"""Validate actual trials, summarize each video, plot variability, and write findings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_utils import ROOT, load_manifest, now, sha256_file, write_json

MODES = ["decode", "dense", "periodic"]
METRIC = "Encoded-clip-to-features latency (ms)"


def validate_trials(frame: pd.DataFrame, metadata: dict, manifest: dict) -> dict:
    columns = ["video_id", "mode", "run_id", "decoded_frames", "candidate_count", "encoder_calls", "cache_reuses", "feature_outputs", "total_ms"]
    if frame.columns.tolist() != columns:
        raise ValueError("Trial CSV columns do not match the benchmark schema")
    if metadata["status"] != "passed" or not metadata["device"].startswith("cuda"):
        raise ValueError("Input is not a successfully completed CUDA measurement")
    if set(metadata["modes"]) != set(MODES) or set(frame["mode"]) != set(MODES):
        raise ValueError("This comparison requires decode, dense, and periodic")
    videos = {v["video_id"]: v for v in metadata["videos"]}
    expected_rows = len(videos) * 3 * metadata["measured_repetitions"]
    if len(frame) != expected_rows or set(frame.video_id) != set(videos) or frame.duplicated(["video_id", "mode", "run_id"]).any():
        raise ValueError("Missing, extra, or duplicated measured trials")
    if frame.isna().any().any() or not np.isfinite(frame.total_ms).all() or not frame.total_ms.gt(0).all():
        raise ValueError("Invalid measured values")
    candidate_count = metadata["candidates"]
    calls = len(range(0, candidate_count, metadata["refresh_every"]))
    expected_counters = {"decode": (0, 0, 0), "dense": (candidate_count, 0, candidate_count), "periodic": (calls, candidate_count - calls, candidate_count)}
    for (video_id, mode), group in frame.groupby(["video_id", "mode"]):
        source = videos[video_id]
        inspected = manifest["videos"][source["relative_path"]]
        if source["file_identity"] != inspected["file_identity"] or source["candidate_indices"] != inspected["candidate_indices"]:
            raise ValueError(f"Manifest differs from measured input: {video_id}")
        if group.run_id.sort_values().tolist() != list(range(1, metadata["measured_repetitions"] + 1)):
            raise ValueError(f"Wrong repeat IDs: {video_id} {mode}")
        if not group.decoded_frames.eq(inspected["decoded_frame_count"]).all() or not group.candidate_count.eq(candidate_count).all():
            raise ValueError(f"Wrong frames or candidates: {video_id} {mode}")
        for column, expected in zip(["encoder_calls", "cache_reuses", "feature_outputs"], expected_counters[mode]):
            if not group[column].eq(expected).all():
                raise ValueError(f"Incorrect {column}: {video_id} {mode}")
    actual_order = [(row.video_id, int(row.run_id), row.mode) for row in frame.itertuples()]
    recorded_order = [(entry["video_id"], entry["run_id"], mode) for entry in metadata["execution_order"] for mode in entry["modes"]]
    if actual_order != recorded_order:
        raise ValueError("CSV row order differs from the recorded execution order")
    warmups = pd.DataFrame(metadata["warmup_order"])
    if metadata["warmup_trials_per_mode"] and not warmups.groupby(["video_id", "mode"]).size().eq(metadata["warmup_trials_per_mode"]).all():
        raise ValueError("Incorrect warmup counts")
    return {"passed": True, "measured_rows": len(frame), "videos": len(videos), "all_frame_counts_and_counters_valid": True,
            "execution_order_matches": True, "expected_rows": expected_rows}


def per_video_summary(frame: pd.DataFrame, metadata: dict, manifest: dict) -> pd.DataFrame:
    rows = []
    for measured in metadata["videos"]:
        video_id = measured["video_id"]
        inspected = manifest["videos"][measured["relative_path"]]
        row = {"video_id": video_id, "decoded_frames": inspected["decoded_frame_count"],
               "duration_seconds": inspected["duration_seconds"], "file_loading_ms_excluded": measured["file_loading_ms_excluded"]}
        for mode in MODES:
            trials = frame.loc[(frame.video_id == video_id) & (frame["mode"] == mode), "total_ms"]
            row.update({f"{mode}_median_ms": float(trials.median()), f"{mode}_min_ms": float(trials.min()), f"{mode}_max_ms": float(trials.max())})
        row["speedup"] = row["dense_median_ms"] / row["periodic_median_ms"]
        row["latency_reduction_pct"] = 100 * (row["dense_median_ms"] - row["periodic_median_ms"]) / row["dense_median_ms"]
        row["median_saving_ms"] = row["dense_median_ms"] - row["periodic_median_ms"]
        row["dense_range_ms"] = row["dense_max_ms"] - row["dense_min_ms"]
        row["periodic_range_ms"] = row["periodic_max_ms"] - row["periodic_min_ms"]
        row["periodic_all_trials_below_dense_all_trials"] = row["periodic_max_ms"] < row["dense_min_ms"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, frame: pd.DataFrame, path: Path) -> None:
    colors = {"decode": "#64748b", "dense": "#2563eb", "periodic": "#ea580c"}
    fig, ax = plt.subplots(figsize=(12.5, max(5.5, 0.77 * len(summary) + 1.5)))
    ypos = np.arange(len(summary))
    height = 0.22
    for position, mode in enumerate(MODES):
        median = summary[f"{mode}_median_ms"].to_numpy()
        lower = median - summary[f"{mode}_min_ms"].to_numpy()
        upper = summary[f"{mode}_max_ms"].to_numpy() - median
        y = ypos + (position - 1) * height
        label = {"decode": "Decode only (no features)", "dense": "Dense (16 encoder calls)", "periodic": "Periodic (4 encoder calls)"}[mode]
        ax.barh(y, median, height=height * 0.88, color=colors[mode], alpha=0.8, label=label,
                xerr=np.vstack([lower, upper]), capsize=3, error_kw={"ecolor": "#111827", "elinewidth": 1.0, "capthick": 1.0})
        for i, video_id in enumerate(summary.video_id):
            values = frame.loc[(frame.video_id == video_id) & (frame["mode"] == mode), "total_ms"].to_numpy()
            offsets = np.linspace(-height * 0.25, height * 0.25, len(values))
            ax.scatter(values, y[i] + offsets, s=13, color=colors[mode], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yticks(ypos, summary.video_id)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel(METRIC, fontsize=11)
    ax.set_ylabel("Validation clip")
    fig.suptitle("Refreshing one-quarter of RGB feature candidates", fontsize=16, x=0.14, y=0.988, ha="left", fontweight="bold")
    fig.text(0.14, 0.948, "Bars: per-video medians  |  Whiskers: trial min–max  |  Dots: individual measured trials", fontsize=10, color="#475569")
    ax.xaxis.grid(True, color="#e2e8f0")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 0.93), ncol=3, frameon=False, fontsize=9)
    fig.text(0.12, 0.014, "Encoded bytes were already in memory. Complete clips were decoded anew in every trial. Warmups are excluded.", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.035, 1, 0.90))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def write_findings(summary: pd.DataFrame, aggregates: dict, metadata: dict, path: Path, single_path: Path | None = None) -> None:
    count = len(summary)
    positive = aggregates["videos_with_positive_latency_reduction"]
    median_reduction = aggregates["median_of_per_video_latency_reductions_pct"]
    median_speedup = aggregates["median_of_per_video_speedups"]
    separated = aggregates["videos_with_periodic_max_below_dense_min"]
    if positive == count and median_reduction >= 5:
        answer = "Refreshing one-quarter of the candidates reduced median latency consistently across all selected videos."
        next_step = "Add motion-vector extraction and measure its overhead using the same encoded-byte boundary, fresh full-clip decoding, fixed settings, and shuffled mode order. Compare appearance-only periodic processing with appearance plus motion extraction."
    elif median_reduction <= 0:
        answer = "Periodic processing did not reduce median latency across this pilot."
        next_step = "Inspect implementation overhead and repeat variability, including CPU/GPU scheduling, cache handling, and transfers, before adding a motion branch."
    else:
        answer = "Reducing encoder calls produced small or inconsistent savings across this pilot."
        next_step = "Profile decoding, RGB conversion, preprocessing, CPU-to-GPU transfers, and encoder execution. Use profiler experiments outside the end-to-end timing run to identify the bottleneck."
    lines = [
        "# Video latency pilot findings", "",
        answer, "",
        f"Encoder calls fell from **{metadata['candidates']} to {len(range(0, metadata['candidates'], metadata['refresh_every']))}**, while dense and periodic both produced {metadata['candidates']} feature outputs and decoded every frame. The **median of the {count} per-video latency reductions was {median_reduction:.2f}%**, and the **median of the per-video speedups was {median_speedup:.3f}×**. Positive reductions occurred in **{positive}/{count} videos**. These are equal-weight aggregates of statistics calculated separately for each video, not ratios of pooled mode medians.", "",
        f"Per-video reductions ranged from **{summary.latency_reduction_pct.min():.2f}% to {summary.latency_reduction_pct.max():.2f}%**; speedups ranged from **{summary.speedup.min():.3f}× to {summary.speedup.max():.3f}×**. Negative reductions, if present, are preserved.", "",
        "## Actual per-video measurements", "",
        "Latencies below are medians, with minimum–maximum measured trial latency in parentheses, in milliseconds.", "",
        "| Video | Decode only | Dense | Periodic | Speedup | Reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        cells = [f"{getattr(row, mode + '_median_ms'):.2f} ({getattr(row, mode + '_min_ms'):.2f}–{getattr(row, mode + '_max_ms'):.2f})" for mode in MODES]
        lines.append(f"| {row.video_id} | {' | '.join(cells)} | {row.speedup:.3f}× | {row.latency_reduction_pct:.2f}% |")
    medians = aggregates["median_of_per_video_median_latencies_ms"]
    decode_fraction = summary.decode_median_ms / summary.dense_median_ms
    lines.extend([
        "", "## Decode baseline and trial variability", "",
        f"The median across per-video median latencies was {medians['decode']:.2f} ms for decode only, {medians['dense']:.2f} ms for dense, and {medians['periodic']:.2f} ms for periodic. Decode-only median latency was {100 * decode_fraction.median():.1f}% of dense latency as a descriptive median of per-video ratios. Decode still processes the full clip, so a 75% reduction in encoder calls need not yield a 75% reduction in total latency.", "",
        "Decode-only is a separate baseline that emits no appearance features. It is not an exact additive component: CPU/GPU work, transfers, scheduling, and power state can interact. Its timing was not subtracted from dense or periodic to estimate isolated encoder time.", "",
        f"For **{separated}/{count} videos**, the slowest measured periodic trial was faster than the fastest measured dense trial. The median per-video saving was {summary.median_saving_ms.median():.2f} ms; the median within-video min–max range was {summary.dense_range_ms.median():.2f} ms for dense and {summary.periodic_range_ms.median():.2f} ms for periodic. These observed ranges describe whether the difference is large relative to the five-trial variability; they are not confidence intervals or a statistical significance test.", "",
        "## Reproducibility and verification", "",
        f"Metric: **{METRIC}**. The measurement starts after encoded bytes are in memory, before a fresh BytesIO/container opens, and stops after container closure and CUDA synchronization. It includes full sequential decoding, required RGB conversion/preprocessing/transfers, encoder calls, and output/cache handling. File loading, inspection, model/weight preparation, correctness checks, and result writing are excluded.", "",
        f"All {aggregates['validation']['measured_rows']} measured rows passed frame-count and encoder/cache/output counter validation. Each mode had {metadata['warmup_trials_per_mode']} full warmups and {metadata['measured_repetitions']} measured trials per video. Warmups are excluded from CSV rows. Mode order was shuffled per measured repetition with seed {metadata['seed']}, with the actual order saved in the metadata sidecar. Verification exercised two fresh trials per mode and checked cross-trial and cross-video cache isolation.", "",
        f"The model was pretrained ResNet18 ({metadata['model']['weights']}) with the classification layer replaced by Identity, producing [1,512] float32 features. It used eval and inference mode on {metadata['gpu_name']}. Documented preprocessing: [torchvision ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). Numerical settings were identical across modes; TF32 and autocast were disabled and deterministic algorithms enabled. Software decoding used {metadata['decoder']['thread_count']} thread, thread type {metadata['decoder']['thread_type']}; PyTorch CPU and interop thread counts were 1.", "",
        f"File-loading time is separately recorded for every video in the metadata and summary; observed values here ranged from {summary.file_loading_ms_excluded.min():.3f} to {summary.file_loading_ms_excluded.max():.3f} ms. Input SHA-256 hashes, candidate indices, weight hash, environment, and source-code hashes are recorded. Reproducible inputs/settings/order do not guarantee identical wall-clock latency on a busy laptop.", "",
        "## Next experiment", "", next_step, "",
        "If counters become incorrect, fix correctness before interpreting performance. If later periodic runs are slower, inspect implementation overhead and variability; if fewer calls produce little saving, profile decoding, conversion, preprocessing, and transfers.", "",
        "## Limits", "",
        "- ResNet18 is an appearance-feature timing prototype.",
        "- This experiment does not measure action-recognition accuracy; reusing a feature may change its usefulness at later positions.",
        "- It does not yet include motion-feature extraction, a motion branch, a learned refresh controller, or an action-classification head.",
        "- File loading is measured separately and excluded from this metric.",
        "- These low-resolution clips do not establish performance for 1080p video.",
        "- Offline clip processing time is not a measurement of live-stream response latency.", "",
        "Artifacts: measured trial CSV and its `.metadata.json` sidecar, `summary.csv`, `aggregate_summary.json`, `latency_comparison.png`, `manifest.json`, and `environment.json`.",
    ])
    overlapping = summary.loc[~summary.periodic_all_trials_below_dense_all_trials, "video_id"].tolist()
    if overlapping:
        lines.extend(["", f"Observed dense/periodic trial ranges overlapped for: {', '.join(overlapping)}. Interpret those clip-level differences with the visible variability."])
    single_path = single_path or ROOT / "results" / "single_video.csv"
    if single_path.exists():
        single = pd.read_csv(single_path)
        if single.video_id.nunique() == 1 and set(single['mode']) == set(MODES):
            video_id = single.video_id.iloc[0]
            pilot_row = summary.loc[summary.video_id == video_id]
            if not pilot_row.empty:
                values = single.groupby('mode').total_ms.median()
                reduction = 100 * (values['dense'] - values['periodic']) / values['dense']
                lines.extend(["", f"The earlier single-video timing check for {video_id} showed {reduction:.2f}% reduction ({values['dense']:.2f} → {values['periodic']:.2f} ms), versus {pilot_row.latency_reduction_pct.iloc[0]:.2f}% in the ten-video run. The CSVs retain both actual runs. This change is evidence that wall-clock results vary between runs; these measurements alone do not identify the cause."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Inspection manifest; defaults to one beside the input CSV, then results/manifest.json")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    metadata = json.loads(args.input.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if sha256_file(args.input) != metadata["csv_sha256"]:
        raise ValueError("CSV changed since measurement")
    manifest_path = args.manifest or args.input.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else load_manifest()
    if args.manifest and not args.manifest.exists():
        raise ValueError(f"Manifest does not exist: {args.manifest}")
    validation = validate_trials(frame, metadata, manifest)
    summary = per_video_summary(frame, metadata, manifest)
    aggregates = {
        "generated_at_utc": now(), "source_csv": str(args.input), "source_csv_sha256": sha256_file(args.input),
        "validation": validation, "aggregation": "Equal weight per video; speedups and reductions calculated before aggregation",
        "median_of_per_video_speedups": float(summary.speedup.median()),
        "mean_of_per_video_speedups": float(summary.speedup.mean()),
        "median_of_per_video_latency_reductions_pct": float(summary.latency_reduction_pct.median()),
        "mean_of_per_video_latency_reductions_pct": float(summary.latency_reduction_pct.mean()),
        "min_per_video_latency_reduction_pct": float(summary.latency_reduction_pct.min()),
        "max_per_video_latency_reduction_pct": float(summary.latency_reduction_pct.max()),
        "median_of_per_video_median_latencies_ms": {mode: float(summary[f"{mode}_median_ms"].median()) for mode in MODES},
        "videos_with_positive_latency_reduction": int(summary.latency_reduction_pct.gt(0).sum()),
        "videos_with_periodic_max_below_dense_min": int(summary.periodic_all_trials_below_dense_all_trials.sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    write_json(args.output.parent / "aggregate_summary.json", aggregates)
    plot_path = args.output.parent / "latency_comparison.png"
    plot(summary, frame, plot_path)
    findings_path = args.output.parent / "findings.md"
    adjacent_single = args.input.parent / "single_video.csv"
    write_findings(summary, aggregates, metadata, findings_path, adjacent_single if adjacent_single.exists() else None)
    print(f"Validated {len(frame)} actual measured rows for {len(summary)} videos; every counter PASS.")
    print(summary[["video_id", "decode_median_ms", "dense_median_ms", "periodic_median_ms", "speedup", "latency_reduction_pct"]].to_string(index=False))
    print(f"Median per-video speedup: {aggregates['median_of_per_video_speedups']:.3f}x; median per-video latency reduction: {aggregates['median_of_per_video_latency_reductions_pct']:.2f}%")
    print(f"Saved: {args.output}\nSaved: {plot_path}\nSaved: {findings_path}")


if __name__ == "__main__":
    main()
