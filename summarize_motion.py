"""Validate six-condition measurements and summarize per-video overhead/savings."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_motion import COLUMNS, MOTION_FIELDS
from motion_utils import MODES
from research_utils import ROOT, now, sha256_file, write_json

METRIC = "Encoded-clip-to-features latency (ms)"
BASES = ["decode", "dense", "periodic"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_trials(frame: pd.DataFrame, metadata: dict, manifest: dict, verification: dict) -> dict:
    if frame.columns.tolist() != COLUMNS or metadata["status"] != "passed" or not metadata["device"].startswith("cuda"):
        raise ValueError("Expected a completed CUDA run with the motion trial schema")
    if set(metadata["modes"]) != set(MODES) or set(frame["mode"]) != set(MODES):
        raise ValueError("All six conditions must be present")
    videos = metadata["videos"]
    ids = [video["video_id"] for video in videos]
    if len(ids) != len(set(ids)) or set(frame.video_id) != set(ids):
        raise ValueError("Video IDs differ, or metadata contains duplicates")
    repeats = metadata["measured_repetitions"]
    expected_keys = set(itertools.product(ids, MODES, range(1, repeats + 1)))
    actual_keys = [(row.video_id, row.mode, row.run_id) for row in frame.itertuples()]
    if len(frame) != len(expected_keys) or metadata["measured_rows"] != len(frame) or set(actual_keys) != expected_keys or len(set(actual_keys)) != len(actual_keys):
        raise ValueError("Missing, extra, or duplicated trial rows/repetition IDs")
    numeric = frame.drop(columns=["video_id", "mode"])
    if frame.isna().any().any() or not np.isfinite(numeric.to_numpy()).all() or not frame.total_ms.gt(0).all():
        raise ValueError("Nonfinite, missing or nonpositive measurements")
    integers = [name for name in numeric.columns if name != "total_ms"]
    if not np.equal(numeric[integers], np.floor(numeric[integers])).all().all() or (numeric[integers] < 0).any().any():
        raise ValueError("Invalid integer trial counters")
    if not verification["passed"] or verification["device"] != metadata["device"] or verification["verification_trials_per_mode"] != 2:
        raise ValueError("Successful two-trial correctness verification is required")
    verification_keys = [(row["video_id"], row["mode"], row["verification_trial"]) for row in verification["rows"]]
    if len(verification_keys) != len(ids)*len(MODES)*2 or set(verification_keys) != set(itertools.product(ids, MODES, (1,2))):
        raise ValueError("Incomplete correctness verification")
    candidates = metadata["candidates"]
    refreshes = len(range(0, candidates, metadata["refresh_every"]))
    appearances = {"decode": (0,0,0), "dense": (candidates,0,candidates),
                   "periodic": (refreshes,candidates-refreshes,candidates)}
    for video in videos:
        inspected = manifest["videos"][video["relative_path"]]
        if video["file_identity"] != inspected["file_identity"] or video["candidate_indices"] != inspected["candidate_indices"]:
            raise ValueError(f"Manifest identity/candidates changed: {video['video_id']}")
        indices = video["candidate_indices"]
        if len(indices) != candidates or sorted(set(indices)) != indices or indices[0] != 0 or indices[-1] != inspected["decoded_frame_count"]-1:
            raise ValueError("Malformed candidate indices")
        for mode in MODES:
            group = frame[(frame.video_id == video["video_id"]) & (frame["mode"] == mode)]
            if not group.decoded_frames.eq(inspected["decoded_frame_count"]).all() or not group.candidate_count.eq(candidates).all():
                raise ValueError(f"Wrong frame/candidate counts: {video['video_id']} {mode}")
            for name, expected in zip(["encoder_calls", "cache_reuses", "feature_outputs"], appearances[mode.removesuffix("_mv")]):
                if not group[name].eq(expected).all():
                    raise ValueError(f"Wrong {name}: {video['video_id']} {mode}")
            expected_motion = verification["motion_by_video"][video["video_id"]]
            for name in MOTION_FIELDS:
                expected = expected_motion[name] if mode.endswith("_mv") else 0
                if not group[name].eq(expected).all():
                    raise ValueError(f"Wrong {name}: {video['video_id']} {mode}")
            if mode.endswith("_mv") and (expected_motion["motion_updates"] != inspected["decoded_frame_count"] or
                    expected_motion["mv_frames_available"] + expected_motion["mv_frames_missing"] != inspected["decoded_frame_count"]):
                raise ValueError("Motion processing missed decoded frames")
    actual_order = [(row.video_id, int(row.run_id), row.mode) for row in frame.itertuples()]
    recorded_order = [(order["video_id"], order["run_id"], mode) for order in metadata["execution_order"] for mode in order["modes"]]
    if actual_order != recorded_order:
        raise ValueError("CSV execution order differs from metadata")
    rng, regenerated_order = random.Random(metadata["seed"]), []
    for video_id in ids:
        for run_id in range(1, repeats+1):
            modes = list(metadata["modes"])
            rng.shuffle(modes)
            regenerated_order.append({"video_id": video_id, "run_id": run_id, "modes": modes})
    if regenerated_order != metadata["execution_order"]:
        raise ValueError("Seed does not reproduce recorded execution order")
    expected_warmups = [{"video_id": video_id, "mode": mode, "warmup_id": warmup_id}
                        for video_id in ids for mode in metadata["modes"]
                        for warmup_id in range(1, metadata["warmup_trials_per_mode"]+1)]
    if expected_warmups != metadata["warmup_order"]:
        raise ValueError("Missing/incorrect warmup records")
    return {"passed": True, "checked_at_utc": now(), "videos": len(ids), "measured_rows": len(frame),
            "warmups_excluded": True, "verification_excluded": True, "all_counters_valid": True,
            "execution_order_matches_and_seed_reproduces": True}


def per_video_summary(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    rows = []
    for video in metadata["videos"]:
        row = {"video_id": video["video_id"], "decoded_frames": video["decoded_frame_count"],
               "file_loading_ms_excluded": video["file_loading_ms_excluded"]}
        for mode in MODES:
            trials = frame.loc[(frame.video_id == video["video_id"]) & (frame["mode"] == mode), "total_ms"]
            row.update({f"{mode}_median_ms": float(trials.median()), f"{mode}_min_ms": float(trials.min()), f"{mode}_max_ms": float(trials.max())})
        for base in BASES:
            delta = row[f"{base}_mv_median_ms"] - row[f"{base}_median_ms"]
            row[f"{base}_mv_configuration_overhead_ms"] = delta
            row[f"{base}_mv_overhead_pct"] = 100*delta/row[f"{base}_median_ms"]
            lo, hi = row[f"{base}_min_ms"], row[f"{base}_max_ms"]
            mv_lo, mv_hi = row[f"{base}_mv_min_ms"], row[f"{base}_mv_max_ms"]
            row[f"{base}_motion_trial_ranges_overlap"] = max(lo,mv_lo) <= min(hi,mv_hi)
            row[f"{base}_overhead_small_vs_larger_trial_range"] = abs(delta) <= max(hi-lo,mv_hi-mv_lo)
        for suffix, label in [("", "control"), ("_mv", "motion")]:
            dense, periodic = row[f"dense{suffix}_median_ms"], row[f"periodic{suffix}_median_ms"]
            row[f"{label}_caching_speedup"] = dense/periodic
            row[f"{label}_latency_reduction_pct"] = 100*(dense-periodic)/dense
            row[f"{label}_median_saving_ms"] = dense-periodic
            row[f"{label}_periodic_all_trials_below_dense_all_trials"] = row[f"periodic{suffix}_max_ms"] < row[f"dense{suffix}_min_ms"]
        row["motion_minus_control_reduction_pp"] = row["motion_latency_reduction_pct"]-row["control_latency_reduction_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_latencies(summary: pd.DataFrame, trials: pd.DataFrame, path: Path) -> None:
    colors = ["#64748b", "#2563eb", "#ea580c", "#94a3b8", "#60a5fa", "#fdba74"]
    fig, ax = plt.subplots(figsize=(13, max(6, len(summary)*1.12+2)))
    ypos, height = np.arange(len(summary)), 0.135
    for position, (mode, color) in enumerate(zip(MODES, colors)):
        y = ypos+(position-2.5)*height
        median = summary[f"{mode}_median_ms"].to_numpy()
        error = np.vstack([median-summary[f"{mode}_min_ms"], summary[f"{mode}_max_ms"]-median])
        ax.barh(y, median, height=height*.88, color=color, label=mode, xerr=error,
                capsize=2, error_kw={"ecolor":"#334155","elinewidth":.85})
        for i, video_id in enumerate(summary.video_id):
            values = trials.loc[(trials.video_id==video_id)&(trials["mode"]==mode),"total_ms"]
            ax.scatter(values, y[i]+np.linspace(-.025,.025,len(values)),s=9,color=color,edgecolor="white",linewidth=.35,zorder=3)
    ax.set_yticks(ypos, summary.video_id)
    ax.invert_yaxis(); ax.set_xlim(left=0)
    ax.set_xlabel(METRIC); ax.set_ylabel("Original validation clip")
    ax.grid(axis="x",color="#e2e8f0"); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Codec motion-vector CPU workload alongside RGB caching",fontsize=16,fontweight="bold",y=.99)
    fig.text(.5,.96,"Bars: per-video medians | Whiskers: trial min-max | Dots: five measured trials",ha="center",color="#475569")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles,labels,ncol=6,loc="upper center",bbox_to_anchor=(.5,.945),frameon=False)
    fig.text(.5,.012,"All six conditions measured in this session. Full sequential decoding; encoded bytes already in memory.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.025,1,.915)); fig.savefig(path,dpi=180); plt.close(fig)


def plot_reductions(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12,max(5.5,len(summary)*.65+1.8)))
    ypos = np.arange(len(summary))
    for offset, suffix, label, color in [(-.17,"","RGB controls","#2563eb"),(.17,"_mv","RGB + CPU motion","#ea580c")]:
        prefix = "motion" if suffix else "control"
        reduction = summary[f"{prefix}_latency_reduction_pct"].to_numpy()
        # Range envelope from independently combining observed extrema, not a CI.
        lower = 100*(1-summary[f"periodic{suffix}_max_ms"].to_numpy()/summary[f"dense{suffix}_min_ms"].to_numpy())
        upper = 100*(1-summary[f"periodic{suffix}_min_ms"].to_numpy()/summary[f"dense{suffix}_max_ms"].to_numpy())
        ax.barh(ypos+offset,reduction,height=.28,color=color,label=label,
                xerr=np.vstack([reduction-lower,upper-reduction]),capsize=3,error_kw={"elinewidth":1,"ecolor":"#334155"})
    ax.set_yticks(ypos,summary.video_id); ax.invert_yaxis(); ax.axvline(0,color="#334155",linewidth=.8)
    ax.set_xlabel("Per-video dense-to-periodic latency reduction (%)")
    ax.grid(axis="x",color="#e2e8f0"); ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,loc="lower right")
    fig.suptitle("How much RGB caching saving remains with motion processing?",fontsize=15,fontweight="bold",y=.99)
    fig.text(.5,.945,"Whiskers: envelope from observed dense/periodic trial extrema; not confidence intervals",ha="center",fontsize=10,color="#475569")
    fig.tight_layout(rect=(0,.01,1,.91)); fig.savefig(path,dpi=180); plt.close(fig)


def findings(summary: pd.DataFrame, metadata: dict, verification: dict, trials: pd.DataFrame) -> str:
    motion = summary.motion_latency_reduction_pct
    controls = summary.control_latency_reduction_pct
    positive = int((motion>0).sum()); separated = int(summary.motion_periodic_all_trials_below_dense_all_trials.sum())
    lines = ["# Motion-vector extraction overhead alongside periodic RGB caching", "",
             f"Actual measurement session: {metadata['started_at_utc']} to {metadata['completed_at_utc']} (UTC). The October 11 directory name is an experiment label, not the collection date.", "",
             f"Validated {len(summary)} original videos and {metadata['measured_rows']} CUDA trials, with two warmups and five measured repetitions per condition. All six controls/workloads were measured in the same session; October 10 numbers were not substituted.", "",
             f"Periodic_mv had a lower median than dense_mv in **{positive}/{len(summary)} videos**. The median of per-video motion-enabled reductions was **{motion.median():.2f}%**, with range **{motion.min():.2f}% to {motion.max():.2f}%**. The median of per-video motion-enabled speedups was **{summary.motion_caching_speedup.median():.3f}x**.", "",
             f"The corresponding same-session RGB-control median per-video reduction was **{controls.median():.2f}%**. The median of per-video motion-minus-control reduction changes was **{summary.motion_minus_control_reduction_pp.median():+.2f} percentage points**, with range {summary.motion_minus_control_reduction_pp.min():+.2f} to {summary.motion_minus_control_reduction_pp.max():+.2f} points. These aggregate statistics weight videos equally; a ratio of aggregate latency medians is a different calculation.", "",
             "## Measured configuration overheads", "",
             "Each difference below is the median latency of an export-and-descriptor configuration minus its corresponding control. It includes decoder interactions, vector access, CPU summaries and descriptor retention, plus scheduling variability. It is not an exact isolated component timing. Negative values are preserved.", "",
             "| Video | decode_mv - decode (ms) | dense_mv - dense (ms) | periodic_mv - periodic (ms) | Control reduction | Motion reduction | Motion speedup | Change (pp) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        lines.append(f"| {row.video_id} | {row.decode_mv_configuration_overhead_ms:+.2f} | {row.dense_mv_configuration_overhead_ms:+.2f} | {row.periodic_mv_configuration_overhead_ms:+.2f} | {row.control_latency_reduction_pct:.2f}% | {row.motion_latency_reduction_pct:.2f}% | {row.motion_caching_speedup:.3f}x | {row.motion_minus_control_reduction_pp:+.2f} |")
    lines += ["", "Across videos, the median configuration overheads were:", ""]
    for base in BASES:
        delta = summary[f"{base}_mv_configuration_overhead_ms"]
        percentage = summary[f"{base}_mv_overhead_pct"]
        lines.append(f"- {base}: {delta.median():+.2f} ms, range {delta.min():+.2f} to {delta.max():+.2f} ms; median per-video relative difference {percentage.median():+.2f}%.")
    lines += ["", "## Trial variability and consistency", "",
              f"Observed periodic_mv and dense_mv ranges were fully separated, with all periodic trials faster, in {separated}/{len(summary)} videos. Five repeats do not establish statistical significance.", "",
              "The table below flags a configuration difference as small relative to variability when its absolute median difference is no larger than the larger of the two observed latency ranges. Range overlap is reported independently; this descriptive rule is not a statistical test.", "",
              "| Video | Small vs trial range: decode / dense / periodic | Ranges overlap: decode / dense / periodic |",
              "| --- | --- | --- |"]
    for _, row in summary.iterrows():
        small = " / ".join("yes" if row[f"{base}_overhead_small_vs_larger_trial_range"] else "no" for base in BASES)
        overlap = " / ".join("yes" if row[f"{base}_motion_trial_ranges_overlap"] else "no" for base in BASES)
        lines.append(f"| {row.video_id} | {small} | {overlap} |")
    paired = trials.pivot(index=["video_id", "run_id"], columns="mode", values="total_ms")
    slower = paired[paired.periodic_mv > paired.dense_mv]
    lines += ["", f"In **{len(slower)}/{len(paired)} same-repetition comparisons**, periodic_mv was slower than dense_mv despite the lower per-video medians. Conditions within a shuffled repetition run sequentially, not simultaneously; this is a descriptive variability check, not a causal/statistical paired test.", ""]
    if len(slower):
        lines += ["| Video | Repetition | Dense_mv (ms) | Periodic_mv (ms) | Repetition reduction |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for (video_id, run_id), row in slower.iterrows():
            reduction = 100*(row.dense_mv-row.periodic_mv)/row.dense_mv
            lines.append(f"| {video_id} | {run_id} | {row.dense_mv:.2f} | {row.periodic_mv:.2f} | {reduction:.2f}% |")
    total_frames = sum(v["decoded_frame_count"] for v in metadata["videos"])
    motion_counts = verification["motion_by_video"]
    available = sum(v["mv_frames_available"] for v in motion_counts.values())
    vectors = sum(v["total_vectors"] for v in motion_counts.values())
    past = sum(v["past_reference_vectors"] for v in motion_counts.values())
    future = sum(v["future_reference_vectors"] for v in motion_counts.values())
    other = sum(v["other_reference_vectors"] for v in motion_counts.values())
    lines += ["", "## Export and descriptor correctness", "",
              f"A single pass over the selected videos decoded {total_frames} frames: {available} with vectors and {total_frames-available} without. It exported {vectors} entries: {past} past-reference, {future} future-reference and {other} other entries. Counts and descriptors matched exactly across all three motion modes and both verification trials; appearance features matched the original dense control at refreshed positions (rtol=1e-5, atol=1e-6).", "",
              "Every decoded frame gets a fresh retained CPU descriptor, including missing-vector frames. Missing data uses zero descriptors with mv_available=false; observed zero displacement retains mv_available=true. dx and dy are computed after float64 conversion; magnitudes use numpy.hypot, an unweighted mean, and numpy.percentile(..., 90, method='linear'). Invalid scales/fields stop processing explicitly.", "",
              "## Interpretation and next experiment", ""]
    worse = int((summary.motion_minus_control_reduction_pp < 0).sum())
    improved = summary.loc[summary.motion_minus_control_reduction_pp > 0, "video_id"].tolist()
    overhead_values = summary[[f"{base}_mv_configuration_overhead_ms" for base in BASES]].to_numpy()
    change = summary.motion_minus_control_reduction_pp.median()
    lines += [f"Median per-video relative configuration differences were {summary.decode_mv_overhead_pct.median():+.2f}% of decode, {summary.dense_mv_overhead_pct.median():+.2f}% of dense, and {summary.periodic_mv_overhead_pct.median():+.2f}% of periodic control latency. Of {overhead_values.size} measured median configuration differences, {int((overhead_values>0).sum())} were positive and {int((overhead_values<0).sum())} were negative. Negative differences are retained.", "",
              f"The percentage caching saving decreased in {worse}/{len(summary)} videos. The median paired change was {change:+.2f} percentage points; it is calculated from each video's change, not by subtracting the two aggregate medians.", ""]
    if improved:
        lines += [f"The saving increased for {', '.join(improved)}; this mixed direction is reported rather than forced into the expected trend. Interpret that exception with its observed trial ranges. Motion configuration overhead was small relative to the larger trial range in {int(summary.dense_overhead_small_vs_larger_trial_range.sum())}/{len(summary)} dense comparisons and {int(summary.periodic_overhead_small_vs_larger_trial_range.sum())}/{len(summary)} periodic comparisons.", ""]
    if positive == len(summary):
        lines.append("RGB caching still saves processing time in every video after adding the CPU motion workload. Assess motion usefulness and recognition accuracy next, while profiling vector export and percentile computation to understand the added cost. Use longer runs and control laptop power/background load before attributing small changes to a component.")
    elif positive:
        lines.append("Motion-enabled caching savings are inconsistent. Profile decoder/export, vector conversion, percentile computation and retention overhead before adding a learned motion branch, and repeat longer runs to separate variability from implementation cost.")
    else:
        lines.append("Motion-enabled periodic caching did not improve median latency in this pilot. Inspect overhead and variability before adding a learned motion branch or interpreting an accuracy/latency tradeoff.")
    lines += ["", "## Scope and limits", "",
              "- These are simple CPU codec displacement descriptors, not optical flow, physical velocity, measured object motion, or an evaluated motion-recognition branch. No motion encoder or descriptor GPU transfer is included.",
              "- The original encodings contain B-frames; exported vectors can reference future frames. This is offline full-clip overhead measurement and does not establish strictly causal online inference. Preserve these encodings; test controlled I/P-only encoding separately.",
              "- ResNet18 remains an appearance-feature timing prototype. Recognition accuracy, stale-feature effects and motion usefulness for adaptive refresh decisions remain untested; there is no action head or learned refresh controller.",
              "- File loading and diagnostic inspection are measured separately and excluded. All modes decode the full clip anew. Decode-only is a separate baseline, not an exact additive component.",
              "- These low-resolution clips do not establish 1080p performance. Ten videos, five repeats and one laptop are preliminary; offline clip latency is not live-stream response latency.",
              "- GPU telemetry before and after the measurement session is in metadata. Concurrent graphics activity and power/scheduling state can affect observed trials.", "",
              "![Six-condition per-video latency](latency_comparison.png)", "",
              "![Caching reductions with and without motion](caching_reduction_comparison.png)", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = read_json(args.input.with_suffix(".metadata.json"))
    manifest = read_json(args.input.parent/"manifest.json")
    verification_path = args.input.parent/Path(metadata["verification_file"]).name
    verification = read_json(verification_path)
    if sha256_file(args.input) != metadata["csv_sha256"] or sha256_file(verification_path) != metadata["verification_sha256"]:
        raise ValueError("Raw measurements or verification bytes changed")
    for name, expected in metadata["source_code_sha256"].items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError(f"Measurement source differs from recorded hash: {name}")
    frame = pd.read_csv(args.input)
    validation = validate_trials(frame,metadata,manifest,verification)
    summary = per_video_summary(frame,metadata)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    summary.to_csv(args.output,index=False)
    plot_latencies(summary,frame,args.output.parent/"latency_comparison.png")
    plot_reductions(summary,args.output.parent/"caching_reduction_comparison.png")
    (args.output.parent/"findings.md").write_text(findings(summary,metadata,verification,frame),encoding="utf-8")
    validation.update({"raw_csv_sha256": sha256_file(args.input), "measurement_source_hashes_match": True,
                       "analysis_source_sha256": sha256_file(Path(__file__)),
                       "median_per_video_motion_reduction_pct": float(summary.motion_latency_reduction_pct.median()),
                       "median_per_video_control_reduction_pct": float(summary.control_latency_reduction_pct.median()),
                       "median_per_video_motion_speedup": float(summary.motion_caching_speedup.median()),
                       "median_per_video_reduction_change_pp": float(summary.motion_minus_control_reduction_pp.median())})
    write_json(args.output.parent/"validation.json",validation)
    print(f"Validated {len(frame)} measured rows, {len(summary)} videos; counters, order, hashes PASS.")
    print(summary[["video_id","decode_mv_configuration_overhead_ms","dense_mv_configuration_overhead_ms",
                   "periodic_mv_configuration_overhead_ms","control_latency_reduction_pct","motion_latency_reduction_pct","motion_caching_speedup"]].to_string(index=False))
    print(f"Median per-video motion-enabled reduction: {summary.motion_latency_reduction_pct.median():.2f}%; speedup: {summary.motion_caching_speedup.median():.3f}x")


if __name__ == "__main__":
    main()
