"""Fixed train/validation/test drift models and paired whole-video uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_cache_drift import (PREDICTORS, TARGET, fit_predict, focused_checks, metrics_for_videos)
from cache_drift_v2_utils import (COMPARISONS, PROTOCOL, SEED, freeze_protocol, read_json,
                                 validate_split, verify_preservation)
from collect_cache_drift import (AGE_INPUTS, MOTION_INPUTS, OBSERVATION_COLUMNS,
                                fresh_directory, observations_for_video)
from motion_utils import write_csv
from research_utils import ROOT, now, relative_path, sha256_file, write_json


def apply_fitted(frame: pd.DataFrame, state: dict) -> np.ndarray:
    # Read predictor columns only: held-out targets cannot affect predictions.
    columns = state["columns"]
    if not columns:
        return np.full(len(frame), state["intercept"], dtype=np.float64)
    x = frame[columns].to_numpy(dtype=np.float64)
    return state["intercept"] + ((x-np.array(state["means"]))/np.array(state["scales"])) @ np.array(state["coefficients"])


def audit_collection(directory: Path, check_files: bool = False) -> tuple[pd.DataFrame, dict, dict]:
    metadata = read_json(directory / "collection_metadata.json")
    if metadata["status"] != "collected" or metadata["device"] != "cuda:0":
        raise ValueError(f"Incomplete real CUDA dataset: {metadata.get('status')}; {metadata.get('blocker')}")
    manifest = read_json(directory / "split_manifest.json")
    assert manifest["status"] == "frozen" and not manifest["feature_eligibility_pending"]
    assert sha256_file(directory / "split_manifest.json") == metadata["split_manifest_sha256"]
    split_validation = validate_split(manifest, check_files=check_files)
    assert read_json(directory / "protocol.json")["protocol"] == PROTOCOL
    assert sha256_file(directory / "protocol.json") == metadata["protocol_sha256"]
    verify_preservation(metadata["protected_original_sha256"])
    for name, digest in metadata["source_code_sha256"].items():
        assert sha256_file(ROOT / name) == digest, f"Collection source changed: {name}"
    frame = pd.read_csv(directory / "observations.csv")
    assert sha256_file(directory / "observations.csv") == metadata["observations_sha256"]
    assert list(frame.columns) == OBSERVATION_COLUMNS+["label", "experiment_split"]
    assert not frame.observation_id.duplicated().any() and len(frame) == 3000
    progress = read_json(directory / "collection_progress.json")
    entries = {e["video_id"]: e for e in manifest["videos"]}
    audits = []
    for video in progress["videos"]:
        entry = video["entry"]
        video_id = entry["video_id"]
        assert entry == entries[video_id]
        for name, digest in video["artifact_sha256"].items():
            assert sha256_file(directory/name) == digest
        side = read_json(directory / "features" / f"{video_id}.json")
        with np.load(directory / "features" / f"{video_id}.npz", allow_pickle=False) as saved:
            features, indices, stamps = saved["features"], saved["frame_indices"], saved["timestamps_seconds"]
        assert features.shape == (16, 512) and features.dtype == np.float32 and np.isfinite(features).all()
        assert indices.tolist() == entry["candidate_indices"]
        np.testing.assert_array_equal(stamps, [c["timestamp_seconds"] for c in side["candidates"]])
        assert np.isfinite(stamps).all() and (np.diff(stamps) > 0).all()
        motion = pd.read_csv(directory / "motion_frames" / f"{video_id}.csv")
        assert len(motion) == entry["decoded_frame_count"]
        assert motion.decoded_frame_index.tolist() == list(range(len(motion)))
        assert (motion.mv_available == (motion.vector_count > 0)).all()
        assert (motion.past_reference_vectors+motion.future_reference_vectors+motion.other_reference_vectors == motion.vector_count).all()
        missing = motion.loc[~motion.mv_available]
        assert (missing[["mean_magnitude_px", "p90_magnitude_px"]] == 0).all().all()
        assert np.isfinite(motion[["mean_magnitude_px", "p90_magnitude_px"]]).all().all()
        observed = frame.loc[frame.video_id == video_id].reset_index(drop=True)
        assert len(observed) == 15 and observed.cached_candidate_ordinal.tolist() == [0,0,0,0,4,4,4,4,8,8,8,8,12,12,12]
        rows = observations_for_video(entry, features, side["candidates"], motion.to_dict("records"), 4)
        rebuilt = pd.DataFrame([{**r, "label": entry["label"], "experiment_split": entry["experiment_split"]} for r in rows])[OBSERVATION_COLUMNS+["label", "experiment_split"]]
        pd.testing.assert_frame_equal(observed, rebuilt, check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-12)
        for row in observed.to_dict("records"):
            t, r = row["candidate_ordinal"], row["cached_candidate_ordinal"]
            a, b = features[t].astype(float), features[r].astype(float)
            independent = 1-np.clip(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)), -1, 1)
            assert abs(independent-row[TARGET]) < 1e-12
            interval = motion.loc[(motion.decoded_frame_index > indices[r]) & (motion.decoded_frame_index <= indices[t])]
            assert len(interval) == row["interval_frame_count"] and row["interval_end_frame"] == indices[t]
            available = interval.loc[interval.mv_available]
            assert len(available) == row["interval_available_frame_count"]
            if len(available):
                assert abs(available.mean_magnitude_px.mean()-row["interval_mean_available_magnitude_px"]) < 1e-10
                assert abs(available.p90_magnitude_px.max()-row["interval_max_available_p90_px"]) < 1e-10
            else:
                assert row["interval_mean_available_magnitude_px"] == row["interval_max_available_p90_px"] == row["interval_mv_available_fraction"] == 0
        audits.append({"video_id": video_id, "feature_count": 16, "observation_count": 15,
                       "independent_labels_intervals_and_missing_vector_checks": True})
    assert len(entries) == 200 and len(audits) == 200
    assert frame.groupby("experiment_split").size().to_dict() == {"train": 1800, "validation": 600, "test": 600}
    assert np.isfinite(frame[AGE_INPUTS+MOTION_INPUTS+[TARGET]]).all().all()
    return frame, metadata, {"passed": True, "input_video_hashes_checked_locally": check_files, "split_validation": split_validation, "videos": audits,
                             "originals_unchanged": True, "all_feature_and_artifact_hashes": True}


def fit_models(train: pd.DataFrame) -> dict:
    states = {}
    for name, columns in PREDICTORS.items():
        _, state = fit_predict(train, train.iloc[:1], columns)
        assert state["training_rows"] == len(train) and abs(state["intercept"]-float(train[TARGET].mean())) < 1e-12
        if columns:
            np.testing.assert_allclose(state["means"], train[columns].mean().to_numpy(), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(state["raw_standard_deviations"], train[columns].std(ddof=0).to_numpy(), rtol=1e-12, atol=1e-12)
        states[name] = state
    return states


def predict(frame: pd.DataFrame, models: dict) -> pd.DataFrame:
    rows = []
    for name, state in models.items():
        raw = apply_fitted(frame, state)
        changed = frame.copy()
        changed[TARGET] = 1e9
        np.testing.assert_array_equal(raw, apply_fitted(changed, state))
        assert np.isfinite(raw).all()
        clipped = np.clip(raw, 0, 2)
        for i, (_, row) in enumerate(frame.iterrows()):
            rows.append({"observation_id": row.observation_id, "video_id": row.video_id, "source_video_id": row.source_video_id,
                         "label": row.label, "experiment_split": row.experiment_split,
                         "candidate_ordinal": int(row.candidate_ordinal), "cached_candidate_ordinal": int(row.cached_candidate_ordinal),
                         "predictor": name, "observed_drift": row[TARGET], "prediction_raw": float(raw[i]), "prediction_clipped": float(clipped[i])})
    result = pd.DataFrame(rows)
    assert len(result) == len(frame)*4 and not result.duplicated(["observation_id", "predictor"]).any()
    assert (result.groupby("observation_id").size() == 4).all()
    assert set(result.observation_id) == set(frame.observation_id)
    return result


def aggregate(metrics: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    result = {}
    for name, rows in metrics.groupby("predictor"):
        result[name] = {"video_count": len(rows)}
        for metric in ["mae", "rmse", "spearman"]:
            valid = rows[metric].dropna()
            result[name][metric] = {"mean_per_video": float(valid.mean()) if len(valid) else None,
                                    "median_per_video": float(valid.median()) if len(valid) else None,
                                    "defined": len(valid), "undefined": len(rows)-len(valid)}
        result[name]["undefined_correlation_reasons"] = rows.loc[rows.spearman.isna(), "spearman_note"].value_counts().to_dict()
        part = predictions.loc[predictions.predictor == name]
        result[name]["clipped_prediction_count"] = int((part.prediction_raw != part.prediction_clipped).sum())
    return result


def paired_comparisons(metrics: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    pivot = metrics.pivot(index="video_id", columns="predictor", values="mae").sort_index()
    source = metrics.groupby("video_id").source_video_id.first()
    assert source.nunique() == len(source), "Bootstrap requires distinct source video groups"
    rows, intervals = [], []
    for candidate, reference in COMPARISONS:
        difference = pivot[reference]-pivot[candidate]
        percent = 100*difference/pivot[reference].where(pivot[reference] != 0, np.nan)
        outcomes = np.where(difference > 1e-12, "improved", np.where(difference < -1e-12, "worsened", "tied"))
        for video_id, delta, pct, outcome in zip(pivot.index, difference, percent, outcomes):
            rows.append({"video_id": video_id, "source_video_id": source[video_id], "candidate": candidate, "reference": reference,
                         "mae_candidate": float(pivot.loc[video_id, candidate]), "mae_reference": float(pivot.loc[video_id, reference]),
                         "absolute_mae_improvement": float(delta), "percentage_mae_improvement": float(pct) if pd.notna(pct) else None,
                         "percentage_note": "undefined: reference MAE zero" if pd.isna(pct) else "defined", "outcome": outcome})
        rng = np.random.default_rng(SEED)
        indices = rng.integers(0, len(pivot), size=(10000, len(pivot)))
        means = difference.to_numpy()[indices].mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5], method="linear")
        net = float(difference.sum())
        positives = difference[difference > 1e-12].sort_values(ascending=False)
        top_two = positives.head(2)
        intervals.append({"candidate": candidate, "reference": reference,
                          "mean_paired_absolute_mae_improvement": float(difference.mean()),
                          "median_paired_absolute_mae_improvement": float(difference.median()),
                          "mean_per_video_percentage_improvement": float(percent.mean()) if percent.notna().any() else None,
                          "median_per_video_percentage_improvement": float(percent.median()) if percent.notna().any() else None,
                          "undefined_percentage_count": int(percent.isna().sum()),
                          "improved": int(np.count_nonzero(outcomes == "improved")), "tied": int(np.count_nonzero(outcomes == "tied")),
                          "worsened": int(np.count_nonzero(outcomes == "worsened")), "ci95_low": float(low), "ci95_high": float(high),
                          "bootstrap_resamples": 10000, "bootstrap_seed": SEED, "bootstrap_unit": "whole source video",
                          "ci_interpretation": "uncertainty over held-out videos conditional on fixed fitted models",
                          "conclusion": "improvement" if low > 0 else "worse" if high < 0 else "inconclusive",
                          "net_sum_improvement": net, "top_two_gain_video_ids": top_two.index.tolist(),
                          "top_two_positive_gains_pct_of_net": 100*float(top_two.sum())/net if net > 1e-12 else None,
                          "concentration_denominator": "net summed paired MAE improvement; undefined if <=1e-12; can exceed 100% when losses offset gains"})
    return pd.DataFrame(rows), intervals


def plot_results(metrics: pd.DataFrame, pairs: pd.DataFrame, predictions: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    pivot = metrics.pivot(index="video_id", columns="predictor", values="mae").sort_values("age_only")
    ids = pivot.index.tolist()
    colors = {"constant": "#888888", "age_only": "#2878b5", "motion_only": "#2ca02c", "age_plus_motion": "#f28e2b"}
    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, name in enumerate(PREDICTORS):
        ax.plot(x, pivot[name], marker="o", ms=3, lw=1, label=name.replace("_", " "), color=colors[name])
    ax.set_xticks(x, [v[:11] for v in ids], rotation=90, fontsize=7)
    ax.set_ylabel("Test-video MAE of appearance-feature drift")
    ax.set_title("Fixed predictors on 40 previously unseen source videos; lower is better")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "test_mae.png", dpi=170)
    plt.close(fig)
    primary = pairs.loc[(pairs.candidate == "age_plus_motion") & (pairs.reference == "age_only")].sort_values("absolute_mae_improvement")
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.bar(np.arange(len(primary)), primary.absolute_mae_improvement,
           color=np.where(primary.absolute_mae_improvement >= 0, "#2ca02c", "#c43d3d"))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(primary)), primary.video_id.str[:11], rotation=90, fontsize=7)
    ax.set_ylabel("MAE(age only) - MAE(age plus motion)")
    ax.set_title("Per-video paired test improvement; positive values favor adding motion")
    fig.tight_layout()
    fig.savefig(output / "test_paired_improvements.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    limit = max(predictions.observed_drift.max(), predictions.prediction_clipped.max())*1.05
    for ax, name in zip(axes.flat, PREDICTORS):
        part = predictions.loc[predictions.predictor == name]
        ax.scatter(part.observed_drift, part.prediction_clipped, s=12, alpha=0.4, color=colors[name])
        ax.plot([0, limit], [0, limit], ls="--", color="gray", lw=1)
        ax.set_title(name.replace("_", " "))
        ax.set_xlim(-0.01, limit)
        ax.set_ylim(-0.01, limit)
        ax.grid(alpha=0.2)
    fig.supxlabel("Observed appearance-feature drift")
    fig.supylabel("Test predicted drift, clipped to [0,2]")
    fig.suptitle("600 correlated positions from 40 unseen videos; fixed training models")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.96))
    fig.savefig(output / "test_predicted_vs_observed.png", dpi=170)
    plt.close(fig)


def findings(aggregates: dict, intervals: list[dict], metadata: dict, validation_candidate: str) -> str:
    test = aggregates["test"]
    primary = intervals[0]
    def value(v, digits=6):
        return "undefined" if v is None else f"{v:.{digits}f}"
    lines = ["# cache_drift_v2: previously unseen-video prediction", "",
        f"Real CUDA collection: {metadata['started_at_utc']} through {metadata['completed_at_utc']}. "
        "200 new Kinetics-400 source videos from 20 classes, with 120/40/40 custom training/validation/test videos. "
        "3,200 reference features and 3,000 pre-decision drift labels; 600 test positions within 40 independent source videos. Original ten pilot sources are excluded.", "",
        "The sampling plan, eligibility, split, alpha=1.0, input definitions, comparisons, and bootstrap protocol were fixed before test evaluation. "
        "All four models and training-only scalers are fitted on the same 1,800 training observations; validation never changes parameters. "
        f"The predefined lowest-validation-MAE rule nominates **{validation_candidate}** for subsequent research, before inspecting test results. "
        "Primary metrics use [0,2]-clipped predictions; raw predictions and clipping counts are saved.", "",
        "## Test prediction quality", "",
        "Smaller MAE/RMSE indicates more accurate drift magnitude prediction. Spearman measures whether larger predicted drift corresponds to larger observed drift within a video. "
        "Undefined constant-target/prediction correlations are missing, not zero; Spearman aggregates include only defined per-video values.", "",
        "| Predictor | Mean MAE | Median MAE | Mean RMSE | Median RMSE | Mean Spearman | Median Spearman | Undefined rho | Clipped predictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in PREDICTORS:
        a = test[name]
        lines.append(f"| {name} | {value(a['mae']['mean_per_video'])} | {value(a['mae']['median_per_video'])} | {value(a['rmse']['mean_per_video'])} | {value(a['rmse']['median_per_video'])} | {value(a['spearman']['mean_per_video'],4)} | {value(a['spearman']['median_per_video'],4)} | {a['spearman']['undefined']} | {a['clipped_prediction_count']} |")
    lines += ["", "## Predefined paired comparisons", "",
        "Positive differences are MAE(reference) minus MAE(candidate), calculated per video before averaging. "
        "95% percentile intervals use 10,000 whole-source-video paired resamples with seed 20261013 and fixed fitted models. "
        "They describe uncertainty over held-out videos conditional on these models, not uncertainty from training-set or class selection. Candidate positions are not independently resampled.", "",
        "| Candidate versus reference | Mean absolute improvement | 95% bootstrap interval | Mean per-video % improvement | Improved / tied / worsened | Interpretation |",
        "| --- | ---: | --- | ---: | --- | --- |"]
    for a in intervals:
        lines.append(f"| {a['candidate']} vs {a['reference']} | {a['mean_paired_absolute_mae_improvement']:+.6f} | [{a['ci95_low']:+.6f}, {a['ci95_high']:+.6f}] | {value(a['mean_per_video_percentage_improvement'],2)}% | {a['improved']} / {a['tied']} / {a['worsened']} | {a['conclusion']} |")
    lines += ["", f"Adding motion beyond age is **{primary['conclusion']}** under the primary comparison: "
        f"{primary['improved']}/40 videos improve and {primary['worsened']}/40 worsen. "
        f"Median paired absolute improvement is {primary['median_paired_absolute_mae_improvement']:+.6f}; "
        f"median per-video percentage improvement is {value(primary['median_per_video_percentage_improvement'],2)}%.", ""]
    concentration = primary["top_two_positive_gains_pct_of_net"]
    if concentration is None:
        lines.append("Primary-comparison gain concentration is undefined because the net summed improvement is nonpositive or numerically negligible. Negative results are preserved.")
    else:
        lines.append(f"The largest two positive gains ({', '.join(primary['top_two_gain_video_ids'])}) account for {concentration:.2f}% of net summed improvement. "
                     "This descriptive net-gain percentage can exceed 100% if losses offset gains. The per-video plot shows whether gains are broadly distributed.")
    for a in intervals[1:]:
        lines.append(f"{a['candidate']} compared with {a['reference']}: **{a['conclusion']}**, with {a['improved']}/40 improved videos and conditional interval [{a['ci95_low']:+.6f}, {a['ci95_high']:+.6f}].")
    comparison = intervals[-1]["conclusion"]
    lines.append("Motion-only is stronger than age plus motion by this fixed-model paired MAE comparison." if comparison == "worse"
                 else "Age plus motion is stronger than motion-only by this fixed-model paired MAE comparison." if comparison == "improvement"
                 else "The paired MAE difference between motion-only and age plus motion is inconclusive.")
    age_rho, both_rho = test["age_only"]["spearman"]["mean_per_video"], test["age_plus_motion"]["spearman"]["mean_per_video"]
    lines += ["", f"Mean defined per-video ranking correlation is {value(both_rho,4)} for age plus motion, {value(age_rho,4)} for age only, "
        f"and {value(test['motion_only']['spearman']['mean_per_video'],4)} for motion only. Numerical-error improvement and ranking improvement must be assessed separately.", "",
        "![Test MAE](test_mae.png)", "", "![Paired improvements](test_paired_improvements.png)", "",
        "![Predicted versus observed drift](test_predicted_vs_observed.png)", "", "## Recommendation and limits", ""]
    if primary["ci95_low"] > 0 and intervals[3]["ci95_low"] > 0 and both_rho is not None and age_rho is not None and both_rho > age_rho and primary["improved"] >= 28:
        lines.append("The fixed motion inputs show broadly distributed magnitude and ranking improvements beyond age on this custom test set. "
                     "This supports a separate adaptive-policy experiment, with new held-out videos, recognition accuracy, and actual refresh/latency measurements. No controller is implemented here.")
    else:
        lines.append("Evidence for moving directly to an adaptive controller is limited or mixed. Resolve the specific magnitude/ranking and baseline comparisons above on additional untouched videos first. "
                     "A lower prediction error alone does not demonstrate useful refresh decisions. No adaptive controller is implemented here.")
    totals = {key: sum(v["counts"][key] for v in read_json(ROOT/metadata["output_directory"]/"collection_progress.json")["videos"])
              for key in ["decoded_frames", "mv_frames_available", "mv_frames_missing", "vector_count", "future_reference_vectors"]}
    lines += ["", f"Collection recorded {totals['decoded_frames']} frames, {totals['mv_frames_available']} with vectors and {totals['mv_frames_missing']} without, "
        f"and {totals['future_reference_vectors']} future-reference entries among {totals['vector_count']} vectors. "
        "Motion is a simple CPU codec-displacement summary, not optical flow, physical velocity, or a learned motion-recognition branch. "
        "Missing-vector frames remain valid with zero summaries and explicit availability.", "",
        "Existing B-frames/future-reference vectors make this an offline diagnostic, despite aggregating only current/preceding decoded positions. "
        "Original bytes are preserved; a controlled I/P-only experiment is separate. Appearance-feature drift is not action-recognition error or measured refresh benefit. "
        "Better drift prediction does not establish maintained recognition accuracy or adaptive-refresh speedup.", "",
        "This is a custom diagnostic subset from a seeded prefix of CVDF official validation archives, not an official full-dataset recognition evaluation. "
        "Twenty selected classes, 40 test videos, one encoder and one encoding distribution limit generalization. Bootstrap intervals do not include fitting or sampling uncertainty. "
        "No feature, alpha, or model tuning used test outcomes. Validation's candidate nomination is exploratory, not proof of policy effectiveness.", "",
        "## Audit artifacts", "", "See `sampling_plan.json`, `dataset_manifest.json`, `split_manifest.json`, `acquisition.json`, `exclusions.json`, "
        "`collection_metadata.json`, `features/`, `motion_frames/`, `observations.csv`, `fitted_predictors.json`, `validation_predictions.csv`, "
        "`test_predictions.csv`, `per_video_metrics.csv`, `paired_comparisons.csv`, `bootstrap_intervals.json`, `aggregate_metrics.json`, and `verification.json`. "
        "All original experiment sources/snapshots and release validation remain intact. Dataset archives, videos, and pretrained weights are excluded from Git.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/cache_drift_v2"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--check-video-files", action="store_true")
    args = parser.parse_args()
    output = args.output or args.input
    if (output / "aggregate_metrics.json").exists() and not args.verify_only:
        output = fresh_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    checks = focused_checks()
    frame, metadata, audits = audit_collection(args.input, check_files=args.check_video_files)
    train = frame.loc[frame.experiment_split == "train"].copy()
    validation = frame.loc[frame.experiment_split == "validation"].copy()
    test = frame.loc[frame.experiment_split == "test"].copy()
    assert len(train) == 1800 and len(validation) == len(test) == 600
    for a, b in [(train, validation), (train, test), (validation, test)]:
        assert set(a.source_video_id).isdisjoint(set(b.source_video_id))
    models = fit_models(train)
    changed = frame.copy()
    changed.loc[changed.experiment_split != "train", TARGET] = 1e9
    assert models == fit_models(changed.loc[changed.experiment_split == "train"])
    release = subprocess.run([sys.executable, str(ROOT/"validate_release.py")], cwd=ROOT, capture_output=True, text=True)
    assert release.returncode == 0, release.stdout+release.stderr
    verification = {"passed": True, "checked_at_utc": now(), "focused_checks": checks, "collection_audit": audits,
                    "training_only_scalers_models_constant": True, "held_out_target_mutation_cannot_change_fit": True,
                    "release_validator": {"returncode": release.returncode, "stdout": release.stdout.strip()},
                    "source_sha256": {"analyze_cache_drift_v2.py": sha256_file(ROOT/"analyze_cache_drift_v2.py")}}
    if args.verify_only:
        write_json(output / "verification.json", verification)
        print("V2 collection/training verification PASS (no validation/test prediction metrics inspected)")
        return 0
    # Serialize the fixed models before evaluating validation or test performance.
    fit_record = {"fitted_at_utc": now(), "training_video_ids": sorted(train.video_id.unique()),
                  "training_source_video_ids": sorted(train.source_video_id.unique()), "training_observation_ids": train.observation_id.tolist(),
                  "training_observations": len(train), "models": models,
                  "protocol_sha256": metadata["protocol_sha256"], "split_manifest_sha256": metadata["split_manifest_sha256"]}
    write_json(output / "fitted_predictors.json", fit_record)
    vp = predict(validation, models)
    vm = metrics_for_videos(vp)
    va = aggregate(vm, vp)
    candidate = min(PREDICTORS, key=lambda name: (va[name]["mae"]["mean_per_video"], name))
    write_csv(output / "validation_predictions.csv", vp.to_dict("records"), list(vp))
    write_json(output / "validation_selection.json", {"recorded_before_test_evaluation_at_utc": now(), "candidate": candidate,
               "rule": PROTOCOL["validation_candidate_rule"], "models_changed": False,
               "validation_aggregate": va, "fitted_predictors_sha256": sha256_file(output/"fitted_predictors.json")})
    # All four are evaluated regardless of the validation nomination.
    tp = predict(test, models)
    tm = metrics_for_videos(tp)
    ta = aggregate(tm, tp)
    pairs, intervals = paired_comparisons(tm)
    aggregates = {"aggregation": "equal-weight per-video, fixed training models", "validation": va, "test": ta,
                  "primary_comparison": intervals[0], "video_counts": {"train": 120, "validation": 40, "test": 40}}
    write_csv(output / "test_predictions.csv", tp.to_dict("records"), list(tp))
    for label, metrics in [("validation", vm), ("test", tm)]:
        metrics.insert(0, "experiment_split", label)
    pd.concat([vm, tm], ignore_index=True).to_csv(output / "per_video_metrics.csv", index=False, float_format="%.17g", na_rep="")
    pairs.to_csv(output / "paired_comparisons.csv", index=False, float_format="%.17g", na_rep="")
    write_json(output / "bootstrap_intervals.json", {"predefined_protocol": PROTOCOL["bootstrap"], "comparisons": intervals})
    write_json(output / "aggregate_metrics.json", aggregates)
    plot_results(tm, pairs, tp, output)
    (output / "findings.md").write_text(findings(aggregates, intervals, {**metadata, "output_directory": relative_path(args.input)}, candidate), encoding="utf-8")
    # Reload saved predictions and independently reproduce published metrics.
    for name, expected in [("validation", vm), ("test", tm)]:
        # Preserve exact float64 predictions, including near-ties used by ranks.
        saved = pd.read_csv(output / f"{name}_predictions.csv", float_precision="round_trip")
        regenerated = metrics_for_videos(saved)
        pd.testing.assert_frame_equal(expected.drop(columns="experiment_split"), regenerated, check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-12)
        assert (saved.groupby("observation_id").size() == 4).all()
    saved_models = read_json(output/"fitted_predictors.json")["models"]
    for held in [validation, test]:
        for name in PREDICTORS:
            np.testing.assert_array_equal(apply_fitted(held, models[name]), apply_fitted(held, saved_models[name]))
    verify_preservation(metadata["protected_original_sha256"])
    verification.update({"saved_predictions_reproduce_metrics": True, "saved_fitted_models_reproduce_predictions": True,
                         "one_prediction_per_method_per_observation": True, "prediction_csv_float_parser": "round_trip",
                         "validation_predictions": len(vp), "test_predictions": len(tp),
                         "split_counts": {"train": 120, "validation": 40, "test": 40}, "originals_unchanged": True})
    write_json(output/"verification.json", verification)
    write_json(output/"analysis_metadata.json", {"completed_at_utc": now(), "source_sha256": sha256_file(ROOT/"analyze_cache_drift_v2.py"),
               "collection_metadata_sha256": sha256_file(args.input/"collection_metadata.json"),
               "protocol_sha256": metadata["protocol_sha256"], "split_manifest_sha256": metadata["split_manifest_sha256"],
               "input_directory": relative_path(args.input), "artifacts_sha256": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != "analysis_metadata.json"}})
    print("V2 evaluation PASS: fixed 120 training videos; 40 validation and 40 unseen test videos, 2400 predictions each.")
    print(json.dumps({"test": ta, "comparisons": intervals}, indent=2))
    print(f"Saved {relative_path(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
