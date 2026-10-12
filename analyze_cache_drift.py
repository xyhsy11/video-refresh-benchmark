"""Whole-source-video held-out ridge prediction of appearance-feature drift."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collect_cache_drift import (AGE_INPUTS, MOTION_INPUTS, AUDIT_COLUMNS, OBSERVATION_COLUMNS,
                                cache_ordinal, check_protected, cosine_distance, interval_inputs,
                                observations_for_video)
from motion_utils import write_csv
from research_utils import ROOT, now, relative_path, sha256_file, write_json

PREDICTORS = {"constant": [], "age_only": AGE_INPUTS, "motion_only": MOTION_INPUTS,
              "age_plus_motion": AGE_INPUTS + MOTION_INPUTS}
ALPHA = 1.0
TARGET = "appearance_feature_drift"
TIE_TOLERANCE = 1e-12
REGRESSION_CONFIG = {
    "implementation": "NumPy float64 ridge normal equations; no additional dependency",
    "objective": "sum((y - intercept - X_standardized @ coefficients)**2) + alpha * sum(coefficients**2)",
    "alpha": ALPHA, "intercept": "training-target mean; not penalized",
    "standardization": "training-fold column mean and population standard deviation (ddof=0)",
    "constant_columns": "standard deviation <= 1e-12 replaced by 1; centered constant training columns become zero",
    "evaluation": "leave one entire source video out; same folds for all four predictors",
    "tuning": "none; fixed alpha=1.0 specified before evaluation",
    "primary_predictions": "clip raw held-out predictions to [0,2] for all metrics",
    "spearman": "Pearson correlation of average ranks for ties; undefined if either vector is constant",
    "aggregation": "calculate each metric per video, then give videos equal weight",
    "tie_tolerance_absolute_mae": TIE_TOLERANCE,
}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, dict]:
    """Only train supplies fitted parameters. Test supplies predictor inputs only."""
    y = train[TARGET].to_numpy(dtype=np.float64)
    intercept = float(y.mean())
    if not columns:
        return np.full(len(test), intercept), {"intercept": intercept, "training_rows": len(train), "columns": []}
    if columns not in [AGE_INPUTS, MOTION_INPUTS, AGE_INPUTS + MOTION_INPUTS]:
        raise ValueError("Only the explicit target-independent predictor whitelist is permitted")
    x = train[columns].to_numpy(dtype=np.float64)
    x_test = test[columns].to_numpy(dtype=np.float64)
    means = x.mean(axis=0)
    raw_scales = x.std(axis=0, ddof=0)
    scales = np.where(raw_scales <= 1e-12, 1.0, raw_scales)
    z = (x - means) / scales
    rhs = z.T @ (y - intercept)
    lhs = z.T @ z + ALPHA * np.eye(len(columns))
    coefficients = np.linalg.solve(lhs, rhs)
    prediction = intercept + ((x_test - means) / scales) @ coefficients
    if not np.isfinite(prediction).all() or not np.isfinite(coefficients).all():
        raise AssertionError("Nonfinite fitted coefficients/predictions")
    np.testing.assert_allclose(lhs @ coefficients, rhs, rtol=1e-9, atol=1e-10)
    return prediction, {"columns": columns, "intercept": intercept, "training_rows": len(train),
                        "means": means.tolist(), "raw_standard_deviations": raw_scales.tolist(),
                        "scales": scales.tolist(), "constant_columns": [columns[i] for i in np.flatnonzero(raw_scales <= 1e-12)],
                        "coefficients": coefficients.tolist(), "alpha": ALPHA}


def spearman(observed: np.ndarray, predicted: np.ndarray) -> tuple[float | None, str]:
    if np.ptp(observed) == 0:
        return None, "undefined: observed drift is constant"
    if np.ptp(predicted) == 0:
        return None, "undefined: predicted drift is constant"
    ranks_y = pd.Series(observed).rank(method="average").to_numpy()
    ranks_p = pd.Series(predicted).rank(method="average").to_numpy()
    return float(np.corrcoef(ranks_y, ranks_p)[0, 1]), "defined: average ranks for ties"


def focused_checks() -> dict:
    """Checks target ordering, interval boundaries, and fitting isolation directly."""
    checks = {}
    expected = [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12]
    assert [cache_ordinal(t, 4) for t in range(1, 16)] == expected
    checks["explicit_pre_decision_schedule_including_4_8_12"] = True
    feature = np.arange(1, 513, dtype=np.float32)
    assert abs(cosine_distance(feature, feature)) < 1e-12
    try:
        cosine_distance(np.zeros(512), feature)
    except ValueError:
        checks["zero_feature_norm_rejected"] = True
    else:
        raise AssertionError("Zero norm accepted")
    checks["identical_nonzero_feature_zero_drift"] = True
    frames = [{"decoded_frame_index": i, "mv_available": bool(i % 2),
               "mean_magnitude_px": float(i) if i % 2 else 0.0,
               "p90_magnitude_px": float(2*i) if i % 2 else 0.0} for i in range(10)]
    result = interval_inputs(frames, 1, 5)
    assert result["interval_mean_available_magnitude_px"] == 4.0
    assert result["interval_max_available_p90_px"] == 10.0
    assert result["interval_mv_available_fraction"] == 0.5
    modified = [{**f, "mean_magnitude_px": 1e8, "p90_magnitude_px": 2e8} if f["decoded_frame_index"] > 5 else f for f in frames]
    assert interval_inputs(modified, 1, 5) == result
    missing = [{**f, "mv_available": False, "mean_magnitude_px": 0.0, "p90_magnitude_px": 0.0} for f in frames]
    zero = [{**f, "mv_available": True} for f in missing]
    assert interval_inputs(missing, 1, 5)["interval_mv_available_fraction"] == 0
    assert interval_inputs(zero, 1, 5)["interval_mv_available_fraction"] == 1
    checks["interval_excludes_cache_frame_and_future_frames"] = True
    checks["equal_frame_weight_and_missing_versus_zero"] = True
    train = pd.DataFrame({name: np.arange(8, dtype=float) for name in AGE_INPUTS})
    train[AGE_INPUTS[-1]] = 1.0
    train[TARGET] = np.linspace(0.1, 0.4, 8)
    test = train.iloc[:2].copy()
    _, original_state = fit_predict(train, test, AGE_INPUTS)
    test.loc[:, AGE_INPUTS] = 1e9
    test[TARGET] = 1e12
    _, changed_state = fit_predict(train, test, AGE_INPUTS)
    assert original_state == changed_state
    assert original_state["scales"][-1] == 1.0 and AGE_INPUTS[-1] in original_state["constant_columns"]
    # Independent augmented least-squares formulation checks the ridge solve.
    x = train[AGE_INPUTS].to_numpy()
    z = (x - np.array(original_state["means"])) / np.array(original_state["scales"])
    augmented_x = np.vstack([z, np.sqrt(ALPHA)*np.eye(len(AGE_INPUTS))])
    augmented_y = np.concatenate([train[TARGET].to_numpy()-train[TARGET].mean(), np.zeros(len(AGE_INPUTS))])
    independent = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    np.testing.assert_allclose(independent, original_state["coefficients"], rtol=1e-12, atol=1e-12)
    checks["held_out_inputs_and_targets_cannot_change_fitted_scaler_or_predictor"] = True
    checks["constant_columns_safe_and_ridge_matches_independent_least_squares"] = True
    assert spearman(np.arange(4), np.ones(4))[0] is None
    assert spearman(np.ones(4), np.arange(4))[0] is None
    assert spearman(np.arange(4), np.arange(4))[0] == 1.0
    checks["constant_rank_correlation_reported_undefined"] = True
    assert not set(AGE_INPUTS + MOTION_INPUTS) & set(AUDIT_COLUMNS + [TARGET])
    try:
        fit_predict(train, test, AGE_INPUTS + [TARGET])
    except ValueError:
        checks["target_and_identifiers_rejected_as_predictors"] = True
    else:
        raise AssertionError("Target leakage accepted")
    return checks


def load_and_validate(directory: Path) -> tuple[pd.DataFrame, dict, dict]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "collected" or metadata["device"] != "cuda:0":
        raise AssertionError("Completed CUDA reference collection required")
    for name, expected in metadata["collection_artifact_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise AssertionError(f"Collection artifact changed: {name}")
    for name, expected in metadata["source_code_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise AssertionError(f"Collection source changed: {name}")
    check_protected(metadata["protected_original_sha256"])
    frame = pd.read_csv(directory / "observations.csv")
    if list(frame.columns) != OBSERVATION_COLUMNS or frame.observation_id.duplicated().any():
        raise AssertionError("Observation schema/IDs invalid")
    motion = pd.read_csv(directory / "motion_frames.csv")
    if frame.video_id.nunique() != metadata["video_count"] or len(frame) != metadata["observation_count"]:
        raise AssertionError("Video/observation counts invalid")
    checked = []
    for entry in metadata["videos"]:
        video_id = entry["video_id"]
        side = json.loads((directory / "features" / f"{video_id}.json").read_text(encoding="utf-8"))
        with np.load(directory / "features" / f"{video_id}.npz", allow_pickle=False) as saved:
            features = saved["features"]
            indices = saved["frame_indices"]
            stamps = saved["timestamps_seconds"]
        if features.shape != (16, 512) or features.dtype != np.float32 or len(side["candidates"]) != 16:
            raise AssertionError("Expected 16 reference features per video")
        assert indices.tolist() == entry["candidate_indices"] == [c["decoded_frame_index"] for c in side["candidates"]]
        np.testing.assert_array_equal(stamps, [c["timestamp_seconds"] for c in side["candidates"]])
        actual = frame.loc[frame.video_id == video_id].reset_index(drop=True)
        assert len(actual) == 15 and actual.candidate_ordinal.tolist() == list(range(1, 16))
        assert actual.cached_candidate_ordinal.tolist() == [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12]
        frames = motion.loc[motion.video_id == video_id].to_dict("records")
        assert len(frames) == entry["decoded_frame_count"]
        assert [f["decoded_frame_index"] for f in frames] == list(range(len(frames)))
        for f in frames:
            assert np.isfinite([f["mean_magnitude_px"], f["p90_magnitude_px"]]).all()
            assert f["mv_available"] == (f["vector_count"] > 0)
            assert f["past_reference_vectors"] + f["future_reference_vectors"] + f["other_reference_vectors"] == f["vector_count"]
            if not f["mv_available"]:
                assert f["mean_magnitude_px"] == f["p90_magnitude_px"] == 0
        rebuilt = pd.DataFrame(observations_for_video(entry, features, side["candidates"], frames, metadata["refresh_every"]))[OBSERVATION_COLUMNS]
        pd.testing.assert_frame_equal(actual, rebuilt, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
        # Independently audit every interval and label, instead of relying only on
        # the same observation-building helper used during collection.
        for row in actual.to_dict("records"):
            t, r = row["candidate_ordinal"], row["cached_candidate_ordinal"]
            a, b = features[t].astype(np.float64), features[r].astype(np.float64)
            independently_calculated = 1 - np.clip(np.sum(a*b)/(np.sqrt(np.sum(a*a))*np.sqrt(np.sum(b*b))), -1, 1)
            assert abs(row[TARGET] - independently_calculated) <= 1e-12
            assert row["interval_start_frame"] == indices[r]+1 and row["interval_end_frame"] == indices[t]
            available = [f for f in frames if indices[r] < f["decoded_frame_index"] <= indices[t] and f["mv_available"]]
            assert len(available) == row["interval_available_frame_count"]
            assert row["interval_frame_count"] == indices[t]-indices[r]
            if available:
                assert abs(row["interval_mean_available_magnitude_px"] - sum(f["mean_magnitude_px"] for f in available)/len(available)) <= 1e-10
                assert abs(row["interval_max_available_p90_px"] - max(f["p90_magnitude_px"] for f in available)) <= 1e-10
            assert 0 <= row[TARGET] <= 2
        checked.append({"video_id": video_id, "reference_features": 16, "observations": 15,
                        "independent_labels_and_intervals": True, "passed": True})
    assert metadata["reference_feature_count"] == 16 * metadata["video_count"]
    assert len(frame) == 15 * metadata["video_count"]
    assert np.isfinite(frame[AGE_INPUTS+MOTION_INPUTS+[TARGET]].to_numpy(dtype=float)).all()
    return frame, metadata, {"videos": checked, "collection_hashes": True, "original_experiments_unchanged": True}


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    groups = sorted(frame.source_video_id.unique())
    if len(groups) < 2:
        raise ValueError("Leave-one-video-out analysis requires multiple source videos; the smoke collection cannot be evaluated")
    predictions, folds = [], []
    for fold_id, group in enumerate(groups, start=1):
        train = frame.loc[frame.source_video_id != group].copy()
        test = frame.loc[frame.source_video_id == group].copy()
        assert set(train.source_video_id).isdisjoint(set(test.source_video_id))
        assert set(train.video_id).isdisjoint(set(test.video_id))
        assert set(train.observation_id).isdisjoint(set(test.observation_id))
        fold = {"fold_id": fold_id, "held_out_source_video_ids": [group],
                "held_out_video_ids": sorted(test.video_id.unique()), "training_source_video_ids": sorted(train.source_video_id.unique()),
                "training_video_ids": sorted(train.video_id.unique()), "training_observation_ids": train.observation_id.tolist(),
                "held_out_observation_ids": test.observation_id.tolist(), "models": {}}
        for name, columns in PREDICTORS.items():
            raw, state = fit_predict(train, test, columns)
            # Strong held-out-label isolation check on every real fold.
            changed_test = test.copy()
            changed_test[TARGET] = 1e9
            other_raw, other_state = fit_predict(train, changed_test, columns)
            assert state == other_state
            np.testing.assert_array_equal(raw, other_raw)
            if columns:
                np.testing.assert_allclose(state["means"], train[columns].mean().to_numpy(), rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(state["raw_standard_deviations"], train[columns].std(ddof=0).to_numpy(), rtol=1e-12, atol=1e-12)
            assert abs(state["intercept"] - train[TARGET].mean()) < 1e-12 and state["training_rows"] == len(train)
            fold["models"][name] = state
            clipped = np.clip(raw, 0, 2)
            for i, (_, row) in enumerate(test.iterrows()):
                predictions.append({"observation_id": row.observation_id, "video_id": row.video_id,
                                    "source_video_id": row.source_video_id, "candidate_ordinal": int(row.candidate_ordinal),
                                    "cached_candidate_ordinal": int(row.cached_candidate_ordinal), "fold_id": fold_id,
                                    "held_out_source_video_id": group, "predictor": name, "observed_drift": row[TARGET],
                                    "prediction_raw": float(raw[i]), "prediction_clipped": float(clipped[i])})
        folds.append(fold)
    result = pd.DataFrame(predictions)
    assert len(result) == len(frame)*4
    assert not result.duplicated(["observation_id", "predictor"]).any()
    assert (result.groupby("observation_id").size() == 4).all()
    assert set(result.observation_id) == set(frame.observation_id)
    assert np.isfinite(result[["observed_drift", "prediction_raw", "prediction_clipped"]]).all().all()
    assert result.prediction_clipped.between(0, 2).all()
    for _, rows in result.groupby("observation_id"):
        assert set(rows.predictor) == set(PREDICTORS) and rows.fold_id.nunique() == 1
    return result, folds


def metrics_for_videos(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (video_id, predictor), part in predictions.groupby(["video_id", "predictor"], sort=True):
        observed = part.observed_drift.to_numpy()
        predicted = part.prediction_clipped.to_numpy()
        error = predicted - observed
        rho, note = spearman(observed, predicted)
        rows.append({"video_id": video_id, "source_video_id": part.source_video_id.iloc[0], "predictor": predictor,
                     "observation_count": len(part), "mae": float(np.abs(error).mean()),
                     "rmse": float(np.sqrt(np.mean(error**2))), "spearman": rho, "spearman_note": note})
    return pd.DataFrame(rows)


def paired_improvements(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(index="video_id", columns="predictor", values="mae")
    paired = pivot[["age_only", "age_plus_motion"]].rename(columns={"age_only": "mae_age_only", "age_plus_motion": "mae_age_plus_motion"}).reset_index()
    paired["absolute_mae_improvement"] = paired.mae_age_only - paired.mae_age_plus_motion
    denominator = paired.mae_age_only.where(paired.mae_age_only != 0, np.nan)
    paired["percentage_mae_improvement"] = 100*paired.absolute_mae_improvement/denominator
    paired["outcome"] = np.where(paired.absolute_mae_improvement > TIE_TOLERANCE, "improved",
                                  np.where(paired.absolute_mae_improvement < -TIE_TOLERANCE, "worsened", "tied"))
    return paired


def aggregates(metrics: pd.DataFrame, paired: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    per_model = {}
    for model, rows in metrics.groupby("predictor"):
        per_model[model] = {}
        for name in ["mae", "rmse", "spearman"]:
            values = rows[name].dropna()
            per_model[model][name] = {"mean_per_video": float(values.mean()) if len(values) else None,
                                     "median_per_video": float(values.median()) if len(values) else None,
                                     "defined_video_count": len(values), "undefined_video_count": len(rows)-len(values)}
    percent = paired.percentage_mae_improvement.dropna()
    ranked = paired.sort_values("absolute_mae_improvement", ascending=False)
    top = ranked.head(min(2, len(ranked)))
    net = float(paired.absolute_mae_improvement.sum())
    remaining = ranked.iloc[len(top):]
    return {
        "aggregation": "equal-weight per-video metrics and paired improvements; not pooled-row independent inference",
        "video_count": metrics.video_id.nunique(), "source_video_count": metrics.source_video_id.nunique(),
        "observation_count": predictions.observation_id.nunique(), "prediction_count": len(predictions),
        "models": per_model,
        "age_plus_motion_vs_age_only": {
            "positive_means": "age plus motion predicts more accurately (smaller clipped-prediction MAE)",
            "mean_absolute_mae_improvement": float(paired.absolute_mae_improvement.mean()),
            "median_absolute_mae_improvement": float(paired.absolute_mae_improvement.median()),
            "mean_percentage_mae_improvement": float(percent.mean()) if len(percent) else None,
            "median_percentage_mae_improvement": float(percent.median()) if len(percent) else None,
            "undefined_percentage_count": len(paired)-len(percent),
            "improved": int((paired.outcome == "improved").sum()), "tied": int((paired.outcome == "tied").sum()),
            "worsened": int((paired.outcome == "worsened").sum()),
            "min_percentage_mae_improvement": float(percent.min()) if len(percent) else None,
            "max_percentage_mae_improvement": float(percent.max()) if len(percent) else None,
        },
        "clipping_counts_by_predictor": {name: int((part.prediction_raw != part.prediction_clipped).sum()) for name, part in predictions.groupby("predictor")},
        "descriptive_gain_concentration": {
            "largest_two_gain_video_ids": top.video_id.tolist(),
            "largest_two_sum_absolute_mae_improvement": float(top.absolute_mae_improvement.sum()),
            "net_sum_absolute_mae_improvement": net,
            "largest_two_percent_of_net_gain": 100*float(top.absolute_mae_improvement.sum())/net if net > 0 else None,
            "remaining_videos_mean_absolute_mae_improvement": float(remaining.absolute_mae_improvement.mean()) if len(remaining) else None,
            "note": "post-evaluation descriptive concentration, not a changed evaluation set or tuned model",
        },
        "correlation_missing_policy": "undefined constant-vector correlations are missing, with reasons; average only defined per-video correlations",
        "independence_limit": "ten source videos; fifteen observations within each are correlated; no statistical significance claim",
    }


def plots(metrics: pd.DataFrame, paired: pd.DataFrame, predictions: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(len(paired))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x-0.19, paired.mae_age_only, width=0.38, label="Age only", color="#2878b5")
    ax.bar(x+0.19, paired.mae_age_plus_motion, width=0.38, label="Age plus motion", color="#f28e2b")
    ax.set_xticks(x, paired.video_id, rotation=40, ha="right")
    ax.set_ylabel("Held-out MAE of appearance-feature drift (cosine distance)")
    ax.set_title("Whole-video held-out prediction: lower MAE is better")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.5, 0.015, "15 observations per video; fixed ridge alpha=1.0; clipped predictions; no significance claim", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / "mae_age_vs_motion.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    videos = sorted(predictions.video_id.unique())
    palette = plt.get_cmap("tab10")
    limit = max(float(predictions.observed_drift.max()), float(predictions.prediction_clipped.max()))*1.05
    for ax, model in zip(axes.flat, PREDICTORS):
        part = predictions.loc[predictions.predictor == model]
        for i, video in enumerate(videos):
            selected = part.loc[part.video_id == video]
            ax.scatter(selected.observed_drift, selected.prediction_clipped, s=20, alpha=0.7, color=palette(i), label=video)
        ax.plot([0, limit], [0, limit], color="gray", ls="--", lw=1)
        ax.set_title(model.replace("_", " ").title())
        ax.set_xlim(-0.01, limit)
        ax.set_ylim(-0.01, limit)
        ax.grid(alpha=0.2)
    fig.supxlabel("Observed appearance-feature drift (cosine distance)", y=0.125)
    fig.supylabel("Held-out predicted drift, clipped to [0, 2]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Leave-one-source-video-out predictions; colors identify videos for auditing")
    fig.tight_layout(rect=(0.02, 0.15, 1, 0.96))
    fig.savefig(output / "predicted_vs_observed.png", dpi=180)
    plt.close(fig)


def findings(metrics: pd.DataFrame, paired: pd.DataFrame, aggregate: dict, metadata: dict, collection_prefix: str = "") -> str:
    a = aggregate["age_plus_motion_vs_age_only"]
    improved = paired.loc[paired.outcome == "improved", "video_id"].tolist()
    worsened = paired.loc[paired.outcome == "worsened", "video_id"].tolist()
    model_maes = {name: data["mae"]["mean_per_video"] for name, data in aggregate["models"].items()}
    def percentage(value):
        return "undefined" if value is None else f"{value:+.2f}%"
    def correlation(value):
        return "undefined" if value is None else f"{value:.4f}"
    best = min(model_maes, key=model_maes.get)
    concentration = aggregate["descriptive_gain_concentration"]
    concentration_text = (
        f"The two largest absolute improvements ({', '.join(concentration['largest_two_gain_video_ids'])}) account for "
        f"{concentration['largest_two_percent_of_net_gain']:.2f}% of the net summed improvement. "
        f"The other videos' mean absolute improvement is {concentration['remaining_videos_mean_absolute_mae_improvement']:+.6f}. "
        "This describes concentration after evaluation; no observations were removed from primary metrics and no model was retuned."
        if concentration["largest_two_percent_of_net_gain"] is not None and concentration["remaining_videos_mean_absolute_mae_improvement"] is not None
        else "The paired outcomes above describe consistency; primary metrics retain every video."
    )
    motion_comparison = "stronger" if model_maes["motion_only"] < model_maes["age_only"] else "weaker"
    if a["mean_absolute_mae_improvement"] > 0 and a["improved"] >= 8:
        recommendation = "Motion shows fairly consistent incremental predictive information in this small diagnostic. Validate the fixed model on a larger untouched set before designing or evaluating an adaptive refresh controller."
    elif a["mean_absolute_mae_improvement"] > 0:
        recommendation = "The average improvement is mixed across videos. Keep cache age as the baseline and validate motion on a larger untouched set before relying on it for refresh decisions."
    else:
        recommendation = "The chosen motion inputs do not improve average held-out MAE beyond cache age in this experiment. Keep cache age as the simpler baseline; collect a larger untouched set and investigate descriptor quality before building an adaptive controller."
    text = ["# Cache-drift prediction diagnostic", "",
            f"Actual CUDA collection: {metadata['started_at_utc']} through {metadata['completed_at_utc']}. "
            f"{metadata['video_count']} videos, {metadata['reference_feature_count']} fresh float32 ResNet18 features, "
            f"{metadata['observation_count']} pre-decision drift observations. This is reference collection and offline prediction evaluation, not a latency measurement.", "",
            "## What was predicted", "",
            "Appearance-feature drift is cosine distance between each current fresh feature and the previous periodic cache. "
            "Candidate 0 is excluded. At t=4/8/12 the label compares against candidate 0/4/8 BEFORE refreshing, so scheduled refreshes do not create artificial zero labels. "
            "Predictors receive four cache-age inputs, six motion inputs, or both; never RGB features, target-derived inputs, video identity, or future decoded-frame positions. "
            "Interval descriptors cover frames strictly after the cached candidate through the current candidate inclusive. Missing vectors retain availability flags.", "",
            "## Held-out results", "",
            "Whole source videos are held out. Each fold trains on the other nine videos, including fitting standardization there only. "
            "Ridge alpha is fixed at 1.0 with an unpenalized intercept and no tuning. Raw and [0,2]-clipped predictions are saved; primary metrics consistently use clipped predictions. "
            "Smaller MAE/RMSE means more accurate drift prediction. Spearman measures whether larger predictions tend to accompany larger observed drift; constant-vector correlations are undefined and saved as missing.", "",
            "| Predictor | Mean per-video MAE | Median per-video MAE | Mean per-video RMSE | Mean defined per-video Spearman |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for model in PREDICTORS:
        data = aggregate["models"][model]
        rho = data["spearman"]["mean_per_video"]
        text.append(f"| {model} | {data['mae']['mean_per_video']:.6f} | {data['mae']['median_per_video']:.6f} | {data['rmse']['mean_per_video']:.6f} | {'undefined' if rho is None else f'{rho:.4f}'} |")
    text += ["", "Age plus motion versus age only, calculated per video before equal-weight aggregation:", "",
             f"- Mean paired absolute MAE improvement: **{a['mean_absolute_mae_improvement']:+.6f}**; median: **{a['median_absolute_mae_improvement']:+.6f}**.",
             f"- Mean paired percentage improvement: **{percentage(a['mean_percentage_mae_improvement'])}**; median: **{percentage(a['median_percentage_mae_improvement'])}**; range **{percentage(a['min_percentage_mae_improvement'])} to {percentage(a['max_percentage_mae_improvement'])}**. Percentage improvement is undefined when age-only MAE is zero.",
             f"- Improved **{a['improved']}**, tied **{a['tied']}**, worsened **{a['worsened']}** videos. Positive improvement means smaller MAE; negative results are retained.",
             f"- Benefited: {', '.join(improved) or 'none'}. Worsened: {', '.join(worsened) or 'none'}.",
             f"- Motion only is **{motion_comparison}** than age only by equal-weight mean per-video MAE ({model_maes['motion_only']:.6f} versus {model_maes['age_only']:.6f}). The lowest mean MAE among the four models is **{best}**.", "",
             "| Video | Age-only MAE | Age-plus-motion MAE | Absolute improvement | Percentage improvement | Outcome |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in paired.itertuples():
        percent = "undefined" if pd.isna(row.percentage_mae_improvement) else f"{row.percentage_mae_improvement:+.2f}%"
        text.append(f"| {row.video_id} | {row.mae_age_only:.6f} | {row.mae_age_plus_motion:.6f} | {row.absolute_mae_improvement:+.6f} | {percent} | {row.outcome} |")
    text += ["", "Consistency is assessed from these ten paired outcomes, not from treating 150 within-video observations as independent samples. "
             "Large relative changes for videos with small age-only MAE can influence the mean percentage improvement; the median and absolute improvements are shown alongside it. "
             "These are descriptive results, with no statistical significance claim. Shared training folds also make out-of-fold errors dependent.", "",
             concentration_text, "",
             f"The constant training-mean baseline has mean per-video MAE {model_maes['constant']:.6f}, compared with {model_maes['age_only']:.6f} for age only. "
             "The richer models should therefore be judged against the constant baseline as well as age alone. "
             f"Age-plus-motion mean per-video Spearman is {correlation(aggregate['models']['age_plus_motion']['spearman']['mean_per_video'])}, "
             f"versus {correlation(aggregate['models']['age_only']['spearman']['mean_per_video'])} for age only. "
             "Lower average MAE does not establish better within-video ranking, which matters for selecting refresh positions.", "",
             "![Per-video MAE](mae_age_vs_motion.png)", "", "![Held-out predictions](predicted_vs_observed.png)", "",
             "## Recommendation and limits", "", recommendation, "",
             "A drift predictor could eventually inform when to recompute appearance features, but this diagnostic does not evaluate refresh decisions, measured refresh benefit, "
             "latency savings, or action-recognition accuracy. ResNet18 is an appearance-feature prototype, with no evaluated action-classification head. "
             "Low drift need not preserve semantic decisions; high drift need not indicate an action error.", "",
             "Motion inputs are simple CPU summaries of codec displacement, not optical flow, physical velocity, or an evaluated motion-recognition branch. "
             "The original encodings include B-frames and future-reference vectors. Aggregation uses only current/preceding decoded-frame positions, "
             "but the underlying vectors can refer to future frames: this is an offline diagnostic, not strictly causal streaming inference. Original encodings are preserved; controlled I/P-only encoding is a separate experiment.", "",
             "Only ten independent source videos and one encoder are evaluated. Generalization to new actions, compression settings, resolutions, and hardware, "
             "the usefulness of motion for refresh decisions, and recognition accuracy remain uncertain. No alpha or descriptor tuning was performed on these held-out videos. "
             "Use a new, larger evaluation set for subsequent model choices rather than repeatedly optimizing this pilot.", "",
             "## Reproduction and audit", "",
             f"See [observations]({collection_prefix}observations.csv), [predictions](predictions_oof.csv), [per-video metrics](per_video_metrics.csv), "
             "[paired comparisons](paired_improvements.csv), [aggregates](aggregate_metrics.json), [folds and fitted parameters](folds.json), "
             f"[collection metadata](metadata.json), and [verification](verification.json). [Candidate features]({collection_prefix}features/) and "
             f"[per-frame motion]({collection_prefix}motion_frames.csv) permit independent interval audits. "
             "Metadata records real timestamps, video/weight/source hashes, software, decoder/export settings, and all definitions. "
             "Original release validation and preservation checks passed. No videos or downloaded model weights are included.", ""]
    return "\n".join(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    if (args.output / "aggregate_metrics.json").exists():
        raise ValueError("Completed analysis preserved. Choose a fresh --output directory for recomputation.")
    checks = focused_checks()
    frame, metadata, validation = load_and_validate(args.input)
    np.random.seed(args.seed)  # Ridge and sorted folds are deterministic; no randomized split or tuning.
    predictions, folds = evaluate(frame)
    metrics = metrics_for_videos(predictions)
    paired = paired_improvements(metrics)
    aggregate = aggregates(metrics, paired, predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "predictions_oof.csv", predictions.to_dict("records"), list(predictions))
    metrics.to_csv(args.output / "per_video_metrics.csv", index=False, float_format="%.17g", na_rep="")
    paired.to_csv(args.output / "paired_improvements.csv", index=False, float_format="%.17g", na_rep="")
    write_json(args.output / "aggregate_metrics.json", aggregate)
    write_json(args.output / "folds.json", {"seed": args.seed, "seed_role": "recorded for reproduction; fixed sorted folds and ridge are deterministic",
                                           "regression": REGRESSION_CONFIG, "folds": folds})
    plots(metrics, paired, predictions, args.output)
    if args.input.resolve() != args.output.resolve():
        # Preserve the source collection while allowing offline recomputation.
        write_json(args.output / "metadata.json", {**metadata, "collection_directory": relative_path(args.input)})
    prefix = "" if args.input.resolve() == args.output.resolve() else Path(os.path.relpath(args.input.resolve(), args.output.resolve())).as_posix() + "/"
    (args.output / "findings.md").write_text(findings(metrics, paired, aggregate, metadata, prefix), encoding="utf-8")
    release = subprocess.run([sys.executable, str(ROOT / "validate_release.py")], cwd=ROOT, capture_output=True, text=True)
    if release.returncode:
        raise AssertionError(f"Original release validator failed: {release.stdout}\n{release.stderr}")
    check_protected(metadata["protected_original_sha256"])
    verification = json.loads((args.input / "verification.json").read_text(encoding="utf-8"))
    verification["analysis"] = {"passed": True, "checked_at_utc": now(), "seed": args.seed,
                                "focused_checks": checks, "collection_validation": validation,
                                "whole_source_video_folds_disjoint": True, "scalers_and_models_training_only": True,
                                "held_out_target_mutation_checks": True, "one_prediction_per_observation_per_predictor": True,
                                "finite_inputs_labels_raw_and_clipped_predictions": True,
                                "prediction_count": len(predictions), "fold_count": len(folds),
                                "original_release_validator": {"returncode": release.returncode, "stdout": release.stdout.strip()},
                                "regression": REGRESSION_CONFIG,
                                "analysis_source_sha256": {"analyze_cache_drift.py": sha256_file(ROOT / "analyze_cache_drift.py")}}
    write_json(args.output / "verification.json", verification)
    write_json(args.output / "analysis_metadata.json", {"completed_at_utc": now(), "input_directory": relative_path(args.input),
               "collection_metadata_sha256": sha256_file(args.input / "metadata.json"), "regression": REGRESSION_CONFIG,
               "seed": args.seed, "source_sha256": sha256_file(ROOT / "analyze_cache_drift.py"),
               "artifacts_sha256": {p.name: sha256_file(p) for p in args.output.iterdir() if p.is_file() and p.name != "analysis_metadata.json"}})
    print(f"Verification PASS: {metadata['video_count']} videos, {metadata['reference_feature_count']} features, {len(frame)} observations, {len(folds)} folds, {len(predictions)} predictions.")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(json.dumps(aggregate["age_plus_motion_vs_age_only"], indent=2))
    print(release.stdout.strip())
    print(f"Saved analysis: {relative_path(args.output)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
