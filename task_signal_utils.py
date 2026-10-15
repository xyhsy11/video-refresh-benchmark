"""Fixed cheap signals and signed equal-budget probes; no encoder execution.

The feature builder accepts only observations available before the proposed
refresh. Labels and current/future reference features belong to the evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from collect_cache_drift import MOTION_INPUTS, interval_inputs
from task_aware_oracle_utils import cache_indices, feature_logits, rank_correlation

TOL = 1e-12
PERIODIC = [0, 4, 8, 12]
AGE = ["candidate_position_normalized", "cached_age_seconds",
       "remaining_refresh_budget", "positions_until_planned_refresh"]
CONFIDENCE = [f"{prefix}_{name}" for prefix in ["cached", "past_output"]
              for name in ["entropy_nats", "max_probability", "top_two_probability_margin"]]
GROUPS = {"A": AGE, "B": AGE + MOTION_INPUTS, "C": AGE + CONFIDENCE,
          "D": AGE + MOTION_INPUTS + CONFIDENCE}
PREDICTORS = ["zero", "training_mean", *GROUPS]
TARGET = "refresh_gain"


@dataclass(frozen=True)
class CheapObservation:
    t: int
    last_refresh: int
    next_planned: int
    timestamps_prefix: tuple
    indices_prefix: tuple
    motion_prefix: tuple
    cached_logits: np.ndarray
    past_output_logits: np.ndarray


def confidence(logits: np.ndarray) -> list[float]:
    z = np.asarray(logits, dtype=np.float64)
    assert z.shape == (20,) and np.isfinite(z).all()
    log_p = z - np.logaddexp.reduce(z)
    p = np.exp(log_p)
    top = np.sort(p)[-2:]
    return [float(-(p * log_p).sum()), float(top[-1]), float(top[-1] - top[-2])]


def cheap_features(observation: CheapObservation) -> dict[str, float]:
    """No feature arrays, identity, labels, evaluator scores or future rows."""
    o = observation
    t, r = o.t, o.last_refresh
    assert 0 <= r < t < o.next_planned <= 12
    assert o.next_planned == r + 4 and r in [0, 4, 8]
    assert len(o.indices_prefix) == len(o.timestamps_prefix) == t + 1
    assert len(o.motion_prefix) == o.indices_prefix[t] + 1
    assert o.past_output_logits.shape == (t, 20)
    values = dict(zip(AGE, [t / 15, o.timestamps_prefix[t] - o.timestamps_prefix[r],
                            3 - r // 4, o.next_planned - t]))
    motion = interval_inputs(o.motion_prefix, o.indices_prefix[r], o.indices_prefix[t])
    # The reused helper also returns interval audit fields; they are not inputs.
    values.update({name: motion[name] for name in MOTION_INPUTS})
    summaries = confidence(o.cached_logits) + confidence(o.past_output_logits.mean(axis=0))
    values.update(zip(CONFIDENCE, summaries))
    assert set(values) == set(GROUPS["D"]) and np.isfinite(list(values.values())).all()
    return {key: float(values[key]) for key in GROUPS["D"]}


def observed_inputs(video, t: int, parameters: dict) -> dict[str, float]:
    """Read only previously computed periodic refresh features, never f_t."""
    r = (t // 4) * 4
    assert t % 4 in [1, 2, 3] and r < t
    cache = cache_indices(PERIODIC)[:t]
    past = feature_logits(video.references[cache], parameters)
    observation = CheapObservation(t, r, r + 4, video.cheap.timestamps[:t+1],
        video.cheap.indices[:t+1], video.cheap.motion[:video.cheap.indices[t]+1],
        feature_logits(video.references[[r]], parameters)[0], past)
    return cheap_features(observation)


def make_folds(entries: list[dict], classes: list[str], seed: int) -> dict:
    rows = []
    for label in classes:
        group = [e for e in entries if e["recognition_role"] == "development" and e["label"] == label]
        assert len(group) == 5
        group.sort(key=lambda e: (hashlib.sha256(
            f"task_signal_diagnostic_v1|fold|{seed}|{label}|{e['source_video_id']}".encode()
        ).hexdigest(), e["source_video_id"]))
        rows.extend({"source_video_id": e["source_video_id"], "video_id": e["video_id"],
                     "label": label, "fold": i} for i, e in enumerate(group))
    assert len(rows) == len({r["source_video_id"] for r in rows}) == 100
    return {"seed": seed, "rule": "Within each original class, sort by SHA256(experiment|fold|seed|label|source); assign ranks 0..4.",
            "videos": rows, "folds": [
                {"fold": f, "training_sources": sorted(r["source_video_id"] for r in rows if r["fold"] != f),
                 "validation_sources": sorted(r["source_video_id"] for r in rows if r["fold"] == f)}
                for f in range(5)]}


def fit_ridge(table: pd.DataFrame, group: str) -> dict:
    columns = GROUPS[group]
    x = table[columns].to_numpy(dtype=np.float64)
    y = table[TARGET].to_numpy(dtype=np.float64)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    means = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    scales = np.where(std <= TOL, 1.0, std)
    z = (x - means) / scales
    intercept = float(y.mean())
    lhs = z.T @ z + np.eye(len(columns))
    rhs = z.T @ (y - intercept)
    coefficients = np.linalg.solve(lhs, rhs)
    np.testing.assert_allclose(lhs @ coefficients, rhs, rtol=1e-9, atol=1e-10)
    return {"group": group, "columns": columns, "target": TARGET, "alpha": 1.0,
            "objective": "sum squared residuals + alpha * squared coefficient norm",
            "intercept_penalized": False, "means": means.tolist(), "scales": scales.tolist(),
            "raw_standard_deviations": std.tolist(), "coefficients": coefficients.tolist(),
            "intercept": intercept, "training_rows": len(table),
            "training_source_ids": sorted(table.source_video_id.unique()),
            "training_example_ids": sorted(table.example_id),
            "constant_columns": [c for c, s in zip(columns, std) if s <= TOL]}


def predict(table: pd.DataFrame, model: dict) -> np.ndarray:
    assert model["columns"] == GROUPS[model["group"]] and model["target"] == TARGET
    x = table[model["columns"]].to_numpy(dtype=np.float64)
    y = model["intercept"] + ((x - model["means"]) / model["scales"]) @ np.array(model["coefficients"])
    assert np.isfinite(y).all()
    return y


def auc(scores: np.ndarray, event: np.ndarray) -> dict:
    event = np.asarray(event, dtype=bool)
    n1, n0 = int(event.sum()), int((~event).sum())
    if not n1 or not n0:
        return {"AUROC": None, "positive_events": n1, "negative_events": n0,
                "undefined_reason": "outcome has one class"}
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    value = (ranks[event].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return {"AUROC": float(value), "positive_events": n1, "negative_events": n0, "undefined_reason": None}


def correlation(x: np.ndarray, y: np.ndarray) -> dict:
    rho, reason = rank_correlation(np.asarray(x), np.asarray(y))
    return {"spearman": rho, "undefined_reason": reason}


def signs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.where(x > TOL, 1, np.where(x < -TOL, -1, 0))


def summarize_predictions(predictions: pd.DataFrame, seed: int) -> tuple[dict, pd.DataFrame, dict, dict]:
    per_video = []
    for (cohort, name, source), g in predictions.groupby(["cohort", "predictor", "source_video_id"], sort=True):
        assert len(g) == 9 and not g.example_id.duplicated().any()
        target, p = g[TARGET].to_numpy(), g.prediction.to_numpy()
        per_video.append({"cohort": cohort, "predictor": name, "source_video_id": source,
            "MAE": float(np.abs(target-p).mean()), "RMSE": float(np.sqrt(np.mean((target-p)**2))),
            "sign_accuracy": float((signs(target) == signs(p)).mean()),
            **correlation(p, target)})
    per_video = pd.DataFrame(per_video)
    aggregate, intervals, index_arrays = [], [], {}
    for cohort, subset in predictions.groupby("cohort", sort=True):
        sources = sorted(subset.source_video_id.unique())
        indices = np.random.Generator(np.random.PCG64(seed)).integers(0, len(sources), (10000, len(sources)))
        index_arrays[cohort] = indices
        digest = hashlib.sha256(indices.tobytes()).hexdigest()
        for name in PREDICTORS:
            v = per_video[(per_video.cohort == cohort) & (per_video.predictor == name)].set_index("source_video_id").loc[sources]
            g = subset[subset.predictor == name]
            actual, estimated = g[TARGET].to_numpy(), g.prediction.to_numpy()
            q = np.quantile(v.MAE.to_numpy()[indices].mean(axis=1), [.025, .975], method="linear")
            a, p = signs(actual), signs(estimated)
            recalls = [float((p[a == label] == label).mean()) for label in np.unique(a)]
            aggregate.append({"cohort": cohort, "predictor": name, "videos": len(v), "examples": len(g),
                "equal_video_MAE": float(v.MAE.mean()), "equal_video_RMSE": float(v.RMSE.mean()),
                "mean_defined_video_spearman": float(v.spearman.mean()) if v.spearman.notna().any() else None,
                "undefined_video_spearman_count": int(v.spearman.isna().sum()),
                "descriptive_example_spearman": correlation(estimated, actual),
                "sign_accuracy": float((a == p).mean()), "sign_macro_recall_observed_classes": float(np.mean(recalls)),
                "beneficial_moves": int((a == 1).sum()), "neutral_moves": int((a == 0).sum()),
                "harmful_moves": int((a == -1).sum()), "predicted_positive": int((p == 1).sum()),
                "positive_precision": float((a[p == 1] == 1).mean()) if (p == 1).any() else None,
                "positive_recall": float((p[a == 1] == 1).mean()) if (a == 1).any() else None,
                "positive_gain_AUROC": auc(estimated, a == 1),
                "MAE_mean_CI95_lower": float(q[0]), "MAE_mean_CI95_upper": float(q[1])})
        pivot = per_video[per_video.cohort == cohort].pivot(index="source_video_id", columns="predictor", values="MAE").loc[sources]
        delta = (pivot.A-pivot.D).to_numpy()
        bounds = np.quantile(delta[indices].mean(axis=1), [.025, .975], method="linear")
        intervals.append({"cohort": cohort, "comparison": "D versus A", "primary": cohort == "development_oof",
            "positive": "MAE(A) - MAE(D), equal source-video weight", "mean": float(delta.mean()),
            "median": float(np.median(delta)), "CI95_lower": float(bounds[0]), "CI95_upper": float(bounds[1]),
            "improved": int((delta > TOL).sum()), "tied": int((abs(delta) <= TOL).sum()),
            "worsened": int((delta < -TOL).sum()), "resamples": 10000, "seed": seed,
            "rng": "numpy.Generator(PCG64)", "percentile_method": "linear", "resampling_indices_sha256": digest,
            "source_order": sources, "limitation": "Conditional on saved fitted probes; no full training or selection uncertainty."})
    return {"probe_metrics": aggregate}, per_video, {"comparisons": intervals}, index_arrays


def clip_summary(table: pd.DataFrame) -> list[dict]:
    result = []
    for (cohort, policy), g in table[table.policy != "dense"].groupby(["cohort", "policy"], sort=True):
        dense_correct = g[g.dense_correct == 1]
        result.append({"cohort": cohort, "policy": policy, "videos": len(g),
            "feature_error_vs_delta_CE": correlation(g.feature_cosine_error, g.delta_CE),
            "feature_error_vs_absolute_classification_error": correlation(g.feature_cosine_error, 1-g.policy_correct),
            "absolute_classification_error_AUROC": auc(g.feature_cosine_error, g.policy_correct == 0),
            "dense_correct_videos": len(dense_correct),
            "harmful_transitions": int((dense_correct.policy_correct == 0).sum()),
            "harmful_fraction_among_dense_correct": float((dense_correct.policy_correct == 0).mean()) if len(dense_correct) else None,
            "harmful_transition_AUROC_among_dense_correct": auc(dense_correct.feature_cosine_error, dense_correct.policy_correct == 0),
            "prediction_disagreements": int(g.prediction_disagreement.sum()),
            "transitions": {str(k): int(v) for k, v in g.transition.value_counts().items()}})
    return result
