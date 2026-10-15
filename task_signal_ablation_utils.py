"""Nested cached/current inputs and fixed float64 ridge probes.

Only the separate privileged builder accepts current RGB-derived logits.
No encoder, video decoder, recognition fitting, or sequential policy is used.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

import task_signal_utils as prior
from task_aware_oracle_utils import cache_indices, feature_logits
from fresh_confirmation_data import strict_json
from research_utils import sha256_file

TOL = 1e-12
SETS = {
    "S": prior.GROUPS["D"],
    "C": prior.GROUPS["D"] + [f"{prefix}_centered_logit_class_{i:02d}"
                             for prefix in ["cached", "past_output"] for i in range(20)],
}
SETS["P"] = SETS["C"] + [f"privileged_current_centered_logit_class_{i:02d}" for i in range(20)]
SETS["A"] = prior.GROUPS["A"]
CONFIGS = [f"{s}-{kind}" for s in ["S", "C", "P"] for kind in ["linear", "nonlinear"]]
BASELINES = ["zero", "training_mean", "training_median", "A-linear"]
PREDICTORS = CONFIGS + BASELINES
REFERENCES = ["keep_reference", "hindsight_positive_gain_reference"]
CONTRASTS = [("S-nonlinear", "S-linear"), ("C-linear", "S-linear"),
             ("C-nonlinear", "S-nonlinear"), ("P-nonlinear", "C-nonlinear")]


def access(name: str) -> dict:
    privileged = name.startswith("P-") or name == REFERENCES[1]
    return {"configuration": name, "privileged": privileged,
            "input_access": "privileged_current_RGB" if name.startswith("P-") else
                "privileged_actual_outcome" if name == REFERENCES[1] else "cached_and_prefix_only"}


def centered(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    assert logits.shape == (20,) and np.isfinite(logits).all()
    return logits-logits.mean()


def cached_observation(video, t: int, head: dict) -> prior.CheapObservation:
    """Read only already refreshed candidates; no feature at candidate t."""
    assert t % 4 in [1, 2, 3]
    r = 4*(t//4)
    past_cache = cache_indices(prior.PERIODIC)[:t]
    return prior.CheapObservation(t, r, r+4, video.cheap.timestamps[:t+1],
        video.cheap.indices[:t+1], video.cheap.motion[:video.cheap.indices[t]+1],
        feature_logits(video.references[[r]], head)[0],
        feature_logits(video.references[past_cache], head))


def cached_inputs(observation: prior.CheapObservation) -> dict:
    """S/C interface: labels, RGB arrays and current logits are not accepted."""
    result = prior.cheap_features(observation)
    vectors = np.r_[centered(observation.cached_logits),
                    centered(observation.past_output_logits.mean(axis=0))]
    result.update(zip(SETS["C"][16:], map(float, vectors)))
    assert list(result) == SETS["C"] and np.isfinite(list(result.values())).all()
    return result


def privileged_inputs(cached: dict, current_logits: np.ndarray) -> dict:
    """P diagnostic only; observing f_t can add an RGB call if periodic is kept."""
    assert list(cached) == SETS["C"]
    return {**cached, **dict(zip(SETS["P"][-20:], map(float, centered(current_logits))))}


def scale_fit(x: np.ndarray) -> tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2 and np.isfinite(x).all()
    means, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    scales = np.where(std <= TOL, 1., std)
    return (x-means)/scales, {"means": means.tolist(), "raw_standard_deviations": std.tolist(),
                            "scales": scales.tolist(), "constant_column_indices": np.flatnonzero(std <= TOL).tolist()}


def kernel(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    assert x.ndim == z.ndim == 2 and x.shape[1] == z.shape[1]
    distance = np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(z*z, axis=1)[None, :] - 2*x@z.T, 0.)
    return np.exp(-distance/x.shape[1])


def fit_array(x: np.ndarray, y: np.ndarray, nonlinear: bool) -> tuple[dict, dict]:
    z, scaler = scale_fit(x)
    y = np.asarray(y, dtype=np.float64)
    assert y.shape == (len(z),) and np.isfinite(y).all()
    mean = float(y.mean())
    state = {**scaler, "intercept": mean, "regularization": 1., "intercept_penalized": False,
             "training_rows": len(z), "input_dimension": z.shape[1], "nonlinear": nonlinear}
    if nonlinear:
        k = kernel(z, z)
        column_mean, row_mean, grand = k.mean(axis=0), k.mean(axis=1), float(k.mean())
        kc = k-column_mean[None, :]-row_mean[:, None]+grand
        dual = np.linalg.solve(kc+np.eye(len(z)), y-mean)
        np.testing.assert_allclose((kc+np.eye(len(z)))@dual, y-mean, atol=1e-10, rtol=1e-9)
        state.update(kernel="exp(-squared_distance/input_dimension)", kernel_grand_mean=grand,
                     centering="training column/grand means; each query row mean over training examples")
        arrays = {"training_features": np.asarray(x, dtype=np.float64),
                  "standardized_training_features": z, "training_targets": y,
                  "kernel_column_means": column_mean, "dual_coefficients": dual}
    else:
        lhs = z.T@z+np.eye(z.shape[1]); rhs = z.T@(y-mean)
        state["coefficients"] = np.linalg.solve(lhs, rhs).tolist()
        np.testing.assert_allclose(lhs@np.asarray(state["coefficients"]), rhs, atol=1e-10, rtol=1e-9)
        arrays = {}
    return state, arrays


def predict_array(x: np.ndarray, state: dict, arrays: dict) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64)-state["means"])/state["scales"]
    if state["nonlinear"]:
        k = kernel(z, arrays["standardized_training_features"])
        centered_k = k-arrays["kernel_column_means"][None, :]-k.mean(axis=1)[:, None]+state["kernel_grand_mean"]
        result = state["intercept"]+centered_k@arrays["dual_coefficients"]
    else:
        result = state["intercept"]+z@np.asarray(state["coefficients"])
    assert np.isfinite(result).all()
    return result


def fit_bundle(table: pd.DataFrame, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    registry = {"training_mean": float(table.refresh_gain.mean()),
                "training_median": float(table.refresh_gain.median()), "models": {}}
    for name in CONFIGS+["A-linear"]:
        input_set, kind = name.split("-")
        columns = SETS[input_set]
        state, arrays = fit_array(table[columns].to_numpy(), table.refresh_gain.to_numpy(), kind == "nonlinear")
        state.update(access(name), input_set=input_set, columns=columns, target="refresh_gain",
                     training_example_ids=table.example_id.tolist(),
                     training_source_ids=sorted(table.source_video_id.unique()))
        stem = name+("_privileged" if state["privileged"] else "")
        if arrays:
            np.savez_compressed(folder/(stem+".npz"), **arrays)
            state.update(array_file=stem+".npz", array_sha256=sha256_file(folder/(stem+".npz")))
        strict_json(folder/(stem+".json"), state)
        registry["models"][name] = {"json": stem+".json", "sha256": sha256_file(folder/(stem+".json")),
                                     "privileged": state["privileged"]}
    strict_json(folder/"registry.json", registry)
    return registry


def load_model(folder: Path, entry: dict) -> tuple[dict, dict]:
    import json
    assert sha256_file(folder/entry["json"]) == entry["sha256"]
    state = json.loads((folder/entry["json"]).read_text(encoding="utf-8"))
    arrays = {}
    if "array_file" in state:
        assert sha256_file(folder/state["array_file"]) == state["array_sha256"]
        with np.load(folder/state["array_file"], allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
    return state, arrays


def predictions(table: pd.DataFrame, folder: Path, registry: dict, cohort: str, fold: int) -> pd.DataFrame:
    parts = []
    for name in PREDICTORS:
        if name in registry["models"]:
            state, arrays = load_model(folder, registry["models"][name])
            values = predict_array(table[state["columns"]].to_numpy(), state, arrays)
        else:
            values = np.full(len(table), 0. if name == "zero" else registry[name])
        g = table[["example_id", "video_id", "source_video_id", "t", "last_refresh", "refresh_gain"]].copy()
        for key, value in {**access(name), "cohort": cohort, "fold": fold}.items():
            g[key] = value
        g["prediction"] = values
        g["earlier_selected"] = values > TOL
        g["realized_gain"] = g.refresh_gain*g.earlier_selected
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def discrimination(scores: np.ndarray, outcomes: np.ndarray) -> dict:
    scores = np.asarray(scores)
    outcomes = np.asarray(outcomes, dtype=bool)
    positives, negatives = int(outcomes.sum()), int((~outcomes).sum())
    reason = "outcome has one class" if not positives or not negatives else "constant score" if np.ptp(scores) <= TOL else None
    return {"positive_events": positives, "negative_events": negatives,
            "AUROC": None if reason else prior.auc(scores, outcomes)["AUROC"], "undefined_reason": reason}


def summarize(raw: pd.DataFrame, seed: int) -> tuple[dict, pd.DataFrame, dict, dict]:
    per_video, aggregate, comparisons, indices_by_role = [], [], [], {}
    for (cohort, name, source), g in raw.groupby(["cohort", "configuration", "source_video_id"], sort=True):
        assert len(g) == 9 and not g.example_id.duplicated().any()
        y, p = g.refresh_gain.to_numpy(), g.prediction.to_numpy()
        selected = p > TOL
        np.testing.assert_allclose(g.realized_gain, y*selected, atol=0, rtol=0)
        per_video.append({**access(name), "cohort": cohort, "source_video_id": source,
            "MAE": float(np.abs(y-p).mean()), "MSE": float(np.mean((y-p)**2)),
            "realized_gain": float(np.mean(y*selected)), "selected_count": int(selected.sum()),
            "helpful_selected": int(((y > TOL)&selected).sum()), "harmful_selected": int(((y < -TOL)&selected).sum()),
            "neutral_selected": int(((abs(y) <= TOL)&selected).sum()),
            "selected_gain_sum": float(y[selected].sum()), "examples": 9})
    unique = raw.drop_duplicates(["cohort", "example_id"])
    for (cohort, source), g in unique.groupby(["cohort", "source_video_id"], sort=True):
        y = g.refresh_gain.to_numpy()
        for name in REFERENCES:
            selected = y > 0 if name == REFERENCES[1] else np.zeros(len(y), dtype=bool)
            per_video.append({**access(name), "cohort": cohort, "source_video_id": source,
                "MAE": None, "MSE": None, "realized_gain": float(np.maximum(y, 0).mean()) if name == REFERENCES[1] else 0.,
                "selected_count": int(selected.sum()), "helpful_selected": int(((y > TOL)&selected).sum()),
                "harmful_selected": 0, "neutral_selected": int(((abs(y) <= TOL)&selected).sum()),
                "selected_gain_sum": float(y[selected].sum()), "examples": 9})
    video = pd.DataFrame(per_video)
    for role, role_table in video.groupby("cohort", sort=True):
        sources = sorted(role_table.source_video_id.unique())
        indices = np.random.Generator(np.random.PCG64(seed)).integers(0, len(sources), (10000, len(sources)))
        indices_by_role[role] = indices
        digest = hashlib.sha256(indices.tobytes()).hexdigest()
        for name, g in role_table.groupby("configuration", sort=True):
            g = g.set_index("source_video_id").loc[sources]
            selected_count = int(g.selected_count.sum())
            n = len(g)*9
            observations = raw[(raw.cohort == role)&(raw.configuration == name)]
            result = {**access(name), "cohort": role, "videos": len(g), "opportunities": n,
                "mean_realized_gain": float(g.realized_gain.mean()), "fraction_selected": selected_count/n,
                "selected_count": selected_count, "helpful_selected": int(g.helpful_selected.sum()),
                "harmful_selected": int(g.harmful_selected.sum()), "neutral_selected": int(g.neutral_selected.sum()),
                "helpful_fraction_selected": float(g.helpful_selected.sum()/selected_count) if selected_count else None,
                "mean_gain_selected": float(g.selected_gain_sum.sum()/selected_count) if selected_count else None,
                "selected_mean_undefined_reason": None if selected_count else "no opportunities selected"}
            for metric in ["MAE", "MSE", "realized_gain"]:
                if g[metric].notna().any():
                    values = g[metric].to_numpy(dtype=np.float64)
                    interval = np.quantile(values[indices].mean(axis=1), [.025, .975], method="linear")
                    result[metric] = {"mean": float(values.mean()), "CI95_lower": float(interval[0]), "CI95_upper": float(interval[1])}
                else:
                    result[metric] = None
            result["spearman"] = prior.correlation(observations.prediction.to_numpy(), observations.refresh_gain.to_numpy()) if len(observations) else {"spearman": None, "undefined_reason": "outcome reference, not a predictor"}
            result["positive_gain_AUROC"] = discrimination(observations.prediction.to_numpy(), observations.refresh_gain.to_numpy() > TOL) if len(observations) else None
            aggregate.append(result)
        for candidate, baseline, metric, primary in [("C-nonlinear", "keep_reference", "realized_gain", True)] + [
                (c, b, m, False) for c, b in CONTRASTS for m in ["MAE", "realized_gain"]]:
            pivot = role_table.pivot(index="source_video_id", columns="configuration", values=metric).loc[sources]
            delta = (pivot[baseline]-pivot[candidate] if metric == "MAE" else pivot[candidate]-pivot[baseline]).to_numpy()
            q = np.quantile(delta[indices].mean(axis=1), [.025, .975], method="linear")
            comparisons.append({"cohort": role, "candidate": candidate, "baseline": baseline, "metric": metric,
                "candidate_privileged": access(candidate)["privileged"], "baseline_privileged": access(baseline)["privileged"],
                "primary": primary and role == "development_oof", "status": "primary" if primary and role == "development_oof" else "exploratory marginal unadjusted",
                "positive": "baseline minus candidate" if metric == "MAE" else "candidate minus baseline",
                "mean": float(delta.mean()), "median": float(np.median(delta)), "CI95_lower": float(q[0]), "CI95_upper": float(q[1]),
                "videos_improved": int((delta > TOL).sum()), "videos_tied": int((abs(delta) <= TOL).sum()),
                "videos_worsened": int((delta < -TOL).sum()), "resamples": 10000, "seed": seed,
                "rng": "PCG64", "percentile_method": "linear", "source_order": sources,
                "resampling_indices_sha256": digest,
                "limitation": "conditional on saved probes; excludes full training/model-selection/repeated-research uncertainty"})
    return {"configurations": aggregate}, video, {"comparisons": comparisons}, indices_by_role
