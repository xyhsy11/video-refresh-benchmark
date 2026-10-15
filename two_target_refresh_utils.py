"""Fixed signed-target probes and source-level metrics for saved tables only."""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import task_signal_ablation_utils as previous
from fresh_confirmation_data import strict_json
from research_utils import sha256_file

TOL = 1e-12
ATOL, RTOL = 1e-10, 1e-9
# This is a fixed whitelist, never a selection of numeric columns in a table.
COLUMNS = [
    "candidate_position_normalized", "cached_age_seconds", "remaining_refresh_budget",
    "positions_until_planned_refresh", "current_mean_magnitude_px", "current_p90_magnitude_px",
    "current_mv_available", "interval_mean_available_magnitude_px", "interval_max_available_p90_px",
    "interval_mv_available_fraction", "cached_entropy_nats", "cached_max_probability",
    "cached_top_two_probability_margin", "past_output_entropy_nats", "past_output_max_probability",
    "past_output_top_two_probability_margin",
] + [f"{prefix}_centered_logit_class_{i:02d}" for prefix in ["cached", "past_output"] for i in range(20)]
AGE_COLUMNS = COLUMNS[:4]
KEYS = ["example_id", "video_id", "source_video_id", "t", "last_refresh"]
TARGETS = ["B", "C", "G", "B_averaged", "C_averaged"]
TARGET_COLUMNS = {"B": "early_addition_benefit", "C": "later_removal_cost", "G": "net_gain",
                  "B_averaged": "average_addition_benefit", "C_averaged": "average_removal_cost"}
RF_PARAMETERS = dict(criterion="squared_error", max_depth=4, min_samples_leaf=20,
                     max_features=1.0, bootstrap=True, oob_score=False, random_state=20261020, n_jobs=1,
                     min_samples_split=2, min_weight_fraction_leaf=0., max_leaf_nodes=None,
                     min_impurity_decrease=0., warm_start=False, ccp_alpha=0., max_samples=None,
                     monotonic_cst=None)
FORESTS = [("Direct_G_RF200", "G", 200), ("Direct_G_RF400", "G", 400),
           ("Two_target_RF200x2_forward", "B", 200), ("Two_target_RF200x2_forward", "C", 200),
           ("Two_target_RF200x2_averaged", "B_averaged", 200),
           ("Two_target_RF200x2_averaged", "C_averaged", 200)]
PAIRS = [(f"C_{kind}_two_target_{order}", f"C_{kind}", b, c)
         for kind in ["linear", "nonlinear"]
         for order, b, c in [("forward", "B", "C"), ("averaged", "B_averaged", "C_averaged")]]
PAIRS += [("Two_target_RF200x2_forward", "Two_target_RF200x2_forward", "B", "C"),
          ("Two_target_RF200x2_averaged", "Two_target_RF200x2_averaged", "B_averaged", "C_averaged")]
DIRECT = [("training_mean", "training_mean"), ("training_median", "training_median"),
          ("age_budget_ridge", "age_budget_ridge"), ("C_linear_direct", "C_linear"),
          ("C_nonlinear_direct", "C_nonlinear"), ("Direct_G_RF200", "Direct_G_RF200"),
          ("Direct_G_RF400", "Direct_G_RF400")]
METHODS = ["keep_reference"] + [m for m, _ in DIRECT] + [m for m, *_ in PAIRS]


def design(table: pd.DataFrame, age: bool = False) -> np.ndarray:
    columns = AGE_COLUMNS if age else COLUMNS
    x = table.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    assert x.shape == (len(table), len(columns)) and np.isfinite(x).all()
    return x


def fit_bundle(table: pd.DataFrame, folder: Path) -> dict:
    """Only the caller's explicit fold-training table is accepted."""
    folder.mkdir(parents=True, exist_ok=True)
    x = design(table)
    common = {"training_example_ids": table.example_id.tolist(),
              "training_source_ids": sorted(table.source_video_id.unique()),
              "training_rows": len(table), "input_columns": COLUMNS,
              "training_C_matrix_sha256": hashlib.sha256(x.tobytes()).hexdigest()}
    registry = {**common, "models": [], "constants": {}}
    for target in TARGETS:
        y = table[target].to_numpy(dtype=np.float64)
        registry["constants"][target] = {"training_mean": float(y.mean()), "training_median": float(np.median(y))}
        for nonlinear in [False, True]:
            recipe = "C_nonlinear" if nonlinear else "C_linear"
            state, arrays = previous.fit_array(x, y, nonlinear)
            state.update(common, target=target, recipe=recipe)
            stem = f"{recipe}_{target}"
            if arrays:
                np.savez_compressed(folder/(stem+".npz"), **arrays)
                state.update(array_file=stem+".npz", array_sha256=sha256_file(folder/(stem+".npz")))
            strict_json(folder/(stem+".json"), state)
            registry["models"].append({"recipe": recipe, "target": target, "file": stem+".json",
                                       "sha256": sha256_file(folder/(stem+".json")), "type": "ridge"})
    state, arrays = previous.fit_array(design(table, age=True), table.G.to_numpy(), False)
    assert not arrays
    state.update(common, target="G", recipe="age_budget_ridge", input_columns=AGE_COLUMNS)
    strict_json(folder/"age_budget_ridge_G.json", state)
    registry["models"].append({"recipe": "age_budget_ridge", "target": "G", "file": "age_budget_ridge_G.json",
                               "sha256": sha256_file(folder/"age_budget_ridge_G.json"), "type": "ridge"})
    z, scaler = previous.scale_fit(x)
    for recipe, target, trees in FORESTS:
        forest = RandomForestRegressor(n_estimators=trees, **RF_PARAMETERS)
        forest.fit(z, table[target].to_numpy(dtype=np.float64))
        stem = f"{recipe}_{target}"
        joblib.dump(forest, folder/(stem+".joblib"), compress=3, protocol=5)
        state = {**common, **scaler, "recipe": recipe, "target": target,
                 "parameters": forest.get_params(), "trees": len(forest.estimators_),
                 "node_count": sum(t.tree_.node_count for t in forest.estimators_),
                 "model_file": stem+".joblib", "model_sha256": sha256_file(folder/(stem+".joblib")),
                 "serialized_model_bytes": (folder/(stem+".joblib")).stat().st_size,
                 "input_arithmetic": "float64 training-only population scaling; sklearn tree prediction uses float32 inputs",
                 "separate_target_forest": True}
        strict_json(folder/(stem+".json"), state)
        registry["models"].append({"recipe": recipe, "target": target, "file": stem+".json",
                                   "sha256": sha256_file(folder/(stem+".json")), "type": "forest"})
    strict_json(folder/"registry.json", registry)
    return registry


def load_model(folder: Path, entry: dict) -> tuple[dict, dict, object | None]:
    import json
    assert sha256_file(folder/entry["file"]) == entry["sha256"]
    state = json.loads((folder/entry["file"]).read_text(encoding="utf-8"))
    arrays, forest = {}, None
    if "array_file" in state:
        assert sha256_file(folder/state["array_file"]) == state["array_sha256"]
        with np.load(folder/state["array_file"], allow_pickle=False) as f:
            arrays = {k: f[k].copy() for k in f.files}
    if entry["type"] == "forest":
        assert sha256_file(folder/state["model_file"]) == state["model_sha256"]
        forest = joblib.load(folder/state["model_file"])
    return state, arrays, forest


def predict_bundle(table: pd.DataFrame, folder: Path, registry: dict, cohort: str, fold: int) -> pd.DataFrame:
    parts = []
    def append(recipe: str, target: str, values: np.ndarray) -> None:
        assert values.shape == (len(table),) and np.isfinite(values).all()
        part = table[KEYS].copy()
        part["cohort"], part["fold"], part["recipe"], part["target"] = cohort, fold, recipe, target
        part["actual"], part["prediction"] = table[target].to_numpy(), values
        part["training_median_prediction"] = registry["constants"][target]["training_median"]
        parts.append(part)
    for target in TARGETS:
        for recipe in ["training_mean", "training_median"]:
            append(recipe, target, np.full(len(table), registry["constants"][target][recipe]))
    for e in registry["models"]:
        state, arrays, forest = load_model(folder, e)
        x = table.loc[:, state["input_columns"]].to_numpy(dtype=np.float64)
        if forest is None:
            values = previous.predict_array(x, state, arrays)
        else:
            values = forest.predict((x-state["means"])/state["scales"])
        append(e["recipe"], e["target"], values)
    return pd.concat(parts, ignore_index=True)


def decision_rows(components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """No fitting: construct net decisions and the uncentered MSE identity."""
    parts, error_parts = [], []
    keys = ["cohort", "fold", *KEYS]
    g = components[components.target == "G"].drop_duplicates("example_id").set_index("example_id")
    for method in METHODS:
        if method == "keep_reference":
            result = g[keys[:-len(KEYS)]+KEYS[1:]].reset_index()
            result["actual_G"] = g.actual.to_numpy()
            result["prediction_G"] = 0.
        elif method in dict(DIRECT):
            recipe = dict(DIRECT)[method]
            result = components[(components.recipe == recipe)&(components.target == "G")][keys+["actual", "prediction"]].copy()
            result = result.rename(columns={"actual": "actual_G", "prediction": "prediction_G"})
        else:
            _, recipe, b, c = next(p for p in PAIRS if p[0] == method)
            one = components[(components.recipe == recipe)&(components.target == b)][keys+["actual", "prediction"]]
            two = components[(components.recipe == recipe)&(components.target == c)][keys+["actual", "prediction"]]
            result = one.merge(two, on=keys, suffixes=("_B", "_C"), validate="one_to_one", sort=False)
            result["actual_G"] = g.loc[result.example_id, "actual"].to_numpy()
            np.testing.assert_allclose(result.actual_B-result.actual_C, result.actual_G, atol=TOL, rtol=0)
            result["prediction_G"] = result.prediction_B-result.prediction_C
            eb, ec = result.prediction_B-result.actual_B, result.prediction_C-result.actual_C
            err = result[keys].copy()
            err["method"] = method
            err["squared_error_B"], err["squared_error_C"] = eb**2, ec**2
            err["uncentered_error_cross_product"] = eb*ec
            err["squared_error_G"] = (result.prediction_G-result.actual_G)**2
            np.testing.assert_allclose(err.squared_error_G, eb**2+ec**2-2*eb*ec, atol=ATOL, rtol=RTOL)
            error_parts.append(err)
        result["method"] = method
        result["selected"] = result.prediction_G > TOL
        result["realized_gain"] = result.actual_G*result.selected
        parts.append(result)
    return pd.concat(parts, ignore_index=True), pd.concat(error_parts, ignore_index=True)


def augment_net_quality(components: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Also measure G quality for differences of component predictions and zero."""
    baseline = components[(components.recipe == "training_median")&(components.target == "G")].set_index("example_id")
    additions = []
    for method in ["keep_reference"]+[p[0] for p in PAIRS]:
        g = decisions[decisions.method == method]
        part = g[["cohort", "fold", *KEYS]].copy()
        part["recipe"], part["target"] = method, "G"
        part["actual"], part["prediction"] = g.actual_G.to_numpy(), g.prediction_G.to_numpy()
        part["training_median_prediction"] = baseline.loc[g.example_id, "prediction"].to_numpy()
        additions.append(part)
    return pd.concat([components, *additions], ignore_index=True)


def interval(values: np.ndarray, indices: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all()
    low, high = np.quantile(values[indices].mean(axis=1), [.025, .975], method="linear")
    return {"mean": float(values.mean()), "CI95_lower": float(low), "CI95_upper": float(high)}


def summarize(decisions: pd.DataFrame, components: pd.DataFrame, errors: pd.DataFrame,
              seed: int, bootstrap: int) -> tuple[dict, pd.DataFrame, dict, dict]:
    video_rows, quality_rows, aggregates, quality, comparisons, index_matrices = [], [], [], [], [], {}
    for (cohort, method, source), g in decisions.groupby(["cohort", "method", "source_video_id"], sort=True):
        assert len(g) == 9 and g.example_id.nunique() == 9
        y, selected = g.actual_G.to_numpy(), g.selected.to_numpy(dtype=bool)
        np.testing.assert_array_equal(g.realized_gain, y*selected)
        video_rows.append({"cohort": cohort, "method": method, "source_video_id": source, "opportunities": 9,
            "utility": float((y*selected).mean()), "selected": int(selected.sum()),
            "helpful": int(((y > TOL)&selected).sum()), "harmful": int(((y < -TOL)&selected).sum()),
            "neutral": int(((abs(y) <= TOL)&selected).sum()), "selected_gain_sum": float(y[selected].sum())})
    for (cohort, recipe, target, source), g in components.groupby(["cohort", "recipe", "target", "source_video_id"], sort=True):
        assert len(g) == 9 and g.example_id.nunique() == 9
        y, pred, median = g.actual.to_numpy(), g.prediction.to_numpy(), g.training_median_prediction.to_numpy()
        mae = float(abs(y-pred).mean())
        quality_rows.append({"cohort": cohort, "recipe": recipe, "target": target, "source_video_id": source,
            "MAE": mae, "MSE": float(((y-pred)**2).mean()),
            "median_baseline_MAE": float(abs(y-median).mean()),
            "MAE_improvement_over_training_median": float(abs(y-median).mean()-mae)})
    video, quality_video = pd.DataFrame(video_rows), pd.DataFrame(quality_rows)
    for cohort, group in video.groupby("cohort", sort=True):
        sources = sorted(group.source_video_id.unique())
        indices = np.random.Generator(np.random.PCG64(seed)).integers(0, len(sources), (bootstrap, len(sources)))
        index_matrices[cohort] = indices
        for method, g in group.groupby("method", sort=True):
            g = g.set_index("source_video_id").loc[sources]
            count, n = int(g.selected.sum()), len(g)*9
            aggregates.append({"cohort": cohort, "method": method, "videos": len(g), "opportunities": n,
                "utility": interval(g.utility.to_numpy(), indices), "selected": count,
                "selected_fraction": count/n, "helpful": int(g.helpful.sum()), "harmful": int(g.harmful.sum()),
                "neutral": int(g.neutral.sum()), "helpful_fraction_among_selected": float(g.helpful.sum()/count) if count else None,
                "mean_gain_among_selected": float(g.selected_gain_sum.sum()/count) if count else None,
                "selected_mean_denominator": count, "selected_mean_undefined_reason": None if count else "no selected opportunities"})
        qv = quality_video[quality_video.cohort == cohort]
        for (recipe, target), g in qv.groupby(["recipe", "target"], sort=True):
            g = g.set_index("source_video_id").loc[sources]
            raw = components[(components.cohort == cohort)&(components.recipe == recipe)&(components.target == target)]
            quality.append({"cohort": cohort, "recipe": recipe, "target": target, "videos": len(g),
                **{name: interval(g[name].to_numpy(), indices) for name in ["MAE", "MSE", "MAE_improvement_over_training_median"]},
                "median_baseline_MAE": float(g.median_baseline_MAE.mean()),
                "descriptive_pooled_spearman": previous.prior.correlation(raw.prediction.to_numpy(), raw.actual.to_numpy()),
                "descriptive_positive_target_AUROC": previous.discrimination(raw.prediction.to_numpy(), raw.actual.to_numpy() > TOL)})
        pivot = group.pivot(index="source_video_id", columns="method", values="utility").loc[sources]
        declared = [("Two_target_RF200x2_forward", "Direct_G_RF400", "primary_relative"),
                    ("Two_target_RF200x2_forward", "keep_reference", "required_absolute"),
                    ("Two_target_RF200x2_forward", "Direct_G_RF200", "secondary"),
                    ("Two_target_RF200x2_forward", "C_nonlinear_direct", "secondary"),
                    ("Two_target_RF200x2_forward", "C_linear_direct", "secondary"),
                    ("Two_target_RF200x2_averaged", "Two_target_RF200x2_forward", "attribution_sensitivity"),
                    ("Two_target_RF200x2_averaged", "Direct_G_RF400", "attribution_sensitivity")]
        for candidate, baseline, status in declared:
            delta = (pivot[candidate]-pivot[baseline]).to_numpy()
            comparisons.append({"cohort": cohort, "candidate": candidate, "baseline": baseline,
                "status": status if cohort == "development_oof" else "exploratory_"+status,
                "positive_favors": candidate, **interval(delta, indices), "median": float(np.median(delta)),
                "videos_improved": int((delta > TOL).sum()), "videos_tied": int((abs(delta) <= TOL).sum()),
                "videos_worsened": int((delta < -TOL).sum()), "source_order": sources,
                "resampling_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()})
    err_video = errors.groupby(["cohort", "method", "source_video_id"], sort=True)[
        ["squared_error_B", "squared_error_C", "uncentered_error_cross_product", "squared_error_G"]].mean().reset_index()
    err_aggregate = []
    for (cohort, method), g in err_video.groupby(["cohort", "method"], sort=True):
        eb, ec, cross, eg = [float(g[c].mean()) for c in ["squared_error_B", "squared_error_C", "uncentered_error_cross_product", "squared_error_G"]]
        np.testing.assert_allclose(eg, eb+ec-2*cross, atol=ATOL, rtol=RTOL)
        err_aggregate.append({"cohort": cohort, "method": method, "MSE_B": eb, "MSE_C": ec,
            "mean_uncentered_error_cross_product": cross, "MSE_G": eg,
            "identity_absolute_residual": abs(eg-(eb+ec-2*cross))})
    return {"decision_methods": aggregates, "prediction_quality": quality, "component_error_identity": err_aggregate}, video, {
        "comparisons": comparisons, "resamples": bootstrap, "seed": seed, "rng": "PCG64", "percentile_method": "linear",
        "unit": "whole source video; all nine opportunities; shared indices per cohort",
        "limitations": "conditional on saved fits; reused developmental cohorts; exclude training/model-selection/repeated-research uncertainty; secondary intervals unadjusted"}, {
            "indices": index_matrices, "quality_per_video": quality_video, "error_per_video": err_video}
