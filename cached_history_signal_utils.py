"""Observation-only cached history, fixed RF400 probes and source metrics."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import two_target_refresh_utils as previous
from fresh_confirmation_data import strict_json
from research_utils import sha256_file

TOL, ATOL, RTOL = 1e-12, 1e-10, 1e-9
C_COLUMNS = previous.COLUMNS.copy()
CACHED_COLUMNS, PAST_COLUMNS = C_COLUMNS[16:36], C_COLUMNS[36:56]
T_COLUMNS = ([f"history_latest_score_difference_{i:02d}" for i in range(20)]
             + [f"history_previous_score_difference_{i:02d}" for i in range(20)]
             + ["history_has_one_transition", "history_has_two_transitions",
                "history_margin_population_std", "history_top1_switch_rate"])
AUGMENTED_COLUMNS = C_COLUMNS+T_COLUMNS
KEYS = previous.KEYS
METHODS = ["C_RF400", "C_plus_T_RF400", "keep_reference", "training_mean", "training_median"]


def margin_and_top(scores: np.ndarray) -> tuple[float, int]:
    """Stable softmax; np.argmax chooses the lowest class index in a tie."""
    probability = np.exp(scores-np.max(scores))
    probability /= probability.sum()
    top_two = np.partition(probability, -2)[-2:]
    return float(top_two.max()-top_two.min()), int(np.argmax(scores))


def assemble_history(scores: np.ndarray, margins: np.ndarray, top: np.ndarray,
                     output: np.ndarray) -> np.ndarray:
    """Only already-available distinct refresh vectors/statistics are accepted."""
    k = len(scores)
    output.fill(0.)
    if k >= 2:
        output[:20] = scores[-1]-scores[-2]
        output[40] = 1.
        output[42] = np.std(margins, ddof=0)
        output[43] = np.count_nonzero(top[1:] != top[:-1])/(k-1)
    if k >= 3:
        output[20:40] = scores[-2]-scores[-3]
        output[41] = 1.
    return output


def features_from_available_history(scores: np.ndarray) -> np.ndarray:
    """Pure observation-only reference builder, no video/target/future argument."""
    pairs = [margin_and_top(q) for q in scores]
    return assemble_history(scores, np.array([p[0] for p in pairs]),
                            np.array([p[1] for p in pairs]), np.zeros(44, dtype=np.float64))


class CachedHistory:
    """Incremental per-video state; repeated cache outputs do not add events.

    Inputs are prevalidated outside CPU timing. The hot path has no assertions,
    target access or future-row access. Returned T is a view, consumed immediately.
    """
    __slots__ = ("scores", "margins", "top", "refreshes", "summary", "count")

    def __init__(self) -> None:
        self.scores = np.zeros((3, 20), dtype=np.float64)
        self.margins = np.zeros(3, dtype=np.float64)
        self.top = np.zeros(3, dtype=np.int64)
        self.refreshes = np.full(3, -1, dtype=np.int64)
        self.summary = np.zeros(44, dtype=np.float64)
        self.count = 0

    def advance(self, last_refresh: int, cached_scores: np.ndarray) -> np.ndarray:
        if self.count and last_refresh == self.refreshes[self.count-1]:
            return self.summary
        i = self.count
        self.scores[i] = cached_scores
        self.refreshes[i] = last_refresh
        self.margins[i], self.top[i] = margin_and_top(cached_scores)
        self.count += 1
        return assemble_history(self.scores[:self.count], self.margins[:self.count],
                                self.top[:self.count], self.summary)

    def memory_accounting(self) -> dict:
        arrays = [self.scores, self.margins, self.top, self.refreshes, self.summary]
        return {"ndarray_payload_bytes": sum(a.nbytes for a in arrays),
                "python_counter_bytes": sys.getsizeof(self.count),
                "python_object_and_arrays_sizeof_estimate_bytes": sys.getsizeof(self)+sys.getsizeof(self.count)+sum(sys.getsizeof(a) for a in arrays),
                "maximum_distinct_scores": 3, "history_score_float64_values": 60,
                "cached_T_float64_values": 44, "retained_margin_float64_values": 3,
                "retained_top1_and_refresh_int64_values": 6,
                "transient_augmented_input_bytes": 100*8,
                "incremental_input_bytes_vs_56_columns": 44*8,
                "exclusions": "Python allocator/transient arrays/RF model/workspace not fully captured by sizeof;model sizes reported separately"}


def augment_observations(observations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Chronological reconstruction, with target-free schema and fresh video state."""
    columns = ["cohort", *KEYS, *C_COLUMNS]
    data = observations.loc[:, columns]
    assert data.example_id.is_unique
    added, algebra_residual, duplicate_residual, independent_residual = [], 0., 0., 0.
    events = 0
    for (_, _), group in data.groupby(["source_video_id", "video_id"], sort=False):
        history = CachedHistory()
        assert len(group) == 9 and sorted(group.t) == [1, 2, 3, 5, 6, 7, 9, 10, 11]
        for _, row in group.sort_values("t").iterrows():
            t, r = int(row.t), int(row.last_refresh)
            assert r == (t//4)*4 and r in [0, 4, 8]
            cached = row[CACHED_COLUMNS].to_numpy(dtype=np.float64)
            assert np.isfinite(cached).all() and abs(cached.mean()) <= ATOL
            if history.count and r == history.refreshes[history.count-1]:
                error = float(abs(cached-history.scores[history.count-1]).max())
                duplicate_residual = max(duplicate_residual, error)
                np.testing.assert_array_equal(cached, history.scores[history.count-1])
            else:
                events += 1
            result = history.advance(r, cached).copy()
            expected_count = r//4+1
            assert history.count == expected_count
            np.testing.assert_array_equal(history.refreshes[:history.count], [0, 4, 8][:expected_count])
            independent = features_from_available_history(history.scores[:history.count])
            independent_residual = max(independent_residual, float(abs(result-independent).max()))
            np.testing.assert_allclose(result, independent, atol=TOL, rtol=0)
            past = row[PAST_COLUMNS].to_numpy(dtype=np.float64)
            if expected_count == 1:
                reconstructed = history.scores[0]
            elif expected_count == 2:
                reconstructed = (4*history.scores[0]+(t-4)*history.scores[1])/t
                recovered = (t*past-(t-4)*cached)/4
                np.testing.assert_allclose(recovered, history.scores[0], atol=ATOL, rtol=RTOL)
            else:
                reconstructed = (4*history.scores[0]+4*history.scores[1]+(t-8)*history.scores[2])/t
            algebra_residual = max(algebra_residual, float(abs(past-reconstructed).max()))
            np.testing.assert_allclose(past, reconstructed, atol=ATOL, rtol=RTOL)
            added.append({"example_id": row.example_id, "history_length": history.count,
                          **dict(zip(T_COLUMNS, map(float, result)))})
        assert history.count == 3
    result = data.merge(pd.DataFrame(added), on="example_id", validate="one_to_one", sort=False)
    assert len(result.columns) == len(columns)+45 and np.isfinite(result[AUGMENTED_COLUMNS].to_numpy()).all()
    return result, {"distinct_history_updates": events, "opportunities": len(data),
        "max_duplicate_cached_score_discrepancy": duplicate_residual,
        "max_weighted_past_mean_algebra_discrepancy": algebra_residual,
        "max_incremental_vs_independent_T_discrepancy": independent_residual,
        "no_state_carry_between_videos": True}


def information_fixture() -> dict:
    q0, q4 = np.zeros(20), np.zeros(20)
    q0[0], q4[1] = 2., 4.
    q0 -= q0.mean(); q4 -= q4.mean()
    q8 = q4.copy()
    def old_row(history):
        past = (4*history[0]+4*history[1]+history[2])/9
        row = np.zeros(56)
        row[:4] = [9/15, 1., 1., 3.]
        row[10:16] = previous.previous.prior.confidence(q8)+previous.previous.prior.confidence(past)
        row[16:36], row[36:56] = q8, past
        return row
    original, swapped = np.stack([q0, q4, q8]), np.stack([q4, q0, q8])
    before, after = old_row(original), old_row(swapped)
    np.testing.assert_array_equal(before, after)
    t0, t1 = features_from_available_history(original), features_from_available_history(swapped)
    assert not np.array_equal(t0, t1)
    return {"software_fixture_only": True, "candidate_t": 9, "original_history": original.tolist(),
            "swapped_history": swapped.tolist(), "old_C_unchanged": True,
            "max_old_C_difference": float(abs(before-after).max()), "T_changes": True,
            "changed_T_columns": [c for c, a, b in zip(T_COLUMNS, t0, t1) if abs(a-b) > TOL],
            "original_T": t0.tolist(), "swapped_T": t1.tolist()}


def fit_bundle(training: pd.DataFrame, folder: Path, baseline_folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    baseline_path = baseline_folder/"Direct_G_RF400_G.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["training_example_ids"] == training.example_id.tolist()
    assert baseline["training_source_ids"] == sorted(training.source_video_id.unique())
    for name in [baseline_path.name, baseline["model_file"]]:
        shutil.copyfile(baseline_folder/name, folder/name)
    x = training[AUGMENTED_COLUMNS].to_numpy(dtype=np.float64)
    z, scaler = previous.previous.scale_fit(x)
    for key in ["means", "scales", "raw_standard_deviations"]:
        np.testing.assert_allclose(scaler[key][:56], baseline[key], atol=TOL, rtol=0)
    forest = RandomForestRegressor(**baseline["parameters"])
    forest.fit(z, training.G.to_numpy(dtype=np.float64))
    joblib.dump(forest, folder/"C_plus_T_RF400.joblib", compress=3, protocol=5)
    state = {**scaler, "input_columns": AUGMENTED_COLUMNS, "target": "G", "parameters": forest.get_params(),
        "training_example_ids": training.example_id.tolist(), "training_source_ids": sorted(training.source_video_id.unique()),
        "training_rows": len(training), "trees": len(forest.estimators_),
        "node_count": sum(t.tree_.node_count for t in forest.estimators_), "model_file": "C_plus_T_RF400.joblib",
        "model_sha256": sha256_file(folder/"C_plus_T_RF400.joblib"),
        "serialized_model_bytes": (folder/"C_plus_T_RF400.joblib").stat().st_size}
    strict_json(folder/"C_plus_T_RF400.json", state)
    registry = {"training_mean": float(training.G.mean()), "training_median": float(training.G.median()),
                "models": {"C_RF400": {"file": baseline_path.name, "sha256": sha256_file(folder/baseline_path.name), "type": "forest"},
                           "C_plus_T_RF400": {"file": "C_plus_T_RF400.json", "sha256": sha256_file(folder/"C_plus_T_RF400.json"), "type": "forest"}},
                "baseline_refitted": False}
    strict_json(folder/"registry.json", registry)
    return registry


def predictions(table: pd.DataFrame, folder: Path, registry: dict, cohort: str, fold: int) -> pd.DataFrame:
    parts = []
    for method in METHODS:
        if method in registry["models"]:
            state, _, forest = previous.load_model(folder, registry["models"][method])
            values = forest.predict((table[state["input_columns"]].to_numpy(dtype=np.float64)-state["means"])/state["scales"])
        else:
            values = np.full(len(table), 0. if method == "keep_reference" else registry[method])
        part = table[[*KEYS, "history_length", "G"]].copy()
        part["cohort"], part["fold"], part["method"] = cohort, fold, method
        part["prediction"] = values
        part["training_median_prediction"] = registry["training_median"]
        part["selected"] = values > TOL
        part["realized_gain"] = part.G*part.selected
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def summarize(raw: pd.DataFrame, seed: int, resamples: int) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict, dict]:
    video_rows, history_rows, aggregate, comparisons, matrices = [], [], [], [], {}
    for (role, method, source), group in raw.groupby(["cohort", "method", "source_video_id"], sort=True):
        assert len(group) == 9 and group.example_id.nunique() == 9
        for length in [0, 1, 2, 3]:
            g = group if length == 0 else group[group.history_length == length]
            assert len(g) == (9 if length == 0 else 3)
            y, p, selected = g.G.to_numpy(), g.prediction.to_numpy(), g.selected.to_numpy(dtype=bool)
            np.testing.assert_array_equal(g.realized_gain, y*selected)
            row = {"cohort": role, "method": method, "source_video_id": source, "history_length": length,
                "opportunities": len(g), "MAE": float(abs(y-p).mean()), "MSE": float(((y-p)**2).mean()),
                "MAE_improvement_over_training_median": float(abs(y-g.training_median_prediction.to_numpy()).mean()-abs(y-p).mean()),
                "utility": float((y*selected).mean()), "selected": int(selected.sum()),
                "helpful": int(((y > TOL)&selected).sum()), "harmful": int(((y < -TOL)&selected).sum()),
                "neutral": int(((abs(y) <= TOL)&selected).sum()), "selected_gain_sum": float(y[selected].sum())}
            (video_rows if length == 0 else history_rows).append(row)
    video, history = pd.DataFrame(video_rows), pd.DataFrame(history_rows)
    full = pd.concat([video, history], ignore_index=True)
    for role, role_table in full.groupby("cohort", sort=True):
        sources = sorted(role_table.source_video_id.unique())
        indices = np.random.Generator(np.random.PCG64(seed)).integers(0, len(sources), (resamples, len(sources)))
        matrices[role] = indices
        for (length, method), g in role_table.groupby(["history_length", "method"], sort=True):
            g = g.set_index("source_video_id").loc[sources]
            obs = raw[(raw.cohort == role)&(raw.method == method)]
            if length: obs = obs[obs.history_length == length]
            count = int(g.selected.sum())
            aggregate.append({"cohort": role, "method": method, "history_length": int(length), "videos": len(g),
                "opportunities": int(g.opportunities.sum()), "selected": count, "helpful": int(g.helpful.sum()),
                "harmful": int(g.harmful.sum()), "neutral": int(g.neutral.sum()),
                "mean_selected_gain": float(g.selected_gain_sum.sum()/count) if count else None,
                "mean_selected_gain_denominator": count, "undefined_selected_reason": None if count else "no selected opportunities",
                **{m: previous.interval(g[m].to_numpy(), indices) for m in ["MAE", "MSE", "MAE_improvement_over_training_median", "utility"]},
                "descriptive_pooled_spearman": previous.previous.prior.correlation(obs.prediction.to_numpy(), obs.G.to_numpy()),
                "positive_gain_AUROC": previous.previous.discrimination(obs.prediction.to_numpy(), obs.G.to_numpy() > TOL)})
        for length in [0, 1, 2, 3]:
            g = role_table[role_table.history_length == length]
            for metric, baseline in [("utility", "C_RF400"), ("utility", "keep_reference"), ("MAE", "C_RF400")]:
                pivot = g.pivot(index="source_video_id", columns="method", values=metric).loc[sources]
                delta = (pivot[baseline]-pivot.C_plus_T_RF400 if metric == "MAE" else pivot.C_plus_T_RF400-pivot[baseline]).to_numpy()
                comparisons.append({"cohort": role, "history_length": length, "candidate": "C_plus_T_RF400", "baseline": baseline,
                    "metric": metric, "primary": role == "development_oof" and length == 0 and metric == "utility" and baseline == "C_RF400",
                    "status": "primary" if role == "development_oof" and length == 0 and metric == "utility" and baseline == "C_RF400" else "secondary marginal unadjusted",
                    **previous.interval(delta, indices), "videos_improved": int((delta > TOL).sum()),
                    "videos_tied": int((abs(delta) <= TOL).sum()), "videos_worsened": int((delta < -TOL).sum()),
                    "source_order": sources})
    return {"history_length_zero_means": "all nine opportunities;valid observation histories have lengths1,2,3", "metrics": aggregate}, video, history, {"comparisons": comparisons, "seed": seed,
        "resamples": resamples, "rng": "PCG64", "percentile_method": "linear", "unit": "whole source;all9 opportunities;shared indices within cohort",
        "limits": "fixed fitted models,reused developmental cohorts;excludes full training/model-selection/repeated-research uncertainty;secondary intervals unadjusted"}, matrices
