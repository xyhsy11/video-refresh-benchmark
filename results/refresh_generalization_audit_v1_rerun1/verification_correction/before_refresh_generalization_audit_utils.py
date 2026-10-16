"""Independent saved-model checking and single-move audit arithmetic.

No original experiment prediction/report function is used. Forest predictions
are checked by traversing saved trees after sklearn's float32 input conversion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

TOL = 1e-12
ATOL, RTOL = 1e-10, 1e-9
C_COLUMNS = [
    "candidate_position_normalized", "cached_age_seconds", "remaining_refresh_budget",
    "positions_until_planned_refresh", "current_mean_magnitude_px", "current_p90_magnitude_px",
    "current_mv_available", "interval_mean_available_magnitude_px", "interval_max_available_p90_px",
    "interval_mv_available_fraction", "cached_entropy_nats", "cached_max_probability",
    "cached_top_two_probability_margin", "past_output_entropy_nats", "past_output_max_probability",
    "past_output_top_two_probability_margin",
] + [f"{prefix}_centered_logit_class_{i:02d}" for prefix in ["cached", "past_output"] for i in range(20)]
T_COLUMNS = ([f"history_latest_score_difference_{i:02d}" for i in range(20)]
             + [f"history_previous_score_difference_{i:02d}" for i in range(20)]
             + ["history_has_one_transition", "history_has_two_transitions",
                "history_margin_population_std", "history_top1_switch_rate"])
KEYS = ["example_id", "video_id", "source_video_id", "t", "last_refresh"]
LEARNED = ["direct_RF400", "two_target_RF200x2", "history_RF400"]
METHODS = ["keep_periodic", "random_move_expected", "motion_median_move", *LEARNED]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def frame(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def csv(path: Path, data: pd.DataFrame) -> None:
    data.to_csv(path, index=False, float_format="%.17g")


def cache(schedule: list[int]) -> np.ndarray:
    assert schedule == sorted(set(schedule)) and schedule[0] == 0 and schedule[-1] < 16
    result, current = [], 0
    for t in range(16):
        if t in schedule:
            current = t
        result.append(current)
    return np.array(result)


def cross_entropy(logits: np.ndarray, label: int) -> float:
    maximum = float(np.max(logits))
    return float(maximum + np.log(np.exp(logits-maximum).sum())-logits[label])


def independent_history(table: pd.DataFrame) -> np.ndarray:
    result = pd.DataFrame(0., index=table.index, columns=T_COLUMNS)
    for _, group in table.groupby("source_video_id", sort=False):
        scores, refreshes = [], []
        for idx, row in group.sort_values("t").iterrows():
            r = int(row.last_refresh)
            q = row[C_COLUMNS[16:36]].to_numpy(dtype=float)
            if not refreshes or r != refreshes[-1]:
                scores.append(q); refreshes.append(r)
            else:
                np.testing.assert_array_equal(q, scores[-1])
            assert refreshes == [0, 4, 8][:int(row.t)//4+1]
            k = len(scores)
            margins, tops = [], []
            for s in scores:
                p = np.exp(s-s.max()); p /= p.sum()
                top = np.sort(p)[-2:]
                margins.append(top[1]-top[0]); tops.append(int(s.argmax()))
            values = np.zeros(44)
            if k >= 2:
                values[:20] = scores[-1]-scores[-2]
                values[40] = 1.
                values[42] = np.sqrt(np.mean((np.array(margins)-np.mean(margins))**2))
                values[43] = sum(a != b for a, b in zip(tops, tops[1:]))/(k-1)
            if k == 3:
                values[20:40] = scores[-2]-scores[-3]; values[41] = 1.
            result.loc[idx] = values
    return result.to_numpy()


def walk_forest(forest, standardized: np.ndarray) -> np.ndarray:
    """Independent tree traversal; no call to forest/tree predict or apply."""
    x = np.asarray(standardized, dtype=np.float32)
    answer = np.zeros(len(x), dtype=np.float64)
    for estimator in forest.estimators_:
        tree = estimator.tree_
        nodes = np.zeros(len(x), dtype=np.int64)
        active = np.ones(len(x), dtype=bool)
        while active.any():
            rows = np.flatnonzero(active)
            n = nodes[rows]
            leaves = tree.children_left[n] == -1
            if leaves.any():
                rr = rows[leaves]
                answer[rr] += tree.value[n[leaves], 0, 0]
                active[rr] = False
            rows, n = rows[~leaves], n[~leaves]
            if len(rows):
                left = x[rows, tree.feature[n]] <= tree.threshold[n]
                nodes[rows] = np.where(left, tree.children_left[n], tree.children_right[n])
    return answer/len(forest.estimators_)


def load_forest(folder: Path, filename: str, training: pd.DataFrame) -> tuple[dict, object, dict]:
    state = read(folder/filename)
    assert digest(folder/state["model_file"]) == state["model_sha256"]
    ids = state["training_example_ids"]
    assert len(ids) == len(set(ids)) == len(training)
    assert set(ids) == set(training.example_id)
    assert state["training_source_ids"] == sorted(training.source_video_id.unique())
    columns = state["input_columns"]
    assert columns in [C_COLUMNS, C_COLUMNS+T_COLUMNS]
    ordered = training.set_index("example_id").loc[ids]
    x = ordered[columns].to_numpy(dtype=np.float64)
    means, std = x.mean(0), x.std(0, ddof=0)
    scales = np.where(std <= TOL, 1., std)
    discrepancies = []
    for key, expected in [("means", means), ("raw_standard_deviations", std), ("scales", scales)]:
        discrepancies.append(float(abs(np.asarray(state[key])-expected).max()))
        np.testing.assert_allclose(state[key], expected, atol=TOL, rtol=0)
    forest = joblib.load(folder/state["model_file"])
    assert forest.get_params() == state["parameters"] and forest.n_jobs == 1
    assert forest.random_state == 20261020
    y = ordered[state["target"]].to_numpy()
    maximum = 0.
    for tree, sample in zip(forest.estimators_, forest.estimators_samples_):
        delta = abs(float(tree.tree_.value[0, 0, 0])-float(y[sample].mean()))
        maximum = max(maximum, delta)
        assert delta <= ATOL
    return state, forest, {"training_sources": len(state["training_source_ids"]), "training_rows": len(training),
        "max_scaler_discrepancy": max(discrepancies), "bootstrap_root_checks": len(forest.estimators_),
        "max_bootstrap_root_discrepancy": maximum, "trees": len(forest.estimators_),
        "nodes": sum(e.tree_.node_count for e in forest.estimators_),
        "serialized_bytes": (folder/state["model_file"]).stat().st_size}


def predict(table: pd.DataFrame, state: dict, forest) -> tuple[np.ndarray, float]:
    x = (table[state["input_columns"]].to_numpy(dtype=float)-state["means"])/state["scales"]
    independent = walk_forest(forest, x)
    library = forest.predict(x)
    difference = float(abs(independent-library).max())
    np.testing.assert_allclose(independent, library, atol=TOL, rtol=0)
    return independent, difference


def motion_threshold(training: pd.DataFrame) -> float | None:
    available = training.current_mv_available.to_numpy() == 1
    assert np.isin(training.current_mv_available, [0, 1]).all()
    return float(np.median(training.loc[available, "current_p90_magnitude_px"])) if available.any() else None


def accept_motion(table: pd.DataFrame, threshold: float | None) -> np.ndarray:
    if threshold is None:
        return np.zeros(len(table), dtype=bool)
    return (table.current_mv_available.to_numpy() == 1)&(table.current_p90_magnitude_px.to_numpy() > threshold)


def auc(scores: np.ndarray, positive: np.ndarray) -> dict:
    n1, n0 = int(positive.sum()), int((~positive).sum())
    reason = "outcome has one class" if not n1 or not n0 else "constant scores" if np.ptp(scores) <= TOL else None
    value = None
    if reason is None:
        ranks = pd.Series(scores).rank(method="average").to_numpy()
        value = float((ranks[positive].sum()-n1*(n1+1)/2)/(n1*n0))
    return {"AUROC": value, "positive_events": n1, "negative_events": n0, "undefined_reason": reason}


def ratio(numerator: float, denominator: float) -> tuple[float | None, str | None]:
    return (float(numerator/denominator), None) if denominator else (None, "zero denominator")


def per_source(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, method, source), group in decisions.groupby(["cohort", "method", "source_video_id"], sort=True):
        assert len(group) == group.example_id.nunique() == 9
        y, accept = group.G.to_numpy(), group.accept_probability.to_numpy()
        assert np.isin(accept, [0., .5, 1.]).all()
        np.testing.assert_allclose(group.utility_contribution, y*accept, atol=TOL, rtol=0)
        helpful, harmful, neutral = y > TOL, y < -TOL, abs(y) <= TOL
        selected = float(accept.sum())
        harmful_n = float(accept[harmful].sum())
        missed = float((1-accept[helpful]).sum())
        mean_gain, gain_reason = ratio(float((y*accept).sum()), selected)
        harm_rate, harm_reason = ratio(harmful_n, selected)
        miss_rate, miss_reason = ratio(missed, int(helpful.sum()))
        prediction = group.predicted_G.to_numpy(dtype=float)
        regression = np.isfinite(prediction).all()
        rows.append({"cohort": cohort, "method": method, "source_video_id": source,
            "class": group['class'].iloc[0], "opportunities": 9,
            "expected_random_counts": method == "random_move_expected", "selected": selected,
            "selection_rate": selected/9, "helpful_selected": float(accept[helpful].sum()),
            "harmful_selected": harmful_n, "neutral_selected": float(accept[neutral].sum()),
            "beneficial_total": int(helpful.sum()), "beneficial_unselected": missed,
            "utility": float(np.mean(y*accept)), "selected_gain_sum": float((y*accept).sum()),
            "mean_selected_gain": mean_gain, "mean_selected_gain_reason": gain_reason,
            "harmful_selection_rate": harm_rate, "harmful_selection_rate_reason": harm_reason,
            "missed_benefit_rate": miss_rate, "missed_benefit_rate_reason": miss_reason,
            "MAE": float(abs(y-prediction).mean()) if regression else None,
            "training_median_MAE": float(abs(y-group.training_median_G.to_numpy()).mean()) if regression else None,
            "MAE_improvement_over_training_median": float((abs(y-group.training_median_G.to_numpy())-abs(y-prediction)).mean()) if regression else None,
            "regression_reason": None if regression else "decision-only method;no gain prediction"})
    return pd.DataFrame(rows)


def interval(values: np.ndarray, indices: np.ndarray) -> dict:
    bounds = np.percentile(values[indices].mean(1), [2.5, 97.5], method="linear")
    return {"mean": float(values.mean()), "lower": float(bounds[0]), "upper": float(bounds[1])}


def summarize(decisions: pd.DataFrame, sources: pd.DataFrame, seed: int, repeats: int):
    summaries, comparisons, matrices = [], [], {}
    for cohort, group in sources.groupby("cohort", sort=True):
        ids = sorted(group.source_video_id.unique())
        indices = np.random.Generator(np.random.PCG64(seed)).integers(0, len(ids), (repeats, len(ids)))
        matrices[cohort] = indices
        for method in METHODS:
            rows = group[group.method == method].set_index("source_video_id").loc[ids]
            raw = decisions[(decisions.cohort == cohort)&(decisions.method == method)]
            selected = float(rows.selected.sum())
            total = int(rows.opportunities.sum())
            harmful = float(rows.harmful_selected.sum())
            missed, beneficial = float(rows.beneficial_unselected.sum()), int(rows.beneficial_total.sum())
            gain, gain_reason = ratio(float(rows.selected_gain_sum.sum()), selected)
            harm, harm_reason = ratio(harmful, selected)
            miss, miss_reason = ratio(missed, beneficial)
            utility = interval(rows.utility.to_numpy(), indices)
            regression = method in LEARNED
            scores = raw.predicted_G.to_numpy() if regression else raw.current_p90_magnitude_px.to_numpy()
            subset = np.ones(len(raw), dtype=bool) if regression else raw.current_mv_available.to_numpy() == 1
            discrimination = auc(scores[subset], raw.G.to_numpy()[subset] > TOL) if regression or method == "motion_median_move" else {"AUROC": None, "positive_events": int((raw.G > TOL).sum()), "negative_events": int((raw.G <= TOL).sum()), "undefined_reason": "decision-only baseline;no ranking score"}
            summaries.append({"cohort": cohort, "method": method, "source_videos": len(ids), "opportunities": total,
                "counts_are_analytical_expectations": method == "random_move_expected", "selected": selected,
                "selection_rate": selected/total, "helpful_selected": float(rows.helpful_selected.sum()),
                "harmful_selected": harmful, "neutral_selected": float(rows.neutral_selected.sum()),
                "utility_nats_per_opportunity": utility["mean"], "utility_CI95_lower": utility["lower"], "utility_CI95_upper": utility["upper"],
                "mean_gain_selected": gain, "mean_gain_selected_denominator": selected, "mean_gain_selected_reason": gain_reason,
                "harmful_selection_rate": harm, "harmful_selection_numerator": harmful, "harmful_selection_denominator": selected,
                "harmful_selection_reason": harm_reason, "missed_benefit_rate": miss,
                "missed_benefit_numerator": missed, "missed_benefit_denominator": beneficial, "missed_benefit_reason": miss_reason,
                "source_utility_min": float(rows.utility.min()), "source_utility_q25": float(rows.utility.quantile(.25)),
                "source_utility_median": float(rows.utility.median()), "source_utility_q75": float(rows.utility.quantile(.75)),
                "source_utility_max": float(rows.utility.max()),
                "MAE": float(rows.MAE.mean()) if regression else None,
                "training_median_MAE": float(rows.training_median_MAE.mean()) if regression else None,
                "MAE_improvement_over_training_median": float(rows.MAE_improvement_over_training_median.mean()) if regression else None,
                "regression_reason": None if regression else "decision-only method;regression metrics not applicable",
                "positive_gain_AUROC": discrimination["AUROC"], "AUROC_positive_events": discrimination["positive_events"],
                "AUROC_negative_events": discrimination["negative_events"], "AUROC_reason": discrimination["undefined_reason"],
                "AUROC_scope": "all opportunities;descriptive" if regression else "available-motion opportunities only;descriptive" if method == "motion_median_move" else "not applicable"})
        pivot = group.pivot(index="source_video_id", columns="method", values="utility").loc[ids]
        for j, candidate in enumerate(METHODS):
            for baseline in METHODS[:j]:
                delta = (pivot[candidate]-pivot[baseline]).to_numpy()
                comparisons.append({"cohort": cohort, "candidate": candidate, "baseline": baseline,
                    "comparison_role": "absolute utility" if baseline == "keep_periodic" else "secondary marginal unadjusted",
                    "unit": "nats/opportunity", "sources": len(ids), **interval(delta, indices),
                    "improved": int((delta > TOL).sum()), "tied": int((abs(delta) <= TOL).sum()), "worsened": int((delta < -TOL).sum())})
    return pd.DataFrame(summaries), comparisons, matrices
