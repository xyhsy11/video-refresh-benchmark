"""Signed whole-clip attribution for fixed cached-feature schedules."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from task_aware_oracle_utils import cache_indices, classification_loss

SIGN_TOL = 1e-12
IDENTITY_ATOL = 1e-12
REPLAY_ATOL = 1e-10
REPLAY_RTOL = 1e-10
REFERENCE = [0, 4, 8, 12]
COUNTS = {"A": 4, "B": 5, "C": 4, "D": 3}
ORDERS = {
    "forward": ("early_addition_benefit", "later_removal_cost"),
    "reverse": ("addition_after_removal_benefit", "removal_first_cost"),
    "averaged": ("average_addition_benefit", "average_removal_cost"),
}
CONTINUOUS = ["mean_net_gain", "mean_benefit", "mean_cost", "mean_abs_benefit",
              "mean_abs_cost", "mean_order_interaction", "mean_abs_order_interaction",
              "selected_realized_gain_over_all9"]
RATES = {
    "cancellation_all": ("cancellation_count", "opportunity_count"),
    "cancellation_among_harmful": ("cancellation_count", "harmful_moves"),
    "cancellation_among_beneficial_additions": ("cancellation_count", "beneficial_additions"),
    "benefit_sign_changes": ("benefit_sign_changed_count", "opportunity_count"),
    "cost_sign_changes": ("cost_sign_changed_count", "opportunity_count"),
    "cancellation_order_changes": ("cancellation_order_changed_count", "opportunity_count"),
    "robust_forward_reverse_cancellation_all": ("robust_cancellation_count", "opportunity_count"),
}


def schedules(t: int, r: int) -> dict[str, list[int]]:
    if r not in [4, 8, 12] or t not in range(r-3, r):
        raise ValueError("Use the nine saved candidate-ordinal opportunities only")
    result = {"A": REFERENCE.copy(), "B": sorted(REFERENCE+[t]),
              "C": sorted([p for p in REFERENCE if p != r]+[t]),
              "D": [p for p in REFERENCE if p != r]}
    for name, positions in result.items():
        assert positions == sorted(set(positions)) and positions[0] == 0
        assert len(positions) == COUNTS[name]
    return result


def replay(logits: np.ndarray, positions: list[int], label: int) -> tuple[float, np.ndarray]:
    """Fresh cache-index state per call; frozen affine head precedes mean pool."""
    assert logits.shape == (16, 20) and np.isfinite(logits).all()
    caches = cache_indices(positions)
    output = logits[caches]
    assert output.shape == (16, 20)
    return float(classification_loss(output.mean(axis=0)[None], label)[0]), caches


def sign(x):
    return np.where(np.asarray(x) > SIGN_TOL, 1, np.where(np.asarray(x) < -SIGN_TOL, -1, 0))


def attributions(LA: float, LB: float, LC: float, LD: float) -> dict:
    net = LA-LC
    benefit, cost = LA-LB, LC-LB
    reverse_benefit, reverse_cost = LD-LC, LD-LA
    average_benefit = (benefit+reverse_benefit)/2
    average_cost = (cost+reverse_cost)/2
    np.testing.assert_allclose([benefit-cost, reverse_benefit-reverse_cost,
                               average_benefit-average_cost], net, atol=IDENTITY_ATOL, rtol=0)
    forward_cancel = benefit > SIGN_TOL and net < -SIGN_TOL
    reverse_cancel = reverse_benefit > SIGN_TOL and net < -SIGN_TOL
    return {"net_gain": net, "early_addition_benefit": benefit, "later_removal_cost": cost,
            "removal_first_cost": reverse_cost, "addition_after_removal_benefit": reverse_benefit,
            "average_addition_benefit": average_benefit, "average_removal_cost": average_cost,
            "order_interaction": reverse_benefit-benefit,
            "forward_cancellation": bool(forward_cancel), "reverse_cancellation": bool(reverse_cancel),
            "averaged_cancellation": bool(average_benefit > SIGN_TOL and net < -SIGN_TOL),
            "benefit_sign_changed": bool(sign(benefit) != sign(reverse_benefit)),
            "cost_sign_changed": bool(sign(cost) != sign(reverse_cost)),
            "cancellation_order_changed": bool(forward_cancel != reverse_cancel),
            "robust_forward_reverse_cancellation": bool(forward_cancel and reverse_cancel)}


def check_boundaries(caches: dict, t: int, r: int) -> None:
    end = r+4 if r < 12 else 16
    np.testing.assert_array_equal(np.flatnonzero(caches["A"] != caches["B"]), np.arange(t, r))
    np.testing.assert_array_equal(np.flatnonzero(caches["B"] != caches["C"]), np.arange(r, end))
    np.testing.assert_array_equal(caches["A"][:t], caches["C"][:t])
    for values in caches.values():
        assert values.shape == (16,) and values[0] == 0


def evaluate(videos: list, opportunities: pd.DataFrame, predictions: pd.DataFrame,
             head: dict, classes: list[str]) -> pd.DataFrame:
    from task_aware_oracle_utils import feature_logits
    rows = []
    by_id = {v.entry["video_id"]: v for v in videos}
    prediction_lookup = predictions.set_index("example_id")
    for video_id, group in opportunities.groupby("video_id", sort=False):
        v = by_id[video_id]
        assert len(group) == 9
        logits = feature_logits(v.references, head)
        for e in group.itertuples():
            assert e.source_video_id == v.entry["source_video_id"] and e.label == v.entry["label"]
            y = classes.index(e.label)
            assert e.label_index == y and e.last_refresh == e.replaced_refresh-4
            plans = schedules(e.t, e.replaced_refresh)
            assert plans["A"] == json.loads(e.baseline_positions) and plans["C"] == json.loads(e.early_positions)
            losses, caches = {}, {}
            for name, positions in plans.items():
                losses[name], caches[name] = replay(logits, positions, y)
            check_boundaries(caches, e.t, e.replaced_refresh)
            p = prediction_lookup.loc[e.example_id]
            assert p.video_id == video_id and p.source_video_id == e.source_video_id
            assert p.t == e.t and p.last_refresh == e.last_refresh
            assert p.refresh_gain == e.refresh_gain
            assert p.earlier_selected == (p.prediction > SIGN_TOL)
            values = attributions(losses["A"], losses["B"], losses["C"], losses["D"])
            np.testing.assert_allclose(values["net_gain"], e.refresh_gain, atol=REPLAY_ATOL, rtol=REPLAY_RTOL)
            delta = abs(values["net_gain"]-e.refresh_gain)
            relative = delta/abs(e.refresh_gain) if abs(e.refresh_gain) > SIGN_TOL else None
            rows.append({"cohort": e.cohort, "example_id": e.example_id, "video_id": video_id,
                "source_video_id": e.source_video_id, "class": e.label, "label_index": y,
                "t": e.t, "r": e.replaced_refresh, "last_refresh": e.last_refresh,
                "t_decoded_frame_index": v.cheap.indices[e.t],
                "r_decoded_frame_index": v.cheap.indices[e.replaced_refresh],
                **{f"schedule_{name}": json.dumps(positions) for name, positions in plans.items()},
                **{f"loss_{name}": losses[name] for name in plans},
                **{f"simulated_refreshes_{name}": len(plans[name]) for name in plans},
                **{f"feature_outputs_{name}": 16 for name in plans},
                **{f"cache_indices_{name}": json.dumps(caches[name].tolist()) for name in plans},
                "original_saved_target": e.refresh_gain, **values,
                "absolute_target_discrepancy": delta, "relative_target_discrepancy": relative,
                "relative_discrepancy_reason": None if relative is not None else "original target within sign tolerance of zero",
                "C_nonlinear_prediction": float(p.prediction), "C_nonlinear_selected": bool(p.earlier_selected),
                "prediction_cohort": p.cohort, "prediction_fold": int(p.fold)})
    return pd.DataFrame(rows)


def video_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, source), all_rows in table.groupby(["cohort", "source_video_id"], sort=True):
        assert len(all_rows) == 9
        realized = float((all_rows.net_gain*all_rows.C_nonlinear_selected).mean())
        for scope in ["all", "C_nonlinear_selected"]:
            group = all_rows if scope == "all" else all_rows[all_rows.C_nonlinear_selected]
            for order, (benefit, cost) in ORDERS.items():
                values = group[benefit].to_numpy()
                n = len(group)
                rows.append({"cohort": cohort, "source_video_id": source,
                    "video_id": all_rows.video_id.iloc[0], "class": all_rows["class"].iloc[0],
                    "scope": scope, "order": order, "opportunity_count": n,
                    "harmful_moves": int((group.net_gain < -SIGN_TOL).sum()),
                    "helpful_moves": int((group.net_gain > SIGN_TOL).sum()),
                    "neutral_moves": int((abs(group.net_gain) <= SIGN_TOL).sum()),
                    "beneficial_additions": int((values > SIGN_TOL).sum()),
                    "cancellation_count": int(((values > SIGN_TOL)&(group.net_gain < -SIGN_TOL)).sum()),
                    "benefit_sign_changed_count": int(group.benefit_sign_changed.sum()),
                    "cost_sign_changed_count": int(group.cost_sign_changed.sum()),
                    "cancellation_order_changed_count": int(group.cancellation_order_changed.sum()),
                    "robust_cancellation_count": int(group.robust_forward_reverse_cancellation.sum()),
                    "mean_net_gain": float(group.net_gain.mean()) if n else None,
                    "mean_benefit": float(group[benefit].mean()) if n else None,
                    "mean_cost": float(group[cost].mean()) if n else None,
                    "mean_abs_benefit": float(abs(group[benefit]).mean()) if n else None,
                    "mean_abs_cost": float(abs(group[cost]).mean()) if n else None,
                    "mean_order_interaction": float(group.order_interaction.mean()) if n else None,
                    "mean_abs_order_interaction": float(abs(group.order_interaction).mean()) if n else None,
                    "selected_realized_gain_over_all9": realized})
    return pd.DataFrame(rows)


def percentile(valid_values: np.ndarray, total: int, reason: str) -> dict:
    finite = valid_values[np.isfinite(valid_values)]
    bounds = np.quantile(finite, [.025, .975], method="linear") if len(finite) else [None, None]
    return {"CI95_lower": float(bounds[0]) if bounds[0] is not None else None,
            "CI95_upper": float(bounds[1]) if bounds[1] is not None else None,
            "valid_resamples": len(finite), "undefined_resamples": total-len(finite),
            "interval_undefined_reason": None if len(finite) else reason}


def summarize(video: pd.DataFrame, seed: int, resamples: int) -> tuple[dict, dict, dict]:
    aggregates, intervals, index_arrays = [], [], {}
    for cohort, part in video.groupby("cohort", sort=True):
        sources = sorted(part.source_video_id.unique())
        indices = np.random.Generator(np.random.PCG64(seed)).integers(0, len(sources), (resamples, len(sources)))
        index_arrays[cohort] = indices
        digest = hashlib.sha256(indices.tobytes()).hexdigest()
        for (scope, order), g in part.groupby(["scope", "order"], sort=True):
            g = g.set_index("source_video_id").loc[sources]
            identity = {"cohort": cohort, "scope": scope, "order": order}
            result = {**identity, "source_videos": len(g),
                "sources_with_opportunities": int((g.opportunity_count > 0).sum()),
                "opportunities": int(g.opportunity_count.sum()), "harmful_moves": int(g.harmful_moves.sum()),
                "helpful_moves": int(g.helpful_moves.sum()), "neutral_moves": int(g.neutral_moves.sum()),
                "beneficial_additions": int(g.beneficial_additions.sum()), "cancellation_count": int(g.cancellation_count.sum()),
                "rates": {}, "continuous": {}}
            for name, (numerator, denominator) in RATES.items():
                num = g[numerator].to_numpy(); den = g[denominator].to_numpy()
                summed_num, summed_den = num[indices].sum(axis=1), den[indices].sum(axis=1)
                bootstrap = np.divide(summed_num, summed_den, out=np.full(resamples, np.nan), where=summed_den > 0)
                value = float(num.sum()/den.sum()) if den.sum() else None
                statistics = {"numerator": int(num.sum()), "denominator": int(den.sum()), "value": value,
                              "undefined_reason": None if den.sum() else "zero conditional denominator",
                              **percentile(bootstrap, resamples, "zero denominator in every resample")}
                result["rates"][name] = statistics
                intervals.append({**identity, "metric": name, "kind": "ratio_of_counts",
                    "primary": scope == "all" and order == "forward" and name == "cancellation_all",
                    **statistics, "resampling_indices_sha256": digest})
            for name in CONTINUOUS:
                x = g[name].to_numpy(dtype=np.float64)
                matrix = x[indices]
                valid_counts = np.isfinite(matrix).sum(axis=1)
                bootstrap = np.divide(np.nansum(matrix, axis=1), valid_counts,
                    out=np.full(resamples, np.nan), where=valid_counts > 0)
                statistics = {"value": float(np.nanmean(x)) if np.isfinite(x).any() else None,
                              "defined_sources": int(np.isfinite(x).sum()),
                              "undefined_reason": None if np.isfinite(x).any() else "no sources with defined selected mean",
                              **percentile(bootstrap, resamples, "no defined source means in every resample")}
                result["continuous"][name] = statistics
                intervals.append({**identity, "metric": name, "kind": "equal_video_continuous",
                                  "primary": False, **statistics, "resampling_indices_sha256": digest})
            aggregates.append(result)
    return {"rows": aggregates}, {"seed": seed, "resamples": resamples, "rng": "PCG64",
        "percentile_method": "linear", "scope": "descriptive conditional diagnostics,not independent confirmation",
        "conditional_rates": "numerator/denominator recomputed inside each whole-video resample;invalid ratios excluded with counts",
        "rows": intervals}, index_arrays
