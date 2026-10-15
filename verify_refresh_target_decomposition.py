"""Independent cache-copy, loss, identity and grouped-bootstrap verification."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_task_signals as original
import recognition_baseline_data as data
import refresh_target_decomposition_utils as u
from research_utils import ROOT, sha256_file, now
from verify_task_signals import same


def explicit_replay(features: np.ndarray, positions: list[int], head: dict, label: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Independent loop: reset, copy latest feature, mean pool, affine head, CE."""
    assert positions == sorted(set(positions)) and positions[0] == 0
    latest = None
    outputs, caches = [], []
    for t in range(16):
        if t in positions:
            latest = t
        assert latest is not None
        outputs.append(features[latest].copy())
        caches.append(latest)
    outputs = np.stack(outputs)
    pooled = outputs.mean(axis=0, dtype=np.float64)
    normalized = (pooled-np.array(head["means"]))/np.array(head["scales"])
    logits = np.array(head["weights"])@normalized+np.array(head["bias"])
    peak = logits.max()
    loss = float(peak+np.log(np.exp(logits-peak).sum())-logits[label])
    for t, cached in enumerate(caches):
        np.testing.assert_array_equal(outputs[t], features[cached])
    return loss, np.array(caches), outputs


def audit(recognition: Path, prior: Path, output: Path, p: dict) -> dict:
    import analyze_refresh_target_decomposition as a
    manifest, head, hashes = a.identities(recognition, prior)
    assert hashes == p["input_sha256"]
    assert {n: sha256_file(ROOT/n) for n in p["source_sha256"]} == p["source_sha256"]
    original.preserved(p)
    targets, predictions = a.load_opportunities(prior, manifest)
    table = pd.read_csv(output/"per_opportunity_decomposition.csv", float_precision="round_trip")
    assert table.groupby("cohort").size().to_dict() == {"development": 900, "exploratory": 360}
    assert table.example_id.nunique() == 1260 and table.groupby("source_video_id").size().eq(9).all()
    assert set(table.example_id) == set(targets.example_id)
    gt = targets.set_index("example_id").loc[table.example_id]
    for field, original_name in [("video_id", "video_id"), ("source_video_id", "source_video_id"),
                                 ("cohort", "cohort"), ("t", "t"), ("r", "replaced_refresh"),
                                 ("class", "label"), ("label_index", "label_index"),
                                 ("original_saved_target", "refresh_gain")]:
        np.testing.assert_array_equal(table[field], gt[original_name])
    prediction = predictions.set_index("example_id").loc[table.example_id]
    np.testing.assert_array_equal(table.C_nonlinear_prediction, prediction.prediction)
    np.testing.assert_array_equal(table.C_nonlinear_selected, prediction.earlier_selected)
    algebra_errors, replay_errors = [], []
    with a.frozen_analysis_only():
        for role in ["development", "exploratory"]:
            videos = {v.entry["video_id"]: v for v in data.load_role(manifest["videos"], role)}
            for row in table[table.cohort == role].itertuples():
                v = videos[row.video_id]
                plans = u.schedules(row.t, row.r)
                caches = {}
                losses = []
                for name, positions in plans.items():
                    assert json.loads(getattr(row, f"schedule_{name}")) == positions
                    assert getattr(row, f"simulated_refreshes_{name}") == u.COUNTS[name]
                    assert getattr(row, f"feature_outputs_{name}") == 16
                    independent, cache, outputs = explicit_replay(v.references, positions, head, row.label_index)
                    assert outputs.shape == (16, 512)
                    np.testing.assert_array_equal(cache, json.loads(getattr(row, f"cache_indices_{name}")))
                    np.testing.assert_allclose(independent, getattr(row, f"loss_{name}"),
                                               atol=u.REPLAY_ATOL, rtol=u.REPLAY_RTOL)
                    replay_errors.append(abs(independent-getattr(row, f"loss_{name}")))
                    caches[name] = cache
                    losses.append(getattr(row, f"loss_{name}"))
                u.check_boundaries(caches, row.t, row.r)
                recomputed = u.attributions(*losses)
                for name, value in recomputed.items():
                    if isinstance(value, bool):
                        assert getattr(row, name) == value
                    else:
                        np.testing.assert_allclose(getattr(row, name), value, atol=u.IDENTITY_ATOL, rtol=0)
                for benefit, cost in u.ORDERS.values():
                    algebra_errors.append(abs(getattr(row, benefit)-getattr(row, cost)-row.net_gain))
    np.testing.assert_allclose(table.net_gain, table.original_saved_target, atol=u.REPLAY_ATOL, rtol=u.REPLAY_RTOL)
    absolute = abs(table.net_gain-table.original_saved_target)
    relative_mask = abs(table.original_saved_target) > u.SIGN_TOL
    relative = absolute[relative_mask]/abs(table.original_saved_target[relative_mask])
    np.testing.assert_allclose(table.absolute_target_discrepancy, absolute, atol=0, rtol=0)
    np.testing.assert_allclose(table.loc[relative_mask, "relative_target_discrepancy"], relative, atol=1e-15, rtol=1e-12)
    assert table.loc[~relative_mask, "relative_target_discrepancy"].isna().all()
    video = u.video_summary(table)
    pd.testing.assert_frame_equal(video, pd.read_csv(output/"per_video_summary.csv", float_precision="round_trip"),
                                 check_exact=False, atol=1e-12, rtol=1e-10)
    aggregate, intervals, indices = u.summarize(video, p["seed"], p["bootstrap"]["resamples"])
    same(aggregate, data.read(output/"aggregate_metrics.json"))
    same(intervals, data.read(output/"bootstrap_intervals.json"))
    # Independently reconstruct every per-video count and signed mean from the opportunity CSV.
    for row in video.itertuples():
        g = table[(table.cohort == row.cohort)&(table.source_video_id == row.source_video_id)]
        if row.scope != "all":
            g = g[g.C_nonlinear_selected]
        benefit, cost = u.ORDERS[row.order]
        cancelled = (g[benefit] > u.SIGN_TOL)&(g.net_gain < -u.SIGN_TOL)
        assert len(g) == row.opportunity_count and int(cancelled.sum()) == row.cancellation_count
        assert int((g.net_gain < -u.SIGN_TOL).sum()) == row.harmful_moves
        assert int((g[benefit] > u.SIGN_TOL).sum()) == row.beneficial_additions
        if len(g):
            np.testing.assert_allclose([g[benefit].mean(), g[cost].mean(),
                abs(g[benefit]).mean(), abs(g[cost]).mean(), g.net_gain.mean()],
                [row.mean_benefit, row.mean_cost, row.mean_abs_benefit, row.mean_abs_cost, row.mean_net_gain], atol=1e-12)
        else:
            assert np.isnan(row.mean_benefit) and np.isnan(row.mean_cost)
    # Ratio numerators and denominators are recomputed in each resample, including zero cases.
    with np.load(output/"bootstrap_resampling_indices.npz", allow_pickle=False) as saved:
        for role, matrix in indices.items():
            np.testing.assert_array_equal(saved[role], matrix)
        for r in intervals["rows"]:
            g = video[(video.cohort == r["cohort"])&(video.scope == r["scope"])&(video.order == r["order"])]
            g = g.set_index("source_video_id").sort_index()
            matrix = indices[r["cohort"]]
            if r["kind"] == "ratio_of_counts":
                numerator, denominator = u.RATES[r["metric"]]
                numerator_samples = g[numerator].to_numpy()[matrix].sum(axis=1)
                denominator_samples = g[denominator].to_numpy()[matrix].sum(axis=1)
                valid = denominator_samples > 0
                values = numerator_samples[valid]/denominator_samples[valid]
            else:
                data_matrix = g[r["metric"]].to_numpy(dtype=np.float64)[matrix]
                counts = np.isfinite(data_matrix).sum(axis=1)
                valid = counts > 0
                values = np.nansum(data_matrix[valid], axis=1)/counts[valid]
            assert int((~valid).sum()) == r["undefined_resamples"]
            if len(values):
                np.testing.assert_allclose(np.quantile(values, [.025, .975], method="linear"),
                    [r["CI95_lower"], r["CI95_upper"]], atol=1e-12)
            else:
                assert r["CI95_lower"] is None and r["CI95_upper"] is None
    old_aggregate = data.read(prior/"aggregate_metrics.json")["configurations"]
    for role in ["development", "exploratory"]:
        g = table[table.cohort == role]
        old_row = next(r for r in old_aggregate if r["cohort"] == ("development_oof" if role == "development" else role)
                       and r["configuration"] == "C-nonlinear")
        assert int(g.C_nonlinear_selected.sum()) == old_row["selected_count"]
        assert int(((g.net_gain > u.SIGN_TOL)&g.C_nonlinear_selected).sum()) == old_row["helpful_selected"]
        assert int(((g.net_gain < -u.SIGN_TOL)&g.C_nonlinear_selected).sum()) == old_row["harmful_selected"]
        realized = g.assign(realized=g.net_gain*g.C_nonlinear_selected).groupby("source_video_id").realized.mean().mean()
        np.testing.assert_allclose(realized, old_row["mean_realized_gain"], atol=u.REPLAY_ATOL, rtol=u.REPLAY_RTOL)
    original.preserved(p)
    discrepancies = [{"cohort": role, "max_absolute": float(x.absolute_target_discrepancy.max()),
        "max_relative_defined": float(x.relative_target_discrepancy.max()) if x.relative_target_discrepancy.notna().any() else None,
        "relative_defined_count": int(x.relative_target_discrepancy.notna().sum())}
        for role, x in table.groupby("cohort", sort=True)]
    return {"passed": True, "checked_at_utc": now(), "development_sources": 100, "exploratory_sources": 40,
        "development_opportunities": 900, "exploratory_opportunities": 360,
        "independent_schedule_replays": len(replay_errors), "exact_refresh_counts": u.COUNTS,
        "latest_feature_reuse_and_cache_boundary_checks": True,
        "original_targets_and_C_nonlinear_selections_reproduced": True,
        "max_absolute_target_discrepancy": float(absolute.max()),
        "max_relative_target_discrepancy_defined": float(relative.max()) if len(relative) else None,
        "relative_undefined_count": int((~relative_mask).sum()), "by_cohort_discrepancies": discrepancies,
        "max_absolute_algebra_residual": float(max(algebra_errors)),
        "max_absolute_independent_loss_discrepancy": float(max(replay_errors)),
        "all_attribution_orders_and_grouped_conditional_bootstraps_reproduced": True,
        "previous_tracked_files_preserved": len(p["previous_tracked_blob_identities"]),
        "classifier_unchanged": True, "actual_RGB_encoder_executions": 0, "video_decodes": 0, "downloads": 0,
        "recognition_training_calls": 0, "predictor_training_calls": 0, "fresh_confirmation_parsed": 0}
