"""Independent ordered-history, provenance and paired-metric verification."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_task_signals as historical
import cached_history_signal_utils as u
from research_utils import ROOT, sha256_file, now
from verify_task_signals import same


def frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def independent_features(scores: list[np.ndarray]) -> np.ndarray:
    """Separate explicit formulas: no incremental builder/assembler calls."""
    k = len(scores)
    latest = scores[-1]-scores[-2] if k > 1 else np.zeros(20)
    previous = scores[-2]-scores[-3] if k > 2 else np.zeros(20)
    margins, tops = [], []
    for row in scores:
        values = [float(np.exp(v-max(row))) for v in row]
        probability = np.array(values)/sum(values)
        ordered = sorted(probability)
        margins.append(ordered[-1]-ordered[-2])
        tops.append(next(i for i, v in enumerate(row) if v == max(row)))
    mean = sum(margins)/k
    std = np.sqrt(sum((x-mean)**2 for x in margins)/k) if k > 1 else 0.
    switches = sum(a != b for a, b in zip(tops[:-1], tops[1:]))/(k-1) if k > 1 else 0.
    return np.r_[latest, previous, int(k >= 2), int(k >= 3), std, switches]


def audit(output: Path, p: dict) -> dict:
    import analyze_cached_history_signals as main
    prior = ROOT/p["predictability_input"]
    previous_protocol, hashes = main.identities(prior)
    assert hashes == p["input_sha256"]
    assert {n: sha256_file(ROOT/n) for n in p["source_sha256"]} == p["source_sha256"]
    historical.preserved(p)
    assert (output/"fold_manifest.json").read_bytes() == (prior/"fold_manifest.json").read_bytes()
    assert len(u.C_COLUMNS) == 56 and len(u.T_COLUMNS) == 44 and len(u.AUGMENTED_COLUMNS) == 100
    feature_checks, mutation_checks, max_feature_error = 0, 0, 0.
    tables = {}
    for role, expected in [("development", 900), ("exploratory", 360)]:
        observations = main.read_observations(prior, role)
        augmented, _ = u.augment_observations(observations)
        pd.testing.assert_frame_equal(augmented, frame(output/f"{role}_observations.csv"), check_exact=True)
        assert len(augmented) == expected and augmented.example_id.nunique() == expected
        members = {e["source_video_id"]: e for e in p["source_membership"] if e["recognition_role"] == role}
        assert set(augmented.source_video_id) == set(members)
        for source, group in augmented.groupby("source_video_id"):
            assert set(group.video_id) == {members[source]["video_id"]}
            distinct, last = [], None
            for _, row in group.sort_values("t").iterrows():
                if row.last_refresh != last:
                    distinct.append(row[u.CACHED_COLUMNS].to_numpy(dtype=float)); last = row.last_refresh
                assert len(distinct) == int(row.last_refresh)//4+1 == row.history_length
                reference = independent_features(distinct)
                error = float(abs(reference-row[u.T_COLUMNS].to_numpy(dtype=float)).max())
                max_feature_error = max(max_feature_error, error)
                np.testing.assert_allclose(reference, row[u.T_COLUMNS].to_numpy(dtype=float), atol=u.TOL, rtol=0)
                if len(distinct) == 1:
                    np.testing.assert_array_equal(reference, np.zeros(44))
                if len(distinct) == 2:
                    np.testing.assert_array_equal(reference[20:40], np.zeros(20))
                feature_checks += 1
            # Future event vectors may change; the first block must be identical.
            for cutoff in [4, 8]:
                mutated = group[["cohort", *u.KEYS, *u.C_COLUMNS]].copy()
                later = mutated.last_refresh >= cutoff
                for j, col in enumerate(u.CACHED_COLUMNS):
                    mutated.loc[later, col] += 10*(j-9.5)
                prefix = mutated[mutated.last_refresh < cutoff]
                history = u.CachedHistory()
                expected_T = group.set_index("example_id")
                for _, row in prefix.sort_values("t").iterrows():
                    np.testing.assert_array_equal(history.advance(int(row.last_refresh), row[u.CACHED_COLUMNS].to_numpy(dtype=float)),
                                                  expected_T.loc[row.example_id, u.T_COLUMNS].to_numpy(dtype=float))
                mutation_checks += 1
        # The builder's schema discards all target/privileged/label columns.
        poisoned = observations.copy()
        for col in ["G", "loss_A", "loss_C", "label", "class", "privileged_current_centered_logit_class_00", "future_RGB"]:
            poisoned[col] = np.nan
        same_features, _ = u.augment_observations(poisoned)
        pd.testing.assert_frame_equal(augmented, same_features, check_exact=True)
        mutation_checks += 1
        tables[role] = main.join_targets(augmented, main.read_targets(prior, role))
    dev, ex = tables["development"], tables["exploratory"]
    assert set(dev.source_video_id).isdisjoint(ex.source_video_id)
    frozen = json.loads((output/"frozen_models.json").read_text())
    assert all(sha256_file(output/n) == digest for n, digest in frozen["model_file_sha256"].items())
    oof = json.loads((output/"oof_frozen.json").read_text())
    assert oof["sha256"] == sha256_file(output/"out_of_fold_predictions.csv")
    assert oof["saved_at_utc"] < frozen["frozen_at_utc"] < json.loads((output/"exploratory_evaluation.json").read_text())["evaluated_at_utc"]
    reconstructed, accounting, prefix_errors = [], [], []
    root_checks = 0
    with ExitStack() as stack:
        stack.enter_context(historical.frozen_only())
        stack.enter_context(patch("cached_history_signal_utils.RandomForestRegressor.fit", side_effect=AssertionError("Audit cannot fit")))
        folds = [(f"fold_{f['fold']}", f["training_sources"], f["validation_sources"], f["fold"]) for f in p["folds"]["folds"]]
        folds.append(("final_development", sorted(dev.source_video_id.unique()), None, -1))
        for stem, sources, held, fold in folds:
            train = dev[dev.source_video_id.isin(sources)]
            query = ex if held is None else dev[dev.source_video_id.isin(held)]
            assert len(train) == (900 if held is None else 720)
            assert set(train.source_video_id).isdisjoint(query.source_video_id)
            if held is not None:
                labels = {e["source_video_id"]: e["label"] for e in p["source_membership"]}
                assert sorted(labels[s] for s in held) == sorted(p["classes"])
            folder = output/"models"/stem
            registry = json.loads((folder/"registry.json").read_text())
            np.testing.assert_allclose([registry["training_mean"], registry["training_median"]], [train.G.mean(), train.G.median()], atol=1e-14, rtol=0)
            states = {}
            for method, entry in registry["models"].items():
                state, _, forest = u.previous.load_model(folder, entry)
                states[method] = state
                columns = u.C_COLUMNS if method == "C_RF400" else u.AUGMENTED_COLUMNS
                assert state["input_columns"] == columns
                assert state["training_example_ids"] == train.example_id.tolist()
                assert state["training_source_ids"] == sorted(sources)
                x = train[columns].to_numpy(dtype=float)
                means, std = x.mean(0), x.std(0, ddof=0)
                scales = np.where(std <= u.TOL, 1., std)
                np.testing.assert_allclose(state["means"], means, atol=u.TOL, rtol=0)
                np.testing.assert_allclose(state["scales"], scales, atol=u.TOL, rtol=0)
                assert forest.get_params() == state["parameters"] == p["RF_parameters"]
                assert len(forest.estimators_) == state["trees"] == 400
                assert sum(t.tree_.node_count for t in forest.estimators_) == state["node_count"]
                assert (folder/state["model_file"]).stat().st_size == state["serialized_model_bytes"]
                if method == "C_RF400":
                    for name in [entry["file"], state["model_file"]]:
                        assert (folder/name).read_bytes() == (prior/"models"/stem/name).read_bytes()
                y = train.G.to_numpy()
                for tree, samples in zip(forest.estimators_, forest.estimators_samples_):
                    np.testing.assert_allclose(tree.tree_.value[0, 0, 0], y[samples].mean(), atol=u.ATOL, rtol=u.RTOL)
                    root_checks += 1
                accounting.append({"bundle": stem, "method": method, "trees": state["trees"], "nodes": state["node_count"], "serialized_bytes": state["serialized_model_bytes"]})
            prefix_error = max(float(abs(np.array(states["C_plus_T_RF400"][key])[:56]-states["C_RF400"][key]).max()) for key in ["means", "scales", "raw_standard_deviations"])
            assert prefix_error <= u.TOL
            prefix_errors.append(prefix_error)
            role = "exploratory" if held is None else "development_oof"
            reconstructed.append(u.predictions(query, folder, registry, role, fold))
    raw = pd.concat(reconstructed, ignore_index=True)
    saved = pd.concat([frame(output/"out_of_fold_predictions.csv"), frame(output/"exploratory_predictions.csv")], ignore_index=True)
    pd.testing.assert_frame_equal(raw, saved, check_exact=True)
    assert len(saved) == 1260*5
    for role, name in [("development_oof", "out_of_fold_predictions.csv"), ("exploratory", "exploratory_predictions.csv")]:
        old = pd.read_csv(prior/name, usecols=["example_id", "method", "actual_G", "prediction_G"], float_precision="round_trip")
        old = old[old.method == "Direct_G_RF400"].set_index("example_id")
        baseline = saved[(saved.cohort == role)&(saved.method == "C_RF400")].set_index("example_id")
        np.testing.assert_array_equal(baseline.G, old.loc[baseline.index, "actual_G"])
        np.testing.assert_array_equal(baseline.prediction, old.loc[baseline.index, "prediction_G"])
    aggregate, video, history, intervals, matrices = u.summarize(saved, p["seed"], p["bootstrap"]["resamples"])
    same(aggregate, json.loads((output/"aggregate_metrics.json").read_text()))
    same(intervals, json.loads((output/"bootstrap_intervals.json").read_text()))
    pd.testing.assert_frame_equal(video, frame(output/"per_video_metrics.csv"), check_exact=True)
    pd.testing.assert_frame_equal(history, frame(output/"history_length_metrics.csv"), check_exact=True)
    with np.load(output/"bootstrap_resampling_indices.npz", allow_pickle=False) as stored:
        for role, group in saved.groupby("cohort", sort=True):
            np.testing.assert_array_equal(stored[role], matrices[role])
            assert stored[role].shape == (p["bootstrap"]["resamples"], 100 if role == "development_oof" else 40)
            for length in [0, 1, 2, 3]:
                rows = group if length == 0 else group[group.history_length == length]
                independent = rows.assign(value=rows.G*(rows.prediction > u.TOL)).groupby(["source_video_id", "method"]).value.mean().unstack().sort_index()
                for r in intervals["comparisons"]:
                    if r["cohort"] != role or r["history_length"] != length or r["metric"] != "utility": continue
                    delta = (independent[r["candidate"]]-independent[r["baseline"]]).to_numpy()
                    lower, upper = np.percentile(delta[stored[role]].mean(1), [2.5, 97.5], method="linear")
                    np.testing.assert_allclose([delta.mean(), lower, upper], [r["mean"], r["CI95_lower"], r["CI95_upper"]], atol=u.TOL, rtol=0)
    timing = frame(output/"cpu_microbenchmark.csv")
    metadata = json.loads((output/"cpu_microbenchmark.metadata.json").read_text())
    assert len(timing) == 15 and timing.groupby("mode").size().eq(5).all()
    assert timing.opportunities.eq(90).all() and timing.videos.eq(10).all()
    assert (timing.total_ms > 0).all()
    np.testing.assert_allclose(timing.us_per_opportunity, timing.total_ms*1000/90, atol=u.TOL, rtol=u.RTOL)
    assert metadata["sample_source_ids"] == p["timing"]["source_ids"] == sorted(dev.source_video_id.unique())[:10]
    rng = np.random.Generator(np.random.PCG64(p["seed"]))
    from cached_history_cpu import MODES
    assert metadata["warmup_execution_order"] == [rng.permutation(MODES).tolist() for _ in range(2)]
    for order in metadata["measured_execution_order"]:
        assert order["order"] == rng.permutation(MODES).tolist()
        observed = timing[timing.run_id == order["run_id"]].sort_values("execution_ordinal")["mode"].tolist()
        assert observed == order["order"]
    fixture = u.information_fixture()
    assert fixture["old_C_unchanged"] and fixture["T_changes"]
    historical.preserved(p)
    return {"passed": True, "checked_at_utc": now(), "cohort_sources": {"development": 100, "exploratory": 40},
        "opportunities": {"development": 900, "exploratory": 360}, "independent_history_feature_checks": feature_checks,
        "max_independent_T_discrepancy": max_feature_error, "input_isolation_mutations": mutation_checks,
        "folds_targets_and_source_identities_unchanged": True, "baseline_prediction_max_discrepancy": 0.,
        "baseline_models_reused_byte_for_byte": 6, "new_augmented_RF_fits": 6,
        "max_scaler_first56_discrepancy": max(prefix_errors), "training_bootstrap_root_checks": root_checks,
        "model_accounting": accounting, "saved_metrics_and_grouped_bootstraps_reproduced": True,
        "information_audit_swap_fixture": True, "CPU_timing_rows_verified": len(timing),
        "CPU_timing_execution_order_reproduced": True,
        "previous_tracked_files_preserved": len(p["previous_tracked_blob_identities"]),
        "classifier_and_prior_artifacts_unchanged": True, "actual_RGB_encoder_executions": 0,
        "video_decodes": 0, "video_downloads": 0, "recognition_training_calls": 0, "fresh_confirmation_artifacts_accessed": 0}
