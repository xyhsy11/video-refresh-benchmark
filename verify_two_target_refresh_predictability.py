"""Independent training-provenance, target-linearity and saved-metric checks."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_task_signals as historical
import two_target_refresh_utils as u
from research_utils import ROOT, sha256_file, now
from verify_task_signals import same


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def equivalence(components: pd.DataFrame, ablation: Path) -> dict:
    rows, previous_rows = [], []
    for role, g in components.groupby("cohort", sort=True):
        for recipe in ["C_linear", "C_nonlinear"]:
            model = g[g.recipe == recipe]
            pivot = model.pivot(index="example_id", columns="target", values="prediction")
            for order, b, c in [("forward", "B", "C"), ("averaged", "B_averaged", "C_averaged")]:
                direct, separate = pivot.G.to_numpy(), (pivot[b]-pivot[c]).to_numpy()
                np.testing.assert_allclose(direct, separate, atol=u.ATOL, rtol=u.RTOL)
                difference = abs(direct-separate)
                disagreements = (direct > u.TOL) != (separate > u.TOL)
                band = u.ATOL+u.RTOL*np.maximum(abs(direct), abs(separate))
                near = np.maximum(abs(direct-u.TOL), abs(separate-u.TOL)) <= band
                assert not np.any(disagreements&~near), "Decision difference outside numerical equivalence band"
                rows.append({"cohort": role, "recipe": recipe, "attribution": order,
                    "max_absolute_discrepancy": float(difference.max()), "decision_disagreements": int(disagreements.sum()),
                    "disagreements_only_within_numerical_band": True,
                    "minimum_direct_distance_to_threshold": float(abs(direct-u.TOL).min())})
        name = "out_of_fold_predictions.csv" if role == "development_oof" else "exploratory_predictions.csv"
        old = read_csv(ablation/name)
        for recipe, config in [("C_linear", "C-linear"), ("C_nonlinear", "C-nonlinear"), ("age_budget_ridge", "A-linear")]:
            target = g[(g.recipe == recipe)&(g.target == "G")].set_index("example_id").sort_index()
            prior = old[old.configuration == config].set_index("example_id").loc[target.index]
            np.testing.assert_array_equal(target.actual, prior.refresh_gain)
            np.testing.assert_allclose(target.prediction, prior.prediction, atol=u.ATOL, rtol=u.RTOL)
            previous_rows.append({"cohort": role, "recipe": recipe, "previous_configuration": config,
                "rows": len(target), "max_absolute_discrepancy": float(abs(target.prediction-prior.prediction).max()),
                "decision_disagreements": int(((target.prediction > u.TOL)!=(prior.prediction > u.TOL)).sum())})
    return {"passed": True, "absolute_tolerance": u.ATOL, "relative_tolerance": u.RTOL,
        "controls_are_linear_in_targets_not_new_two_head_methods": True,
        "target_linearity": rows, "previous_direct_predictions": previous_rows,
        "max_target_linearity_discrepancy": max(r["max_absolute_discrepancy"] for r in rows),
        "max_previous_prediction_discrepancy": max(r["max_absolute_discrepancy"] for r in previous_rows)}


def audit(output: Path, p: dict) -> dict:
    import analyze_two_target_refresh_predictability as main
    prior, hashes = main.identities(ROOT/p["decomposition_input"], ROOT/p["ablation_input"])
    assert hashes == p["input_sha256"]
    assert {n: sha256_file(ROOT/n) for n in p["source_sha256"]} == p["source_sha256"]
    historical.preserved(p)
    assert (output/"fold_manifest.json").read_bytes() == (ROOT/p["ablation_input"]/"fold_manifest.json").read_bytes()
    dev = main.load_table(ROOT/p["decomposition_input"], ROOT/p["ablation_input"], "development")
    ex = main.load_table(ROOT/p["decomposition_input"], ROOT/p["ablation_input"], "exploratory")
    for role, table in [("development", dev), ("exploratory", ex)]:
        pd.testing.assert_frame_equal(table, read_csv(output/f"{role}_analysis_table.csv"), check_exact=True)
        expected = {e["source_video_id"]: e for e in p["source_membership"] if e["recognition_role"] == role}
        assert set(table.source_video_id) == set(expected)
        for row in table.itertuples():
            assert row.video_id == expected[row.source_video_id]["video_id"]
        for source, group in table.groupby("source_video_id"):
            assert set(group["class"]) == {expected[source]["label"]}
            assert sorted(zip(group.r, group.t)) == [(r, t) for r in [4, 8, 12] for t in range(r-3, r)]
    assert set(dev.source_video_id).isdisjoint(ex.source_video_id)
    frozen = json.loads((output/"frozen_models.json").read_text())
    assert all(sha256_file(output/n) == digest for n, digest in frozen["all_model_file_sha256"].items())
    oof_frozen = json.loads((output/"oof_frozen.json").read_text())
    assert oof_frozen["decisions_sha256"] == sha256_file(output/"out_of_fold_predictions.csv")
    assert oof_frozen["components_sha256"] == sha256_file(output/"component_predictions_oof.csv")
    assert oof_frozen["saved_at_utc"] < frozen["frozen_at_utc"] < json.loads((output/"exploratory_evaluation.json").read_text())["evaluated_at_utc"]
    regenerated_parts, forest_accounting, partition_differences = [], [], []
    checked_models, root_checks, mutation_checks = 0, 0, 0
    with ExitStack() as stack:
        stack.enter_context(historical.frozen_only())
        stack.enter_context(patch("two_target_refresh_utils.RandomForestRegressor.fit", side_effect=AssertionError("Audit must not fit forests")))
        stack.enter_context(patch("task_signal_ablation_utils.fit_array", side_effect=AssertionError("Audit must not fit ridges")))
        bundles = [(f"fold_{f['fold']}", f["training_sources"], f["validation_sources"], f["fold"]) for f in p["folds"]["folds"]]
        bundles.append(("final_development", sorted(dev.source_video_id.unique()), None, -1))
        for stem, sources, heldout, fold in bundles:
            tr = dev[dev.source_video_id.isin(sources)]
            query = ex if heldout is None else dev[dev.source_video_id.isin(heldout)]
            assert set(tr.source_video_id).isdisjoint(query.source_video_id)
            if heldout is not None:
                assert len(tr) == 720 and len(query) == 180
                assert sorted(query.groupby("source_video_id")["class"].first()) == sorted(p["classes"])
            folder = output/"models"/stem
            registry = json.loads((folder/"registry.json").read_text())
            assert registry["training_example_ids"] == tr.example_id.tolist()
            assert registry["training_source_ids"] == sorted(sources)
            assert registry["training_C_matrix_sha256"] == hashlib.sha256(u.design(tr).tobytes()).hexdigest()
            for target in u.TARGETS:
                np.testing.assert_allclose([registry["constants"][target]["training_mean"], registry["constants"][target]["training_median"]],
                                          [tr[target].mean(), tr[target].median()], atol=1e-14, rtol=0)
            partitions = {}
            for entry in registry["models"]:
                state, arrays, forest = u.load_model(folder, entry)
                checked_models += 1
                assert state["training_example_ids"] == tr.example_id.tolist() and state["training_source_ids"] == sorted(sources)
                columns = u.AGE_COLUMNS if entry["recipe"] == "age_budget_ridge" else u.COLUMNS
                assert state["input_columns"] == columns
                x = tr[columns].to_numpy(dtype=np.float64)
                mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
                scale = np.where(std <= u.TOL, 1., std)
                np.testing.assert_array_equal(state["means"], mean)
                np.testing.assert_array_equal(state["scales"], scale)
                z, y = (x-mean)/scale, tr[entry["target"]].to_numpy()
                if forest is None:
                    np.testing.assert_allclose(state["intercept"], y.mean(), atol=1e-14, rtol=0)
                    if not state["nonlinear"]:
                        np.testing.assert_allclose((z.T@z+np.eye(z.shape[1]))@state["coefficients"], z.T@(y-y.mean()), atol=u.ATOL, rtol=u.RTOL)
                    else:
                        np.testing.assert_array_equal(arrays["training_features"], x)
                        np.testing.assert_array_equal(arrays["training_targets"], y)
                        k = u.previous.kernel(z, z)
                        np.testing.assert_allclose(arrays["kernel_column_means"], k.mean(axis=0), atol=1e-14)
                        kc = k-k.mean(axis=0)[None]-k.mean(axis=1)[:, None]+k.mean()
                        np.testing.assert_allclose((kc+np.eye(len(z)))@arrays["dual_coefficients"], y-y.mean(), atol=u.ATOL, rtol=u.RTOL)
                else:
                    for name, value in u.RF_PARAMETERS.items():
                        assert forest.get_params()[name] == value, name
                    assert forest.get_params() == state["parameters"] and forest.n_features_in_ == 56
                    assert len(forest.estimators_) == state["trees"]
                    assert sum(t.tree_.node_count for t in forest.estimators_) == state["node_count"]
                    assert (folder/state["model_file"]).stat().st_size == state["serialized_model_bytes"]
                    for tree, sample in zip(forest.estimators_, forest.estimators_samples_):
                        np.testing.assert_allclose(tree.tree_.value[0, 0, 0], y[sample].mean(), atol=u.ATOL, rtol=u.RTOL)
                        root_checks += 1
                    signature = hashlib.sha256()
                    for tree in forest.estimators_:
                        signature.update(tree.tree_.feature.tobytes()); signature.update(tree.tree_.threshold.tobytes())
                    partitions[(entry["recipe"], entry["target"])] = signature.hexdigest()
                    forest_accounting.append({"bundle": stem, "recipe": entry["recipe"], "target": entry["target"],
                        "trees": state["trees"], "nodes": state["node_count"], "serialized_bytes": state["serialized_model_bytes"],
                        "model_file": (folder/state["model_file"]).relative_to(output).as_posix()})
            for recipe, b, c in [("Two_target_RF200x2_forward", "B", "C"), ("Two_target_RF200x2_averaged", "B_averaged", "C_averaged")]:
                different = partitions[(recipe, b)] != partitions[(recipe, c)]
                assert different, "Unexpected identical complete B/C forest partitions;investigate target-dependent fitting"
                partition_differences.append({"bundle": stem, "recipe": recipe, "partitions_differ": different})
            role = "exploratory" if heldout is None else "development_oof"
            raw = u.predict_bundle(query, folder, registry, role, fold)
            altered = query.copy()
            for target in u.TARGETS:
                altered[target] = -37*altered[target]+123
            altered["class"] = "MUTATED_NOT_AN_INPUT"
            np.testing.assert_array_equal(raw.prediction, u.predict_bundle(altered, folder, registry, role, fold).prediction)
            mutation_checks += 1
            regenerated_parts.append(raw)
    raw = pd.concat(regenerated_parts, ignore_index=True)
    decisions, errors = u.decision_rows(raw)
    components = u.augment_net_quality(raw, decisions)
    saved_components = pd.concat([read_csv(output/"component_predictions_oof.csv"), read_csv(output/"component_predictions_exploratory.csv")], ignore_index=True)
    saved_decisions = pd.concat([read_csv(output/"out_of_fold_predictions.csv"), read_csv(output/"exploratory_predictions.csv")], ignore_index=True)
    # Row ordering differs only because augmentation is concatenated after both cohorts.
    sortc, sortd = ["cohort", "recipe", "target", "example_id"], ["cohort", "method", "example_id"]
    pd.testing.assert_frame_equal(components.sort_values(sortc).reset_index(drop=True), saved_components.sort_values(sortc).reset_index(drop=True), check_exact=False, atol=u.ATOL, rtol=u.RTOL)
    pd.testing.assert_frame_equal(decisions.sort_values(sortd).reset_index(drop=True), saved_decisions.sort_values(sortd).reset_index(drop=True), check_exact=False, atol=u.ATOL, rtol=u.RTOL)
    assert len(saved_decisions) == 1260*len(u.METHODS)
    eq = equivalence(saved_components, ROOT/p["ablation_input"])
    aggregate, video, intervals, extra = u.summarize(saved_decisions, saved_components, errors, p["seed"], p["bootstrap"]["resamples"])
    same(aggregate, json.loads((output/"aggregate_metrics.json").read_text()))
    same(intervals, json.loads((output/"bootstrap_intervals.json").read_text()))
    pd.testing.assert_frame_equal(video, read_csv(output/"per_video_metrics.csv"), check_exact=False, atol=u.ATOL, rtol=u.RTOL)
    pd.testing.assert_frame_equal(extra["quality_per_video"], read_csv(output/"per_video_component_metrics.csv"), check_exact=False, atol=u.ATOL, rtol=u.RTOL)
    pd.testing.assert_frame_equal(extra["error_per_video"], read_csv(output/"per_video_error_decomposition.csv"), check_exact=False, atol=u.ATOL, rtol=u.RTOL)
    # Independent calculation of utility/MAE and each paired interval from saved predictions.
    with np.load(output/"bootstrap_resampling_indices.npz", allow_pickle=False) as matrices:
        for role, group in saved_decisions.groupby("cohort", sort=True):
            matrix = matrices[role]
            np.testing.assert_array_equal(matrix, extra["indices"][role])
            independent = group.assign(realized=group.actual_G*(group.prediction_G > u.TOL)).groupby(["source_video_id", "method"]).realized.mean().unstack()
            sources = sorted(independent.index)
            for row in intervals["comparisons"]:
                if row["cohort"] != role: continue
                delta = (independent.loc[sources, row["candidate"]]-independent.loc[sources, row["baseline"]]).to_numpy()
                low, high = np.percentile(np.mean(delta[matrix], axis=1), [2.5, 97.5], method="linear")
                np.testing.assert_allclose([delta.mean(), low, high], [row["mean"], row["CI95_lower"], row["CI95_upper"]], atol=u.ATOL, rtol=u.RTOL)
        quality = saved_components.copy()
        quality["ae"] = abs(quality.actual-quality.prediction)
        independent_mae = quality.groupby(["cohort", "recipe", "target", "source_video_id"]).ae.mean().groupby(["cohort", "recipe", "target"]).mean()
        for r in aggregate["prediction_quality"]:
            np.testing.assert_allclose(independent_mae.loc[(r["cohort"], r["recipe"], r["target"])], r["MAE"]["mean"], atol=u.ATOL, rtol=u.RTOL)
    historical.preserved(p)
    return {"passed": True, "checked_at_utc": now(), "cohort_sources": {"development": 100, "exploratory": 40},
        "opportunities": {"development": 900, "exploratory": 360}, "folds_copied_unchanged": True,
        "exact_whitelist_columns": 56, "models_and_training_only_scalers_checked": checked_models,
        "forest_bootstrap_root_means_checked": root_checks, "target_dependent_partition_checks": partition_differences,
        "heldout_target_and_label_prediction_mutations": mutation_checks,
        "equivalence_checks": eq, "forest_model_accounting": forest_accounting,
        "saved_predictions_metrics_and_whole_video_bootstrap_reproduced": True,
        "all_nine_opportunities_retained_per_source": True,
        "previous_tracked_files_preserved": len(p["previous_tracked_blob_identities"]),
        "classifier_and_prior_artifacts_unchanged": True, "actual_RGB_encoder_executions": 0,
        "video_decodes": 0, "video_downloads": 0, "recognition_training_calls": 0,
        "fresh_confirmation_artifacts_accessed": 0}
