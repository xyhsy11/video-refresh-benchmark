"""Read-only reconstruction of inputs, predictions and paired decision utility."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_task_signals as old
import recognition_baseline_data as data
import task_signal_ablation_utils as u
from research_utils import ROOT, sha256_file, now
from task_aware_oracle_utils import feature_logits
from verify_task_signals import direct_replay, same


def independent_prediction(table: pd.DataFrame, state: dict, arrays: dict) -> np.ndarray:
    z = (table[state["columns"]].to_numpy()-np.asarray(state["means"]))/np.asarray(state["scales"])
    if not state["nonlinear"]:
        return state["intercept"]+z@np.asarray(state["coefficients"])
    # Direct broadcasted squared differences; no dot-product distance shortcut.
    training = arrays["standardized_training_features"]
    parts = []
    for start in range(0, len(z), 24):
        similarity = np.exp(-np.sum((z[start:start+24, None, :]-training[None, :, :])**2, axis=2)/z.shape[1])
        centered = similarity-similarity.mean(axis=1)[:, None]-arrays["kernel_column_means"][None, :]+state["kernel_grand_mean"]
        parts.append(state["intercept"]+centered@arrays["dual_coefficients"])
    return np.concatenate(parts)


def audit(recognition: Path, diagnostic: Path, output: Path, p: dict) -> dict:
    import analyze_task_signal_ablation as a
    manifest, head, hashes = a.identities(recognition, diagnostic)
    assert hashes == p["input_sha256"]
    assert {n: sha256_file(ROOT/n) for n in p["source_sha256"]} == p["source_sha256"]
    old.preserved(p)
    assert sha256_file(output/"fold_manifest.json") == sha256_file(diagnostic/"fold_manifest.json")
    assert sha256_file(output/"counterfactual_targets.csv") == sha256_file(diagnostic/"counterfactual_examples.csv")
    targets = pd.read_csv(output/"counterfactual_targets.csv", float_precision="round_trip")
    inputs = pd.read_csv(output/"inputs_including_privileged_P.csv", float_precision="round_trip")
    assert len(inputs) == 1260 and inputs.example_id.nunique() == 1260
    np.testing.assert_array_equal(inputs[u.SETS["S"]], targets[u.SETS["S"]])
    predictions = pd.concat([pd.read_csv(output/n, float_precision="round_trip") for n in
        ["out_of_fold_predictions.csv", "exploratory_predictions.csv"]], ignore_index=True)
    assert predictions.groupby("cohort").size().to_dict() == {"development_oof": 9000, "exploratory": 3600}
    assert not predictions.duplicated(["example_id", "configuration"]).any()
    assert predictions.groupby("example_id").size().eq(10).all()
    target_lookup = targets.set_index("example_id")
    np.testing.assert_array_equal(predictions.refresh_gain, target_lookup.loc[predictions.example_id].refresh_gain)
    assert (predictions.earlier_selected == (predictions.prediction > u.TOL)).all()
    np.testing.assert_array_equal(predictions.realized_gain, predictions.refresh_gain*predictions.earlier_selected)
    assert (predictions.privileged == predictions.configuration.str.startswith("P-")).all()
    isolation, independent_targets = 0, 0
    with old.frozen_only():
        for role in ["development", "exploratory"]:
            videos = data.load_role(manifest["videos"], role)
            rows = targets[targets.cohort == role]
            regenerated = a.input_table(videos, rows, head)
            saved = inputs[inputs.cohort == role]
            np.testing.assert_allclose(regenerated[u.SETS["P"]], saved[u.SETS["P"]], atol=1e-11, rtol=1e-10)
            by_id = {v.entry["video_id"]: v for v in videos}
            for row in rows.itertuples():
                v = by_id[row.video_id]
                base, history = direct_replay(v.references, [0, 4, 8, 12], head, row.label_index)
                alternative, early_history = direct_replay(v.references, json.loads(row.early_positions), head, row.label_index)
                np.testing.assert_array_equal(history[:row.t], early_history[:row.t])
                np.testing.assert_allclose(base-alternative, row.refresh_gain, atol=1e-10, rtol=1e-10)
                independent_targets += 1
                if row.t not in [2, 6, 10]:
                    continue
                observed = {i for i in [0, 4, 8, 12] if i < row.t}
                rgb = v.references.copy()
                for i in range(16):
                    if i not in observed:
                        rgb[i] = np.arange(512, dtype=np.float32)+500+i
                motion = tuple(dict(m, mean_magnitude_px=9e5, p90_magnitude_px=1e6)
                               if i > v.cheap.indices[row.t] else m for i, m in enumerate(v.cheap.motion))
                mutant = replace(v, entry={**v.entry, "label": "mutated"},
                                 cheap=replace(v.cheap, motion=motion), references=rgb)
                actual = u.cached_inputs(u.cached_observation(v, row.t, head))
                changed = u.cached_inputs(u.cached_observation(mutant, row.t, head))
                same(actual, changed)
                # P must ignore labels and later RGB while legitimately depending on f_t.
                future = v.references.copy()
                future[row.t+1:] = rgb[row.t+1:]
                future_only = replace(mutant, references=future)
                p_original = u.privileged_inputs(actual, feature_logits(v.references[[row.t]], head)[0])
                p_future = u.privileged_inputs(u.cached_inputs(u.cached_observation(future_only, row.t, head)),
                                               feature_logits(future_only.references[[row.t]], head)[0])
                same(p_original, p_future)
                p_current = u.privileged_inputs(changed, feature_logits(mutant.references[[row.t]], head)[0])
                assert not np.allclose([p_original[c] for c in u.SETS["P"][-20:]],
                                       [p_current[c] for c in u.SETS["P"][-20:]])
                # Independently walk past outputs; current candidate is excluded.
                past, latest = [], 0
                for t in range(row.t):
                    if t in [0, 4, 8, 12]:
                        latest = t
                    past.append(feature_logits(v.references[[latest]], head)[0])
                np.testing.assert_allclose([actual[c] for c in u.SETS["C"][36:]],
                                           u.centered(np.stack(past).mean(axis=0)), atol=1e-11, rtol=1e-10)
                np.testing.assert_allclose([actual[c] for c in u.SETS["C"][16:36]],
                                           u.centered(feature_logits(v.references[[row.last_refresh]], head)[0]), atol=1e-11, rtol=1e-10)
                isolation += 1
    data_table = a.join_targets(inputs, targets)
    development = data_table[data_table.cohort == "development"]
    folds = data.read(output/"fold_manifest.json")
    frozen = data.read(output/"frozen_models.json")
    assert frozen["oof_sha256"] == sha256_file(output/"out_of_fold_predictions.csv")
    fits = [(f["fold"], f["training_sources"], f["validation_sources"], output/"models"/f"fold_{f['fold']}") for f in folds["folds"]]
    fits.append((-1, sorted(development.source_video_id.unique()),
        sorted(data_table[data_table.cohort == "exploratory"].source_video_id.unique()), output/"models/final_development"))
    checked_models = 0
    for fold, training, heldout, folder in fits:
        assert set(training).isdisjoint(heldout)
        if fold == -1:
            assert sha256_file(folder/"registry.json") == frozen["final_registry_sha256"]
        else:
            assert sha256_file(folder/"registry.json") == frozen["folds"][str(fold)]["registry_sha256"]
        train = development[development.source_video_id.isin(training)]
        held = data_table[data_table.source_video_id.isin(heldout)]
        assert len(train) == (900 if fold == -1 else 720)
        assert len(held) == (360 if fold == -1 else 180)
        registry = data.read(folder/"registry.json")
        np.testing.assert_allclose([registry["training_mean"], registry["training_median"]],
                                  [train.refresh_gain.mean(), train.refresh_gain.median()], atol=1e-12)
        for name in u.PREDICTORS:
            stored = predictions[(predictions.configuration == name)&(predictions.fold == fold)].set_index("example_id").loc[held.example_id]
            if name in registry["models"]:
                state, arrays = u.load_model(folder, registry["models"][name])
                assert state["columns"] == u.SETS[state["input_set"]]
                assert state["training_source_ids"] == sorted(training)
                assert state["training_example_ids"] == train.example_id.tolist()
                assert state["privileged"] == name.startswith("P-")
                x = train[state["columns"]].to_numpy()
                mean = x.mean(axis=0); std = x.std(axis=0, ddof=0); scale = np.where(std <= u.TOL, 1., std)
                np.testing.assert_array_equal(state["means"], mean)
                np.testing.assert_array_equal(state["scales"], scale)
                z = (x-mean)/scale; y = train.refresh_gain.to_numpy()
                np.testing.assert_allclose(state["intercept"], y.mean(), atol=1e-12)
                if state["nonlinear"]:
                    np.testing.assert_array_equal(arrays["training_features"], x)
                    np.testing.assert_array_equal(arrays["training_targets"], y)
                    np.testing.assert_array_equal(arrays["standardized_training_features"], z)
                    k = u.kernel(z, z)
                    np.testing.assert_allclose(arrays["kernel_column_means"], k.mean(axis=0), atol=1e-12)
                    np.testing.assert_allclose(state["kernel_grand_mean"], k.mean(), atol=1e-12)
                    kc = k-k.mean(axis=0)[None, :]-k.mean(axis=1)[:, None]+k.mean()
                    np.testing.assert_allclose((kc+np.eye(len(k)))@arrays["dual_coefficients"], y-y.mean(), atol=1e-10, rtol=1e-9)
                else:
                    independent = np.linalg.lstsq(np.vstack([z, np.eye(z.shape[1])]),
                                                 np.r_[y-y.mean(), np.zeros(z.shape[1])], rcond=None)[0]
                    np.testing.assert_allclose(state["coefficients"], independent, atol=1e-10, rtol=1e-9)
                expected = independent_prediction(held, state, arrays)
                checked_models += 1
            else:
                expected = np.full(len(held), 0. if name == "zero" else registry[name])
            np.testing.assert_allclose(stored.prediction, expected, atol=1e-10, rtol=1e-9)
    for filename, role in [("out_of_fold_predictions.csv", "development_oof"), ("exploratory_predictions.csv", "exploratory")]:
        previous = pd.read_csv(diagnostic/filename, float_precision="round_trip").set_index(["example_id", "predictor"])
        for name, original in [("S-linear", "D"), ("A-linear", "A")]:
            g = predictions[(predictions.cohort == role)&(predictions.configuration == name)]
            values = previous.loc[[(e, original) for e in g.example_id]].prediction.to_numpy()
            np.testing.assert_allclose(g.prediction, values, atol=1e-11, rtol=1e-10)
    aggregate, per_video, intervals, indices = u.summarize(predictions, p["seed"])
    same(aggregate, data.read(output/"aggregate_metrics.json"))
    same(intervals, data.read(output/"bootstrap_intervals.json"))
    saved_video = pd.read_csv(output/"per_video_metrics.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(per_video, saved_video, check_exact=False, atol=1e-11, rtol=1e-10)
    with np.load(output/"bootstrap_resampling_indices.npz", allow_pickle=False) as saved:
        for role, index in indices.items():
            np.testing.assert_array_equal(saved[role], index)
            sources = sorted(predictions[predictions.cohort == role].source_video_id.unique())
            manual = []
            for source in sources:
                g = predictions[(predictions.cohort == role)&(predictions.source_video_id == source)&
                                (predictions.configuration == "C-nonlinear")]
                assert len(g) == 9
                manual.append(float(np.mean(g.refresh_gain.to_numpy()*(g.prediction.to_numpy() > u.TOL))))
            repeated_means = np.array([np.mean(np.array(manual)[i]) for i in index])
            comparison = next(r for r in intervals["comparisons"] if r["cohort"] == role and r["candidate"] == "C-nonlinear" and r["baseline"] == "keep_reference")
            np.testing.assert_allclose(np.quantile(repeated_means, [.025, .975], method="linear"),
                [comparison["CI95_lower"], comparison["CI95_upper"]], atol=1e-12)
    old.preserved(p)
    return {"passed": True, "checked_at_utc": now(), "unchanged_source_grouped_folds": True,
            "development_sources": 100, "exploratory_sources": 40, "development_examples": 900,
            "exploratory_examples": 360, "predictions": 12600, "independent_target_replays": independent_targets,
            "S_exact_prior_D_inputs": True, "S_linear_prior_D_predictions": True, "age_baseline_reproduced": True,
            "cached_and_privileged_input_isolation_checks": isolation, "independently_checked_models": checked_models,
            "training_only_scalers_and_kernel_centering": True, "independent_kernel_query_distances": True,
            "metrics_and_whole_video_bootstrap_reproduced": True,
            "previous_tracked_files_preserved": len(p["previous_tracked_blob_identities"]),
            "fresh_confirmation_parsed": 0, "RGB_encoder_executions": 0, "video_decodes": 0, "downloads": 0,
            "recognition_classifier_training": 0, "P": "privileged current RGB,not a cheap gate or matched-cost deployment"}
