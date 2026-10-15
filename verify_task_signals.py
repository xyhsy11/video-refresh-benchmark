"""Focused independent saved-result and information-isolation checks."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

import recognition_baseline_data as data
import task_aware_oracle_utils as task
import task_signal_utils as s
from research_utils import ROOT, sha256_file, now


def direct_replay(features: np.ndarray, positions: list[int], parameters: dict, label: int) -> tuple[float, np.ndarray]:
    """Independent explicit feature copying, affine head and log-sum-exp."""
    assert len(positions) == len(set(positions)) == 4 and positions[0] == 0
    outputs, used, latest = [], [], 0
    for t in range(16):
        if t in positions:
            latest = t
        outputs.append(features[latest].astype(np.float64).copy())
        used.append(latest)
    pooled = np.stack(outputs).mean(axis=0)
    z = (pooled-np.array(parameters["means"])) / np.array(parameters["scales"])
    logits = np.array(parameters["weights"]) @ z + np.array(parameters["bias"])
    maximum = float(logits.max())
    loss = maximum + np.log(np.exp(logits-maximum).sum()) - logits[label]
    return float(loss), np.array(used)


def same(left, right) -> None:
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        np.testing.assert_allclose(left, right, atol=1e-11, rtol=1e-10)
    else:
        assert left == right, (left, right)


def audit(directory: Path, output: Path, protocol: dict) -> dict:
    import analyze_task_signals as a
    manifest, head, controllers, hashes = a.identities(directory)
    assert hashes == protocol["input_sha256"]
    assert {n: sha256_file(ROOT/n) for n in protocol["source_sha256"]} == protocol["source_sha256"]
    a.preserved(protocol)
    folds = data.read(output/"fold_manifest.json")
    assert folds == s.make_folds(manifest["videos"], manifest["classes"], protocol["seed"])
    all_examples = pd.read_csv(output/"counterfactual_examples.csv", float_precision="round_trip")
    clips = pd.read_csv(output/"clip_error_analysis.csv", float_precision="round_trip")
    assert len(clips) == 420 and len(all_examples) == 1260
    assert all_examples.example_id.nunique() == 1260
    assert all_examples.groupby("source_video_id").size().eq(9).all()
    prior = pd.read_csv(directory/"per_video_metrics.csv", float_precision="round_trip")
    prior = prior[(prior.training_per_class == 20) & prior.policy.isin(["dense", "periodic"])]
    lookup = clips.set_index(["cohort", "video_id", "policy"])
    assert len(prior) == 280
    for row in prior.itertuples():
        current = lookup.loc[(row.cohort, row.video_id, row.policy)]
        assert current.policy_prediction == row.prediction_index and current.policy_correct == row.correct
        np.testing.assert_allclose([current.policy_CE, current.feature_cosine_error],
                                  [row.cross_entropy, row.feature_cosine_error], atol=1e-10, rtol=1e-10)
    real_examples, real_clips = [], []
    isolation_checks = 0
    with a.frozen_only():
        for role in ["development", "exploratory"]:
            videos = data.load_role(manifest["videos"], role)
            c, e = a.evaluate(videos, role, head, controllers, protocol["classes"])
            real_examples.append(e); real_clips.append(c)
            by_id = {v.entry["video_id"]: v for v in videos}
            for row in all_examples[all_examples.cohort == role].itertuples():
                video = by_id[row.video_id]
                baseline_loss, baseline_history = direct_replay(video.references, s.PERIODIC, head, row.label_index)
                early_loss, early_history = direct_replay(video.references, json.loads(row.early_positions), head, row.label_index)
                np.testing.assert_array_equal(baseline_history[:row.t], early_history[:row.t])
                assert len(early_history) == 16
                np.testing.assert_allclose(baseline_loss-early_loss, row.refresh_gain, atol=1e-10, rtol=1e-10)
                assert row.interval_start_frame == video.cheap.indices[row.last_refresh]+1
                assert row.interval_end_frame == video.cheap.indices[row.t]
                # Change labels, unobserved (including current/future) RGB, and future motion.
                # Legitimate cached RGB and prefix codec observations stay fixed.
                if row.t in [2, 6, 10]:
                    rgb = video.references.copy()
                    observed = {p for p in s.PERIODIC if p < row.t}
                    for t in range(16):
                        if t not in observed:
                            rgb[t] = np.arange(512, dtype=np.float32)+1000+t
                    end = video.cheap.indices[row.t]
                    motion = tuple(dict(m, mean_magnitude_px=1e6, p90_magnitude_px=2e6)
                                   if i > end else m for i, m in enumerate(video.cheap.motion))
                    mutant = replace(video, entry={**video.entry, "label": "mutated", "source_video_id": "mutated"},
                                     references=rgb, cheap=replace(video.cheap, motion=motion))
                    same(s.observed_inputs(video, row.t, head), s.observed_inputs(mutant, row.t, head))
                    isolation_checks += 1
    regenerated = pd.concat(real_examples, ignore_index=True).set_index("example_id").sort_index()
    saved = all_examples.set_index("example_id").sort_index()
    np.testing.assert_allclose(regenerated[s.GROUPS["D"]+[s.TARGET, "privileged_current_drift"]],
                               saved[s.GROUPS["D"]+[s.TARGET, "privileged_current_drift"]], atol=1e-11, rtol=1e-10)
    regenerated_clips = pd.concat(real_clips, ignore_index=True).sort_values(["cohort", "video_id", "policy"])
    pd.testing.assert_frame_equal(regenerated_clips.reset_index(drop=True),
        clips.sort_values(["cohort", "video_id", "policy"]).reset_index(drop=True), check_exact=False, atol=1e-10, rtol=1e-10)
    parameters = data.read(output/"probe_parameters.json")
    assert sha256_file(output/"out_of_fold_predictions.csv") == parameters["oof_predictions_sha256"]
    predictions = pd.concat([pd.read_csv(output/n, float_precision="round_trip") for n in
                             ["out_of_fold_predictions.csv", "exploratory_predictions.csv"]], ignore_index=True)
    assert len(predictions) == 7560 and not predictions.duplicated(["example_id", "predictor"]).any()
    assert predictions.groupby("example_id").size().eq(6).all()
    np.testing.assert_allclose(predictions.refresh_gain, saved.loc[predictions.example_id, s.TARGET])
    dev = saved[saved.cohort == "development"].reset_index()
    expected_parts = []
    all_fits = [(f["fold"], f["training_sources"], f["validation_sources"], parameters["fold_models"][str(f["fold"])])
                for f in folds["folds"]]
    all_fits.append((-1, sorted(dev.source_video_id.unique()),
        sorted(saved[saved.cohort == "exploratory"].source_video_id.unique()),
        {"models": parameters["full_development_models"], "training_mean": parameters["full_development_mean"]}))
    for fold, training, heldout, state in all_fits:
        assert set(training).isdisjoint(heldout)
        train = dev[dev.source_video_id.isin(training)]
        validation = saved[saved.source_video_id.isin(heldout)].reset_index()
        assert len(train) == (900 if fold == -1 else 720)
        np.testing.assert_allclose(state["training_mean"], train.refresh_gain.mean(), atol=1e-12)
        for group, model in state["models"].items():
            assert model["training_source_ids"] == sorted(training)
            assert model["training_example_ids"] == sorted(train.example_id)
            columns = s.GROUPS[group]
            assert model["columns"] == columns
            x = train[columns].to_numpy(dtype=np.float64)
            mean = x.mean(axis=0); std = x.std(axis=0, ddof=0); scale = np.where(std <= s.TOL, 1., std)
            np.testing.assert_allclose(model["means"], mean, atol=1e-12)
            np.testing.assert_allclose(model["scales"], scale, atol=1e-12)
            z = (x-mean)/scale
            # Independent augmented least-squares solution (rather than normal equations).
            design = np.vstack([z, np.eye(len(columns))])
            target = np.r_[train.refresh_gain.to_numpy()-train.refresh_gain.mean(), np.zeros(len(columns))]
            coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
            np.testing.assert_allclose(model["coefficients"], coefficient, atol=1e-10, rtol=1e-9)
            np.testing.assert_allclose(model["intercept"], train.refresh_gain.mean(), atol=1e-12)
        expected_parts.append(a.prediction_rows(validation, "exploratory" if fold == -1 else "development_oof",
                                               fold, state["models"], state["training_mean"]))
    keys = ["example_id", "predictor"]
    expected = pd.concat(expected_parts, ignore_index=True).sort_values(keys).reset_index(drop=True)
    actual = predictions.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual, check_exact=False, atol=1e-11, rtol=1e-10)
    aggregate, per_video, intervals, index_arrays = s.summarize_predictions(predictions, protocol["seed"])
    published = data.read(output/"aggregate_metrics.json")
    same(aggregate["probe_metrics"], published["probe_metrics"])
    same(s.clip_summary(clips), published["clip_error_analysis"])
    same(intervals, data.read(output/"bootstrap_intervals.json"))
    with np.load(output/"bootstrap_resampling_indices.npz", allow_pickle=False) as arrays:
        for role, indices in index_arrays.items():
            np.testing.assert_array_equal(arrays[role], indices)
            manual = []
            for source in sorted(predictions[predictions.cohort == role].source_video_id.unique()):
                g = predictions[(predictions.cohort == role) & (predictions.source_video_id == source)]
                errors = {name: np.abs(x.prediction-x.refresh_gain).mean() for name, x in g.groupby("predictor")}
                manual.append(errors["A"]-errors["D"])
            means = np.array([np.mean(np.asarray(manual)[i]) for i in indices])
            found = next(r for r in intervals["comparisons"] if r["cohort"] == role)
            np.testing.assert_allclose(np.quantile(means, [.025, .975], method="linear"),
                                      [found["CI95_lower"], found["CI95_upper"]], atol=1e-12)
    a.preserved(protocol)
    return {"passed": True, "checked_at_utc": now(), "development_videos": 100, "exploratory_videos": 40,
        "development_examples": 900, "exploratory_examples": 360, "saved_predictions": 7560,
        "dense_periodic_reproduced_rows": 280, "clip_rows": 420, "independent_explicit_counterfactual_replays": 1260,
        "source_grouped_class_stratified_folds": True, "training_only_scalers_and_independent_ridge_solution": True,
        "input_isolation_mutations": isolation_checks, "independent_video_MAE_and_paired_bootstrap": True,
        "frozen_classifier_and_controller_unchanged": True, "previous_tracked_files_preserved": len(protocol["previous_tracked_blob_identities"]),
        "fresh_confirmation_arrays_or_results_parsed": 0, "actual_RGB_encoder_calls": 0,
        "video_decodes": 0, "downloads": 0, "recognition_training_calls": 0}
