"""Focused synthetic software fixtures, excluded from scientific outputs."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import task_signal_ablation_utils as u
import task_signal_utils as prior
from refresh_policy_utils import CheapVideo, ReplayVideo
from task_aware_oracle_utils import feature_logits


class AblationChecks(unittest.TestCase):
    def test_kernel_centering_independent_H_and_intercept_system(self):
        rng = np.random.default_rng(51)
        x, y, query = rng.normal(size=(9, 4)), rng.normal(size=9), rng.normal(size=(3, 4))
        state, arrays = u.fit_array(x, y, True)
        z = (x-x.mean(axis=0))/x.std(axis=0)
        k = np.exp(-np.sum((z[:, None]-z[None, :])**2, axis=2)/4)
        h = np.eye(9)-np.ones((9, 9))/9
        expected = np.linalg.solve(h@k@h+np.eye(9), y-y.mean())
        np.testing.assert_allclose(arrays["dual_coefficients"], expected, atol=1e-12)
        q = (query-x.mean(axis=0))/x.std(axis=0)
        cross = np.exp(-np.sum((q[:, None]-z[None, :])**2, axis=2)/4)
        centered = cross-cross.mean(axis=1)[:, None]-k.mean(axis=0)[None, :]+k.mean()
        result = u.predict_array(query, state, arrays)
        np.testing.assert_allclose(result, y.mean()+centered@expected, atol=1e-12)
        saddle = np.block([[k+np.eye(9), np.ones((9, 1))], [np.ones((1, 9)), np.zeros((1, 1))]])
        solution = np.linalg.solve(saddle, np.r_[y, 0.])
        np.testing.assert_allclose(result, cross@solution[:9]+solution[9], atol=1e-12)
        np.testing.assert_allclose(result[:1], u.predict_array(query[:1], state, arrays), atol=1e-12)

    def test_linear_reproduces_prior_D_recipe(self):
        rng = np.random.default_rng(12)
        table = pd.DataFrame(rng.normal(size=(30, 16)), columns=u.SETS["S"])
        table["refresh_gain"] = rng.normal(size=30)
        table["example_id"] = [str(i) for i in range(30)]
        table["source_video_id"] = [f"s{i//3}" for i in range(30)]
        previous = prior.fit_ridge(table, "D")
        state, arrays = u.fit_array(table[u.SETS["S"]].to_numpy(), table.refresh_gain.to_numpy(), False)
        for key in ["means", "scales", "coefficients", "intercept"]:
            np.testing.assert_array_equal(state[key], previous[key])
        np.testing.assert_allclose(u.predict_array(table[u.SETS["S"]].to_numpy(), state, arrays),
                                   prior.predict(table, previous), atol=1e-12)

    def test_cached_and_current_interfaces_isolate_privileged_information(self):
        rng = np.random.default_rng(5)
        head = {"means": [0.]*512, "scales": [1.]*512, "bias": [0.]*20,
                "weights": (rng.normal(size=(20, 512))*.02).tolist()}
        rgb = rng.normal(size=(16, 512)).astype(np.float32)
        frames = tuple({"decoded_frame_index": i, "mv_available": False,
                        "mean_magnitude_px": 0., "p90_magnitude_px": 0.} for i in range(31))
        video = ReplayVideo({"label": "unused"}, CheapVideo("source", tuple(range(0, 32, 2)),
                tuple(np.arange(16)*.5), frames), rgb)
        base = u.cached_inputs(u.cached_observation(video, 6, head))
        mutated = rgb.copy()
        mutated[[i for i in range(16) if i not in [0, 4]]] = 1000.
        changed = replace(video, references=mutated, entry={"label": "different"})
        self.assertEqual(base, u.cached_inputs(u.cached_observation(changed, 6, head)))
        privileged = u.privileged_inputs(base, feature_logits(rgb[[6]], head)[0])
        other = u.privileged_inputs(base, feature_logits(mutated[[6]], head)[0])
        self.assertNotEqual(privileged, other)
        np.testing.assert_allclose([base[c] for c in u.SETS["C"][16:36]],
                                   u.centered(feature_logits(rgb[[4]], head)[0]), atol=1e-12)
        self.assertTrue(all(base[c] == 0 for c in prior.MOTION_INPUTS))
        self.assertFalse(set(u.SETS["C"]) & {"label", "refresh_gain", "dense_prediction"})

    def test_constant_training_columns_and_kernel(self):
        x = np.ones((8, 3)); y = np.arange(8, dtype=float)
        state, arrays = u.fit_array(x, y, True)
        np.testing.assert_array_equal(state["scales"], np.ones(3))
        np.testing.assert_allclose(u.predict_array(x[:2], state, arrays), np.full(2, y.mean()), atol=1e-12)
        self.assertTrue(np.isfinite(arrays["dual_coefficients"]).all())

    def test_heldout_targets_cannot_change_actual_fold_models(self):
        from analyze_task_signal_ablation import fit_oof
        rng = np.random.default_rng(42)
        table = pd.DataFrame(rng.normal(size=(90, 76)), columns=u.SETS["P"])
        table["source_video_id"] = [f"s{i//9}" for i in range(90)]
        table["video_id"] = [f"v{i//9}" for i in range(90)]
        table["example_id"] = [f"e{i}" for i in range(90)]
        table["t"] = np.tile([1, 2, 3, 5, 6, 7, 9, 10, 11], 10)
        table["last_refresh"] = 0
        table["refresh_gain"] = rng.normal(size=90)
        folds = {"folds": [{"fold": f, "training_sources": [f"s{i}" for i in range(10) if i%2 != f],
                           "validation_sources": [f"s{i}" for i in range(10) if i%2 == f]} for f in range(2)]}
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            fit_oof(table, folds, Path(left))
            mutated = table.copy()
            mutated.loc[mutated.source_video_id.isin(folds["folds"][0]["validation_sources"]), "refresh_gain"] += 500.
            fit_oof(mutated, folds, Path(right))
            import json
            for name in u.CONFIGS+["A-linear"]:
                folder1, folder2 = Path(left)/"models/fold_0", Path(right)/"models/fold_0"
                r1 = json.loads((folder1/"registry.json").read_text())
                r2 = json.loads((folder2/"registry.json").read_text())
                a, x = u.load_model(folder1, r1["models"][name])
                b, z = u.load_model(folder2, r2["models"][name])
                self.assertEqual(a, b)
                for key in x:
                    np.testing.assert_array_equal(x[key], z[key])
            a = pd.read_csv(Path(left)/"out_of_fold_predictions.csv")
            b = pd.read_csv(Path(right)/"out_of_fold_predictions.csv")
            np.testing.assert_array_equal(a[a.fold == 0].prediction, b[b.fold == 0].prediction)

    def test_decision_utility_whole_video_grouping_and_outcome_reference(self):
        y = np.array([-.4, -.2, 0., .1, .2, -.1, .3, .4, -.2])
        rows = []
        for source in ["a", "b"]:
            for name in u.PREDICTORS:
                score = np.array([0., 1., 1e-12, 1.1e-12, 1., -1., 1., -1., 0.])
                for i in range(9):
                    rows.append({**u.access(name), "cohort": "development_oof", "source_video_id": source,
                        "example_id": f"{source}:{i}", "refresh_gain": y[i], "prediction": score[i],
                        "realized_gain": y[i]*(score[i] > 1e-12)})
        summary, per_video, intervals, arrays = u.summarize(pd.DataFrame(rows), 20261018)
        actual = next(r for r in summary["configurations"] if r["configuration"] == "C-nonlinear")
        expected = np.mean(y*np.array([0, 1, 0, 1, 1, 0, 1, 0, 0]))
        self.assertAlmostEqual(actual["mean_realized_gain"], expected)
        self.assertEqual(arrays["development_oof"].shape, (10000, 2))
        oracle = next(r for r in summary["configurations"] if r["configuration"] == "hindsight_positive_gain_reference")
        self.assertAlmostEqual(oracle["mean_realized_gain"], np.maximum(y, 0).mean())
        keep = next(r for r in summary["configurations"] if r["configuration"] == "keep_reference")
        self.assertIsNone(keep["mean_gain_selected"])
        self.assertEqual(len(per_video), 24)

    def test_undefined_discrimination(self):
        self.assertIsNone(u.discrimination(np.zeros(4), np.array([0, 1, 0, 1]))["AUROC"])
        self.assertIsNone(u.discrimination(np.arange(4), np.ones(4))["AUROC"])


if __name__ == "__main__":
    unittest.main()
