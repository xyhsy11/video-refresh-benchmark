"""Small information/implementation fixtures, never research measurements."""
from __future__ import annotations

import os
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "1"

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import cached_history_signal_utils as u
from verify_cached_history_signals import independent_features


class CachedHistoryChecks(unittest.TestCase):
    def test_exact_formulas_missing_history_and_ties(self):
        rng = np.random.default_rng(3)
        scores = rng.normal(size=(3, 20))
        history = u.CachedHistory()
        for i in range(3):
            got = history.advance(i*4, scores[i]).copy()
            np.testing.assert_allclose(got, independent_features(list(scores[:i+1])), atol=u.TOL, rtol=0)
            self.assertEqual(history.count, i+1)
            np.testing.assert_array_equal(history.advance(i*4, scores[i]), got)
            self.assertEqual(history.count, i+1)
        np.testing.assert_array_equal(u.features_from_available_history(scores[:1]), np.zeros(44))
        np.testing.assert_array_equal(u.features_from_available_history(scores[:2])[20:40], np.zeros(20))
        margin, top = u.margin_and_top(np.full(20, 1e4))
        self.assertEqual((margin, top), (0., 0))
        self.assertEqual(len(u.T_COLUMNS), 44)
        self.assertEqual(len(u.AUGMENTED_COLUMNS), 100)
        # New refresh events still count if their observed scores happen to match.
        same = u.CachedHistory()
        same.advance(0, np.zeros(20))
        unchanged = same.advance(4, np.zeros(20)).copy()
        self.assertEqual(same.count, 2)
        self.assertEqual(unchanged[40], 1.)
        self.assertEqual(unchanged[41], 0.)
        np.testing.assert_array_equal(unchanged[:40], np.zeros(40))

    def test_two_history_recovery_and_three_history_order_audit(self):
        rng = np.random.default_rng(11)
        a, b = rng.normal(size=(2, 20))
        for t in [5, 6, 7]:
            mean = (4*a+(t-4)*b)/t
            np.testing.assert_allclose((t*mean-(t-4)*b)/4, a, atol=u.TOL)
        fixture = u.information_fixture()
        self.assertTrue(fixture["old_C_unchanged"])
        self.assertTrue(fixture["T_changes"])
        self.assertIn("history_top1_switch_rate", fixture["changed_T_columns"])

    def test_prefix_isolation_and_video_reset(self):
        rng = np.random.default_rng(16)
        a, b, future = rng.normal(size=(3, 20))
        h = u.CachedHistory()
        first, second = h.advance(0, a).copy(), h.advance(4, b).copy()
        changed = u.CachedHistory()
        np.testing.assert_array_equal(changed.advance(0, a), first)
        np.testing.assert_array_equal(changed.advance(4, b), second)
        changed.advance(8, future*1e4)
        np.testing.assert_array_equal(second, h.summary)
        fresh = u.CachedHistory()
        np.testing.assert_array_equal(fresh.advance(0, future), np.zeros(44))
        self.assertEqual(fresh.count, 1)
        self.assertEqual(h.memory_accounting()["ndarray_payload_bytes"], 904)

    def test_observation_builder_ignores_privileged_and_outcome_columns(self):
        rows = []
        scores = np.zeros((3, 20)); scores[0, 0] = 2; scores[1, 1] = 4; scores[2, 2] = 3
        scores -= scores.mean(axis=1)[:, None]
        for t in [1, 2, 3, 5, 6, 7, 9, 10, 11]:
            i, r = t//4, (t//4)*4
            past = scores[0] if i == 0 else (4*scores[0]+(t-4)*scores[1])/t if i == 1 else (4*scores[0]+4*scores[1]+(t-8)*scores[2])/t
            row = dict.fromkeys(u.C_COLUMNS, 0.)
            row.update(zip(u.CACHED_COLUMNS, scores[i])); row.update(zip(u.PAST_COLUMNS, past))
            rows.append({**row, "cohort": "software_fixture", "example_id": str(t), "video_id": "fixture", "source_video_id": "fixture", "t": t, "last_refresh": r})
        table = pd.DataFrame(rows)
        expected, info = u.augment_observations(table)
        self.assertEqual(info["distinct_history_updates"], 3)
        for name in ["G", "loss_A", "loss_C", "label", "class", "privileged_current_centered_logit_class_00", "future_RGB"]:
            table[name] = np.nan
        got, _ = u.augment_observations(table)
        pd.testing.assert_frame_equal(got, expected)

    def test_component_pass_includes_all_events_and_batch_one_calls(self):
        import cached_history_cpu as cpu
        videos = []
        for _ in range(2):
            video = []
            for r in [0, 0, 0, 4, 4, 4, 8, 8, 8]:
                q = np.arange(20, dtype=float)*r
                row = np.zeros(56); row[16:36] = q
                video.append(cpu.Observation(r, q, row))
            videos.append(video)
        class Forest:
            def __init__(self): self.shapes = []
            def predict(self, x): self.shapes.append(x.shape); return np.zeros(1)
        class History(u.CachedHistory):
            resets, calls, updates = 0, 0, 0
            def __init__(self): super().__init__(); History.resets += 1
            def advance(self, r, q):
                History.calls += 1
                before = self.count
                answer = super().advance(r, q)
                History.updates += self.count-before
                return answer
        baseline, augmented = Forest(), Forest()
        predictors = {"C_RF400": (baseline, np.zeros(56), np.ones(56)), "C_plus_T_RF400": (augmented, np.zeros(100), np.ones(100))}
        with patch.object(cpu.u, "CachedHistory", History):
            cpu.component_pass(videos, predictors, "history_T")
            cpu.component_pass(videos, predictors, "baseline_scale_RF1")
            cpu.component_pass(videos, predictors, "augmented_history_scale_RF1")
        self.assertEqual((History.resets, History.calls, History.updates), (4, 36, 12))
        self.assertEqual(baseline.shapes, [(1, 56)]*18)
        self.assertEqual(augmented.shapes, [(1, 100)]*18)

    def test_equal_video_utility_and_shared_bootstrap(self):
        rows = []
        for source, y in [("one", np.arange(9)-4.), ("two", np.arange(9)-3.)]:
            for method in u.METHODS:
                prediction = np.ones(9) if method == "C_plus_T_RF400" else np.zeros(9)
                for i in range(9):
                    selected = prediction[i] > u.TOL
                    rows.append({"cohort": "software_fixture", "method": method, "source_video_id": source,
                        "example_id": source+str(i), "history_length": i//3+1, "G": y[i], "prediction": prediction[i],
                        "training_median_prediction": 0., "selected": selected, "realized_gain": y[i]*selected})
        aggregate, video, history, intervals, matrices = u.summarize(pd.DataFrame(rows), 20261021, 100)
        primary = next(r for r in intervals["comparisons"] if r["baseline"] == "C_RF400" and r["history_length"] == 0 and r["metric"] == "utility")
        self.assertAlmostEqual(primary["mean"], .5)
        self.assertEqual(matrices["software_fixture"].shape, (100, 2))
        self.assertEqual(len(video), 10)
        self.assertEqual(len(history), 30)
        keep = next(r for r in aggregate["metrics"] if r["method"] == "keep_reference" and r["history_length"] == 0)
        self.assertEqual(keep["utility"]["mean"], 0.)
        self.assertIsNone(keep["mean_selected_gain"])


if __name__ == "__main__":
    unittest.main()
