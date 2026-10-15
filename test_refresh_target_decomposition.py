"""Synthetic software checks only; no synthetic experimental observations."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

import refresh_target_decomposition_utils as u
from task_aware_oracle_utils import feature_logits
from verify_refresh_target_decomposition import explicit_replay


class DecompositionChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(72)
        self.features = rng.normal(size=(16, 512)).astype(np.float32)
        self.head = {"means": [0.]*512, "scales": [1.]*512, "bias": [0.]*20,
                     "weights": (rng.normal(size=(20, 512))*.02).tolist()}
        self.logits = feature_logits(self.features, self.head)

    def test_all_nine_schedules_cache_boundaries_and_loss(self):
        for r in [4, 8, 12]:
            for t in range(r-3, r):
                plans = u.schedules(t, r)
                cache = {}
                for name, positions in plans.items():
                    self.assertEqual(len(positions), u.COUNTS[name])
                    self.assertEqual(len(set(positions)), u.COUNTS[name])
                    actual, cache[name] = u.replay(self.logits, positions, 4)
                    independent, history, outputs = explicit_replay(self.features, positions, self.head, 4)
                    np.testing.assert_array_equal(cache[name], history)
                    self.assertEqual(outputs.shape, (16, 512))
                    self.assertAlmostEqual(actual, independent, places=12)
                u.check_boundaries(cache, t, r)
                if r == 12:
                    self.assertTrue((cache["B"][r:] == r).all())
                    self.assertTrue((cache["C"][r:] == t).all())

    def test_state_reset_and_ordinal_validation(self):
        plans = u.schedules(2, 4)
        first, cache = u.replay(self.logits, plans["A"], 2)
        u.replay(self.logits, plans["B"], 2)
        u.replay(self.logits, plans["D"], 2)
        again, after = u.replay(self.logits, plans["A"], 2)
        self.assertEqual(first, again)
        np.testing.assert_array_equal(cache, after)
        with self.assertRaises(ValueError):
            u.schedules(20, 80)
        with self.assertRaises(ValueError):
            u.schedules(4, 4)

    def test_signed_order_sensitive_fixture_and_averaged_identity(self):
        values = u.attributions(1., .7, 1.1, .9)
        self.assertTrue(values["forward_cancellation"])
        self.assertFalse(values["reverse_cancellation"])
        self.assertTrue(values["benefit_sign_changed"])
        self.assertLess(values["removal_first_cost"], 0)
        self.assertAlmostEqual(values["average_addition_benefit"]-values["average_removal_cost"], -.1)
        # A signed negative benefit and negative cost must not be clipped.
        values = u.attributions(1., 1.2, .9, 1.3)
        self.assertLess(values["early_addition_benefit"], 0)
        self.assertLess(values["later_removal_cost"], 0)

    def fixture(self, selected: bool) -> pd.DataFrame:
        rows = []
        for source in ["a", "b"]:
            for i in range(9):
                losses = (1., .7, 1.1, .9) if source == "a" else (1., 1.1, .9, 1.3)
                rows.append({"cohort": "development", "source_video_id": source, "video_id": source,
                    "class": "synthetic", "example_id": f"{source}:{i}",
                    "C_nonlinear_selected": selected and source == "a", **u.attributions(*losses)})
        return pd.DataFrame(rows)

    def test_conditional_bootstrap_zero_denominators_and_whole_videos(self):
        video = u.video_summary(self.fixture(True))
        aggregate, intervals, arrays = u.summarize(video, 20261019, 200)
        self.assertEqual(arrays["development"].shape, (200, 2))
        row = next(r for r in aggregate["rows"] if r["scope"] == "all" and r["order"] == "forward")
        self.assertEqual(row["rates"]["cancellation_all"]["numerator"], 9)
        self.assertEqual(row["rates"]["cancellation_all"]["denominator"], 18)
        selected = next(r for r in aggregate["rows"] if r["scope"] != "all" and r["order"] == "forward")
        self.assertEqual(selected["rates"]["cancellation_all"]["value"], 1.)
        self.assertGreater(selected["rates"]["cancellation_all"]["undefined_resamples"], 0)
        self.assertEqual(selected["continuous"]["mean_benefit"]["defined_sources"], 1)
        none, _, _ = u.summarize(u.video_summary(self.fixture(False)), 20261019, 200)
        selected = next(r for r in none["rows"] if r["scope"] != "all" and r["order"] == "forward")
        self.assertIsNone(selected["rates"]["cancellation_all"]["value"])
        self.assertIsNone(selected["rates"]["cancellation_all"]["CI95_lower"])
        self.assertIsNone(selected["continuous"]["mean_benefit"]["value"])


if __name__ == "__main__":
    unittest.main()
