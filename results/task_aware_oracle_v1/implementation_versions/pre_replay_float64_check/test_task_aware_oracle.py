"""Focused checks for classifier isolation, exact search, replay and paired metrics."""
from __future__ import annotations

from itertools import combinations
from unittest import TestCase, main

import numpy as np
import torch

import task_aware_oracle_utils as u


class TaskAwareOracleTests(TestCase):
    def test_cache_latest_exact_budget(self):
        for budget in [2, 4, 8, 16]:
            positions, caches, weights = u.schedule_table(16, budget)
            self.assertEqual(len(positions), {2: 15, 4: 455, 8: 6435, 16: 1}[budget])
            self.assertTrue((positions[:, 0] == 0).all())
            for row in [0, len(positions)//2, len(positions)-1]:
                np.testing.assert_array_equal(caches[row], u.cache_indices(positions[row].tolist()))
                self.assertEqual(len(set(positions[row])), budget)
            np.testing.assert_allclose(weights.sum(axis=1), 1)

    def test_cross_entropy_independent_formula_and_torch(self):
        logits = np.array([[1000., 999., 998.], [-1000., -999., -998.]], dtype=float)
        expected = np.log(np.exp(logits-logits.max(axis=1, keepdims=True)).sum(axis=1)) + logits.max(axis=1)-logits[:, 1]
        np.testing.assert_allclose(u.classification_loss(logits, 1), expected, atol=1e-12)
        np.testing.assert_allclose(u.classification_loss(logits, 1), torch.nn.functional.cross_entropy(torch.tensor(logits), torch.ones(2, dtype=torch.int64), reduction="none").numpy(), atol=1e-12)

    def test_exact_task_oracle_small_independent_enumeration(self):
        logits = np.random.default_rng(17).normal(size=(6, 3))
        for budget in [2, 3, 5]:
            direct = []
            for tail in combinations(range(1, 6), budget-1):
                positions = [0, *tail]; r = 0; outputs = []
                for t in range(6):
                    if t in positions: r = t
                    outputs.append(logits[r])
                pooled = np.mean(outputs, axis=0)
                loss = np.log(np.exp(pooled-pooled.max()).sum())+pooled.max()-pooled[1]
                direct.append((loss, positions))
            best = min(x[0] for x in direct)
            winner = next(p for loss, p in direct if loss <= best+u.TOL)
            actual = u.exact_task_oracle(logits, 1, budget)
            self.assertEqual(actual["positions"], winner)
            self.assertAlmostEqual(actual["minimum_cross_entropy"], best, places=12)

    def test_oracle_tie_rule(self):
        oracle = u.exact_task_oracle(np.zeros((16, 3)), 0, 4)
        self.assertEqual(oracle["positions"], [0, 1, 2, 3])

    def test_affine_pooling_matches_explicit_features(self):
        rng = np.random.default_rng(22); features = rng.normal(size=(16, 5))
        parameters = {"means": [1.]*5, "scales": [2.]*5, "weights": rng.normal(size=(3, 5)).tolist(), "bias": [1., 2., 3.]}
        cached = u.cache_indices([0, 4, 8, 12])
        np.testing.assert_allclose(u.feature_logits(features, parameters)[cached].mean(axis=0), u.feature_logits(features[cached].mean(axis=0)[None], parameters)[0], atol=1e-12)

    def test_deterministic_training_constant_column_and_train_only_scaling(self):
        x = np.array([[0., 7., -1.], [1., 7., 1.], [2., 7., -1.], [3., 7., 1.]])
        y = np.array([0, 1, 0, 1])
        a, info = u.fit_classifier(x, y, .1, 23); b, _ = u.fit_classifier(x, y, .1, 23)
        self.assertEqual(a, b); self.assertEqual(info["constant_input_columns"], 1)
        np.testing.assert_array_equal(a["means"], x.mean(axis=0)); self.assertEqual(a["scales"][1], 1)
        heldout_targets = np.array([0, 1]); heldout = np.array([[100., 7., 1.], [-100., 7., -1.]])
        before = u.feature_logits(heldout, a)
        heldout_targets[:] = [1, 0]
        np.testing.assert_array_equal(before, u.feature_logits(heldout, a))
        self.assertEqual(a, b)

    def test_shared_whole_video_bootstrap_against_loop(self):
        deltas = np.array([.2, -.1, .5, 0.])
        indices = np.random.default_rng(24).integers(0, 4, size=(10000, 4))
        actual = u.bootstrap_difference(deltas, indices)
        expected = np.quantile([sum(deltas[row])/4 for row in indices], [.025, .975], method="linear")
        np.testing.assert_allclose([actual["CI95_lower"], actual["CI95_upper"]], expected)
        self.assertEqual((actual["improved"], actual["tied"], actual["worsened"]), (2, 1, 1))

    def test_constant_correlation_missing(self):
        self.assertEqual(u.rank_correlation(np.zeros(4), np.arange(4))[0], None)
        self.assertAlmostEqual(u.rank_correlation(np.arange(4), np.arange(4))[0], 1)


if __name__ == "__main__":
    main()
