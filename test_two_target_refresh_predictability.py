"""Small synthetic software fixtures; not experimental observations."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import two_target_refresh_utils as u


class TwoTargetChecks(unittest.TestCase):
    def test_explicit_whitelist_ignores_privileged_targets_and_ids(self):
        table = pd.DataFrame(np.arange(56*7, dtype=float).reshape(7, 56), columns=u.COLUMNS)
        x = u.design(table)
        for name in ["label", "source_video_id", "G", "B", "loss_A", "privileged_current_centered_logit_class_00", "future_RGB"]:
            table[name] = np.nan
        np.testing.assert_array_equal(x, u.design(table))
        self.assertEqual(len(u.COLUMNS), 56)
        self.assertEqual(u.COLUMNS, u.previous.SETS["C"])

    def test_linear_and_kernel_target_linearity_with_constant_column(self):
        rng = np.random.default_rng(21)
        x, query = rng.normal(size=(35, 6)), rng.normal(size=(8, 6))
        x[:, -1], query[:, -1] = 4., 999.
        b, c = rng.normal(size=(2, 35))
        for nonlinear in [False, True]:
            fitted = [u.previous.fit_array(x, y, nonlinear) for y in [b, c, b-c]]
            pb, pc, pg = [u.previous.predict_array(query, s, a) for s, a in fitted]
            np.testing.assert_allclose(pb-pc, pg, atol=u.ATOL, rtol=u.RTOL)
            self.assertEqual(fitted[0][0]["scales"][-1], 1.)
            np.testing.assert_array_equal(fitted[0][0]["means"], x.mean(axis=0))

    def test_training_only_kernel_centering_independently(self):
        rng = np.random.default_rng(4)
        x, q, y = rng.normal(size=(12, 3)), rng.normal(size=(4, 3)), rng.normal(size=12)
        state, arrays = u.previous.fit_array(x, y, True)
        z, zz = (x-x.mean(0))/x.std(0), (q-x.mean(0))/x.std(0)
        k = np.exp(-((z[:, None]-z[None])**2).sum(2)/3)
        h = np.eye(12)-np.ones((12, 12))/12
        dual = np.linalg.solve(h@k@h+np.eye(12), y-y.mean())
        query = np.exp(-((zz[:, None]-z[None])**2).sum(2)/3)
        centered = query-k.mean(0)-query.mean(1)[:, None]+k.mean()
        np.testing.assert_allclose(u.previous.predict_array(q, state, arrays), y.mean()+centered@dual, atol=u.ATOL)
        altered_labels = rng.normal(size=4)
        self.assertEqual(len(altered_labels), len(q))
        # No heldout target argument exists; extreme heldout X must not change the scaler.
        u.previous.predict_array(q*1e6, state, arrays)
        np.testing.assert_array_equal(state["means"], x.mean(0))

    def test_random_forests_use_separate_target_dependent_partitions(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(240, 4))
        b, c = x[:, 0]**2+2*x[:, 1], 5*x[:, 2]-x[:, 3]**2
        f = RandomForestRegressor(n_estimators=200, **u.RF_PARAMETERS).fit(x, b)
        g = RandomForestRegressor(n_estimators=200, **u.RF_PARAMETERS).fit(x, c)
        self.assertIsNot(f, g)
        self.assertTrue(any(not np.array_equal(a.tree_.feature, z.tree_.feature) or
                            not np.array_equal(a.tree_.threshold, z.tree_.threshold)
                            for a, z in zip(f.estimators_, g.estimators_)))
        rerun = RandomForestRegressor(n_estimators=200, **u.RF_PARAMETERS).fit(x, b)
        np.testing.assert_array_equal(f.predict(x), rerun.predict(x))

    def test_utility_uses_zero_unselected_and_nine_per_video(self):
        actual = np.array([2., -3., 1., -1., 0., 4., -5., 2., 1.])
        prediction = np.array([1., 1., -1., 1e-12, 2e-12, -1., 1., 1., -1.])
        selected = prediction > u.TOL
        self.assertEqual(selected.sum(), 5)
        self.assertAlmostEqual(float((actual*selected).mean()), -4/9)
        self.assertNotEqual(float((actual*selected).mean()), float(actual[selected].mean()))

    def test_uncentered_error_product_is_required(self):
        eb, ec = np.array([2., 4., 7.]), np.array([1., 3., 8.])
        mse = np.mean((eb-ec)**2)
        self.assertAlmostEqual(mse, np.mean(eb**2)+np.mean(ec**2)-2*np.mean(eb*ec))
        self.assertNotAlmostEqual(mse, np.mean(eb**2)+np.mean(ec**2)-2*np.cov(eb, ec, ddof=0)[0, 1])

    def test_source_bootstrap_keeps_nine_rows_together(self):
        by_video = np.array([-2., 1., 5.])
        indices = np.random.Generator(np.random.PCG64(20261020)).integers(0, 3, (10000, 3))
        explicit = np.repeat(by_video, 9).reshape(3, 9)[indices].mean(axis=(1, 2))
        np.testing.assert_allclose(explicit, by_video[indices].mean(axis=1))
        result = u.interval(by_video, indices)
        np.testing.assert_allclose([result["CI95_lower"], result["CI95_upper"]], np.percentile(explicit, [2.5, 97.5]))
        undefined = u.previous.discrimination(np.zeros(3), np.array([True, True, True]))
        self.assertIsNone(undefined["AUROC"])
        self.assertIsNotNone(undefined["undefined_reason"])


if __name__ == "__main__":
    unittest.main()
