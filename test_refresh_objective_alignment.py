"""Explicitly SYNTHETIC unit fixtures; none are experimental observations."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import refresh_pooling_utils as pool
import refresh_objective_alignment_utils as obj


class ObjectiveFixtures(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(37)
        self.f = rng.normal(size=(16, 7))
        self.head = dict(means=rng.normal(size=7).tolist(), scales=rng.uniform(.4, 2., 7).tolist(),
                         weights=rng.normal(size=(20, 7)).tolist(), bias=rng.normal(size=20).tolist())
        self.table = pool.schedules(16, 4)

    def test_independent_objective_loops_pooling_and_jensen(self):
        result = obj.evaluate(self.f, self.head, 3, self.table)
        for j in [0, 1, 107, 288, 454]:
            checked = obj.scalar_check(self.f, self.head, 3, self.table['positions'][j].tolist(), result, j)
            self.assertLess(checked['maximum_difference'], 1e-10)

    def test_centered_logit_common_shift_and_affine_standardization(self):
        a = obj.evaluate(self.f, self.head, 3, self.table)
        shifted = copy.deepcopy(self.head)
        w = np.asarray(shifted['weights']) + np.arange(7)[None, :]*.3
        shifted['weights'] = w.tolist()
        shifted['bias'] = (np.asarray(shifted['bias'])+4.).tolist()
        b = obj.evaluate(self.f, shifted, 3, self.table)
        for name in ['Z_local', 'Z_pool', 'loss_C', 'loss_U']:
            np.testing.assert_allclose(a[name], b[name], atol=1e-10, rtol=1e-10)
        raw_w = np.asarray(self.head['weights']) / self.head['scales']
        raw_b = np.asarray(self.head['bias'])-raw_w @ self.head['means']
        np.testing.assert_allclose(self.f @ raw_w.T+raw_b, pool.logits(self.f, self.head), atol=1e-10)

    def test_schedule_counts_independent_minima_and_lex_ties(self):
        self.assertEqual([len(pool.schedules(16, b)['positions']) for b in [2, 4, 8]], [15, 455, 6435])
        result = obj.evaluate(self.f, self.head, 3, self.table)
        # Scalar explicit cache loops determine each objective independently.
        x = (self.f-self.head['means'])/self.head['scales']
        w = np.asarray(self.head['weights'])
        scalar = {name: [] for name in obj.OBJECTIVES}
        for positions in self.table['positions'].tolist():
            errors = []
            last = 0
            for t in range(16):
                if t in positions:
                    last = t
                errors.append(x[last]-x[t])
            e = np.array(errors)
            a = np.array([w @ row for row in e]); a -= a.mean(axis=1, keepdims=True)
            for prefix, values in [('F', e), ('Z', a)]:
                scalar[prefix+'_local'].append(float(np.square(values).mean()))
                scalar[prefix+'_pool'].append(float(np.square(values.mean(0)).mean()))
        for key in obj.OBJECTIVES:
            np.testing.assert_allclose(result[key], scalar[key], atol=1e-10, rtol=1e-10)
            minimum = min(scalar[key])
            independent = next(j for j, value in enumerate(scalar[key]) if value <= minimum+1e-12)
            self.assertEqual(independent, result['choices'][key]['index'])
        identical = np.repeat(self.f[:1], 16, axis=0)
        tied = obj.evaluate(identical, self.head, 3, self.table)
        for key in obj.OBJECTIVES:
            self.assertEqual(tied['choices'][key]['index'], 0)
            self.assertEqual(tied['choices'][key]['tie_count'], 455)
        self.assertEqual(pool.first_tie(np.array([2e-12, 0., .5e-12]))['index'], 1)

    def test_historical_raw_cosine_and_random_expectation(self):
        value = obj.evaluate(self.f, self.head, 3, self.table)
        d = np.empty((16, 16))
        for t in range(16):
            for r in range(16):
                d[t, r] = 1-np.clip(np.dot(self.f[t], self.f[r])/(np.linalg.norm(self.f[t])*np.linalg.norm(self.f[r])), -1, 1)
        expected = [sum(d[t, int(r)] for t, r in enumerate(cache))/16 for cache in self.table['cache']]
        np.testing.assert_allclose(value['LegacyCosine'], expected, atol=1e-12, rtol=0)
        self.assertAlmostEqual(float(value['loss_C'].mean()), sum(float(x) for x in value['loss_C'])/455, places=12)

    def test_seal_rejects_mutation_and_missing_real_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root/'real_identity.txt'
            path.write_text('original', encoding='utf-8')
            seal = {'sha256': {'real_identity.txt': pool.sha(path)}}
            obj.verify_seal(root, seal)
            path.write_text('changed', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                obj.verify_seal(root, seal)
            path.unlink()
            with self.assertRaises(RuntimeError):
                obj.verify_seal(root, seal)

    def test_source_stratification_and_objective_label_isolation(self):
        sources = ['a0', 'a1', 'b0', 'b1']
        labels = dict(zip(sources, ['a', 'a', 'b', 'b']))
        idx = pool.stratified_indices(sources, labels, ['a', 'b'], 100, 20261025)
        self.assertTrue(np.all((idx < 2).sum(axis=1) == 2))
        a, b = obj.evaluate(self.f, self.head, 1, self.table), obj.evaluate(self.f, self.head, 17, self.table)
        for key in ['LegacyCosine', *obj.OBJECTIVES]:
            np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(a['choices'][key], b['choices'][key])

    def test_missing_video_stream_is_input_rejection_not_hidden_programming_error(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from refresh_objective_alignment_data import inspect_technical
        with patch('refresh_objective_alignment_data.inspect_encoded', side_effect=IndexError('fixture')):
            with patch('av.open') as mocked:
                mocked.return_value.__enter__.return_value = SimpleNamespace(streams=SimpleNamespace(video=[]))
                with self.assertRaisesRegex(ValueError, 'no video stream'):
                    inspect_technical(Path('SYNTHETIC_NO_VIDEO_FIXTURE'))
                mocked.return_value.__enter__.return_value = SimpleNamespace(streams=SimpleNamespace(video=[object()]))
                with self.assertRaises(IndexError):
                    inspect_technical(Path('SYNTHETIC_PROGRAM_ERROR_FIXTURE'))


if __name__ == '__main__':
    unittest.main()
