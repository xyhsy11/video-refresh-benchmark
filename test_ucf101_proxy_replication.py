"""Explicitly SYNTHETIC unit tests; no fixture is a scientific measurement."""
import copy
import inspect
from pathlib import Path
import tempfile
import unittest
import zipfile
import io
from unittest.mock import patch
import numpy as np
import pandas as pd
import refresh_pooling_utils as pool
from temporal_refresh_common import snippet_grid
from ucf101_proxy_common import verify_hashes, sha, ROOT, Blocker, rank
from ucf101_proxy_data import parse_metadata, require_population
from ucf101_proxy_math import *


class SyntheticProtocolTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(52)
        self.f = rng.normal(size=(16, 9))
        self.head = dict(means=rng.normal(size=9), scales=rng.uniform(.3, 2., 9),
                         weights=rng.normal(size=(101, 9)), bias=rng.normal(size=101))
        self.table = pool.schedules(16, 4)

    def test_generic101_affine_pooling_scalar_objectives_and_cancellation(self):
        values = select_proxies(self.f, self.head, self.table)
        self.assertEqual(values['candidate_logits'].shape, (16, 101))
        for j in [0, 17, 234, 454]:
            self.assertLess(scalar_replay(self.f, self.head, 100, self.table['positions'][j].tolist(), values, j), 1e-10)
        for j in [0, 234]:
            cache = self.table['cache'][j]
            x = (self.f-self.head['means'])/self.head['scales']
            e = x[cache]-x
            w = self.head['weights']
            a = e @ w.T; a -= a.mean(1, keepdims=True)
            self.assertAlmostEqual(values['objectives']['Z_local'][j], np.square(a).mean(), places=12)
            self.assertAlmostEqual(values['objectives']['Z_pool'][j], np.square(a.mean(0)).mean(), places=12)

    def test_label_isolation_and_exact_minima_ties(self):
        self.assertNotIn('label', inspect.signature(select_proxies).parameters)
        values = select_proxies(self.f, self.head, self.table)
        for key in PROXIES:
            a = values['objectives'][key]
            self.assertEqual(values['choices'][key]['index'], next(i for i, v in enumerate(a) if v <= min(a)+1e-12))
        tied = select_proxies(np.repeat(self.f[:1], 16, axis=0), self.head, self.table)
        for key in PROXIES:
            self.assertEqual(tied['choices'][key]['index'], 0)
            self.assertEqual(tied['choices'][key]['tie_count'], 455)
        self.assertEqual([len(pool.schedules(16, b)['positions']) for b in [2, 4, 8]], [15, 455, 6435])

    def test_raw_cosine_and_candidate_reuse_counts(self):
        v = select_proxies(self.f, self.head, self.table)
        for j in [0, 115, 454]:
            explicit = sum(1-np.clip(np.dot(self.f[t], self.f[r])/(np.linalg.norm(self.f[t])*np.linalg.norm(self.f[r])), -1, 1)
                           for t, r in enumerate(self.table['cache'][j]))/16
            self.assertAlmostEqual(explicit, v['objectives']['LegacyCosine'][j], places=12)
            for t in range(16):
                self.assertEqual(self.table['cache_weights'][j, t], np.sum(self.table['cache'][j] == t)/16)

    def test_stable_exact_decomposition_direction_plus_KL(self):
        rng = np.random.default_rng(20)
        for scale in [1., 500.]:
            dense, a, b = rng.normal(size=(3, 101))*scale
            da, db = decomposition(dense, a, 100), decomposition(dense, b, 100)
            self.assertAlmostEqual(da['CE']-db['CE'], da['directional']-db['directional']+da['KL']-db['KL'], places=9)
            self.assertGreaterEqual(da['KL'], -1e-10)
            shifted = decomposition(dense+6000, a+999, 100)
            self.assertAlmostEqual(da['CE'], shifted['CE'], places=9)

    def test_weighted_class_group_clip_moments(self):
        entries = [dict(label=c, group=g) for c, groups in [('B', ['g01']), ('A', ['g01', 'g02'])] for g in groups for _ in range(2)]
        sw = hierarchical_weights(entries, ['B', 'A'])
        np.testing.assert_allclose(sw, [.25, .25, .125, .125, .125, .125])
        frame = pd.DataFrame(entries).assign(value=[1, 3, 5, 7, 9, 11])
        self.assertAlmostEqual(float(sw @ frame.value), balanced(frame, ['value'])['value'])
        head, record = fit_head(np.arange(18).reshape(6, 3), np.array([0, 0, 1, 1, 1, 1]), entries, ['B', 'A'], .1)
        expected = sw @ np.arange(18).reshape(6, 3)
        np.testing.assert_allclose(head['means'], expected, atol=1e-12)
        self.assertLess(record['gradient_max_abs'], 1e-4)

    def test_group_bootstrap_unequal_counts_nonalpha_class_order(self):
        groups = pd.DataFrame(dict(label=['A', 'A', 'B'], group=['g01', 'g02', 'g01']))
        canonical, indices, weights = bootstrap_plan(groups, ['B', 'A'], repeats=200)
        self.assertEqual(canonical.label.tolist(), ['B', 'A', 'A'])
        self.assertTrue((indices[:, 0] == 0).all())
        self.assertTrue((indices[:, 1:] >= 1).all())
        r = interval(np.array([8., 1., 3.]), indices, weights)
        self.assertAlmostEqual(r['mean'], 5.)
        np.testing.assert_array_equal(indices, bootstrap_plan(groups, ['B', 'A'], repeats=200)[1])

    def test_real_PTS_floor_math_on_labeled_synthetic_fixture(self):
        pts = np.arange(160)/25
        grid = snippet_grid(pts)
        requested = np.array(grid['requested_timestamps'])
        actual = np.array(grid['actual_timestamps'])
        self.assertTrue((actual <= requested+1e-9).all())
        for i, row in enumerate(grid['frame_indices']):
            for j, r in enumerate(row):
                self.assertEqual(r, max(k for k, p in enumerate(pts) if p <= requested[i, j]+1e-9))
        self.assertEqual(len(set(grid['endpoint_indices'])), 16)
        with self.assertRaises(ValueError):
            snippet_grid([0., 0., 1.])
        with self.assertRaises(ValueError):
            snippet_grid(np.arange(20)/30)

    def test_native_AVI_presentation_order_preserves_real_frame_identity(self):
        from ucf101_proxy_data import presentation_order
        frames = ['a', 'b', 'd', 'c']
        pts = [1., 2., 4., 3.]
        identities = [dict(index=i, pts=int(t), time_base='1/25') for i, t in enumerate(pts)]
        f, p, ids, n = presentation_order(frames, pts, identities)
        self.assertEqual(f, ['a', 'b', 'c', 'd'])
        self.assertEqual(p, [1., 2., 3., 4.])
        self.assertEqual([i['index'] for i in ids], [0, 1, 3, 2])
        self.assertEqual(n, 2)
        with self.assertRaises(ValueError):
            presentation_order(['a', 'b'], [1., 1.], identities[:2])

    def test_official_group_roles_and_class_scoped_g01(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'SYNTHETIC_METADATA.zip'
            classes = ['Class%03d' % i for i in range(101)]
            train, test = [], []
            for ci, c in enumerate(classes, 1):
                for g in range(1, 26):
                    for clip in range(1, 3):
                        name = f'{c}/v_{c}_g{g:02d}_c{clip:02d}.avi'
                        (test if g <= 7 else train).append(name if g <= 7 else name+' '+str(ci))
            def write(train_rows):
                with zipfile.ZipFile(p, 'w') as z:
                    z.writestr('ucfTrainTestlist/classInd.txt', '\n'.join(f'{i} {c}' for i, c in enumerate(classes, 1)))
                    z.writestr('ucfTrainTestlist/trainlist01.txt', '\n'.join(train_rows))
                    z.writestr('ucfTrainTestlist/testlist01.txt', '\n'.join(test))
            write(train)
            plan = parse_metadata(p)
            self.assertEqual(sum(v['role'] == 'development' for v in plan['clips']), 606)
            self.assertEqual(len({(v['label'], v['group']) for v in plan['clips']}), 2525)
            require_population(plan['clips'], classes)
            for c in classes:
                development = {v['group'] for v in plan['clips'] if v['label'] == c and v['role'] == 'development'}
                self.assertEqual(development, set(sorted([f'g{i:02d}' for i in range(8, 26)], key=lambda g: rank(c, g))[:3]))
            write(train+[test[0]+' 1'])
            with self.assertRaisesRegex(ValueError, 'GROUP leakage'):
                parse_metadata(p)

    def test_seal_enforcement(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = Path(tmp)/'SYNTHETIC_SEAL.txt'; p.write_text('one')
            records = {p.relative_to(ROOT).as_posix(): sha(p)}
            verify_hashes(records)
            p.write_text('two')
            with self.assertRaises(Blocker):
                verify_hashes(records)
            p.unlink()
            with self.assertRaises(Blocker):
                verify_hashes(records)

    def test_completed_resume_is_read_only(self):
        import run_ucf101_proxy_replication as runner
        from ucf101_proxy_common import save
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='SYNTHETIC_UNIT_ONLY_') as tmp:
            out = Path(tmp)
            save(out/'status.json', dict(current_stage='final', status='verified'))
            before = (out/'status.json').read_bytes()
            with patch.object(runner, 'OUT', out), patch.object(runner, 'run') as run, \
                 patch('ucf101_proxy_verify.audit', return_value={'synthetic_read_only_audit': True}), \
                 patch('sys.argv', ['runner', '--resume']), patch('sys.stdout', new_callable=io.StringIO):
                runner.main()
            run.assert_not_called()
            self.assertEqual(before, (out/'status.json').read_bytes())
            self.assertFalse((out/'run.lock').exists())

    def test_complete_real_feature_analysis_path_with_explicit_synthetic_arrays(self):
        import ucf101_proxy_analysis as analysis
        from ucf101_proxy_common import save
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='SYNTHETIC_UNIT_ONLY_') as tmp:
            base = Path(tmp); out, data = base/'outputs', base/'synthetic_arrays'
            entry = dict(clip='SYNTHETIC_clip.avi', label='C100', group='g01', role='test')
            classes = [f'C{i:03d}' for i in range(101)]
            head = {k: np.asarray(v).tolist() for k, v in self.head.items()}
            save(out/'heads/parameters.json', dict(parameters=head))
            save(out/'test_collection.json', {entry['clip']: dict(sha256='SYNTHETIC_ONLY')})
            with patch.object(analysis, 'OUT', out), patch.object(analysis, 'DATA', data), \
                 patch.object(analysis, 'read_features', return_value=self.f), patch.object(analysis, 'seal_check'):
                result = analysis.evaluate([entry], 'test', classes, head)
                self.assertEqual(set(result.budget), {2, 4, 8, 16})
                contrasts = analysis.pairs(result)
                np.testing.assert_allclose(contrasts.g, contrasts.directional_g+contrasts.KL_g, atol=1e-10)
                self.assertTrue((out/'test_prediction_artifacts.json').exists())
                # Resume uses genuine-checkpoint identity checks even in this labeled unit test.
                resumed = analysis.evaluate([entry], 'test', classes, head)
                self.assertEqual(len(result), len(resumed))
                # Explicit synthetic ALL-class unequal-group summary fixture.
                copies, offsets = [], []
                for ci, c in enumerate(classes):
                    ng = 2+ci % 2
                    offsets.append(.01*(ng-1)/2)
                    for gi in range(ng):
                        for clip in range(2):
                            f = result.copy()
                            f['label'] = c; f['group'] = f'g{gi:02d}'
                            f['clip'] = f'SYNTHETIC_{c}_{gi}_{clip}'
                            mask = f.selector == 'Z_pool'
                            f.loc[mask, 'CE'] += .01*gi
                            f.loc[mask, 'directional'] += .01*gi
                            copies.append(f)
                save(out/'stratum_thresholds.json', dict(dense_margin_terciles=[0., 1.]))
                primary = analysis.summarize(pd.concat(copies, ignore_index=True), classes)
                original_g = float(contrasts[(contrasts.budget == 4) & (contrasts.pool == 'C')].g.iloc[0])
                g = next(r for r in primary if r['contrast'] == 'g')
                self.assertAlmostEqual(g['mean'], original_g+np.mean(offsets), places=11)
                self.assertEqual(g['groups'], sum(2+i % 2 for i in range(101)))
                # Precision must also serialize its planning flags using native JSON types.
                synthetic_dev = pd.concat(copies, ignore_index=True).assign(role='development')
                planning = analysis.precision(synthetic_dev, dict(classes=classes,
                    counts={'test': {c: 7 for c in classes}}))
                self.assertIsInstance(planning['contrasts']['g']['target_met'], bool)
                self.assertTrue((out/'precision_estimates.json').is_file())


if __name__ == '__main__':
    unittest.main()
