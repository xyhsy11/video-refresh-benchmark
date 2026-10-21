"""Synthetic/software checks only. Never used to populate empirical results."""
import unittest
import numpy as np
import analyze_safe_schedule_density as a


class SafeDensityTests(unittest.TestCase):
    def test_enumeration_against_independent_combinations(self):
        from itertools import combinations
        tables = a.verify_schedules()
        for k in (4,8,16):
            expected = [(0,*rest) for rest in combinations(range(1,16),k-1)]
            self.assertEqual(tables[k]['positions'].tolist(),[list(p) for p in expected])
            for sid in (0,len(expected)//2,len(expected)-1):
                p = expected[sid]
                cache = [max(j for j in p if j <= t) for t in range(16)]
                self.assertEqual(tables[k]['cache'][sid].tolist(),cache)
                want = np.bincount(cache,minlength=16)/16
                np.testing.assert_array_equal(tables[k]['cache_weights'][sid],want)

    def test_stable_ce_and_random_is_not_ensemble(self):
        z = np.array([[1000.,1001.],[1000.,990.]])
        losses = a.ce(z,0)
        self.assertTrue(np.isfinite(losses).all())
        r,_,_ = a.density_record(z,0,z.mean(0),0.,0)
        self.assertEqual(r['RandomExact_CE'],float(losses.mean()))
        self.assertGreater(abs(r['RandomExact_CE']-float(a.ce(z.mean(0),0))),.1)

    def test_wrong_dense_agreement_is_not_safe(self):
        z = np.array([[0.,3.],[3.,0.],[0.,4.]])
        dense = np.array([0.,2.])
        r,_,_ = a.density_record(z,0,dense,0.,0)
        self.assertIsNone(r['P_safe'])
        self.assertEqual(r['correct_and_CE_count'],1)
        self.assertEqual(r['dense_agreement_count'],2)
        self.assertEqual(r['transition_D0_S0_count'],2)
        self.assertEqual(r['transition_D0_S1_count'],1)

    def test_epsilon_tolerance_and_joint_components(self):
        dense = np.array([3.,0.])
        z = np.array([[3.,0.],[2.9,0.],[2.8,0.],[0.,3.]])
        counts = []
        for eps in a.EPSILONS:
            r,_,_ = a.density_record(z,0,dense,eps,0)
            self.assertLessEqual(r['correct_and_CE_count'],r['correct_count'])
            self.assertLessEqual(r['correct_and_CE_count'],r['CE_nondegrade_count'])
            counts.append(r['correct_and_CE_count'])
        self.assertEqual(counts,sorted(counts))
        self.assertEqual(a.density_record(z,0,dense,0.,0)[0]['correct_and_CE_count'],1)

    def test_empty_condition_is_na(self):
        r = a.estimate(np.array([1.,0.]),np.array([False,False]),np.array([.5,.5]),None)
        self.assertIsNone(r['estimate'])
        self.assertIsNone(r['CI95_lower'])
        self.assertIsNone(a.weighted_quantiles(np.array([0.,1.]),np.array([0.,0.]))['median'])

    def test_hierarchy_and_paired_bootstrap(self):
        videos = [dict(clip='a',label='A',group='g1'),dict(clip='b',label='A',group='g1'),
                  dict(clip='c',label='A',group='g2'),dict(clip='d',label='B',group='g1')]
        np.testing.assert_array_equal(a.weights(videos),[.125,.125,.25,.5])
        groups,draws,bw = a.bootstrap(videos)
        self.assertEqual(groups,[('A','g1'),('A','g2'),('B','g1')])
        self.assertEqual(draws.shape,(2000,3))
        np.testing.assert_array_equal(bw[:,0],bw[:,1])
        np.testing.assert_array_equal(bw[:,3],np.full(2000,.5))
        # Independent RNG block construction verifies ordering and single-group behavior.
        rng = np.random.Generator(np.random.PCG64(20261021))
        expected = np.c_[rng.integers(2,size=(2000,2)),np.full(2000,2)]
        np.testing.assert_array_equal(draws,expected)

    def test_pilot_selection_ignores_outcomes(self):
        videos = [dict(clip=str(i),label=str(i),sha256=str(i)) for i in range(30)]
        provenance = {str(i):dict(original_feature=dict(sha256='feature'+str(i))) for i in range(30)}
        before = [v['clip'] for v in a.pilot_order(videos,provenance)]
        for v in videos:
            v['imaginary_outcome'] = -1
        self.assertEqual(before,[v['clip'] for v in a.pilot_order(videos,provenance)])


if __name__ == '__main__':
    unittest.main()
