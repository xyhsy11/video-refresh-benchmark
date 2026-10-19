"""SYNTHETIC TEST FIXTURES ONLY. No empirical UCF arrays or results are opened."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.special import logsumexp
import task_utility_observability as m
import task_utility_observability_study as s


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        # SYNTHETIC TEST FIXTURE, never scientific evidence.
        self.rng=np.random.Generator(np.random.PCG64(71))
        self.f=self.rng.normal(size=(16,7));self.head=dict(means=np.zeros(7),scales=np.ones(7),
            weights=self.rng.normal(size=(3,7)),bias=np.array([.2,.4,-.1]))
        self.z=m.affine(self.f,self.head)

    def test_all_455_exact_B4_schedules(self):
        self.assertEqual(len(m.POSITIONS),455)
        self.assertEqual(len(set(map(tuple,m.POSITIONS))),455)
        self.assertTrue(np.all(m.POSITIONS[:,0]==0))
        self.assertTrue(np.all(np.diff(m.POSITIONS,axis=1)>0))
        np.testing.assert_allclose(m.WEIGHTS.sum(1),1)

    def test_independent_completions_and_forced_decisions(self):
        loss=m.ce(m.WEIGHTS @ self.z,1)
        scalar=np.array([np.logaddexp.reduce(self.z[c].mean(0))-self.z[c].mean(0)[1] for c in m.CACHE])
        for schedule in [[0,4,8,12],[0,1,2,3],[0,13,14,15]]:
            for t in range(1,16):
                history=[p for p in schedule if p<t]
                r=m.task_targets(self.z,1,t,history,loss)
                local,q=m.independent_completion_targets(self.z,1,t,history,scalar)
                self.assertAlmostEqual(local,r['gain_local'],places=12)
                for name,value in zip(['Q_hold','Q_refresh'],q):
                    if value is None:self.assertIsNone(r[name])
                    else:self.assertAlmostEqual(value,r[name],places=12)
                self.assertEqual(m.discretionary(t,history),r['gain_budget'] is not None)

    def test_sign_conventions(self):
        z=np.zeros((16,2));z[1]=[3.,-3.]
        r=m.task_targets(z,0,1,[0]);self.assertGreater(r['gain_local'],0)
        self.assertAlmostEqual(r['gain_budget'],r['Q_hold']-r['Q_refresh'])
        self.assertEqual(m.ce(np.array([1000.,999.]),0),logsumexp([1000.,999.])-1000)

    def test_affine_cache_pooling(self):
        for i in [0,25,454]:
            raw=m.affine(self.f[m.CACHE[i]].mean(0),self.head)
            np.testing.assert_allclose(raw,m.WEIGHTS[i] @ self.z,atol=1e-12)

    def test_labels_never_enter_observation(self):
        obs=m.causal_inputs(4,[0],{0:self.f[0]},np.zeros(32))
        with self.assertRaises(RuntimeError):_ = obs.label
        with self.assertRaises(RuntimeError):_ = obs.current_feature
        self.assertNotIn('label',obs.__dict__)
        self.assertEqual(set(obs.__dict__),{'t','age','remaining','history','cheap'})

    def test_future_feature_rejection_and_invariance(self):
        with self.assertRaises(RuntimeError):m.causal_inputs(4,[0,4],{0:self.f[0],4:self.f[4]},np.zeros(32))
        with self.assertRaises(RuntimeError):m.causal_inputs(4,[0],{0:self.f[0],5:self.f[5]},np.zeros(32))
        a=m.causal_inputs(4,[0],{0:self.f[0]},np.zeros(32));before=m.observation_vector(a,self.head)
        self.f[4:]+=10000
        b=m.causal_inputs(4,[0],{0:self.f[0]},np.zeros(32))
        np.testing.assert_array_equal(before,m.observation_vector(b,self.head))

    def test_H_HM_shared_history_and_M_isolation(self):
        a=m.causal_inputs(4,[0],{0:self.f[0]},np.zeros(32));b=m.causal_inputs(4,[0],{0:self.f[0]},np.ones(32))
        xa,xb=m.observation_vector(a,self.head),m.observation_vector(b,self.head);ids=m.view_indices(3)
        np.testing.assert_array_equal(xa[ids['H']],xb[ids['H']])
        np.testing.assert_array_equal(xb[ids['HM']][:len(ids['H'])],xb[ids['H']])
        self.assertTrue(np.all(xa[ids['M']]!=xb[ids['M']]))
        self.assertEqual(len(m.view_indices(101)['HM']),146)

    def test_class_group_clip_state_weights(self):
        rows=[dict(label='A',group='g1',clip='a1')]*3+[dict(label='A',group='g1',clip='a2')]+[
            dict(label='A',group='g2',clip='a3')]+[dict(label='B',group='g1',clip='b1')]*2
        w=m.hierarchical_weights(rows)
        self.assertAlmostEqual(w[:3].sum(),.125);self.assertAlmostEqual(w[3],.125)
        self.assertAlmostEqual(w[4],.25);self.assertAlmostEqual(w[5:].sum(),.5)

    def test_group_bootstrap_pairing_and_determinism(self):
        rows=[dict(label=c,group=g,clip=c+g) for c in ['A','B'] for g in ['g1','g2'] for _ in range(2)]
        groups,draws=m.group_plan(rows,20,3);g2,d2=m.group_plan(rows,20,3)
        np.testing.assert_array_equal(draws,d2);self.assertEqual(groups,g2)
        w=m.bootstrap_weights(rows,groups,draws);np.testing.assert_allclose(w.sum(1),1)
        np.testing.assert_array_equal(w[:,::2],w[:,1::2])
        outcome=np.arange(len(rows));np.testing.assert_allclose(w @ (outcome+3)-w @ outcome,3)

    def test_split_and_exact_duplicate_safety(self):
        rows=[dict(clip=str(i),label='A',group='g'+str(i),study_role=r,sha256='hash'+str(i),
            official_path='A/'+str(i),official_role='official_train') for i,r in enumerate(['controller_train','selection','execution'])]
        members={r['official_path'] for r in rows};s.validate_roles(rows,['A'],members)
        for field in ['group','sha256']:
            bad=copy.deepcopy(rows);bad[1][field]=bad[0][field]
            with self.assertRaises(RuntimeError):s.validate_roles(bad,['A'],members)
        bad=copy.deepcopy(rows);bad[2]['official_role']='official_test'
        with self.assertRaises(RuntimeError):s.validate_roles(bad,['A'],members)

    def test_forbidden_official_test_and_history_writes(self):
        guard=s.Guard()
        for path in [s.OLD/'findings.md',s.OLD/'test_collection.json',s.SOURCE/'features/v_foreign.avi.npz']:
            with self.assertRaises(RuntimeError):guard.check(path)
        with self.assertRaises(RuntimeError):guard.check(s.PRIOR/'findings.md',True)
        guard.read_only=True
        with self.assertRaises(RuntimeError):guard.check(s.OUT/'anything.json',True)

    def test_identity_mutation_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'SYNTHETIC_TEST_FIXTURE.json';path.write_text('one')
            expected=s.sha(path);path.write_text('two')
            with self.assertRaises(RuntimeError):s.verify({str(path):expected})

    def test_resume_reauthenticates_historical_parent_trials(self):
        # SYNTHETIC TEST FIXTURE: every original parent must be checked before
        # any saved execution-array path can be reused, rather than trusting a
        # potentially changed new manifest's requested paths.
        from unittest.mock import patch
        clip='SYNTHETIC_TEST_FIXTURE.avi'
        roll={clip+'/'+name:{'id':name} for name in ['periodic','RandomB4_seeded','history_motion']}
        dense={clip:{'id':'dense'}};seen=[]
        with patch.object(s,'prior_trial',side_effect=lambda item,inputs:seen.append(item['id'])):
            s.authenticate_cached_execution(clip,roll,dense,{})
        self.assertEqual(seen,['dense','periodic','RandomB4_seeded','history_motion'])
        guard=s.Guard();path=s.ROOT/'data/source/ucf101_causal_development_v1/executed_features/changed.npz'
        with self.assertRaises(RuntimeError):guard.check(path)

    def test_probe_determinism_and_constant_base_rate(self):
        x=self.rng.normal(size=(30,4));gain=self.rng.normal(size=30);w=np.full(30,1/30)
        a=m.fit_linear(x,(gain>0).astype(float),w,.1,True);b=m.fit_linear(x,(gain>0).astype(float),w,.1,True)
        self.assertEqual(a,b);self.assertTrue(a['fit']['success'])
        c=m.fit_linear(x[:,:0],(gain>0).astype(float),w,0,True)
        np.testing.assert_allclose(m.predict(c,x[:,:0]),(gain>0).mean(),atol=1e-10)

    def test_metrics_ties_and_weighted_ranks(self):
        y=np.array([0.,1.,0.,1.]);w=np.full(4,.25)
        roc,ap=m.rank_auc(y,np.ones(4),w);self.assertAlmostEqual(roc,.5);self.assertAlmostEqual(ap,.5)
        self.assertAlmostEqual(m.weighted_spearman(np.arange(4),-np.arange(4),w),-1.)
        self.assertIsNone(m.weighted_spearman(np.ones(4),np.arange(4),w))
        # SYNTHETIC unequal hierarchy, reproducing a floating centering edge.
        unequal=np.r_[np.full(4752,1/4848),np.full(48,1/2424)]
        self.assertIsNone(m.weighted_spearman(np.arange(4800),np.ones(4800),unequal))
        self.assertEqual(m.rank_auc(np.zeros(4),np.arange(4),w),(None,None))


if __name__=='__main__':unittest.main()
