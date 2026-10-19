"""SYNTHETIC TEST FIXTURE only: no videos, real measurements, or GPU calls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import ucf101_causal_development_common as c
import ucf101_causal_development as s
from causal_observability_policy import Policy,Observation,TrainingReplayProvider,execute,vector


class SyntheticDevelopmentTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(20261019)
        self.f=rng.normal(size=(16,7));self.thumbs=rng.integers(0,256,size=(16,32,32),dtype=np.uint8)
        self.head=dict(means=np.zeros(7),scales=np.ones(7),weights=rng.normal(size=(101,7)),bias=np.zeros(101))
    def test_exact_four_first0_and_scalar_logit_replay(self):
        provider=TrainingReplayProvider(self.f,self.thumbs)
        r=s.causal_execute(provider,s.LivePolicy('periodic',self.head),self.head)
        self.assertEqual(r['positions'],[0,4,8,12]);self.assertEqual(provider.forward_completions,4)
        cache=np.repeat(self.f[[0,4,8,12]],4,axis=0)
        z=np.mean([self.head['weights'] @ f for f in cache],axis=0)
        np.testing.assert_allclose(z,r['logits'],atol=1e-12)
    def test_forced_budget_feasibility_always_and_never(self):
        for choose,expected in [(True,[0,1,2,3]),(False,[0,13,14,15])]:
            class Gate:
                needs_cheap=False
                def decide(self,obs):return choose
            p=TrainingReplayProvider(self.f,self.thumbs);r=s.causal_execute(p,Gate(),self.head)
            self.assertEqual(r['positions'],expected);self.assertEqual(p.forward_completions,4)
    def test_live_executor_matches_existing_policy_definition(self):
        for method in ['periodic','RandomB4_seeded','motion']:
            a=execute(TrainingReplayProvider(self.f,self.thumbs),Policy(method,self.head,threshold=.1,clip='SYNTHETIC'),self.head)
            b=s.causal_execute(TrainingReplayProvider(self.f,self.thumbs),s.LivePolicy(method,self.head,threshold=.1,clip='SYNTHETIC'),self.head)
            self.assertEqual(a['positions'],b['positions']);np.testing.assert_allclose(a['logits'],b['logits'],atol=0,rtol=0)
    def test_chronology_and_prohibited_decision_access(self):
        case=self;observed=[]
        class Gate:
            needs_cheap=False
            def decide(self,obs):
                for key in ['label','future_features','current_feature']:
                    with case.assertRaises(RuntimeError):getattr(obs,key)
                observed.append(obs);return False
        r=s.causal_execute(TrainingReplayProvider(self.f,self.thumbs),Gate(),self.head)
        for row in r['trace']:self.assertTrue(all(j<row['t'] for j in row['available_refreshes']))
        self.assertTrue(all(any(np.array_equal(h,f) for f in self.f[:o.t]) for o in observed for h in o.history))
    def test_generic101_and_history_only_current_isolation(self):
        obs=Observation(2,2,3,(tuple(self.f[0]),),tuple(np.zeros(32)))
        other=Observation(2,2,3,obs.history,tuple(np.ones(32)))
        self.assertEqual(vector(obs,self.head,False).shape,(145,))
        np.testing.assert_array_equal(vector(obs,self.head,False),vector(other,self.head,False))
        self.assertFalse(np.array_equal(vector(obs,self.head,True),vector(other,self.head,True)))
        class NoThumb(TrainingReplayProvider):
            def thumbnail(self):raise AssertionError('SYNTHETIC: prohibited thumbnail')
        model=dict(means=np.zeros(145),scales=np.ones(145),w1=np.zeros((32,145)),b1=np.zeros(32),w2=np.zeros(32),b2=0.)
        s.causal_execute(NoThumb(self.f,self.thumbs),s.LivePolicy('history',self.head,model,.9),self.head)
    def test_independent_video_cache_and_seeded_random(self):
        a=s.causal_execute(TrainingReplayProvider(self.f,self.thumbs),s.LivePolicy('RandomB4_seeded',self.head,clip='SYNTHETIC'),self.head)
        b=s.causal_execute(TrainingReplayProvider(self.f*2,self.thumbs),s.LivePolicy('RandomB4_seeded',self.head,clip='SYNTHETIC'),self.head)
        self.assertEqual(a['positions'],b['positions']);self.assertNotEqual(a['logits'],b['logits'])
        self.assertEqual(a['trace'][0]['available_refreshes'],b['trace'][0]['available_refreshes'],[])
    def test_guard_rejects_official_test_outcomes_and_unapproved_arrays(self):
        guard=c.InputGuard()
        for p in [c.OLD/'test_metrics.csv',c.OLD/'test_collection.json',c.OLD/'primary_intervals.json',
                  c.SOURCE/'features/SYNTHETIC_test.avi.npz',c.SOURCE/'videos/UCF-101/SYNTHETIC/test.avi',
                  c.ROOT/'data/source/causal_observability_v1/videos/SYNTHETIC/test.avi']:
            with self.assertRaises(c.Blocker):guard.path_check(p)
        guard.path_check(c.OLD/'head_selection.json')
    def test_historical_write_guard(self):
        guard=c.InputGuard()
        with self.assertRaises(c.Blocker):guard.hook('open',(str(c.OLD/'head_selection.json'),'w',1))
    def test_execution_arrays_unavailable_before_diagnostics(self):
        guard=c.InputGuard();v=dict(clip='SYNTHETIC.avi',study_role='execution',path='data/source/SYNTHETIC.avi')
        guard.set_population(dict(videos=[v]));p=c.SOURCE/'features/SYNTHETIC.avi.npz'
        with self.assertRaises(c.Blocker):guard.path_check(p)
        guard.set_population(dict(videos=[v]),diagnostics=True);guard.path_check(p)
    def test_checkpoint_identity_payload_and_reuse(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_TEST_FIXTURE_') as folder:
            path=Path(folder)/'checkpoint.json';calls=[]
            def compute():calls.append(1);return dict(SYNTHETIC=True,calls=4)
            self.assertEqual(c.checkpoint(path,{'source':'SYNTHETIC'},compute),c.checkpoint(path,{'source':'SYNTHETIC'},compute))
            self.assertEqual(len(calls),1)
            with self.assertRaises(c.Blocker):c.checkpoint(path,{'source':'CHANGED'},compute)
            r=c.read(path);r['payload']['calls']=3;c.save(path,r)
            with self.assertRaises(c.Blocker):c.checkpoint(path,{'source':'SYNTHETIC'},compute)
    def test_freeze_mutation_rejection(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_TEST_FIXTURE_') as folder:
            root=Path(folder);out=root/'out';out.mkdir();f=root/'SYNTHETIC';f.write_bytes(b'SYNTHETIC TEST FIXTURE')
            with patch.object(c,'ROOT',root),patch.object(c,'OUT',out):
                with self.assertRaises(c.Blocker):c.seal_check()
                c.save(out/'development_execution_seal.json',dict(sha256={'SYNTHETIC':c.sha(f)}));c.seal_check()
                f.write_bytes(b'SYNTHETIC CHANGED')
                with self.assertRaises(c.Blocker):c.seal_check()
    def test_weighted_class_group_clip_metrics(self):
        entries=[dict(label='SYNTHETIC_A',group='g1'),dict(label='SYNTHETIC_A',group='g1'),
                 dict(label='SYNTHETIC_A',group='g2'),dict(label='SYNTHETIC_B',group='g1')]
        z=np.array([[2.,0.],[2.,0.],[0.,2.],[0.,2.]])
        m=s.metrics(z,entries,['SYNTHETIC_A','SYNTHETIC_B'])
        self.assertAlmostEqual(m['accuracy'],.75)
    def test_population_rejects_official_test_and_cross_roles(self):
        classes=[f'SYNTHETIC_{i}' for i in range(101)];rows=[]
        for label in classes:
            for role,groups in [('controller_train',['g01','g02']),('selection',['g03']),('execution',['g04'])]:
                for g in groups:
                    for j in range(2):
                        clip=f'{label}_{g}_{j}.avi';rows.append(dict(label=label,group=g,clip=clip,sha256=c.digest(clip),
                            role='train' if role=='controller_train' else 'development',study_role=role,
                            official_role='official_train',official_path=f'{label}/{clip}'))
        manifest=dict(classes=classes,videos=rows);members={v['official_path'] for v in rows}
        with patch.object(c,'original_train_members',return_value=members),patch.object(c,'read',return_value=dict(edges=[])):
            c.validate_population(manifest)
            rows[0]['official_role']='test'
            with self.assertRaises(c.Blocker):c.validate_population(manifest)
            rows[0]['official_role']='official_train';rows[-1]['sha256']=rows[0]['sha256']
            with self.assertRaises(c.Blocker):c.validate_population(manifest)


if __name__=='__main__':unittest.main()
