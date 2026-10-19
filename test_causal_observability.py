"""Explicitly SYNTHETIC fixtures only; no empirical experiment outputs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import causal_observability_common as c
from causal_observability_policy import *
from causal_observability_data import assert_roles,source_family,validate_manifest,isolated_partition
from causal_observability_analysis import cluster_plan,paired_interval


class Fixtures(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(0)
        self.f=rng.normal(size=(16,5));self.thumbs=rng.integers(0,256,size=(16,32,32),dtype=np.uint8)
        self.head=dict(means=np.zeros(5),scales=np.ones(5),weights=rng.normal(size=(3,5)),bias=np.zeros(3))

    def test_prohibited_current_future_label_access(self):
        o=Observation(1,1,3,((1,2),),())
        for key in ['label','future_features','current_feature']:
            with self.assertRaises(RuntimeError):getattr(o,key)

    def test_first_four_cache_pooling(self):
        provider=TrainingReplayProvider(self.f,self.thumbs)
        r=execute(provider,Policy('periodic',self.head),self.head)
        self.assertEqual(r['positions'],[0,4,8,12]);self.assertEqual(provider.forward_completions,4)
        expected=affine(np.repeat(self.f[[0,4,8,12]],4,axis=0).mean(0),self.head)
        np.testing.assert_allclose(r['logits'],expected,atol=1e-12)
        np.testing.assert_array_equal(r['cache_indices'],np.repeat([0,4,8,12],4))

    def test_forced_feasibility_never_or_always(self):
        for choose,expected in [(False,[0,13,14,15]),(True,[0,1,2,3])]:
            class Gate:
                needs_cheap=False
                def decide(self,obs):return choose
            p=TrainingReplayProvider(self.f,self.thumbs);r=execute(p,Gate(),self.head)
            self.assertEqual(r['positions'],expected);self.assertEqual(p.forward_completions,4)

    def test_chronology_and_only_encountered_feature(self):
        p=TrainingReplayProvider(self.f,self.thumbs)
        with self.assertRaises(ValueError):p.advance(1)
        p.advance(0);np.testing.assert_array_equal(p.refresh(),self.f[0])
        with self.assertRaises(ValueError):p.refresh()
        seen=[]
        execute(TrainingReplayProvider(self.f,self.thumbs),Policy('motion',self.head,threshold=.1),self.head,
            on_decision=lambda obs,a,b:seen.append(obs))
        self.assertEqual([o.t for o in seen],list(range(16)))
        for obs in seen:
            self.assertLessEqual(len(obs.history),4)
            for v in obs.history:self.assertTrue(any(np.array_equal(v,row) for row in self.f[:obs.t]))

    def test_independent_video_state(self):
        a=execute(TrainingReplayProvider(self.f,self.thumbs),Policy('RandomB4_seeded',self.head,clip='SYNTHETIC_A'),self.head)
        b=execute(TrainingReplayProvider(self.f*3,self.thumbs),Policy('RandomB4_seeded',self.head,clip='SYNTHETIC_A'),self.head)
        self.assertEqual(a['positions'],b['positions']);self.assertNotEqual(a['logits'],b['logits'])
        self.assertEqual(a['trace'][0]['available_refreshes'],[]);self.assertEqual(b['trace'][0]['available_refreshes'],[])

    def test_history_only_cheap_isolation(self):
        obs=Observation(2,2,3,tuple(tuple(v) for v in self.f[:1]),tuple(np.zeros(32)))
        changed=Observation(2,2,3,obs.history,tuple(np.ones(32)))
        np.testing.assert_array_equal(vector(obs,self.head,False),vector(changed,self.head,False))
        self.assertFalse(np.array_equal(vector(obs,self.head,True),vector(changed,self.head,True)))
        class Poison:
            def __array__(self,*a,**k):raise AssertionError('Forbidden cheap access')
        isolated=Observation(obs.t,obs.age,obs.remaining,obs.history,Poison())
        vector(isolated,self.head,False)

    def test_no_thumbnail_calls_for_history_or_periodic(self):
        class NoThumb(TrainingReplayProvider):
            def thumbnail(self):raise AssertionError('Unexpected current observation')
        execute(NoThumb(self.f,self.thumbs),Policy('periodic',self.head),self.head)

    def test_family_and_duplicate_safety(self):
        self.assertEqual(source_family('Movie_X_walk_u_nm_np1_fr_med_0.avi','walk')[0],'moviex')
        rows=[dict(source_family='SYNTHETIC_F',sha256='SYNTHETIC_H',role='head_train'),dict(source_family='SYNTHETIC_F',sha256='OTHER',role='test')]
        with self.assertRaises(ValueError):assert_roles(rows)
        rows[1].update(source_family='OTHER',sha256='SYNTHETIC_H')
        with self.assertRaises(ValueError):assert_roles(rows)

    def test_cached_population_cannot_bypass_fixed_design(self):
        with self.assertRaises(c.Blocker):validate_manifest(dict(classes=['SYNTHETIC'],videos=[]))
        with self.assertRaises(c.Blocker):validate_manifest(dict(classes=[f'SYNTHETIC_{i}' for i in range(51)],videos=[]))

    def test_positive_family_exact_quota_assignment(self):
        classes=[f'SYNTHETIC_CLASS_{i}' for i in range(51)]
        rows=[]
        for label in classes:
            for family,n in [('SHARED_SYNTHETIC_HEAD',45),(label+'_CONTROLLER',15),(label+'_DEV',10)]:
                rows.extend(dict(label=label,source_family=family) for _ in range(n))
        assignment=isolated_partition(rows,classes)
        self.assertEqual(assignment['SHARED_SYNTHETIC_HEAD'],'head_train')
        for label in classes:
            for role,n in [('head_train',45),('controller_train',15),('development',10)]:
                self.assertEqual(sum(r['label']==label and assignment[r['source_family']]==role for r in rows),n)

    def test_cross_class_grouped_bootstrap(self):
        rows=[dict(label='A',source_family='shared'),dict(label='B',source_family='shared'),dict(label='A',source_family='a'),dict(label='A',source_family='aa'),dict(label='B',source_family='b'),dict(label='B',source_family='bb')]
        values=np.array([2.,10.,4.,6.,12.,14.]);f,idx,strata=cluster_plan(rows,repeats=10000)
        r=paired_interval(rows,values,['A','B'],f,idx)
        self.assertAlmostEqual(r['mean'],8.)
        manual=[]
        for draw in idx:
            multiplicity=np.bincount(draw,minlength=len(f));means=[]
            for label in ['A','B']:
                weights=np.array([multiplicity[f.index(v['source_family'])] if v['label']==label else 0 for v in rows]);means.append(weights @ values/weights.sum())
            manual.append(np.mean(means))
        np.testing.assert_allclose([r['CI95_lower'],r['CI95_upper']],np.percentile(manual,[2.5,97.5]),atol=1e-12)
        self.assertTrue(any(s['classes']==['A','B'] and s['singleton'] for s in strata))

    def test_seal_missing_mutation_and_resume_integrity(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_seal_') as d:
            root=Path(d);out=root/'out'
            with patch.object(c,'ROOT',root),patch.object(c,'OUT',out):
                with self.assertRaises(c.Blocker):c.seal_check()
                p=root/'fixture';p.write_bytes(b'SYNTHETIC')
                c.save(out/'evaluation_seal.json',dict(sha256={'fixture':c.sha(p)}))
                with self.assertRaises(c.Blocker):c.seal_check(True)
                c.save(out/'evaluation_exposure.json',dict(seal_sha256=c.sha(out/'evaluation_seal.json')))
                c.seal_check(True);before=c.sha(out/'evaluation_seal.json');c.seal_check(True);self.assertEqual(before,c.sha(out/'evaluation_seal.json'))
                p.write_bytes(b'SYNTHETIC MUTATION')
                with self.assertRaises(c.Blocker):c.seal_check(True)

    def test_teacher_labels_never_enter_decide(self):
        from causal_observability_learning import teacher
        result=execute(TrainingReplayProvider(self.f,self.thumbs),Policy('periodic',self.head),self.head)
        a=teacher(self.f,self.head,result['trace'],5)
        altered=[dict(v,selected=not v['selected']) if v['t']>5 else v for v in result['trace']]
        self.assertEqual(a,teacher(self.f,self.head,altered,5))

    def test_completed_checkpoint_reuse_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_checkpoint_') as d:
            root=Path(d)
            with patch.object(c,'ROOT',root):
                p=root/'record.json';calls=[]
                def compute():calls.append(1);return dict(SYNTHETIC=True,forwards=4)
                first=c.checkpoint(p,{'input':'SYNTHETIC'},compute)
                self.assertEqual(first,c.checkpoint(p,{'input':'SYNTHETIC'},compute));self.assertEqual(len(calls),1)
                with self.assertRaises(c.Blocker):c.checkpoint(p,{'input':'changed'},compute)
                r=c.read(p);r['payload']['forwards']=5;c.save(p,r)
                with self.assertRaises(c.Blocker):c.checkpoint(p,{'input':'SYNTHETIC'},compute)

    def test_weighted_51class_head_and_equal_learner_shapes(self):
        from causal_observability_learning import fit_head,train_network
        rng=np.random.default_rng(4);classes=[f'SYNTHETIC_CLASS_{i}' for i in range(51)]
        entries=[dict(label=label,source=f'SYNTHETIC_{i}_{j}') for i,label in enumerate(classes) for j in range(2+i%2)]
        x=rng.normal(size=(len(entries),5));y=np.array([classes.index(r['label']) for r in entries])
        head,log=fit_head(x,y,entries,classes,.1)
        expected=np.mean([x[y==i].mean(0) for i in range(51)],axis=0)
        np.testing.assert_allclose(head['means'],expected,atol=1e-12);self.assertEqual(np.shape(head['weights']),(51,5))
        rows=[dict(label=label,source=f'SYNTHETIC_{i}',input=rng.normal(size=95).tolist(),target=(i+j)%2) for i,label in enumerate(classes) for j in range(2)]
        h,_=train_network(rows,False,(1,2));m,_=train_network(rows,True,(1,2))
        self.assertEqual(np.shape(h[2]['w1']),(32,95));np.testing.assert_array_equal(h[2]['means'][-32:],np.zeros(32))
        np.testing.assert_allclose(h[2]['means'][:-32],m[2]['means'][:-32],atol=0,rtol=0)

    def test_dense_diagnostics_require_prior_actual_rollouts(self):
        import causal_observability_runtime as runtime
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_guard_') as d:
            root=Path(d);out=root/'out'
            with patch.object(c,'ROOT',root),patch.object(c,'OUT',out),patch.object(runtime,'ROOT',root),patch.object(runtime,'OUT',out),patch.object(runtime,'DATA',root/'data'):
                entry=dict(label='SYNTHETIC',clip='fixture',role='test')
                with self.assertRaises(c.Blocker):runtime.collect_dense([entry],'test',object())
                c.save(out/'evaluation_seal.json',dict(sha256={}))
                c.save(out/'evaluation_exposure.json',dict(seal_sha256=c.sha(out/'evaluation_seal.json')))
                c.save(out/'test_rollouts.json',dict(all_deployable_rollouts_complete=False))
                with self.assertRaises(c.Blocker):runtime.collect_dense([entry],'test',object())

    def test_end_to_end_synthetic_paired_tables(self):
        import pandas as pd
        import causal_observability_analysis as analysis
        classes=[f'SYNTHETIC_CLASS_{i}' for i in range(3)]
        entries=[dict(label=label,clip=f'SYNTHETIC_CLIP_{j}',source_family=label+str(j),role='test') for label in classes for j in range(2)]
        rows=[]
        for v in entries:
            for method,ce in [('periodic',.7),('history_motion',.5),('history',.6),('Z_pool_offline',.4),('dense16',.3),('task_C_hindsight',.2),('motion',.8),('RandomB4_seeded',.9)]:
                rows.append(dict(key=v['label']+'/'+v['clip'],method=method,CE=ce,correct=1,dense_agreement=1,KL=.01,**v))
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_analysis_') as d:
            out=Path(d)
            with patch.object(analysis,'OUT',out):
                pd.DataFrame(rows).to_csv(out/'per_source_metrics.csv',index=False)
                result=analysis.summarize(dict(classes=classes,videos=entries))
                self.assertAlmostEqual(result['mean'],.2,places=12)
                self.assertAlmostEqual(result['CI95_lower'],.2,places=12)
                self.assertEqual(np.load(out/'bootstrap_cluster_indices.npy').shape,(10000,6))


if __name__=='__main__':unittest.main()
