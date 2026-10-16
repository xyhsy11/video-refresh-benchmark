"""Small independent exhaustive fixtures,not experimental evidence."""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from itertools import combinations
import unittest
import numpy as np
import pandas as pd
import refresh_pooling_utils as u
from verify_refresh_pooling_control import explicit_pool,ce


class PoolingChecks(unittest.TestCase):
    @staticmethod
    def head(n):return {'means':[0.]*n,'scales':[1.]*n,'weights':np.eye(n).tolist(),'bias':[0.]*n}

    def test_counts_order_and_weights(self):
        for b,count in [(1,1),(2,15),(4,455),(8,6435),(16,1)]:
            table=u.schedules(16,b)
            self.assertEqual(len(table['positions']),count)
            self.assertEqual(table['positions'].tolist(),[list((0,*x)) for x in combinations(range(1,16),b-1)])
            self.assertTrue((table['durations']>0).all())
            for key in ['cache_weights','unique_weights']:np.testing.assert_allclose(table[key].sum(1),1.,atol=u.TOL,rtol=0)

    def test_pooling_alone_changes_prediction(self):
        features=np.array([[-3.,0.],[2.,0.],[8.,1.],[9.,2.]])
        a,b=explicit_pool(features,[0,1],self.head(2))
        np.testing.assert_array_equal(a,[.75,0.]);np.testing.assert_array_equal(b,[-.5,0.])
        self.assertEqual((a.argmax(),b.argmax()),(0,1))

    def test_both_oracles_and_exact_random_scalar_enumeration(self):
        rng=np.random.default_rng(19);features=rng.normal(size=(6,3));head=self.head(3)
        for budget in [2,3,4]:
            result=u.evaluate(features,head,1,u.schedules(6,budget));cache_losses=[];unique_losses=[];correct_c=[];correct_u=[]
            for rest in combinations(range(1,6),budget-1):
                schedule=[0,*rest];a,b=explicit_pool(features,schedule,head)
                cache_losses.append(float(ce(a,1)));unique_losses.append(float(ce(b,1)))
                correct_c.append(a.argmax()==1);correct_u.append(b.argmax()==1)
            np.testing.assert_allclose(result['loss_C'],cache_losses,atol=1e-12,rtol=0)
            np.testing.assert_allclose(result['loss_U'],unique_losses,atol=1e-12,rtol=0)
            for key,values in [('SC',cache_losses),('SU',unique_losses)]:
                minimum=min(values);ties=[i for i,v in enumerate(values) if v<=minimum+1e-12]
                self.assertEqual(result[key]['index'],ties[0]);self.assertEqual(result[key]['tie_count'],len(ties))
            self.assertAlmostEqual(result['loss_C'].mean(),sum(cache_losses)/len(cache_losses))
            self.assertAlmostEqual(result['correct_C'].mean(),sum(correct_c)/len(correct_c))
            self.assertAlmostEqual(result['correct_U'].mean(),sum(correct_u)/len(correct_u))
            mean_logits=result['logits_C'].mean(0)
            self.assertGreater(abs(result['loss_C'].mean()-float(ce(mean_logits,1))),1e-6)

    def test_tolerance_first_choice_and_true_minimum(self):
        values=np.array([1.+.5e-12,1.+2e-12,1.])
        result=u.first_tie(values)
        self.assertEqual(result['index'],0);self.assertEqual(result['tie_count'],2)
        self.assertEqual(result['true_minimum'],1.);self.assertGreater(result['achieved_objective'],result['true_minimum'])

    def test_periodic_dense_one_and_identical_features(self):
        rng=np.random.default_rng(23);features=rng.normal(size=(16,3));head=self.head(3)
        for budget in [1,2,4,8,16]:
            schedule=list(range(0,16,16//budget));a,b=explicit_pool(features,schedule,head)
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
            identical=np.tile(features[0],(16,1));a,b=explicit_pool(identical,schedule,head)
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)

    def test_affine_offset_and_nonunit_scaling(self):
        features=np.arange(48,dtype=float).reshape(16,3)/10
        head={'means':[2.,-3.,.5],'scales':[.5,3.,2.],'weights':[[2.,1.,-1.],[1.,2.,4.]],'bias':[-4.,3.]}
        table=u.schedules(16,4);z=u.logits(features,head)
        for i in [0,100,454]:
            a,b=explicit_pool(features,table['positions'][i].tolist(),head)
            np.testing.assert_allclose([a,b],[table['cache_weights'][i]@z,table['unique_weights'][i]@z],atol=1e-12,rtol=0)

    def test_class_stratified_bootstrap_and_csv_nulls(self):
        sources=['a','b','c','d','e','f'];labels=dict(zip(sources,['x']*3+['y']*3))
        idx=u.stratified_indices(sources,labels,['x','y'],200,20261023)
        self.assertTrue((idx[:,:3]<3).all());self.assertTrue((idx[:,3:]>=3).all())
        np.testing.assert_array_equal(idx,u.stratified_indices(sources,labels,['x','y'],200,20261023))
        values=np.arange(6,dtype=float);summary=u.describe(values,idx)
        np.testing.assert_array_equal(np.quantile(values[idx].mean(1),[.025,.975]),[summary['CI95_lower'],summary['CI95_upper']])
        a=pd.DataFrame({'n':[1.,None],'reason':[None,None]});b=pd.DataFrame({'n':[1.,np.nan],'reason':[np.nan,np.nan]})
        u.compare_csv(a,b)


if __name__=='__main__':unittest.main()
