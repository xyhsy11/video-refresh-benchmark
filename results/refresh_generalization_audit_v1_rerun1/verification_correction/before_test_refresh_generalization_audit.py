"""Independent fixtures for audit arithmetic,not experimental observations."""
from __future__ import annotations

import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import refresh_generalization_audit_utils as u


class AuditChecks(unittest.TestCase):
    def test_tree_traversal_and_float32_boundary(self):
        tree=SimpleNamespace(children_left=np.array([1,-1,-1]),children_right=np.array([2,-1,-1]),
            feature=np.array([0,-2,-2]),threshold=np.array([.5,-2,-2]),value=np.array([0.,-2.,3.]).reshape(3,1,1))
        forest=SimpleNamespace(estimators_=[SimpleNamespace(tree_=tree)]*2)
        values=u.walk_forest(forest,np.array([[.1],[.5],[.500000001],[.6]]))
        np.testing.assert_array_equal(values,[-2,-2,-2,3])

    def test_schedule_boundaries_and_target_signs(self):
        logits=np.arange(320,dtype=float).reshape(16,20)/100
        logits[:,3]=np.sin(np.arange(16))
        for r in [4,8,12]:
            for t in [r-3,r-2,r-1]:
                a=[0,4,8,12];b=sorted(a+[t]);c=sorted([x for x in a if x!=r]+[t])
                self.assertEqual([len(a),len(b),len(c)],[4,5,4])
                np.testing.assert_array_equal(u.cache(a)[:t],u.cache(c)[:t])
                losses=[u.cross_entropy(logits[u.cache(s)].mean(0),3) for s in [a,b,c]]
                self.assertAlmostEqual(losses[0]-losses[2],(losses[0]-losses[1])-(losses[2]-losses[1]))
                expected=[];latest=0
                for j in range(16):
                    if j in c:latest=j
                    expected.append(latest)
                np.testing.assert_array_equal(u.cache(c),expected)

    def test_motion_training_only_missing_and_ties(self):
        tr=pd.DataFrame({'current_mv_available':[1,1,0,1],'current_p90_magnitude_px':[2.,4.,10000.,6.],'G':[99.,0.,-99.,4.]})
        threshold=u.motion_threshold(tr);self.assertEqual(threshold,4.)
        changed=tr.copy();changed['G']=-1e10
        self.assertEqual(u.motion_threshold(changed),threshold)
        held=pd.DataFrame({'current_mv_available':[1,1,0,1],'current_p90_magnitude_px':[4.,5.,9000.,1.]})
        np.testing.assert_array_equal(u.accept_motion(held,threshold),[False,True,False,False])
        tr['current_mv_available']=0
        self.assertIsNone(u.motion_threshold(tr))
        self.assertFalse(u.accept_motion(held,None).any())

    def fixture(self):
        rows=[]
        for source,ys in [('a',np.array([2.,-1.,0.]*3)),('b',np.array([4.,-3.,0.]*3))]:
            for method in u.METHODS:
                for i,y in enumerate(ys):
                    p=.5 if method=='random_move_expected' else 0. if method=='keep_periodic' else 1.
                    rows.append({'cohort':'fixture','method':method,'source_video_id':source,'class':'fixture',
                        'example_id':source+str(i),'G':y,'accept_probability':p,'utility_contribution':y*p,
                        'predicted_G':.1 if method in u.LEARNED else np.nan,'training_median_G':0.,
                        'current_p90_magnitude_px':float(i),'current_mv_available':1})
        return pd.DataFrame(rows)

    def test_expected_random_and_metric_denominators(self):
        raw=self.fixture();source=u.per_source(raw)
        random=source[(source.method=='random_move_expected')&(source.source_video_id=='a')].iloc[0]
        self.assertEqual(random.selected,4.5);self.assertEqual(random.helpful_selected,1.5)
        self.assertEqual(random.harmful_selected,1.5);self.assertEqual(random.neutral_selected,1.5)
        self.assertAlmostEqual(random.utility,1/6)
        self.assertEqual(random.harmful_selection_rate,1/3);self.assertEqual(random.missed_benefit_rate,.5)
        keep=source[source.method=='keep_periodic'].iloc[0]
        self.assertTrue(pd.isna(keep.mean_selected_gain));self.assertEqual(keep.missed_benefit_rate,1.)
        self.assertTrue(pd.isna(keep.MAE))

    def test_whole_source_shared_bootstrap(self):
        raw=self.fixture();source=u.per_source(raw)
        summary,pairs,matrices=u.summarize(raw,source,20261022,200)
        indices=matrices['fixture'];self.assertEqual(indices.shape,(200,2))
        random=next(x for x in pairs if x['candidate']=='random_move_expected' and x['baseline']=='keep_periodic')
        values=source[source.method=='random_move_expected'].sort_values('source_video_id').utility.to_numpy()
        np.testing.assert_array_equal(np.percentile(values[indices].mean(1),[2.5,97.5]),[random['lower'],random['upper']])
        again=u.summarize(raw,source,20261022,200)[2]['fixture']
        np.testing.assert_array_equal(indices,again)

    def test_no_outcomes_in_feature_whitelists(self):
        self.assertEqual(len(u.C_COLUMNS),56);self.assertEqual(len(u.T_COLUMNS),44)
        self.assertFalse(set(u.KEYS+['G','B','C','class','loss_A','loss_C','label'])&set(u.C_COLUMNS+u.T_COLUMNS))
        self.assertFalse(any('privileged' in x for x in u.C_COLUMNS+u.T_COLUMNS))

    def test_cross_entropy_and_undefined_statistics(self):
        self.assertAlmostEqual(u.cross_entropy(np.zeros(20),2),np.log(20))
        self.assertAlmostEqual(u.cross_entropy(np.array([10000.,10001.]),1),np.log1p(np.exp(-1)))
        self.assertIsNone(u.auc(np.zeros(3),np.array([True,False,True]))['AUROC'])
        self.assertIsNone(u.auc(np.arange(3),np.ones(3,dtype=bool))['AUROC'])
        self.assertIsNone(u.ratio(0,0)[0])


if __name__=='__main__':unittest.main()
