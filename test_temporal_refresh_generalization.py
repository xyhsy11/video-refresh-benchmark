"""Independent fixtures for the temporal contract and exact diagnostics."""
from __future__ import annotations
from itertools import combinations
import math
import unittest

import numpy as np
import refresh_pooling_utils as p
from temporal_refresh_common import snippet_grid,operational_grid,TIME_TOL


class TemporalContractTests(unittest.TestCase):
    def test_floor_distinct_and_prefix_isolation(self):
        pts=np.arange(301,dtype=float)/30
        grid=snippet_grid(pts)
        idx=np.array(grid['frame_indices']); requested=np.array(grid['requested_timestamps'])
        self.assertEqual(idx.shape,(16,4)); self.assertTrue((pts[idx]<=requested+TIME_TOL).all())
        self.assertTrue(all(len(set(r))==4 for r in idx)); self.assertTrue((np.diff(idx[:,-1])>0).all())
        self.assertEqual(idx[0].tolist(),[0,15,31,47])
        # A floor request just below a frame cannot round into that future frame.
        x=np.searchsorted(pts,np.array([.049])+TIME_TOL,side='right')-1
        self.assertEqual(x[0],1)
        rng=np.random.default_rng(37); pixels=rng.normal(size=(301,5))
        for row in idx:
            original=pixels[row].copy(); changed=pixels.copy()
            changed[row[-1]+1:]+=100
            nonselected=np.setdiff1d(np.arange(301),row); changed[nonselected]+=10
            np.testing.assert_array_equal(original,changed[row])
    def test_incomplete_rejected(self):
        with self.assertRaises(ValueError): snippet_grid(np.arange(16)/30)
        with self.assertRaises(ValueError): snippet_grid([0]*100)
    def test_operational_grid_matches_validated_vfr(self):
        for seed in range(20):
            rng=np.random.default_rng(seed);pts=np.r_[0.,np.cumsum(rng.uniform(.02,.06,299))]
            self.assertEqual(operational_grid(pts),snippet_grid(pts)['frame_indices'])
    def test_counts(self):
        for b,count in [(2,15),(4,455),(8,6435)]:
            table=p.schedules(16,b); self.assertEqual(len(table['positions']),count)
            self.assertTrue((np.diff(table['positions'],axis=1)>0).all())
    def test_evaluation_is_locked_without_seal(self):
        from pathlib import Path
        import tempfile
        from temporal_refresh_common import require_evaluation_seal
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError,'Evaluation locked'):require_evaluation_seal(Path(folder))
    def test_independent_oracles_and_random(self):
        for seed in range(4):
            rng=np.random.default_rng(seed); f=rng.normal(size=(6,5))
            head={'means':rng.normal(size=5),'scales':rng.uniform(.5,2,5),
                  'weights':rng.normal(size=(3,5)),'bias':rng.normal(size=3)}
            label=seed%3; table=p.schedules(6,3); result=p.evaluate(f,head,label,table)
            other_label=p.evaluate(f,head,(label+1)%3,table)
            self.assertEqual(result['F'],other_label['F'])
            lc=[]; lu=[]; errors=[]
            for tail in combinations(range(1,6),2):
                s=(0,*tail); outputs=[]; last=None; err=0
                for t in range(6):
                    if t in s: last=t
                    outputs.append(f[last]); err+=1-np.clip(np.dot(f[t],f[last])/(np.linalg.norm(f[t])*np.linalg.norm(f[last])),-1,1)
                errors.append(err/6)
                for seq,dest in [(outputs,lc),([f[t] for t in s],lu)]:
                    mean=np.mean(seq,axis=0); logits=((mean-head['means'])/head['scales'])@head['weights'].T+head['bias']
                    maximum=max(logits); dest.append(maximum+math.log(sum(math.exp(x-maximum) for x in logits))-logits[label])
            for name,values in [('loss_C',lc),('loss_U',lu),('feature_error',errors)]:
                np.testing.assert_allclose(result[name],values,atol=1e-12,rtol=0)
            for tag,values in [('SC',lc),('SU',lu),('F',errors)]:
                winner=next(i for i,x in enumerate(values) if x<=min(values)+1e-12)
                self.assertEqual(result[tag]['index'],winner)
            self.assertAlmostEqual(result['loss_C'].mean(),sum(lc)/len(lc),places=12)
    def test_pooling_equalities_and_prediction_change(self):
        for b in [1,2,4,8,16]:
            table=p.schedules(16,b); positions=list(range(0,16,16//b))
            j=np.flatnonzero((table['positions']==positions).all(1))[0]
            np.testing.assert_allclose(table['cache_weights'][j],table['unique_weights'][j],atol=1e-12)
        table=p.schedules(4,2); z=np.array([[-3.,0],[2,0],[17,0],[20,0]])
        j=0; np.testing.assert_allclose(table['cache_weights'][j]@z,[.75,0])
        np.testing.assert_allclose(table['unique_weights'][j]@z,[-.5,0])
        self.assertNotEqual((table['cache_weights'][j]@z).argmax(),(table['unique_weights'][j]@z).argmax())
        self.assertEqual(p.first_tie(np.array([1+5e-13,1.]))['index'],0)
        for weights in [table['cache_weights'],table['unique_weights']]:
            np.testing.assert_allclose(weights@np.ones((4,3)),1)


if __name__=='__main__': unittest.main()
