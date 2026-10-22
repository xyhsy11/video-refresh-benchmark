"""Synthetic SOFTWARE fixtures only; never empirical video evidence."""
import unittest
import numpy as np
from executable_value_observability import hog, observation, trajectory, ridge_fit, score, actions, PERIODIC


class SoftwareFixtures(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(9)
        self.head=dict(means=np.zeros(2048),scales=np.ones(2048),weights=rng.normal(size=(51,2048))*.01,bias=np.zeros(51))
        self.f=rng.normal(size=(16,2048))

    def test_hog_constant_and_orientation(self):
        self.assertTrue(np.array_equal(hog(np.full((32,32),128,np.uint8)),np.zeros((16,9))))
        a=np.tile(np.arange(32,dtype=np.uint8)*8,(32,1)); h=hog(a)
        np.testing.assert_allclose(h[:,0],1.,atol=1e-10)
        np.testing.assert_allclose(h[:,1:],0.,atol=1e-10)
        np.testing.assert_allclose(hog(255-a),h,atol=1e-10)

    def test_history_api_rejects_current_and_future(self):
        for t in range(1,13):
            past={u:self.f[u] for u in PERIODIC if u<t}
            self.assertEqual(observation(t,past,self.head).shape,(83,))
            with self.assertRaises(ValueError): observation(t,{**past,t:self.f[t]},self.head)
            with self.assertRaises(ValueError): observation(t,{**past,15:self.f[15]},self.head)

    def test_references_and_future_pixels(self):
        rng=np.random.default_rng(11); thumbs=rng.integers(0,256,(16,32,32),dtype=np.uint8)
        supplied={u:self.f[u] for u in PERIODIC}; x=trajectory(thumbs,supplied,self.head)
        for t in range(1,13):
            r=max(u for u in PERIODIC if u<t)
            np.testing.assert_allclose(x[t-1,115:131],np.linalg.norm(hog(thumbs[t])-hog(thumbs[r]),axis=1))
            np.testing.assert_allclose(x[t-1,131:],np.linalg.norm(hog(thumbs[t])-hog(thumbs[t-1]),axis=1))
            changed=thumbs.copy(); changed[t+1:]=0
            np.testing.assert_array_equal(trajectory(changed,supplied,self.head)[:t],x[:t])

    def test_ridge_objective_weight_scale_and_constant(self):
        rng=np.random.default_rng(3); x=rng.normal(size=(70,4)); x[:,3]=7
        y=x[:,0]*.4+2; w=rng.uniform(.1,2,70)
        m=ridge_fit(x,y,w); n=ridge_fit(x,y,w*17)
        np.testing.assert_allclose(score(m,x),score(n,x),atol=1e-12)
        self.assertEqual(m['scales'][3],1.)
        self.assertLess(m['normal_residual'],1e-12)
        # Independent gradient of normalized half-MSE + half L2.
        z=(x-m['means'])/m['scales']; residual=score(m,x)-y; ww=w/w.sum()
        np.testing.assert_allclose(z.T@(ww*residual)+m['coefficients'],0,atol=1e-12)
        self.assertAlmostEqual(float(ww@residual),0.,places=12)

    def test_sign_and_tie(self):
        np.testing.assert_array_equal(actions(np.array([2e-12,-2e-12,0.,1e-12,-1e-12]),np.array([1,4,4,2,8])),[1,0,1,0,1])


if __name__=='__main__': unittest.main()
