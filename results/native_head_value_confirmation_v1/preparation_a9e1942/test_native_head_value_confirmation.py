"""Exact contracts and genuine stored fitting inputs; no synthetic observations.

These checks never open protected test outcomes, invoke an encoder, or generate
an empirical sample. Combinations enumerate the fixed mathematical action space.
"""
from __future__ import annotations

import csv
from itertools import combinations
import json
from pathlib import Path
import unittest
import tempfile

import numpy as np

from native_head_value_confirmation import (SCHEDULES, PERIODIC, VIEWS, class_mapping,
    pool_weights, repair, sha, native_observation, source_inference, folds)
from verify_native_head_value_confirmation import independent_repair

ROOT = Path(__file__).resolve().parent


class ExactContracts(unittest.TestCase):
    def test_complete_legal_space_and_independent_tail(self):
        self.assertEqual(len(SCHEDULES),455)
        self.assertEqual(len(set(SCHEDULES)),455)
        for s in SCHEDULES:
            self.assertEqual(s[0],0)
            self.assertEqual(len(s),4)
            self.assertEqual(tuple(sorted(set(s))),s)
        for t in range(1,13):
            for a in (0,1):
                main = repair(t,a)
                self.assertEqual(main,independent_repair(t,a))
                self.assertIn(main,SCHEDULES)
                self.assertEqual(tuple(u for u in main if u<t),tuple(u for u in PERIODIC if u<t))
                self.assertEqual(t in main,bool(a))
            self.assertEqual(repair(t,int(t in PERIODIC)),PERIODIC)

    def test_duration_weights_by_explicit_cache_visits(self):
        w = pool_weights()
        for i,schedule in enumerate(SCHEDULES):
            visits = [max(u for u in schedule if u<=t) for t in range(16)]
            for u in range(16):
                self.assertEqual(w[i,u],visits.count(u)/16)
        np.testing.assert_array_equal(w.sum(1),np.ones(455))

    def test_official_native_map_and_real_annotation_labels(self):
        p = ROOT/'data/source/native_head_value_confirmation_v1/metadata'
        with (p/'val_annotations').open(encoding='utf-8-sig') as f:
            labels = {r['label'] for r in csv.DictReader(f)}
        native = json.loads((p/'native_classes').read_text())
        result = class_mapping(native,labels)
        for name,index in native.items():
            cleaned = name[1:-1] if name.startswith('"') and name.endswith('"') else name
            self.assertEqual(result[cleaned],int(index))
        self.assertEqual(result['dying hair'],108)
        self.assertEqual(set(result.values()),set(range(400)))

    def test_real_stored_validation_features_and_native_history_schema(self):
        # This is saved-feature replay on a previously exposed fitting source,
        # not a new E27 smoke, untouched confirmation, or model latency.
        collection = json.loads((ROOT/'results/temporal_refresh_generalization_v1/train_development_collection.json').read_text())
        record = next(iter(collection['videos'].values()))
        path = ROOT/record['path']
        self.assertEqual(sha(path),record['sha256'])
        with np.load(path,allow_pickle=False) as arrays:
            features = arrays['x3d_xs'].astype(float)
        import torch
        state = torch.load(ROOT/'data/source/temporal_refresh_generalization_v1/X3D_XS.pyth',map_location='cpu',weights_only=False)['model_state']
        weight = state['blocks.5.proj.weight'].numpy().astype(float)
        bias = state['blocks.5.proj.bias'].numpy().astype(float)
        # Genuine saved f through actual fixed head; explicitly replay, not
        # genuine new inference or a substitute for the future native smoke.
        z = features @ weight.T+bias
        for t in range(1,13):
            history = {u:(features[u],z[u]) for u in PERIODIC if u<t}
            h = native_observation(t,history)
            self.assertEqual(h.shape,(VIEWS['H'],))
            self.assertAlmostEqual(float(h[22:422].sum()),1.)
        self.assertEqual(VIEWS,dict(S=19,H=432,HG=464,HGH=496))
        with self.assertRaises(ValueError):
            native_observation(4,{u:(features[u],z[u]) for u in (0,4)})
        with self.assertRaises(ValueError):
            native_observation(4,{0:(features[0],z[0,:51])})

    def test_real_fitting_feature_ridge_objective(self):
        from executable_value_observability import ridge_fit
        collection = json.loads((ROOT/'results/temporal_refresh_generalization_v1/train_development_collection.json').read_text())
        record = next(iter(collection['videos'].values()))
        with np.load(ROOT/record['path'],allow_pickle=False) as stored:
            features = stored['x3d_xs'].astype(float)
        x = features[:,:19]
        y = 1-(features @ features[0])/(np.linalg.norm(features,axis=1)*np.linalg.norm(features[0]))
        # Actual equally weighted states from this saved source; this one
        # mechanical verification fit is not an E27 planning/final probe.
        weights = np.repeat(1/len(y),len(y))
        model = ridge_fit(x,y,weights)
        from run_native_head_value_confirmation import event
        event('real_feature_objective_contract_fit',verification_probe_fits=1,
              scope='one archived fitting source, mechanical check only, not E27 planning')
        centered = (x-np.array(model['means']))/np.array(model['scales'])
        beta = np.array(model['coefficients']); intercept = model['intercept']
        residual = centered @ beta+intercept-y
        gradient = np.r_[centered.T @ residual/len(y)+beta,residual.mean()]
        self.assertLess(float(abs(gradient).max()),1e-10)
        expected_mu = np.array([sum(x[:,j])/len(x) for j in range(x.shape[1])])
        np.testing.assert_allclose(model['means'],expected_mu,rtol=0,atol=1e-12)

    def test_bootstrap_units_and_analytic_variance_on_real_official_metadata(self):
        # Resampling genuine annotation timestamps validates arithmetic only.
        # These are not model outcomes or an E27 scientific population/result.
        rows = json.loads((ROOT/'data/source/native_head_value_confirmation_v1/candidate_manifest.json').read_text())
        selected = []
        seen = set()
        for c in range(400):
            members = []
            for row in rows:
                if row['split']=='val' and row['y']==c and row['component'] not in seen:
                    members.append(row);seen.add(row['component'])
                    if len(members)==8:
                        break
            self.assertEqual(len(members),8)
            selected.extend(members)
        values = np.array([int(r['filename'][-17:-11]) for r in selected],dtype=float)
        groups = [np.arange(c*8,(c+1)*8) for c in range(400)]
        expected_mean = sum(float(values[g].mean()) for g in groups)/400
        with tempfile.TemporaryDirectory(prefix='native27-real-metadata-') as temp:
            path = Path(temp)/'draws.npz'
            result = source_inference(selected,values,path)
            with np.load(path,allow_pickle=False) as data:
                draws = data['indices']
            samples = np.mean(values[draws],axis=1)
            np.testing.assert_allclose(result['bootstrap95'][0],np.quantile(samples,[.025,.975],method='linear'),atol=1e-10,rtol=0)
        self.assertAlmostEqual(result['mean'][0],expected_mean,places=10)
        vc = [float(np.var(values[g],ddof=1))/(400**2*8) for g in groups]
        variance = sum(vc);denom = sum(v*v/7 for v in vc)
        df = variance**2/denom if denom else float('inf')
        from scipy.stats import t
        half = float(t.ppf(.975,df))*np.sqrt(variance)
        np.testing.assert_allclose(result['analytic95'][0],[expected_mean-half,expected_mean+half],atol=1e-10,rtol=0)
        fs = folds(selected)
        for g in groups:
            self.assertEqual(sorted(np.bincount(fs[g]).tolist()),[2,2,2,2])


if __name__=='__main__':
    unittest.main()
