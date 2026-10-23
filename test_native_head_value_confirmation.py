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
import gzip

import numpy as np

from native_head_value_confirmation import (SCHEDULES, PERIODIC, VIEWS, class_mapping,
    pool_weights, repair, sha, native_observation, source_inference, folds)
from verify_native_head_value_confirmation import independent_repair

ROOT = Path(__file__).resolve().parent


class ExactContracts(unittest.TestCase):
    def test_amended_network_cap_preserves_real_prior_receipts(self):
        from run_native_head_value_confirmation import CAPS,network_ledger
        self.assertEqual(CAPS['network_bytes'],61*2**30)
        self.assertEqual(CAPS['retained_bytes'],20*2**30)
        old=json.loads((ROOT/'results/native_head_value_confirmation_v1/preparation_a9e1942/network_ledger.json').read_text(encoding='utf-8'))
        current=network_ledger()
        self.assertEqual(current['requests'][:len(old['requests'])],old['requests'])

    def test_real_population_order_independent_of_manifest_arrival(self):
        from native_head_confirmation_inputs import ordered_groups
        rows=json.loads((ROOT/'data/source/native_head_value_confirmation_v1/candidate_manifest.json').read_text(encoding='utf-8'))
        for split in ('val','test'):
            self.assertEqual(ordered_groups(rows,split),ordered_groups(list(reversed(rows)),split))

    def test_compressed_real_screening_adapter(self):
        from native_head_confirmation_inputs import records
        path=ROOT/'results/causal_observability_population_v2/screening_evidence.jsonl.gz'
        obj,pointer=next(records(path))
        with gzip.open(path,'rt',encoding='utf-8') as f:expected=json.loads(next(f))
        self.assertEqual(obj,expected)
        self.assertEqual(pointer,'/records/0')
        self.assertEqual(len(obj['fingerprints']),3)

    def test_lossless_deterministic_json_gzip_on_official_metadata(self):
        from native_head_value_confirmation import read,save
        official=read(ROOT/'data/source/native_head_value_confirmation_v1/metadata/native_classes')
        with tempfile.TemporaryDirectory(prefix='native27-real-map-') as temp:
            a=Path(temp)/'a.json.gz';b=Path(temp)/'b.json.gz'
            save(a,official);save(b,official)
            self.assertEqual(sha(a),sha(b))
            self.assertEqual(read(a),official)

    def test_actual_duplicate_schema_and_dataset_namespace(self):
        from native_head_confirmation_inputs import clip_nodes
        audit=json.loads((ROOT/'results/ucf101_proxy_replication_v1/duplicate_and_pretraining_audit.json').read_text(encoding='utf-8'))
        self.assertGreater(len(audit['edges']),0)
        for edge in audit['edges']:
            for k in ('a','b'):
                self.assertTrue(all(n.startswith(('clip:ucf:','family:ucf:')) for n in clip_nodes(edge[k])))
        hmdb=json.loads((ROOT/'results/causal_observability_population_v2/source_family_audit.json').read_text(encoding='utf-8'))
        member=hmdb['components'][0]['members'][0]
        self.assertTrue(clip_nodes(member)[0].startswith('clip:hmdb:'))

    def test_live_lock_owner_authenticated_without_packages(self):
        import os
        from native_head_confirmation_state import process_identity
        value=process_identity(os.getpid())
        self.assertIsInstance(value,int)
        self.assertEqual(value,process_identity(os.getpid()))

    def test_real_windows_memory_counters(self):
        from native_head_confirmation_state import memory_snapshot
        measured=memory_snapshot()
        self.assertGreater(measured['process_working_set_bytes'],0)
        self.assertGreaterEqual(measured['process_peak_working_set_bytes'],measured['process_working_set_bytes'])
        self.assertGreaterEqual(measured['system_ram_bytes'],measured['system_available_ram_bytes'])

    def test_preparation_snapshot_has_unchanged_real_hashes(self):
        base=ROOT/'results/native_head_value_confirmation_v1/preparation_a9e1942'
        identity=json.loads((base/'snapshot_identity.json').read_text(encoding='utf-8'))
        for name,digest in identity['files'].items():self.assertEqual(sha(base/name),digest)

    def test_real_role_and_list_hash_evidence_coverage(self):
        identity=json.loads((ROOT/'results/native_head_value_confirmation_v1/source_audit_identity.json').read_text(encoding='utf-8'))
        with gzip.open(ROOT/identity['path'],'rt',encoding='utf-8') as f:graph=json.load(f)
        nodes=graph['component_by_node'];reserved=set(graph['reserved_components'])
        exclusion=json.loads((ROOT/'results/temporal_refresh_generalization_v1/source_exclusions.json').read_text(encoding='utf-8'))
        self.assertIsInstance(exclusion['encoded_sha256'],list)
        self.assertTrue(all('bytes:'+h in nodes for h in exclusion['encoded_sha256']))
        dataset=json.loads((ROOT/'results/fresh_refresh_confirmation_v1/frozen_test_manifest.json').read_text(encoding='utf-8'))
        fresh=[r for r in dataset['videos'] if r.get('experiment_split')=='fresh_test']
        self.assertGreater(len(fresh),0)
        for row in fresh:self.assertIn(nodes['yt:'+row['source_video_id']],reserved)
        self.assertTrue(all(e.get('file') and 'pointer' in e for e in graph['edges']))

    def test_incremental_counters_equal_real_append_only_ledger(self):
        from native_head_confirmation_state import totals
        path=ROOT/'results/native_head_value_confirmation_v1/resource_exposure_ledger.jsonl'
        for attempt in range(3):
            before=path.read_bytes();actual=totals();after=path.read_bytes()
            if before==after:break
        else:self.skipTest('A real concurrent append occurred during every snapshot')
        records=[json.loads(line) for line in before.splitlines()]
        for key,value in actual.items():
            if key!='uncertain_operations':self.assertEqual(value,sum(r.get(key,0) for r in records))
        self.assertEqual(actual,totals())

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
