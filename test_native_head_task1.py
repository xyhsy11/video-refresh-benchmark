"""Control-flow and genuine-input contracts; no simulated empirical outcomes."""
import ast
from contextlib import nullcontext
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import fit_plan_native_head_confirmation as task
from native_head_value_confirmation import read,save,sha,folds,precision_plan
from verify_native_head_task1 import expected_fold_membership,precision_independent


class Task1Contracts(unittest.TestCase):
    def test_no_eval_or_test_acquisition_reachable_from_fit_plan(self):
        tree=ast.parse(Path(task.__file__).read_text())
        names={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
        self.assertFalse(names & {'acquire','certify','plan','plan_once','run','execute','finish_evaluation','timing','finalize'})
        collection_calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='collect']
        self.assertEqual(len(collection_calls),1)
        self.assertEqual(ast.literal_eval(collection_calls[0].args[2]),'PROBE_FIT')
        self.assertIn("require(phase=='PROBE_FIT'",inspect.getsource(task.FittingEncoder.forward))

    def test_ordinary_profile_uses_real_receipts_and_excludes_cached_replay(self):
        profile=task.ordinary_profile()
        self.assertTrue(profile['passed'])
        self.assertEqual(profile['genuine_forwards'],128)
        self.assertAlmostEqual(profile['mean_seconds'],3.654882625000255,places=10)
        self.assertEqual(profile['safety_factor'],1.5)
        self.assertGreater(profile['sample_sd_seconds'],0)

    def test_cumulative_caps_and_frozen_code_preserved(self):
        config=read(task.OUT/'frozen_config.json')
        self.assertEqual(config['caps'],task.CAPS)
        self.assertEqual(config['implementation'],{name:sha(task.ROOT/name) for name in task.CODE})
        for key,limit in [('collection_seconds',28800),('cpu_seconds',14400),('timing_seconds',7200)]:self.assertEqual(task.CAPS[key],limit)

    def test_session_pauses_only_at_explicit_safe_boundary(self):
        before=task.totals()
        with tempfile.TemporaryDirectory() as d,patch.object(task,'TASK',Path(d)):
            session=task.Session(.001)
            session.next_operation='next real source'
            session.checkpoint('RUNNING')
            with self.assertRaises(task.PausedSession):session.before('next real source',1)
            receipt=read(Path(d)/'sessions'/(session.id+'.json'))
            self.assertEqual(receipt['status'],'PAUSED_SESSION')
            self.assertEqual(receipt['next_operation'],'next real source')
            self.assertEqual(receipt['counters_before'],receipt['counters'])
        self.assertEqual(before,task.totals())

    def test_existing_authentic_source_reuses_without_model_calls(self):
        row=read(task.OUT/'smoke_manifest.json')[0]
        before=task.totals()
        with patch.object(task.NativeEncoder,'forward',side_effect=AssertionError('unexpected model call')):
            self.assertTrue(task.authenticate_reference(row))
        self.assertEqual(before,task.totals())

    def test_partial_operation_blocks_without_recompute(self):
        with tempfile.TemporaryDirectory() as d,patch.object(task,'TASK',Path(d)):
            save(Path(d)/'operations/unfinished.json',dict(status='started'))
            with self.assertRaisesRegex(RuntimeError,'BLOCKED_ACCOUNTING'):task.boundary()

    def test_resource_ledger_independent_roundoff_bounded_sum(self):
        import math
        records=task.events();counters=task.totals()
        for key,value in counters.items():
            if key=='uncertain_operations':continue
            independent=math.fsum(r.get(key,0) for r in records)
            if key.endswith('_seconds'):self.assertLess(abs(value-independent),1e-8)
            else:self.assertEqual(value,independent)

    def test_fold_membership_and_integrity_guard(self):
        rows=read(task.OUT/'val_manifest.json.gz');actual=folds(rows)
        np.testing.assert_array_equal(actual,expected_fold_membership(rows))
        for c in range(400):self.assertEqual(np.bincount(actual[[i for i,r in enumerate(rows) if r['y']==c]]).tolist(),[2,2,2,2])
        with tempfile.TemporaryDirectory() as d,patch.object(task,'TASK',Path(d)):
            save(Path(d)/'folds/0.json',dict(fold=0,manifest_sha256='deliberately-invalid-identity',folds=actual.tolist()))
            with self.assertRaisesRegex(RuntimeError,'fold membership mismatch'):task.load_fold(0,actual,{},rows)

    def test_precision_arithmetic_on_real_input_metadata_only(self):
        # Actual original file byte lengths exercise the arithmetic; these are
        # not task effects, are never persisted as precision evidence or m.
        rows=read(task.OUT/'val_manifest.json.gz');metadata=[r['encoded_bytes'] for r in rows]
        a=precision_plan(rows,metadata);b=precision_independent(rows,metadata)
        self.assertEqual(a['selected_m'],b['selected_m'])
        np.testing.assert_allclose(a['per_class_variance'],b['per_class_variance'],rtol=1e-14)
        for m in range(8,21):
            for factor in ('1.0','1.5','2.0'):self.assertAlmostEqual(a['projections'][str(m)][factor],b['projections'][str(m)][factor],places=8)
        with self.assertRaises(ValueError):precision_plan(rows,metadata[:-1])


if __name__=='__main__':unittest.main()
