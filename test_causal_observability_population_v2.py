"""SYNTHETIC TEST FIXTURE only. No empirical videos, models or GPU calls."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import inspect
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import causal_observability_population_v2 as p


def fixture(n=19, dt=Fraction(1, 30)):
    """SYNTHETIC TEST FIXTURE: declared rational timestamps, never research evidence."""
    ids = [dict(index=i, pts=i, time_base=str(dt), presentation_index=i) for i in range(n)]
    pts = [float(i*dt) for i in range(n)]
    return dict(key='SYNTHETIC/clip.avi', new_eligible=True, decoded_frames=n,
                presentation_identities=ids, grid=p.adaptive_grid(pts, ids))


def population():
    """SYNTHETIC TEST FIXTURE with cross-class families and fixed test groups."""
    rows, mapping = [], {}
    classes = ['SYNTHETIC_A', 'SYNTHETIC_B']
    for c in classes:
        for i in range(9):
            key = f'{c}/SYNTHETIC_train_{i}.avi'
            rows.append(dict(key=key, class_name=c, filename=key.split('/')[-1], official_flag=1,
                new_eligible=True, old_eligible=False, sha256=p.digest(key), technical_exclusion=None))
            mapping[key] = f'SYNTHETIC_SHARED_{i}'
        for g in range(5):
            for j in range(2):
                key = f'{c}/SYNTHETIC_test_{g}_{j}.avi'
                rows.append(dict(key=key, class_name=c, filename=key.split('/')[-1], official_flag=2,
                    new_eligible=True, old_eligible=False, sha256=p.digest(key), technical_exclusion=None))
                mapping[key] = f'{c}_SYNTHETIC_TEST_{g}'
    return classes, rows, mapping


class SyntheticPopulationTests(unittest.TestCase):
    def test_19_frames_exact_16_endpoints(self):
        r = fixture()
        self.assertEqual(r['grid']['endpoint_indices'], list(range(3, 19)))
        p.scalar_validate_frames(r)

    def test_reject_short_invalid_and_duplicate_PTS(self):
        for a in [np.arange(18), [0., 1., 1.], [0., float('nan')], [2., 1.]]:
            with self.assertRaises(ValueError):
                p.adaptive_grid(a)

    def test_real_identity_contract_and_no_future(self):
        for n in [19, 20, 35, 60, 100, 101, 523]:
            r = fixture(n)
            p.scalar_validate_frames(r)
            for i, indices in zip(r['grid']['endpoint_indices'], r['grid']['frame_indices']):
                self.assertEqual(len(set(indices)), 4)
                self.assertEqual(indices[-1], i)
                self.assertTrue(all(j <= i for j in indices))

    def test_window_extension_only_when_needed(self):
        r = fixture(19, Fraction(2, 1))
        self.assertTrue(all(r['grid']['window_extended']))
        self.assertTrue(all(d == 6 for d in r['grid']['history_window_seconds']))
        p.scalar_validate_frames(r)
        fast = fixture(200, Fraction(1, 100))
        self.assertFalse(any(fast['grid']['window_extended']))

    def test_variable_rate_and_original_decode_ordinals(self):
        from ucf101_proxy_data import presentation_order
        points = np.cumsum([1+i%7 for i in range(50)])
        order = np.arange(50)[::-1]
        ids = [dict(index=j, pts=int(points[i]), time_base='1/100') for j, i in enumerate(order)]
        frames, pts, ids, reordered = presentation_order(list(order), (points[order]/100).tolist(), ids)
        r = dict(new_eligible=True, decoded_frames=50, presentation_identities=ids, grid=p.adaptive_grid(pts, ids))
        self.assertGreater(reordered, 0)
        p.scalar_validate_frames(r)
        self.assertEqual(ids[0]['index'], 49)
        self.assertFalse(np.allclose(np.diff(r['grid']['endpoint_timestamps']), np.diff(r['grid']['endpoint_timestamps'])[0]))

    def test_scalar_replay_rejects_corruption_and_future_frames(self):
        r = fixture(35)
        r['grid']['frame_indices'][0][-1] += 1
        with self.assertRaises(p.PopulationError):
            p.scalar_validate_frames(r)
        r = fixture()
        r['presentation_identities'][3]['pts'] = 2
        with self.assertRaises(p.PopulationError):
            p.scalar_validate_frames(r)

    def test_source_links_and_global_connected_components(self):
        rows = [dict(key=f'SYNTHETIC_{i}', sha256='SAME' if i < 2 else str(i),
                     title_family='TITLE' if i in (1, 2) else str(i), fingerprints=[], pts_span=1.) for i in range(4)]
        edges = p.source_edges(rows)
        mapping, groups = p.components(rows, edges)
        self.assertEqual(mapping['SYNTHETIC_0'], mapping['SYNTHETIC_2'])
        self.assertNotEqual(mapping['SYNTHETIC_0'], mapping['SYNTHETIC_3'])
        audit = dict(edges=edges, components=[dict(source_family=f, members=m) for f, m in groups.items()])
        self.assertEqual(p.graph_replay(rows, audit), mapping)

    def test_fixed_phash_threshold_and_all_matching_edges(self):
        rows = [dict(key=f'SYNTHETIC_{i}', sha256=str(i), title_family=str(i), pts_span=1.,
                     fingerprints=[format(v, '063b')]*3) for i, v in enumerate([0, 15, 31])]
        edges = [e for e in p.source_edges(rows) if e['kind'] == 'near']
        self.assertEqual({(e['first'], e['second']) for e in edges},
                         {('SYNTHETIC_0', 'SYNTHETIC_1'), ('SYNTHETIC_1', 'SYNTHETIC_2')})

    def test_test_priority_including_ineligible_and_unused_bridge(self):
        rows = [dict(key=str(i), official_flag=f, new_eligible=(i != 2), old_eligible=False)
                for i, f in enumerate([1, 0, 2, 1])]
        mapping = {'0':'SYNTHETIC_shared', '1':'SYNTHETIC_shared', '2':'SYNTHETIC_shared', '3':'SYNTHETIC_free'}
        train, test, _ = p.availability(rows, mapping)
        self.assertEqual([r['key'] for r in train], ['3'])
        self.assertEqual(test, [])

    def test_global_MILP_minima_cross_class_and_repeat(self):
        classes, rows, mapping = population()
        counts = p.counts_by_class(rows, classes, mapping)
        with patch.dict(p.CONFIGS, {'SYNTHETIC': [3, 3, 3]}):
            result = p.solve_configuration(rows, classes, mapping, counts, 'SYNTHETIC')
        self.assertEqual(result['status'], 'FEASIBLE')
        self.assertTrue(result['deterministic_repeat_verified'])
        self.assertEqual(len(result['assignments']), 9)
        for r in result['independent_counts']:
            self.assertEqual(r['head_train_clips'], 3)
            self.assertEqual(r['development_families'], 3)

    def test_independent_assignment_rejects_missing_family_or_role(self):
        classes, rows, mapping = population()
        assignment = {f'SYNTHETIC_SHARED_{i}': p.ROLES[i//3] for i in range(9)}
        p.validate_assignment(rows, classes, mapping, assignment, [3, 3, 3])
        assignment.pop('SYNTHETIC_SHARED_0')
        with self.assertRaises(p.PopulationError):
            p.validate_assignment(rows, classes, mapping, assignment, [3, 3, 3])

    def test_necessary_certificate_is_not_solver_timeout(self):
        counts = [dict(class_name='SYNTHETIC', training_available=32, training_families=8,
                       new_test_eligible=9, test_families=4)]
        proof = p.necessary_failures(counts, [20, 8, 5])
        self.assertEqual(len(proof), 4)
        self.assertEqual({r['constraint'] for r in proof},
                         {'training_available', 'training_families', 'new_test_eligible', 'test_families'})

    def test_manifest_corruption_and_foreign_archive_rejected(self):
        classes, rows, mapping = population()
        manifest = p.allocation_manifest(rows, mapping, None, {})
        inventory = dict(videos={r['key']:dict(sha256=r['sha256']) for r in rows})
        manifest['videos'][0]['sha256'] = 'SYNTHETIC_CORRUPTED'
        with self.assertRaises(p.PopulationError):
            p.validate_leakage(manifest, rows, inventory, mapping, {})
        manifest = p.allocation_manifest(rows, mapping, None, {})
        inventory['videos'].pop(rows[0]['key'])
        with self.assertRaises(p.PopulationError):
            p.validate_leakage(manifest, rows, inventory, mapping, {})

    def test_official_unused_cannot_enter_training(self):
        classes, rows, mapping = population()
        rows[0]['official_flag'] = 0
        manifest = p.allocation_manifest(rows, mapping, None, {})
        self.assertEqual(manifest['videos'][0]['role'], 'excluded_official_unused')
        manifest['videos'][0]['role'] = 'head_train'
        inventory = dict(videos={r['key']:dict(sha256=r['sha256']) for r in rows})
        with self.assertRaises(p.PopulationError):
            p.validate_leakage(manifest, rows, inventory, mapping, {})

    def test_exact_duplicates_and_families_cannot_cross_roles(self):
        for link in ['exact', 'family']:
            classes, rows, mapping = population()
            for r in rows:
                r.update({k: v for k, v in fixture().items() if k not in ['key']})
            if link == 'exact':
                rows[1]['sha256'] = rows[0]['sha256']
            else:
                mapping[rows[1]['key']] = mapping[rows[0]['key']]
            manifest = p.allocation_manifest(rows, mapping, None, {})
            manifest['videos'][0]['role'] = 'head_train'
            manifest['videos'][1]['role'] = 'development'
            inventory = dict(videos={r['key']:dict(sha256=r['sha256']) for r in rows})
            with self.assertRaisesRegex(p.PopulationError, 'crosses roles'):
                p.validate_leakage(manifest, rows, inventory, mapping, {})

    def test_official_test_never_becomes_training(self):
        classes, rows, mapping = population()
        manifest = p.allocation_manifest(rows, mapping, None, {})
        i = next(i for i, r in enumerate(rows) if r['official_flag'] == 2)
        manifest['videos'][i]['role'] = 'head_train'
        inventory = dict(videos={r['key']:dict(sha256=r['sha256']) for r in rows})
        # Reach the official flag check before frame replay, using labeled fixtures.
        for r in rows:
            r.update({k: v for k, v in fixture().items() if k != 'key'})
        with self.assertRaisesRegex(p.PopulationError, 'Official test/unused'):
            p.validate_leakage(manifest, rows, inventory, mapping, {})

    def test_checkpoint_resume_corruption_and_tail_preservation(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_TEST_FIXTURE_') as folder:
            path = Path(folder)/'journal.jsonl'
            identity = dict(SYNTHETIC_TEST_FIXTURE=True, rule='SYNTHETIC')
            payload = dict(key='SYNTHETIC_A')
            item = dict(identity=identity, payload=payload, payload_sha256=p.digest(payload))
            path.write_text(json.dumps(item)+'\n', encoding='utf-8')
            self.assertEqual(p.load_journal(path, identity)['SYNTHETIC_A'], payload)
            with self.assertRaises(p.PopulationError):
                p.load_journal(path, {'CHANGED': True})
            with path.open('a') as f:
                f.write('{"interrupted')
            with self.assertRaises(p.PopulationError):
                p.load_journal(path, identity)
            self.assertEqual(p.load_journal(path, identity, recover=True)['SYNTHETIC_A'], payload)
            self.assertEqual(len(list(Path(folder).glob('*.interrupted-*'))), 1)
            item['payload']['key'] = 'SYNTHETIC_TAMPER'
            path.write_text(json.dumps(item)+'\n', encoding='utf-8')
            with self.assertRaises(p.PopulationError):
                p.load_journal(path, identity)

    def test_audit_mutation_rejection_without_execution(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_TEST_FIXTURE_') as folder:
            root = Path(folder); out = root/'out'; out.mkdir()
            target = root/'SYNTHETIC_evidence'; target.write_bytes(b'SYNTHETIC TEST FIXTURE')
            p.save(out/'SHA256SUMS.json', {'SYNTHETIC_evidence':p.sha(target)})
            target.write_bytes(b'SYNTHETIC CORRUPTION')
            with patch.object(p, 'ROOT', root), patch.object(p, 'OUT', out), \
                 patch.object(p, 'screen_clip', side_effect=AssertionError('Decoder forbidden')), \
                 patch.object(p, 'solve_configuration', side_effect=AssertionError('Solver forbidden')), \
                 patch.object(p, 'save', side_effect=AssertionError('Write forbidden')):
                with self.assertRaises(p.PopulationError):
                    p.audit()

    def test_population_code_has_no_model_or_outcome_interface(self):
        import ast
        tree = ast.parse(inspect.getsource(p))
        forbidden = {'torch', 'pytorchvideo', 'causal_observability_runtime', 'causal_observability_learning'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(forbidden.intersection(n.name for n in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, forbidden)
        signature = inspect.signature(p.adaptive_grid)
        self.assertEqual(list(signature.parameters), ['timestamps', 'identities'])
        self.assertNotIn('label', inspect.signature(p.solve_configuration).parameters)


if __name__ == '__main__':
    unittest.main()
