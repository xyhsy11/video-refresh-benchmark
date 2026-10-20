"""Explicitly SYNTHETIC CPU unit fixtures; never scientific observations."""
import copy
import unittest
import numpy as np
from scipy.stats import spearmanr
import refresh_failure_evidence as m
import run_refresh_failure_evidence as runner


class SyntheticAuditContracts(unittest.TestCase):
    def test_weighting_does_not_count_repeated_states_as_independent_sources(self):
        rows = [dict(label='A',group='g1',clip='one')]*9 + [dict(label='A',group='g2',clip='two')]
        w = m.weights(rows)
        self.assertAlmostEqual(w[:9].sum(), .5)
        self.assertAlmostEqual(w[-1], .5)

    def test_class_names_scope_group_ids(self):
        rows = [dict(label='A',group='g1',clip='a'),dict(label='B',group='g1',clip='b')]
        np.testing.assert_array_equal(m.weights(rows), [.5,.5])

    def test_weighted_rank_ties_match_scipy_with_equal_weights(self):
        x, y = [0,0,2,3,3], [3,2,1,1,0]
        self.assertAlmostEqual(m.weighted_spearman(x,y,np.ones(5)/5), spearmanr(x,y).statistic)

    def test_constant_proxy_has_no_correlation(self):
        self.assertIsNone(m.weighted_spearman([1,1],[0,1],[.5,.5]))

    def test_bad_correlation_is_rejected(self):
        with self.assertRaises(RuntimeError):
            m.weighted_spearman([1,np.nan],[0,1],[.5,.5])

    def test_proxy_is_not_an_event_annotation(self):
        rr = [dict(label='A',group='g',clip='a',gain_local=-1.,gain_budget=1.),
              dict(label='A',group='g',clip='a',gain_local=1.,gain_budget=-1.)]
        result = m.proxy_associations(rr, np.zeros((2,32)))
        self.assertEqual(len(result),12)
        self.assertTrue(all(r['weighted_spearman'] is None for r in result))
        self.assertTrue(all('event' not in r for r in result))

    def quality_fixture(self):
        rows, predictions = [], []
        for label, clip in [('A','a'),('B','b')]:
            for method in ('periodic','history','history_motion'):
                rows.append(dict(label=label,clip=clip,group='g',method=method,CE=np.log(2),
                                 correct=int(label=='A'),encoder_starts=4,encoder_completions=4))
                predictions.append(dict(clip=clip,method=method,logits=[0.,0.]))
        return rows,predictions,['A','B'],[('A','g'),('B','g')],np.array([[0,1],[0,1]])

    def test_quality_replays_logits_and_preserves_paired_draws(self):
        result = m.reproduce_quality(*self.quality_fixture())
        self.assertEqual(result['contrasts'][0]['CI95'],[0.,0.])

    def test_saved_mean_cannot_replace_real_logits(self):
        args = self.quality_fixture();args[1][0]['logits']=[1.,0.]
        with self.assertRaisesRegex(RuntimeError,'CE inconsistent'):
            m.reproduce_quality(*args)

    def test_bootstrap_cannot_cross_classes(self):
        args=list(self.quality_fixture());args[-1]=np.array([[1,0]])
        with self.assertRaisesRegex(RuntimeError,'class boundaries'):
            m.reproduce_quality(*args)

    def test_draw_block_order_can_differ_from_group_table(self):
        args=list(self.quality_fixture())
        args[2]=['B','A'];args[4]=np.array([[1,0],[1,0]])
        for r in args[0]:r['correct']=int(r['label']=='B')
        self.assertEqual(m.reproduce_quality(*args)['contrasts'][0]['CI95'],[0.,0.])

    def test_actual_forward_budget_is_verified(self):
        args=self.quality_fixture();args[0][0]['encoder_starts']=5
        with self.assertRaisesRegex(RuntimeError,'forward accounting'):
            m.reproduce_quality(*args)

    def test_unpaired_quality_rejected(self):
        args=list(self.quality_fixture());args[0]=args[0][1:]
        with self.assertRaisesRegex(RuntimeError,'Unpaired'):
            m.reproduce_quality(*args)

    def population_fixture(self):
        records=[]
        for i,role in enumerate(m.ROLES):
            r=dict(label='A',role=role,clip=str(i),family=str(i),duplicate_component=str(i),sha256=str(i))
            r.update({k:True for k in ('authorized','bytes_authenticated','labels_authenticated','pts_verified',
                'family_identity_verified','duplicates_screened','license_verified','storage_verified',
                'compute_verified','outcome_unexposed')})
            records.append(r)
        return records,['A'],dict.fromkeys(m.ROLES,1)

    def test_empty_population_is_blocked(self):
        self.assertEqual(m.validate_population_records([],[],{})['status'],'BLOCKED_INPUT')

    def test_positive_input_fixture_is_not_scientific_go(self):
        result=m.validate_population_records(*self.population_fixture())
        self.assertEqual(result['status'],'PASS_INPUT_CERTIFICATE')
        self.assertFalse(result['scientific_GO'])

    def test_each_missing_input_condition_blocks(self):
        for field in ('authorized','pts_verified','license_verified','compute_verified','outcome_unexposed'):
            args=self.population_fixture();args[0][0].pop(field)
            self.assertEqual(m.validate_population_records(*args)['status'],'BLOCKED_INPUT')

    def test_global_duplicate_and_family_quarantine(self):
        for field in ('family','duplicate_component','sha256'):
            args=self.population_fixture();args[0][0][field]=args[0][1][field]
            self.assertIn('cross_role_identity',m.validate_population_records(*args)['reasons'])

    def test_cannot_drop_a_class_to_pass_quotas(self):
        args=list(self.population_fixture());args[1]=['A','B']
        self.assertEqual(m.validate_population_records(*args)['status'],'BLOCKED_INPUT')

    def test_official_test_and_traversal_paths_denied(self):
        for name in ('results/ucf101_proxy_replication_v1/test_per_clip.csv',
                     'data/source/ucf101_proxy_replication_v1/videos/test.avi',
                     'results/refresh_failure_evidence_v1/../ucf101_proxy_replication_v1/test.json'):
            with self.assertRaises(RuntimeError):runner.check_path(runner.ROOT/name)

    def test_historical_writes_and_network_gpu_decoder_denied(self):
        with self.assertRaises(RuntimeError):runner.check_path(runner.ROOT/'results/task_utility_observability_v1/new.json',True)
        with self.assertRaises(RuntimeError):runner.hook('socket.connect',())
        for module in ('torch','av','cupy'):
            with self.assertRaises(RuntimeError):runner.hook('import',(module,))

    def test_read_only_guard_prohibits_even_current_result_write(self):
        old=runner.READ_ONLY
        try:
            runner.READ_ONLY=True
            with self.assertRaises(RuntimeError):runner.check_path(runner.OUT/'audit.json',True)
        finally:runner.READ_ONLY=old

    def test_incomplete_timing_population_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'timing cohort'):
            m.reconcile_cost([])

    def test_timing_uses_per_clip_medians_and_keeps_nested_fields_separate(self):
        trials=[]
        methods=['periodic','history','history_motion','motion','RandomB4_seeded','dense16','periodic_unused_gate']
        for method in methods:
            for clip in range(30):
                for rep in range(5):
                    seconds=1.+(method=='history_motion')*.2+rep*.01
                    trials.append(dict(method=method,clip=str(clip),warmup='False',repetition=str(rep),
                        encoder_starts=16 if method=='dense16' else 4,
                        encoder_completions=16 if method=='dense16' else 4,
                        seconds=seconds,cheap_thumbnail_seconds=.1,cheap_statistics_seconds=.02,
                        controller_decide_seconds=.03,observation_preparation_seconds=.15))
        result=m.reconcile_cost(trials)
        hm=next(r for r in result['methods'] if r['method']=='history_motion')
        self.assertAlmostEqual(hm['seconds'],1.22)
        self.assertAlmostEqual(result['incremental'][0]['paired_median_extra_seconds'],.2)
        self.assertAlmostEqual(hm['observation_preparation_seconds'],.15)
        self.assertEqual(result['measured_trials'],1050)


if __name__ == '__main__':
    unittest.main()
