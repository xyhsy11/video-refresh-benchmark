"""Focused recognition expansion and accounting fixtures, never real evidence."""
from __future__ import annotations
import io
from pathlib import Path
import tempfile
from unittest import TestCase,main
import urllib.request

import numpy as np

from recognition_baseline_data import ByteBudget,subsets,validate_membership
from task_aware_oracle_utils import exact_task_oracle,cache_indices,classification_loss


class Response(io.BytesIO):
    def __init__(self,body): super().__init__(body);self.headers={'Content-Length':str(len(body))}


class RecognitionChecks(TestCase):
    def test_body_bytes_and_retry_accounting_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            calls=[]
            def opener(req,*args,**kwargs):calls.append(req.get_method());return Response(b'abcde')
            budget=ByteBudget(Path(folder)/'budget.json',10,opener)
            for _ in range(2):
                with budget.open('https://example.invalid/file') as response:self.assertEqual(response.read(),b'abcde')
            self.assertEqual(budget.used(),10)
            with self.assertRaises(RuntimeError):budget.open('https://example.invalid/file')
            self.assertEqual(calls.count('GET'),2)

    def test_interrupted_reservation_conservatively_charged(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'budget.json';budget=ByteBudget(path,10,lambda *a,**k:Response(b'abcde'))
            response=budget.open('https://example.invalid/file');self.assertEqual(response.read(2),b'ab')
            resumed=ByteBudget(path,10,lambda *a,**k:Response(b'abcde'))
            self.assertEqual(resumed.state['actual_body_bytes'],2)
            self.assertEqual(resumed.state['uncertain_interruption_bytes_charged'],3)

    def test_failed_read_charges_remaining_reservation(self):
        with tempfile.TemporaryDirectory() as folder:
            budget=ByteBudget(Path(folder)/'b.json',10,lambda *a,**k:Response(b'abcde'))
            with self.assertRaises(ValueError):
                with budget.open('https://example.invalid/file') as response:
                    response.read(2);raise ValueError('simulated failure')
            self.assertEqual(budget.used(),5)

    def test_balanced_nested_subsets_source_hash_disjoint(self):
        classes=[f'class{i}' for i in range(20)];entries=[]
        for label in classes:
            for role,count in [('train',20),('development',5),('exploratory',2)]:
                for i in range(count):
                    source=f'{label}_{role}_{i}'
                    entries.append({'source_video_id':source,'file_identity':{'sha256':source},'label':label,'recognition_role':role,'candidate_indices':list(range(16)),'decoded_frame_count':16})
        self.assertEqual(validate_membership(entries),{'train':400,'development':100,'exploratory':40})
        selected=subsets(entries,classes);self.assertEqual(selected,subsets(entries[::-1],classes))
        self.assertEqual([len(selected[str(n)]) for n in [4,10,20]],[80,200,400])
        self.assertTrue(set(selected['4'])<=set(selected['10'])<=set(selected['20']))
        entries[-1]['source_video_id']=entries[0]['source_video_id']
        with self.assertRaises(AssertionError):validate_membership(entries)

    def test_independent_exact_oracle_and_cache(self):
        from itertools import combinations
        logits=np.random.default_rng(11).normal(size=(8,3));losses=[]
        for tail in combinations(range(1,8),3):
            selected=[0,*tail];r=0;out=[]
            for t in range(8):
                if t in selected:r=t
                out.append(logits[r])
            mean=np.mean(out,axis=0);high=mean.max();loss=np.log(np.exp(mean-high).sum())+high-mean[1]
            losses.append((loss,selected))
        best=min(v[0] for v in losses);positions=next(p for loss,p in losses if loss<=best+1e-12)
        actual=exact_task_oracle(logits,1,4);self.assertEqual(actual['positions'],positions)
        self.assertAlmostEqual(actual['minimum_cross_entropy'],best,places=12)
        n16=np.random.default_rng(12).normal(size=(16,20));o=exact_task_oracle(n16,3,4)
        self.assertEqual(o['enumerated_schedules'],455)
        self.assertLessEqual(o['minimum_cross_entropy'],classification_loss(n16[cache_indices([0,4,8,12])].mean(axis=0)[None],3)[0]+1e-12)


if __name__=='__main__':main()
