"""Small software fixtures; synthetic data never enters scientific outputs."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import task_signal_utils as s
import task_aware_oracle_utils as task
from refresh_policy_utils import CheapVideo, ReplayVideo
from verify_task_signals import direct_replay


class TaskSignalChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.Generator(np.random.PCG64(19))
        self.rgb = rng.normal(size=(16,512)).astype(np.float32)
        self.head = {"weights": (rng.normal(size=(20,512))*.02).tolist(), "bias": [0.]*20,
                     "means": [0.]*512, "scales": [1.]*512}
        frames = tuple({"decoded_frame_index": i, "mv_available": False,
                        "mean_magnitude_px": 0., "p90_magnitude_px": 0.} for i in range(31))
        self.video = ReplayVideo({"label": "class0"}, CheapVideo("source", tuple(range(0,32,2)),
                                tuple(np.arange(16)*.5), frames), self.rgb)

    def test_exact_budget_histories_signed_independent_targets(self):
        logits = task.feature_logits(self.rgb, self.head)
        for r in [4,8,12]:
            for t in range(r-3,r):
                positions = sorted([p for p in s.PERIODIC if p != r]+[t])
                base, history = direct_replay(self.rgb, s.PERIODIC, self.head, 2)
                early, early_history = direct_replay(self.rgb, positions, self.head, 2)
                np.testing.assert_array_equal(history[:t], early_history[:t])
                losses = task.classification_loss(np.stack([logits[task.cache_indices(p)].mean(axis=0)
                                                  for p in [s.PERIODIC,positions]]), 2)
                self.assertAlmostEqual(base-early, losses[0]-losses[1], places=12)
                self.assertEqual(len(set(positions)),4)

    def test_no_label_current_future_rgb_or_future_motion_inputs(self):
        before = s.observed_inputs(self.video, 6, self.head)
        rgb = self.rgb.copy()
        rgb[[i for i in range(16) if i not in [0,4]]] = 999
        motion = tuple(dict(m, mean_magnitude_px=999.) if i>12 else m
                       for i,m in enumerate(self.video.cheap.motion))
        changed = replace(self.video, references=rgb, entry={"label":"not the target"},
                          cheap=replace(self.video.cheap,motion=motion))
        self.assertEqual(before,s.observed_inputs(changed,6,self.head))
        self.assertEqual(set(before),set(s.GROUPS["D"]))
        self.assertFalse(set(before)&{"label","refresh_gain","privileged_current_drift","dense_prediction"})

    def test_missing_motion_and_confidence(self):
        x = s.observed_inputs(self.video,2,self.head)
        self.assertTrue(all(x[c]==0 for c in s.MOTION_INPUTS))
        np.testing.assert_allclose(s.confidence(np.zeros(20)),[np.log(20),.05,0.],atol=1e-12)
        self.assertEqual(x["remaining_refresh_budget"],3)
        self.assertEqual(x["positions_until_planned_refresh"],2)

    def test_ridge_sum_objective_constant_columns(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame(rng.normal(size=(30,4)),columns=s.AGE)
        frame[s.AGE[0]]=1.
        frame["refresh_gain"]=rng.normal(size=30)
        frame["source_video_id"]=[f"s{i//3}" for i in range(30)]
        frame["example_id"]=[str(i) for i in range(30)]
        model=s.fit_ridge(frame,"A")
        self.assertEqual(model["scales"][0],1.)
        z=(frame[s.AGE].to_numpy()-model["means"])/model["scales"]
        beta=np.linalg.lstsq(np.vstack([z,np.eye(4)]),
            np.r_[frame.refresh_gain-frame.refresh_gain.mean(),np.zeros(4)],rcond=None)[0]
        np.testing.assert_allclose(beta,model["coefficients"],atol=1e-12)

    def test_class_stratified_source_folds(self):
        classes=[f"class{i}" for i in range(20)]
        entries=[{"recognition_role":"development","label":c,"source_video_id":f"{c}:{j}","video_id":f"{c}:{j}"}
                 for c in classes for j in range(5)]
        folds=s.make_folds(entries,classes,20261017)
        self.assertEqual(folds,s.make_folds(list(reversed(entries)),classes,20261017))
        for f in folds["folds"]:
            self.assertEqual(len(f["training_sources"]),80)
            self.assertEqual(len(f["validation_sources"]),20)
            self.assertFalse(set(f["training_sources"])&set(f["validation_sources"]))
            selected=[r["label"] for r in folds["videos"] if r["fold"]==f["fold"]]
            self.assertEqual(set(selected),set(classes))

    def test_heldout_targets_do_not_change_fold_model_or_predictions(self):
        from analyze_task_signals import fit_probes
        rng=np.random.default_rng(9)
        rows=[]
        for i in range(100):
            for t in range(9):
                rows.append({"source_video_id":f"s{i}","video_id":f"v{i}","example_id":f"{i}:{t}",
                    "t":t,"last_refresh":0,"refresh_gain":float(rng.normal()),
                    **dict(zip(s.GROUPS["D"],rng.normal(size=16)))})
        table=pd.DataFrame(rows)
        folds={"folds":[{"fold":f,"training_sources":[f"s{i}" for i in range(100) if i%5!=f],
                         "validation_sources":[f"s{i}" for i in range(100) if i%5==f]} for f in range(5)]}
        with tempfile.TemporaryDirectory() as first,tempfile.TemporaryDirectory() as second:
            p=fit_probes(table,folds,Path(first))
            changed=table.copy()
            changed.loc[changed.source_video_id.isin(folds["folds"][0]["validation_sources"]),"refresh_gain"]+=1000
            q=fit_probes(changed,folds,Path(second))
            self.assertEqual(p["fold_models"]["0"],q["fold_models"]["0"])
            a=pd.read_csv(Path(first)/"out_of_fold_predictions.csv")
            b=pd.read_csv(Path(second)/"out_of_fold_predictions.csv")
            np.testing.assert_allclose(a[a.fold==0].prediction,b[b.fold==0].prediction)

    def test_undefined_statistics_and_ties(self):
        self.assertIsNone(s.auc(np.arange(5),np.ones(5))["AUROC"])
        self.assertIsNone(s.correlation(np.ones(5),np.arange(5))["spearman"])
        self.assertEqual(s.auc(np.zeros(4),np.array([0,1,0,1]))["AUROC"],.5)
        np.testing.assert_array_equal(s.signs(np.array([-1.,-1e-14,0,1e-14,1.])),[-1,0,0,0,1])


if __name__ == "__main__":
    unittest.main()
