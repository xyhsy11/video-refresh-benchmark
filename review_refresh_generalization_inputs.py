"""Supplementary independent input/denominator review;never fits a model.

Run after the main audit. Rebuilds C from allowed periodic observations and
saved motion rows,then checks metrics with explicit scalar accumulation.
"""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def frame(path):return pd.read_csv(path,float_precision='round_trip')


def confidence(z):
    p=np.exp(z-z.max());p/=p.sum()
    q=sorted(p)
    return [-sum(float(x)*np.log(float(x)) for x in p),q[-1],q[-1]-q[-2]]


def run(output):
    p=read(output/'protocol.json')
    assert read(output/'status.json')['status']=='complete'
    recognition=ROOT/'results/recognition_baseline_v2'
    manifest=read(recognition/'dataset_manifest.json')['videos']
    head=read(recognition/'classifiers/20/parameters.json')
    observations=pd.concat([frame(ROOT/'results/cached_history_signal_diagnostic_v1_rerun1'/f'{role}_observations.csv') for role in ['development','exploratory']],ignore_index=True)
    base_columns=p['frozen_model_inputs']['C']
    maximum=0.;checks=0;missing=0;future_vectors=0;bframes=0
    for entry in manifest:
        if entry['recognition_role']=='train':continue
        directory=ROOT/entry['reference_directory']
        motion_path=directory/'motion_frames'/(entry['video_id']+'.csv')
        assert sha(motion_path)==entry['reference_artifact_sha256'][f"motion_frames/{entry['video_id']}.csv"]
        motion=frame(motion_path)
        assert len(motion)==entry['decoded_frame_count']
        # Both stored schemas use the actual zero-based decoded frame index.
        assert motion.decoded_frame_index.tolist()==list(range(len(motion)))
        with np.load(directory/'features'/(entry['video_id']+'.npz'),allow_pickle=False) as arr:x=arr['features'].astype(float)
        logits=((x-np.array(head['means']))/np.array(head['scales']))@np.array(head['weights']).T+np.array(head['bias'])
        group=observations[observations.source_video_id==entry['source_video_id']]
        for row in group.itertuples():
            t,r=int(row.t),int(row.last_refresh)
            assert r==4*(t//4)
            current_index=entry['candidate_indices'][t];cached_index=entry['candidate_indices'][r]
            current=motion.iloc[current_index]
            interval=motion.iloc[cached_index+1:current_index+1]
            available=interval[interval.mv_available==True]
            cached=logits[r];past=np.mean([logits[4*(j//4)] for j in range(t)],axis=0)
            values=[t/15,entry['candidate_timestamps_seconds'][t]-entry['candidate_timestamps_seconds'][r],3-r//4,r+4-t,
                current.mean_magnitude_px,current.p90_magnitude_px,float(current.mv_available),
                float(available.mean_magnitude_px.mean()) if len(available) else 0.,
                float(available.p90_magnitude_px.max()) if len(available) else 0.,len(available)/len(interval),
                *confidence(cached),*confidence(past),*(cached-cached.mean()),*(past-past.mean())]
            old=np.array([getattr(row,c) for c in base_columns])
            np.testing.assert_allclose(values,old,atol=1e-10,rtol=1e-9)
            maximum=max(maximum,float(abs(np.array(values)-old).max()));checks+=1
        missing+=int((~motion.mv_available.astype(bool)).sum())
        if 'future_reference_vectors' in motion:future_vectors+=int(motion.future_reference_vectors.sum())
        bframes+=entry.get('frame_type_counts',{}).get('B',0)
    decisions=frame(output/'opportunity_decisions.csv');summary=frame(output/'summary.csv')
    scalar_checks=0
    for row in summary.itertuples():
        rows=decisions[(decisions.cohort==row.cohort)&(decisions.method==row.method)]
        by_source={};selected=helpful=harmful=neutral=missed=beneficial=0.
        for r in rows.itertuples():
            a,g=float(r.accept_probability),float(r.G)
            by_source.setdefault(r.source_video_id,[]).append(a*g)
            selected+=a
            if g>1e-12:helpful+=a;beneficial+=1;missed+=1-a
            elif g< -1e-12:harmful+=a
            else:neutral+=a
        equal_source=sum(sum(v)/len(v) for v in by_source.values())/len(by_source)
        assert all(len(v)==9 for v in by_source.values())
        np.testing.assert_allclose([equal_source,selected,helpful,harmful,neutral,missed,beneficial],
            [row.utility_nats_per_opportunity,row.selected,row.helpful_selected,row.harmful_selected,row.neutral_selected,row.missed_benefit_numerator,row.missed_benefit_denominator],atol=1e-12,rtol=0)
        if selected:
            np.testing.assert_allclose(harmful/selected,row.harmful_selection_rate,atol=1e-12,rtol=0)
            np.testing.assert_allclose(sum(sum(v) for v in by_source.values())/selected,row.mean_gain_selected,atol=1e-12,rtol=0)
        else:assert pd.isna(row.harmful_selection_rate) and pd.isna(row.mean_gain_selected)
        scalar_checks+=1
    # Independently match original per-source utilities,without calling reports.
    old=frame(ROOT/'results/two_target_refresh_predictability_v1/per_video_metrics.csv')
    history=frame(ROOT/'results/cached_history_signal_diagnostic_v1_rerun1/per_video_metrics.csv')
    source=frame(output/'per_source_metrics.csv');reproduced=0
    for method,previous in [('direct_RF400','Direct_G_RF400'),('two_target_RF200x2','Two_target_RF200x2_forward'),('history_RF400','C_plus_T_RF400')]:
        rows=source[source.method==method]
        before=(history if method=='history_RF400' else old)
        before=before[before.method==previous].set_index(['cohort','source_video_id'])
        for r in rows.itertuples():
            saved=before.loc[(r.cohort,r.source_video_id)]
            np.testing.assert_allclose([r.utility,r.selected,r.helpful_selected,r.harmful_selected],
                [saved.utility,saved.selected,saved.helpful,saved.harmful],atol=1e-12,rtol=0)
            reproduced+=1
    reuse=read(recognition/'reused_membership.json')['videos'];ex=read(recognition/'exclusions.json')
    old_train={e['source_video_id'] for e in reuse if e['recognition_role']=='train'}
    new_train=set(p['membership']['train'])-old_train
    assert len(old_train)==80 and len(new_train)==320 and new_train.isdisjoint(ex['source_ids'])
    assert {e['file_identity']['sha256'] for e in manifest if e['source_video_id'] in new_train}.isdisjoint(ex['file_hashes'])
    result={'passed':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'additional_check_source_sha256':sha(Path(__file__)),
        'scope':'additional verification only;no model/threshold/feature changes',
        'independent_C_input_rebuilds':checks,'maximum_C_input_discrepancy':maximum,
        'explicit_scalar_summary_and_denominator_checks':scalar_checks,'old_per_source_utilities_and_counters_reproduced':reproduced,
        'recorded_B_frames':bframes,'recorded_future_reference_vectors':future_vectors,'frames_with_missing_motion':missing,
        'motion_CSV_hashes_checked':140,'new_training_sources_exclude_prior_reserved_ids_and_hashes':320,
        'previous_exclusively_training_sources_reused':80,'protected_confirmation_outcome_tables_read':0,
        'encoder_decodes_downloads_or_fits':0}
    (output/'independent_review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/refresh_generalization_audit_v1_rerun1'))
    args=parser.parse_args();run((ROOT/args.output).resolve())
