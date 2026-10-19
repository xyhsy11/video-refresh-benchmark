"""Read-only REAL input/provenance/estimand review; --write saves its receipt.

No decoder, CUDA, training, or official-test outcomes. Run after core analysis.
"""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import ucf101_causal_development_common as c


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    c.GUARD.install();manifest=c.read(c.OUT/'population_manifest.json');c.GUARD.set_population(manifest,diagnostics=True)
    c.validate_population(manifest)
    for v in manifest['videos']:
        assert c.sha(ROOT/v['path'])==v['sha256'],'Original video changed'
    head=c.read(c.OLD/'heads/parameters.json')
    lineage=dict(training_collection_matches=head['training_collection_sha256']==c.sha(c.OLD/'train_collection.json'),
        development_collection_matches=head['development_collection_sha256']==c.sha(c.OLD/'development_collection.json'),
        training_manifest_matches=head['training_manifest_sha256']==c.sha(c.OLD/'eligible_manifest.json'))
    assert all(lineage.values()),'Head data lineage mismatch'
    frame=pd.read_csv(c.OUT/'development_per_clip_metrics.csv',float_precision='round_trip')
    results={}
    for method in frame['method'].unique():
        selected=frame[frame['method']==method];class_means=[]
        for label in manifest['classes']:
            members=selected[selected['label']==label];group_means=[]
            for group in sorted(members['group'].unique()):
                clips=members[members['group']==group];group_means.append(float(sum(clips['CE'])/len(clips)))
            class_means.append(sum(group_means)/len(group_means))
        results[method]=sum(class_means)/len(class_means)
    intervals=pd.read_csv(c.OUT/'development_CE_contrasts.csv',float_precision='round_trip')
    primary=intervals[intervals['contrast']=='periodic_minus_history_motion'].iloc[0]
    actual=results['periodic']-results['history_motion']
    assert abs(actual-primary['mean'])<1e-12,'Independent nested class/group/clip contrast differs'
    trials=pd.read_csv(c.OUT/'timing_trials.csv',float_precision='round_trip')
    expected=np.where(trials['method']=='dense16',16,4)
    assert np.array_equal(trials['encoder_starts'],expected) and np.array_equal(trials['encoder_completions'],expected)
    assert len(trials)==1470 and len(trials[~trials['warmup']])==1050
    assert trials['clip'].nunique()==30 and trials['label'].nunique()==30
    event_rows=[json.loads(line) for line in (c.OUT/'encoder_events.jsonl').read_text().splitlines()]
    counts=Counter((r['phase'],r['event']) for r in event_rows)
    assert sum(r['event']=='start' for r in event_rows)==sum(r['event']=='complete' for r in event_rows)
    assert all(r['clip'] in c.GUARD.entries for r in event_rows)
    auxilia={}
    for name in ['refresh_pooling_utils.py','temporal_refresh_timing.py']:
        original=subprocess.check_output(['git','show',c.read(c.OUT/'repository_review.json')['actual_main']+':'+name],cwd=ROOT)
        expected_hash=hashlib.sha256(original).hexdigest();assert c.sha(ROOT/name)==expected_hash
        auxilia[name]=expected_hash
    record=dict(passed=True,scope='REAL DEVELOPMENT input/provenance and independent scalar estimand review',
        original_approved_AVI_hashes_verified=len(manifest['videos']),head_data_lineage=lineage,
        class_group_clip_CE=results,primary_development_contrast=actual,
        timing_trials=len(trials),measured_timing_trials=len(trials[~trials['warmup']]),
        encoder_starts=sum(r['event']=='start' for r in event_rows),encoder_completions=sum(r['event']=='complete' for r in event_rows),
        phase_events=[dict(phase=k[0],event=k[1],count=n) for k,n in sorted(counts.items())],
        auxiliary_helpers_unchanged_from_reviewed_main=auxilia,
        supplementary_check_time=c.now(),source_code_sha256=c.sha(Path(__file__)),
        attempt_UTC_note='execution_attempts.at_utc identifies the attempt start on both start/complete rows; actual encoder clock_seconds and measured seconds record execution timing.',
        calls_or_metrics_from_official_test=0,decodes=0,model_calls=0,CUDA_imported='torch' in sys.modules)
    assert not record['CUDA_imported']
    if args.write:c.save(c.OUT/'independent_integrity_review.json',record)
    print(json.dumps(record,indent=2,allow_nan=False))


if __name__=='__main__':main()
