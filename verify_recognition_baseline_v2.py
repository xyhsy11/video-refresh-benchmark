"""Independent cohort, frozen-source, label and download-accounting checks."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from recognition_baseline_data import read,subsets,validate_membership,CACHE
from fresh_confirmation_data import strict_json
from research_utils import ROOT,now,sha256_file


def verify(output:Path) -> dict:
    protocol=read(output/'protocol.json');plan=read(output/'sampling_plan.json');exclusions=read(output/'exclusions.json')
    for name,key in [('sampling_plan.json','sampling_plan_sha256'),('exclusions.json','exclusions_sha256'),('reused_membership.json','reused_membership_sha256')]:
        assert sha256_file(output/name)==protocol[key],name
    for name,digest in protocol['implementation_sha256'].items():assert sha256_file(ROOT/name)==digest,name
    annotations=pd.read_csv(CACHE/'val.csv');assert sha256_file(CACHE/'val.csv')==plan['annotation_sha256']
    lookup=annotations.set_index(['youtube_id','time_start','time_end'])['label']
    reused=read(output/'reused_membership.json')['videos'];original={e['source_video_id']:e for e in reused}
    complete=(output/'dataset_manifest.json').exists()
    if complete:entries=read(output/'dataset_manifest.json')['videos']
    else:
        progress=read(output/'collection_progress.json') if (output/'collection_progress.json').exists() else {'videos':[]}
        entries=[*reused,*[v['entry'] for v in progress['videos']]]
    counts=validate_membership(entries,complete)
    fresh=read(ROOT/'results/fresh_refresh_confirmation_v1/frozen_test_manifest.json')['videos']
    assert not {e['source_video_id'] for e in entries}&{e['source_video_id'] for e in fresh}
    assert not {e['file_identity']['sha256'] for e in entries}&{e['file_identity']['sha256'] for e in fresh}
    for entry in entries:
        source=entry['source_video_id'];key=(source,entry['time_start'],entry['time_end'])
        assert key in lookup.index and lookup.loc[key]==entry['label'],source
        if source in original:
            expected=original[source]
            assert entry['recognition_role']==expected['recognition_role'] and entry['file_identity']==expected['file_identity']
        else:
            assert source not in exclusions['source_ids'] and entry['file_identity']['sha256'] not in exclusions['file_hashes']
    budget=read(output/'download_accounting.json')
    assert sum(e['actual_bytes'] for e in budget['requests'])==budget['actual_body_bytes']
    assert budget['actual_body_bytes']+budget['uncertain_interruption_bytes_charged']+sum(budget['inflight'].values())<=protocol['max_download_bytes']
    result={'passed':True,'checked_at_utc':now(),'dataset_complete':complete,'available_counts':counts,
            'official_labels_and_source_identities_verified':True,'frozen_reuse_roles_and_protected_evaluation_exclusions':True,
            'all_HTTP_reads_reconciled_and_cap_respected':True,'verification_source_sha256':sha256_file(Path(__file__))}
    if complete:
        chosen=read(output/'training_subsets.json')['subsets']
        assert chosen==subsets(entries,protocol['classes'])
        by_id={e['source_video_id']:e for e in entries}
        for n in [4,10,20]:
            assert Counter(by_id[source]['label'] for source in chosen[str(n)])==Counter({c:n for c in protocol['classes']})
        result['nested_training_subsets_class_balance']=True
        if (output/'frozen_models.json').exists():
            frozen=read(output/'frozen_models.json')['parameter_sha256']
            for n in [4,10,20]:
                folder=output/'classifiers'/str(n);selected=read(folder/'selection.json')
                assert sha256_file(folder/'parameters.json')==frozen[str(n)]==selected['parameters_sha256']
                assert selected['training_source_ids']==chosen[str(n)] and not selected['exploratory_metrics_used']
                result['frozen_selection_and_scaler_identities']=True
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('results/recognition_baseline_v2'));args=parser.parse_args()
    result=verify(args.output);strict_json(args.output/'independent_verification.json',result);print(result)
