"""Independent CPU read-only review of genuine training inputs and old evidence.

Default prints a receipt; --write retains it in this NEW experiment only.
No decoder, Torch, X3D, controller fitting or official-test data access.
"""
from pathlib import Path
import argparse
import gzip
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import task_utility_observability_study as s
import task_utility_observability as m


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    s.GUARD.install();manifest,_,_,_,_=s.load_inputs()
    old=s.read(s.PRIOR/'training_identity.json')
    path=next(p for p in old['files'] if p.endswith('/base_examples.json.gz'))
    s.GUARD.approved.add((s.ROOT/path).resolve());s.verify({path:old['files'][path]})
    with gzip.open(s.ROOT/path,'rt',encoding='utf-8') as f:base=json.load(f)
    byclip={}
    for row in base:byclip.setdefault(row['source'],[]).append(row)
    ledger=s.read(s.OUT/'observation_provenance.json');count=0;max_error=0.
    for clip in byclip:
        state=s.read(s.DATA/'clips'/f'{clip}.json')['states']
        periodic=[(i,r) for i,r in enumerate(state) if r['population']=='periodic' and r['discretionary']]
        with np.load(s.DATA/'clips'/f'{clip}.npz',allow_pickle=False) as f:x=f['observations']
        for (index,new),oldrow in zip(periodic,byclip[clip][:len(periodic)]):
            assert (new['t'],new['age'],new['remaining'])==(oldrow['t'],oldrow['age'],oldrow['remaining'])
            replay=np.r_[x[index,:113],x[index,114:]]
            error=float(abs(replay-np.asarray(oldrow['input'])).max());assert error<1e-12
            assert new['teacher_refresh']==oldrow['target'];max_error=max(max_error,error);count+=1
    # Independent nested scalar weighting of saved gains and squared errors.
    import csv
    with (s.OUT/'probe_metrics.csv').open(newline='',encoding='utf-8') as f:metrics=list(csv.DictReader(f))
    item=next(v for v in s.read(s.OUT/'prediction_inventory.json') if v['role']=='execution' and v['population']=='periodic')
    states,a=s.read_prediction(item);results=[]
    for ti,target in enumerate(m.TARGETS):
        for vi,view in enumerate(m.VIEWS):
            class_means=[]
            for label in manifest['classes']:
                group_means=[]
                for group in sorted({r['group'] for r in states if r['label']==label}):
                    clip_means=[]
                    for clip in sorted({r['clip'] for r in states if r['label']==label and r['group']==group}):
                        ids=[i for i,r in enumerate(states) if r['clip']==clip]
                        squared=[(a['calibrated'][ti,vi,i]-float(a['gains'][ti,i]>0))**2 for i in ids]
                        clip_means.append(sum(squared)/len(squared))
                    group_means.append(sum(clip_means)/len(clip_means))
                class_means.append(sum(group_means)/len(group_means))
            actual=sum(class_means)/len(class_means)
            row=next(r for r in metrics if r['role']=='execution' and r['population']=='periodic' and r['stratum']=='all' and r['calibration']=='selection_calibrated' and r['target']==target and r['view']==view)
            assert abs(actual-float(row['Brier']))<1e-12
            results.append(dict(target=target,view=view,independent_nested_Brier=actual))
    # Reproduce original H+M teacher-imitation counts using unchanged prefixes.
    items=next(v for v in s.read(s.OUT/'prediction_inventory.json') if v['population']=='HM_executed')
    states,a=s.read_prediction(items);conf={k:0 for k in ['TP','FP','TN','FN']}
    for r in states:
        conf['TP' if r['selected'] and r['teacher_refresh'] else 'FP' if r['selected'] else 'FN' if r['teacher_refresh'] else 'TN']+=1
    context=s.read(s.PRIOR/'teacher_imitation_context.json');assert conf==context['confusion']
    assert len(states)==context['decisions']
    out=dict(passed=True,original_periodic_training_states_reproduced=count,
        old_base_examples_sha256=old['files'][path],maximum_shared_input_difference=max_error,
        original_HM_teacher_confusion_reproduced=conf,original_HM_discretionary_states=len(states),
        independently_nested_Brier=results,new_GPU_calls=0,new_decodes=0,source_code_sha256=s.sha(Path(__file__)))
    if args.write:s.save(s.OUT/'independent_replay_review.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
