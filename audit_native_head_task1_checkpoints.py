"""Independent source/forward-journal audit; no decode, model, fitting or EVAL."""
import hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
from native_head_value_confirmation import read,save,sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/native_head_value_confirmation_v1'
CACHE=ROOT/'data/source/native_head_value_confirmation_v1'


def audit():
    errors=[]
    def check(ok,message):
        if not ok:errors.append(message)
    initial=read(OUT/'task1/initial_state.json')
    for name,digest in initial['frozen_code'].items():check(sha(ROOT/name)==digest,'frozen code '+name)
    for name,digest in initial['preserved_files'].items():
        p=OUT/'task1/baseline'/Path(name).name
        check(sha(p)==digest,'baseline snapshot '+name)
    raw=(OUT/'resource_exposure_ledger.jsonl').read_bytes()
    check(raw.startswith((OUT/'task1/baseline/resource_exposure_ledger.jsonl').read_bytes()),'append-only historical prefix')
    check(raw.endswith(b'\n'),'partial ledger tail')
    log=[json.loads(line) for line in raw.splitlines()]
    controls=read(OUT/'frozen_config.json')
    check(sha(OUT/'frozen_config.json')==initial['preserved_files']['results/native_head_value_confirmation_v1/frozen_config.json'],'original frozen configuration')
    operational=read(OUT/'task1/operational_controls.json')
    for name,digest in operational['sha256'].items():check(sha(ROOT/name)==digest,'frozen task1 code '+name)
    check(operational['caps']==controls['caps'],'unchanged cumulative caps')
    model=read(OUT/'model_identity.json')
    check(sha(ROOT/model['path'])==model['expected_sha256']==controls['checkpoint_sha256'],'actual model checkpoint')
    for name,digest in model['upstream_source_hashes'].items():check(sha(ROOT/name)==digest,'upstream model code '+name)
    implementation=dict(controls['implementation'])
    for name in ('protocol.md','frozen_config.json','class_mapping.json','model_identity.json','historical_source_evidence.json'):
        implementation['results/native_head_value_confirmation_v1/'+name]=sha(OUT/name)
    active={}; completed_sources={};journal_pairs={}
    for e in log:
        if e['kind']=='operation_start' and e.get('label')=='native-reference':active[e['operation_id']]=dict(start=e,forwards={},source_attempt=[])
        if e['kind'].startswith('task1_forward_'):
            ids=[key for key,item in active.items() if item['start'].get('phase')=='PROBE_FIT']
            check(len(ids)==1,'forward outside exactly one fitting source operation')
            if ids:
                bucket=active[ids[0]]['forwards'].setdefault(e['forward_id'],[]);bucket.append(e['kind'])
        if e['kind']=='source_collection_attempt':
            for item in active.values():
                if item['start']['source']==e['source']:item['source_attempt'].append(e)
        if e['kind']=='operation_finish' and e['operation_id'] in active:
            item=active.pop(e['operation_id'])
            if e.get('success'):
                source=item['start']['source'];completed_sources.setdefault(source,[]).append(item)
                if item['forwards']:
                    check(len(item['forwards'])==16,'16 durable forwards/source '+source)
                    check(all(v==['task1_forward_intent','task1_forward_completion'] for v in item['forwards'].values()),'forward intent/completion pairing '+source)
                    journal_pairs[source]=len(item['forwards'])
    check(not active,'unfinished native source operation')
    rows=read(OUT/'val_manifest.json.gz');references=[]
    for row in rows:
        key=hashlib.sha256(('native_head_value_confirmation_v1|20261023|feature|'+row['filename']).encode()).hexdigest()
        p=CACHE/'features'/(key+'.npz');receipt=p.with_suffix('.json')
        check(sha(ROOT/row['path'])==row['encoded_sha256'],'original input '+row['filename'])
        if not p.exists() and not receipt.exists():continue
        check(p.exists() and receipt.exists(),'partial reference '+row['filename'])
        if not (p.exists() and receipt.exists()):continue
        r=read(receipt)
        grid=hashlib.sha256(('native_head_value_confirmation_v1|20261023|grid|'+json.dumps(row['grid'],sort_keys=True)).encode()).hexdigest()
        expected=dict(encoded_sha256=row['encoded_sha256'],grid_sha256=grid,checkpoint_sha256=controls['checkpoint_sha256'],phase='PROBE_FIT',implementation=implementation)
        check(r['input']==expected,'source input/code/grid '+row['filename'])
        check(sha(p)==r['array_sha256'],'reference array hash '+row['filename'])
        check(r['encoder_calls']==r['completed_calls']==16,'completed source forwards '+row['filename'])
        source_events=completed_sources.get(row['component'],[])
        check(len(source_events)==1,'exactly one canonical reference collection '+row['filename'])
        if source_events:
            attempts=source_events[0]['source_attempt']
            check(len(attempts)==1 and attempts[0]['encoder_starts']==attempts[0]['encoder_completions']==16,'ledger source completion '+row['filename'])
        with np.load(p,allow_pickle=False) as a:
            check(a['features'].shape==(16,2048) and a['logits'].shape==(16,400) and a['thumbnails'].shape==(16,32,32),'dimensions '+row['filename'])
            check(all(np.isfinite(a[k]).all() for k in a.files),'finite outputs '+row['filename'])
        references.append(dict(component=row['component'],filename=row['filename'],y=row['y'],array_path=p.relative_to(ROOT).as_posix(),array_sha256=sha(p),receipt_sha256=sha(receipt)))
    operations=list((OUT/'task1/operations').glob('*.json'))
    for p in operations:
        r=read(p);check(r['status']=='complete','incomplete atomic checkpoint '+p.name)
        for name,digest in r.get('sha256',{}).items():check(sha(ROOT/name)==digest,'checkpoint artifact '+name)
        check(r['implementation_sha256']==read(OUT/'task1/operational_controls.json')['sha256']['fit_plan_native_head_confirmation.py'],'orchestration hash '+p.name)
    totals={k:sum(e.get(k,0) for e in log) for k in ('encoder_starts','encoder_completions','probe_fits','probe_completions','collection_seconds','cpu_seconds','timing_seconds')}
    from native_head_confirmation_state import totals as cumulative_totals
    canonical=cumulative_totals()
    differences={k:totals[k]-canonical[k] for k in totals}
    for k,difference in differences.items():
        check(abs(difference)<1e-8 if k.endswith('_seconds') else difference==0,'independent cumulative reconciliation '+k)
    check(totals['encoder_starts']==totals['encoder_completions']==len(references)*16+128,'total forwards includes extra historical profile')
    check(len(journal_pairs)==len(references)-8,'every new reference has per-forward journal')
    absence={name:not (OUT/name).exists() for name in ('joint_source_certificate.json','evaluation_seal.json','evaluation_exposure.json','test_manifest.json.gz','eval_source_summaries.json','summary.json','bootstrap_draws.npz')}
    check(all(absence.values()),'evaluation boundary artifact present')
    check(not any(e.get('phase')=='EVAL' for e in log),'EVAL ledger entry')
    check(not list((OUT/'probes').glob('final_*')),'final probe present')
    result=dict(passed=not errors,errors=errors,scope='Independent original-input, saved-reference and durable checkpoint audit; OOF empirical validity reported separately',
        verified_original_sources=len(rows),verified_reference_sources=len(references),new_journaled_sources=len(journal_pairs),
        verified_references=references,counters_independent=totals,evaluation_absence=absence,
        cumulative_reconciliation_differences=differences,
        model_checkpoint_sha256=model['expected_sha256'],class_mapping_sha256=sha(OUT/'class_mapping.json'),
        evaluation_model_calls=0,evaluation_outcomes_accessed=False,all_references_complete=len(references)==3200,
        oof_complete=(OUT/'task1/oof_verification.json').exists(),replay_not_model_throughput=True)
    return result


if __name__=='__main__':
    from run_native_head_value_confirmation import run_lock
    from native_head_confirmation_state import metered
    with run_lock():
        with metered('cpu','task1-independent-checkpoint-audit'):
            result=audit();save(OUT/'task1/checkpoint_verification.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='verified_references'},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
