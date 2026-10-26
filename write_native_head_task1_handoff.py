"""Write a fitting-only handoff from persisted evidence, with no empirical work."""
import json,math,statistics,time
from pathlib import Path
from native_head_value_confirmation import read,save,sha
from run_native_head_value_confirmation import ROOT,OUT,now,git,run_lock,CAPS
from native_head_confirmation_state import metered,state
from fit_plan_native_head_confirmation import accounting,events

RESUME=r'.\.venv_temporal\Scripts\python.exe -B fit_plan_native_head_confirmation.py fit-plan --session-wall-seconds 21600'
TASK2=r'.\.venv_temporal\Scripts\python.exe -B run_native_head_value_confirmation.py acquire --split test'
NEW_CODE=('fit_plan_native_head_confirmation.py','verify_native_head_task1.py','test_native_head_task1.py',
    'audit_native_head_task1_checkpoints.py','report_native_head_task1_diagnostics.py',
    'propose_native_head_task1_resources.py','write_native_head_task1_handoff.py','verify_native_head_task1_resume.py',
    'assemble_native_head_task1_handoff.py')


def build():
    status=read(OUT/'task1_status.json');audit=read(OUT/'task1/checkpoint_verification.json')
    initial=read(OUT/'task1/initial_state.json');log=events()
    checked=read(OUT/'task1/oof_verification.json') if (OUT/'task1/oof_verification.json').exists() else None
    pp=read(OUT/'precision_plan.json') if (OUT/'precision_plan.json').exists() else None
    diagnostic=read(OUT/'task1/oof_prediction_diagnostics.json') if (OUT/'task1/oof_prediction_diagnostics.json').exists() else None
    proposal=read(OUT/'task1_resource_proposal.json')
    phase_costs={}
    for e in log:
        if e['kind']=='operation_finish':
            key=e.get('label','unlabeled');item=phase_costs.setdefault(key,dict(operations=0,successful_operations=0,collection_seconds=0.,cpu_seconds=0.,timing_seconds=0.))
            item['operations']+=1;item['successful_operations']+=bool(e.get('success'))
            for field in ('collection_seconds','cpu_seconds','timing_seconds'):item[field]+=e.get(field,0)
    negative={}
    if (OUT/'fit_oof_source_summaries.json').exists():
        sources=read(OUT/'fit_oof_source_summaries.json')
        for key in ('primary','drift_contrast'):
            values=[r[key] for r in sources]
            negative[key]=dict(mean=math.fsum(values)/len(values),negative=sum(x<0 for x in values),exact_zero=sum(x==0 for x in values),positive=sum(x>0 for x in values))
    code={name:sha(ROOT/name) for name in NEW_CODE if (ROOT/name).exists()}
    frozen={name:dict(before=digest,after=sha(ROOT/name),unchanged=sha(ROOT/name)==digest) for name,digest in initial['frozen_code'].items()}
    r=dict(experiment='native_head_value_confirmation_v1',task=1,at_utc=now(),technical=status['technical'],scientific='NOT_EVALUATED',
        blocker=status['blocker'],completed_stage=state()['stage'],
        fitting=dict(certified_sources=3200,verified_original_sources=audit['verified_original_sources'],
            verified_reference_sources=audit['verified_reference_sources'],all_references_complete=audit['all_references_complete'],
            native_forward_completions=audit['counters_independent']['encoder_completions'],
            new_native_forward_completions=audit['counters_independent']['encoder_completions']-initial['counters']['encoder_completions'],
            extra_historical_profile_forwards=128,completed_oof_fits=audit['counters_independent']['probe_completions'],
            completed_fold_checkpoints=len(list((OUT/'task1/folds').glob('*.json')))),
        independent_verification=dict(checkpoint_audit_passed=audit['passed'],oof=checked,
            authentic_resume=read(OUT/'task1/resume_verification.json') if (OUT/'task1/resume_verification.json').exists() else None,
            fitting_stage_passed=bool(audit['passed'] and checked and checked['passed'] and status['technical']=='OOF_COMPLETE')),
        precision=pp,source_level_development_effects=negative,development_diagnostics=diagnostic,
        session=status['session'],caps=CAPS,phase_resource_receipts=phase_costs,
        process_edge_timing=read(OUT/'task1/production_process_time.json'),
        proposal=dict(path='task1_resource_proposal.md',machine_readable_path='task1_resource_proposal.json',activated=False,cases=proposal['cases']),
        provenance=dict(base_commit=initial['repository_commit'],repository_head_during_execution=git('rev-parse','HEAD'),
            branch=git('branch','--show-current'),delivery_commit='Commit containing this handoff; resolve with git log -1 --format=%H -- results/native_head_value_confirmation_v1/task1_handoff.json',
            original_code=frozen,added_code_sha256=code,frozen_config_sha256=sha(OUT/'frozen_config.json'),
            model_identity=read(OUT/'model_identity.json'),class_mapping_sha256=sha(OUT/'class_mapping.json'),
            operational_amendment='task1_operational_amendment.md',original_exposure_relabelled=False,environment=read(OUT/'task1/current_environment.json')),
        validation=dict(commands=read(OUT/'task1/validation_commands.json'),focused_final=read(OUT/'task1/focused_tests_final.json'),
            final_publication_checks=read(OUT/'task1/publication_checks.json') if (OUT/'task1/publication_checks.json').exists() else None,
            historical_suite_limitation='23/24 passed. Exact float-equality ledger assertion differed by 9.1e-13 seconds under Python 3.12 summation. Frozen test and counters unchanged; independent exact discrete and 1e-8-second floating reconciliation passed.'),
        boundaries=dict(evaluation_source_certification_started=False,evaluation_sealed=False,evaluation_model_calls=0,
            evaluation_outcomes_accessed=False,test_archive_acquisition_started=False,final_probes_fitted=False,final_timing_executed=False),
        resumption=dict(task1_command=RESUME,next_pending_operation=('Task 2 budget authorization required; no Task 1 empirical work remains' if status['technical']=='OOF_COMPLETE' else status['session']['next_operation']),
            task2_recommended_entrypoint=TASK2 if status['technical']=='OOF_COMPLETE' else None,
            task2_prerequisites='Explicit user authorization of revised cumulative limits, operational budget amendment with preserved identity evidence, and a passing complete-study resource gate. Not executed.',
            large_data='Original videos and reference arrays remain in the established ignored data/source directory. A fresh checkout alone cannot resume without those authenticated inputs.'),
        limitations=['OOF is fitting-stage development evidence, not fresh evaluation confirmation.',
            'OOF within-class variances use eight fitting components/class and may not transfer to the official test population. Shared training folds can induce dependence; fixed 1.5 inflation is a prespecified allowance, not a coverage guarantee.',
            'Ordinary profile component times were not separately instrumented. Decode/inference, I/O and collection overhead are measured jointly; offline/authentication work stays separately charged.',
            'Source/fold checkpoints are atomic. A hard interruption with incomplete intent blocks for audit instead of silent reexecution.',
            'All-source independent scalar verification is one read-only operation; the runner may pause before it if its conservative reserve does not fit the remaining session.',
            'Historical failed preparation wall time remains unknown. Previously charged interrupted transfer upper bounds are retained. Git/control network traffic is separate from dataset HTTP-body accounting.'])
    if not audit['passed'] or not all(v['unchanged'] for v in frozen.values()):
        r['technical']='BLOCKED_VERIFICATION';r['blocker']='Final independent checkpoint or frozen identity audit failed'
    return r


def write_report(r):
    c=r['resources'];f=r['fitting'];p=r['precision'];m=p['selected_m'] if p else None
    lines=['# E27 Task 1 handoff','',f"**Stage status: {r['technical']}. Scientific status: NOT_EVALUATED.**",'',
        f"Formal completed stage: {r['completed_stage']}. Blocker: {r['blocker'] or 'none'}.",'',
        f"Verified original fitting sources: {f['verified_original_sources']:,}. Verified complete reference sets: {f['verified_reference_sources']:,}/3,200. Native forward completions: {f['native_forward_completions']:,} ({f['new_native_forward_completions']:,} new in Task 1; total includes 128 extra historical ordinary-profile forwards). Completed OOF fits: {f['completed_oof_fits']}; complete fold checkpoints: {f['completed_fold_checkpoints']}/4.",
        f"Independent source/reference/checkpoint audit: {'PASS' if r['independent_verification']['checkpoint_audit_passed'] else 'FAIL'}. Complete fitting-stage independent verification: {r['independent_verification']['fitting_stage_passed']}.",'',
        '## OOF development evidence','']
    if r['development_diagnostics']:
        lines+=['| Target | View | Held-out MAE | Training-fold constant MAE | Spearman |','|---|---|---:|---:|---:|']
        for target,views in r['development_diagnostics']['targets'].items():
            for view,d in views.items():lines.append(f"| {target} | {view} | {d['mae']:.8f} | {d['constant_heldout_mae']:.8f} | {d['spearman']} |")
        lines+=['',f"Source-level effects (all negative/zero/positive effects retained): `{json.dumps(r['source_level_development_effects'],sort_keys=True)}`.",
            'Per-fold training/held-out errors and class/source distributions are in fit_diagnostics.json, fit_oof_source_summaries.json and task1/oof_prediction_diagnostics.json. Lower drift error does not establish better refresh utility.']
    else:lines+=['Complete OOF development diagnostics are not available. No empirical finding is invented.']
    lines+=['','## Precision selection','']
    if p:
        lines+=[f"Selected m: {m if m is not None else 'none; BLOCKED_PRECISION_OR_POPULATION'}. Sum of 400 unbiased within-class variances: {math.fsum(p['per_class_variance']):.12g}.",
            'Rule: h(m) = 1.96 × 1.5 × sqrt(sum(s_c²) / (400² × m)); choose the smallest integer 8–20 with h(m) ≤ 0.005.','',
            '| m | h, inflation 1 | h, inflation 1.5 | h, inflation 2 |','|---:|---:|---:|---:|']
        for candidate,values in p['projections'].items():lines.append(f"| {candidate} | {values['1.0']:.8f} | {values['1.5']:.8f} | {values['2.0']:.8f} |")
    else:lines+=['Selected m is unknown. Genuine complete OOF effects are required; no evaluation size is assumed.']
    lines+=['','The OOF variance transfer limitation is recorded below. precision_plan.json contains all 400 variance inputs, the full sensitivity table and input hashes.','',
        '## Resources','',f"Compute-session wall: {r['session']['wall_seconds']:.3f} seconds (allowance 21,600). Additional read-only audit/publication work is charged separately to the cumulative ledger. Resource snapshot UTC: {r['resource_snapshot_utc']}.",'',
        '| Resource | Cumulative actual/charged | Authorized cap | Remaining |','|---|---:|---:|---:|']
    for label,key in [('Collection seconds','collection_seconds'),('CPU-budget seconds','cpu_seconds'),('Timing seconds','timing_seconds'),('HTTP response bytes charged','network_bytes'),('Additional retained bytes','retained_bytes')]:
        actual=c['network_cap_charged_bytes'] if key=='network_bytes' else c[key]
        lines.append(f"| {label} | {actual:,.3f} | {CAPS[key]:,.0f} | {c['remaining'][key]:,.3f} |")
    lines += ['',f"Known HTTP bodies: {c['known_response_body_bytes']:,} bytes; conservative excess charge: {c['network_cap_charged_bytes']-c['known_response_body_bytes']:,} bytes. Scratch archive bytes: {c['scratch_bytes']:,}. Free disk: {c['free_disk_bytes']/2**30:.3f} GiB; mandatory reserve: 10 GiB.",
        f"Production process startup wall charged separately: {r['process_edge_timing']['startup_wall_seconds']:.6f} seconds. Session-end-to-first-stopped-observation upper bound charged to CPU: {r['process_edge_timing']['post_session_upper_seconds']:.6f} seconds; this deliberately includes observation delay and is not claimed as exact teardown time.",
        f"Unresolved operations: {json.dumps(c['uncertain_operations'])}; uncertain native calls: {c['uncertain_encoder_calls']}. Acquisition wall upper-bound extra already included in collection: {c['acquisition_upper_bound_extra_seconds']:.3f} seconds. CPU counters are exclusive phase wall-budget seconds, not operating-system CPU-cycle measurements.",'',
        '## Tasks 2–4 proposal','', 'See task1_resource_proposal.md and task1_resource_proposal.json for measured rates, workload calculations, minimum estimates, conservative headroom and four-working-day feasibility. No budget increase is activated.','']
    for candidate,case in r['proposal']['cases'].items():lines.append(f"- m={candidate}: future Tasks 2–4 projected wall {case['expected_future_stage_wall_seconds']/3600:.2f} hours; recommended cumulative limits `{json.dumps(case['recommended_cumulative'],sort_keys=True)}`.")
    lines+=['','## Provenance and validation','',f"Execution base commit: `{r['provenance']['base_commit']}`. Branch: `{r['provenance']['branch']}`. Resolve the delivery commit with `git log -1 --format=%H -- results/native_head_value_confirmation_v1/task1_handoff.json`.",
        'Original frozen scientific code/configuration hashes remain unchanged. Added code hashes, exact model/class identities and environment comparison are in task1_handoff.json. task1/initial_state.json and task1/baseline preserve the initial checkpoint. task1/artifact_sha256_inventory.json binds the delivered evidence; original videos/reference arrays are separately bound by authenticated manifests and the reference audit.',
        r['validation']['historical_suite_limitation'],
        'The focused Task 1 suite passed all nine tests. Initial failed focused-test output is preserved separately from its corrected pre-exposure pass. validate_release.py and final publication checks are reported in the machine-readable validation section. Implementation tests are not empirical confirmation.','',
        '## Execution boundary and resumption','',
        'Evaluation source certification has not started; no joint certificate, test acquisition, final probe, evaluation seal, evaluation model call, outcome access, or final timing run occurred.','',
        'Resume or verify Task 1 with:','','```powershell',RESUME,'```','']
    if r['technical']=='OOF_COMPLETE':lines+=['After explicit revised-budget authorization and a passing complete-study gate, the recommended Task 2 acquisition entrypoint is:','','```powershell',TASK2,'```','','This command was not executed. Task 2 first needs a reviewed operational budget amendment that preserves existing exposure identities.']
    else:lines+=[f"Next pending operation: {r['resumption']['next_pending_operation']}. Resolve any blocker before invoking the resume command. Do not use the old integrated plan/run/resume paths for Task 1."]
    lines+=['','## Limitations','']+['- '+s for s in r['limitations']]+['']
    (OUT/'task1_handoff.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')


if __name__=='__main__':
    with run_lock():
        with metered('cpu','task1-handoff-preparation'):
            result=build()
            inventory={}
            excluded={'task1_handoff.md','task1_handoff.json','task1/artifact_sha256_inventory.json','resource_exposure_ledger.jsonl','run.lock','val_manifest.json','test_manifest.json'}
            for p in OUT.rglob('*'):
                if not p.is_file():continue
                key=p.relative_to(OUT).as_posix()
                if key in excluded or p.name.endswith('.writing') or (p.name in ('val_manifest.json','test_manifest.json') and 'certificates' in p.parts):continue
                inventory[key]=dict(sha256=sha(p),bytes=p.stat().st_size)
        # The preparation receipt is complete before the ledger snapshot is bound.
        inventory['resource_exposure_ledger.jsonl']=dict(sha256=sha(OUT/'resource_exposure_ledger.jsonl'),bytes=(OUT/'resource_exposure_ledger.jsonl').stat().st_size)
        save(OUT/'task1/artifact_sha256_inventory.json',dict(scope='Snapshot before handoff serialization; excludes the handoff and inventory to avoid circular hashes. Original videos/feature arrays are bound by the manifest and independent checkpoint audit.',files=inventory))
        result['resources']=accounting();result['resource_snapshot_utc']=now()
        result['artifact_inventory_sha256']=sha(OUT/'task1/artifact_sha256_inventory.json')
        result['resource_snapshot_scope']='Completed ledger operations through handoff preparation; final serialization, Git publication and interactive control are outside this snapshot.'
        save(OUT/'task1_handoff.json',result);write_report(result)
        print(json.dumps(dict(technical=result['technical'],fitting=result['fitting'],selected_m=result['precision']['selected_m'] if result['precision'] else None,resources=result['resources']),indent=2))
