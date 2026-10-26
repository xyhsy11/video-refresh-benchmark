"""Read-only future workload estimation; writes proposal, never activates caps."""
import json,math,statistics
import numpy as np
from pathlib import Path
from native_head_value_confirmation import read,save,sha
from native_head_confirmation_availability import remaining_work_projection,remaining_release_transfer
from native_head_confirmation_inputs import TRANSFER_IO_RECIPE
from run_native_head_value_confirmation import ROOT,OUT,CAPS
from fit_plan_native_head_confirmation import accounting,events


def proposal():
    log=events();current=accounting()
    profile=read(OUT/'normal_reference_throughput_diagnosis.json')
    observed=[e['collection_seconds'] for e in log if e['kind']=='operation_finish' and e.get('label')=='native-reference' and e.get('success')][8:]
    rate=max(profile['total_seconds']/8,statistics.mean(observed) if observed else 0)
    transfer=[e for e in log if e['kind']=='archive_transfer' and e.get('complete') and e.get('bytes',0)>0 and e.get('io_recipe')==TRANSFER_IO_RECIPE]
    bytes_remaining=remaining_release_transfer('test')
    seconds_per_byte=sum(e['acquisition_seconds'] for e in transfer)/sum(e['bytes'] for e in transfer)
    pp=read(OUT/'precision_plan.json') if (OUT/'precision_plan.json').exists() else None
    selected=pp['selected_m'] if pp else None
    models=list((OUT/'probes').glob('[0-3]_*.json'))
    fold_receipts=[e['cpu_seconds'] for e in log if e['kind']=='operation_finish' and e.get('label')=='task1-oof-fold' and e.get('success')]
    # Outer-fold time includes matrix assembly, scaling, prediction and writes;
    # using only the solver receipt would omit real CPU work.
    final_fit=(statistics.mean(fold_receipts)*(4/3)*1.5) if len(models)==32 and len(fold_receipts)==4 else 1800.
    smoke_rate=sum(read(p)['genuine_resource_elapsed_seconds'] for p in (ROOT/'data/source/native_head_value_confirmation_v1/features').glob('*.json') if read(p).get('native_checks'))/8
    timing=60*7*4*smoke_rate*2
    features=[p for p in (ROOT/'data/source/native_head_value_confirmation_v1/features').glob('*.npz')]
    feature_bytes=statistics.mean(p.stat().st_size+p.with_suffix('.json').stat().st_size for p in features)
    per_source_cpu=0.
    for label in ('offline-source-enumeration','source-input-controls','task1-source-resume-authentication'):
        measured=[e['cpu_seconds'] for e in log if e['kind']=='operation_finish' and e.get('label')==label and e.get('success')]
        if measured:per_source_cpu+=statistics.mean(measured)
    oof_audits=[e['cpu_seconds'] for e in log if e['kind']=='operation_finish' and e.get('label')=='task1-independent-oof-verification' and e.get('success')]
    fitting_memory=read(OUT/'task1_status.json')['resources'].get('memory') if (OUT/'task1_status.json').exists() else None
    cuda_memory={key:max(read(p.with_suffix('.json')).get(key,0) for p in features) for key in ('cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')}
    setups=[e['collection_seconds'] for e in log if e['kind']=='operation_finish' and e.get('label') in
        ('native-model-setup','genuine-normal-reference-model-setup','task1-native-model-setup') and e.get('success')]
    cases={}
    for m in ([selected] if selected else [8,20]):
        work=remaining_work_projection(m,log)
        stages=dict(test_transfer_seconds=bytes_remaining*seconds_per_byte*1.2,
            future_native_model_setup_seconds=2*max(setups)*1.5,
            test_catalog_seconds=work['test_catalog_projection_seconds'],test_screening_seconds=work['test_screening_projection_seconds'],
            test_duplicate_seconds=work['test_duplicate_projection_seconds'],evaluation_reference_seconds=400*m*rate*1.5,
            final_probe_fit_seconds=final_fit,independent_verification_seconds=max(work['independent_verification_projection_seconds'],max(oof_audits,default=0)/3200*(400*m)*1.5),
            evaluation_offline_cpu_seconds=per_source_cpu*400*m*1.5,
            complete_system_timing_seconds=timing,future_audit_analysis_cpu_reserve_seconds=1800.)
        collection=sum(stages[k] for k in ('test_transfer_seconds','test_catalog_seconds','test_screening_seconds','test_duplicate_seconds','evaluation_reference_seconds','future_native_model_setup_seconds'))
        cpu=stages['final_probe_fit_seconds']+stages['independent_verification_seconds']+stages['evaluation_offline_cpu_seconds']+1800
        network=current['network_cap_charged_bytes']+bytes_remaining
        retained=current['retained_bytes']+work['projected_test_original_bytes']+400*m*feature_bytes*1.3
        still_fitting=(3200-len(features))*rate*1.5
        # Incomplete Task 1 cannot disappear from a cumulative proposal.
        verification_done=(OUT/'task1/oof_verification.json').exists() and read(OUT/'task1/oof_verification.json')['passed']
        unfinished_cpu=read(OUT/'task1/fitting_resource_projection.json')['remaining_cpu_projection_seconds'] if not verification_done else 0
        minimum=dict(network_bytes=network,retained_bytes=math.ceil(retained),
            collection_seconds=current['collection_seconds']+still_fitting+collection,
            cpu_seconds=current['cpu_seconds']+unfinished_cpu+cpu,timing_seconds=current['timing_seconds']+timing)
        frozen_collection_gate=minimum['collection_seconds']-stages['evaluation_reference_seconds']+400*m*smoke_rate*1.5
        source_audit_share=min(stages['independent_verification_seconds'],2*sum(e.get('cpu_seconds',0) for e in log
            if e['kind']=='operation_finish' and e.get('label') in ('independent-verification','independent-fitting-source-check')))
        by_task=dict(task2_preparation=sum(stages[k] for k in ('test_transfer_seconds','test_catalog_seconds','test_screening_seconds','test_duplicate_seconds','final_probe_fit_seconds'))+source_audit_share,
            task3_evaluation_execution=stages['evaluation_reference_seconds']+stages['evaluation_offline_cpu_seconds']+stages['future_native_model_setup_seconds']/2,
            task4_analysis_timing_verification=stages['complete_system_timing_seconds']+stages['independent_verification_seconds']-source_audit_share+stages['future_audit_analysis_cpu_reserve_seconds']+stages['future_native_model_setup_seconds']/2)
        if abs(sum(by_task.values())-(collection+cpu+timing))>1e-8:raise ValueError('Future task allocation must preserve total workload')
        recommended=dict(network_gib=max(61,5*math.ceil(network*1.2/2**30/5)),retained_gib=max(20,5*math.ceil(retained*1.2/2**30/5)),
            collection_hours=math.ceil(max(minimum['collection_seconds'],frozen_collection_gate)*1.25/3600),cpu_hours=math.ceil(minimum['cpu_seconds']*1.25/3600),
            timing_hours=math.ceil(minimum['timing_seconds']*1.25/3600),free_disk_reserve_gib=10)
        cases[str(m)]=dict(m=m,evaluation_sources=400*m,genuine_evaluation_forwards=6400*m,stages=stages,
            task_allocation_seconds=by_task,source_certification_scope='Technical screening projection uses complete measured certify-val wall, including authentication, failed attempts and certificate work. Historical source-audit allowance is allocated to Task 2; remaining independent verification to Task 4. No cost is removed or counted twice.',
            minimum_projected_cumulative=minimum,recommended_cumulative= recommended,
            unchanged_full_study_collection_gate_seconds=frozen_collection_gate,
            expected_future_stage_wall_seconds=collection+cpu+timing,unfinished_task1_reference_projection_seconds=still_fitting,
            unfinished_task1_cpu_reserve_seconds=unfinished_cpu,
            projected_recorded_total_machine_wall_seconds=current['collection_seconds']+current['cpu_seconds']+current['timing_seconds']+still_fitting+unfinished_cpu+collection+cpu+timing,
            work_basis=work)
    result=dict(selected_m=selected,scope='Proposal only; caps unchanged. Endpoint scenarios are not selected evaluation sizes.',
        current=current,current_snapshot_scope='Taken during the proposal CPU operation; that receipt closes after writing. The final handoff resource snapshot includes completed post-run work.',reference_seconds_per_source=rate,ordinary_new_sources_measured=len(observed),
        observed_ordinary_reference_distribution=dict(n=len(observed),mean_seconds=statistics.mean(observed),sample_sd_seconds=statistics.stdev(observed),
            quantiles_seconds=dict(zip(('minimum','p05','p25','median','p75','p95','maximum'),np.quantile(observed,[0,.05,.25,.5,.75,.95,1]).tolist()))) if observed else None,
        measured_offline_and_authentication_cpu_seconds_per_source=per_source_cpu,
        measured_full_oof_verification_seconds=oof_audits,fitting_process_memory=fitting_memory,cuda_memory=cuda_memory,
        measured_complete_fold_cpu_seconds=fold_receipts,
        test_archive_bytes=bytes_remaining,source_receipts_sha256=sha(OUT/'normal_reference_throughput_diagnosis.json'),cases=cases,
        assumptions=['Transfer and screening rates may change across test archives.',
            'Final probe time is extrapolated from genuine fixed folds when available; otherwise 1800 seconds is unmeasured planning reserve.',
            'Timing retains the existing conservative full-system smoke gate; no final timing experiment was run.',
            'Future analysis/audit reserve is an unmeasured 1800 CPU seconds.',
            'Retained original size uses measured fitting bytes with 1.3 inflation; scratch peak still needs future-stage validation.',
            'Unknown reuploads, test membership/technical failures and rank-preserving revisits may block the fixed population.',
            'Reference profile has eight sources; subsequent genuine fitting measurements are reported separately.'],
        activated=False)
    save(OUT/'task1_resource_proposal.json',result)
    lines=['# E27 Task 1 resource proposal','',result['scope'],'',
        f"Selected m: {selected if selected is not None else ('infeasible in 8..20 under the frozen rule' if pp else 'unknown; genuine complete OOF not available')}. No evaluation data were acquired or executed for this proposal.",'',
        f"Current cumulative collection / CPU / timing: {current['collection_seconds']:.3f} / {current['cpu_seconds']:.3f} / {current['timing_seconds']:.3f} seconds.",
        f"Network charged: {current['network_cap_charged_bytes']:,} bytes. Additional retained: {current['retained_bytes']:,} bytes. All prior costs remain charged.",'',
        '## Scenario projections','',
        '| m | Evaluation sources | Future Tasks 2–4 wall hours | Unfinished Task 1 reference hours | Proposed cumulative network / retained GiB | Collection / CPU / timing hours |',
        '|---:|---:|---:|---:|---|---|']
    for m,c in cases.items():
        r=c['recommended_cumulative'];lines.append(f"| {m} | {c['evaluation_sources']} | {c['expected_future_stage_wall_seconds']/3600:.2f} | {c['unfinished_task1_reference_projection_seconds']/3600:.2f} | {r['network_gib']} / {r['retained_gib']} | {r['collection_hours']} / {r['cpu_hours']} / {r['timing_hours']} |")
    for m,c in cases.items():
        lines+=['',f'### m={m} workload','', '| Operation | Additional projected hours |','|---|---:|']
        for name,value in c['stages'].items():lines.append(f"| {name.removesuffix('_seconds').replace('_',' ')} | {value/3600:.3f} |")
        lines+=['','| Future task | Additional projected hours |','|---|---:|']
        for name,value in c['task_allocation_seconds'].items():lines.append(f"| {name.replace('_',' ')} | {value/3600:.3f} |")
        lines+=['',c['source_certification_scope'],'The split of setup time equally between Tasks 3 and 4 is a planning allocation, not a future measurement.']
        minimum=c['minimum_projected_cumulative']
        lines+=['',f"Minimum projected cumulative collection / CPU / timing: {minimum['collection_seconds']/3600:.3f} / {minimum['cpu_seconds']/3600:.3f} / {minimum['timing_seconds']/3600:.3f} hours. These are conditional estimates, not guaranteed minimum execution times.",
            f"Minimum projected network / retained: {minimum['network_bytes']/2**30:.3f} / {minimum['retained_bytes']/2**30:.3f} GiB.",
            f"The unchanged integrated gate still uses smoke timing for future references and requires a projected cumulative collection floor of {c['unchanged_full_study_collection_gate_seconds']/3600:.3f} hours. The recommendation covers the greater of this operational gate and the ordinary-reference forecast, plus headroom. Task 1 does not weaken or bypass that future gate."]
    lines+=['','## Measurements and uncertainties','',
        f"Ordinary eight-source mean: {profile['total_seconds']/8:.6f} s/source. Additional genuine ordinary sources now measured: {len(observed)}. Projection rate: {rate:.6f} s/source; reference safety factor stays 1.5.",
        f"Measured new ordinary-reference timing distribution: {json.dumps(result['observed_ordinary_reference_distribution'],sort_keys=True)}. Includes actual decode, inference, durable journaling and collection overhead; cached replay is excluded.",
        f"Authenticated remaining test archive route: {bytes_remaining:,} response-body bytes. Transfer factor 1.2, screening/catalog factor 1.5, original/feature retention factor 1.3. Recommended cumulative limits add 25% time and 20% byte headroom over these already inflated estimates.",
        'Keep 10 GiB free disk plus the largest compressed archive (about 1.51 GiB), decoded working buffers, and the projected retained increment. No paid compute or environment upgrade is proposed.','']
    lines += [f"Measured source-level offline processing and authentication: {per_source_cpu:.6f} CPU-budget seconds/source. These costs are projected separately for every evaluation source.",
        f"Fitting-run process memory receipt: {json.dumps(fitting_memory,sort_keys=True)}. CUDA peak allocated/reserved receipts: {json.dumps(cuda_memory,sort_keys=True)}.",
        'These memory peaks describe their recorded process scopes, not complete-system RAM. Future all-source data assembly may scale with 400m; reserve and recheck scratch and available RAM before that operation.','']
    lines.extend('- '+s for s in result['assumptions'])
    lines+=['','## Four working days','',
        'Using eight hours/day as a planning convention gives 32 hours. Recorded prior collection/CPU/timing are included in the totals below, along with unfinished Task 1 reserves and future work. These phase-wall sums are a conservative serial machine-work accounting convention, not a measurement of human working hours. Historical unknown preparation time, review, transfer stalls, safe-session pauses and input revisits remain additional uncertainty.',
        *[f"For m={m}, recorded-plus-projected total machine work is {c['projected_recorded_total_machine_wall_seconds']/3600:.2f} hours; {'within 32 hours before unknown costs and human review, so four days remains conditional' if c['projected_recorded_total_machine_wall_seconds']<=32*3600 else 'above 32 hours before unknown costs and human review, so four eight-hour working days is not supported by this projection'}." for m,c in cases.items()],
        '', '## Authorization','',
        'This document does not activate any cap amendment. Task 2 must wait for explicit user authorization of revised cumulative limits, completion and verification of Task 1, and a fresh full-study resource check. The old integrated plan/run/resume paths must not be used to continue Task 1.','']
    (OUT/'task1_resource_proposal.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    return result


if __name__=='__main__':
    from run_native_head_value_confirmation import run_lock
    from native_head_confirmation_state import metered
    with run_lock(),metered('cpu','task1-future-resource-proposal'):
        result=proposal()
        print(json.dumps({k:v['recommended_cumulative'] for k,v in result['cases'].items()},indent=2))
