"""Append observed resource use and conditional research decision to findings.

Reads retained outputs; does not train, extract or change endpoint values.
"""
from pathlib import Path
from datetime import datetime
import gzip
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    events=[json.loads(line) for line in (OUT/'exposure_resource_ledger.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    resource=read(OUT/'collection_resources.json') if (OUT/'collection_resources.json').exists() else None
    ledger_forward_starts=sum(e['kind']=='HMDB_forward_start' for e in events)
    ledger_forward_completions=sum(e['kind']=='HMDB_forward_complete' for e in events)
    disk=sum(p.stat().st_size for directory in [OUT,ROOT/'data/source/continuation_value_gap_v1'] if directory.exists() for p in directory.rglob('*') if p.is_file())
    result=dict(UCF_analysis_seconds=next(e['seconds'] for e in events if e['kind']=='UCF_complete'),
        HMDB_collection_seconds=None if resource is None else resource['cumulative_seconds'],
        HMDB_forward_starts=ledger_forward_starts,HMDB_forward_completions=ledger_forward_completions,
        failed_or_incomplete_forwards=ledger_forward_starts-ledger_forward_completions,
        feature_cache_reuses=0 if resource is None else resource['reuse'],additional_disk_bytes=disk,
        new_data_or_checkpoint_downloads=0,environment_modifications=0,new_UCF_encoder_calls=0,
        collection_wall_cap_seconds=28800,additional_disk_cap_bytes=20*2**30,
        scope='Diagnostic collection/analysis resource use, not adaptive inference latency')
    if resource and resource['attempts']:
        complete=[a for a in resource['attempts'] if a['status']=='complete']
        first=min(datetime.fromisoformat(a['at_utc']) for a in resource['attempts'])
        last=max(datetime.fromisoformat(a['at_utc']).timestamp()+a['seconds'] for a in complete)
        result['conservative_collection_envelope_seconds']=last-first.timestamp()
        result['envelope_scope']='All wall time from first collection attempt start to last completion; includes restarts, provenance waits and head fitting, hence conservatively bounds collection and its bookkeeping overhead.'
        assert result['conservative_collection_envelope_seconds']<=28800, 'Collection envelope exceeds cap: reconcile exact active collection wall before claiming cap compliance'
    if (OUT/'process_memory_samples.jsonl').exists():
        samples=[json.loads(l) for l in (OUT/'process_memory_samples.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        result['observed_adapter_peak_working_set_bytes']=max(r['peak_working_set_bytes'] for r in samples)
        result['peak_memory_scope']='OS high-water mark sampled during live HMDB adapter; includes provenance/collection/head. Not an isolated CPU-analysis peak.'
    if (OUT/'hmdb_smoke_verification.json').exists():
        smoke=read(OUT/'hmdb_smoke_verification.json')
        result['smoke_peak_cuda_allocated_bytes']=smoke['peak_cuda_allocated_bytes'];result['smoke_peak_cuda_reserved_bytes']=smoke['peak_cuda_reserved_bytes']
    (OUT/'resource_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['','## Observed resources and verification','',
        f"UCF CPU endpoint calculation: {result['UCF_analysis_seconds']:.2f} s; zero new encoder calls, decodes or fits. "+
        f"HMDB collection: {result['HMDB_collection_seconds']} s including per-clip reads, decode, preprocessing, inference, synchronization and writes. "+
        f"Forward starts/completions: {ledger_forward_starts:,}/{ledger_forward_completions:,}. Additional retained disk: {disk/2**20:.2f} MiB. "+
        'The resource ledger includes the successful first smoke clip before the timestamp bookkeeping failure and its hash-verified reuse; that mechanical failure and original code/seal are retained. No failed model forward is hidden.',
        '', 'The existing `.venv_temporal` executable uses the already configured shared `.venv` Torch/torchvision libraries; actual paths and versions are in `actual_runtime_paths.json`. '+
        'No environment was replaced or upgraded. CPU-analysis peak RAM was not separately measured. Smoke GPU and sampled adapter peak RAM are reported with their measurement scope in `resource_summary.json`.']
    if 'conservative_collection_envelope_seconds' in result:
        envelope=result['conservative_collection_envelope_seconds']
        lines+=['',f"The per-clip timer sum is distinct from the conservative elapsed collection envelope: {envelope:.2f} s ({envelope/3600:.4f} h). "+
            'The envelope includes directory/hash bookkeeping outside clip timers, restart gaps, and head fitting, so it bounds the complete feature-collection wall time conservatively. '+
            'Cap compliance is also checked against this broader window; neither quantity is deployed adaptive-inference latency.']
    if (OUT/'temporal_support.json').exists():
        temporal=read(OUT/'temporal_support.json');u=temporal['UCF'];h=temporal['HMDB_EVAL']
        lines+=['','Input-only physical support: UCF snippet-span median '+f"{u['snippet_span_seconds'][4]:.4g} s versus HMDB EVAL {h['snippet_span_seconds'][4]:.4g} s; "+
            f"HMDB EVAL 5th–95th percentile snippet spans {h['snippet_span_seconds'][2]:.4g}–{h['snippet_span_seconds'][6]:.4g} s. "+
            f"Median candidate gaps are {u['candidate_gap_seconds'][4]:.4g} s (UCF) and {h['candidate_gap_seconds'][4]:.4g} s (HMDB). "+
            'These are unweighted metadata distributions over all included clips/snippets, not independent outcome replicates. Complete probabilities/denominators are in `temporal_support.json`.']
    p=OUT/'independent_verification.json'
    if p.exists():
        v=read(p);lines+=['',f"Independent artifact verification passed: {v['passed']}; historical result blobs preserved: {v.get('historical_result_files_unchanged')}. "+
            'Suffix enumeration, scalar state reconstruction, E22 overlap and all455 scalar raw-feature checks on fixed hash-selected real clips are separate from synthetic fixture validation.']
        for name in ['ucf','hmdb']:
            if name in v:
                a=v[name];lines += [f"- {name.upper()}: {a['states']:,} real state values; max scalar state discrepancy {a['scalar_state_max_error']:.3g} nats; "+
                    f"raw-feature replay on {a['raw_feature_clips']} clips/all455 schedules: max CE discrepancy {a['raw_feature_all455_CE_max_error']:.3g} nats."]
    for name in ['ucf','hmdb']:
        p=OUT/name/'estimates.json'
        if not p.exists():continue
        e=read(p);s=pd.read_csv(OUT/name/'per_state.csv.gz',float_precision='round_trip');c=pd.read_csv(OUT/name/'per_clip.csv',float_precision='round_trip')
        w=np.repeat(c.weight.to_numpy()/12,12);base=np.repeat(c.Periodic_CE.to_numpy(),12)
        sensitivity=[]
        for tol in [0,1e-12,1e-10,1e-8]:
            a=np.where(abs(s.D_star.to_numpy())<=tol,s.original_action.to_numpy(),(s.D_star.to_numpy()>0).astype(int))
            q=np.where(a,s.Q_P_R,s.Q_P_H)
            sensitivity.append(dict(tolerance=tol,tie_states=int((abs(s.D_star)<=tol).sum()),G_star=float(w@(base-q))))
        # HMDB shares the prospective sensitivities frozen before UCF endpoints.
        if not (OUT/name/'tie_sensitivity.json').exists():(OUT/name/'tie_sensitivity.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
        lines+=['',f'## {name.upper()} diagnostics and meaning','']
        identity_residual=float(np.max(abs((s.D_P-s.D_star)-(s.gap_H-s.gap_R))))
        min_gap=float(min(s.gap_H.min(),s.gap_R.min()))
        lines += [f"Accounting check: r_a=Q_P(a)-Q_star(a) has minimum {min_gap:.3g} nats; "+
            f"max residual in D_P-D_star=r_HOLD-r_REFRESH is {identity_residual:.3g} nats. "+
            'Different action-specific continuation gaps explain how an optimistic branch preference can change under a fixed tail. This is an accounting verification, not a new theorem. '+
            'Exact branch cardinalities are in `branch_sizes.csv`; their differences are genuine feasible-space differences, not Monte Carlo or noisy-minimum bias. Uniform branch means average individual schedule losses.','']
        for tail in ['P','U']:
            for stratum in ['all','material']:
                d=e[f'disagreement_{tail}_{stratum}']
                lines += [f"- {tail} tail, {stratum}: {d['disagree_states']}/{d['states']} state disagreements; hierarchical fraction {d['fraction']:.4%}; "+
                          f"conditional paired wrong-branch loss {d['paired_regret_conditional']:.7g} nats; whole-cohort regret contribution {d['paired_regret_whole']:.7g} nats."]
        logo=pd.read_csv(OUT/name/'leave_one_family_out.csv')
        lines+=['',f"Leave-one-global-family-out G_star range {logo['mean'].min():.8g} to {logo['mean'].max():.8g} nats; "+
                f"{logo['mean'].isna().sum()} omissions would remove a fixed class and remain undefined. This is influence analysis, not a replacement confidence interval.",
                '',f"Numerical ties: {int(s.numerical_tie.sum())}/{len(s)} states. Sensitivity G_star range "+
                f"{min(a['G_star'] for a in sensitivity):.10g} to {max(a['G_star'] for a in sensitivity):.10g} nats. "+
                f"Primary tie-choice gain range: {e['estimates']['tie_gain_low']['mean']:.10g} to {e['estimates']['tie_gain_high']['mean']:.10g} nats.",
                '',f"Matched single-intervention CE is {e['estimates']['Periodic_CE']['mean']-e['estimates']['G_matched']['mean']:.9g}; "+
                f"the 455-schedule task oracle CE is {e['estimates']['Task_oracle_CE']['mean']:.9g}. These have different intervention freedom. "+
                'Z_pool gain uses the original centered-logit pooled-squared-error teacher, lexicographic first within1e-12, under exactly the same prefix/tail.']
    h=read(OUT/'hmdb/estimates.json') if (OUT/'hmdb/estimates.json').exists() else None
    lines+=['','## Precise research decision','']
    if h:
        p=h['primary'];op=h['estimates']['G_matched']['mean'];reg=h['estimates']['R_star']['mean'];ratio=reg/op if op>0 else None
        population=read(OUT/'hmdb_population_diagnostics.json')['roles'];train=population['HEAD_TRAIN'];ev=population['EVAL']
        lines += [f"Recognition context: Dense top1={h['estimates']['Dense_accuracy']['mean']:.4%}. HEAD_TRAIN raw per-class counts range "+
            f"{min(train['per_class_clips'].values())}–{max(train['per_class_clips'].values())}, with only {train['min_families_per_class']} families in the smallest training class. "+
            'Class/family weighting addresses this declared estimand; it does not replace missing independent sources or guarantee a strong recognition head. '+
            f"EVAL uses {ev['global_components']} global components / {ev['sum_class_specific_families']} class-family cells, {ev['cross_class_components']} cross-class components, "+
            f"maximum included component size {ev['maximum_component_size']}, effective weight concentration {ev['effective_components']:.3f}. No recognition-based exclusions or head improvements were made.",'']
        lines+=[f"HMDB primary classification under the prospective .01-nat margin: **{p.get('outcome','unresolved')}**. "+
                f"G_star={p['mean']:.9g}, 95% CI={p['interval']}, df={p['df']:.6g}. Matched opportunity={op:.9g}, regret={reg:.9g} nats "+
                f"({ratio:.2%} of matched opportunity)."]
        if p.get('outcome')=='material harm' and op>=.01:
            lines+=['The tested periodic-repair tail turns oracle-preferred actions into material mean harm despite meaningful matched-tail opportunity. '+
                    'This supports continuation mismatch in this exact setting. Next: a new prospective study of whether permitted causal observations predict matched-tail advantage, with separately isolated training and evaluation. No controller retraining is justified by this diagnostic alone.']
        elif p['mean']>0:
            lines+=['The oracle-selected intervention improves mean CE in this validation population; continuation mismatch produces measurable regret but does not establish an explanation based on broad oracle-action harm. '+
                    'Keep the claim about continuation dependence and the measured lost opportunity. Next: a separately preregistered causal-observation predictability study targeting the executable-tail advantage, only if its opportunity magnitude warrants the acquisition cost. Do not retune the present head or controllers on these test outcomes.']
        elif op<.01:
            lines+=['The matched single-intervention family has little mean opportunity under the prospective practical margin. Next: retain this narrow local-opportunity finding and stop controller development on this diagnostic; adaptive scheduling in general remains unresolved.']
        else:
            lines+=['The primary result remains unresolved at the prospective practical margin. Next: preserve the frozen adverse/uncertain result and specify a genuinely independent replication with an adequate precision plan before another recognition-head or controller experiment.']
    else:
        gate=read(OUT/'hmdb_gate.json');lines += [f"HMDB was not evaluated ({gate['status']}): {gate.get('blocker','see gate artifact')}. "+
            'The independent-evidence gap remains open. UCF supports only exposed-development continuation dependence: its positive mean gain does not support material mean harm, and its full-cohort inferential uncertainty is unavailable. '+
            'Next: resolve the documented technical/resource gate under an explicit prospective amendment, preserving the frozen population and exposure history; do not select a smaller or easier evaluation cohort.']
    lines += ['','## Figures','',
        'All plots use the frozen class/family/clip weighting and include adverse outcomes. The full UCF cohort has no valid CR2 interval; no error bar is substituted.','',
        '![Oracle versus periodic-repair advantage](figure_1_advantages.png)','',
        '[Figure 1 PDF](figure_1_advantages.pdf): all reset states; marker area is hierarchical state weight; full symmetric equal x/y scales within each dataset.','',
        '![Single-intervention gains and regret](figure_2_endpoints.png)','',
        '[Figure 2 PDF](figure_2_endpoints.pdf): Periodic CE baseline for periodic repair; V_U baseline for the uniform tail. HMDB intervals, when available, are approximate and secondary intervals are unadjusted.','',
        '![Signed clip gain and regret distributions](figure_3_distributions.png)','',
        '[Figure 3 PDF](figure_3_distributions.pdf): complete weighted clip CDFs, including adverse tails; dotted thresholds mark the prospective practical margin.']
    text=(OUT/'findings.md').read_text(encoding='utf-8');marker='\n## Observed resources and verification\n'
    text=text.replace('unresolved (invalid/unavailable uncertainty)','No inferential label: uncertainty unavailable')
    gate_text=json.dumps(read(OUT/'hmdb_gate.json'),ensure_ascii=False,indent=2)
    text=text.replace(gate_text,'```json\n'+gate_text+'\n```')
    if marker in text:text=text.split(marker)[0]
    (OUT/'findings.md').write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
    verified=read(OUT/'independent_verification.json') if (OUT/'independent_verification.json').exists() else {}
    inputs=read(OUT/'supplemental_input_verification.json') if (OUT/'supplemental_input_verification.json').exists() else {}
    gate=read(OUT/'hmdb_gate.json')
    hmdb_execution=gate['status']
    if hmdb_execution=='COMPLETE' and not (verified.get('hmdb',{}).get('passed') and inputs.get('passed') and inputs.get('feature_checks')!='PENDING_COLLECTION'):
        hmdb_execution='VERIFICATION_FAILED'
    execution=dict(UCF=dict(execution='COMPLETE' if verified.get('ucf',{}).get('passed') else 'VERIFICATION_FAILED',
        scientific_outcome='NO_INFERENTIAL_LABEL_UNCERTAINTY_UNAVAILABLE',point_estimate=read(OUT/'ucf/estimates.json')['primary']['mean']),
        HMDB=dict(execution=hmdb_execution,scientific_outcome=h['primary'].get('outcome','NO_INFERENTIAL_LABEL') if h else 'NOT_EVALUATED'))
    (OUT/'execution_summary.json').write_text(json.dumps(execution,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
