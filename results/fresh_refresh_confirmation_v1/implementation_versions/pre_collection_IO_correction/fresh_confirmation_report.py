"""Frozen-cohort confirmation figures and candid interval-based findings."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from budget_aware_refresh_report import LABELS
from fresh_confirmation_data import strict_json
from research_utils import now


def create_report(output:Path,protocol:dict,per,trajectories,traces,aggregate:dict,intervals:dict,classes,frozen_run:Path) -> None:
    metrics={(r['budget'],r['policy']):r for r in aggregate['policy_metrics']}
    ci={(r['budget'],r['candidate'],r['reference']):r for r in intervals['comparisons']}
    primary=ci[4,'value_age_motion_budget','periodic']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors={policy:plt.get_cmap('tab10')(i) for i,policy in enumerate(protocol['policies'])}
    fig,ax=plt.subplots(figsize=(12,7))
    for policy in protocol['policies']:
        ax.plot(protocol['budgets'],[metrics[B,policy]['mean_output_error'] for B in protocol['budgets']],marker='o',color=colors[policy],
            label=LABELS[policy],ls='--' if policy in ['oracle','privileged_actual_drift','random'] else '-')
    var={r['budget']:r for r in aggregate['random_seed_variability']}
    ax.fill_between(protocol['budgets'],[var[B]['min_seed_mean'] for B in protocol['budgets']],
        [var[B]['max_seed_mean'] for B in protocol['budgets']],alpha=.1,color=colors['random'],label='Random: ten seed-level mean range')
    ax.scatter([16],[metrics[16,'dense']['mean_output_error']],marker='*',s=140,color='black',label='Dense,16')
    ax.set(xticks=[2,4,8,16],xlabel='Simulated refresh count',ylabel='Mean output-feature cosine error',title='Fresh confirmation:100 videos, equal video weight; models/settings frozen')
    ax.grid(alpha=.2); ax.legend(ncol=2,fontsize=8); fig.tight_layout(); fig.savefig(output/'feature_error_vs_budget.png',dpi=170); plt.close(fig)
    b4=per[(per.budget==4)&(per.policy=='value_age_motion_budget')].sort_values(['absolute_improvement_vs_periodic','source_video_id'])
    fig,ax=plt.subplots(figsize=(16,6))
    delta=b4.absolute_improvement_vs_periodic.to_numpy()
    ax.bar(range(100),delta,color=np.where(delta>0,'#238c36','#bd3434')); ax.axhline(0,color='black',lw=.7)
    ax.set_xticks(range(100),b4.source_video_id.tolist(),rotation=90,fontsize=5)
    ax.set(xlabel='Fresh source video, ordered by paired gain',ylabel='Periodic error minus motion-value-controller error',title='B=4 primary comparison: each bar is one new source video')
    ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(output/'B4_per_video_improvement.png',dpi=170); plt.close(fig)
    ordered=sorted(intervals['comparisons'],key=lambda r:(r['budget'],r['candidate'],r['reference']))
    fig,ax=plt.subplots(figsize=(12,8))
    for i,r in enumerate(ordered):
        color='#c43a2f' if r['comparison_type']=='primary' else '#2878b5'
        ax.errorbar(r['mean_paired_absolute_improvement'],i,xerr=[[r['mean_paired_absolute_improvement']-r['CI95_lower']],[r['CI95_upper']-r['mean_paired_absolute_improvement']]],fmt='o',color=color,capsize=3)
    ax.axvline(0,color='black',lw=.8)
    ax.set_yticks(range(len(ordered)),[f"{'PRIMARY ' if r['comparison_type']=='primary' else ''}B{r['budget']}: {LABELS[r['candidate']]} vs {LABELS[r['reference']]}" for r in ordered],fontsize=8)
    ax.set(xlabel='Mean paired feature-error improvement (95% percentile bootstrap interval)',title='100 whole-source-video pairs; fixed models; secondary intervals marginal/unadjusted')
    ax.grid(axis='x',alpha=.2); fig.tight_layout(); fig.savefig(output/'paired_improvements_intervals.png',dpi=170); plt.close(fig)
    best=b4.sort_values(['absolute_improvement_vs_periodic','source_video_id'],ascending=[False,True]).iloc[0]
    worst=b4.sort_values(['absolute_improvement_vs_periodic','source_video_id']).iloc[0]
    examples=[]; fig,axes=plt.subplots(2,3,figsize=(17,8),sharex=True)
    for i,(kind,row) in enumerate([('largest_gain',best),('largest_loss',worst)]):
        examples.append({'kind':kind,'source_video_id':row.source_video_id,'video_id':row.video_id,'label':row.label,'B4_absolute_gain':float(row.absolute_improvement_vs_periodic),
            'rule':'predeclared largest gain/loss versus periodic, source-ID tie break; descriptive, not selected for tuning'})
        for j,policy in enumerate(['periodic','paced_age_plus_motion','value_age_motion_budget']):
            g=traces[(traces.video_id==row.video_id)&(traces.policy==policy)&(traces.budget==4)].sort_values('candidate_ordinal')
            axes[i,j].plot(g.candidate_ordinal,g.output_feature_error,label='Output error',marker='o',markersize=3,color=colors[policy])
            axes[i,j].plot(g.candidate_ordinal,g.evaluator_true_pre_decision_drift,label='Pre-decision drift (privileged)',ls='--',color='gray')
            refresh=g[g.refresh]; axes[i,j].scatter(refresh.candidate_ordinal,refresh.output_feature_error,color='black',marker='*',s=80,label='Simulated refresh')
            axes[i,j].set_title(f'{kind}: {row.source_video_id}\n{LABELS[policy]}',fontsize=9); axes[i,j].grid(alpha=.2); axes[i,j].legend(fontsize=6)
            if j==0: axes[i,j].set_ylabel('Feature cosine error')
            if i==1: axes[i,j].set_xlabel('Candidate ordinal (offline)')
    fig.tight_layout(); fig.savefig(output/'refresh_timelines.png',dpi=160); plt.close(fig)
    strict_json(output/'example_selection.json',{'examples':examples})
    acquisition=json.loads((output/'acquisition.json').read_text()); exclusion=json.loads((output/'exclusions.json').read_text())
    collection=json.loads((output/'collection_metadata.json').read_text())
    old=json.loads((frozen_run/'development_aggregate_metrics.json').read_text())
    old_metrics={(r['budget'],r['policy']):r for r in old['policy_metrics']}
    fmt=lambda x,p=6:'undefined' if x is None else f'{x:.{p}f}'
    lines=['# Fresh-video confirmation of frozen budget-aware refresh','',f"Written {now()} (UTC). Primary result: **{primary['interpretation']}**.",'',
        '## Cohort, reference collection and frozen methods','',
        f"Exactly100 new source videos:five per frozen class across twenty existing classes. Frozen prior exclusions contained210 sources and210 hashes; this run conservatively adds{len(exclusion['additional_sources'])} prior inspected/collected source IDs and{len(exclusion['additional_hashes'])} hashes. No source/hash overlap or duplicates remain. Selection and replacements used only predetermined archive/candidate rank and eligibility; no policy performance was examined before the complete manifest froze.",
        f"Verified cached compressed bytes {acquisition['cached_archive_bytes']:,}; new reserved compressed bytes {acquisition['newly_reserved_archive_bytes']:,} ({acquisition['newly_reserved_archive_bytes']/1024**3:.3f} GiB), under12GiB. Observed response body transfer bytes {acquisition['transfer_response_body_bytes']:,}; headers/metadata and retransmissions can differ from reservation. Archives/encoded videos remain local, unmodified.",
        f"Retained references:1,600. Actual ResNet18 forward entries:{collection['actual_forward_calls_entered']}; extra smoke/retry calls:{collection['extra_forward_calls']}. Collection uses the original verified weights, preprocessing, CUDA float32 batch1, eval/inference mode, numerical settings, software decoder and per-frame CPU motion descriptors. Missing vectors do not disqualify clips. Reference collection is actual encoder work; policy replay performs **zero encoder calls**.",
        '', 'Frozen coefficient/scaler and selected-setting digests matched before acquisition and after evaluation. No regressors/scalers were refitted, no grids recalculated, no thresholds/lambdas/features changed. Original forty-video compatibility reproduced all1120 policy/budget metrics and2200 schedules. Replay uses all nine sparse methods at2,4,8 and dense16; all ten random seeds are averaged within each source before aggregating sources.',
        '', '## Primary and predefined secondary comparisons','',
        'Positive differences favor the candidate. Intervals use the same10,000x100 resampling matrix of whole source videos, PCG64 seed20261014 and linear percentiles. Secondary intervals are marginal and unadjusted.','',
        '| B | Candidate | Reference | Mean paired gain |95% interval|Improved/tied/worsened|Mean per-video gain (%)|Status|',
        '|---:|---|---|---:|---|---|---:|---|']
    for r in ordered:
        lines.append(f"|{r['budget']}|{LABELS[r['candidate']]}|{LABELS[r['reference']]}|{r['mean_paired_absolute_improvement']:+.6f}|[{r['CI95_lower']:+.6f},{r['CI95_upper']:+.6f}]|{r['improved']}/{r['tied']}/{r['worsened']}|{fmt(r['mean_per_video_percentage_improvement'],2)}|{r['interpretation']} {'(PRIMARY)' if r['comparison_type']=='primary' else '(secondary)'}|")
    lines+=['', '## Fresh-test feature error','', '|Policy|B2 mean|B4 mean|B4 median per-video|B8 mean|','|---|---:|---:|---:|---:|']
    for policy in protocol['policies']:
        lines.append(f"|{LABELS[policy]}|{metrics[2,policy]['mean_output_error']:.6f}|{metrics[4,policy]['mean_output_error']:.6f}|{metrics[4,policy]['median_per_video_output_error']:.6f}|{metrics[8,policy]['mean_output_error']:.6f}|")
    lines+=['',f"Dense B16 mean error:{metrics[16,'dense']['mean_output_error']:.3e}. Actual drift threshold and full-future oracle are privileged diagnostics, not efficient deployable methods.",
        '', '## Development results, shown separately','', '|Policy|Development B4 mean|Fresh B4 mean|','|---|---:|---:|']
    for policy in ['periodic','paced_age_plus_motion','value_age_budget','value_age_motion_budget']:
        lines.append(f"|{LABELS[policy]}|{old_metrics[4,policy]['mean_output_error']:.6f}|{metrics[4,policy]['mean_output_error']:.6f}|")
    lines+=['', 'Development and fresh results are not pooled. A positive development point estimate does not override an inconclusive or negative primary fresh interval.','', '## Does motion add value, and does the controller beat pacing?','']
    for B in protocol['budgets']:
        for reference in ['value_age_budget','paced_age_plus_motion']:
            r=ci[B,'value_age_motion_budget',reference]
            lines.append(f"B{B} motion value versus {LABELS[reference]}: {r['interpretation']}; paired gain{r['mean_paired_absolute_improvement']:+.6f}, interval[{r['CI95_lower']:+.6f},{r['CI95_upper']:+.6f}].")
    concentration=aggregate['concentration']
    lines+=['', '## Distribution, oracle headroom and schedules','',
        f"Primary improvements occur in{primary['improved']}/100 videos, ties{primary['tied']}, losses{primary['worsened']}; {concentration['primary_classes_improved_in_mean']}/20 classes improve descriptively. Top-two-positive-gains share of positive net summed gain:{fmt(concentration['top_two_positive_gains_percent_of_positive_net'],2)}%; {concentration['reason']}. No class significance claims are made from five videos/class.",
        '', 'Percentages computed from aggregate means are saved separately from mean per-video percentages. Oracle-gap ratios can be negative despite a positive absolute average gain; small denominators amplify losses. Undefined denominators remain null/blank with reasons.','',
        '|B|Mean oracle headroom|Motion value mean oracle-gap closure (%)|Undefined gaps|Motion value mean forced refreshes|Mean last refresh|Mean outputs after exhaustion|',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for B in protocol['budgets']:
        r=metrics[B,'value_age_motion_budget']
        lines.append(f"|{B}|{r['mean_oracle_headroom']:.6f}|{fmt(100*r['mean_defined_oracle_gap_closure'] if r['mean_defined_oracle_gap_closure'] is not None else None,2)}|{r['undefined_oracle_gaps']}|{r['mean_forced_refreshes']:.2f}|{r['mean_last_refresh_position']:.2f}|{r['mean_outputs_after_budget_exhausted']:.2f}|")
    lines+=['', 'Random-seed variability is separate descriptive variability; seeds are not independently bootstrapped videos. All bootstrap comparisons retain whole-video pairing. Action regrets and saved traces reproduce mean-error excess above the exact oracle. Low forced counts alone do not imply good fidelity.','', '## Interpretation and next action','']
    motion_ci=ci[4,'value_age_motion_budget','value_age_budget']; age_ci=ci[4,'value_age_budget','periodic']
    if primary['CI95_lower']>0:
        conclusion='The primary frozen comparison provides evidence of lower appearance-feature error on these new videos. A separate recognition-quality and actual adaptive-latency experiment is justified; neither is measured here.'
    elif primary['CI95_upper']<0:
        conclusion='The primary frozen controller has evidence of worse appearance-feature error. Preserve this failure for these fixed methods. Do not proceed as though the motion controller has been confirmed.'
    else:
        conclusion='The primary confirmation is inconclusive:the interval includes zero. This is neither confirmed improvement nor equivalence. Do not add more videos to this cohort until the interval becomes favorable.'
    lines.append(conclusion)
    if motion_ci['CI95_lower']<=0:
        lines.append('Motion has not demonstrated an improvement beyond the corresponding age/budget controller in the primary-budget comparison; its extra complexity and potential processing cost are not justified by this confirmation.')
    if age_ci['CI95_lower']>0 and primary['CI95_lower']<=0:
        lines.append('The without-motion controller has a favorable predefined secondary interval versus periodic, but that marginal/unadjusted result does not replace the original primary comparison. If pursued, evaluate that simpler rule in a separate clearly specified recognition/latency stage with this qualification retained.')
    lines+=['', 'Secondary outcomes across budgets remain visible above; favorable isolated comparisons do not establish broad superiority. No model redesign or retuning was performed.','',
        '## Limits','',
        'Appearance fidelity is not recognition accuracy. Reference-collection elapsed time is not adaptive inference latency. Replay refresh counts are simulated; no measured adaptive speedup is established. Full-clip candidates and future-reference codec vectors keep this offline. Test videos are new sources from existing classes, not unseen classes. The intervals condition on fixed fitted models and the sampled class population, exclude training uncertainty, and resample whole sources rather than candidates, frames, states or random seeds.','',
        '## Figures and reproduction','', '![Feature error versus simulated budget](feature_error_vs_budget.png)','', '![Primary per-video improvements](B4_per_video_improvement.png)','',
        '![Paired mean gains and intervals](paired_improvements_intervals.png)','', '![Predetermined example timelines](refresh_timelines.png)','',
        '```powershell', '& .\\.venv\\Scripts\\python.exe .\\run_fresh_refresh_confirmation.py --frozen-run results/budget_aware_refresh_v2 --output results/fresh_refresh_confirmation_v1 --device cuda --max-download-gib 12 --download-workers 8',
        '```', '', 'For an incomplete run, append `--resume` with exactly the same configuration. Completed runs are audited/reproduced without acquiring another cohort. See verification.json, frozen identities, execution/sampling plans, acquisition/retry records and bootstrap_intervals.json.']
    (output/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
