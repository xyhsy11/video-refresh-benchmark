"""Reports only measured offline replay quality, preserving negative findings."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import budget_aware_refresh_utils as u
from research_utils import ROOT, now, sha256_file, write_json

LABELS={'periodic':'Periodic','random':'Random (ten seeds)','age_only':'Age drift threshold',
        'age_plus_motion':'Age + motion drift threshold','paced_age_plus_motion':'Drift + budget pacing',
        'value_age_budget':'Action value: age + budget','value_age_motion_budget':'Action value: age + motion + budget',
        'privileged_actual_drift':'Actual drift (privileged)','oracle':'Full-future oracle (privileged)','dense':'Dense (16)'}


def create_report(output:Path,per:pd.DataFrame,trajectories:pd.DataFrame,traces:pd.DataFrame,aggregate:dict,metadata:dict) -> list:
    metrics={(r['budget'],r['policy']):r for r in aggregate['policy_metrics']}
    paired={(r['budget'],r['candidate'],r['reference']):r for r in aggregate['paired_comparisons']}
    def fmt(x,precision=6): return 'undefined' if x is None else f'{x:.{precision}f}'
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors={p:plt.get_cmap('tab10')(i) for i,p in enumerate(u.POLICIES)}
    fig,ax=plt.subplots(figsize=(12,7))
    for policy in u.POLICIES:
        ax.plot([2,4,8],[metrics[B,policy]['mean_output_error'] for B in [2,4,8]],marker='o',
            color=colors[policy],label=LABELS[policy],ls='--' if policy in ['oracle','privileged_actual_drift','random'] else '-')
    variability={r['budget']:r for r in aggregate['random_seed_variability']}
    ax.fill_between([2,4,8],[variability[B]['min_seed_mean_error'] for B in [2,4,8]],
        [variability[B]['max_seed_mean_error'] for B in [2,4,8]],color=colors['random'],alpha=.1,
        label='Random: range of ten seed-level means')
    ax.scatter([16],[metrics[16,'dense']['mean_output_error']],color='black',marker='*',s=150,label=LABELS['dense'])
    ax.set(xticks=[2,4,8,16],xlabel='Simulated refresh count',ylabel='Mean post-decision output-feature cosine error',
           title='40 development videos; equal weight per video; thresholds selected on these videos')
    ax.grid(alpha=.2); ax.legend(ncol=2,fontsize=8); fig.tight_layout()
    fig.savefig(output/'error_vs_refresh_budget.png',dpi=180); plt.close(fig)
    B4=per[(per.budget==4)&(per.policy=='value_age_motion_budget')].sort_values(['absolute_improvement_vs_periodic','source_video_id'])
    order=B4.video_id.tolist(); pivot=per[per.budget==4].pivot(index='video_id',columns='policy',values='mean_output_error').loc[order]
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(15,8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for policy in ['periodic','paced_age_plus_motion','value_age_budget','value_age_motion_budget']:
        ax1.plot(range(40),pivot[policy].to_numpy(),marker='.',label=LABELS[policy],color=colors[policy])
    ax1.set_ylabel('Mean output-feature error'); ax1.legend(fontsize=8); ax1.grid(alpha=.2)
    ax1.set_title('B=4 development comparison, ordered by paired action-value-controller improvement')
    reductions=B4.absolute_improvement_vs_periodic.to_numpy()
    ax2.bar(range(40),reductions,color=['#248f35' if x>u.TOL else '#bf3535' if x<-u.TOL else '#999999' for x in reductions])
    ax2.axhline(0,color='black',lw=.7); ax2.grid(axis='y',alpha=.2)
    ax2.set_ylabel('Periodic error minus\ncontroller error'); ax2.set_xticks(range(40),B4.source_video_id.tolist(),rotation=90,fontsize=7)
    ax2.set_xlabel('Development source video'); fig.tight_layout()
    fig.savefig(output/'per_video_B4_improvement.png',dpi=170); plt.close(fig)
    examples=[]
    largest=B4.sort_values(['absolute_improvement_vs_periodic','source_video_id'],ascending=[False,True]).iloc[0]
    smallest=B4.sort_values(['absolute_improvement_vs_periodic','source_video_id']).iloc[0]
    chosen=[('largest_gain' if largest.absolute_improvement_vs_periodic>u.TOL else 'least_loss',largest),
            ('failure' if smallest.absolute_improvement_vs_periodic<-u.TOL else 'smallest_gain_no_failure_observed',smallest)]
    fig,axes=plt.subplots(2,3,figsize=(17,8),sharex=True)
    for i,(kind,row) in enumerate(chosen):
        examples.append({'kind':kind,'video_id':row.video_id,'source_video_id':row.source_video_id,'label':row.label,
            'absolute_improvement_vs_periodic':float(row.absolute_improvement_vs_periodic),'selection':'predefined B4 largest gain/loss; source-ID tie break; descriptive only'})
        for j,policy in enumerate(['periodic','paced_age_plus_motion','value_age_motion_budget']):
            g=traces[(traces.video_id==row.video_id)&(traces.budget==4)&(traces.policy==policy)].sort_values('candidate_ordinal')
            axes[i,j].plot(g.candidate_ordinal,g.output_feature_error,label='Post-decision output error',marker='o',markersize=3,color=colors[policy])
            axes[i,j].plot(g.candidate_ordinal,g.evaluator_true_pre_decision_drift,label='True pre-decision drift (evaluator)',color='gray',ls='--')
            r=g[g.refresh]; axes[i,j].scatter(r.candidate_ordinal,r.output_feature_error,marker='*',s=90,color='black',label='Actual simulated refresh')
            axes[i,j].set_title(f'{kind}: {row.source_video_id}\n{LABELS[policy]}',fontsize=9)
            axes[i,j].grid(alpha=.2); axes[i,j].legend(fontsize=6)
            if j==0: axes[i,j].set_ylabel('Appearance-feature cosine error')
            if i==1: axes[i,j].set_xlabel('Candidate ordinal (offline)')
    fig.suptitle('B=4 successes/failures: cache follows actual refreshes; evaluator values never enter deployable scorers')
    fig.tight_layout(); fig.savefig(output/'example_timelines.png',dpi=160); plt.close(fig)
    write_json(output/'example_selection.json',{'frozen_rule':'B4 largest absolute gain and largest loss versus periodic; source-ID tie break','examples':examples})
    manifest=json.loads((ROOT/'results/cache_drift_v2/split_manifest.json').read_text())
    pilot=json.loads((ROOT/'results/pilot_2026_10_10/manifest.json').read_text())
    pilot_rows=list(pilot['videos'].values())
    fresh={'status':'frozen prospective protocol; not acquired or evaluated','written_at_utc':now(),
        'videos':100,'classes':sorted({e['label'] for e in manifest['videos']}),'videos_per_class':5,
        'sampling_seed':20261014,'eligibility':['preserve H.264 bytes','5..15 seconds','>=16 decoded frames','finite increasing candidate timestamps','finite reference extraction'],
        'excluded_source_ids':sorted({e['source_video_id'] for e in manifest['videos']}|{e['video_id'] for e in pilot_rows}),
        'excluded_file_SHA256':sorted({e['file_identity']['sha256'] for e in manifest['videos']}|{e['file_identity']['sha256'] for e in pilot_rows}),
        'fixed_models_sha256':sha256_file(output/'fitted_predictors.json'),'fixed_settings_sha256':sha256_file(output/'selected_thresholds.json'),
        'policies':u.POLICIES,'budgets':[2,4,8],'primary_comparison':'B4 value_age_motion_budget vs periodic',
        'additional_predefined_comparisons':['B4 value_age_motion_budget vs pacing','B4 value_age_motion_budget vs value_age_budget','secondary B2/B8 same comparisons'],
        'candidate_and_motion_inputs':'identical sixteen endpoints-including candidates and existing definitions; do not alter after results',
        'uncertainty':{'unit':'whole source videos; paired model results retained together','method':'95% percentile bootstrap of mean paired per-video error difference','resamples':10000,'seed':20261014,'models':'fixed'},
        'rule':'freeze acquisition identities before evaluation; no model/setting/feature retuning after test results; no recognition or latency claims'}
    json.dumps(fresh,allow_nan=False); write_json(output/'fresh_test_protocol.json',fresh)
    lines=['# Budget-aware oracle-action-value refresh: development findings','',
        f"Actual execution started {metadata['started_at_utc']} (UTC); client research date 2026-10-14. This is saved-artifact offline replay.",
        '', '## Data and frozen methods','',
        f"Only the original 120 v2 training sources were used: four training and two development videos per class, seed 20261014, yielding **80 training /40 development videos**. Original validation/test feature arrays, motion tables, predictions and evaluation results were not loaded. Identity manifests and preservation hashes are separate from data access.",
        f"The same eighty sources train both drift baselines (**9,600 pairs**) and six budget-specific action-value regressors (**{metadata['action_value_training_states']:,} optional states**). Scalers use training-only population statistics, alpha=1.0, unpenalized intercept, float64 arithmetic. State counts are equal within each budget across training videos.",
        '', 'An exact DP minimizes remaining SUM of output errors with exactly b refreshes. The signed label `(Q_reuse-Q_refresh)/16` is generated only for reachable states with both actions feasible. Future RGB generates training labels, never deployable inputs. Optional action-value decisions refresh iff predicted advantage >0; no clipping, threshold search or retraining occurs for these models.',
        '', 'Drift thresholds and the four predefined pacing lambdas are selected globally on these same forty development videos. All candidates, settings and tie rules were frozen beforehand. Pacing lambda=0 reproduces static decisions. Random averages all ten fixed seeds within each video. Mandatory refresh0 and exact B constraints apply to all sparse methods. Dense is a separate B=16 reference.',
        '', '## Results','', 'Lower mean post-decision cosine error is better; every video has equal weight.','',
        '| Policy | B=2 | B=4 | B=8 |','| --- | ---: | ---: | ---: |']
    for policy in u.POLICIES: lines.append(f"| {LABELS[policy]} | "+' | '.join(fmt(metrics[B,policy]['mean_output_error']) for B in [2,4,8])+' |')
    lines += ['',f"Dense B=16 mean error: {metrics[16,'dense']['mean_output_error']:.3e} (approximately zero). Refresh counts are simulated logical calls; **actual encoder executions=0**.",
        '', 'Primary B=4 comparison. Percentage improvements are computed separately per video before averaging.','',
        '| Policy | Mean error | Mean absolute gain vs periodic | Mean per-video gain (%) | Improved/tied/worsened | Mean oracle-gap closure (%) |',
        '| --- | ---: | ---: | ---: | --- | ---: |']
    for policy in u.POLICIES:
        m=metrics[4,policy]; gap=m['mean_defined_oracle_gap_closure']
        lines.append(f"| {LABELS[policy]} | {fmt(m['mean_output_error'])} | {fmt(m['mean_absolute_improvement_vs_periodic'])} | {fmt(m['mean_per_video_percentage_improvement'],2)} | {m['improved']}/{m['tied']}/{m['worsened']} | {fmt(100*gap if gap is not None else None,2)} |")
    lines += ['', 'Absolute means, mean per-video percentages, and mean per-video oracle-gap closures have different denominators and can differ in sign. Ratios of aggregate means are not mean per-video percentages. Negative values are retained; zero/small denominators are missing with reasons. The oracle is a full-future lower bound; privileged actual-drift thresholding is **not** a bound or deployable method.', '', '## Does budget-aware action supervision help?','']
    for B in [4,2,8]:
        m=metrics[B,'value_age_motion_budget']
        lines.append(f"B={B}: action value with motion has mean absolute improvement versus periodic {m['mean_absolute_improvement_vs_periodic']:+.6f}, mean per-video percentage {fmt(m['mean_per_video_percentage_improvement'],2)}%; {m['improved']} improved, {m['tied']} tied, {m['worsened']} worsened videos.")
        for ref in ['paced_age_plus_motion','value_age_budget','age_plus_motion']:
            c=paired[B,'value_age_motion_budget',ref]
            lines.append(f"Versus {LABELS[ref]}: mean absolute improvement {c['mean_absolute_improvement']:+.6f}; {c['improved']}/{c['tied']}/{c['worsened']} improved/tied/worsened; mean per-video percentage {fmt(c['mean_per_video_percentage_improvement'],2)}%.")
        priv=paired[B,'privileged_actual_drift','age_plus_motion']
        lines.append(f"Privileged actual-drift threshold versus predicted-drift static threshold: mean absolute improvement {priv['mean_absolute_improvement']:+.6f}; {priv['improved']}/{priv['tied']}/{priv['worsened']}. A weak actual-drift threshold despite oracle headroom implicates the greedy threshold rule; a large privileged/predicted gap implicates prediction as well.")
        periodic=metrics[B,'periodic']['mean_output_error']; oracle=metrics[B,'oracle']
        lines.append(f"Oracle mean headroom {periodic-oracle['mean_output_error']:.6f}; mean per-video possible reduction {fmt(oracle['mean_per_video_percentage_improvement'],2)}%; undefined oracle gaps {oracle['undefined_oracle_gaps']}.")
    motion=paired[4,'value_age_motion_budget','value_age_budget']; pace=paired[4,'value_age_motion_budget','paced_age_plus_motion']
    lines += ['',f"At B=4, adding motion {'lowers' if motion['mean_absolute_improvement']>u.TOL else 'raises' if motion['mean_absolute_improvement']<-u.TOL else 'ties'} mean error relative to age/budget alone by {abs(motion['mean_absolute_improvement']):.6f}. The controller {'beats' if pace['mean_absolute_improvement']>u.TOL else 'does not beat'} pacing in average development error. Mixed per-video counts above must be assessed alongside means; no significance is claimed.",
        '', '## Early spending, delayed spending, and approximation error','',
        'The evaluator logs both feasible oracle action values. Positive regret on a refresh identifies a costly spend-now decision; positive regret on reuse identifies a costly delay. Summed regrets reproduce the policy mean-error excess over the exact oracle. Future oracle values are diagnostic only. Forced-refresh counts by themselves do not establish success.','',
        '| Policy | B | Mean forced refreshes | Mean last refresh | Mean outputs after exhaustion | Mean summed action regret |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for B in [2,4,8]:
        for policy in ['age_plus_motion','paced_age_plus_motion',*u.VALUE_COLUMNS]:
            m=metrics[B,policy]
            lines.append(f"| {LABELS[policy]} | {B} | {m['mean_forced_refreshes']:.2f} | {m['mean_last_refresh_position']:.2f} | {m['mean_outputs_after_budget_exhausted']:.2f} | {m['mean_sum_action_regret']:.6f} |")
    for policy in [*u.VALUE_COLUMNS,'paced_age_plus_motion','age_plus_motion']:
        g=traces[(traces.policy==policy)&(traces.budget==4)&(traces.candidate_ordinal>0)]
        spend=g[(g.refresh)&(g.evaluator_action_regret>u.TOL)]; delay=g[(~g.refresh)&(g.evaluator_action_regret>u.TOL)]
        lines.append(f"{LABELS[policy]} B=4: {len(spend)} costly refresh decisions (total regret {spend.evaluator_action_regret.sum():.6f}) and {len(delay)} costly reuse decisions (total regret {delay.evaluator_action_regret.sum():.6f}) across forty videos.")
        worst=g.sort_values(['evaluator_action_regret','source_video_id','candidate_ordinal'],ascending=[False,True,True]).iloc[0]
        lines.append(f"Largest action-regret example: {worst.source_video_id}, candidate {int(worst.candidate_ordinal)}, cached {int(worst.cached_before_ordinal)}, remaining {int(worst.remaining_budget_before)}, action {'refresh' if worst.refresh else 'reuse'}, reason {worst.decision_reason}, predicted score {fmt(worst.score_decision)}, regret {worst.evaluator_action_regret:.6f}.")
    for example in examples: lines.append(f"Timeline {example['kind']}: {example['source_video_id']} ({example['label']}), B4 gain {example['absolute_improvement_vs_periodic']:+.6f}; chosen by the frozen descriptive rule.")
    primary=metrics[4,'value_age_motion_budget']
    if primary['mean_absolute_improvement_vs_periodic']>u.TOL and pace['mean_absolute_improvement']>u.TOL:
        recommendation='The development mean supports a frozen fresh-video confirmation of the proposed controller against periodic, pacing and age/budget controls. This is preliminary evidence, with per-video losses retained; it does not justify recognition or timing claims. Run the predefined hundred-video confirmation without retuning before considering those later experiments.'
    else:
        recommendation='The proposed controller does not provide a clear average advantage over both periodic and pacing in this development study. Preserve this negative/mixed result. A fresh confirmation may audit these fixed rules, but do not promote it to recognition or GPU timing on this evidence. Inspect signed-value approximation and budget decisions on training or separate development data without repeatedly tuning these forty evaluated videos.'
    lines += ['', '## Recommendation and limits','',recommendation,'',
        'Failure applies to these input definitions, ridge models, targets and rules; it does not prove every adaptive policy must fail. Development settings were selected on evaluated development videos. Features measure appearance fidelity, not action-recognition accuracy. Replay measures no decoding, controller, motion-processing, transfer or GPU latency. Full-clip candidate selection and future-reference codec vectors make this offline. Oracle labels use privileged future RGB during training. Many states from one source are correlated, not independent videos. No confidence intervals, significance, accuracy or speedup estimates are manufactured.',
        '', 'The frozen prospective `fresh_test_protocol.json` requires **100 new videos, five per existing class**, excluding all prior source IDs and duplicate hashes. It fixes these models/settings and paired whole-video uncertainty estimation. That set was not acquired or evaluated.','',
        '## Plots and reproduction','', '![Error versus simulated refresh budget](error_vs_refresh_budget.png)','',
        '![B4 per-video gains](per_video_B4_improvement.png)','', '![Success and failure timelines](example_timelines.png)','',
        'Run from the repository root with the existing environment:','', '```powershell',
        '& .\\.venv\\Scripts\\python.exe .\\replay_budget_aware_policies.py --input results/cache_drift_v2 --output results/budget_aware_refresh_v2 --budgets 2 4 8 --seed 20261014',
        '```','', 'Existing completed and partial directories are preserved by choosing a fresh directory. Scientific missing values are blank in CSV and null in JSON; use round-trip float parsing for audits. See protocol, split, training tables, predictor parameters, all threshold results, traces, per-video metrics, aggregate metrics, oracle verification and verification files.']
    (output/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return examples
