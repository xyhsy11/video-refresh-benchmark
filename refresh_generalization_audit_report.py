"""Readable audit reports derived only from the new saved audit tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import refresh_generalization_audit_utils as u

LABELS = {'keep_periodic':'Keep periodic','random_move_expected':'Random single move (expected)',
          'motion_median_move':'Motion-median single move','direct_RF400':'Direct RF400',
          'two_target_RF200x2':'Forward two-target RF200x2','history_RF400':'Cached-history RF400'}


def report(output: Path, protocol: dict, summary: pd.DataFrame, training: pd.DataFrame, identity: dict) -> None:
    root=Path(__file__).resolve().parent
    output_name=output.relative_to(root).as_posix()
    roles=pd.DataFrame([
        {'cohort':'recognition training pool','sources':400,'opportunities_in_this_audit':0,'original_purpose':'class-balanced recognition learning curve;custom CVDF validation-source split','recognition_head_use':'weights and normalization fitted on400;smaller nested subsets secondary','gain_predictor_use':'not used for the audited gain forests','previously_inspected_outcomes':True,'untouched_confirmation':False,'note':'Source/hash disjoint from100 development and40 exploratory;0 audit opportunities does not mean no saved candidate features'},
        {'cohort':'development','sources':100,'opportunities_in_this_audit':900,'original_purpose':'40 old development +60 additions;recognition regularization selection','recognition_head_use':'dense top1 then CE selected regularization0.01;no gradient training','gain_predictor_use':'five folds:80 training/20 heldout each;final forest uses all100;features/models previously investigated','previously_inspected_outcomes':True,'untouched_confirmation':False,'note':'Controller OOF excludes held-out source from forest/scaler;head selection already used all100'},
        {'cohort':'exploratory','sources':40,'opportunities_in_this_audit':360,'original_purpose':'cache_drift_v2 original test membership;previously evaluated in several studies','recognition_head_use':'not used for frozen head fitting or regularization selection','gain_predictor_use':'final probes frozen on100 development;no exploratory fitting;research outcomes repeatedly inspected','previously_inspected_outcomes':True,'untouched_confirmation':False,'note':'Separate held-out predictions conditional on fixed recipe;not untouched research confirmation'},
        {'cohort':'protected completed fresh confirmation','sources':100,'opportunities_in_this_audit':None,'original_purpose':'completed frozen budget-aware feature-fidelity confirmation','recognition_head_use':'excluded via existing recognition exclusion identities','gain_predictor_use':'not accessed or repurposed','previously_inspected_outcomes':True,'untouched_confirmation':False,'note':'Only prior exclusion evidence referenced;no protected outcome/feature table parsed'},
    ])
    u.csv(output/'cohort_roles.csv',roles)
    inputs=pd.DataFrame([
        ['age/budget4','candidate position,elapsed seconds,remaining calls,next planned refresh distance','current prefix plus saved candidate plan','bookkeeping','no current RGB/labels;full-clip16-candidate endpoints use whole-clip inspection'],
        ['codec motion6','current mean/p90/availability and interval mean/max/available fraction','decoded positions after actual periodic cache through current candidate','software codec export and CPU summaries','no current RGB;B-frame/future-reference codec vectors can depend on future encoded references'],
        ['cached confidence6','entropy,max probability,margin for cached and mean past-output logits','only already refreshed features;past outputs stop at t-1','linear head/softmax/bookkeeping','no true-label score;not full-video dense prediction'],
        ['cached centered logits40','20 cached scores and20 mean past scores;fixed class order','periodic cache refreshed0,4,8 strictly before decision','linear head on cached features','no current unencoded or skipped dense feature;no identities as model inputs'],
        ['history T44','ordered cached-score changes,masks,margin variability,top1 switches','distinct past periodic refresh events only','retain history and compute differences/softmax summaries','no new RGB;length2 recoverable from C;length3 retains individual/order information'],
        ['privileged P (excluded)','current fresh RGB logits','only after paying current encoding','extra current RGB work potentially needed','excluded from all audited models and motion baseline'],
        ['targets B,C,G (excluded)','whole-video loss differences using true label and full references','retrospective evaluator only','offline reference replay','labels/future features permitted for targets,not predictor inputs'],
    ],columns=['group','contains','availability','additional_computation','limitations'])
    u.csv(output/'input_availability.csv',inputs)
    # Four in-sample fold appearances are averaged within source, never counted as400 independent videos.
    metric=['utility','MAE','training_median_MAE','MAE_improvement_over_training_median']
    fold_training=training[training.cohort!='final_development'].groupby(['method','source_video_id'],as_index=False)[metric].mean()
    fold_training['phase']='fold-training in-sample (four models/source averaged)'
    final_training=training[training.cohort=='final_development'][['method','source_video_id',*metric]].copy()
    final_training['phase']='final-model in-sample'
    held=u.frame(output/'per_source_metrics.csv')
    held=held[held.method.isin(u.LEARNED)][['cohort','method','source_video_id',*metric]].copy()
    held['phase']=held.cohort.map({'development_oof':'development OOF','exploratory':'exploratory held out'})
    diagnostics=pd.concat([fold_training,final_training,held.drop(columns='cohort')],ignore_index=True)
    u.csv(output/'training_vs_heldout_source_metrics.csv',diagnostics)
    diagnostic_summary=diagnostics.groupby(['phase','method'],sort=True).agg(sources=('source_video_id','nunique'),utility=('utility','mean'),MAE=('MAE','mean')).reset_index()
    u.csv(output/'training_vs_heldout.csv',diagnostic_summary)
    class_summary=u.frame(output/'per_source_metrics.csv').groupby(['cohort','method','class'],sort=True).agg(sources=('source_video_id','nunique'),utility=('utility','mean'),selected=('selected','sum'),harmful_selected=('harmful_selected','sum')).reset_index()
    u.csv(output/'per_class_descriptive.csv',class_summary)
    leakage='''# Leakage, provenance and evaluation-design audit

## Verified implementation checks

Original source-video IDs, not clip filenames, define folds. All nine opportunities stay together. Every held-out forest records only its fold-training source/example IDs; training-only population means, standard deviations, constant-column handling and720/900 training rows are independently reconstructed. Saved tree bootstrap root targets are checked. The motion rule receives only availability and p90 values from training rows to construct its median; it never optimizes against G. Final exploratory thresholds use development only. All frozen forest hyperparameters and >1e-12 decisions are retained. No model or threshold is refitted to improve audit outcomes.

Direct, benefit-minus-cost and history predictions are reproduced with explicit float32 tree traversal independent of original prediction/report functions; library predictions and archived selections are cross-checked. B=LA-LB,C=LC-LB,G=LA-LC=B-C are checked against saved arrays using separate explicit cache replay and stable cross entropy. Predictor matrices use whitelisted56 or100 columns; target/class/source metadata are excluded. No verified coding leakage was found in these checked paths. This is not proof against every possible defect or all upstream provenance errors.

## Permitted, inspected development data and design limitations

The frozen recognition head is fitted on400 source-disjoint training videos. Its normalization is independently reproduced from those dense mean-pooled references. Regularization0.01 was selected using all100 development videos (dense accuracy, then CE, then grid order). Therefore gain-predictor OOF is source-held-out only for that predictor; it is not a nested, fully independent head-plus-predictor evaluation. Recognition selection dependence is a design limitation, not an accidental coding bug.

The40 exploratory videos were excluded from head/probe fitting and selection in these frozen runs, but their outcomes had already been inspected in earlier research. Repeated representation/model choices across studies use reused cohorts. Saved per-run protocols can show fixed recipes before that run; they cannot prove decisions were uninformed by all previous outcomes or unrecorded human choices. Fixed seeds and no within-run tuning do not remove repeated-research selection uncertainty.

## Identity and unresolved provenance

All540 recognition-manifest records have distinct original source IDs and encoded SHA256 values. Source IDs agree with clip source prefixes; no same-parent windows cross cohorts, so no adjacent windows from a recorded parent leak across these cohorts. The100/40 audit memberships and original class-stratified five folds match saved identities. All540 saved reference-array hashes and candidate order/timestamps are independently checked. Encoded-file hashes are taken from the previously validated manifest; this audit does not rehash or decode video files.

Different source IDs/encoded hashes do not rule out reuploads, compilations, semantically near-duplicate content or a provenance error. No perceptual duplicate search or new raw-video inspection is conducted. Original Kinetics labels are reused; label correctness is not independently re-annotated. The data are a custom subset from CVDF validation archives, not an official full recognition benchmark. These unresolved matters are reported rather than assumed absent.

## Availability and offline scope

See input_availability.csv. Cached logits/history are legitimate pre-refresh observations under the unmodified periodic history, and repeated outputs add no new event. Candidate positions include full-clip first/last uniform selection. Codec vectors include future references in B-frames; restricting descriptor aggregation to a decoded prefix does not establish strict streaming causality. Privileged current RGB columns, true-label scores, future dense features and retrospective loss targets are not predictor inputs. Future references and labels are confined to the evaluator/target path. No nine independent moves are combined into a sequential schedule.

## Confirmation status

No genuinely independent test was run. Both current cohorts have been inspected. The completed protected100-video feature-fidelity confirmation is not repurposed, and its outcomes, features and motion tables are not read. No eligible untouched saved cohort is established. A prospective protocol is supplied without acquisition or evaluation.
'''
    (output/'leakage_audit.md').write_text(leakage,encoding='utf-8')
    exclusions=u.read(root/'results/recognition_baseline_v2/exclusions.json')
    excluded=sorted(set(exclusions['source_ids'])|{s for ids in protocol['membership'].values() for s in ids})
    hashes=sorted(set(exclusions['file_hashes'])|{e['file_identity']['sha256'] for e in u.read(root/'results/recognition_baseline_v2/dataset_manifest.json')['videos']})
    u.save(output/'fresh_test_exclusions.json',{'source_ids':excluded,'file_hashes':hashes,'rule':'union existing recognition exclusions and all540 known recognition sources;refresh from all subsequent persisted memberships before prospective acquisition;do not exclude own partial new cohort on resume'})
    fresh=f'''# Prospective fresh-source protocol (not executed)

No genuinely independent test was run.

This document specifies an optional confirmation of the frozen single-move methods, not evidence that another cached-feature variant is warranted. No acquisition is launched. Before acquisition, date/sign this protocol, verify no new memberships have appeared, and freeze an execution plan and encoded/reference/observation manifests before reading outcomes.

Population:100 genuinely new source videos,five per each of the same20 recorded classes,from the same documented Kinetics-derived population and fixed eligibility/ResNet18 collection protocol. This feasible class-balanced size gives100 independent source units; it is not a power guarantee or a claim that five/class characterize unseen classes. Freeze a precision/power calculation from prior per-source variability and a scientifically justified smallest effect before acquisition if a formal decision is required; do not change sample size after outcomes. Do not keep adding sources until intervals become favorable.

Exclude the {len(excluded)} source IDs and {len(hashes)} encoded hashes in fresh_test_exclusions.json,plus any later persisted/reserved membership. Preserve original source identity across windows and all classes. Deduplicate source IDs and encoded hashes; document unresolved perceptual duplicates. Never reuse the completed protected confirmation cohort. Freeze deterministic ranking,eligibility,replacement and retry rules without looking at G or policy outcomes.

Freeze the current primary recognition head by its SHA256 {identity['head_sha256']}. Freeze final-development Direct_G_RF400,forward benefit/cost RF200 models and C+T_RF400 JSON/joblib identities from this audit protocol. No normalization,feature,forest,threshold or head refitting. The motion threshold is the saved final-development available-p90 median; missing=>reject. Gain acceptance remains >1e-12. Include keep_periodic,analytical random0.5,motion median and all three learned methods. Nine prescribed independent counterfactuals/video;never combine them into one schedule.

Primary comparison,if this specific frozen representation is confirmed: C+T_RF400 versus keep_periodic absolute utility. Success requires a positive point estimate and an entirely positive95% paired whole-source percentile interval; an interval containing0 is inconclusive,not equivalence. All other methods/comparisons,including motion and random baselines,are predefined secondary marginal/unadjusted analyses. Do not declare broad superiority from favorable secondary intervals. Positive absolute utility alone does not establish runtime value or a sequential controller.

Use10,000 whole-source resamples,PCG64 seed20261022,linear2.5/97.5 percentiles,one saved index matrix across paired methods. Keep nine opportunities together,average within source then equally across sources. Report selected/helpful/harmful/missed-benefit counts and denominators,MAE relative to frozen training-median,undefined AUROCs and random expectations. Evaluate once,report negative outcomes,no retuning. Intervals condition on fixed models and sampled classes and omit training uncertainty. This is an offline task-loss diagnostic,not action accuracy,streaming or latency evidence.
'''
    (output/'fresh_test_protocol.md').write_text(fresh,encoding='utf-8')
    figures=output/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(14,5.8),sharey=True)
    for ax,cohort,title in zip(axes,['development_oof','exploratory'],['Development OOF (reused100 sources)','Exploratory (reused40 sources)']):
        s=summary[summary.cohort==cohort].set_index('method').loc[u.METHODS]
        y=np.arange(len(s));v=s.utility_nats_per_opportunity.to_numpy()
        ax.errorbar(v,y,xerr=np.stack([v-s.utility_CI95_lower.to_numpy(),s.utility_CI95_upper.to_numpy()-v]),fmt='o',capsize=4)
        ax.axvline(0,color='black',lw=1);ax.set_yticks(y,[LABELS[m] for m in s.index]);ax.invert_yaxis()
        ax.set_xlabel('Single-move utility (nats/opportunity)\n95% whole-source percentile interval');ax.set_title(title);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Frozen single-move decisions;not sequential refresh controllers');fig.tight_layout();fig.savefig(figures/'utility_intervals.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for ax,metric,title in zip(axes,['harmful_selection_rate','missed_benefit_rate'],['Harmful selected / all selected','Beneficial rejected / all beneficial']):
        for offset,cohort,color in [(-.18,'development_oof','#2468a2'),(.18,'exploratory','#ca7b21')]:
            s=summary[summary.cohort==cohort].set_index('method').loc[u.METHODS]
            ax.bar(np.arange(6)+offset,s[metric].to_numpy(dtype=float),width=.34,color=color,label=cohort)
        ax.set_xticks(np.arange(6),[LABELS[m].replace(' ','\n') for m in u.METHODS],fontsize=8)
        ax.set_ylim(0,1.05);ax.set_title(title);ax.set_ylabel('Fraction;undefined harmful rate for no selections');ax.legend(fontsize=8)
    fig.suptitle('Descriptive rates;random values are ratios of expected counts');fig.tight_layout();fig.savefig(figures/'harmful_and_missed.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    phase=['fold-training in-sample (four models/source averaged)','development OOF','final-model in-sample','exploratory held out']
    for ax,metric,title in zip(axes,['utility','MAE'],['Decision utility (nats/opportunity)','Gain MAE (nats)']):
        for method in u.LEARNED:
            s=diagnostic_summary[diagnostic_summary.method==method].set_index('phase').loc[phase]
            ax.plot(range(4),s[metric],marker='o',label=LABELS[method])
        ax.set_xticks(range(4),['Fold training\n100 unique sources','Development OOF\n100 sources','Final training\n100 sources','Exploratory\n40 sources'],fontsize=8)
        ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].axhline(0,color='black',lw=1);fig.suptitle('In-sample versus held-out diagnostics;not independent confirmation');fig.tight_layout();fig.savefig(figures/'training_vs_heldout.png',dpi=180);plt.close(fig)
    lines=['# refresh_generalization_audit_v1 findings','',
        'This bounded independent audit retains the frozen head,models,targets,folds and1e-12 decision threshold. It evaluates nine independent single-refresh moves per source,not sequential controller rollouts. Both cohorts are developmental and previously inspected. No genuinely independent test was run.','',
        '## What passed','',
        f'All540 reference-array hashes/source identities and training-only recognition normalization pass. Recognition regularization selection reproduces0.01 from development dense accuracy/loss. Dense accuracy reproduces72% development and65% exploratory;periodic70%/65%. All1260 loss/cache counterfactuals reproduce independently (maximum target discrepancy {identity["maximum_target_discrepancy"]:.3g}). Saved forests,scalers,source-held-out membership and selections are independently checked;see verification.json and model_checks.json. No verified implementation leakage was found in the checked paths.','',
        '## Fair single-move decisions','',
        '|Cohort|Method|Utility and95% interval (nats/opportunity)|Selected / total|Helpful / harmful / neutral|Harmful rate (denominator)|Missed benefit (denominator)|MAE vs training-median|',
        '|---|---|---|---:|---|---|---|---|']
    for row in summary.itertuples():
        counts=f'{row.selected:g}/{row.opportunities}'+(' expected' if row.counts_are_analytical_expectations else '')
        harmful='undefined:no selections' if pd.isna(row.harmful_selection_rate) else f'{row.harmful_selection_rate:.3%} ({row.harmful_selection_numerator:g}/{row.harmful_selection_denominator:g})'
        missed='undefined:no benefits' if pd.isna(row.missed_benefit_rate) else f'{row.missed_benefit_rate:.3%} ({row.missed_benefit_numerator:g}/{row.missed_benefit_denominator:g})'
        mae='not a regression method' if pd.isna(row.MAE) else f'{row.MAE:.6f} vs {row.training_median_MAE:.6f}'
        lines.append(f'|{row.cohort}|{LABELS[row.method]}|{row.utility_nats_per_opportunity:+.6f} [{row.utility_CI95_lower:+.6f},{row.utility_CI95_upper:+.6f}]|{counts}|{row.helpful_selected:g}/{row.harmful_selected:g}/{row.neutral_selected:g}|{harmful}|{missed}|{mae}|')
    lines+=['','Random accepts independently with probability0.5. Counts and utility are analytical expectations;its interval resamples source heterogeneity of expected utility,not Bernoulli decision noise. The motion baseline is training-available p90 median,strictly exceeded;missing=>reject. Neither baseline is a complete Random Refresh/Motion Threshold controller. All A-versus-C moves retain4 simulated refreshes and16 outputs. Four encoder calls do not imply equal total runtime.','',
        '## Training and generalization','', '|Phase|Method|Unique sources|Utility|MAE|','|---|---|---:|---:|---:|']
    for r in diagnostic_summary.itertuples():lines.append(f'|{r.phase}|{LABELS[r.method]}|{r.sources}|{r.utility:+.6f}|{r.MAE:.6f}|')
    train=diagnostic_summary[diagnostic_summary.phase==phase[0]].set_index('method')
    held=diagnostic_summary[diagnostic_summary.phase==phase[1]].set_index('method')
    if (train.utility > held.utility).all() and (train.MAE < held.MAE).all():
        interpretation='Every frozen forest is better in sample than on its held-out sources in utility and MAE. This supports an overfitting/generalization concern for these tested representations/models. In-sample performance is not a new method result. It does not identify distribution shift,prove sufficient input information,or establish inherent unpredictability.'
    else:
        interpretation='Inspect the model-specific training/held-out gaps above. Weak performance in both would suggest limited usefulness of the tested model/representation;strong training with weak held-out results supports a generalization concern. Neither establishes distribution shift or inherent unpredictability.'
    positive=summary[(summary.method.isin(u.LEARNED+['motion_median_move']))&(summary.utility_CI95_lower>0)]
    verdict='No audited nontrivial method establishes positive utility over keeping periodic on these reused cohorts.' if positive.empty else 'Some predefined descriptive absolute intervals are positive;inspect all paired baselines and reused-cohort limitations before claiming useful generalization.'
    lines+=['',interpretation,'',verdict,
        'A relative improvement over another harmful method is not useful positive utility. No regression metrics are assigned to decision-only methods. AUROC is pooled descriptive discrimination with explicit event counts;constant scores/single-class outcomes are missing with reasons. Per-source distributions and per-class results are descriptive;no favorable subgroup is promoted to confirmation.','',
        '## What failed or remains unverifiable','',
        'Untouched full-pipeline independence is absent. Recognition-head regularization used all100 development videos;controller OOF does not undo that selection. Exploratory outcomes were already inspected. Near-duplicate/reuploaded content under different IDs,unrecorded human outcome access and annotation correctness remain unresolved. Exact source IDs,windows and encoded hashes rule out only their recorded forms of overlap. These are design/provenance limits,not evidence of an accidental code leak. No protected confirmation outcomes are accessed. See leakage_audit.md and cohort_roles.csv.','',
        '## Computation','',
        'Actual RGB encoder executions,decodes,downloads,recognition fits and new predictor fits are zero. Sparse calls are simulated. New cpu_component_timings.csv measures only preloaded scaling/saved sklearn batch1 prediction/decision or motion threshold work,two warmups/five interleaved passes,fixed threads. It excludes observation construction,history reconstruction,codec extraction,decoding,RGB and recognition. The two-target model requires two200-tree predictions;equal total trees do not imply equal cost. Model nodes/serialized sizes are in model_checks.json. Existing history state payload904bytes and history-construction timing are historical component evidence,not measurements to add to these timings. Full-pipeline latency and adaptive speedup remain unmeasured.','',
        '## One next experiment','',
        'Design a separately preregistered temporal-recognition backbone/task benchmark with stronger temporal representation and genuinely new source videos before another cached-feature controller variant. First establish an adequate recognition/task objective and freeze source exclusions and comparisons. Do not launch an architecture search or acquisition in this audit. A frozen-method fresh-test protocol is provided for later use,without claiming these methods earned confirmation.','',
        'All intervals use10,000 paired whole-source resamples,PCG64 seed20261022,linear percentiles,shared method indices within cohort. All nine opportunities stay together;cohorts are separate. Intervals condition on already fitted models and reused cohorts,excluding full training/model-selection/repeated-research uncertainty;paired secondary intervals are marginal/unadjusted. This audit establishes no recognition-accuracy improvement,streaming causality,successful adaptive controller or end-to-end speedup.','',
        '## Reproduce from the repository root','', '```powershell',
        r'& .\.venv\Scripts\python.exe -m unittest test_refresh_generalization_audit -v',
        rf'& .\.venv\Scripts\python.exe .\audit_refresh_generalization.py --output {output_name} --seed 20261022 --bootstrap 10000',
        rf'& .\.venv\Scripts\python.exe .\audit_refresh_generalization.py --output {output_name} --seed 20261022 --bootstrap 10000 --audit-only',
        '```','', 'Completed runs audit without changing results or retiming. Use a versioned fresh directory if inputs/protocol change.']
    (output/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'README.md').write_text('# refresh_generalization_audit_v1\n\nSee [findings](findings.md),[cohort roles](cohort_roles.csv),[leakage audit](leakage_audit.md),[summary](summary.csv),[verification](verification.json) and [prospective protocol](fresh_test_protocol.md). No genuinely independent test was run.\n',encoding='utf-8')
