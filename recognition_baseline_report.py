"""Learning-curve findings from saved measurements; no arbitrary quality gate."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from task_aware_oracle_report import save
from recognition_baseline_data import read

COMMANDS = r"""```powershell
& .\.venv\Scripts\python.exe -m unittest test_recognition_baseline_v2 -v
& .\.venv\Scripts\python.exe .\run_recognition_baseline_v2.py --preflight-only --max-download-gib 20
& .\.venv\Scripts\python.exe .\run_recognition_baseline_v2.py --resume --max-download-gib 20 --download-workers 8
& .\.venv\Scripts\python.exe .\run_recognition_baseline_v2.py --audit-only
& .\.venv\Scripts\python.exe .\validate_release.py
```"""
LABELS={'dense':'Dense B16','periodic':'Periodic B4','feature_oracle':'Feature oracle B4 (hindsight)','task_oracle':'Task oracle B4 (label + hindsight)'}


def incomplete_report(output:Path,status:dict):
    acquisition=read(output/'acquisition.json') if (output/'acquisition.json').exists() else {}
    lines=['# recognition_baseline_v2: incomplete','',str(status.get('blocker')),'',
           'The planned400 training/100 development/40 exploratory diagnostic has not passed acquisition and verification. No reduced-cohort recognition or oracle result is substituted.','',
           '|Class|Training missing|Development missing|','|---|---:|---:|']
    for label,row in acquisition.get('shortfall_by_class',{}).items():lines.append(f"|{label}|{row['train']}|{row['development']}|")
    lines+=['','Preserved acquisition/progress and charged HTTP body bytes are in acquisition.json,collection_progress.json and download_accounting.json. Network/CUDA failures do not cause cohort replacement.','',COMMANDS]
    (output/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'README.md').write_text('# recognition_baseline_v2\n\nStatus: incomplete. See [findings](findings.md) for exact shortages and resume commands.\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,4));ax.text(.5,.5,'INCOMPLETE: planned full dataset required\nNo recognition measurements reported',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
    folder=output/'figures';folder.mkdir(exist_ok=True);save(fig,folder,'dense_recognition_learning_curve.png')


def report(output:Path):
    folder=output/'figures';folder.mkdir(exist_ok=True)
    curve=read(output/'learning_curve.json')['rows'];agg=pd.DataFrame(read(output/'aggregate_metrics.json')['rows'])
    raw=pd.read_csv(output/'per_video_metrics.csv',float_precision='round_trip');intervals=read(output/'bootstrap_intervals.json')['comparisons']
    primary=agg[agg.training_per_class==20];roles=['development','exploratory']
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,role in zip(axes,roles):
        training=[r for r in curve if r['cohort']=='training'];evaluated=[r for r in curve if r['cohort']==role]
        ax.plot([r['training_per_class']*20 for r in training],[100*r['top1_accuracy'] for r in training],marker='o',label='Training dense')
        ax.plot([r['training_per_class']*20 for r in evaluated],[100*r['top1_accuracy'] for r in evaluated],marker='s',label=f'{role} dense')
        ax.axhline(5,color='grey',linestyle=':',label='Chance/majority5%')
        ax.set(title=role+' (fixed cohort)',xlabel='Nested training source videos',xticks=[80,200,400],ylim=(0,105));ax.legend()
    axes[0].set_ylabel('Dense top-1 action classification accuracy (%)')
    fig.suptitle('Same encoder/head recipe; 400-video classifier designated primary before training')
    fig.tight_layout(rect=[0,0,1,.93]);save(fig,folder,'dense_recognition_learning_curve.png')
    for metric,name,ylabel,scale in [('top1_accuracy','primary_accuracy_by_policy.png','Top-1 action classification accuracy (%)',100),('cross_entropy_nats','classification_loss_by_policy.png','Cross-entropy (nats/video)',1)]:
        fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
        for ax,role in zip(axes,roles):
            g=primary[primary.cohort==role].set_index('policy').loc[list(LABELS)]
            ax.bar(range(4),g[metric]*scale,color=['#688cac','#498f82','#a69974','#b66a51'])
            ax.set(title=role,xlabel='Same frozen400-video-trained classifier',xticks=range(4),xticklabels=list(LABELS.values()))
            ax.tick_params(axis='x',rotation=20)
        axes[0].set_ylabel(ylabel);fig.suptitle('Oracle bars are privileged hindsight; sparse budgets exactly4')
        fig.tight_layout(rect=[0,0,1,.93]);save(fig,folder,name)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,role in zip(axes,roles):
        for policy in list(LABELS)[1:]:
            g=raw[(raw.training_per_class==20)&(raw.cohort==role)&(raw.policy==policy)]
            ax.scatter(g.feature_cosine_error,g.cross_entropy,s=22,alpha=.6,label=LABELS[policy])
        ax.set(title=role,xlabel='Mean output-feature cosine error',ylabel='Cross-entropy (nats/video)');ax.legend(fontsize=8)
    fig.suptitle('Primary400-video classifier; descriptive paired source observations');fig.tight_layout(rect=[0,0,1,.93]);save(fig,folder,'feature_error_vs_classification_loss.png')
    lines=['# recognition_baseline_v2 findings','',
           'This is a custom CVDF Kinetics validation-source diagnostic, not an official full-dataset benchmark or independent confirmation. The official training archive list was available, but no training shards were cached; the cached validation route was fixed before training for bounded acquisition. No completed100-video fresh-confirmation feature or policy data was used. Its source IDs/file hashes were excluded.','',
           '## Learning curve','',
           'The400-video classifier is primary by protocol, not selected by oracle gains. All sizes use class-balanced nested subsets of one400-video pool and the same100-video development cohort. The new80-video subset may differ from the old80-video v1 cohort. Original custom validation/test sources never enter training. Existing40 development sources remain, with60 new additions; existing40 exploratory sources remain separate.','',
           '|Training videos|Cohort|Dense top1/macro (%)|CE (nats)|','|---:|---|---:|---:|']
    for r in curve:lines.append(f"|{r['training_per_class']*20}|{r['cohort']}|{100*r['top1_accuracy']:.2f}|{r['cross_entropy']:.6f}|")
    lines+=['','Normalization and every fit use training-subset data only. Four unchanged regularization values (.001,.01,.1,1) use the v1 deterministic float64 linear cross-entropy/LBFGS recipe. Development dense top1,then lower loss,then grid order select lambda. All grids and frozen scalers/weights are saved; no model is refitted on development. Every selected model freezes before exploratory evaluation.','',
            '## Primary400-video classifier','', '|Cohort|Policy|Correct/videos|Top1/macro (%)|CE (nats)|Feature cosine error|','|---|---|---|---:|---:|---:|']
    for r in primary.itertuples():lines.append(f"|{r.cohort}|{LABELS[r.policy]}|{r.correct_videos}/{r.videos}|{100*r.top1_accuracy:.2f}|{r.cross_entropy_nats:.6f}|{r.mean_feature_cosine_error:.6f}|")
    lines+=['','## Paired whole-video uncertainty','',
            'Positive accuracy gains are candidate minus baseline; positive loss reductions are baseline minus candidate.10,000 whole-source-video paired resamples use PCG64 seed20261016 and linear percentiles. One saved index identity is shared across all compared policies within each cohort. Classifier parameters remain fixed. Intervals condition on these fitted models and sampled classes; training/subset/regularization-selection uncertainty is excluded. Development is tuned; exploratory sources were previously evaluated. Secondary intervals are marginal/unadjusted.','',
            '|Cohort|Candidate vs baseline|Outcome|Mean gain|95% interval|Improved/tied/worsened|','|---|---|---|---:|---|---|']
    for r in intervals:
        scale=100 if r['metric']=='correct' else 1;unit='pp' if scale==100 else 'nats'
        lines.append(f"|{r['cohort']}|{r['candidate']} vs {r['reference']}|{r['metric']}|{scale*r['mean']:+.6f} {unit}|[{scale*r['CI95_lower']:+.6f},{scale*r['CI95_upper']:+.6f}]|{r['improved']}/{r['tied']}/{r['worsened']}|")
    by={(r['training_per_class'],r['cohort']):r for r in curve}
    lines+=['','## Research interpretation','']
    for role in roles:
        low=by[(4,role)];high=by[(20,role)];tr_low=by[(4,'training')];tr_high=by[(20,'training')]
        lines.append(f"1–2. {role}: additional training changes dense accuracy from {100*low['top1_accuracy']:.2f}% to {100*high['top1_accuracy']:.2f}% ({100*(high['top1_accuracy']-low['top1_accuracy']):+.2f}pp). The training/evaluation accuracy gap changes from {100*(tr_low['top1_accuracy']-low['top1_accuracy']):.2f}pp to {100*(tr_high['top1_accuracy']-high['top1_accuracy']):.2f}pp. Interpret only matched new nested subsets, not differences from the unmatched v1 cohort.")
    for role in roles:
        ac=next(r for r in intervals if r['cohort']==role and r['primary'] and r['metric']=='correct')
        ce=next(r for r in intervals if r['cohort']==role and r['primary'] and r['metric']=='cross_entropy')
        lines.append(f"3. {role} primary task-oracle accuracy gain {100*ac['mean']:+.2f}pp, interval[{100*ac['CI95_lower']:+.2f},{100*ac['CI95_upper']:+.2f}], and loss reduction {ce['mean']:+.6f} nats. {'Correctness headroom is supported for this privileged comparison.' if ac['CI95_lower']>0 else 'Correctness improvement is inconclusive or absent; a positive point estimate is not confirmation.'} Loss cannot be worse than periodic by construction: all455 schedules include periodic. This oracle optimizes true-label loss, not top1, and is neither deployable accuracy nor an accuracy-maximizing upper bound.")
        p=primary[primary.cohort==role].set_index('policy');f=p.loc['feature_oracle'];q=p.loc['periodic']
        lines.append(f"4. {role}: feature oracle lowers feature cosine error by {q.mean_feature_cosine_error-f.mean_feature_cosine_error:+.6f}, with accuracy change {100*(f.top1_accuracy-q.top1_accuracy):+.2f}pp and loss reduction {q.cross_entropy_nats-f.cross_entropy_nats:+.6f} nats. These are distinct objectives; reducing cosine error need not improve classification.")
    lines+=['','5. Yes: caching repeats a refreshed feature for several candidate outputs. For a linear mean-pooled head this exactly changes the temporal weights of candidate logits. A true-label oracle can overweight class-favorable appearance and suppress ambiguous frames. Its advantage can therefore include hindsight temporal reweighting rather than better recognition of causal action segments. No temporal localization annotation or label-free schedule demonstrates semantics here.','',
            '6. The measured oracle headroom can motivate a separately frozen,label-free controller feasibility study; it does not establish that available causal inputs predict task value. First compare against a label-free temporal-selection/reweighting baseline on development sources, freeze its settings,and confirm on genuinely new sources before measuring real latency. If correctness headroom is inconclusive, resolve the recognition/temporal-weighting limitation before investing in a controller. No arbitrary accuracy cutoff is used as a publication gate.','',
            '7. The first representation limitation to investigate is the static ImageNet appearance encoder with temporal mean pooling: it ignores action dynamics and can classify context rather than motion. Check class-wise errors,normalization/regularization convergence,and a stronger pretrained temporal action representation in a separate study. No new representation or recognition model was downloaded here.','',
            'Original encodings,16 uniform decoded candidates,existing ResNet18 weights/preprocessing,float32 batch1 CUDA references are fixed. Every sparse replay has mandatory0,exactly4 refreshes and16 outputs with latest-cache reuse. Saved-feature replay executes zero encoders. Reference collection elapsed time is not adaptive inference latency. Full-clip candidates and codec future references remain offline; no streaming causality,speedup or publication-level novelty is claimed.','',
            '## Reproduction','',COMMANDS,'',
            'Protocol,source hashes,dataset identities,exclusions,all acquisition retries/body-byte accounting,frozen models and verification accompany this snapshot. Cached archives do not charge the new-download budget; all new HTTP body reads and conservative interrupted reservations do. Existing completed outputs are audited rather than overwritten.']
    (output/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'README.md').write_text('# recognition_baseline_v2\n\nStatus: complete; primary400-video classifier predefined.100 development and40 previously used exploratory videos are separate.\n\nSee [findings](findings.md),[protocol](protocol.json),[learning curve](learning_curve.json),[metrics](aggregate_metrics.json),[intervals](bootstrap_intervals.json),[verification](verification.json).\n\n'+COMMANDS+'\n\n![Dense learning curve](figures/dense_recognition_learning_curve.png)\n\n![Primary accuracy](figures/primary_accuracy_by_policy.png)\n',encoding='utf-8')
