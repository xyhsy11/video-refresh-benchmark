"""Tables,paired plots and follow-up design from saved pooling-control results."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import refresh_pooling_utils as u

NAMES={'periodic':'Periodic','feature_oracle':'Feature oracle (dense-feature hindsight)',
       'cache_task_oracle':'Cache task oracle (true-label hindsight)',
       'unique_task_oracle':'Unique task oracle (true-label hindsight)',
       'RandomExact':'RandomExact (expected)','dense':'Dense16 reference'}


def report(output:Path,p:dict):
    intervals=u.read(output/'bootstrap_intervals.json')['contrasts'];aggregate=u.read(output/'aggregate_metrics.json')['rows']
    raw=u.frame(output/'per_video_metrics.csv');pairs=u.frame(output/'paired_comparisons.csv');overlap=u.frame(output/'schedule_overlap.csv')
    def result(cohort,name,b=4):return next(r for r in intervals if r['cohort']==cohort and r['contrast']==name and r['budget']==b)
    def metric(cohort,policy,pool,b=4):return next(r for r in aggregate if r['cohort']==cohort and r['policy']==policy and r['pool']==pool and r['budget']==b)
    def fmt(r):return f"{r['mean']:+.6f} [{r['CI95_lower']:+.6f},{r['CI95_upper']:+.6f}]"
    figures=output/'figures';figures.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(figures/(name+'.'+ext),dpi=180,bbox_inches='tight',pad_inches=.15)
        plt.close(fig)
    titles=['Development100 (primary Delta)','Previously inspected exploratory40 (secondary)']
    cohorts=['development','exploratory']
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.2),sharey=True)
    for ax,cohort,title in zip(axes,cohorts,titles):
        for i,(name,color) in enumerate([('H_C','#226ea6'),('H_U','#d47b20'),('Delta','#26844f')]):
            r=result(cohort,name);v=r['mean']
            ax.errorbar(v,i,xerr=[[v-r['CI95_lower']],[r['CI95_upper']-v]],fmt='o',color=color,capsize=5)
        ax.set_yticks([0,1,2],['Cache-optimized H_C','Unique-optimized H_U','Delta = H_C minus H_U'])
        ax.axvline(0,color='black',lw=1);ax.grid(axis='x',alpha=.2);ax.set_title(title,fontsize=11)
        ax.set_xlabel('CE headroom / difference (nats per video)\n95% class-stratified whole-source interval')
    fig.suptitle('B=4 privileged label-aware oracle headroom;reused cohorts',fontsize=13);save(fig,'primary_headroom_delta')
    matrices=[np.array([[result(c,'L_C_SC')['mean'],result(c,'L_U_SC')['mean']],
                        [result(c,'L_C_SU')['mean'],result(c,'L_U_SU')['mean']]]) for c in cohorts]
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained')
    lo,hi=min(m.min() for m in matrices),max(m.max() for m in matrices)
    for ax,cohort,m in zip(axes,cohorts,matrices):
        image=ax.imshow(m,cmap='YlOrRd',vmin=lo,vmax=hi)
        ax.set_xticks([0,1],['Cache pooling C','Unique pooling U'])
        ax.set_yticks([0,1],['Selected by cache oracle S_C','Selected by unique oracle S_U'])
        ax.set_title('Development100' if cohort=='development' else 'Previously inspected exploratory40')
        for i,policy in enumerate(['cache_task_oracle','unique_task_oracle']):
            for j,pool in enumerate(['C','U']):
                a=metric(cohort,policy,pool)['top1_accuracy']
                ax.text(j,i,f'CE {m[i,j]:.4f} nats\nTop-1 {100*a:.1f}%',ha='center',va='center',fontsize=11,color='black')
    fig.colorbar(image,ax=axes,shrink=.8,label='Mean CE (nats/video)')
    fig.suptitle('B=4 cross-evaluation;positions fixed within each row (privileged oracles)')
    for ext in ['png','pdf']:fig.savefig(figures/('cross_evaluation_matrix.'+ext),dpi=180,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),sharey=True)
    for ax,cohort in zip(axes,cohorts):
        for name,shift,color,marker in [('W_C',-.1,'#226ea6','o'),('W_U',.1,'#d47b20','x')]:
            g=pairs[(pairs.cohort==cohort)&(pairs.budget==4)&(pairs.contrast==name)].sort_values('source_video_id')
            ax.scatter(np.arange(1,len(g)+1)+shift,g.paired_value,s=20,alpha=.75,color=color,marker=marker,label=name+' (same frames)')
        ax.axhline(0,color='black',lw=1);ax.set_title('Development100' if cohort=='development' else 'Previously inspected exploratory40')
        ax.set_xlabel('Source index (fixed source-ID order)');ax.legend(fontsize=9);ax.grid(alpha=.15)
    axes[0].set_ylabel('CE(U) minus CE(C),nats/video\nPositive:cache weights lower loss for this exact set')
    fig.suptitle('B=4 fixed-schedule pooling effects;no reoptimization within a point');save(fig,'fixed_schedule_pooling_effects')
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.2),sharey=True)
    for ax,cohort in zip(axes,cohorts):
        for name,color in [('H_C','#226ea6'),('H_U','#d47b20'),('Delta','#26844f')]:
            rs=[result(cohort,name,b) for b in [2,4,8]];y=np.array([r['mean'] for r in rs])
            ax.errorbar([2,4,8],y,yerr=[y-np.array([r['CI95_lower'] for r in rs]),np.array([r['CI95_upper'] for r in rs])-y],marker='o',color=color,capsize=3,label=name)
        ax.axhline(0,color='black',lw=.8);ax.axvline(4,color='gray',ls=':',lw=1)
        ax.set_xticks([2,4,8]);ax.set_xlabel('Simulated RGB refresh budget');ax.legend();ax.grid(alpha=.15)
        ax.set_title('Development100' if cohort=='development' else 'Previously inspected exploratory40')
    axes[0].set_ylabel('CE headroom / difference (nats/video)')
    fig.suptitle('Privileged oracle pooling sensitivity;B=4 primary,B=2/8 secondary');save(fig,'headroom_vs_budget')
    primary=result('development','Delta')
    sign=('positive:larger optimized headroom under cache pooling' if primary['CI95_lower']>0 else
          'negative:larger optimized headroom under unique pooling' if primary['CI95_upper']<0 else
          'inconclusive:the interval includes zero;this is not equivalence')
    lines=['# refresh_pooling_control_v1 findings','',
        f"**Primary development B=4 Delta = {fmt(primary)} nats/video.** The interval is {sign}. This is a frozen analysis on reused data,not independent preregistration or fresh confirmation.",
        '', '|Cohort|Budget|Cache headroom H_C|Unique headroom H_U|Delta H_C-H_U|','|---|---:|---|---|---|']
    for c in cohorts:
        for b in [2,4,8]:lines.append(f'|{c}|{b}|{fmt(result(c,"H_C",b))}|{fmt(result(c,"H_U",b))}|{fmt(result(c,"Delta",b))}|')
    lines+=['','Intervals are95% class-stratified whole-source percentiles. Exploratory40 and B=2/8 are predefined secondary,marginal/unadjusted. No budget,class or subgroup replaces the primary result. Positive oracle headroom follows from periodic being feasible;it is not evidence of a deployable policy.','',
        '## Same frames versus reoptimized choices','',
        '|Cohort|L_C(S_C)|L_U(S_C)|L_C(S_U)|L_U(S_U)|W_C (U-C at S_C)|W_U (U-C at S_U)|',
        '|---|---:|---:|---:|---:|---|---|']
    for c in cohorts:
        cells=[result(c,k)['mean'] for k in ['L_C_SC','L_U_SC','L_C_SU','L_U_SU']]
        lines.append(f'|{c}|'+''.join(f'{v:.6f}|' for v in cells)+f'{fmt(result(c,"W_C"))}|{fmt(result(c,"W_U"))}|')
        o=overlap[(overlap.cohort==c)&(overlap.budget==4)&(overlap['first']=='cache_task_oracle')&(overlap['second']=='unique_task_oracle')]
        lines.append('')
        lines.append(f"For {c},S_C/S_U mean Jaccard is {o.Jaccard.mean():.4f};{int(o.same_schedule.sum())}/{len(o)} schedules are identical. The cache-optimal fixed set has W_C {fmt(result(c,'W_C'))};after reoptimizing for U,the remaining unique headroom is {fmt(result(c,'H_U'))}. These are different comparisons. Changing pooling at S_C never silently substitutes S_U.")
    lines+=['','H_C=[L_U(P)-L_U(S_C)]+W_C is verified per source. The first term uses S_C under U;it is not H_U unless that schedule is also U-optimal. This is a path-dependent algebraic decomposition,not independently identified causal contributions or a percentage of semantic importance. Duration weights count candidate outputs,not seconds.','',
        '## B=4 recognition and feature-fidelity control','',
        '|Cohort|Selected schedule|Pool|CE nats/video|Top-1 / macro (%)|Last-cache cosine error|Effective weight count|',
        '|---|---|---|---:|---:|---:|---:|']
    for c in cohorts:
        for policy in u.POLICIES+['dense']:
            for pool in (['C=U'] if policy=='dense' else ['C','U']):
                r=metric(c,policy,pool,16 if policy=='dense' else 4)
                lines.append(f"|{c}|{NAMES[policy]}|{pool}|{r['CE']['mean']:.6f}|{100*r['top1_accuracy']:.2f} / {100*r['macro_top1_accuracy']:.2f}|{r['mean_last_cache_feature_error']:.6f}|{r['mean_effective_weight_count']:.3f}|")
        pc,pf=metric(c,'periodic','C'),metric(c,'feature_oracle','C')
        uf=metric(c,'feature_oracle','U')
        lines+=['',f"For {c},feature oracle reduces last-cache cosine error by {pc['mean_last_cache_feature_error']-pf['mean_last_cache_feature_error']:.6f},while its CE relative to periodic changes by {pf['CE']['mean']-pc['CE']['mean']:+.6f} under C and {uf['CE']['mean']-pc['CE']['mean']:+.6f} under U. Its accuracy is {100*pf['top1_accuracy']:.2f}%/{100*uf['top1_accuracy']:.2f}% (C/U),versus {100*pc['top1_accuracy']:.2f}% periodic. The original cache-pool feature-fidelity/task-quality mismatch reproduces;unique pooling is a separate frozen-head aggregation comparison."]
    lines+=['','RandomExact rows are expectations over every feasible individual schedule,not an ensemble. Its expected correct counts and entire outcome distributions are saved;accuracy is not inferred from averaged logits or probabilities. Feature error always describes the hypothetical16-output latest-cache timeline;it is not a reconstruction metric for U. Effective count is1/sum(w^2),a descriptive weight-concentration statistic,not effective sample size or encoder cost. Unique pooling produces one final classification from B unique features,not16 time-indexed predictions.','',
        '## Verification and scope','',
        'The primary400-training-video head,regularization0.01,class order,means,scales,weights and bias stay frozen. Exactly100 development and40 previously inspected exploratory sources are used,with16 float32 references each,cast to float64. Training arrays,protected confirmation outcomes/reference arrays/motion tables are not opened. Archived primary-head B=4 dense,periodic,feature-oracle and cache-task-oracle schedules/logits/losses/accuracy reproduce. Every selected schedule is independently cache-copied and unique-feature averaged;all966,700 feasible source/schedule outcomes are independently checked. Synthetic scalar exhaustive tests cover both minima,near ties,RandomExact,affine identities and the prediction-flip fixture. Saved per-source statistics and bootstrap endpoints are independently reproduced. See verification.json.','',
        'The chosen oracle is lexicographically first within1e-12 of the numerical minimum;minimum,achieved objective and tie count are all retained. CE minimization is label-guided hindsight and does not guarantee higher top-1 accuracy. Any nonnegative headroom/own-objective regret or exact affine identity is a computation check,not controller evidence.','',
        'The head was trained on dense mean features. Sparse C/U aggregations can create different representation distributions. These results isolate pooling/selection under that fixed head,not separately retrained best-case heads. Ground-truth labels and future reference candidates make both task oracles privileged/noncausal;the feature oracle also has dense references. No temporal localization evidence establishes semantic frame importance.','',
        'Bootstrap uses10,000 PCG64 seed20261023 resamples,linear percentiles,within each fixed class5 sources for development and2 for exploratory,with shared indices across budgets/pools/contrasts. Sources—not schedules or candidates—are independent resampling units. Cohorts stay separate. Intervals condition on the frozen head and chosen classes,exclude training/model-selection/repeated-research uncertainty,and are particularly limited by two exploratory sources/class. No gap ratios,post-hoc effect threshold or publication gate are used.','',
        '## Supported next experiment','',
        'Run the separately frozen temporal-backbone benchmark specified in next_temporal_backbone_protocol.md before another cached-feature controller variant. It uses one efficient temporal representation,valid complete snippets and matched snippet-call budgets,with pooling controls and genuinely new train/development/evaluation sources. No acquisition,backbone execution or fitting is launched here. The present experiment establishes no fresh generalization,streaming causality,successful adaptive controller,end-to-end speedup or publication acceptance.','',
        '## Reproduction','', 'From the repository root:','', '```powershell']
    prefix=rf'& .\.venv\Scripts\python.exe .\analyze_refresh_pooling_control.py --input {p["input"]} --output {output.relative_to(Path(__file__).resolve().parent).as_posix()} --budgets 2 4 8 --bootstrap 10000 --seed 20261023'
    lines += [r'& .\.venv\Scripts\python.exe -m unittest test_refresh_pooling_control -v',prefix+' --preflight-only',prefix,
        '# Inspect all four rendered figures,then explicitly finalize the visual review:',prefix+' --finalize-reviewed',prefix+' --audit-only','```','',
        'Completed invocations perform read-only audits;they do not rerun or overwrite scientific measurements. Changed inputs/protocol require a fresh versioned directory. Analysis runtime is an offline CPU analysis cost,not inference latency. Actual new downloads,decodes,encoder forwards,model/scaler/threshold fits and package upgrades are zero.']
    (output/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'README.md').write_text('# refresh_pooling_control_v1\n\n'+lines[2]+'\n\nSee [findings](findings.md),[frozen reused-data analysis protocol](protocol.json),[input provenance](input_provenance.json),[selected schedules](selected_schedules.json),[paired contrasts](paired_comparisons.csv),[intervals](bootstrap_intervals.json),[verification](verification.json),and [next benchmark design](next_temporal_backbone_protocol.md).\n',encoding='utf-8')
    (output/'next_temporal_backbone_protocol.md').write_text(NEXT_PROTOCOL,encoding='utf-8')


NEXT_PROTOCOL=r'''# Proposed next benchmark: one temporal backbone,matched snippet budgets

This is a concrete follow-up design,not an executed experiment or a claim that a controller is justified. Freeze the final checkpoint,software,dataset identities and complete protocol before collecting/evaluating new outcomes. Do not launch acquisition or training from this pooling diagnostic.

## Minimum first benchmark

Use **PyTorchVideo X3D-XS**,one efficient pretrained temporal backbone,without an architecture search. The official recipe uses four RGB samples and a nominal4x12 window at30fps,with182-pixel short-side/center crop and mean0.45/std0.225 normalization. Its published factory uses input_clip_length4 and input_crop_size160,while the hub inference transform documents182;this discrepancy must be resolved by a pinned-recipe forward-shape/dense-baseline check before acquisition. Do not silently alter the pretrained architecture. Sources:[official X3D inference recipe](https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/),[pinned factory](https://github.com/facebookresearch/pytorchvideo/blob/f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0/pytorchvideo/models/hub/x3d.py).

Proposed checkpoint URL is https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_XS.pyth. The exact checkpoint SHA256,license,compatible PyTorchVideo/torch versions,pre-logit output dimension and transformation details are **unverified prerequisites**,not identities certified by this experiment. Pin upstream commit f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0;verify official checkpoint loading and preprocessing in a separate environment without upgrading the existing stack. If compatibility/recipe cannot be verified,stop the follow-up rather than substituting another model after outcomes.

## Exact refresh/snippet unit and temporal support

One refresh is one complete X3D-XS forward on a four-frame snippet,not a single still frame and not one decoded video frame. On a documented30fps virtual sampling grid,a closed trailing window has48 grid positions ending at time u. Freeze UniformTemporalSubsample-style floor(linspace(0,47,4))=[0,15,31,47],giving timestamps[u-47/30,u-32/30,u-16/30,u]. This spans47/30seconds;4x12 names the nominal48-frame/1.6-second window,not four consecutive frames or an exact12-frame stride under this specified uniform subsampler. Resolve real VFR timestamps by nearest available PTS with deterministic earlier-frame tie handling;reject/record clips lacking valid distinct support rather than padding unavailable beginning frames.

All four actual sampled frames are available by u. The network can mix them noncausally **within that closed observed snippet**;it must have no access to video frames after u or other snippets. Internal convolution padding stays within the four-input computation. Validate the pinned model's actual temporal dependency with an impulse/prefix mutation fixture before claiming this support. Never repeat one stale RGB frame four times as if it were a valid temporal snippet. No hidden recurrent state crosses snippets.

Choose16 candidate snippet end-times uniformly over the interval[first valid complete-window end,last valid end],using complete-clip metadata and recording exact source timestamps. Candidate0 is the first valid snippet,so startup waits for the initial full temporal support. This remains offline clip selection;it does not establish a streaming gate. Codec reordering/DTS versus PTS and lookahead must be separately accounted for before any streaming claim.

Within this same backbone,compare dense16 snippet forwards with periodic B=2,4,8 snippet forwards and mandatory first refresh. Each forward processes four RGB inputs:nominally64 frame-input evaluations for dense versus8/16/32 for sparse,while overlapping decoded frames,unique pixels and cached decoder outputs are counted separately. Do not compare these counts directly to the old single-frame ResNet18 budgets. Cache only the complete snippet representation after each actual refresh;reset state per source/trial.

## Recognition and pooling controls

Freeze backbone weights,replace only its classification projection with an audited pre-logit feature interface,and train one linear20-class head on temporal dense-mean representations. Match the same head across dense and sparse policies. Training-only normalization;one predefined regularization grid chosen on development dense performance only;freeze before evaluation. No gate/controller in the minimum benchmark.

Validate dense recognition first against ground-truth labels,chance/majority baselines,class-wise confusion,CE,train/development gap and reproducible forward outputs. Verify the canonical pretrained400-class dense inference on a documented development-only reference subset to catch loading,preprocessing,label-map and temporal-sampling errors. If recognition remains weak,report that limitation and investigate that baseline before interpreting oracle headroom;do not invent an outcome-selected accuracy cutoff.

At matched within-backbone budgets,use periodic,uniform RandomExact and clearly privileged feature/cache-task/unique-task oracles as diagnostics. Retain both candidate-duration cache pooling and unique-snippet pooling. Cross-evaluate fixed schedules under both pools;do not conflate reoptimization with changing weights. Do not pool probabilities. Inspect whether temporal mean pooling still discards useful ordering before proposing a gate. Budget4 is primary;2/8 are fixed sensitivities;dense16 appears once. Label-guided oracle results remain hindsight selection bounds,not deployable accuracy.

## New-source roles,exclusions and pretraining overlap

Proposed minimum:400 new training videos(20/class),100 new development(5/class),100 new evaluation(5/class),same20 classes. Freeze deterministic ranking and source roles before feature collection;original parent video IDs and encoded hashes cannot cross roles. Exclude every source/hash in all prior experiment memberships,including the protected completed confirmation cohort,using membership identities only. Exclude adjacent windows from an already used parent. Preserve original encoding;record failures and deterministic eligibility replacements without looking at recognition outcomes.

X3D weights were trained on Kinetics-400. New to this repository is not necessarily absent from pretraining. Before data acquisition,obtain verified original checkpoint-training source manifests and compare parent IDs,windows and hashes where available. Prefer eligible sources outside all known pretraining memberships. If checkpoint membership is incomplete,record overlap uncertainty explicitly and do not label evaluation pretraining-clean. A stronger independence claim may require newly captured/licensed videos outside Kinetics;availability,annotation quality,rights and600-source feasibility are unresolved. Do not silently fall back to contaminated or duplicated clips to fill class quotas.

Fit only training data;development chooses the predefined head regularization and validates implementation. Freeze all settings and evaluation membership before reading evaluation outcomes. Evaluate once;no evaluation-based threshold,representation,seed or budget selection. Use paired class-stratified whole-source uncertainty,shared10,000 PCG64 resamples and saved indices. Report all fixed comparisons,negative outcomes,undefined cases and conditional uncertainty. Do not repeatedly add sources to obtain favorable intervals.

## Full inference-cost accounting and later scope

First reproduce quality with saved temporal references. Then,in a separate explicitly authorized timing stage,measure encoded bytes already in memory through completed final classification under one consistent hardware/software protocol. Include container open/close,demux/decode,reordering/snippet buffering,timestamp sampling,RGB conversion,preprocessing,H2D transfer,complete temporal forward,caching/pooling,head and any future controller. Synchronize CUDA around each complete trial;fixed decoder threads,batch1/dtype,numerical settings,two full warmups,five seeded interleaved measured repetitions. Measure disk loading and setup separately;keep model construction/checkpoint download and offline oracle search out of inference timing. Oracles are not efficient deployable comparators.

Record peak CPU/GPU memory,model residency,snippet-buffer storage,unique decoded frames,frame-input evaluations,actual forwards and controller overhead. GPU compute/decoder sharing,overlapping snippets,CPU contention and transfer costs must be measured rather than inferred from B. Report startup support delay and end-to-end live-stream response separately if later tested. No speedup follows from replay call counts alone.

This minimum is one backbone/one fixed20-class population with new train/development/evaluation sources. Eventual multi-dataset coverage,motion-sensitive datasets,unseen classes,robust streaming latency and a deployable causal task-aware gate belong to separately frozen later studies. Do not expand into them or acquire a large checkpoint/dataset in the present task.
'''
