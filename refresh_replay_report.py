"""Developmental policy-quality figures and reports from actual replay results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from refresh_policy_utils import LEARNED, POLICIES, TOL
from research_utils import write_json

COLORS={"periodic":"#2878b5","random":"#888888","simple_motion":"#9b6abb",
        "age_only":"#ad7142","motion_only":"#2ca02c","age_plus_motion":"#f28e2b",
        "oracle":"#151515","dense":"#c43d3d"}


def create_report(output: Path, per_video: pd.DataFrame, trajectories: pd.DataFrame,
                  traces: pd.DataFrame, aggregate: dict, metadata: dict) -> list[dict]:
    summaries={(r["budget"],r["policy"]):r for r in aggregate["policy_metrics"]}
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(11,6))
    for policy in POLICIES:
        values=[summaries[b,policy]["mean_output_error"] for b in [2,4,8]]
        ax.plot([2,4,8],values,marker="o",label=policy.replace("_"," "),color=COLORS[policy],
                ls="--" if policy in ["oracle","random"] else "-")
    variability={r["budget"]:r for r in aggregate["random_seed_variability"]}
    ax.fill_between([2,4,8],[variability[b]["min_seed_equal_video_mean_error"] for b in [2,4,8]],
                    [variability[b]["max_seed_equal_video_mean_error"] for b in [2,4,8]],color=COLORS["random"],alpha=0.15,
                    label="random: range of ten seed-level video means")
    ax.scatter([16],[summaries[16,"dense"]["mean_output_error"]],marker="*",s=130,color=COLORS["dense"],label="dense reference (16)")
    ax.set_xticks([2,4,8,16])
    ax.set_xlabel("Simulated refresh count")
    ax.set_ylabel("Mean output-feature cosine error (equal weight per video)")
    ax.set_title("40 validation videos; globally validation-tuned thresholds")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2,fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"error_vs_refresh_budget.png",dpi=170)
    plt.close(fig)

    primary=per_video.loc[(per_video.budget==4)&(per_video.policy=="age_plus_motion")].sort_values(["absolute_reduction_vs_periodic","video_id"])
    periodic=per_video.loc[(per_video.budget==4)&(per_video.policy=="periodic")].set_index("video_id")
    ids=primary.video_id.tolist()
    x=np.arange(40)
    fig,axes=plt.subplots(2,1,figsize=(15,8),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    axes[0].plot(x,periodic.loc[ids,"mean_output_error"],marker="o",ms=3,lw=1,color=COLORS["periodic"],label="periodic")
    axes[0].plot(x,primary.mean_output_error,marker="o",ms=3,lw=1,color=COLORS["age_plus_motion"],label="age plus motion")
    axes[0].set_ylabel("Mean output-feature error")
    axes[0].set_title("B=4: developmental validation comparison, sorted by paired improvement")
    axes[0].legend()
    axes[1].bar(x,primary.absolute_reduction_vs_periodic,color=np.where(primary.absolute_reduction_vs_periodic>=0,"#2ca02c","#c43d3d"))
    axes[1].axhline(0,color="black",lw=0.8)
    axes[1].set_ylabel("Periodic error minus\ncombined-policy error")
    axes[1].set_xticks(x,[v[:11] for v in ids],rotation=90,fontsize=7)
    axes[1].set_xlabel("Validation source video")
    for ax in axes: ax.grid(axis="y",alpha=0.2)
    fig.tight_layout()
    fig.savefig(output/"per_video_B4.png",dpi=170)
    plt.close(fig)

    best=primary.sort_values(["absolute_reduction_vs_periodic","video_id"],ascending=[False,True]).iloc[0]
    worst=primary.iloc[0]
    examples=[]
    for role,row in [("largest_gain",best),("failure" if worst.absolute_reduction_vs_periodic < -TOL else "weakest_gain",worst)]:
        examples.append({"role":role,"video_id":row.video_id,"source_video_id":row.source_video_id,"label":row.label,
            "absolute_reduction_vs_periodic":float(row.absolute_reduction_vs_periodic),
            "selection":"B=4 age-plus-motion largest absolute gain/loss versus periodic, video-ID tie break; descriptive only"})
    write_json(output/"example_selection.json",{"examples":examples,"not_used_for_policy_tuning":True})
    fig,axes=plt.subplots(len(examples),2,figsize=(14,7),sharex=True)
    for i,example in enumerate(examples):
        for j,policy in enumerate(["periodic","age_plus_motion"]):
            group=traces.loc[(traces.video_id==example["video_id"])&(traces.budget==4)&(traces.policy==policy)].sort_values("candidate_ordinal")
            ax=axes[i,j]
            ax.plot(group.candidate_ordinal,group.output_feature_error,marker="o",color=COLORS[policy],label="post-decision output error")
            ax.plot(group.candidate_ordinal,group.pre_decision_true_drift,ls="--",lw=1,color="#777777",label="pre-decision true drift (evaluator only)")
            refresh=group.loc[group.refresh]
            ax.scatter(refresh.candidate_ordinal,refresh.output_feature_error,marker="*",s=120,color="black",label="actual simulated refresh")
            for ordinal in refresh.candidate_ordinal: ax.axvline(ordinal,color="black",lw=0.6,alpha=0.2)
            ax.set_title(f"{example['role']}: {example['source_video_id']} — {policy.replace('_',' ')}")
            ax.set_xticks(range(16))
            ax.set_xlabel("Candidate ordinal (offline)")
            ax.set_ylabel("Appearance-feature cosine error")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
    fig.suptitle("B=4 trajectories; refresh error approximately zero; reference features never enter policy callbacks")
    fig.tight_layout(rect=(0,0,1,0.94))
    fig.savefig(output/"example_timelines.png",dpi=170)
    plt.close(fig)

    def val(number,digits=6): return "undefined" if number is None else f"{number:.{digits}f}"
    lines=["# adaptive_refresh_v1: fixed-budget offline policy replay","",
        f"Actual replay began {metadata['started_at_utc']}. Saved references from 120 training videos and 40 validation videos were used; zero test videos were evaluated. "
        "The training table has 14,400 cache/current pairs (120 per source video), not 14,400 independent videos. All three ridge models use the same pairs, alpha=1.0, and training-only scalers.","",
        "The protocol and threshold procedure were frozen before validation evaluation. Training-score quantiles define the grid, and one global threshold per scored policy/budget minimizes equal-weight validation-video output error. "
        "Every grid setting's forty results are saved. Models are fitted once and remain unchanged. These are **validation-tuned developmental results**, not unbiased unseen-test results or a statistical-significance claim.","",
        "## Output-feature quality","",
        "Lower error means the feature actually returned after the refresh/reuse decision is closer to the current fresh reference. Mean errors include all sixteen output positions, with approximately zero error at refreshes. "
        "Pre-decision drift is a separate evaluator-only trace field. All sparse trajectories make exactly their stated number of simulated refreshes, including candidate 0; no actual encoder executes.","",
        "Random metrics average all ten seeds within each video before giving each video equal weight. Its maximum-error metric averages trajectory maxima; worst-seed error is also saved. Seed variability is descriptive, not a confidence interval. "
        "Percentage reductions and oracle-gap closure are calculated per video before aggregation; undefined denominators remain missing and negative outcomes remain negative.",""]
    for budget in [2,4,8]:
        lines += [f"### B={budget}","",
            "| Policy | Mean output error | Mean candidate maximum | Mean per-video reduction vs periodic | Improved / tied / worsened | Mean oracle-gap closure | Mean forced refreshes |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
        for policy in POLICIES:
            r=summaries[budget,policy]
            gap=100*r["mean_defined_oracle_gap_closure"] if r["mean_defined_oracle_gap_closure"] is not None else None
            lines.append(f"| {policy} | {r['mean_output_error']:.6f} | {r['mean_max_candidate_error']:.6f} | {val(r['mean_per_video_percentage_reduction'],2)}% | {r['improved']} / {r['tied']} / {r['worsened']} | {val(gap,2)}% | {r['mean_forced_refreshes']:.2f} |")
        periodic_error=summaries[budget,"periodic"]["mean_output_error"]
        oracle=summaries[budget,"oracle"]
        lines += ["",f"Oracle headroom: periodic {periodic_error:.6f} versus exact offline oracle {oracle['mean_output_error']:.6f}; "
            f"absolute equal-video mean headroom {periodic_error-oracle['mean_output_error']:.6f}, mean per-video possible reduction {val(oracle['mean_per_video_percentage_reduction'],2)}%. "
            f"Oracle-gap closure is undefined in {oracle['undefined_oracle_gap_videos']} videos at this budget.",""]
    lines += [f"Dense is a separate **B=16** reference: mean error {summaries[16,'dense']['mean_output_error']:.3e}; all sixteen positions refresh.","",
        "## Does adaptive refresh help?",""]
    for budget in [2,4,8]:
        combined=summaries[budget,"age_plus_motion"]
        lines.append(f"At B={budget}, age plus motion {'lowers' if combined['mean_absolute_reduction_vs_periodic']>TOL else 'raises' if combined['mean_absolute_reduction_vs_periodic']<-TOL else 'ties'} equal-video mean error versus periodic by "
            f"{abs(combined['mean_absolute_reduction_vs_periodic']):.6f}, improving {combined['improved']}/40 videos and worsening {combined['worsened']}/40. "
            f"It recovers {val(100*combined['mean_defined_oracle_gap_closure'] if combined['mean_defined_oracle_gap_closure'] is not None else None,2)}% of the per-video oracle gap on average.")
        for reference in ["random","simple_motion","motion_only","age_only"]:
            c=next(c for c in aggregate["paired_comparisons"] if c["budget"]==budget and c["candidate"]=="age_plus_motion" and c["reference"]==reference)
            lines.append(f"Against {reference}, combined-policy absolute mean reduction is {c['mean_absolute_reduction']:+.6f}: {c['improved']} improved / {c['tied']} tied / {c['worsened']} worsened videos.")
    lines += ["","Adding age beyond motion-only must be assessed as a policy comparison above; prediction MAE does not directly determine refresh quality. "
        "All-pair refitting changes the training cache-history distribution. Sequential decisions also change future ages and intervals, so the previous periodic-label prediction ranking is not a policy ranking.","",
        "Mean absolute reduction, mean per-video percentage reduction, and mean oracle-gap closure use different denominators. "
        "They can have different signs when gains and losses occur in videos with different baseline errors or oracle headroom. "
        "A ratio of aggregate errors is not the mean of per-video percentages; small oracle gaps can make individual gap-closure values very negative. These values are preserved.","",
        "## Forced spending and variability","",
        "The required rule spends the remaining budget at the end when remaining slots equal remaining refreshes. It is part of every sparse policy; forced counts are recorded separately from threshold-triggered refreshes.",""]
    for budget in [2,4,8]:
        for policy in [*LEARNED,"simple_motion"]:
            r=summaries[budget,policy]
            lines.append(f"{policy}, B={budget}: mean {r['mean_forced_refreshes']:.2f} forced refreshes; {100*r['fraction_trajectories_with_forced_refresh']:.1f}% of trajectories have at least one; "
                f"{100*r['mean_forced_share_of_nonmandatory_refreshes']:.1f}% of nonmandatory refreshes are forced.")
        variability=next(r for r in aggregate["random_seed_variability"] if r["budget"]==budget)
        lines.append(f"Random B={budget} equal-video mean error ranges {variability['min_seed_equal_video_mean_error']:.6f}–{variability['max_seed_equal_video_mean_error']:.6f} across all ten fixed seeds.")
    lines += ["","## Recommendation","",
        "Oracle results show the fidelity improvement achievable by ideal offline scheduling under this setup. A large oracle gap with weak learned-policy recovery points to a predictor/decision-rule problem; small headroom limits the opportunity regardless of predictor quality.",""]
    primary=summaries[4,"age_plus_motion"]
    if primary["mean_absolute_reduction_vs_periodic"]>TOL and primary["mean_per_video_percentage_reduction"]>0 and primary["improved"]>=28:
        lines.append("The B=4 combined policy improves validation feature fidelity in most videos, supporting a **fresh-test replay evaluation with these models and thresholds frozen**. "
            "Do not retune using test outcomes. Actual execution timing and a recognition probe are separate follow-ups after generalization is checked; neither speedup nor maintained recognition accuracy is established here.")
    else:
        lines.append("The combined policy provides **no consistent improvement over periodic refresh** across the developmental results. "
            f"At the primary B=4, the equal-video mean absolute change is {primary['mean_absolute_reduction_vs_periodic']:+.6f}, "
            f"but mean per-video percentage change is {primary['mean_per_video_percentage_reduction']:+.2f}% and {primary['worsened']}/40 videos worsen. "
            "The sizable oracle headroom and weak recovery point to the predictor/decision rule, rather than a lack of possible scheduling gains. "
            "A fresh-test replay can audit this fixed negative/mixed result if its thresholds remain frozen; it is not a promotion to a successful controller. "
            "**Do not proceed directly to actual GPU timing or a recognition probe for this rule.** Inspect budget allocation, exhaustion, and forced spending using training or new development data first. "
            "Do not repeatedly tune these same validation videos until the result becomes positive.")
    lines += ["","No regression, feature definition, budget rule, or threshold-grid procedure was retuned after observing these validation outcomes.","",
        "## Figures and examples","",
        "![Error versus simulated refresh budget](error_vs_refresh_budget.png)","",
        "![B=4 per-video comparison](per_video_B4.png)","",
        "![Actual refresh timelines](example_timelines.png)",""]
    for example in examples:
        lines.append(f"Example {example['role']}: {example['video_id']} ({example['label']}), absolute B=4 reduction {example['absolute_reduction_vs_periodic']:+.6f}. "
                     "Selected by the frozen largest-gain/largest-loss rule, with video-ID tie break, for description only.")
    lines += ["","## Limits and reproduction","",
        "Thresholds were selected on these same forty validation videos; this is developmental evidence. Appearance-feature fidelity is not action-recognition accuracy. "
        "Replay measures no codec, motion-processing, controller, transfer, or GPU execution latency. Counts are simulated refreshes/logical encoder calls, not actual encoder executions. "
        "Future-reference codec vectors and full-clip candidate selection make this offline, despite sequential access to current/preceding motion descriptors. "
        "The oracle uses all reference features and is an unavailable offline diagnostic bound. Sixteen low-resolution candidates, one appearance encoder, and twenty source classes limit generalization. "
        "Candidate observations and random seeds are correlated within each of forty videos; no confidence intervals or significance claims are made.","",
        "Run from the repository root in the existing environment:","","```powershell",
        "& .\\.venv\\Scripts\\python.exe .\\replay_refresh_policies.py --input results/cache_drift_v2 --output results/adaptive_refresh_v1 --budgets 2 4 8 --seed 20261013",
        "```","",
        "Completed/existing outputs are preserved by choosing a fresh directory. For saved-CSV audits use `float_precision='round_trip'`. "
        "See `protocol.json`, `metadata.json`, `train_pairs.csv`, `fitted_predictors.json`, `threshold_grid.json`, `threshold_search.csv`, "
        "`selected_thresholds.json`, `policy_traces.csv`, `validation_trajectories.csv`, `validation_per_video_metrics.csv`, "
        "`validation_pairwise_comparisons.csv`, `validation_aggregate_metrics.json`, `oracle_schedules.json`, and `verification.json`.",""]
    (output/"findings.md").write_text("\n".join(lines),encoding="utf-8")
    return examples
