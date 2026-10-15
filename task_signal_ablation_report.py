"""Plots and cautious findings from saved ablation predictions and metrics."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import task_signal_ablation_utils as u
from recognition_baseline_data import read
from task_aware_oracle_report import save


def label(name: str) -> str:
    return name+" (privileged)" if u.access(name)["privileged"] else name


def number(x, digits: int = 4) -> str:
    return "undefined" if x is None else f"{x:.{digits}f}"


def report(output: Path) -> None:
    aggregate = read(output/"aggregate_metrics.json")["configurations"]
    intervals = read(output/"bootstrap_intervals.json")["comparisons"]
    lookup = {(r["cohort"], r["configuration"]): r for r in aggregate}
    roles = ["development_oof", "exploratory"]
    folder = output/"figures"; folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    values = [lookup[role, name]["MAE"]["mean"] for role in roles for name in u.CONFIGS]
    for ax, role in zip(axes, roles):
        grid = np.array([lookup[role, name]["MAE"]["mean"] for name in u.CONFIGS]).reshape(3, 2)
        image = ax.imshow(grid, cmap="YlOrRd", vmin=min(values), vmax=max(values), aspect="auto")
        ax.set_xticks([0, 1], ["Linear ridge", "RBF kernel ridge"])
        ax.set_yticks([0, 1, 2], ["S: summaries", "C: cached class scores", "P: current RGB (privileged)"])
        ax.set_title(role)
        for i in range(3):
            for j in range(2):
                ax.text(j, i, f"{grid[i,j]:.4f}", ha="center", va="center",
                        color="white" if grid[i,j] > (min(values)+max(values))*.5 else "black")
    fig.subplots_adjust(left=.25, right=.87, wspace=.8, top=.8)
    cax = fig.add_axes([.9, .17, .02, .58])
    fig.colorbar(image, cax=cax, label="Equal-video MAE (nats), lower is better")
    fig.suptitle("Fixed 3-input x 2-predictor ablation; P is privileged")
    save(fig, folder, "mae_3x2_comparison.png")
    order = u.CONFIGS+u.BASELINES+u.REFERENCES
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, role in zip(axes, roles):
        for i, name in enumerate(order):
            r = lookup[role, name]["realized_gain"]
            color = "#8c42a3" if u.access(name)["privileged"] else "#2470a3"
            ax.errorbar(r["mean"], i, xerr=[[r["mean"]-r["CI95_lower"]], [r["CI95_upper"]-r["mean"]]],
                        fmt="o", capsize=3, color=color)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_yticks(range(len(order)), [label(n) for n in order])
        ax.set(title=role, xlabel="Mean realized CE gain (nats per independent decision)")
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Decision utility and 95% whole-video intervals\nNo sequential rollout; P can incur additional RGB computation")
    fig.tight_layout(rect=[0, 0, 1, .92]); save(fig, folder, "decision_utility_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, role in zip(axes, roles):
        helpful = np.array([lookup[role, n]["helpful_selected"] for n in order])
        harmful = np.array([lookup[role, n]["harmful_selected"] for n in order])
        neutral = np.array([lookup[role, n]["neutral_selected"] for n in order])
        ax.barh(range(len(order)), helpful, color="#2a9259", label="Helpful selected")
        ax.barh(range(len(order)), harmful, left=helpful, color="#c6473b", label="Harmful selected")
        ax.barh(range(len(order)), neutral, left=helpful+harmful, color="#999999", label="Neutral selected")
        ax.set_yticks(range(len(order)), [label(n) for n in order])
        ax.set(title=role, xlabel="Selected opportunities (not independent videos)")
    axes[0].invert_yaxis(); axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Helpful and harmful selected moves; fixed threshold >1e-12\nP/current RGB and actual-outcome reference are privileged")
    fig.tight_layout(rect=[0, 0, 1, .92]); save(fig, folder, "selected_opportunity_breakdown.png")
    lines = ["# task_signal_ablation_v2 findings", "",
        "This fixed 3-input x 2-predictor experiment reuses exactly 100 development and 40 exploratory sources, 900/360 saved signed counterfactual targets, the original five source-grouped class-stratified folds, and the frozen primary recognition head. The new seed 20261018 affects bootstrap bookkeeping only; fold assignment remains the saved seed-20261017 assignment.", "",
        "Both cohorts are developmental. Recognition regularization already used development labels, and exploratory outcomes were previously inspected. The completed fresh-confirmation cohort was not parsed. Actual RGB encoder executions, video decodes, downloads and recognition training calls are zero.", "",
        "## Inputs, probes and cost", "",
        "S is exactly the previous 16-dimensional group D. C adds 40 class-centered logits: cached feature logits and mean output logits for past candidates 0 through t-1 only. P adds 20 centered current-feature logits and is privileged. P is not a cheap gate: keeping the reference schedule after observing f_t can add an RGB call. Four-refresh counterfactual schedules do not establish matched deployment cost for P.", "",
        "Linear ridge uses training-only population standardization, sum squared residuals plus coefficient norm squared, alpha=1, and an unpenalized intercept. RBF kernel ridge uses exp(-squared distance/p), lambda=1, training-kernel/target centering and training-only query centering. Kernel width, regularization, features and seeds were not searched. Signed predictions are unclipped. Saved models include raw/standardized training rows, scaler and kernel statistics and coefficients.", "",
        "The target is one prescribed equal-budget schedule change: CE(periodic) minus CE(earlier single refresh), not an optimal action value. All compared schedules contain initial refresh 0, four refreshes and sixteen latest-cache outputs. The nine counterfactuals per source are not combined into a sequential policy.", ""]
    for role in roles:
        lines += [f"## {role}", "",
            "|Configuration|MAE (nats)|MSE (nats squared)|Spearman|Positive-gain AUROC|Realized gain (nats/decision)|Selected helpful/harmful/neutral|",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for name in u.CONFIGS+u.BASELINES:
            r = lookup[role, name]
            lines.append(f"|{label(name)}|{r['MAE']['mean']:.6f}|{r['MSE']['mean']:.6f}|{number(r['spearman']['spearman'])}|{number(r['positive_gain_AUROC']['AUROC'])}|{r['mean_realized_gain']:+.6f}|{r['helpful_selected']}/{r['harmful_selected']}/{r['neutral_selected']}|")
        for name in u.REFERENCES:
            r = lookup[role, name]
            lines.append(f"|{label(name)}|not a regression probe|not applicable|undefined|undefined|{r['mean_realized_gain']:+.6f}|{r['helpful_selected']}/{r['harmful_selected']}/{r['neutral_selected']}|")
        lines += ["", "|Configuration|Selected / opportunities|Selection fraction|Helpful / selected|Mean gain among selected (nats)|", "|---|---:|---:|---:|---:|"]
        for name in order:
            r = lookup[role, name]
            lines.append(f"|{label(name)}|{r['selected_count']}/{r['opportunities']}|{r['fraction_selected']:.4f}|{r['helpful_selected']}/{r['selected_count']} ({number(r['helpful_fraction_selected'])})|{number(r['mean_gain_selected'],6)}|")
        lines += [""]
    lines += ["## Primary and secondary comparisons", "",
        "Earlier refresh is chosen only when prediction exceeds 1e-12. Realized gain is actual signed gain times this indicator. Average nine decisions within a source first, then average sources. Always keeping the reference gives zero. The separate hindsight reference max(actual gain,0) knows the outcome and is not deployable.", "",
        "|Cohort|Candidate minus baseline comparison|Metric (positive favors candidate)|Mean improvement|95% interval|",
        "|---|---|---|---:|---|"]
    for r in intervals:
        lines.append(f"|{r['cohort']}|{label(r['candidate'])} vs {label(r['baseline'])}|{r['metric']}|{r['mean']:+.6f}|[{r['CI95_lower']:+.6f}, {r['CI95_upper']:+.6f}]|")
    lines += ["", "For MAE, improvement is baseline minus candidate. For decision utility, it is candidate minus baseline. The fixed primary is development OOF C-nonlinear versus always keeping reference. Exploratory is reported separately; secondary intervals are exploratory, marginal and unadjusted.", "",
        "All intervals use 10,000 paired whole-source-video resamples with PCG64 seed20261018, linear percentiles, and one shared index matrix across configurations within each cohort. Nine repeated moves stay together. Intervals condition on saved probes and omit full training, model-selection and repeated-research uncertainty. Spearman/AUROC are descriptive across correlated opportunities. One-class outcomes and constant scores have explicit null/reason, never fabricated discrimination.", "",
        "## Interpretation and next action", ""]
    for role in roles:
        primary = next(r for r in intervals if r["cohort"] == role and r["candidate"] == "C-nonlinear" and r["baseline"] == "keep_reference")
        status = "evidence of positive decision utility for this fixed comparison" if primary["CI95_lower"] > 0 else "evidence of harmful decision utility" if primary["CI95_upper"] < 0 else "inconclusive decision utility"
        lines.append(f"{role}: {status}; mean {primary['mean']:+.6f} nats/decision, interval [{primary['CI95_lower']:+.6f},{primary['CI95_upper']:+.6f}].")
        for c, b in u.CONTRASTS:
            mae = next(r for r in intervals if r["cohort"] == role and r["candidate"] == c and r["baseline"] == b and r["metric"] == "MAE")
            utility = next(r for r in intervals if r["cohort"] == role and r["candidate"] == c and r["baseline"] == b and r["metric"] == "realized_gain")
            lines.append(f"- {label(c)} versus {label(b)}: MAE improvement {mae['mean']:+.6f}; utility improvement {utility['mean']:+.6f}. These diagnose capacity, retained class-score information, or current appearance, not a newly selected winning method.")
    primary = next(r for r in intervals if r["primary"])
    if primary["CI95_lower"] > 0:
        recommendation = "Freeze C-nonlinear unchanged and confirm its single-decision utility on genuinely new sources before any sequential-controller study. Do not select a different configuration from the exploratory ranking."
    else:
        recommendation = "Do not build a sequential controller from an attractive regression score. Revisit the target's dependence on future content, data support and the static appearance representation before controller training. If privileged P has a useful contrast, investigate a genuinely inexpensive observation in a separate frozen study; P itself cannot be deployed as a cheap gate."
    lines += ["", recommendation, "",
        "Improved S-nonlinear would suggest interactions missed by the linear summary probe. C-over-S gains would suggest summaries discarded useful class-score information. Gains limited to P would suggest information missing from cached observations. No improvement would motivate revisiting target/representation/data. A single fixed RBF kernel cannot establish that all nonlinear models fail, and P is not an information-theoretic upper bound.", "",
        "Any promising result still needs new-source confirmation. Full-clip candidate selection and future-reference codec vectors keep this offline. No sequential recognition result, adaptive latency, speedup, or matched P deployment cost is measured.", "",
        "## Reproduction", ""]
    fence = chr(96)*3
    command = "& .\\.venv\\Scripts\\python.exe .\\analyze_task_signal_ablation.py --recognition-input results/recognition_baseline_v2 --diagnostic-input results/task_signal_diagnostic_v1 --output results/task_signal_ablation_v2 --seed 20261018"
    lines += [fence+"powershell", "& .\\.venv\\Scripts\\python.exe -m unittest test_task_signal_ablation -v",
              command, command+" --audit-only", fence, "",
              "Completed outputs are audited without overwriting. Incomplete runs resume only with matching frozen input/source/configuration identities. Previous experiments remain unchanged.", ""]
    (output/"findings.md").write_text("\n".join(lines), encoding="utf-8")
    (output/"README.md").write_text("# task_signal_ablation_v2\n\nSee [findings](findings.md), [protocol](protocol.json), [verification](verification.json), [metrics](aggregate_metrics.json), [paired intervals](bootstrap_intervals.json) and [frozen models](frozen_models.json).\n\nP is privileged current RGB. This is independent single-decision replay, not a sequential policy or matched-cost P deployment.\n", encoding="utf-8")
