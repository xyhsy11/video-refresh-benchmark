"""Readable diagnostic findings and figures, generated only from saved results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recognition_baseline_data import read
from task_aware_oracle_report import save

LABELS = {"zero": "Constant zero", "training_mean": "Training mean", "A": "A: age/budget",
          "B": "B: + motion", "C": "C: + cached prediction", "D": "D: both"}
COLORS = {"both_correct": "#28844c", "dense_correct_policy_wrong": "#bf3b32",
          "dense_wrong_policy_correct": "#2063b2", "both_wrong": "#858585"}


def fmt(value, digits: int = 4) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


def report(output: Path) -> None:
    summary = read(output/"aggregate_metrics.json")
    intervals = read(output/"bootstrap_intervals.json")["comparisons"]
    clips = pd.read_csv(output/"clip_error_analysis.csv", float_precision="round_trip")
    examples = pd.read_csv(output/"counterfactual_examples.csv", float_precision="round_trip")
    folder = output/"figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for i, role in enumerate(["development", "exploratory"]):
        for j, policy in enumerate(["periodic", "value_age_motion_budget"]):
            ax = axes[i, j]
            for category, color in COLORS.items():
                g = clips[(clips.cohort == role) & (clips.policy == policy) & (clips.transition == category)]
                ax.scatter(g.feature_cosine_error, g.delta_CE, s=25, color=color, alpha=.75,
                           label=category.replace("_", " "))
            ax.axhline(0, color="black", linewidth=.6)
            ax.set(title=f"{role}: {policy}", xlabel="Mean candidate-wise feature cosine error",
                   ylabel="Policy CE minus dense CE (nats)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Classification transitions under four-refresh caching")
    fig.tight_layout(rect=[0, .09, 1, .95])
    save(fig, folder, "feature_drift_vs_loss_change.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, role in zip(axes, ["development", "exploratory"]):
        g = examples[examples.cohort == role]
        ax.scatter(g.privileged_current_drift, g.refresh_gain, s=13, alpha=.5)
        ax.axhline(0, color="black", linewidth=.6)
        ax.set(title=f"{role}: nine prescribed moves/video", xlabel="True current feature drift (privileged)",
               ylabel="CE(periodic) minus CE(earlier refresh), nats")
    fig.suptitle("True drift versus equal-budget refresh gain; evaluator information")
    fig.tight_layout(rect=[0, 0, 1, .92]); save(fig, folder, "true_drift_vs_refresh_gain.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = list(LABELS)
    for ax, role in zip(axes, ["development_oof", "exploratory"]):
        rows = {r["predictor"]: r for r in summary["probe_metrics"] if r["cohort"] == role}
        values = np.array([rows[n]["equal_video_MAE"] for n in order])
        low = np.array([rows[n]["MAE_mean_CI95_lower"] for n in order])
        high = np.array([rows[n]["MAE_mean_CI95_upper"] for n in order])
        ax.errorbar(range(6), values, yerr=np.array([values-low, high-values]), fmt="o", capsize=4)
        ax.set_xticks(range(6), [LABELS[n] for n in order], rotation=35, ha="right")
        ax.set(title=role, ylabel="Equal-video mean absolute error (nats)")
    fig.suptitle("Probe prediction error; 95% whole-video bootstrap intervals\nConditional on saved probes; developmental cohorts")
    fig.tight_layout(rect=[0, 0, 1, .88]); save(fig, folder, "probe_prediction_error.png")

    lines = ["# task_signal_diagnostic_v1 findings", "",
        "This offline diagnostic uses the frozen 400-video-trained linear recognition head and existing saved ResNet18 references. It evaluates 100 development videos and 40 previously inspected exploratory videos separately. The head previously used development data for regularization selection. Neither cohort is a fresh independent confirmation. No completed fresh-confirmation arrays or results were parsed.", "",
        "Actual RGB encoder executions, video decodes, downloads and recognition-classifier training calls are all zero. Dense uses sixteen candidate refreshes, not every decoded frame. Sparse replay uses four simulated refreshes and sixteen latest-cache outputs. End-to-end latency is unmeasured.", "",
        "## Caching and classification changes", "",
        "A policy mistake is not automatically caused by caching. Harmful transitions are restricted to dense-correct sources that become wrong; prediction disagreement is recorded separately.", "",
        "|Cohort|Policy|Correct / videos|Mean CE (nats)|Mean feature error|Disagreements|Harmful / dense correct|",
        "|---|---|---:|---:|---:|---:|---:|"]
    for (role, policy), g in clips.groupby(["cohort", "policy"], sort=True):
        eligible = g[g.dense_correct == 1]
        harmful = int((eligible.policy_correct == 0).sum())
        lines.append(f"|{role}|{policy}|{g.policy_correct.sum()}/{len(g)}|{g.policy_CE.mean():.6f}|{g.feature_cosine_error.mean():.6f}|{g.prediction_disagreement.sum()}|{harmful}/{len(eligible)}|")
    lines += ["", "|Cohort|Policy|Spearman: feature error vs delta CE|Feature error vs policy wrong AUROC|",
              "|---|---|---:|---:|"]
    for r in summary["clip_error_analysis"]:
        lines.append(f"|{r['cohort']}|{r['policy']}|{fmt(r['feature_error_vs_delta_CE']['spearman'])}|{fmt(r['absolute_classification_error_AUROC']['AUROC'])}|")
    lines += ["", "These correlations are calculated separately for each policy/cohort. Dense and oracle observations are not pooled into them. The AUROC event is a policy classification error, which includes errors already present under dense inference. Conditional harmful-transition AUROCs and event counts are saved separately; one-class outcomes produce null plus a reason.", "",
        "## Equal-budget target and probes", "",
        "Each source supplies nine replacements: move planned refresh 4, 8 or 12 earlier by one, two or three candidate positions. Gain is periodic CE minus early-schedule CE in nats. Histories are identical before the decision, and both schedules use exactly four refreshes. This is one prescribed counterfactual change, not a globally optimal action value.", "",
        "A uses t/15, elapsed seconds since the cached refresh, remaining refresh count and ordinal distance to the next planned refresh. B adds the existing six motion summaries; C adds cached and past-output logit entropy/max-probability/top-two probability margin; D adds both. Interval descriptors use decoded frames strictly after the cached position through the current candidate. Availability flags/fractions distinguish missing vectors from observed zero displacement. Confidence is never conditioned on the true label.", "",
        "Ridge alpha is fixed at 1 for the sum-squared-residual objective, with training-only population standardization and an unpenalized intercept. Predictions are signed and unclipped. Five source-grouped class-stratified folds provide development OOF predictions. Only after saving all OOF predictions are probes fitted on all development sources, frozen, and evaluated on exploratory sources. No grid or seed selection was performed.", "",
        "|Cohort|Videos / examples|Beneficial / neutral / harmful moves|Privileged drift vs gain Spearman|",
        "|---|---:|---:|---:|"]
    for r in summary["privileged_drift_vs_gain"]:
        role = "development_oof" if r["cohort"] == "development" else r["cohort"]
        events = next(x for x in summary["probe_metrics"] if x["cohort"] == role and x["predictor"] == "A")
        lines.append(f"|{r['cohort']}|{r['videos']}/{r['examples']}|{events['beneficial_moves']}/{events['neutral_moves']}/{events['harmful_moves']}|{fmt(r['spearman'])}|")
    lines += ["", "All signed gains enter regression; only values within 1e-12 of zero are called floating-point ties for sign/count reporting. True current drift requires the uncomputed RGB feature and is a privileged diagnostic, never a predictor input.", "",
        "|Cohort|Probe|Equal-video MAE (nats)|Predicted vs gain Spearman|Sign accuracy|Positive-gain AUROC|",
        "|---|---|---:|---:|---:|---:|"]
    for r in summary["probe_metrics"]:
        lines.append(f"|{r['cohort']}|{LABELS[r['predictor']]}|{r['equal_video_MAE']:.6f}|{fmt(r['descriptive_example_spearman']['spearman'])}|{100*r['sign_accuracy']:.2f}%|{fmt(r['positive_gain_AUROC']['AUROC'])}|")
    lines += ["", "Spearman in this table is descriptive across repeated examples within each cohort. Per-source correlations, undefined counts, RMSE, sign macro recall and positive precision/recall are also saved. Undefined correlations are missing, not zero.", "",
        "## Primary paired uncertainty", "",
        "|Cohort|MAE(A) minus MAE(D), nats|95% percentile interval|Videos improved / tied / worsened|",
        "|---|---:|---|---:|"]
    for r in intervals:
        lines.append(f"|{r['cohort']}|{r['mean']:+.6f}|[{r['CI95_lower']:+.6f}, {r['CI95_upper']:+.6f}]|{r['improved']}/{r['tied']}/{r['worsened']}|")
    lines += ["", "Intervals use 10,000 PCG64 whole-source-video resamples, seed 20261017, with linear percentiles and shared indices across probes within each cohort. All nine examples remain together. They condition on saved fitted probes; overlapping OOF training folds and full training/model-selection uncertainty are not modeled. The exploratory comparison is separate and previously inspected.", "",
        "## Answers and next decision", "",
        "1. Feature error is not a sufficient explanation of harmful recognition changes. The policy-specific drift/loss correlations and conditional transition counts above distinguish additional caching errors from dense errors; correlation does not establish causation.",
        "2. The privileged drift/gain correlations above test association for the nine prescribed moves only. They do not make true drift a cheap trigger or establish a global action-value signal."]
    for role in ["development_oof", "exploratory"]:
        rows = {r["predictor"]: r for r in summary["probe_metrics"] if r["cohort"] == role}
        delta = lambda left, right: rows[left]["equal_video_MAE"]-rows[right]["equal_video_MAE"]
        lines += [f"3–4. {role}: cached signals change MAE by {delta('A','C'):+.6f} nats versus A and {delta('B','D'):+.6f} versus B. Motion changes MAE by {delta('A','B'):+.6f} versus A and {delta('C','D'):+.6f} versus C. Positive values favor the added signals. These component comparisons are descriptive, not independently tuned tests."]
    primary, secondary = (next(r for r in intervals if r["cohort"] == role) for role in ["development_oof", "exploratory"])
    status = lambda r: "lower MAE for D" if r["CI95_lower"] > 0 else "higher MAE for D" if r["CI95_upper"] < 0 else "inconclusive D-versus-A difference"
    lines += [f"5. Development: {status(primary)}. Exploratory: {status(secondary)}. Every predefined probe and constant baseline is reported, including negative outcomes. Improvement over A alone does not establish useful prediction if constants remain competitive.",
        "6. The next single hypothesis should be whether cached prediction uncertainty can identify beneficial early refreshes better than an age-only and constant baseline on a separate, frozen development cohort. Do not assemble the nine independent counterfactual predictions into a claimed sequential controller: later cache states would change. If the present probes fail, these particular signals and linear probes have not demonstrated useful prediction; that does not rule out every task-aware policy.", "",
        "Candidate indices use the full clip and codec vectors can reference future frames. Prefix descriptor access therefore remains an offline diagnostic, not strict streaming causality. Ground-truth labels and full RGB references appear only in targets/evaluation, never the inexpensive feature builder. Appearance/recognition probes do not establish maintained recognition for an adaptive controller, actual latency, or speedup.", "",
        "## Reproduction", ""]
    fence = chr(96)*3
    lines += [fence+"powershell",
        "& .\\.venv\\Scripts\\python.exe -m unittest test_task_signals -v",
        "& .\\.venv\\Scripts\\python.exe .\\analyze_task_signals.py --input results/recognition_baseline_v2 --output results/task_signal_diagnostic_v1 --seed 20261017",
        "& .\\.venv\\Scripts\\python.exe .\\analyze_task_signals.py --input results/recognition_baseline_v2 --output results/task_signal_diagnostic_v1 --seed 20261017 --audit-only",
        fence, "", "Completed outputs are audited rather than overwritten. The protocol freezes input/source identities and folds before outcome comparisons. No acquisition, reference collection, recognition fitting or fresh-confirmation evaluation is permitted.", ""]
    (output/"findings.md").write_text("\n".join(lines), encoding="utf-8")
    (output/"README.md").write_text("# task_signal_diagnostic_v1\n\nSee [findings](findings.md), [protocol](protocol.json), [verification](verification.json), [metrics](aggregate_metrics.json) and [paired intervals](bootstrap_intervals.json).\n\nSaved-feature analysis only: zero RGB encoder calls, no measured end-to-end latency.\n", encoding="utf-8")
