"""Readable figures and findings from saved results; no fitting or selection."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"training_mean": "Training mean", "training_median": "Training median", "keep_reference": "Keep reference",
    "age_budget_ridge": "Age/budget ridge", "C_linear": "C linear", "C_nonlinear": "C kernel ridge",
    "C_linear_direct": "C linear direct G", "C_nonlinear_direct": "C kernel direct G",
    "Direct_G_RF200": "Direct G RF200", "Direct_G_RF400": "Direct G RF400",
    "Two_target_RF200x2_forward": "Two-target RF200x2 (forward)",
    "Two_target_RF200x2_averaged": "Two-target RF200x2 (averaged)"}


def ci(x: dict) -> str:
    return f"{x['mean']:+.6f} [{x['CI95_lower']:+.6f}, {x['CI95_upper']:+.6f}]"


def report(output: Path) -> None:
    aggregate = json.loads((output/"aggregate_metrics.json").read_text())
    intervals = json.loads((output/"bootstrap_intervals.json").read_text())
    checks = json.loads((output/"verification.json").read_text())
    decisions, quality = aggregate["decision_methods"], aggregate["prediction_quality"]
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    roles = ["development_oof", "exploratory"]
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(17, 12))
    for i, role in enumerate(roles):
        for j, target in enumerate(["B", "C", "G"]):
            axis = axes[i, j]
            methods = ["training_median", "training_mean", "C_linear", "C_nonlinear", "Two_target_RF200x2_forward"]
            if target == "G":
                methods = ["training_median", "training_mean", "age_budget_ridge", "C_linear", "C_nonlinear",
                           "Direct_G_RF200", "Direct_G_RF400", "Two_target_RF200x2_forward", "Two_target_RF200x2_averaged"]
            for k, name in enumerate(methods):
                r = next(r for r in quality if r["cohort"] == role and r["target"] == target and r["recipe"] == name)
                value = r["MAE_improvement_over_training_median"]
                axis.errorbar(value["mean"], k, xerr=[[value["mean"]-value["CI95_lower"]], [value["CI95_upper"]-value["mean"]]],
                              fmt="o", capsize=3, color="#236a9b" if "RF" in name else "#606060")
            axis.axvline(0, color="black", lw=.8, ls="--")
            axis.set_yticks(range(len(methods)), [LABELS.get(n, n) for n in methods])
            axis.invert_yaxis()
            axis.set_title(f"{role.replace('_', ' ')}: {dict(B='early-addition benefit', C='later-removal cost', G='net gain')[target]}")
            axis.set_xlabel("MAE improvement over own training median (nats)\nPositive = lower error; whole-video 95% interval")
            axis.grid(axis="x", alpha=.2)
    fig.suptitle("Component quality relative to each target's own constant baseline\nForward attribution primary; raw B/C errors have different scales", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(figures/"component_prediction_quality.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    methods = ["keep_reference", "training_mean", "training_median", "age_budget_ridge", "C_linear_direct",
               "C_nonlinear_direct", "Direct_G_RF200", "Direct_G_RF400", "Two_target_RF200x2_forward", "Two_target_RF200x2_averaged"]
    for axis, role in zip(axes, roles):
        for k, method in enumerate(methods):
            r = next(r for r in decisions if r["cohort"] == role and r["method"] == method)
            v = r["utility"]
            axis.errorbar(v["mean"], k, xerr=[[v["mean"]-v["CI95_lower"]], [v["CI95_upper"]-v["mean"]]], fmt="o", capsize=3,
                          color="#236a9b" if method == "Two_target_RF200x2_forward" else "#8550a1" if "averaged" in method else "#606060")
        axis.set_yticks(range(len(methods)), [LABELS.get(n, n) for n in methods])
        axis.invert_yaxis()
        axis.axvline(0, color="black", lw=.8, ls="--")
        axis.grid(axis="x", alpha=.2)
        axis.set_xlabel("Equal-video realized CE gain (nats / independent decision)\nPositive = helpful; whole-video 95% interval")
        axis.set_title(role.replace("_", " "))
    fig.suptitle("Fixed threshold > 1e-12; independent counterfactual moves, not a sequential policy\nBoth cohorts developmental; intervals conditional on saved fits", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(figures/"decision_utility.png", dpi=160)
    plt.close(fig)
    lines = ["# two_target_refresh_predictability_v1 findings", "",
        "This fixed saved-table diagnostic uses exactly 100 development sources/900 opportunities and 40 exploratory sources/360 opportunities. All nine observations from a source share its original fold. Both cohorts are developmental, not fresh confirmations: recognition regularization previously used development sources, and exploratory sources have been inspected. No RGB encoder, video decoder, video acquisition or recognition training runs. New fixed probes are fitted; the recognition head and prior artifacts remain unchanged.", "",
        "Primary inputs are the exact 56 C columns, never the privileged P columns. Targets B=LA-LB, C=LC-LB and G=LA-LC remain signed nats. Averaged attribution is one predeclared sensitivity analysis. The four schedules retain the old meanings and unequal-budget B/D controls. The labels are retrospective whole-video task outcomes, not online observations.", "",
        "## Decision quality", "",
        "Each independent move is selected only when predicted G>1e-12. Utility is actual G times selection, averaged over all nine opportunities within source (including zeros), then over sources. Keeping reference is zero. No independent moves are combined into a policy.", "",
        "|Cohort|Method|Utility and 95% interval (nats/decision)|Selected / total|Helpful / harmful / neutral|Mean selected gain (denominator)|",
        "|---|---|---|---:|---|---|"]
    for r in decisions:
        selected_mean = "undefined: no selections" if r["mean_gain_among_selected"] is None else f"{r['mean_gain_among_selected']:+.6f} ({r['selected_mean_denominator']})"
        lines.append(f"|{r['cohort']}|{r['method']}|{ci(r['utility'])}|{r['selected']}/{r['opportunities']}|{r['helpful']}/{r['harmful']}/{r['neutral']}|{selected_mean}|")
    lines += ["", "## Predeclared comparisons", "", "Positive paired differences favor the candidate. Primary is forward two-target RF200x2 minus direct-G RF400 on development OOF videos. Absolute utility must also be positive: a less negative result is not a useful policy. Secondary intervals are marginal and unadjusted.", "",
        "|Cohort|Candidate minus baseline|Mean paired gain and 95% interval (nats)|Improved / tied / worsened videos|",
        "|---|---|---|---|"]
    for r in intervals["comparisons"]:
        lines.append(f"|{r['cohort']}|{r['candidate']} minus {r['baseline']}|{ci(r)}|{r['videos_improved']}/{r['videos_tied']}/{r['videos_worsened']}|")
    primary = next(r for r in intervals["comparisons"] if r["cohort"] == "development_oof" and r["status"] == "primary_relative")
    absolute = next(r for r in decisions if r["cohort"] == "development_oof" and r["method"] == "Two_target_RF200x2_forward")["utility"]
    lines += ["", f"Development primary relative utility: {ci(primary)}. Absolute forward two-target utility: {ci(absolute)}."]
    if primary["CI95_lower"] > 0 and absolute["CI95_lower"] > 0:
        conclusion = "Both predeclared requirements have positive intervals. This supports a new-source confirmation of the frozen forward two-target method, not deployment."
        next_action = "Run a frozen new-source confirmation of this exact forward method and direct-G RF400 comparator, with no retuning."
    else:
        conclusion = "The predeclared joint requirements are not established. Do not claim successful decomposition-based decisions or that two targets are necessary."
        next_action = "Before another controller variant, freeze a task/temporal-representation diagnostic with greater independent-source support, testing whether richer legitimately cached temporal observations predict this same target beyond constant and direct-G baselines."
    lines += [conclusion, "", "## Component predictability", "",
        "Compare each target with its own training-median predictor. Raw B-MAE versus raw C-MAE is not an easier/harder comparison because scales differ. All means give sources equal weight. Pooled Spearman and positive-target AUROC are descriptive; nine opportunities per video are correlated. Constant scores or single-class outcomes are null with reasons, not invented values.", "",
        "|Cohort|Target|Recipe|MAE (nats)|Own median MAE|MAE improvement and 95% interval|MSE (nats squared)|Spearman|Positive-target AUROC|",
        "|---|---|---|---:|---:|---|---:|---|---|"]
    for r in quality:
        rho, auc = r["descriptive_pooled_spearman"], r["descriptive_positive_target_AUROC"]
        rs = str(rho["spearman"]) if rho["spearman"] is not None else "undefined: "+str(rho["undefined_reason"])
        az = str(auc["AUROC"]) if auc["AUROC"] is not None else "undefined: "+str(auc["undefined_reason"])
        lines.append(f"|{r['cohort']}|{r['target']}|{r['recipe']}|{r['MAE']['mean']:.6f}|{r['median_baseline_MAE']:.6f}|{ci(r['MAE_improvement_over_training_median'])}|{r['MSE']['mean']:.6f}|{rs}|{az}; events {auc['positive_events']}/{auc['negative_events']}|")
    lines += ["", "For forward B/C, the following comparisons have an entirely positive MAE-improvement interval over their own training median (absence is not proof of no possible signal):"]
    for role in roles:
        signals = [f"{r['target']}:{r['recipe']}" for r in quality if r["cohort"] == role and r["target"] in ["B", "C"] and r["MAE_improvement_over_training_median"]["CI95_lower"] > 0]
        lines.append(f"- {role}: {', '.join(signals) if signals else 'none of the fixed probes establishes a positive interval' }.")
    eq = checks["equivalence_checks"]
    lines += ["", "## Equivalence, prediction errors and resource accounting", "",
        f"Linear/kernel identities pass for both attributions and cohorts. Maximum heldout direct-G versus B-minus-C discrepancy is {eq['max_target_linearity_discrepancy']:.12g}; previous direct predictions reproduce within {eq['max_previous_prediction_discrepancy']:.12g}. These recipes are linear in training targets, including the input-nonlinear RBF estimator; component heads do not introduce another decision method mathematically.",
        f"Threshold decision disagreements across equivalence controls: {sum(r['decision_disagreements'] for r in eq['target_linearity'])}. Any disagreements must lie within the documented numerical equivalence band; they cannot support an algorithmic gain.", "",
        "Separate target-dependent forest partitions are confirmed. Equal total tree count does not guarantee equal node count, capacity, serialized bytes or latency. Compressed joblib sizes use protocol5/compress3 and include estimator state, not a latency measurement.", "",
        "|Final-development forest|Target|Trees|Nodes|Serialized bytes|", "|---|---|---:|---:|---:|"]
    for r in checks["forest_model_accounting"]:
        if r["bundle"] == "final_development":
            lines.append(f"|{r['recipe']}|{r['target']}|{r['trees']}|{r['nodes']}|{r['serialized_bytes']}|")
    lines += ["", "The signed net error is eB-eC. Its MSE includes the **uncentered** error cross-product; correlated component errors can cancel or reinforce. This is not a covariance-only decomposition.", "",
        "|Cohort|Method|MSE B|MSE C|Mean eB*eC|MSE G|Identity residual|", "|---|---|---:|---:|---:|---:|---:|"]
    for r in aggregate["component_error_identity"]:
        lines.append(f"|{r['cohort']}|{r['method']}|{r['MSE_B']:.6f}|{r['MSE_C']:.6f}|{r['mean_uncentered_error_cross_product']:.6f}|{r['MSE_G']:.6f}|{r['identity_absolute_residual']:.3g}|")
    lines += ["", "## Interpretation and next experiment", "",
        "Forward is primary; the averaged-attribution comparisons above are sensitivity results and never replace a losing primary method. Evaluate whether either attribution meets both relative and absolute requirements; similar component accuracy alone is insufficient. A difference between orders reflects nonlinear whole-video attribution, not a unique causal decomposition.",
        "Unknown true labels, future-content dependence, limited source support and the cached representation remain unresolved. This fixed study cannot rule out every predictor. Full-clip candidates and future-reference codec vectors prevent a verified streaming-causal interpretation. End-to-end latency and adaptive recognition accuracy are unmeasured; no adaptive speedup or publication-level novelty follows.", "",
        "Recommended single next experiment: "+next_action, "",
        f"Verification passes: {checks['models_and_training_only_scalers_checked']} model/scaler checks, {checks['forest_bootstrap_root_means_checked']} bootstrap-root checks, unchanged folds/targets/prior artifacts, saved prediction/metric reproduction and shared whole-source bootstrap reproduction. Actual RGB encoder executions, video decodes, video downloads and recognition training calls are zero. Optional scikit-learn dependencies were installed separately without changing existing package versions.", "",
        f"Intervals use {intervals['resamples']} PCG64 whole-source resamples, seed {intervals['seed']}, linear percentile convention. They condition on saved fitted probes and omit full training/model-selection/repeated-research uncertainty. Cohorts are separate and both previously inspected.", "",
        "## Reproduce from the repository root", "", "```powershell",
        "& .\\.venv\\Scripts\\python.exe -m pip install -r .\\requirements-two-target.txt",
        "& .\\.venv\\Scripts\\python.exe -m unittest test_two_target_refresh_predictability -v",
        "& .\\.venv\\Scripts\\python.exe .\\analyze_two_target_refresh_predictability.py --decomposition-input results/refresh_target_decomposition_v1 --ablation-input results/task_signal_ablation_v2 --output results/two_target_refresh_predictability_v1 --seed 20261020 --bootstrap 10000",
        "& .\\.venv\\Scripts\\python.exe .\\analyze_two_target_refresh_predictability.py --decomposition-input results/refresh_target_decomposition_v1 --ablation-input results/task_signal_ablation_v2 --output results/two_target_refresh_predictability_v1 --seed 20261020 --bootstrap 10000 --audit-only",
        "```", "", "A completed invocation audits without overwriting. Source/input identity mismatches block reuse; choose a clearly versioned fresh directory for a changed protocol. No old scientific artifact is modified."]
    (output/"findings.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (output/"README.md").write_text("# two_target_refresh_predictability_v1\n\nSee [findings](findings.md), [protocol](protocol.json), [verification](verification.json), [equivalence](equivalence_checks.json), [metrics](aggregate_metrics.json) and [intervals](bootstrap_intervals.json).\n\nFixed saved-table diagnostic; no sequential rollout, encoder execution or latency measurement.\n", encoding="utf-8")
