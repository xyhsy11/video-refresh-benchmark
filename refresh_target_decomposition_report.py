"""Figures, measured examples and descriptive attribution findings."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import refresh_target_decomposition_utils as u
from fresh_confirmation_data import strict_json
from recognition_baseline_data import read
from task_aware_oracle_report import save


def numeric(value, digits: int = 6) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


def rate_text(rate: dict) -> str:
    return f"{rate['numerator']}/{rate['denominator']} ({100*rate['value']:.2f}%)" if rate["value"] is not None else f"{rate['numerator']}/{rate['denominator']} (undefined: zero denominator)"


def interval_text(value: dict, percent: bool = False) -> str:
    scale = 100 if percent else 1
    if value["CI95_lower"] is None:
        return "undefined"
    return f"[{scale*value['CI95_lower']:.4f}, {scale*value['CI95_upper']:.4f}]"+("%" if percent else "")


def measured_examples(table: pd.DataFrame) -> list[dict]:
    ordered = table.sort_values(["cohort", "source_video_id", "r", "t"], kind="stable")
    candidates = [
        ("beneficial_move", ordered[ordered.net_gain > u.SIGN_TOL].sort_values("net_gain", ascending=False, kind="stable"),
         "largest positive net gain"),
        ("harmful_despite_beneficial_addition", ordered[ordered.forward_cancellation].sort_values("net_gain", kind="stable"),
         "most negative net among forward cancellations"),
        ("attribution_sensitive", ordered[ordered.benefit_sign_changed | ordered.cost_sign_changed | ordered.cancellation_order_changed]
             .assign(absolute_interaction=lambda g: abs(g.order_interaction)).sort_values("absolute_interaction", ascending=False, kind="stable"),
         "largest absolute interaction among sign/cancellation-sensitive opportunities")]
    used, selected = set(), []
    fields = ["cohort", "example_id", "video_id", "source_video_id", "class", "t", "r",
              *[f"schedule_{n}" for n in u.COUNTS], *[f"loss_{n}" for n in u.COUNTS],
              "net_gain", "early_addition_benefit", "later_removal_cost",
              "addition_after_removal_benefit", "removal_first_cost",
              "average_addition_benefit", "average_removal_cost", "order_interaction",
              "forward_cancellation", "reverse_cancellation", "averaged_cancellation",
              "benefit_sign_changed", "cost_sign_changed", "C_nonlinear_selected"]
    for name, g, rule in candidates:
        if not len(g):
            selected.append({"category": name, "rule": rule, "example": None, "reason": "no qualifying measured opportunity"})
            continue
        unique = g[~g.example_id.isin(used)]
        row = (unique if len(unique) else g).iloc[0]
        example = {key: row[key].item() if isinstance(row[key], np.generic) else row[key] for key in fields}
        selected.append({"category": name, "rule": rule, "example": example, "reused_example": row.example_id in used})
        used.add(row.example_id)
    return selected


def report(output: Path) -> None:
    table = pd.read_csv(output/"per_opportunity_decomposition.csv", float_precision="round_trip")
    aggregates = read(output/"aggregate_metrics.json")["rows"]
    verification = read(output/"verification.json")
    lookup = {(r["cohort"], r["scope"], r["order"]): r for r in aggregates}
    roles = ["development", "exploratory"]
    folder = output/"figures"; folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, role in zip(axes, roles):
        g = table[table.cohort == role]
        for mask, color, label in [(g.net_gain > u.SIGN_TOL, "#26804e", "Beneficial move"),
                                   (g.net_gain < -u.SIGN_TOL, "#c84a40", "Harmful move"),
                                   (abs(g.net_gain) <= u.SIGN_TOL, "#888888", "Floating-point tie")]:
            x = g[mask]
            ax.scatter(x.early_addition_benefit, x.later_removal_cost, color=color, alpha=.55, s=18, label=label)
        selected = g[g.C_nonlinear_selected]
        ax.scatter(selected.early_addition_benefit, selected.later_removal_cost, facecolors="none",
                   edgecolors="black", linewidths=.5, s=32, label="Saved C-nonlinear selection")
        values = np.r_[g.early_addition_benefit, g.later_removal_cost]
        lo, hi = float(values.min()), float(values.max())
        pad = max((hi-lo)*.06, .01); lo -= pad; hi += pad
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=.8)
        ax.axvline(0, color="#777777", linewidth=.4); ax.axhline(0, color="#777777", linewidth=.4)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), title=f"{role}: {len(g)} opportunities / {g.source_video_id.nunique()} sources",
               xlabel="Early-addition benefit (cross-entropy change, nats)",
               ylabel="Later-removal cost (cross-entropy change, nats)")
        ax.set_aspect("equal", adjustable="box")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("Forward attribution: points above y=x have negative net gain\nB uses five refreshes; addition/removal components are unequal-budget controls")
    fig.tight_layout(rect=[0, .10, 1, .90]); save(fig, folder, "benefit_vs_cost.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, scope in enumerate(["all", "C_nonlinear_selected"]):
        for j, role in enumerate(roles):
            ax = axes[i, j]
            for shift, metric, color, label in [(-.12, "mean_benefit", "#2177a6", "Benefit"),
                                               (.12, "mean_cost", "#bc5741", "Cost")]:
                for k, order in enumerate(u.ORDERS):
                    v = lookup[role, scope, order]["continuous"][metric]
                    if v["value"] is not None:
                        ax.errorbar(k+shift, v["value"],
                            yerr=[[v["value"]-v["CI95_lower"]], [v["CI95_upper"]-v["value"]]],
                            fmt="o", color=color, capsize=4, label=label if k == 0 else None)
            ax.axhline(0, color="black", linewidth=.7)
            ax.set_xticks(range(3), list(u.ORDERS))
            ax.set(title=f"{role}: {scope}", ylabel="Equal-source mean CE change (nats)")
            ax.legend(fontsize=8)
    fig.suptitle("Signed whole-video components with 95% whole-source-video intervals\nSelected-subset means exclude sources with no selected opportunities; counts are reported")
    fig.tight_layout(rect=[0, 0, 1, .9]); save(fig, folder, "component_summary.png")
    examples = measured_examples(table)
    strict_json(output/"measured_examples.json", {"selection_rules_frozen_in_protocol": True,
        "not_representative": "descriptive extremes;ties by cohort/source/r/t;prefer distinct examples", "examples": examples})
    lines = ["# refresh_target_decomposition_v1 findings", "",
        "This retrospective diagnostic uses the unchanged frozen primary recognition head, saved candidate features, 100 development sources and 40 exploratory sources. There are nine prescribed opportunities per source (900/360), not 900/360 independent videos. Both cohorts are developmental; neither is an untouched confirmation. No predictor or recognition head was fitted, and actual encoder executions, video decodes and downloads are zero.", "",
        "## Four controls and signed attribution", "",
        "A is [0,4,8,12]. B adds t while retaining r (five refreshes); C replaces r with t (four); D removes r without adding t (three). All produce sixteen latest-cache feature outputs. B and D are unequal-budget diagnostic controls, never equal-cost deployment improvements.", "",
        "Forward benefit is LA-LB and cost is LC-LB. Reverse benefit is LD-LC and cost is LD-LA. The two-order averages preserve net gain LA-LC. All components remain signed: a negative removal cost means removing that refresh lowers loss in that context. Cross entropy is nonlinear, so neither ordering is a unique causal explanation. Even the early-addition component depends on the whole-video prediction context and true label; it is not an observable online signal.", "",
        "## Primary and conditional cancellation rates", "",
        "Cancellation means benefit >1e-12 and net gain < -1e-12. The primary is the forward count divided by all opportunities, separately in each cohort. Reverse and averaged rows are secondary attribution checks.", "",
        "|Cohort|Scope|Order|Cancellation / opportunities|95% interval (%)|Cancellation / harmful moves|Cancellation / beneficial additions|",
        "|---|---|---|---:|---|---:|---:|"]
    for r in aggregates:
        rates = r["rates"]
        lines.append(f"|{r['cohort']}|{r['scope']}|{r['order']}|{rate_text(rates['cancellation_all'])}|{interval_text(rates['cancellation_all'],True)}|{rate_text(rates['cancellation_among_harmful'])}|{rate_text(rates['cancellation_among_beneficial_additions'])}|")
    lines += ["", "All counts/denominators and conditional intervals are saved. Bootstrap numerators and denominators are recomputed after sampling whole source videos; zero denominators are undefined, with invalid-resample counts and null intervals if all are invalid. No conditional rate is imputed as zero.", "",
        "## Signed means and absolute component sizes", "",
        "|Cohort|Scope|Order|Defined component sources / all sources|Mean benefit|Mean cost|Mean absolute benefit|Mean absolute cost|",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in aggregates:
        c = r["continuous"]
        lines.append(f"|{r['cohort']}|{r['scope']}|{r['order']}|{c['mean_benefit']['defined_sources']}/{r['source_videos']}|{numeric(c['mean_benefit']['value'])}|{numeric(c['mean_cost']['value'])}|{numeric(c['mean_abs_benefit']['value'])}|{numeric(c['mean_abs_cost']['value'])}|")
    lines += ["", "Continuous outcomes are averaged within source video first, in nats. Selected-subset component means then give equal weight to sources with at least one selected opportunity; empty sources are undefined and their counts are shown. Count ratios retain opportunity numerators/denominators. Selected realized gain over all nine positions is separately saved and reproduces the failed predictor's original utility.", "",
        "## Connection to C-nonlinear and attribution sensitivity", ""]
    for role in roles:
        f, rev, avg = [lookup[role, "all", order] for order in u.ORDERS]
        s, sr, sa = [lookup[role, "C_nonlinear_selected", order] for order in u.ORDERS]
        lines += [f"{role}: forward cancellation occurs in {rate_text(f['rates']['cancellation_all'])} of all opportunities and {rate_text(f['rates']['cancellation_among_harmful'])} of harmful moves. Reverse gives {rate_text(rev['rates']['cancellation_all'])}; averaged gives {rate_text(avg['rates']['cancellation_all'])}.",
            f"Saved C-nonlinear selects {s['opportunities']} opportunities: {s['helpful_moves']} helpful, {s['harmful_moves']} harmful and {s['neutral_moves']} tied. Of harmful selections, beneficial early addition occurs in {rate_text(s['rates']['cancellation_among_harmful'])} forward, {rate_text(sr['rates']['cancellation_among_harmful'])} reverse and {rate_text(sa['rates']['cancellation_among_harmful'])} averaged.",
            f"Across all opportunities, benefit signs change in {rate_text(f['rates']['benefit_sign_changes'])}, cost signs in {rate_text(f['rates']['cost_sign_changes'])}, and cancellation classification in {rate_text(f['rates']['cancellation_order_changes'])}. Forward/reverse agree on cancellation for {rate_text(f['rates']['robust_forward_reverse_cancellation_all'])}. Among selections, cancellation classification changes in {rate_text(s['rates']['cancellation_order_changes'])}; robust overlap is {rate_text(s['rates']['robust_forward_reverse_cancellation_all'])}.", ""]
    lines += ["These measurements show exactly which interpretation survives the attribution order; a flag change is not an encoder or recognition accuracy measurement. Rates and intervals are descriptive, conditional on the saved features/head and sampled classes. They exclude training, model-selection and repeated-research uncertainty. The two cohorts are never pooled.", "",
        "## Three measured examples", "",
        "Selection was frozen: largest beneficial net gain, most negative net among forward cancellations, and largest absolute interaction among sign/cancellation-sensitive rows. Ties use cohort/source/r/t ordering and distinct examples are preferred. These are descriptive extremes, not representative samples.", ""]
    for selected in examples:
        e = selected["example"]
        lines += [f"### {selected['category']}", ""]
        if e is None:
            lines += ["No qualifying measured example exists.", ""]
            continue
        lines += [f"{e['cohort']}; source {e['source_video_id']}; video {e['video_id']}; class {e['class']}; t={e['t']}, r={e['r']}. Saved C-nonlinear selected: {e['C_nonlinear_selected']}. Rule: {selected['rule']}.", "",
            "|Schedule|Candidate refresh ordinals|Simulated refreshes|Whole-video CE (nats)|", "|---|---|---:|---:|"]
        for name in u.COUNTS:
            lines.append(f"|{name}|{e['schedule_'+name]}|{u.COUNTS[name]}|{e['loss_'+name]:.12f}|")
        lines += [f"\nNet {e['net_gain']:+.9f}; forward benefit/cost {e['early_addition_benefit']:+.9f}/{e['later_removal_cost']:+.9f}; reverse benefit/cost {e['addition_after_removal_benefit']:+.9f}/{e['removal_first_cost']:+.9f}; averaged {e['average_addition_benefit']:+.9f}/{e['average_removal_cost']:+.9f}.", ""]
    lines += ["## Verification and limits", "",
        f"Independent explicit feature-copy replay checks {verification['independent_schedule_replays']} schedules, latest-cache equality and exact temporal difference boundaries. Maximum original-target absolute discrepancy: {verification['max_absolute_target_discrepancy']:.12g}; maximum defined relative discrepancy: {numeric(verification['max_relative_target_discrepancy_defined'],12)}; undefined relative denominators: {verification['relative_undefined_count']}. Maximum algebra residual: {verification['max_absolute_algebra_residual']:.12g}; independent-loss discrepancy: {verification['max_absolute_independent_loss_discrepancy']:.12g}. Sign tolerance remains1e-12; independent replay uses the previously documented1e-10 absolute/relative tolerances. No tolerance was loosened.", "",
        "Representation quality, limited source/class support, unavailable true labels, and the predictability of future content remain unresolved. Retrospective cancellation does not prove future information is impossible to predict. Full-clip candidates and future-reference codec vectors keep this offline. There is no accuracy improvement, adaptive speedup, successful online controller or publication-ready contribution demonstrated here.", "",
        "The next action should depend on the reported magnitude and forward/reverse overlap. If cancellation accounts for many harmful moves robustly, reconsider the replace-later action and supervision before fitting another controller. If it is uncommon or attribution-sensitive, this diagnosis has not established the proposed explanation. Do not change a threshold, predictor or cohort using this analysis.", "",
        "## Reproduction", ""]
    fence = chr(96)*3
    command = "& .\\.venv\\Scripts\\python.exe .\\analyze_refresh_target_decomposition.py --recognition-input results/recognition_baseline_v2 --ablation-input results/task_signal_ablation_v2 --output results/refresh_target_decomposition_v1 --seed 20261019 --bootstrap 10000"
    lines += [fence+"powershell", "& .\\.venv\\Scripts\\python.exe -m unittest test_refresh_target_decomposition -v",
              command, command+" --audit-only", fence, "", "Completed outputs are audited without overwriting. Missing prerequisites block the run; no synthetic measurements substitute for the cached dataset.", ""]
    (output/"findings.md").write_text("\n".join(lines), encoding="utf-8")
    (output/"README.md").write_text("# refresh_target_decomposition_v1\n\nSee [findings](findings.md), [protocol](protocol.json), [verification](verification.json), [metrics](aggregate_metrics.json) and [intervals](bootstrap_intervals.json).\n\nSaved-feature retrospective controls; B has five refreshes and D has three. No training, decoding, RGB execution or latency measurement.\n", encoding="utf-8")
