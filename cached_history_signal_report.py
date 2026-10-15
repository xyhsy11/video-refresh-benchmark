"""Saved-result reporting for the fixed cached-history diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_utils import relative_path

LABELS = {"C_RF400": "C RF400 (reused)", "C_plus_T_RF400": "C + T RF400", "keep_reference": "Keep reference",
          "training_mean": "Training mean", "training_median": "Training median"}


def ci(row: dict) -> str:
    return f"{row['mean']:+.6f} [{row['CI95_lower']:+.6f}, {row['CI95_upper']:+.6f}]"


def report(output: Path) -> None:
    read = lambda name: json.loads((output/name).read_text(encoding="utf-8"))
    protocol, aggregate, intervals, checks = [read(n) for n in ["protocol.json", "aggregate_metrics.json", "bootstrap_intervals.json", "verification.json"]]
    metrics, comparisons = aggregate["metrics"], intervals["comparisons"]
    roles = ["development_oof", "exploratory"]
    def metric(role, method, length=0):
        return next(r for r in metrics if r["cohort"] == role and r["method"] == method and r["history_length"] == length)
    def contrast(role, baseline="C_RF400", length=0):
        return next(r for r in comparisons if r["cohort"] == role and r["baseline"] == baseline and r["history_length"] == length and r["metric"] == "utility")
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11})
    methods = ["keep_reference", "training_mean", "training_median", "C_RF400", "C_plus_T_RF400"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, role in zip(axes, roles):
        for i, method in enumerate(methods):
            value = metric(role, method)["utility"]
            axis.errorbar(value["mean"], i, xerr=[[value["mean"]-value["CI95_lower"]], [value["CI95_upper"]-value["mean"]]],
                          fmt="o", capsize=4, color="#186f99" if method == "C_plus_T_RF400" else "#606060")
        axis.set_yticks(range(len(methods)), [LABELS[n] for n in methods])
        axis.invert_yaxis(); axis.axvline(0, color="black", linestyle="--", linewidth=.8)
        axis.grid(axis="x", alpha=.2); axis.set_title(role.replace("_", " "))
        axis.set_xlabel("Realized CE gain (nats / independent opportunity)\nEqual-video mean and whole-source 95% interval")
    fig.suptitle("Cached-history diagnostic: fixed RF400, prediction > 1e-12\nBoth cohorts developmental; independent counterfactuals, not a policy rollout", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(figures/"decision_utility.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for axis, role in zip(axes, roles):
        for method, shift, color in [("C_RF400", -.06, "#686868"), ("C_plus_T_RF400", .06, "#186f99")]:
            values = [metric(role, method, k)["utility"] for k in [1, 2, 3]]
            means = np.array([r["mean"] for r in values])
            axis.errorbar(np.arange(1, 4)+shift, means,
                yerr=[means-np.array([r["CI95_lower"] for r in values]), np.array([r["CI95_upper"] for r in values])-means],
                fmt="o-", capsize=4, label=LABELS[method], color=color)
        axis.set_xticks([1, 2, 3], ["1 (t=1-3)", "2 (t=5-7)", "3 (t=9-11)"])
        axis.axhline(0, color="black", linestyle="--", linewidth=.8); axis.grid(alpha=.2)
        axis.set_xlabel("Distinct legitimately refreshed score vectors available")
        axis.set_ylabel("Realized CE gain (nats / opportunity)\nEqual-video mean and whole-source 95% interval")
        axis.set_title(role.replace("_", " ")); axis.legend()
    fig.suptitle("History-length breakdown: secondary descriptive analysis\nSame full-data model in every group; no subgroup fitting or selection", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(figures/"history_length_comparison.png", dpi=170); plt.close(fig)
    construction = read("history_construction_checks.json")
    audit_text = ["# Cached-history information audit", "",
        "T contains exactly 44 deterministic functions of already-computed periodic cached scores, not 44 independent new observations. This audit is algebra/implementation evidence, not a prediction-performance result.", "",
        "The history follows only refreshes at candidate ordinals 0,4,8 before the nine decisions. Each vector is added once; repeated outputs at the next two opportunities add no event. No skipped past RGB, future refreshed scores, hypothetical alternative histories, labels or outcomes enter the builder.", "",
        "For t=1,2,3, only q0 is available. All44 T values are zero, including masks, standard deviation and switch rate. T adds no information.", "",
        "For t=5,6,7, m=[4*q0+(t-4)*q4]/t, so q0=[t*m-(t-4)*q4]/4. All T values can already be derived from old C (including t and q4). Differences, nonlinear softmax/margin functions and masks make information explicit for the fixed shallow forest; they are not new observations.", "",
        "For t=9,10,11, m=[4*q0+4*q4+(t-8)*q8]/t, so old C identifies q0+q4 but generally not their individual values or order. Ordered score differences can preserve information lost by that compression. The controlled fixture swaps distinct q0 and q4, holds q8, ages, budgets, motion and all other C observations fixed, and leaves old C unchanged while T changes. It is not a performance example.", "",
        "Masks are redundant with the candidate block. Centered20-vectors each sum to zero. In the final block, given old C and latest difference L=q8-q4, q4=q8-L and q0=(q0+q4)-q4; the previous difference is then determined. Margin standard deviation is permutation-invariant, while the consecutive top1 switch rate can depend on order. Confidence statistics are nonlinear derived features. Do not count all44 columns as independent new information.", "",
        "Softmax uses exp(q-max(q))/sum in float64; argmax ties use the lowest class index. Margin standard deviation is population ddof0, with k1 zero. All numeric definitions, order and missing-history conventions were frozen before fitting.", "",
        f"Controlled fixture C discrepancy: {construction['fixture']['max_old_C_difference']}; changed T fields: {', '.join(construction['fixture']['changed_T_columns'])}.", "",
        "Real-data weighted-past-mean maximum discrepancies: "+", ".join(f"{role}={construction[role]['max_weighted_past_mean_algebra_discrepancy']:.12g}" for role in ["development", "exploratory"])+". Cached score repetitions agree exactly. Incremental T matches the independent builder.", "",
        "Full-clip candidate selection and existing future-reference codec vectors keep this offline. Prefix-only score handling does not validate strict streaming causality. No temporal video backbone is evaluated."]
    (output/"feature_information_audit.md").write_text("\n".join(audit_text)+"\n", encoding="utf-8")
    lines = ["# cached_history_signal_diagnostic_v1 findings (Experiment13)", "",
        "This completed offline study changes only observation representation. It preserves the frozen recognition head, signed net-gain target,100 development/40 exploratory sources,900/360 opportunities,original five folds,RF400 recipe/seed20261020 and decision threshold1e-12. The baseline's six fitted models are reused byte-for-byte; only six augmented forests are fitted. Both cohorts were previously inspected, and recognition regularization used development. Neither is independent confirmation.", "",
        "## Answers", "",
        "1. **Information added:** no new information at history length1; length2 is algebraically recoverable from old C; length3 can preserve individual/order information discarded by the weighted past mean. T also makes differences and nonlinear statistics explicit. The44 features are dependent derived quantities, not44 new independent observations. See the controlled information audit.",
        f"2. **Relative utility:** development augmented minus baseline is {ci(contrast('development_oof'))}; exploratory is {ci(contrast('exploratory'))} nats/opportunity.",
        f"3. **Absolute utility:** augmented development is {ci(metric('development_oof','C_plus_T_RF400')['utility'])}; exploratory is {ci(metric('exploratory','C_plus_T_RF400')['utility'])}. Keeping reference is zero. A less negative result does not establish useful decisions.",
        "4. **History length:** all three predeclared subgroup comparisons are below, using the same full-data models. They are descriptive, marginal and unadjusted. In length1, T is entirely zero, so prediction changes can reflect the refitted forest's partitions rather than new information at that opportunity. No favorable subgroup is selected as a new primary result.",
        "5. **Incremental work:** state stores three cached vectors, three margins/top1 indices/refresh ordinals,one cached44-vector and a counter. Each of3 distinct events/video computes one stable20-class softmax/top1 and rebuilds T;6 repeated observations/video reuse T. Actual memory and component timings are below."]
    primary, absolute = contrast("development_oof"), metric("development_oof", "C_plus_T_RF400")["utility"]
    encouraging = primary["CI95_lower"] > 0 and absolute["CI95_lower"] > 0
    next_step = ("Freeze this exact representation/model and plan new-source confirmation; do not call the reused-cohort result independent confirmation or a successful adaptive system."
                 if encouraging else "Move to a separately designed temporal-recognition backbone/task benchmark before another cached-feature variant. Freeze its task and independent-source evaluation protocol first; do not automatically launch it here.")
    lines += ["6. **Next step:** "+next_step, "",
        "The predeclared positive-relative and positive-absolute utility requirements "+("are supported on development within this conditional diagnostic." if encouraging else "are not both established; useful positive utility remains unproven."), "",
        "## Overall measurements", "",
        "Utility averages all9 opportunities within source, including zero for unselected moves, then gives sources equal weight. Signed targets and predictions are unclipped. These independent moves are never combined into a sequential schedule.", "",
        "|Cohort|Method|MAE|MSE|MAE improvement vs training median|Utility and95% interval|Selected/total|Helpful/harmful/neutral|Mean selected gain (denominator)|",
        "|---|---|---:|---:|---|---|---:|---|---|"]
    for role in roles:
        for method in methods:
            r = metric(role, method)
            mean = "undefined:no selections" if r["mean_selected_gain"] is None else f"{r['mean_selected_gain']:+.6f} ({r['mean_selected_gain_denominator']})"
            lines.append(f"|{role}|{method}|{r['MAE']['mean']:.6f}|{r['MSE']['mean']:.6f}|{ci(r['MAE_improvement_over_training_median'])}|{ci(r['utility'])}|{r['selected']}/{r['opportunities']}|{r['helpful']}/{r['harmful']}/{r['neutral']}|{mean}|")
    lines += ["", "All MAE/utility units are nats; MSE is nats squared. Spearman and positive-gain AUROC are pooled descriptive statistics, with event counts and null/reason for undefined cases in aggregate_metrics.json. Candidate observations are correlated within source.", "",
        "## History-length comparisons (secondary)", "",
        "|Cohort|Distinct refreshes|Baseline utility|Augmented utility|Paired augmented-minus-baseline and95% interval|Improved/tied/worsened sources|",
        "|---|---:|---:|---:|---|---|"]
    for role in roles:
        for k in [1, 2, 3]:
            r = contrast(role, length=k)
            lines.append(f"|{role}|{k}|{metric(role,'C_RF400',k)['utility']['mean']:+.6f}|{metric(role,'C_plus_T_RF400',k)['utility']['mean']:+.6f}|{ci(r)}|{r['videos_improved']}/{r['videos_tied']}/{r['videos_worsened']}|")
    timing = pd.read_csv(output/"cpu_microbenchmark.csv", float_precision="round_trip")
    tm = read("cpu_microbenchmark.metadata.json")
    env = read("environment.json")
    lines += ["", "## Component CPU work and model accounting", "",
        f"CPU: {env['cpu']}. Python{env['python']};RF n_jobs1;OMP/MKL/OpenBLAS threads1. Timings use10 development sources selected lexicographically before outcomes,90 opportunities/pass,2 warmup and5 measured passes. All observations/models are preloaded. Methods are interleaved with PCG64 seed20261021;actual orders and identities are recorded.", "",
        "|Component|Median microseconds/opportunity|Min|max|Measured passes|", "|---|---:|---:|---:|---:|"]
    for mode, g in timing.groupby("mode", sort=True):
        values = g.us_per_opportunity
        lines.append(f"|{mode}|{values.median():.3f}|{values.min():.3f}|{values.max():.3f}|{len(g)}|")
    state = tm["history_state"]
    lines += ["", f"Retained history ndarray payload is {state['ndarray_payload_bytes']}bytes plus a Python integer counter ({state['python_counter_bytes']}bytes in this environment). The slots/array/counter sizeof estimate is {state['python_object_and_arrays_sizeof_estimate_bytes']}bytes, excluding allocator/transient/RF workspace. A100-value float64 augmented input is800bytes versus448bytes for C, an additional352bytes per transient input. Scaling and sklearn float32 buffers add transient workspace; forest memory is separate.",
        "History_T includes per-video state allocation/reset,all30 distinct cached events and90 feature accesses/pass. Stable softmax is computed once per distinct event; T is cached for repetitions. Baseline includes56-column scaling plus90 batch-size1 RF calls; augmented includes history handling,100-column concatenation/scaling and90 batch-size1 RF calls. Library prediction validation is included; external correctness assertions,loading,fitting and logging are excluded.",
        "These are cached-observation component CPU timings. They exclude video decoding,RGB extraction,codec-vector extraction and full recognition execution. They are not adaptive inference latency, cannot be subtracted from the old pilot, and establish no end-to-end speedup. Five passes describe local variability, not statistical significance.", "",
        "|Final-development model|Trees|Nodes|Compressed serialized bytes|", "|---|---:|---:|---:|"]
    for r in checks["model_accounting"]:
        if r["bundle"] == "final_development": lines.append(f"|{r['method']}|{r['trees']}|{r['nodes']}|{r['serialized_bytes']}|")
    lines += ["", "Equal400-tree counts do not imply equal partitions, effective capacity, memory or inference cost. Extra explicit features can change the shallow forest's partitions even where they contain no new information.", "",
        "## Verification and limits", "",
        f"Baseline prediction discrepancy is {checks['baseline_prediction_max_discrepancy']};{checks['independent_history_feature_checks']} independent T checks and{checks['input_isolation_mutations']} input-isolation mutations pass. Max first56 scaler discrepancy is {checks['max_scaler_first56_discrepancy']:.12g}. All folds/targets/source memberships,training-only scaling,latest history membership,missing-history zeros,cache-event deduplication,saved metrics and whole-source bootstrap reproduction pass. Earlier scientific artifacts and the head remain unchanged.",
        "Actual RGB encoder executions,video decodes,video downloads and recognition-head training calls are zero. No fresh-confirmation artifact is accessed. Full-clip candidates and future-reference codec vectors retain the offline limitation; this does not evaluate a temporal video backbone or validate streaming causality.",
        f"Intervals use{protocol['bootstrap']['resamples']} paired whole-source resamples,PCG64 seed{protocol['seed']},linear percentiles. All9 opportunities stay together,shared indices cover every method/group within cohort,and cohorts remain separate. Intervals condition on fitted models/reused cohorts and omit full training/model-selection/repeated-research uncertainty. Secondary intervals are unadjusted. No accuracy improvement,successful adaptive controller or end-to-end speedup is established.", "",
        "## Reproduce", "", "```powershell", "& .\\.venv\\Scripts\\python.exe -m unittest test_cached_history_signals -v"]
    command = ("& .\\.venv\\Scripts\\python.exe .\\analyze_cached_history_signals.py --predictability-input "+protocol["predictability_input"]
               +" --output "+relative_path(output)+f" --seed {protocol['seed']} --bootstrap {protocol['bootstrap']['resamples']}")
    lines += [command, command+" --audit-only", "```", "", "Completed invocations audit without overwriting or retiming;changed protocols select a clearly versioned new output directory. Models,observations,predictions,timing rows and indices are retained for independent audit."]
    (output/"findings.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (output/"README.md").write_text("# cached_history_signal_diagnostic_v1\n\nSee [findings](findings.md), [information audit](feature_information_audit.md), [protocol](protocol.json), [verification](verification.json), [metrics](aggregate_metrics.json), [intervals](bootstrap_intervals.json) and [CPU timings](cpu_microbenchmark.csv).\n\nCached-score offline diagnostic;no temporal video backbone,recognition training or adaptive latency measurement.\n", encoding="utf-8")
