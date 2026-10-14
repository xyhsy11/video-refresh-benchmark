"""Figures and candid interpretation derived only from saved experiment results."""
from __future__ import annotations

import json
import io
from pathlib import Path
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fresh_confirmation_data import strict_json
from research_utils import ROOT, sha256_file
import task_aware_oracle_utils as u


def save(figure, directory: Path, name: str) -> None:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    with tempfile.NamedTemporaryFile("wb", dir=directory, suffix=".tmp", delete=False) as handle:
        handle.write(buffer.getvalue())
        temporary = Path(handle.name)
    for attempt in range(12):
        try:
            temporary.replace(directory/name)
            break
        except OSError:
            if attempt == 11:
                raise
            time.sleep(min(.025*2**attempt,.4))
    plt.close(figure)


def report(output: Path) -> None:
    figures = output/"figures"; figures.mkdir(exist_ok=True)
    table = pd.read_csv(output/"per_video_metrics.csv", float_precision="round_trip")
    raw = pd.read_csv(output/"trajectory_metrics.csv", float_precision="round_trip")
    dense = pd.read_csv(output/"dense_classifier_predictions.csv", float_precision="round_trip")
    aggregates = pd.DataFrame(u.read_json(output/"aggregate_metrics.json")["rows"])
    intervals = u.read_json(output/"bootstrap_intervals.json")["comparisons"]
    classifier = u.read_json(output/"classifier_metrics.json")
    configuration = u.read_json(output/"model_configuration.json"); classes = configuration["classes"]
    seed = u.read_json(output/"protocol.json")["seed"]
    schedules = u.read_json(output/"oracle_schedules.json")["rows"]
    roles = ["development", "exploratory_test"]
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .2})
    for metric, filename, ylabel in [("top1_accuracy", "accuracy_vs_refresh_budget.png", "Top-1 action classification accuracy (%)"),
                                     ("cross_entropy", "classification_loss_vs_budget.png", "True-class cross-entropy (nats/video)")]:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        for ax, role in zip(axes, roles):
            for policy in [*u.POLICIES, "dense"]:
                g = aggregates[(aggregates.role == role) & (aggregates.policy == policy)].sort_values("budget")
                ax.plot(g.budget, g[metric]*(100 if metric == "top1_accuracy" else 1), marker="o", label=u.DISPLAY[policy])
            if metric == "top1_accuracy": ax.axhline(5, color="grey", linestyle=":", label="Chance / majority: 5%")
            ax.set(title=role.replace("_", " "), xlabel="Simulated feature refreshes B (no encoder execution)", xticks=[2,4,8,16])
        axes[0].set_ylabel(ylabel); axes[1].legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
        fig.suptitle("Frozen linear mean-pooled head; oracles are privileged hindsight")
        save(fig, figures, filename)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, role in zip(axes, roles):
        for policy in u.POLICIES:
            g = table[(table.role == role) & (table.budget == 4) & (table.policy == policy)]
            ax.scatter(g.feature_cosine_error, g.cross_entropy, s=23, alpha=.65, label=u.DISPLAY[policy])
        ax.set(title=role.replace("_", " "), xlabel="Mean post-decision feature cosine error", ylabel="Classification cross-entropy (nats/video)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
    fig.suptitle("B=4; source-video observations, descriptive association")
    save(fig, figures, "feature_error_vs_classification_loss.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, role in zip(axes, roles):
        g = table[(table.role == role) & (table.budget == 4)].pivot(index="video_id", columns="policy", values="cross_entropy")
        ax.scatter(g.periodic, g.task_oracle, s=28)
        end = max(g.periodic.max(), g.task_oracle.max())*1.05
        ax.plot([0,end], [0,end], color="grey", linestyle="--")
        ax.set(title=role.replace("_", " "), xlabel="Periodic CE (nats/video)", ylabel="Task oracle CE (label + hindsight)")
    fig.suptitle("B=4 paired losses: selection uses each evaluation video's true label")
    save(fig, figures, "periodic_vs_task_oracle.png")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    for ax, role in zip(axes, roles):
        g = table[(table.role == role) & (table.budget == 4)].pivot(index="video_id", columns="policy", values="cross_entropy")
        delta = (g.periodic-g.task_oracle).sort_values()
        ax.bar(np.arange(len(delta)), delta, color="#597dab")
        ax.set(title=role.replace("_", " "), ylabel="Periodic minus task-oracle CE (nats)", xlabel="Source videos sorted by paired improvement")
        ax.axhline(0, color="black", linewidth=.8)
    fig.suptitle("B=4 hindsight loss headroom, not deployable performance")
    fig.tight_layout(rect=[0,0,1,.94])
    save(fig, figures, "per_video_task_oracle_improvement.png")
    # Freeze the descriptive example rule in the protocol, not by visual appeal.
    dev = table[(table.role == "development") & (table.budget == 4)].pivot(index="video_id", columns="policy", values="cross_entropy")
    example_ids = (dev.feature_oracle-dev.task_oracle).rename("gain").reset_index().sort_values(["gain","video_id"], ascending=[False,True]).head(3).video_id.tolist()
    fig, axes = plt.subplots(3, 1, figsize=(13, 9))
    comparisons = []
    for ax, vid in zip(axes, example_ids):
        v = next(e for e in u.read_json(output/"dataset_split.json")["videos"] if e["entry"]["video_id"] == vid)["entry"]
        records = next(e for e in schedules if e["role"] == "development" and e["video_id"] == vid and e["budget"] == 4)
        input_directory = u.read_json(output/"metadata.json")["input_directory"]
        with np.load(ROOT/input_directory/f"features/{vid}.npz", allow_pickle=False) as data: features = data["features"]
        logits = u.feature_logits(features, u.read_json(output/"classifier_parameters.json")); y = classes.index(v["label"])
        others = logits.copy(); others[:, y] = -np.inf
        ax.plot(np.arange(16), logits[:, y]-others.max(axis=1), marker=".", color="grey", label="Single-feature true-class logit margin")
        for kind, color, marker in [("feature", "#4777aa", "o"), ("task", "#c76637", "s")]:
            positions = records[kind]["positions"]
            level = min((logits[:,y]-others.max(axis=1)).min(),0)-(.5 if kind == "feature" else 1.5)
            ax.scatter(positions, [level]*4, color=color, marker=marker, s=65, label=f"{kind} oracle refreshes (hindsight)")
        ax.set(title=f"{vid}: {v['label']}", xlabel="Chronological candidate ordinal", ylabel="Logit margin / schedule tracks", xticks=range(16))
        ax.legend(loc="upper right", fontsize=8)
        comparisons.append({"video_id": vid, "feature_positions": records["feature"]["positions"], "task_positions": records["task"]["positions"], "label": v["label"]})
    fig.suptitle("Predefined development examples: largest task-versus-feature loss gains\nMargins describe the classifier; they do not establish semantic action localization")
    fig.tight_layout(rect=[0,0,1,.94]); save(fig, figures, "oracle_schedule_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(23, 11))
    matrices = {}
    for ax, role in zip(axes, roles):
        g = dense[dense.role == role]; matrix = np.zeros((20,20), dtype=int)
        for row in g.itertuples(): matrix[row.label_index, row.prediction_index] += 1
        matrices[role] = matrix.tolist()
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=2)
        for i,j in zip(*np.nonzero(matrix)): ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=8)
        ax.set(title=f"Dense classifier: {role.replace('_',' ')} (2 clips/class)", xlabel="Predicted class", ylabel="Ground-truth class",
               xticks=range(20), yticks=range(20), xticklabels=classes, yticklabels=classes)
        ax.tick_params(axis="x", rotation=90); ax.grid(False)
    fig.tight_layout(); save(fig, figures, "confusion_matrix.png")
    strict_json(output/"confusion_matrices.json", {"classes": classes, "counts": matrices})
    schedule_summary = []
    for role in roles:
        for budget in [2,4,8]:
            rows = [e for e in schedules if e["role"] == role and e["budget"] == budget]
            overlap = [len(set(e["task"]["positions"]) & set(e["feature"]["positions"])) for e in rows]
            schedule_summary.append({"role": role, "budget": budget, "videos": len(rows),
                                     "identical_schedules": sum(e["task"]["positions"] == e["feature"]["positions"] for e in rows),
                                     "mean_shared_refresh_fraction": float(np.mean(overlap)/budget),
                                     "mean_jaccard": float(np.mean([n/(2*budget-n) for n in overlap]))})
    examples = []
    for role in roles:
        g = table[(table.role == role) & (table.policy == "periodic") & (table.budget == 4)]
        median = g.feature_cosine_error.median()
        for kind, selected in [("large_error_still_correct", g[(g.feature_cosine_error >= median) & (g.correct == 1)]),
                               ("small_error_classification_changed", g[(g.feature_cosine_error <= median) & (g.disagrees_with_dense == 1)])]:
            examples.append({"role": role, "criterion": kind, "error_median_cutoff": float(median), "count": len(selected),
                             "first_three_by_video_id": selected.sort_values("video_id").head(3)[["video_id","label","feature_cosine_error","cross_entropy","correct","disagrees_with_dense"]].to_dict("records")})
    strict_json(output/"analysis_examples.json", {"schedule_examples": comparisons, "schedule_similarity": schedule_summary, "feature_recognition_gap_examples": examples})
    primary = [r for r in intervals if r["primary"]]
    test_ce = next(r for r in primary if r["role"] == "exploratory_test" and r["metric"] == "cross_entropy")
    test_accuracy = next(r for r in primary if r["role"] == "exploratory_test" and r["metric"] == "correct")
    adequate = all(classifier[role]["top1_accuracy"] >= .5 for role in roles)
    near_chance = any(classifier[role]["top1_accuracy"] <= .1 for role in roles)
    decision = "REVISE" if not adequate else "CONTINUE" if test_accuracy["mean"] >= .05 and test_ce["mean"] >= .05 else "PIVOT"
    text = ["# Task-aware oracle feasibility v1", "",
            f"Decision: **{decision}**. This is a development / separately identified exploratory hindsight study, not an independent confirmation or deployable accuracy result.", "",
            "## Classifier quality and data", "",
            "Eighty original-training videos (four/class) fit the linear head and training-only population normalization. Forty development videos (two/class) select one of four predefined regularization values; forty previously used original-test videos are evaluated only after freezing. Original validation arrays and the 100-video confirmation cohort are unused. Official CVDF annotation identity and labels are checked. The class-index order is the original frozen twenty-class list.", "",
            f"Selected lambda={configuration['selected_regularization']}; CPU float64 deterministic cross-entropy training; mean pooling; frozen ResNet18 references. Chance and majority-class accuracy are 5%. Training: {classifier['train']['top1_accuracy']*100:.2f}% top1, CE {classifier['train']['cross_entropy']:.4f}. Development: {classifier['development']['top1_accuracy']*100:.2f}% top1, CE {classifier['development']['cross_entropy']:.4f}. Exploratory test: {classifier['exploratory_test']['top1_accuracy']*100:.2f}% top1, CE {classifier['exploratory_test']['cross_entropy']:.4f}. Macro equals top1 here because classes have equal counts. Class-wise values and confusion counts are saved.", "",
            f"Predefined exploratory quality gate (dense >=50% on both evaluation cohorts): {'passed' if adequate else 'failed'}. {'Performance is close to chance, so oracle conclusions about useful understanding are unreliable.' if near_chance else 'Performance exceeds chance, but only two evaluation videos per class and eighty training clips limit reliability; this is not a validated action-recognition system.'} Training minus exploratory accuracy is {100*(classifier['train']['top1_accuracy']-classifier['exploratory_test']['top1_accuracy']):.2f} percentage points. This large gap and classes with no correct dense predictions call for a stronger recognition baseline before publication or claims about useful video understanding.", "",
            "## B=4 recognition and feature fidelity", "",
            "| Cohort | Method | Top1 / macro (%) | CE (nats) | Feature cosine error | Disagreement vs dense (%) |", "|---|---|---:|---:|---:|---:|"]
    for role in roles:
        for policy in ["dense", *u.POLICIES]:
            g = aggregates[(aggregates.role == role) & (aggregates.policy == policy) & (aggregates.budget == (16 if policy == "dense" else 4))].iloc[0]
            text.append(f"|{role}|{u.DISPLAY[policy]}|{g.top1_accuracy*100:.2f}|{g.cross_entropy:.6f}|{g.feature_cosine_error:.6f}|{g.classification_disagreement_vs_dense*100:.2f}|")
    text += ["", "## Paired primary hindsight headroom", "",
             f"Positive gain favors the task oracle. Cross-entropy gains use periodic minus oracle loss; correctness gains use oracle minus periodic. Bootstrap uses 10,000 whole-source-video paired resamples, PCG64 seed{seed}, linear percentiles. Models remain fixed. Frames, candidate schedules and random seeds are not independent sampling units. Intervals are conditional on the selected head/classes and do not account for model selection or small training-sample uncertainty.", "",
             "|Cohort|Outcome|Mean gain|95% interval|Improved/tied/worsened|", "|---|---|---:|---|---|"]
    for r in primary:
        scale = 100 if r["metric"] == "correct" else 1; unit = "pp" if scale == 100 else "nats"
        text.append(f"|{r['role']}|{r['metric']}|{r['mean']*scale:+.6f} {unit}|[{r['CI95_lower']*scale:+.6f},{r['CI95_upper']*scale:+.6f}]|{r['improved']}/{r['tied']}/{r['worsened']}|")
    text += ["", "The oracle evaluates every valid schedule using the evaluation video's true label and future candidate features: 15/455/6,435 schedules for B2/4/8. It has a large selection advantage. Since periodic is among these schedules, its minimum CE cannot be worse by construction. Positive loss intervals quantify this privileged opportunity, not predictive generalization. CE optimization need not maximize top1 correctness. All budgets are exact. Dense is a separate B16 reference, not necessarily the task-loss optimum across sparse budgets.", "", "## Research questions and recommendation", ""]
    dev_b4 = aggregates[(aggregates.role == "development") & (aggregates.budget == 4)].set_index("policy")
    test_b4 = aggregates[(aggregates.role == "exploratory_test") & (aggregates.budget == 4)].set_index("policy")
    text += [f"A. Task-aware versus periodic: development top1 difference {100*(dev_b4.loc['task_oracle','top1_accuracy']-dev_b4.loc['periodic','top1_accuracy']):+.2f}pp; exploratory {test_accuracy['mean']*100:+.2f}pp. Exploratory CE headroom {test_ce['mean']:+.6f} nats. These are privileged results; see paired intervals above.", "",
             f"B. Feature oracle: exploratory B4 top1 {100*test_b4.loc['feature_oracle','top1_accuracy']:.2f}%, versus task oracle {100*test_b4.loc['task_oracle','top1_accuracy']:.2f}%; CE {test_b4.loc['feature_oracle','cross_entropy']:.6f} versus {test_b4.loc['task_oracle','cross_entropy']:.6f}. Feature fidelity and downstream loss are distinct objectives.", ""]
    rho = next(r for r in u.read_json(output/"feature_task_correlations.json")["rows"] if r["role"] == "exploratory_test" and r["policy"] == "periodic" and r["budget"] == 4 and r["outcome"] == "loss_change_vs_dense")
    sim = next(r for r in schedule_summary if r["role"] == "exploratory_test" and r["budget"] == 4)
    text += [f"C. Exploratory periodic B4 feature error versus CE change from dense: Spearman {rho['spearman'] if rho['spearman'] is not None else 'undefined (constant outcome)'}. All within-policy/budget correlations are saved, avoiding a pooled cross-budget association. This small descriptive association does not validate cosine error as a task surrogate. There are correct classifications with large feature errors and changed classifications with relatively small errors; exact counts/examples are in analysis_examples.json. Oracles choose identical B4 schedules for {sim['identical_schedules']}/40 exploratory videos; mean shared-refresh fraction {sim['mean_shared_refresh_fraction']:.3f}. Candidate margins cannot establish action-relevant temporal localization without independent annotations.", "",
             f"D. Recommendation **{decision}** under the frozen gate. " + ("First improve/validate the recognition representation and classifier on a larger, source-disjoint action dataset. A stronger pretrained temporal action encoder is a proposed separate study, with its weights/data terms checked before acquisition. Do not train a controller from this oracle result yet." if decision == "REVISE" else "Investigate a causal task-aware controller in a separately frozen development study, then confirm on genuinely new sources and measure recognition quality and actual latency separately." if decision == "CONTINUE" else "This defined representation/head offers insufficient useful oracle headroom under the predefined gate. Explore partial or spatial feature recomputation or motion-compensated propagation in a separate feasibility study."), "",
             "E. Classifier reliability remains limited: twenty classes, four training clips/class, mean-pooled ImageNet appearance features without a trained temporal backbone. Development hyperparameters were selected on reported development videos. The reused original test is exploratory, not pristine. The same existing controller coefficients are reused, not task-trained; absence of their gains would not test all possible causal controllers.", "", "## Limits", "",
             "Zero actual ResNet18 encoder executions and zero video decodes occurred. Head training and saved-feature replay are actual CPU work; simulated B refreshes are not measured GPU calls or latency/speedup. Every replay produces sixteen outputs using the latest cache. Motion controllers consume only current/past decoded positions, but codec vectors may reference future frames; full-clip candidate selection also makes this offline. Task oracle uses true labels/future RGB and is unavailable at inference. Neither improved feature fidelity nor this weak diagnostic establishes real action-recognition preservation in a deployed system. No 100-video confirmation feature/motion/accuracy artifact was used, and no historical experiment was rewritten.", "", "## Reproduction", "", "Run from the repository root using the existing environment:", "", "```powershell", "& .\\.venv\\Scripts\\python.exe -m unittest test_task_aware_oracle -v", "& .\\.venv\\Scripts\\python.exe .\\run_task_aware_oracle.py --input results/cache_drift_v2 --frozen-run results/budget_aware_refresh_v2 --output results/task_aware_oracle_v1 --seed 20261015", "& .\\.venv\\Scripts\\python.exe .\\run_task_aware_oracle.py --input results/cache_drift_v2 --output results/task_aware_oracle_v1 --audit-only", "& .\\.venv\\Scripts\\python.exe .\\validate_release.py", "```", "", "Recollection is unnecessary. Full refitting checks cached data/source/cache_drift_v2/val.csv against its original hash; --annotations selects another copy. Saved-result auditing needs only the published feature/motion artifacts and saved head, not encoded videos or the annotation download. Existing output directories are preserved and a fresh run directory is chosen. Parameters and training configuration are classifier_parameters.json and model_configuration.json. Exact source and input identities are in metadata.json; checksums cover this snapshot."]
    (output/"findings.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    (output/"README.md").write_text("# Task-aware oracle v1\n\n"+f"Status: completed after verification; decision **{decision}**. 80 training,40 development,40 exploratory test videos; fresh confirmation excluded.\n\n"+"See [findings](findings.md), [classifier quality](classifier_metrics.json), [per-video metrics](per_video_metrics.csv), [oracle schedules](oracle_schedules.json), [paired intervals](bootstrap_intervals.json), [verification](verification.json), [head parameters](classifier_parameters.json) and [configuration](model_configuration.json). Reproduction commands are in findings; full refitting requires the original small annotation CSV, audit requires no video assets.\n\n"+"![Classification accuracy](figures/accuracy_vs_refresh_budget.png)\n\n![Task loss](figures/classification_loss_vs_budget.png)\n\nTask oracle is privileged and label-aware. No deployable accuracy or latency improvement is claimed.\n", encoding="utf-8")
