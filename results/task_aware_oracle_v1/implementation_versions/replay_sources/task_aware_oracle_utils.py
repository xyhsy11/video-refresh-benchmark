"""Small frozen-feature classifier and exact hindsight schedule evaluation.

RGB references and labels enter only the classifier/evaluator. Existing policy
callbacks continue to receive CheapVideo and their previously frozen coefficients.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

import budget_aware_refresh_utils as budget_utils
from fresh_confirmation_eval import load_references
from research_utils import sha256_file

TOL = 1e-12
POLICIES = ["periodic", "random", "value_age_budget", "value_age_motion_budget",
            "feature_oracle", "task_oracle"]
DISPLAY = {"dense": "Dense", "periodic": "Periodic", "random": "Random (10 seeds)",
           "value_age_budget": "Frozen age/budget", "value_age_motion_budget": "Frozen age/motion/budget",
           "feature_oracle": "Feature oracle (hindsight)", "task_oracle": "Task oracle (label + hindsight)"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(directory: Path, frozen: Path, annotations: Path) -> tuple[list, list, list, dict, pd.DataFrame]:
    """Read original-training data first. Original validation arrays are never opened."""
    manifest = read_json(directory / "split_manifest.json")
    split = read_json(frozen / "development_split.json")
    train, dev, record = budget_utils.load_training_artifacts(directory, manifest, split)
    plan = read_json(directory / "sampling_plan.json")
    if not annotations.is_file():
        raise FileNotFoundError("Ground-truth annotation CSV missing; use --annotations with the original CVDF val.csv")
    assert sha256_file(annotations) == plan["annotations_sha256"], "Annotation identity mismatch"
    labels = pd.read_csv(annotations)
    assert {"youtube_id", "label", "time_start", "time_end"} <= set(labels)
    keys = ["youtube_id", "time_start", "time_end"]
    assert not labels.duplicated(keys).any(), "Ambiguous annotation clip identity"
    indexed = labels.set_index(keys)
    tests = [e for e in manifest["videos"] if e["experiment_split"] == "test"]
    groups = {"train": train, "development": dev}
    # Check all permitted evaluation identities/labels before any test feature access.
    entries = [(role, v.entry) for role, vs in groups.items() for v in vs]
    entries += [("exploratory_test", e) for e in tests]
    audit = []
    for role, entry in entries:
        key = (entry["source_video_id"], entry["time_start"], entry["time_end"])
        assert key in indexed.index and indexed.loc[key, "label"] == entry["label"], entry["video_id"]
        audit.append({"role": role, "video_id": entry["video_id"], "source_video_id": key[0],
                      "time_start": key[1], "time_end": key[2], "label": entry["label"],
                      "encoded_sha256": entry["file_identity"]["sha256"]})
    table = pd.DataFrame(audit)
    assert len(table) == 160 and table.source_video_id.nunique() == table.encoded_sha256.nunique() == 160
    assert table.groupby("role").size().to_dict() == {"train": 80, "development": 40, "exploratory_test": 40}
    assert (table.groupby(["role", "label"]).size() == table.role.map({"train": 4, "development": 2, "exploratory_test": 2}).groupby([table.role, table.label]).first()).all()
    record.update(annotation_sha256=sha256_file(annotations), development_split_sha256=sha256_file(frozen / "development_split.json"),
                  test_entries=tests, classes=manifest["classes"], test_artifacts_opened_before_classifier_freeze=0)
    return train, dev, tests, record, table


def load_exploratory(directory: Path, entries: list) -> tuple[list, dict]:
    ledger = dict((name, digest) for digest, name in
                  (line.split("  ", 1) for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()))
    hashes = {name: ledger[name] for e in entries for name in
              [f"features/{e['video_id']}.npz", f"features/{e['video_id']}.json", f"motion_frames/{e['video_id']}.csv"]}
    return load_references(entries, directory, hashes), hashes


def classification_loss(logits: np.ndarray, label: int) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    assert x.ndim == 2 and np.isfinite(x).all() and 0 <= label < x.shape[1]
    return np.logaddexp.reduce(x, axis=1) - x[:, label]


def feature_logits(features: np.ndarray, parameters: dict) -> np.ndarray:
    x = (np.asarray(features, dtype=np.float64) - np.asarray(parameters["means"])) / np.asarray(parameters["scales"])
    return x @ np.asarray(parameters["weights"]).T + np.asarray(parameters["bias"])


def fit_classifier(features: np.ndarray, labels: np.ndarray, regularization: float, seed: int) -> tuple[dict, dict]:
    """Mean cross entropy + lambda/2 * squared weights; unpenalized intercept."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    x = np.asarray(features, dtype=np.float64)
    means, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    scales = np.where(std <= TOL, 1.0, std)
    tensor = torch.tensor((x - means) / scales, dtype=torch.float64)
    target = torch.tensor(labels, dtype=torch.int64)
    classes = int(target.max()) + 1
    head = torch.nn.Linear(x.shape[1], classes, dtype=torch.float64)
    optimizer = torch.optim.LBFGS(head.parameters(), lr=1.0, max_iter=400, max_eval=800,
                                 tolerance_grad=1e-9, tolerance_change=1e-12,
                                 history_size=50, line_search_fn="strong_wolfe")
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(tensor), target) + regularization / 2 * head.weight.square().sum()
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)
    objective = float(closure().detach())
    gradients = max(float(p.grad.abs().max()) for p in head.parameters())
    state = optimizer.state[head.weight]
    parameters = {"means": means.tolist(), "scales": scales.tolist(),
                  "weights": head.weight.detach().numpy().tolist(), "bias": head.bias.detach().numpy().tolist()}
    assert np.isfinite(np.array(parameters["weights"])).all()
    return parameters, {"regularization": regularization, "objective": objective,
                        "gradient_max_abs": gradients, "closure_calls": evaluations,
                        "iterations": int(state["n_iter"]), "constant_input_columns": int((std <= TOL).sum())}


def dense_metrics(videos: list, classes: list[str], parameters: dict) -> tuple[dict, pd.DataFrame]:
    rows = []
    for video in videos:
        y = classes.index(video.entry["label"])
        logits = feature_logits(video.references, parameters).mean(axis=0)
        rows.append({"video_id": video.entry["video_id"], "source_video_id": video.entry["source_video_id"],
                     "label": video.entry["label"], "label_index": y, "prediction_index": int(logits.argmax()),
                     "correct": int(logits.argmax() == y), "cross_entropy": float(classification_loss(logits[None], y)[0]),
                     "logits": json.dumps(logits.tolist())})
    table = pd.DataFrame(rows)
    return {"videos": len(table), "top1_accuracy": float(table.correct.mean()),
            "macro_top1_accuracy": float(table.groupby("label").correct.mean().mean()),
            "cross_entropy": float(table.cross_entropy.mean()),
            "class_wise": {name: {"videos": len(g), "accuracy": float(g.correct.mean()), "cross_entropy": float(g.cross_entropy.mean())}
                           for name, g in table.groupby("label")}}, table


def cache_indices(positions: list[int], n: int = 16) -> np.ndarray:
    assert positions == sorted(set(positions)) and positions[0] == 0 and positions[-1] < n
    return np.maximum.accumulate(np.where(np.isin(np.arange(n), positions), np.arange(n), 0))


@lru_cache(None)
def schedule_table(n: int, budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.array([(0, *rest) for rest in combinations(range(1, n), budget - 1)], dtype=np.int64)
    marks = np.zeros((len(positions), n), dtype=np.int64)
    marks[np.arange(len(positions))[:, None], positions] = positions
    caches = np.maximum.accumulate(marks, axis=1)
    weights = np.stack([(caches == i).sum(axis=1) / n for i in range(n)], axis=1)
    return positions, caches, weights


def exact_task_oracle(logits: np.ndarray, label: int, budget: int) -> dict:
    positions, _, weights = schedule_table(len(logits), budget)
    losses = classification_loss(weights @ logits, label)
    minimum = float(losses.min())
    winner = int(np.flatnonzero(losses <= minimum + TOL)[0])
    return {"positions": positions[winner].tolist(), "minimum_cross_entropy": minimum,
            "selected_cross_entropy": float(losses[winner]), "enumerated_schedules": len(positions)}


def replay(videos: list, role: str, parameters: dict, classes: list, models: dict) -> tuple[pd.DataFrame, list, list]:
    rows, oracles, traces = [], [], []
    for index, video in enumerate(videos, 1):
        vid = video.entry["video_id"]; y = classes.index(video.entry["label"])
        logits = feature_logits(video.references, parameters)
        dense_prediction = int(logits.mean(axis=0).argmax())
        distances = budget_utils.distance_matrix(video.references)
        for budget in [2, 4, 8, 16]:
            if budget != 16:
                task = exact_task_oracle(logits, y, budget)
                feature = budget_utils.exact_oracle(distances, budget)
                oracles.append({"role": role, "video_id": vid, "budget": budget, "task": task, "feature": feature})
            for policy in ["dense"] if budget == 16 else POLICIES:
                for seed in range(20261014, 20261024) if policy == "random" else [None]:
                    if policy in ["task_oracle", "feature_oracle"]:
                        positions = (task if policy == "task_oracle" else feature)["positions"]
                    else:
                        plan = budget_utils.plan_deployable(video.cheap, policy, budget, models, seed=seed)
                        positions = [r["candidate_ordinal"] for r in plan if r["refresh"]]
                        if policy.startswith("value_"):
                            traces.extend({"role": role, "video_id": vid, "policy": policy, "budget": budget, **r} for r in plan)
                    assert len(positions) == budget
                    cached = cache_indices(positions)
                    output_logits = logits[cached].mean(axis=0)
                    loss = float(classification_loss(output_logits[None], y)[0])
                    error = float(distances[np.arange(16), cached].mean())
                    if budget != 16:
                        assert loss >= task["minimum_cross_entropy"] - TOL
                        assert error >= feature["exact_minimum_error"] - TOL
                    else:
                        assert error <= TOL
                    # Explicit feature replay independently checks the affine pooling shortcut.
                    np.testing.assert_allclose(output_logits, feature_logits(video.references[cached].mean(axis=0, dtype=np.float64)[None], parameters)[0], atol=1e-10, rtol=1e-10)
                    prediction = int(output_logits.argmax())
                    rows.append({"role": role, "video_id": vid, "source_video_id": video.entry["source_video_id"],
                                 "label": video.entry["label"], "label_index": y, "policy": policy, "budget": budget,
                                 "random_seed": seed, "refresh_count": budget, "cache_reuses": 16-budget, "feature_outputs": 16,
                                 "refresh_positions": json.dumps(positions), "prediction_index": prediction,
                                 "correct": int(prediction == y), "dense_prediction_index": dense_prediction,
                                 "disagrees_with_dense": int(prediction != dense_prediction), "cross_entropy": loss,
                                 "feature_cosine_error": error, "logits": json.dumps(output_logits.tolist())})
        print(f"{role}: replayed {index}/{len(videos)} videos (all exact oracle budgets)", flush=True)
    return pd.DataFrame(rows), oracles, traces


def average_random(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["role", "video_id", "source_video_id", "label", "label_index", "policy", "budget"]
    numeric = ["refresh_count", "cache_reuses", "feature_outputs", "correct", "disagrees_with_dense", "cross_entropy", "feature_cosine_error"]
    result = raw.groupby(keys, sort=True, as_index=False)[numeric].mean()
    counts = raw.groupby(keys, sort=True).size().to_numpy()
    result["trajectory_count"] = counts
    assert (result.loc[result.policy == "random", "trajectory_count"] == 10).all()
    assert (result.loc[result.policy != "random", "trajectory_count"] == 1).all()
    dense = result[result.policy == "dense"].set_index(["role", "video_id"]).correct
    result["accuracy_loss_vs_dense"] = [float(dense.loc[(r.role, r.video_id)] - r.correct) for r in result.itertuples()]
    return result


def rank_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float | None, str | None]:
    if np.ptp(x) <= TOL or np.ptp(y) <= TOL:
        return None, "constant input or outcome"
    return float(np.corrcoef(pd.Series(x).rank(method="average"), pd.Series(y).rank(method="average"))[0, 1]), None


def bootstrap_difference(deltas: np.ndarray, indices: np.ndarray) -> dict:
    assert np.isfinite(deltas).all() and indices.shape[1] == len(deltas)
    bounds = np.quantile(deltas[indices].mean(axis=1), [0.025, 0.975], method="linear")
    return {"mean": float(deltas.mean()), "median": float(np.median(deltas)), "CI95_lower": float(bounds[0]),
            "CI95_upper": float(bounds[1]), "improved": int((deltas > TOL).sum()),
            "tied": int((abs(deltas) <= TOL).sum()), "worsened": int((deltas < -TOL).sum())}


def summarize(table: pd.DataFrame, seed: int) -> tuple[dict, pd.DataFrame, dict, pd.DataFrame, list]:
    import hashlib
    aggregates, comparisons, intervals, per_class, correlations = [], [], [], [], []
    for (role, policy, budget), group in table.groupby(["role", "policy", "budget"], sort=True):
        aggregates.append({"role": role, "policy": policy, "budget": int(budget), "videos": len(group),
                           "top1_accuracy": float(group.correct.mean()),
                           "macro_top1_accuracy": float(group.groupby("label").correct.mean().mean()),
                           "cross_entropy": float(group.cross_entropy.mean()),
                           "feature_cosine_error": float(group.feature_cosine_error.mean()),
                           "accuracy_loss_vs_dense_pp": float(100*group.accuracy_loss_vs_dense.mean()),
                           "classification_disagreement_vs_dense": float(group.disagrees_with_dense.mean()),
                           "refresh_count": int(budget)})
        per_class.extend({"role": role, "policy": policy, "budget": int(budget), "label": name,
                          "videos": len(g), "accuracy": float(g.correct.mean()), "cross_entropy": float(g.cross_entropy.mean())}
                         for name, g in group.groupby("label"))
        if policy != "dense":
            dense = table[(table.role == role) & (table.policy == "dense")].set_index("video_id")
            for outcome, y in [("classification_loss", group.cross_entropy.to_numpy()),
                               ("loss_change_vs_dense", group.cross_entropy.to_numpy()-dense.loc[group.video_id].cross_entropy.to_numpy()),
                               ("disagreement_vs_dense", group.disagrees_with_dense.to_numpy()),
                               ("correctness", group.correct.to_numpy())]:
                rho, reason = rank_correlation(group.feature_cosine_error.to_numpy(), y)
                correlations.append({"role": role, "policy": policy, "budget": int(budget), "outcome": outcome,
                                     "spearman": rho, "undefined_reason": reason, "videos": len(group)})
    for role, role_table in table.groupby("role", sort=True):
        subset = role_table[role_table.budget == 4]
        sources = sorted(subset.source_video_id.unique())
        indices = np.random.Generator(np.random.PCG64(seed)).integers(0, len(sources), size=(10000, len(sources)), dtype=np.int64)
        index_digest = hashlib.sha256(indices.tobytes()).hexdigest()
        for candidate, reference in [("task_oracle", "periodic"), ("feature_oracle", "periodic"), ("task_oracle", "feature_oracle")]:
            for metric in ["cross_entropy", "correct"]:
                pivot = subset.pivot(index="source_video_id", columns="policy", values=metric).loc[sources]
                delta = (pivot[reference]-pivot[candidate]) if metric == "cross_entropy" else (pivot[candidate]-pivot[reference])
                result = bootstrap_difference(delta.to_numpy(), indices)
                interval = {"role": role, "budget": 4, "candidate": candidate, "reference": reference,
                            "metric": metric, "primary": candidate == "task_oracle" and reference == "periodic",
                            "unit": "nats per video" if metric == "cross_entropy" else "correctness fraction",
                            "resamples": 10000, "seed": seed, "rng": "numpy.Generator(PCG64)", "percentile_method": "linear",
                            "resampling_indices_sha256": index_digest, **result}
                if metric == "correct":
                    interval["mean_accuracy_difference_pp"] = 100*result["mean"]
                    interval["CI95_accuracy_difference_pp"] = [100*result["CI95_lower"], 100*result["CI95_upper"]]
                intervals.append(interval)
                for source in sources:
                    ref = float(pivot.loc[source, reference]); value = float(delta.loc[source])
                    comparisons.append({"role": role, "source_video_id": source, "budget": 4, "candidate": candidate,
                                        "reference": reference, "metric": metric, "paired_improvement": value,
                                        "relative_loss_improvement_percent": 100*value/ref if metric == "cross_entropy" and ref > TOL else None})
    return {"aggregation": "equal video weight; random averaged within video before aggregation", "rows": aggregates}, pd.DataFrame(comparisons), {"comparisons": intervals}, pd.DataFrame(per_class), correlations
