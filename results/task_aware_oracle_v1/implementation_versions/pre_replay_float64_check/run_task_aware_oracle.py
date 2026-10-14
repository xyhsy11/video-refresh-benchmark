"""Train a small action head and replay exact task-aware hindsight oracles."""
from __future__ import annotations

import os
for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

import task_aware_oracle_utils as u
from fresh_confirmation_data import strict_json, check_frozen
from motion_utils import current_environment
from research_utils import ROOT, now, relative_path, sha256_file

SOURCES = ["run_task_aware_oracle.py", "task_aware_oracle_utils.py", "task_aware_oracle_report.py", "test_task_aware_oracle.py"]


def csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, float_format="%.17g", na_rep="")


def inventory() -> dict:
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    return {name: sha256_file(ROOT/name) for name in files if name and name not in ["README.md", ".gitignore"]}


def audit(output: Path, input_dir: Path) -> dict:
    """Independent saved-result audit, never trains a classifier or runs an encoder."""
    protocol = u.read_json(output/"protocol.json")
    metadata = u.read_json(output/"metadata.json")
    for name, digest in metadata["source_code_sha256"].items(): assert sha256_file(ROOT/name) == digest, name
    for name, digest in metadata["protected_original_sha256"].items(): assert sha256_file(ROOT/name) == digest, name
    for name, digest in metadata["input_artifact_sha256"].items(): assert sha256_file(input_dir/name) == digest, name
    model = u.read_json(output/"classifier_parameters.json")
    assert sha256_file(output/"classifier_parameters.json") == metadata["classifier_sha256"]
    config = u.read_json(output/"model_configuration.json"); classes = config["classes"]
    raw = pd.read_csv(output/"trajectory_metrics.csv", float_precision="round_trip")
    assert len(raw) == 3680 and not raw.duplicated(["role", "video_id", "policy", "budget", "random_seed"]).any()
    for (role, vid), group in raw.groupby(["role", "video_id"]):
        with np.load(input_dir/f"features/{vid}.npz", allow_pickle=False) as data: features = data["features"].copy()
        logits = u.feature_logits(features, model); y = int(group.label_index.iloc[0])
        assert group.label.nunique() == 1 and classes[y] == group.label.iloc[0]
        distances = u.budget_utils.distance_matrix(features)
        for row in group.itertuples():
            positions = json.loads(row.refresh_positions); assert len(positions) == row.budget == row.refresh_count
            assert row.cache_reuses == 16-row.budget and row.feature_outputs == 16
            cached = u.cache_indices(positions)
            explicit = u.feature_logits(features[cached].mean(axis=0)[None], model)[0]
            np.testing.assert_allclose(explicit, json.loads(row.logits), atol=1e-10, rtol=1e-10)
            maximum = explicit.max()
            independent_loss = np.log(np.exp(explicit-maximum).sum()) + maximum-explicit[y]
            assert abs(independent_loss-row.cross_entropy) <= 1e-10
            assert int(explicit.argmax() == y) == row.correct
            assert int(explicit.argmax()) == row.prediction_index
            assert abs(distances[np.arange(16), cached].mean()-row.feature_cosine_error) <= 1e-12
            assert int(explicit.argmax() != logits.mean(axis=0).argmax()) == row.disagrees_with_dense
        for budget in [2, 4, 8]:
            result = u.exact_task_oracle(logits, y, budget)
            stored = group[(group.policy == "task_oracle") & (group.budget == budget)].iloc[0]
            assert result["positions"] == json.loads(stored.refresh_positions)
            assert (group[group.budget == budget].cross_entropy >= result["minimum_cross_entropy"]-u.TOL).all()
    table = u.average_random(raw)
    stored_table = pd.read_csv(output/"per_video_metrics.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(table, stored_table, check_dtype=False, atol=1e-12, rtol=1e-12)
    aggregate, pairs, intervals, _, _ = u.summarize(stored_table, protocol["seed"])
    assert aggregate == u.read_json(output/"aggregate_metrics.json")
    assert intervals == u.read_json(output/"bootstrap_intervals.json")
    for comparison in intervals["comparisons"]:
        matched = pairs[(pairs.role == comparison["role"]) & (pairs.candidate == comparison["candidate"]) &
                        (pairs.reference == comparison["reference"]) & (pairs.metric == comparison["metric"])].sort_values("source_video_id")
        deltas = matched.paired_improvement.to_numpy()
        idx = np.random.default_rng(protocol["seed"]).integers(0, 40, size=(10000, 40), dtype=np.int64)
        bounds = np.quantile([sum(deltas[row])/40 for row in idx], [.025, .975], method="linear")
        np.testing.assert_allclose(bounds, [comparison["CI95_lower"], comparison["CI95_upper"]], atol=1e-12, rtol=1e-12)
    return {"passed": True, "raw_trajectories": 3680, "per_video_rows": 1520,
            "all_logits_losses_predictions_replay_and_exact_task_schedules_reproduced": True,
            "bootstrap_independently_reproduced_from_video_pairs": True,
            "previous_tracked_files_unchanged": len(metadata["protected_original_sha256"])}


def execute(args) -> Path:
    output = args.output
    if output.exists():
        output = output.with_name(output.name + "_" + now().replace(":", "").replace("-", "").split(".")[0].replace("+", ""))
    output.mkdir(parents=True, exist_ok=False)
    protected = inventory()
    protocol = {"experiment": "task_aware_oracle_v1", "frozen_at_utc": now(), "seed": args.seed,
                "splits": {"train": 80, "development": 40, "exploratory_test": 40},
                "fresh_confirmation_data_use": "identity exclusions only; no features, motion, labels, performance or tuning",
                "classifier": "frozen ResNet18 references; temporal mean; training-only population normalization; linear 20-class head",
                "regularization_grid": [.001, .01, .1, 1.0], "objective": "mean cross entropy + lambda/2 * squared weights; intercept unpenalized",
                "training": "CPU float64 torch LBFGS strong_wolfe; max_iter400/max_eval800; deterministic same initialization per lambda",
                "selection": "highest development dense top1; then lowest development cross entropy; then earliest grid order; no retraining on development",
                "quality_gate": {"near_chance": "dense accuracy <=10% (chance5%)", "exploratory_adequate": "dense development AND exploratory test top1 >=50%", "limitation": "only four training and two evaluation clips per class; adequacy is an exploratory gate, not validated recognition"},
                "decision_gate": "REVISE if classifier inadequate; otherwise CONTINUE cautiously if B4 hindsight correctness gain >=5pp and mean CE gain>=0.05 nats on exploratory set; otherwise PIVOT for this representation/head. No causal feasibility established by oracle alone.",
                "budgets": [2, 4, 8, 16], "primary": "task oracle versus periodic at B4, separately on development and previously used exploratory test",
                "policies": ["dense", *u.POLICIES], "random_seeds": list(range(20261014, 20261024)),
                "controllers": "unchanged budget_aware_refresh_v2 models trained on same80 sources; original motion inputs contain future-reference codec vectors, so offline",
                "oracle": "exact all schedules with mandatory0; CE true-label objective; lexicographically first within1e-12; feature oracle unchanged",
                "exact_schedules_per_video": {"2": 15, "4": 455, "8": 6435},
                "bootstrap": "10,000 paired whole-source-video resamples; PCG64 seed; linear2.5/97.5 percentiles; same matrix across comparisons within each cohort",
                "secondary_B4_comparisons": [["feature_oracle", "periodic"], ["task_oracle", "feature_oracle"]],
                "example_rule": "development B4 largest three feature-oracle minus task-oracle CE differences; video-ID tie break; chronological margins descriptive only",
                "actual_encoder_calls": 0, "limits": ["true-label/future schedule selection optimism", "not ordinary held-out generalization", "test cohort previously evaluated", "no latency/speedup claims", "no semantic temporal localization annotations"]}
    strict_json(output/"protocol.json", protocol)
    strict_json(output/"environment.json", current_environment())
    train, dev, test_entries, inputs, annotation_table = u.load_dataset(args.input, args.frozen_run, args.annotations)
    fresh_ids = {e["source_video_id"] for e in u.read_json(ROOT/"results/fresh_refresh_confirmation_v1/frozen_test_manifest.json")["videos"]}
    assert not set(annotation_table.source_video_id) & fresh_ids
    classes = inputs["classes"]
    frozen_protocol = u.read_json(args.frozen_run/"fresh_test_protocol.json")
    models, _ = check_frozen(args.frozen_run, frozen_protocol)
    fitted = u.read_json(args.frozen_run/"fitted_predictors.json")
    assert set(fitted["training_source_ids"]) == {v.entry["source_video_id"] for v in train}
    csv(output/"annotation_records.csv", annotation_table)
    split_rows = [{"role": role, "entry": v.entry} for role, vs in [("train", train), ("development", dev)] for v in vs]
    split_rows += [{"role": "exploratory_test", "entry": e} for e in test_entries]
    strict_json(output/"dataset_split.json", {"classes": classes, "counts": protocol["splits"], "videos": split_rows,
                                               "input_manifest_sha256": inputs["manifest_sha256"], "development_split_sha256": inputs["development_split_sha256"], "annotations_sha256": inputs["annotation_sha256"]})
    x = np.stack([v.references.astype(np.float64).mean(axis=0) for v in train])
    y = np.array([classes.index(v.entry["label"]) for v in train])
    assert set(y) == set(range(20))
    search, candidates = [], []
    for index, regularization in enumerate(protocol["regularization_grid"]):
        parameters, info = u.fit_classifier(x, y, regularization, args.seed)
        train_metrics, _ = u.dense_metrics(train, classes, parameters)
        dev_metrics, _ = u.dense_metrics(dev, classes, parameters)
        search.append({"grid_order": index, **info, "train_accuracy": train_metrics["top1_accuracy"],
                       "development_accuracy": dev_metrics["top1_accuracy"], "development_cross_entropy": dev_metrics["cross_entropy"]})
        candidates.append(parameters)
        print(f"Classifier lambda={regularization}: train={train_metrics['top1_accuracy']:.3f}, development={dev_metrics['top1_accuracy']:.3f}", flush=True)
    winner = min(range(len(search)), key=lambda i: (-search[i]["development_accuracy"], search[i]["development_cross_entropy"], i))
    parameters = candidates[winner]
    strict_json(output/"classifier_parameters.json", parameters)
    classifier_hash = sha256_file(output/"classifier_parameters.json")
    configuration = {"frozen_at_utc": now(), "classes": classes, "class_to_index": {c: i for i, c in enumerate(classes)},
                     "selected_grid_order": winner, "selected_regularization": search[winner]["regularization"],
                     "training_configuration": protocol, "hyperparameter_search": search, "parameter_artifact": "classifier_parameters.json",
                     "parameter_sha256": classifier_hash, "training_source_ids": sorted(v.entry["source_video_id"] for v in train),
                     "encoder_identity": inputs["collection_configuration"]["model"], "arithmetic": "CPU float64; original saved features float32",
                     "test_features_loaded_when_frozen": False}
    strict_json(output/"model_configuration.json", configuration)
    # This is the first original-test feature/motion read, after model and choices freeze.
    exploratory, test_hashes = u.load_exploratory(args.input, test_entries)
    inputs["artifact_sha256"].update(test_hashes)
    # Check real-controller information boundaries on development sources only.
    isolation_cases = 0
    for video in dev[:2]:
        changed_motion = tuple({**row, "mean_magnitude_px": row["mean_magnitude_px"]+1234.,
                                "p90_magnitude_px": row["p90_magnitude_px"]+1234.}
                               if index > video.cheap.indices[8] else dict(row)
                               for index, row in enumerate(video.cheap.motion))
        future = replace(video.cheap, motion=changed_motion)
        changed_rgb_label = replace(video, references=video.references[::-1], entry={**video.entry, "label": "unused changed evaluation target"})
        for budget in [2, 4, 8]:
            for policy in ["value_age_budget", "value_age_motion_budget"]:
                original = u.budget_utils.plan_deployable(video.cheap, policy, budget, models)
                mutation = u.budget_utils.plan_deployable(future, policy, budget, models)
                assert original[:9] == mutation[:9], "Future-motion leakage"
                assert original == u.budget_utils.plan_deployable(changed_rgb_label.cheap, policy, budget, models)
                isolation_cases += 1
    dense_results, dense_tables = {}, []
    counts = Counter(annotation_table[annotation_table.role == "train"].label)
    majority = next(c for c in classes if counts[c] == max(counts.values()))
    for role, videos in [("train", train), ("development", dev), ("exploratory_test", exploratory)]:
        metrics, table = u.dense_metrics(videos, classes, parameters)
        metrics.update(chance_accuracy=1/20, majority_class=majority, majority_accuracy=sum(v.entry["label"] == majority for v in videos)/len(videos))
        dense_results[role] = metrics; dense_tables.append(table.assign(role=role))
    strict_json(output/"classifier_metrics.json", dense_results)
    csv(output/"dense_classifier_predictions.csv", pd.concat(dense_tables, ignore_index=True))
    raw_tables, oracle_rows, controller_rows = [], [], []
    with patch.object(u, "fit_classifier", side_effect=AssertionError("Classifier frozen; evaluation must never fit")):
        for role, videos in [("development", dev), ("exploratory_test", exploratory)]:
            raw, oracles, traces = u.replay(videos, role, parameters, classes, models)
            raw_tables.append(raw); oracle_rows.extend(oracles); controller_rows.extend(traces)
    raw = pd.concat(raw_tables, ignore_index=True); table = u.average_random(raw)
    assert len(raw) == 3680 and len(table) == 1520
    csv(output/"trajectory_metrics.csv", raw); csv(output/"per_video_metrics.csv", table)
    csv(output/"controller_traces.csv", pd.DataFrame(controller_rows))
    strict_json(output/"oracle_schedules.json", {"privileged": True, "true_labels_used_by_task_oracle": True, "rows": oracle_rows})
    aggregates, comparisons, intervals, per_class, correlations = u.summarize(table, args.seed)
    strict_json(output/"aggregate_metrics.json", aggregates); strict_json(output/"bootstrap_intervals.json", intervals)
    strict_json(output/"feature_task_correlations.json", {"scope": "within policy/budget/cohort, source videos; descriptive, not causal", "rows": correlations})
    csv(output/"paired_comparisons.csv", comparisons); csv(output/"per_class_metrics.csv", per_class)
    metadata = {"started_at_utc": protocol["frozen_at_utc"], "completed_replay_at_utc": now(), "status": "replay_complete_pending_audit",
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_code_sha256": {n: sha256_file(ROOT/n) for n in SOURCES}, "protected_original_sha256": protected,
                "input_artifact_sha256": inputs["artifact_sha256"], "original_collection": inputs["collection_configuration"],
                "actual_encoder_calls": 0, "actual_video_decodes": 0, "classifier_sha256": classifier_hash,
                "frozen_policy_models_sha256": sha256_file(args.frozen_run/"fitted_predictors.json"),
                "input_directory": relative_path(args.input),
                "input_loading": "only original80train/40dev/40exploratory; validation arrays never opened; fresh confirmation identity manifest only",
                "counts": {"train": 80, "development": 40, "exploratory_test": 40, "reference_features_reused": 2560,
                           "raw_trajectories": 3680, "per_video_rows": 1520, "task_objective_schedule_evaluations": 80*(15+455+6435)}}
    strict_json(output/"metadata.json", metadata)
    verification = audit(output, args.input)
    from task_aware_oracle_report import report
    report(output)
    tests = subprocess.run([sys.executable, "-m", "unittest", "test_task_aware_oracle", "-v"], cwd=ROOT, capture_output=True, text=True)
    assert tests.returncode == 0, tests.stdout+tests.stderr
    release = subprocess.run([sys.executable, "validate_release.py"], cwd=ROOT, capture_output=True, text=True)
    assert release.returncode == 0, release.stdout+release.stderr
    for name, digest in protected.items(): assert sha256_file(ROOT/name) == digest, name
    assert sha256_file(output/"classifier_parameters.json") == classifier_hash
    check_frozen(args.frozen_run, frozen_protocol)
    verification.update(split_source_and_hash_disjoint=True, annotation_labels_verified=True, test_loaded_only_after_classifier_freeze=True,
                        fresh_confirmation_features_or_motion_opened=0, classifier_unchanged_across_policies=True,
                        exact_budget_and_latest_cache_semantics=True, exact_oracle_enumeration_counts={"2": 15, "4": 455, "8": 6435},
                        real_deployable_information_isolation_cases=isolation_cases,
                        future_motion_cannot_change_earlier_decisions=True, evaluator_RGB_and_labels_cannot_change_deployable_schedules=True,
                        focused_unit_tests={"returncode": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr},
                        release_validator={"returncode": release.returncode, "stdout": release.stdout})
    strict_json(output/"verification.json", verification)
    metadata.update(status="complete", completed_at_utc=now()); strict_json(output/"metadata.json", metadata)
    files = sorted(p for p in output.rglob("*") if p.is_file() and p.suffix not in [".tmp", ".part"] and p.name != "SHA256SUMS.txt")
    (output/"SHA256SUMS.txt").write_text("".join(f"{sha256_file(p)}  {p.relative_to(output).as_posix()}\n" for p in files), encoding="utf-8")
    print(f"Complete: {relative_path(output)}; 80 training,40 development,40 exploratory; actual encoder calls0.", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/cache_drift_v2"))
    parser.add_argument("--frozen-run", type=Path, default=Path("results/budget_aware_refresh_v2"))
    parser.add_argument("--annotations", type=Path, default=Path("data/source/cache_drift_v2/val.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/task_aware_oracle_v1"))
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(audit(args.output, args.input))
    else:
        try: execute(args)
        except Exception:
            print("INCOMPLETE: preserve partial outputs; no result may be claimed complete until verification passes.", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
