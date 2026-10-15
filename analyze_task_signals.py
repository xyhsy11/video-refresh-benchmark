"""Audit caching errors and probe signed equal-budget task value using saved data."""
from __future__ import annotations

import os
for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch
from importlib.metadata import version

import numpy as np
import pandas as pd

import budget_aware_refresh_utils as budget
import recognition_baseline_data as data
import task_aware_oracle_utils as task
import task_signal_utils as signals
from fresh_confirmation_data import strict_json
from fresh_confirmation_eval import no_fitting
from research_utils import ROOT, now, relative_path, sha256_file

FROZEN = ROOT / "results/budget_aware_refresh_v2"
SOURCES = ["analyze_task_signals.py", "task_signal_utils.py", "task_signal_report.py", "verify_task_signals.py",
           "test_task_signals.py", "task_aware_oracle_utils.py", "recognition_baseline_data.py",
           "budget_aware_refresh_utils.py", "refresh_policy_utils.py", "collect_cache_drift.py",
           "fresh_confirmation_eval.py", "fresh_confirmation_data.py", "research_utils.py"]


def csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, float_format="%.17g", na_rep="")


def identities(directory: Path) -> tuple[dict, dict, dict, dict]:
    manifest = data.read(directory / "dataset_manifest.json")
    data.validate_membership(manifest["videos"])
    head = data.read(directory / "classifiers/20/parameters.json")
    expected = data.read(directory / "frozen_models.json")["parameter_sha256"]["20"]
    prior_metadata = data.read(directory / "metadata.json")
    assert sha256_file(directory / "dataset_manifest.json") == prior_metadata["dataset_manifest_sha256"]
    assert expected == prior_metadata["classifier_sha256"]["20"]
    assert sha256_file(directory / "classifiers/20/parameters.json") == expected
    assert data.read(directory / "classifiers/20/selection.json")["parameters_sha256"] == expected
    prior_protocol = data.read(directory / "protocol.json")
    assert manifest["classes"] == prior_protocol["classes"]
    assert prior_protocol["class_to_index"] == {c: i for i, c in enumerate(manifest["classes"])}
    assert manifest["classes"] == data.read(ROOT / "results/cache_drift_v2/split_manifest.json")["classes"]
    assert np.asarray(head["weights"]).shape == (20, 512) and min(head["scales"]) > 0
    frozen = data.read(FROZEN / "fresh_test_protocol.json")
    assert sha256_file(FROZEN / "fitted_predictors.json") == frozen["fixed_models_sha256"]
    controllers = data.read(FROZEN / "fitted_predictors.json")
    selected = [e for e in manifest["videos"] if e["recognition_role"] != "train"]
    assert set(controllers["training_source_ids"]).isdisjoint(e["source_video_id"] for e in selected)
    paths = [directory / p for p in ["protocol.json", "metadata.json", "dataset_manifest.json", "frozen_models.json",
             "classifiers/20/parameters.json", "classifiers/20/selection.json", "per_video_metrics.csv"]]
    paths += [FROZEN / p for p in ["fitted_predictors.json", "fresh_test_protocol.json", "development_split.json"]]
    paths += [ROOT / "results/cache_drift_v2/split_manifest.json"]
    for e in selected:
        paths.append(ROOT / e["reference_directory"] / "collection_metadata.json")
        for name, digest in e["reference_artifact_sha256"].items():
            p = ROOT / e["reference_directory"] / name
            if not p.is_file():
                raise FileNotFoundError(f"Required saved artifact missing: {relative_path(p)}")
            assert sha256_file(p) == digest, relative_path(p)
            paths.append(p)
    hashes = {relative_path(p): sha256_file(p) for p in sorted(set(paths))}
    return manifest, head, controllers["models"], hashes


def original_inventory() -> dict:
    # Git object identities only: do not parse or evaluate the completed fresh cohort.
    rows = subprocess.check_output(["git", "ls-files", "-s", "-z"], cwd=ROOT).decode().split("\0")
    return {name: header.split()[1] for row in rows if row
            for header, name in [row.split("\t", 1)] if name not in ["README.md", ".gitignore"]}


def preserved(protocol: dict) -> None:
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only", protocol["starting_commit"]], cwd=ROOT).decode().splitlines())
    assert not changed.intersection(protocol["previous_tracked_blob_identities"]), "Historical artifacts modified"
    # Include unstaged files, including modifications after staging.
    unstaged = set(subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT).decode().splitlines())
    assert not unstaged.intersection(protocol["previous_tracked_blob_identities"])


def freeze(directory: Path, output: Path, seed: int) -> dict:
    manifest, _, _, hashes = identities(directory)
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCES}
    if (output / "protocol.json").exists():
        p = data.read(output / "protocol.json")
        assert p["seed"] == seed and p["input_directory"] == relative_path(directory)
        assert p["source_sha256"] == source_hashes and p["input_sha256"] == hashes
        preserved(p)
        return p
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty output has no protocol; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    folds = signals.make_folds(manifest["videos"], manifest["classes"], seed)
    protocol = {"experiment": "task_signal_diagnostic_v1", "frozen_at_utc": now(),
        "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "input_directory": relative_path(directory), "source_sha256": source_hashes, "input_sha256": hashes,
        "previous_tracked_blob_identities": original_inventory(), "seed": seed,
        "classes": manifest["classes"], "classifier": "frozen recognition_baseline_v2 primary classifier/20",
        "cohorts": {"development": 100, "exploratory": 40}, "examples_per_video": 9,
        "reference_schedule": signals.PERIODIC, "early_moves": "replace r in 4,8,12 with t=r-3,r-2,r-1; identical history before t",
        "target": "refresh_gain = CE(periodic,true label)-CE(early,true label); signed nats, no clipping or exclusion; not global action value",
        "feature_groups": signals.GROUPS,
        "age_definition": "t/15; seconds since last refresh; remaining refresh count before t; next planned refresh ordinal minus t",
        "motion_definition": "original six CPU descriptors; current frame and decoded frames strictly after last refresh through t, inclusive; zero magnitudes plus availability flag/fraction when missing",
        "prediction_definition": "softmax entropy(nats),max probability,top-two probability difference of cached logits; same summaries of mean past output logits at candidates0..t-1; frozen affine head, no true-label quantities",
        "privileged_diagnostic": "current feature cosine drift uses f_t and is evaluator-only",
        "folds": folds["folds"], "fold_rule": folds["rule"], "groups": "whole source videos; one video/class/fold",
        "regression": {"alpha": 1.0, "objective": "sum squared residuals + alpha*coefficient norm squared",
                       "standardization": "fold-training population means/std; std<=1e-12 uses scale1",
                       "intercept": "unpenalized", "arithmetic": "float64", "predictions": "signed, unclipped"},
        "baselines": ["constant zero", "fold-training mean"],
        "primary": "development out-of-fold equal-video MAE(A)-MAE(D)",
        "bootstrap": {"resamples": 10000, "seed": seed, "rng": "PCG64", "percentile_method": "linear",
                      "unit": "whole source video; all nine examples together; shared indices across groups within cohort",
                      "uncertainty": "conditional on saved fitted probes; no full training/model-selection uncertainty"},
        "additional_metrics": ["equal-video RMSE", "descriptive example and per-video Spearman",
                               "three-class sign accuracy,macro recall", "positive gain AUROC",
                               "per-cohort/per-policy feature error vs delta CE and absolute error",
                               "harmful transition counts conditional on dense correct; disagreement separately"],
        "ties": "absolute value<=1e-12 only for floating-point ties; regression retains every signed target",
        "sequence": "save all development OOF predictions; fit/freeze probes on all development; then load exploratory arrays and evaluate once",
        "actual_RGB_encoder_calls": 0, "decodes": 0,
        "limitations": ["development used by recognition regularization selection", "exploratory previously inspected",
                       "offline full-clip candidates and future-reference codec vectors",
                       "no fresh-confirmation cohort parsed", "no controller,latency,speedup or independent confirmation claim",
                       "repeated examples are not independent source videos"]}
    strict_json(output / "protocol.json", protocol)
    strict_json(output / "fold_manifest.json", folds)
    strict_json(output / "environment.json", {"recorded_at_utc": now(), "python": platform.python_version(),
        "os": platform.platform(), "packages": {n: version(n) for n in ["numpy", "pandas", "torch", "torchvision", "av", "matplotlib", "Pillow"]},
        "analysis_device": "CPU saved-array float64 arithmetic", "actual_RGB_encoder_calls": 0, "decodes": 0,
        "CUDA_required": False})
    return protocol


def evaluate(videos: list, cohort: str, head: dict, models: dict, classes: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    clips, examples = [], []
    for video in videos:
        e = video.entry
        y = classes.index(e["label"])
        identity = {"cohort": cohort, "video_id": e["video_id"], "source_video_id": e["source_video_id"],
                    "label": e["label"], "label_index": y}
        logits = task.feature_logits(video.references, head)
        distances = budget.distance_matrix(video.references)
        dense = logits.mean(axis=0)
        dense_pred = int(dense.argmax())
        dense_ce = float(task.classification_loss(dense[None], y)[0])
        baseline_ce = None
        for policy in ["dense", "periodic", "value_age_motion_budget"]:
            if policy == "dense":
                positions = list(range(16))
            else:
                plan = budget.plan_deployable(video.cheap, policy, 4, models)
                positions = [r["candidate_ordinal"] for r in plan if r["refresh"]]
                assert len(positions) == 4
            cache = task.cache_indices(positions)
            output = logits[cache].mean(axis=0)
            pred = int(output.argmax())
            ce = float(task.classification_loss(output[None], y)[0])
            transition = ("both_correct" if dense_pred == y and pred == y else
                          "dense_correct_policy_wrong" if dense_pred == y else
                          "dense_wrong_policy_correct" if pred == y else "both_wrong")
            clips.append({**identity, "policy": policy,
                "feature_cosine_error": float(distances[np.arange(16), cache].mean()),
                "dense_prediction": dense_pred, "policy_prediction": pred,
                "dense_correct": int(dense_pred == y), "policy_correct": int(pred == y),
                "prediction_disagreement": int(pred != dense_pred), "dense_CE": dense_ce, "policy_CE": ce,
                "delta_CE": ce-dense_ce, "transition": transition, "refresh_positions": json.dumps(positions),
                "simulated_encoder_calls": len(positions), "feature_outputs": 16})
            if policy == "periodic":
                baseline_ce = ce
        assert baseline_ce is not None
        baseline_cache = task.cache_indices(signals.PERIODIC)
        for planned in [4, 8, 12]:
            for t in range(planned-3, planned):
                r = planned-4
                positions = sorted([p for p in signals.PERIODIC if p != planned]+[t])
                early_cache = task.cache_indices(positions)
                assert len(set(positions)) == 4 and positions[0] == 0
                np.testing.assert_array_equal(baseline_cache[:t], early_cache[:t])
                early_logits = logits[early_cache].mean(axis=0)
                early_ce = float(task.classification_loss(early_logits[None], y)[0])
                examples.append({**identity, "example_id": f"{e['video_id']}:move{planned}to{t}",
                    "t": t, "last_refresh": r, "replaced_refresh": planned,
                    "cached_frame_index": video.cheap.indices[r], "current_frame_index": video.cheap.indices[t],
                    "interval_start_frame": video.cheap.indices[r]+1, "interval_end_frame": video.cheap.indices[t],
                    "baseline_positions": json.dumps(signals.PERIODIC), "early_positions": json.dumps(positions),
                    "simulated_encoder_calls_each": 4, "feature_outputs_each": 16,
                    **signals.observed_inputs(video, t, head),
                    "privileged_current_drift": float(distances[t, r]),
                    "baseline_CE": baseline_ce, "early_CE": early_ce,
                    signals.TARGET: baseline_ce-early_ce})
    return pd.DataFrame(clips), pd.DataFrame(examples)


def prediction_rows(table: pd.DataFrame, cohort: str, fold: int, models: dict, mean: float) -> pd.DataFrame:
    rows = []
    for name in signals.PREDICTORS:
        values = np.zeros(len(table)) if name == "zero" else np.full(len(table), mean) if name == "training_mean" else signals.predict(table, models[name])
        for e, value in zip(table.to_dict("records"), values):
            rows.append({key: e[key] for key in ["example_id", "video_id", "source_video_id", "t", "last_refresh", signals.TARGET]}
                        | {"cohort": cohort, "fold": fold, "predictor": name, "prediction": float(value)})
    return pd.DataFrame(rows)


def fit_probes(development: pd.DataFrame, folds: dict, output: Path) -> dict:
    all_predictions, fold_models = [], {}
    for f in folds["folds"]:
        tr = development[development.source_video_id.isin(f["training_sources"])]
        va = development[development.source_video_id.isin(f["validation_sources"])]
        assert len(tr) == 720 and len(va) == 180 and set(tr.source_video_id).isdisjoint(va.source_video_id)
        models = {name: signals.fit_ridge(tr, name) for name in signals.GROUPS}
        fold_models[str(f["fold"])] = {"models": models, "training_mean": float(tr.refresh_gain.mean())}
        all_predictions.append(prediction_rows(va, "development_oof", f["fold"], models, tr.refresh_gain.mean()))
    csv(output / "out_of_fold_predictions.csv", pd.concat(all_predictions, ignore_index=True))
    # Full-development fits happen only after all OOF predictions have been saved.
    state = {"development_predictions_saved_at_utc": now(), "oof_predictions_sha256": sha256_file(output / "out_of_fold_predictions.csv"),
        "fold_models": fold_models, "full_development_models": {g: signals.fit_ridge(development, g) for g in signals.GROUPS},
        "full_development_mean": float(development.refresh_gain.mean()), "frozen_at_utc": now()}
    strict_json(output / "probe_parameters.json", state)
    return state


def frozen_only():
    stack = ExitStack()
    stack.enter_context(no_fitting())
    for target in ["task_aware_oracle_utils.fit_classifier", "benchmark.Encoder.__init__",
                   "urllib.request.urlopen", "av.open"]:
        stack.enter_context(patch(target, side_effect=AssertionError("Saved-feature diagnostic forbids encoder,decoding,downloads,recognition fitting")))
    return stack


def run(args) -> None:
    directory = (ROOT / args.input).resolve()
    output = (ROOT / args.output).resolve()
    if args.audit_only and not (output / "verification.json").is_file():
        raise FileNotFoundError("Audit requires an existing evaluated run and verification.json")
    p = freeze(directory, output, args.seed)
    if args.audit_only or ((output / "status.json").exists() and data.read(output / "status.json")["status"] == "complete"):
        from verify_task_signals import audit
        print(json.dumps(audit(directory, output, p), indent=2))
        return
    strict_json(output / "status.json", {"status": "running", "started_at_utc": now()})
    manifest, head, controllers, _ = identities(directory)
    with frozen_only():
        dev = data.load_role(manifest["videos"], "development")
        dev_clips, dev_examples = evaluate(dev, "development", head, controllers, p["classes"])
        csv(output / "development_examples.csv", dev_examples)
        state = fit_probes(dev_examples, data.read(output / "fold_manifest.json"), output)
        print("All development OOF predictions saved and probes frozen; opening exploratory references.", flush=True)
        exploratory = data.load_role(manifest["videos"], "exploratory")
        ex_clips, ex_examples = evaluate(exploratory, "exploratory", head, controllers, p["classes"])
        csv(output / "exploratory_predictions.csv", prediction_rows(ex_examples, "exploratory", -1,
            state["full_development_models"], state["full_development_mean"]))
    examples = pd.concat([dev_examples, ex_examples], ignore_index=True)
    clips = pd.concat([dev_clips, ex_clips], ignore_index=True)
    assert len(examples) == 1260 and len(clips) == 420
    csv(output / "clip_error_analysis.csv", clips)
    csv(output / "counterfactual_examples.csv", examples)
    predictions = pd.concat([pd.read_csv(output / name, float_precision="round_trip") for name in
                             ["out_of_fold_predictions.csv", "exploratory_predictions.csv"]], ignore_index=True)
    summary, per_video, intervals, indices = signals.summarize_predictions(predictions, args.seed)
    summary["clip_error_analysis"] = signals.clip_summary(clips)
    summary["privileged_drift_vs_gain"] = [
        {"cohort": role, "videos": g.source_video_id.nunique(), "examples": len(g),
         **signals.correlation(g.privileged_current_drift, g.refresh_gain),
         "positive_gain_AUROC": signals.auc(g.privileged_current_drift, g.refresh_gain > signals.TOL),
         "gain_quantiles": {str(q): float(g.refresh_gain.quantile(q)) for q in [0, .1, .25, .5, .75, .9, 1]}}
        for role, g in examples.groupby("cohort", sort=True)]
    strict_json(output / "aggregate_metrics.json", summary)
    strict_json(output / "bootstrap_intervals.json", intervals)
    csv(output / "per_video_probe_metrics.csv", per_video)
    np.savez_compressed(output / "bootstrap_resampling_indices.npz", **indices)
    strict_json(output / "metadata.json", {"completed_analysis_at_utc": now(), "actual_RGB_encoder_calls": 0,
        "video_decodes": 0, "classification_training_calls": 0, "downloads": 0,
        "development_videos": 100, "exploratory_videos": 40, "development_examples": 900,
        "exploratory_examples": 360, "oof_predictions": 5400, "exploratory_predictions": 2160,
        "probe_parameters_sha256": sha256_file(output / "probe_parameters.json"), "input_sha256": p["input_sha256"]})
    from verify_task_signals import audit
    verification = audit(directory, output, p)
    tests = subprocess.run([sys.executable, "-m", "unittest", "test_task_signals", "-v"], cwd=ROOT, capture_output=True, text=True)
    assert tests.returncode == 0, tests.stdout+tests.stderr
    release = subprocess.run([sys.executable, "validate_release.py"], cwd=ROOT, capture_output=True, text=True)
    assert release.returncode == 0, release.stdout+release.stderr
    verification.update(focused_tests={"returncode": 0, "output": tests.stdout+tests.stderr},
                        release_validation={"returncode": 0, "output": release.stdout+release.stderr})
    strict_json(output / "verification.json", verification)
    from task_signal_report import report
    report(output)
    strict_json(output / "status.json", {"status": "complete", "completed_at_utc": now(), "blocker": None})
    files = sorted(f for f in output.rglob("*") if f.is_file() and f.name != "SHA256SUMS.txt")
    (output / "SHA256SUMS.txt").write_text("".join(f"{sha256_file(f)}  {f.relative_to(output).as_posix()}\n" for f in files), encoding="utf-8")
    print(f"Complete and verified: {relative_path(output)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/recognition_baseline_v2"))
    parser.add_argument("--output", type=Path, default=Path("results/task_signal_diagnostic_v1"))
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        message = f"{type(exc).__name__}: {str(exc).replace(str(ROOT), '.')}"
        status_path = (ROOT / args.output) / "status.json"
        if (status_path.parent / "protocol.json").exists() and not args.audit_only:
            existing = data.read(status_path) if status_path.exists() else {}
            if existing.get("status") != "complete":
                strict_json(status_path, {"status": "incomplete", "blocked_at_utc": now(), "blocker": message})
        print(f"INCOMPLETE: {message}", file=sys.stderr)
        raise
