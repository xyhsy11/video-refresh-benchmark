"""Fixed 3-input x 2-predictor saved-feature task-value ablation."""
from __future__ import annotations

import os
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "1"

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd

import analyze_task_signals as old
import recognition_baseline_data as data
import task_signal_ablation_utils as u
from fresh_confirmation_data import strict_json
from research_utils import ROOT, relative_path, sha256_file, now
from task_aware_oracle_utils import feature_logits

NEW_SOURCES = ["analyze_task_signal_ablation.py", "task_signal_ablation_utils.py",
               "verify_task_signal_ablation.py", "task_signal_ablation_report.py", "test_task_signal_ablation.py"]


def csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, float_format="%.17g", na_rep="")


def identities(recognition: Path, diagnostic: Path) -> tuple[dict, dict, dict]:
    for path in [recognition/"classifiers/20/parameters.json", diagnostic/"SHA256SUMS.txt"]:
        if not path.is_file():
            raise FileNotFoundError(f"Required saved prerequisite missing: {relative_path(path)}")
    manifest, head, _, hashes = old.identities(recognition)
    p = data.read(diagnostic/"protocol.json")
    assert p["input_sha256"] == hashes and p["input_directory"] == relative_path(recognition)
    assert data.read(diagnostic/"status.json")["status"] == "complete"
    assert data.read(diagnostic/"verification.json")["passed"]
    assert {name: sha256_file(ROOT/name) for name in p["source_sha256"]} == p["source_sha256"]
    ledger = dict((n, d) for d, n in (line.split("  ", 1) for line in
                  (diagnostic/"SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()))
    for name, digest in ledger.items():
        path = diagnostic/name
        if not path.is_file():
            raise FileNotFoundError(f"Required prior artifact missing: {relative_path(path)}")
        assert sha256_file(path) == digest, relative_path(path)
        hashes[relative_path(path)] = digest
    hashes[relative_path(diagnostic/"SHA256SUMS.txt")] = sha256_file(diagnostic/"SHA256SUMS.txt")
    folds = data.read(diagnostic/"fold_manifest.json")
    selected = [e for e in manifest["videos"] if e["recognition_role"] == "development"]
    labels = {e["source_video_id"]: e["label"] for e in selected}
    assert len(folds["videos"]) == 100 and len(folds["folds"]) == 5
    for f in folds["folds"]:
        assert len(f["training_sources"]) == 80 and len(f["validation_sources"]) == 20
        assert set(f["training_sources"]).isdisjoint(f["validation_sources"])
        assert set(f["training_sources"]) | set(f["validation_sources"]) == set(labels)
        assert sorted(labels[s] for s in f["validation_sources"]) == sorted(manifest["classes"])
    return manifest, head, hashes


def freeze(recognition: Path, diagnostic: Path, output: Path, seed: int) -> dict:
    manifest, _, hashes = identities(recognition, diagnostic)
    sources = NEW_SOURCES+old.SOURCES
    source_hashes = {n: sha256_file(ROOT/n) for n in sorted(set(sources))}
    if (output/"protocol.json").exists():
        p = data.read(output/"protocol.json")
        assert p["seed"] == seed and p["source_sha256"] == source_hashes and p["input_sha256"] == hashes
        assert p["recognition_input"] == relative_path(recognition) and p["diagnostic_input"] == relative_path(diagnostic)
        old.preserved(p)
        return p
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Unrecognized nonempty output; choose a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    p = {"experiment": "task_signal_ablation_v2", "frozen_at_utc": now(),
         "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
         "previous_tracked_blob_identities": old.original_inventory(), "source_sha256": source_hashes,
         "recognition_input": relative_path(recognition), "diagnostic_input": relative_path(diagnostic),
         "input_sha256": hashes, "seed": seed, "classes": manifest["classes"],
         "class_to_index": {c: i for i, c in enumerate(manifest["classes"])},
         "cohorts": {"development": {"videos": 100, "examples": 900}, "exploratory": {"videos": 40, "examples": 360}},
         "folds": "unchanged diagnostic_v1 bytes and seed20261017; never regenerate with new seed",
         "task": "unchanged nine single replacements/video from [0,4,8,12]; four refreshes,16 latest-cache outputs,identical pre-t history",
         "target": "saved refresh_gain=CE(reference,label)-CE(early,label),signed nats; never combine independent moves",
         "input_sets": {s: {"columns": u.SETS[s], "privileged": s == "P", "dimension": len(u.SETS[s])} for s in ["S", "C", "P"]},
         "S": "exact old group D;16 age/budget,motion,cached/past confidence summaries",
         "C": "S plus all20 cached-feature logits and all20 mean-past-output logits,each vector subtracting its own class mean;past candidates0..t-1 only",
         "P": "C plus centered20 current f_t logits;privileged,not a cheap gate;keeping reference can add paid RGB computation;no matched four-call deployment claim",
         "linear": "float64,training-only population standardization,std<=1e-12 scale1;sum residual squares+coefficient norm squared,alpha1,unpenalized intercept",
         "nonlinear": "same scaling;K=exp(-squared_distance/p),p includes constant columns;H K H centering;solve(Kc+I)a=y-ybar;query center uses training column/grand means and query row mean over training exemplars",
         "hyperparameter_search": False, "configurations": [u.access(n) for n in u.CONFIGS],
         "baselines": u.BASELINES, "reference_baselines": [u.access(n) for n in u.REFERENCES],
         "decision": "earlier iff predicted gain>1e-12;otherwise reference;realized gain=actual_gain*selection;not a sequential policy",
         "outcome_reference": "max(actual_gain,0) independently per opportunity;privileged actual-outcome knowledge,not deployable",
         "sequence": "save all development OOF predictions;fit/freeze all final probes on all development;then open exploratory arrays",
         "primary": "C-nonlinear vs keep_reference:development OOF equal-video mean realized gain",
         "secondary": [{"candidate": c, "baseline": b, "metrics": ["MAE", "realized_gain"]} for c, b in u.CONTRASTS],
         "prediction_metrics": "equal-video MAE,MSE;descriptive Spearman;positive-gain AUROC with event counts",
         "undefined": "null+reason for one-class AUROC or constant scores;constant-score AUROC intentionally undefined rather than imputed",
         "ties": "1e-12 for floating-point ties/counts;all signed targets retained;exact max(gain,0) for hindsight reference",
         "bootstrap": {"resamples": 10000, "seed": seed, "rng": "PCG64", "percentile_method": "linear",
                       "unit": "whole source video,all nine examples;shared index matrix across configurations within cohort"},
         "limits": ["both cohorts developmental", "recognition regularization used development", "exploratory previously inspected",
                    "conditional fitted-probe intervals omit full training/model-selection/repeated-research uncertainty",
                    "secondary intervals exploratory and unadjusted", "P not information-theoretic upper bound",
                    "one RBF model does not test every nonlinear model", "full-clip candidates and future-reference codec vectors:offline",
                    "no sequential controller,adaptive latency,speedup or matched P deployment-cost claim",
                    "no completed fresh-confirmation cohort parsed"],
         "actual_RGB_encoder_executions": 0, "video_decodes": 0, "downloads": 0, "recognition_training": 0}
    strict_json(output/"protocol.json", p)
    shutil.copyfile(diagnostic/"fold_manifest.json", output/"fold_manifest.json")
    shutil.copyfile(diagnostic/"counterfactual_examples.csv", output/"counterfactual_targets.csv")
    strict_json(output/"input_provenance.json", {"recorded_at_utc": now(), "hashes": hashes, "feature_definitions": p["input_sets"],
        "classifier_sha256": sha256_file(recognition/"classifiers/20/parameters.json"),
        "folds_copied_byte_for_byte": True, "targets_copied_byte_for_byte": True,
        "fresh_confirmation_artifacts_parsed": 0})
    strict_json(output/"environment.json", {"recorded_at_utc": now(), "python": platform.python_version(),
        "os": platform.platform(), "packages": {n: version(n) for n in ["numpy", "pandas", "torch", "torchvision", "av", "matplotlib"]},
        "device": "CPU float64 saved-array analysis", "new_dependencies": [], "encoder_calls": 0, "video_decodes": 0, "downloads": 0})
    return p


def input_table(videos: list, targets: pd.DataFrame, head: dict) -> pd.DataFrame:
    by_id = {v.entry["video_id"]: v for v in videos}
    rows = []
    for row in targets.itertuples():
        v = by_id[row.video_id]
        assert v.entry["source_video_id"] == row.source_video_id and v.entry["label"] == row.label
        observation = u.cached_observation(v, row.t, head)
        cached = u.cached_inputs(observation)
        np.testing.assert_array_equal([cached[c] for c in u.SETS["S"]], [getattr(row, c) for c in u.SETS["S"]])
        privileged = u.privileged_inputs(cached, feature_logits(v.references[[row.t]], head)[0])
        rows.append({key: getattr(row, key) for key in ["example_id", "video_id", "source_video_id", "t", "last_refresh"]}
                    | {"contains_privileged_P_current_inputs": True, **privileged})
    return pd.DataFrame(rows)


def join_targets(inputs: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    columns = ["example_id", "video_id", "source_video_id", "t", "last_refresh"]
    return inputs.merge(targets[columns+["refresh_gain"]], on=columns, validate="one_to_one", sort=False)


def fit_oof(table: pd.DataFrame, folds: dict, output: Path) -> dict:
    parts, registry = [], {}
    for f in folds["folds"]:
        train = table[table.source_video_id.isin(f["training_sources"])]
        heldout = table[table.source_video_id.isin(f["validation_sources"])]
        assert set(train.source_video_id).isdisjoint(heldout.source_video_id)
        folder = output/"models"/f"fold_{f['fold']}"
        state = u.fit_bundle(train, folder)
        registry[str(f["fold"])] = {"path": folder.relative_to(output).as_posix(),
                                   "registry_sha256": sha256_file(folder/"registry.json")}
        parts.append(u.predictions(heldout, folder, state, "development_oof", f["fold"]))
    csv(output/"out_of_fold_predictions.csv", pd.concat(parts, ignore_index=True))
    return {"folds": registry, "oof_saved_at_utc": now(), "oof_sha256": sha256_file(output/"out_of_fold_predictions.csv")}


def run(args) -> None:
    recognition, diagnostic, output = [(ROOT/p).resolve() for p in [args.recognition_input, args.diagnostic_input, args.output]]
    if args.audit_only and not (output/"verification.json").is_file():
        raise FileNotFoundError("Audit requires a completed evaluated run")
    p = freeze(recognition, diagnostic, output, args.seed)
    if args.audit_only or ((output/"status.json").exists() and data.read(output/"status.json")["status"] == "complete"):
        from verify_task_signal_ablation import audit
        print(json.dumps(audit(recognition, diagnostic, output, p), indent=2))
        return
    strict_json(output/"status.json", {"status": "running", "started_at_utc": now()})
    manifest, head, _ = identities(recognition, diagnostic)
    targets = pd.read_csv(output/"counterfactual_targets.csv", float_precision="round_trip")
    assert targets.groupby("cohort").size().to_dict() == {"development": 900, "exploratory": 360}
    assert targets.groupby("source_video_id").size().eq(9).all() and targets.example_id.nunique() == 1260
    with old.frozen_only():
        dev_targets = targets[targets.cohort == "development"]
        dev_inputs = input_table(data.load_role(manifest["videos"], "development"), dev_targets, head)
        dev = join_targets(dev_inputs, dev_targets)
        models = fit_oof(dev, data.read(output/"fold_manifest.json"), output)
        print("All development OOF predictions saved; fitting final fixed probes.", flush=True)
        final = output/"models"/"final_development"
        registry = u.fit_bundle(dev, final)
        models.update(final_registry_sha256=sha256_file(final/"registry.json"), final_models_frozen_at_utc=now())
        strict_json(output/"frozen_models.json", models)
        print("Final probes frozen; opening exploratory references once.", flush=True)
        ex_targets = targets[targets.cohort == "exploratory"]
        ex_inputs = input_table(data.load_role(manifest["videos"], "exploratory"), ex_targets, head)
        exploratory = join_targets(ex_inputs, ex_targets)
        csv(output/"exploratory_predictions.csv", u.predictions(exploratory, final, registry, "exploratory", -1))
    csv(output/"inputs_including_privileged_P.csv", pd.concat([dev_inputs.assign(cohort="development"),
        ex_inputs.assign(cohort="exploratory")], ignore_index=True))
    predictions = pd.concat([pd.read_csv(output/n, float_precision="round_trip") for n in
                             ["out_of_fold_predictions.csv", "exploratory_predictions.csv"]], ignore_index=True)
    assert len(predictions) == 12600
    aggregate, per_video, intervals, indices = u.summarize(predictions, args.seed)
    strict_json(output/"aggregate_metrics.json", aggregate)
    csv(output/"per_video_metrics.csv", per_video)
    strict_json(output/"bootstrap_intervals.json", intervals)
    np.savez_compressed(output/"bootstrap_resampling_indices.npz", **indices)
    strict_json(output/"metadata.json", {"evaluated_at_utc": now(), "development_examples": 900, "exploratory_examples": 360,
        "OOF_predictions": 9000, "exploratory_predictions": 3600, "actual_RGB_encoder_executions": 0,
        "video_decodes": 0, "downloads": 0, "recognition_training": 0, "models_frozen_before_exploratory": models,
        "P": "privileged current appearance;no matched four-call deployment-cost claim", "fresh_confirmation_parsed": 0})
    from verify_task_signal_ablation import audit
    verification = audit(recognition, diagnostic, output, p)
    test = subprocess.run([sys.executable, "-m", "unittest", "test_task_signal_ablation", "-v"], cwd=ROOT, capture_output=True, text=True)
    assert test.returncode == 0, test.stdout+test.stderr
    release = subprocess.run([sys.executable, "validate_release.py"], cwd=ROOT, capture_output=True, text=True)
    assert release.returncode == 0, release.stdout+release.stderr
    verification.update(focused_tests={"returncode": 0, "output": test.stdout+test.stderr},
                        release_validator={"returncode": 0, "output": release.stdout+release.stderr})
    strict_json(output/"verification.json", verification)
    from task_signal_ablation_report import report
    report(output)
    strict_json(output/"status.json", {"status": "complete", "completed_at_utc": now(), "blocker": None})
    write_ledger(output)
    print(f"Complete: {relative_path(output)}", flush=True)


def write_ledger(output: Path) -> None:
    files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (output/"SHA256SUMS.txt").write_text("".join(f"{sha256_file(p)}  {p.relative_to(output).as_posix()}\n" for p in files), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recognition-input", type=Path, default=Path("results/recognition_baseline_v2"))
    parser.add_argument("--diagnostic-input", type=Path, default=Path("results/task_signal_diagnostic_v1"))
    parser.add_argument("--output", type=Path, default=Path("results/task_signal_ablation_v2"))
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        message = f"{type(exc).__name__}: {str(exc).replace(str(ROOT), '.')}"
        path = ROOT/args.output/"status.json"
        if (path.parent/"protocol.json").exists() and not args.audit_only:
            existing = data.read(path) if path.exists() else {}
            if existing.get("status") != "complete":
                strict_json(path, {"status": "incomplete", "blocked_at_utc": now(), "blocker": message})
        print(f"INCOMPLETE: {message}", file=sys.stderr)
        raise
