"""Two-target refresh diagnostic: saved cached inputs, fixed grouped folds."""
from __future__ import annotations

import os
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[key] = "1"

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from importlib.metadata import distributions, version

import numpy as np
import pandas as pd

import analyze_task_signals as historical
import recognition_baseline_data as data
import two_target_refresh_utils as u
from fresh_confirmation_data import strict_json
from research_utils import ROOT, relative_path, sha256_file, now

SOURCES = ["analyze_two_target_refresh_predictability.py", "two_target_refresh_utils.py",
           "verify_two_target_refresh_predictability.py", "two_target_refresh_report.py",
           "test_two_target_refresh_predictability.py", "requirements-two-target.txt",
           "task_signal_ablation_utils.py", "task_signal_utils.py", "refresh_target_decomposition_utils.py",
           "analyze_task_signals.py", "fresh_confirmation_data.py", "research_utils.py"]


def csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, float_format="%.17g", na_rep="")


def identities(decomposition: Path, ablation: Path) -> tuple[dict, dict]:
    hashes = {}
    for directory in [decomposition, ablation]:
        for name in ["status.json", "verification.json", "protocol.json", "SHA256SUMS.txt"]:
            if not (directory/name).is_file():
                raise FileNotFoundError(f"Required prerequisite missing: {relative_path(directory/name)}")
        assert data.read(directory/"status.json")["status"] == "complete"
        assert data.read(directory/"verification.json")["passed"]
        for line in (directory/"SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            p = directory/name
            if not p.is_file():
                raise FileNotFoundError(f"Required prerequisite missing: {relative_path(p)}")
            assert sha256_file(p) == expected, relative_path(p)
            hashes[relative_path(p)] = expected
        hashes[relative_path(directory/"SHA256SUMS.txt")] = sha256_file(directory/"SHA256SUMS.txt")
        prior = data.read(directory/"protocol.json")
        assert {n: sha256_file(ROOT/n) for n in prior["source_sha256"]} == prior["source_sha256"]
    p = data.read(decomposition/"protocol.json")
    assert p["ablation_input"] == relative_path(ablation)
    ab = data.read(ablation/"protocol.json")
    assert p["classes"] == ab["classes"] and ab["input_sets"]["C"]["columns"] == u.COLUMNS == u.previous.SETS["C"]
    classifier = ROOT/p["classifier"]["path"]
    assert sha256_file(classifier) == p["classifier"]["sha256"]
    hashes[relative_path(classifier)] = sha256_file(classifier)
    return p, hashes


def load_table(decomposition: Path, ablation: Path, role: str) -> pd.DataFrame:
    """usecols excludes every privileged P column before pandas reads values."""
    inputs = pd.read_csv(ablation/"inputs_including_privileged_P.csv", usecols=u.KEYS+u.COLUMNS+["cohort"], float_precision="round_trip")
    inputs = inputs[inputs.cohort == role].copy()
    fields = ["cohort", "example_id", "video_id", "source_video_id", "class", "t", "r", "last_refresh",
              "original_saved_target", "loss_A", "loss_B", "loss_C", "loss_D", *u.TARGET_COLUMNS.values()]
    target = pd.read_csv(decomposition/"per_opportunity_decomposition.csv", usecols=fields, float_precision="round_trip")
    target = target[target.cohort == role]
    target = target.rename(columns={v: k for k, v in u.TARGET_COLUMNS.items()})
    table = inputs.merge(target, on=["cohort", *u.KEYS], validate="one_to_one", sort=False)
    expected = 900 if role == "development" else 360
    assert len(inputs) == len(target) == len(table) == expected
    assert table.example_id.nunique() == expected and table.groupby("source_video_id").size().eq(9).all()
    assert table.source_video_id.nunique() == expected//9
    np.testing.assert_array_equal(table.G, table.original_saved_target)
    for name, values in {"B": table.loss_A-table.loss_B, "C": table.loss_C-table.loss_B,
                         "G": table.loss_A-table.loss_C,
                         "B_averaged": ((table.loss_A-table.loss_B)+(table.loss_D-table.loss_C))/2,
                         "C_averaged": ((table.loss_C-table.loss_B)+(table.loss_D-table.loss_A))/2}.items():
        np.testing.assert_allclose(table[name], values, atol=u.TOL, rtol=0)
    np.testing.assert_allclose(table.B-table.C, table.G, atol=u.TOL, rtol=0)
    np.testing.assert_allclose(table.B_averaged-table.C_averaged, table.G, atol=u.TOL, rtol=0)
    assert np.isfinite(table[u.TARGETS].to_numpy()).all()
    u.design(table)
    return table


def freeze(args, output: Path) -> dict:
    decomposition, ablation = (ROOT/args.decomposition_input).resolve(), (ROOT/args.ablation_input).resolve()
    prior, hashes = identities(decomposition, ablation)
    sources = {n: sha256_file(ROOT/n) for n in SOURCES}
    if (output/"protocol.json").is_file():
        p = data.read(output/"protocol.json")
        assert p["input_sha256"] == hashes and p["source_sha256"] == sources
        assert p["seed"] == args.seed and p["bootstrap"]["resamples"] == args.bootstrap
        assert p["decomposition_input"] == relative_path(decomposition) and p["ablation_input"] == relative_path(ablation)
        historical.preserved(p)
        return p
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty output with a different protocol; choose a versioned fresh --output")
    output.mkdir(parents=True, exist_ok=True)
    folds = data.read(ablation/"fold_manifest.json")
    p = {"experiment": "two_target_refresh_predictability_v1", "frozen_at_utc": now(),
        "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "previous_tracked_blob_identities": historical.original_inventory(), "source_sha256": sources,
        "input_sha256": hashes, "decomposition_input": relative_path(decomposition), "ablation_input": relative_path(ablation),
        "source_membership": prior["source_membership"], "classes": prior["classes"], "classifier": prior["classifier"],
        "cohorts": {"development": {"sources": 100, "opportunities": 900}, "exploratory": {"sources": 40, "opportunities": 360}},
        "folds": folds, "fold_manifest_sha256": sha256_file(ablation/"fold_manifest.json"),
        "input_columns": u.COLUMNS, "input_dimension": 56, "age_baseline_columns": u.AGE_COLUMNS,
        "input_access": "Read only whitelisted saved C columns; no current/future RGB, labels, identities, losses or targets in X. Class-centered cached and mean past-output logits; old candidate and codec limitations remain.",
        "targets": u.TARGET_COLUMNS, "definition": "B=LA-LB,C=LC-LB,G=LA-LC=B-C;signed nats;no target scaling or clipping",
        "sensitivity": "Only the predeclared averaged-attribution B/C pair; G remains unchanged; forward primary",
        "ridge": "Reuse task_signal_ablation_utils.fit_array: float64, population training-only standardization, std<=1e-12 scale1; alpha1, unpenalized intercept, sum squared residual objective",
        "kernel_ridge": "Same preprocessing and alpha1; exp(-squared_distance/56); training-only centered kernel and targets, unpenalized intercept",
        "equivalence": {"identity": "heldout direct G approximately predicted B-predicted C; both forward/averaged,linear/kernel",
                        "absolute_tolerance": u.ATOL, "relative_tolerance": u.RTOL,
                        "rationale": "same float64 solve tolerances as prior fit residual checks; report maximum discrepancies and any threshold-sensitive decisions"},
        "forests": {"parameters": u.RF_PARAMETERS, "fits": u.FORESTS,
                    "preprocessing": "identical C representation; population scaler fit on training rows only; sklearn trees internally convert inputs to float32",
                    "partitioning": "separate target-dependent forests for each B/C; same fixed seed but no shared fixed tree partitions",
                    "accounting": "save actual node counts and compressed joblib protocol5 sizes; tree count is neither a latency nor equal-capacity claim"},
        "baselines": ["keep reference", "training-only mean and median of every target", "existing age/budget ridge G", "existing C-linear and C-nonlinear direct G reproduced"],
        "decision_methods": u.METHODS, "decision_threshold": u.TOL, "decision": "select independent earlier move iff predicted_G>1e-12;utility=actual_G*selected;never combine moves",
        "sign_events": "positive target>1e-12,negative target<-1e-12,neutral otherwise;all original signed magnitudes retained in regression and utility",
        "primary": "development OOF equal-video utility(Two_target_RF200x2_forward)-utility(Direct_G_RF400)",
        "required_absolute": "utility(Two_target_RF200x2_forward)-0;positive relative and absolute intervals required for encouraging evidence",
        "secondary": ["vs Direct_G_RF200", "vs C-linear and C-nonlinear", "averaged versus forward attribution; no selection of favorable order"],
        "metrics": ["per-target equal-video MAE,MSE,MAE improvement over own fold-training-median baseline",
                    "descriptive pooled Spearman and positive-target AUROC;single-class/constant scores null with reason",
                    "utility,selected/total,helpful/harmful/neutral selections,selected mean and denominator",
                    "MSE_G=MSE_B+MSE_C-2*mean(eB*eC);uncentered cross-product,not covariance"],
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "rng": "PCG64", "percentile_method": "linear",
                      "unit": "whole source video;all9 examples;shared matrix across all methods/targets within cohort"},
        "sequence": "save all development OOF predictions;fit/freeze final probes;then compute exploratory predictions once",
        "seed": args.seed, "hyperparameter_search": False,
        "limits": ["both cohorts developmental and previously inspected;recognition regularization used development",
                   "conditional fitted-model intervals omit full training/model-selection/repeated-research uncertainty;secondary unadjusted",
                   "retrospective true-label and whole-video targets are not observable online signals",
                   "full-clip candidates and future-reference codec vectors:offline,not verified streaming causal",
                   "no sequential controller,recognition accuracy improvement,adaptive latency,speedup or publication novelty claim",
                   "separate fresh-confirmation cohort not accessed"],
        "actual_RGB_encoder_executions": 0, "video_decodes": 0, "video_downloads": 0, "recognition_training_calls": 0}
    strict_json(output/"protocol.json", p)
    shutil.copyfile(ablation/"fold_manifest.json", output/"fold_manifest.json")
    strict_json(output/"input_provenance.json", {"recorded_at_utc": now(), "input_sha256": hashes,
        "feature_columns": u.COLUMNS, "feature_source": relative_path(ablation/"inputs_including_privileged_P.csv"),
        "read_policy": "pandas usecols reads only cached C columns and auditing keys;privileged P values never loaded",
        "target_source": relative_path(decomposition/"per_opportunity_decomposition.csv"),
        "folds_copied_byte_for_byte": True, "target_definitions": u.TARGET_COLUMNS,
        "classifier": prior["classifier"], "fresh_confirmation_artifacts_accessed": 0})
    packages = {d.metadata["Name"].lower(): d.version for d in distributions()}
    before_path = ROOT/".git/two_target_environment_before.json"
    before = data.read(before_path) if before_path.is_file() else None
    if before is not None:
        assert all(packages.get(k) == v for k, v in before.items()), "Existing environment packages changed"
    strict_json(output/"environment.json", {"recorded_at_utc": now(), "python": platform.python_version(),
        "os": platform.platform(), "packages": packages, "existing_packages_unchanged": True if before else None,
        "added_packages": {k: v for k, v in packages.items() if before is not None and k not in before},
        "device": "CPU saved-table analysis;no CUDA operation", "reference_encoder_calls": 0,
        "video_downloads": 0, "package_installation": "new optional pinned dependencies;pip check passed;no research-stack upgrades",
        "random_forest_docs": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html"})
    return p


def write_ledger(output: Path) -> None:
    files = sorted(f for f in output.rglob("*") if f.is_file() and f.name != "SHA256SUMS.txt")
    (output/"SHA256SUMS.txt").write_text("".join(f"{sha256_file(f)}  {f.relative_to(output).as_posix()}\n" for f in files), encoding="utf-8")


def run(args) -> None:
    output = (ROOT/args.output).resolve()
    if args.audit_only and not (output/"verification.json").is_file():
        raise FileNotFoundError("Audit requires an evaluated run and verification.json")
    p = freeze(args, output)
    decomposition, ablation = ROOT/p["decomposition_input"], ROOT/p["ablation_input"]
    if args.audit_only or ((output/"status.json").exists() and data.read(output/"status.json")["status"] == "complete"):
        from verify_two_target_refresh_predictability import audit
        print(json.dumps(audit(output, p), indent=2))
        return
    strict_json(output/"status.json", {"status": "running", "started_at_utc": now()})
    with historical.frozen_only():
        dev = load_table(decomposition, ablation, "development")
        csv(output/"development_analysis_table.csv", dev)
        all_parts, models = [], {}
        for f in p["folds"]["folds"]:
            tr, va = dev[dev.source_video_id.isin(f["training_sources"])], dev[dev.source_video_id.isin(f["validation_sources"])]
            assert len(tr) == 720 and len(va) == 180 and set(tr.source_video_id).isdisjoint(va.source_video_id)
            folder = output/"models"/f"fold_{f['fold']}"
            registry = u.fit_bundle(tr, folder)
            all_parts.append(u.predict_bundle(va, folder, registry, "development_oof", f["fold"]))
            models[str(f["fold"])] = {"path": folder.relative_to(output).as_posix(), "registry_sha256": sha256_file(folder/"registry.json")}
            print(f"Fold {f['fold']}:80 training/20 heldout sources;fixed probes saved", flush=True)
        oof = pd.concat(all_parts, ignore_index=True)
        decisions, errors = u.decision_rows(oof)
        oof = u.augment_net_quality(oof, decisions)
        csv(output/"component_predictions_oof.csv", oof)
        csv(output/"out_of_fold_predictions.csv", decisions)
        strict_json(output/"oof_frozen.json", {"saved_at_utc": now(),
            "components_sha256": sha256_file(output/"component_predictions_oof.csv"),
            "decisions_sha256": sha256_file(output/"out_of_fold_predictions.csv")})
        print("All development OOF predictions saved;fitting final probes before exploratory prediction.", flush=True)
        final = output/"models/final_development"
        registry = u.fit_bundle(dev, final)
        strict_json(output/"frozen_models.json", {"frozen_at_utc": now(), "fold_models": models,
            "final_path": final.relative_to(output).as_posix(), "final_registry_sha256": sha256_file(final/"registry.json"),
            "all_model_file_sha256": {f.relative_to(output).as_posix(): sha256_file(f) for f in sorted((output/"models").rglob("*")) if f.is_file()}})
        ex = load_table(decomposition, ablation, "exploratory")
        csv(output/"exploratory_analysis_table.csv", ex)
        xp = u.predict_bundle(ex, final, registry, "exploratory", -1)
        xd, xe = u.decision_rows(xp)
        xp = u.augment_net_quality(xp, xd)
        csv(output/"component_predictions_exploratory.csv", xp)
        csv(output/"exploratory_predictions.csv", xd)
        strict_json(output/"exploratory_evaluation.json", {"evaluated_at_utc": now(), "final_models_unchanged": True,
            "prediction_sha256": sha256_file(output/"exploratory_predictions.csv")})
        all_decisions, all_components, all_errors = pd.concat([decisions, xd]), pd.concat([oof, xp]), pd.concat([errors, xe])
        aggregate, video, bootstrap, extra = u.summarize(all_decisions, all_components, all_errors, args.seed, args.bootstrap)
    csv(output/"per_video_metrics.csv", video)
    csv(output/"per_video_component_metrics.csv", extra["quality_per_video"])
    csv(output/"per_video_error_decomposition.csv", extra["error_per_video"])
    strict_json(output/"aggregate_metrics.json", aggregate)
    strict_json(output/"bootstrap_intervals.json", bootstrap)
    np.savez_compressed(output/"bootstrap_resampling_indices.npz", **extra["indices"])
    from verify_two_target_refresh_predictability import audit
    checks = audit(output, p)
    unit = subprocess.run([sys.executable, "-m", "unittest", "test_two_target_refresh_predictability", "-v"], cwd=ROOT, capture_output=True, text=True)
    assert unit.returncode == 0, unit.stdout+unit.stderr
    release = subprocess.run([sys.executable, "validate_release.py"], cwd=ROOT, capture_output=True, text=True)
    assert release.returncode == 0, release.stdout+release.stderr
    checks.update(focused_tests={"returncode": 0, "output": unit.stdout+unit.stderr}, release_validator={"returncode": 0, "output": release.stdout+release.stderr})
    strict_json(output/"equivalence_checks.json", checks["equivalence_checks"])
    csv(output/"forest_model_accounting.csv", pd.DataFrame(checks["forest_model_accounting"]))
    strict_json(output/"verification.json", checks)
    from two_target_refresh_report import report
    report(output)
    strict_json(output/"status.json", {"status": "complete", "completed_at_utc": now(), "blocker": None})
    write_ledger(output)
    print(f"Complete:{relative_path(output)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition-input", type=Path, default=Path("results/refresh_target_decomposition_v1"))
    parser.add_argument("--ablation-input", type=Path, default=Path("results/task_signal_ablation_v2"))
    parser.add_argument("--output", type=Path, default=Path("results/two_target_refresh_predictability_v1"))
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    try:
        run(args)
    except Exception as exc:
        path = ROOT/args.output/"status.json"
        message = f"{type(exc).__name__}:{str(exc).replace(str(ROOT), '.')}"
        if (path.parent/"protocol.json").exists() and not args.audit_only:
            existing = data.read(path) if path.exists() else {}
            if existing.get("status") != "complete":
                strict_json(path, {"status": "incomplete", "blocked_at_utc": now(), "blocker": message})
        print(f"INCOMPLETE:{message}", file=sys.stderr)
        raise
