"""Experiment13: fixed cached-history representation, RF400 and saved targets."""
from __future__ import annotations

import os
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from importlib.metadata import distributions

import numpy as np
import pandas as pd

import analyze_task_signals as historical
import cached_history_signal_utils as u
from fresh_confirmation_data import strict_json
from research_utils import ROOT, relative_path, sha256_file, now

SOURCES = ["analyze_cached_history_signals.py", "cached_history_signal_utils.py", "cached_history_cpu.py",
           "verify_cached_history_signals.py", "test_cached_history_signals.py", "cached_history_signal_report.py",
           "two_target_refresh_utils.py", "task_signal_ablation_utils.py", "task_signal_utils.py",
           "analyze_task_signals.py", "fresh_confirmation_data.py", "research_utils.py"]


def csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, float_format="%.17g", na_rep="")


def identities(prior: Path) -> tuple[dict, dict]:
    for name in ["status.json", "verification.json", "protocol.json", "SHA256SUMS.txt"]:
        if not (prior/name).is_file():
            raise FileNotFoundError(f"Required prerequisite missing:{relative_path(prior/name)}")
    p = json.loads((prior/"protocol.json").read_text(encoding="utf-8"))
    assert json.loads((prior/"status.json").read_text())["status"] == "complete"
    assert json.loads((prior/"verification.json").read_text())["passed"]
    assert p["input_columns"] == u.C_COLUMNS and len(u.C_COLUMNS) == 56
    assert {n: sha256_file(ROOT/n) for n in p["source_sha256"]} == p["source_sha256"]
    hashes = {}
    for line in (prior/"SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if not (prior/name).is_file():
            raise FileNotFoundError(f"Required prerequisite missing:{relative_path(prior/name)}")
        assert sha256_file(prior/name) == digest, name
        hashes[relative_path(prior/name)] = digest
    hashes[relative_path(prior/"SHA256SUMS.txt")] = sha256_file(prior/"SHA256SUMS.txt")
    assert sha256_file(ROOT/p["classifier"]["path"]) == p["classifier"]["sha256"]
    hashes[p["classifier"]["path"]] = p["classifier"]["sha256"]
    return p, hashes


def read_observations(prior: Path, role: str) -> pd.DataFrame:
    columns = ["cohort", *u.KEYS, *u.C_COLUMNS]
    table = pd.read_csv(prior/f"{role}_analysis_table.csv", usecols=columns, float_precision="round_trip")
    assert set(table.cohort) == {role}
    return table.loc[:, columns]


def read_targets(prior: Path, role: str) -> pd.DataFrame:
    return pd.read_csv(prior/f"{role}_analysis_table.csv", usecols=["cohort", *u.KEYS, "G"], float_precision="round_trip")


def join_targets(observations: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    assert "G" not in observations and "G" in targets
    table = observations.merge(targets, on=["cohort", *u.KEYS], validate="one_to_one", sort=False)
    assert len(table) == len(observations) == len(targets)
    assert table.example_id.is_unique and table.groupby("source_video_id").size().eq(9).all()
    assert np.isfinite(table.G).all()
    return table


def freeze(args, output: Path) -> dict:
    prior = (ROOT/args.predictability_input).resolve()
    previous, hashes = identities(prior)
    source_hashes = {n: sha256_file(ROOT/n) for n in SOURCES}
    if (output/"protocol.json").is_file():
        p = json.loads((output/"protocol.json").read_text())
        assert p["input_sha256"] == hashes and p["source_sha256"] == source_hashes
        assert p["seed"] == args.seed and p["bootstrap"]["resamples"] == args.bootstrap
        assert p["predictability_input"] == relative_path(prior)
        historical.preserved(p)
        return p
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty output must be preserved;select a clearly versioned --output")
    output.mkdir(parents=True, exist_ok=True)
    members = previous["source_membership"]
    dev_sources = sorted(e["source_video_id"] for e in members if e["recognition_role"] == "development")
    timing_sources = dev_sources[:10]
    parameters = json.loads((prior/"models/final_development/Direct_G_RF400_G.json").read_text())["parameters"]
    assert parameters["random_state"] == 20261020 and parameters["n_estimators"] == 400
    p = {"experiment": "cached_history_signal_diagnostic_v1", "experiment_number": 13, "frozen_at_utc": now(),
        "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "previous_tracked_blob_identities": historical.original_inventory(), "source_sha256": source_hashes,
        "input_sha256": hashes, "predictability_input": relative_path(prior), "classifier": previous["classifier"],
        "source_membership": members, "classes": previous["classes"], "folds": previous["folds"],
        "fold_manifest_sha256": sha256_file(prior/"fold_manifest.json"),
        "cohorts": {"development": {"sources": 100, "opportunities": 900}, "exploratory": {"sources": 40, "opportunities": 360}},
        "target": "exact saved signed G=net_gain=loss_A-loss_C;no clipping,redefinition or decomposition",
        "C_columns": u.C_COLUMNS, "T_columns": u.T_COLUMNS, "augmented_columns": u.AUGMENTED_COLUMNS,
        "dimensions": {"C": 56, "T": 44, "C_plus_T": 100},
        "history": "reference periodic[0,4,8,12];t1,2,3 history[0];t5,6,7 history[0,4];t9,10,11 history[0,4,8];append distinct events once;reset every video",
        "T_formulas": {"latest20": "qk-q(k-1),else zero", "previous20": "q(k-1)-q(k-2),else zero",
                       "mask1": "k>=2", "mask2": "k>=3", "margin_std": "population std of per-vector softmax top-two probability margins;zero k1",
                       "switch_rate": "consecutive distinct-event top1 switches/(k-1);zero k1"},
        "softmax": "float64 exp(q-max(q))/sum;argmax ties choose first/lowest fixed class index",
        "input_isolation": "observation-only builder accepts available cached vectors/statistics;no whole video,targets,labels,future rows,current RGB or P columns;targets joined only after construction",
        "information_audit": "k1 no extra information;k2 q0 recoverable from current q4,past mean and t;k3 old C preserves q0+q4 but not their order;swap fixture frozen in tests;44 columns not independent new observations",
        "numerical_tolerances": {"sign": u.TOL, "duplicate_cached_vector": "exact equality", "scaler_prefix_absolute": u.TOL,
                                 "prediction_and_algebra_absolute": u.ATOL, "prediction_and_algebra_relative": u.RTOL},
        "RF_parameters": parameters, "RF_seed": 20261020, "baseline": "copy saved Direct_G_RF400 JSON/joblib bytes;do not refit",
        "scaling": "unchanged population training-only scaling,constant std<=1e-12 replaced with1;augmented first56 match baseline",
        "methods": u.METHODS, "threshold": u.TOL,
        "primary": "development OOF equal-video utility(C_plus_T_RF400)-utility(C_RF400)",
        "required_absolute": "augmented utility versus keep_reference zero;positive relative and absolute utility required for encouraging result",
        "utility": "G*indicator(prediction>1e-12);mean all9 per source,including unselected zeros;then equal-source mean",
        "secondary": "exploratory separately;MAE/MSE/median-baseline improvement,descriptive pooled Spearman/AUROC;history lengths1,2,3 withsame full-data models,no subgroup selection",
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "rng": "PCG64", "percentile_method": "linear",
                      "unit": "whole source,all9 opportunities;shared matrix for every method/history subgroup per cohort"},
        "timing": {"sample_selection": "first10 development source IDs in stable lexicographic order;before new outcomes;no outcome-dependent or stochastic sample selection",
                   "source_ids": timing_sources, "seed": args.seed, "warmup_passes": 2, "measured_passes": 5,
                   "modes": ["history_T", "baseline_scale_RF1", "augmented_history_scale_RF1"],
                   "order": "PCG64 seeded permutation per pass;one pass processes all10 sources and90 opportunities sequentially",
                   "boundary": "preloaded C observations through incremental T or batch-size1 scaled RF output;include per-video reset and all30 distinct history updates",
                   "exclusions": "loading,fitting,logging,assertions,decoding,RGB extraction,codec-vector extraction,full recognition",
                   "model_scope": "final development models;timing is component work,not heldout prediction quality or video latency"},
        "sequence": "first reproduce baseline-only OOF and exploratory predictions;save all new development OOF;fit/freeze final augmented model;then evaluate exploratory",
        "seed": args.seed, "hyperparameter_search": False,
        "limits": ["both cohorts previously inspected and developmental;recognition regularization used development",
                   "new history summaries are not a temporal video backbone",
                   "full-clip candidates and future-reference codec vectors:not verified streaming causal",
                   "independent counterfactual moves,not a sequential policy;no adaptive accuracy or speedup claim",
                   "fixed-model conditional intervals exclude training/model-selection/repeated-research uncertainty;secondary unadjusted",
                   "completed fresh-confirmation cohort never accessed"],
        "actual_RGB_encoder_executions": 0, "video_decodes": 0, "video_downloads": 0, "recognition_training_calls": 0}
    strict_json(output/"protocol.json", p)
    shutil.copyfile(prior/"fold_manifest.json", output/"fold_manifest.json")
    strict_json(output/"input_provenance.json", {"recorded_at_utc": now(), "input_sha256": hashes,
        "read_policy": "pandas usecols:only C observations and keys;G loaded separately and joined after T construction;no privileged columns/feature arrays",
        "folds_copied_byte_for_byte": True, "C_columns": u.C_COLUMNS, "T_columns": u.T_COLUMNS,
        "classifier": previous["classifier"], "fresh_confirmation_artifacts_accessed": 0})
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"], capture_output=True, text=True)
    strict_json(output/"environment.json", {"recorded_at_utc": now(), "os": platform.platform(), "python": platform.python_version(),
        "cpu": json.loads(cpu.stdout) if cpu.returncode == 0 and cpu.stdout.strip() else platform.processor(),
        "packages": {d.metadata["Name"].lower(): d.version for d in distributions()},
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "new_dependencies": [],
        "device": "CPU cached-table fitting,analysis and component microbenchmark"})
    return p


def baseline_reproduction(prior: Path) -> dict:
    checks = []
    for role, filename in [("development", "out_of_fold_predictions.csv"), ("exploratory", "exploratory_predictions.csv")]:
        observations, targets = read_observations(prior, role), read_targets(prior, role)
        original = pd.read_csv(prior/filename, usecols=["example_id", "method", "actual_G", "prediction_G"], float_precision="round_trip")
        original = original[original.method == "Direct_G_RF400"].set_index("example_id")
        folds = json.loads((prior/"fold_manifest.json").read_text())["folds"] if role == "development" else [{"fold": -1, "validation_sources": observations.source_video_id.unique().tolist()}]
        for f in folds:
            group = observations[observations.source_video_id.isin(f["validation_sources"])]
            stem = f"fold_{f['fold']}" if role == "development" else "final_development"
            folder = prior/"models"/stem
            state = json.loads((folder/"Direct_G_RF400_G.json").read_text())
            entry = {"file": "Direct_G_RF400_G.json", "sha256": sha256_file(folder/"Direct_G_RF400_G.json"), "type": "forest"}
            _, _, forest = u.previous.load_model(folder, entry)
            values = forest.predict((group[u.C_COLUMNS].to_numpy(dtype=np.float64)-state["means"])/state["scales"])
            expected = original.loc[group.example_id]
            np.testing.assert_array_equal(targets.set_index("example_id").loc[group.example_id, "G"], expected.actual_G)
            np.testing.assert_allclose(values, expected.prediction_G, atol=u.ATOL, rtol=u.RTOL)
            checks.append({"cohort": role, "fold": f["fold"], "opportunities": len(group),
                "max_absolute_prediction_discrepancy": float(abs(values-expected.prediction_G.to_numpy()).max()),
                "decision_disagreements": int(((values > u.TOL)!=(expected.prediction_G.to_numpy() > u.TOL)).sum())})
    return {"passed": True, "checked_at_utc": now(), "checks": checks,
            "max_absolute_discrepancy": max(r["max_absolute_prediction_discrepancy"] for r in checks)}


def write_ledger(output: Path) -> None:
    files = sorted(f for f in output.rglob("*") if f.is_file() and f.name != "SHA256SUMS.txt")
    (output/"SHA256SUMS.txt").write_text("".join(f"{sha256_file(f)}  {f.relative_to(output).as_posix()}\n" for f in files), encoding="utf-8")


def run(args) -> None:
    output = (ROOT/args.output).resolve()
    if args.audit_only and not (output/"verification.json").is_file():
        raise FileNotFoundError("Audit requires an evaluated run")
    # A changed protocol gets a fresh directory, never new expected hashes in an old run.
    if not args.audit_only and output.exists() and any(output.iterdir()):
        old = json.loads((output/"protocol.json").read_text()) if (output/"protocol.json").exists() else {}
        _, inputs = identities((ROOT/args.predictability_input).resolve())
        sources = {n: sha256_file(ROOT/n) for n in SOURCES}
        if old.get("source_sha256") != sources or old.get("input_sha256") != inputs or old.get("seed") != args.seed or old.get("bootstrap", {}).get("resamples") != args.bootstrap:
            output = output.with_name(output.name+"_"+now().replace("-", "").replace(":", "").replace(".", "_").replace("+", "_"))
            print(f"Preserved earlier outputs;using {relative_path(output)}", flush=True)
    args.actual_output = output
    p = freeze(args, output)
    prior = ROOT/p["predictability_input"]
    if args.audit_only or ((output/"status.json").exists() and json.loads((output/"status.json").read_text())["status"] == "complete"):
        from verify_cached_history_signals import audit
        print(json.dumps(audit(output, p), indent=2))
        return
    strict_json(output/"status.json", {"status": "running", "started_at_utc": now()})
    with historical.frozen_only():
        strict_json(output/"baseline_reproduction.json", baseline_reproduction(prior))
        print("Saved RF400 baseline reproduced before augmentation comparisons.", flush=True)
        dev_obs, dev_info = u.augment_observations(read_observations(prior, "development"))
        csv(output/"development_observations.csv", dev_obs)
        dev = join_targets(dev_obs, read_targets(prior, "development"))
        parts = []
        for f in p["folds"]["folds"]:
            train, held = dev[dev.source_video_id.isin(f["training_sources"])], dev[dev.source_video_id.isin(f["validation_sources"])]
            assert len(train) == 720 and len(held) == 180 and set(train.source_video_id).isdisjoint(held.source_video_id)
            folder = output/"models"/f"fold_{f['fold']}"
            registry = u.fit_bundle(train, folder, prior/"models"/folder.name)
            parts.append(u.predictions(held, folder, registry, "development_oof", f["fold"]))
            print(f"Fold {f['fold']}:baseline reused;fixed augmented RF400 saved", flush=True)
        oof = pd.concat(parts, ignore_index=True)
        csv(output/"out_of_fold_predictions.csv", oof)
        strict_json(output/"oof_frozen.json", {"saved_at_utc": now(), "sha256": sha256_file(output/"out_of_fold_predictions.csv")})
        final = output/"models/final_development"
        registry = u.fit_bundle(dev, final, prior/"models/final_development")
        strict_json(output/"frozen_models.json", {"frozen_at_utc": now(),
            "model_file_sha256": {f.relative_to(output).as_posix(): sha256_file(f) for f in sorted((output/"models").rglob("*")) if f.is_file()}})
        print("All development OOF predictions saved;final augmented model frozen before exploratory evaluation.", flush=True)
        ex_obs, ex_info = u.augment_observations(read_observations(prior, "exploratory"))
        csv(output/"exploratory_observations.csv", ex_obs)
        exploratory = join_targets(ex_obs, read_targets(prior, "exploratory"))
        xp = u.predictions(exploratory, final, registry, "exploratory", -1)
        csv(output/"exploratory_predictions.csv", xp)
        strict_json(output/"exploratory_evaluation.json", {"evaluated_at_utc": now(), "sha256": sha256_file(output/"exploratory_predictions.csv")})
        aggregate, video, history, intervals, matrices = u.summarize(pd.concat([oof, xp]), args.seed, args.bootstrap)
        csv(output/"per_video_metrics.csv", video)
        csv(output/"history_length_metrics.csv", history)
        strict_json(output/"aggregate_metrics.json", aggregate)
        strict_json(output/"bootstrap_intervals.json", intervals)
        np.savez_compressed(output/"bootstrap_resampling_indices.npz", **matrices)
        strict_json(output/"history_construction_checks.json", {"development": dev_info, "exploratory": ex_info, "fixture": u.information_fixture()})
        from cached_history_cpu import run as microbenchmark
        timing, timing_metadata = microbenchmark(dev_obs, p["timing"]["source_ids"], final, registry, args.seed)
        csv(output/"cpu_microbenchmark.csv", timing)
        strict_json(output/"cpu_microbenchmark.metadata.json", timing_metadata)
    from verify_cached_history_signals import audit
    verification = audit(output, p)
    tests = subprocess.run([sys.executable, "-m", "unittest", "test_cached_history_signals", "-v"], cwd=ROOT, capture_output=True, text=True)
    assert tests.returncode == 0, tests.stdout+tests.stderr
    release = subprocess.run([sys.executable, "validate_release.py"], cwd=ROOT, capture_output=True, text=True)
    assert release.returncode == 0, release.stdout+release.stderr
    verification.update(focused_tests={"returncode": 0, "output": tests.stdout+tests.stderr},
                        release_validator={"returncode": 0, "output": release.stdout+release.stderr})
    strict_json(output/"verification.json", verification)
    from cached_history_signal_report import report
    report(output)
    strict_json(output/"status.json", {"status": "complete", "completed_at_utc": now(), "blocker": None})
    write_ledger(output)
    print(f"Complete:{relative_path(output)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictability-input", type=Path, default=Path("results/two_target_refresh_predictability_v1"))
    parser.add_argument("--output", type=Path, default=Path("results/cached_history_signal_diagnostic_v1"))
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 1: parser.error("--bootstrap must be positive")
    try:
        run(args)
    except Exception as exc:
        path = getattr(args, "actual_output", ROOT/args.output)/"status.json"
        message = f"{type(exc).__name__}:{str(exc).replace(str(ROOT), '.')}"
        if hasattr(args, "actual_output") and (path.parent/"protocol.json").exists() and not args.audit_only:
            old = json.loads(path.read_text()) if path.exists() else {}
            if old.get("status") != "complete":
                strict_json(path, {"status": "incomplete", "blocked_at_utc": now(), "blocker": message})
        print(f"INCOMPLETE:{message}", file=sys.stderr)
        raise
