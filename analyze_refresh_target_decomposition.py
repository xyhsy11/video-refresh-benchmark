"""Retrospective add/move/remove refresh decomposition; saved features only."""
from __future__ import annotations

import os
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "1"

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

import analyze_task_signal_ablation as ablation
import analyze_task_signals as original
import recognition_baseline_data as data
import refresh_target_decomposition_utils as u
from fresh_confirmation_data import strict_json
from research_utils import ROOT, relative_path, sha256_file, now

NEW_SOURCES = ["analyze_refresh_target_decomposition.py", "refresh_target_decomposition_utils.py",
               "verify_refresh_target_decomposition.py", "refresh_target_decomposition_report.py",
               "test_refresh_target_decomposition.py"]


def csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, float_format="%.17g", na_rep="")


def identities(recognition: Path, prior: Path) -> tuple[dict, dict, dict]:
    for name in ["protocol.json", "SHA256SUMS.txt", "counterfactual_targets.csv",
                 "out_of_fold_predictions.csv", "exploratory_predictions.csv"]:
        if not (prior/name).is_file():
            raise FileNotFoundError(f"Required cached prerequisite missing: {relative_path(prior/name)}")
    p = data.read(prior/"protocol.json")
    assert p["recognition_input"] == relative_path(recognition)
    manifest, head, hashes = ablation.identities(recognition, ROOT/p["diagnostic_input"])
    assert p["input_sha256"] == hashes
    assert {n: sha256_file(ROOT/n) for n in p["source_sha256"]} == p["source_sha256"]
    assert data.read(prior/"status.json")["status"] == "complete"
    assert data.read(prior/"verification.json")["passed"]
    for line in (prior/"SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if not (prior/name).is_file():
            raise FileNotFoundError(f"Required prior artifact missing: {relative_path(prior/name)}")
        assert sha256_file(prior/name) == digest, name
        hashes[relative_path(prior/name)] = digest
    hashes[relative_path(prior/"SHA256SUMS.txt")] = sha256_file(prior/"SHA256SUMS.txt")
    return manifest, head, hashes


def frozen_analysis_only():
    stack = ExitStack()
    stack.enter_context(original.frozen_only())
    for target in ["task_signal_utils.fit_ridge", "task_signal_ablation_utils.fit_array",
                   "task_signal_ablation_utils.fit_bundle", "analyze_task_signal_ablation.fit_oof",
                   "analyze_task_signals.fit_probes"]:
        stack.enter_context(patch(target, side_effect=AssertionError("Decomposition must not fit predictors")))
    return stack


def freeze(recognition: Path, prior: Path, output: Path, seed: int, resamples: int) -> dict:
    manifest, _, hashes = identities(recognition, prior)
    sources = {name: sha256_file(ROOT/name) for name in sorted(set(NEW_SOURCES+ablation.NEW_SOURCES+original.SOURCES))}
    if (output/"protocol.json").exists():
        p = data.read(output/"protocol.json")
        assert p["seed"] == seed and p["bootstrap"]["resamples"] == resamples
        assert p["source_sha256"] == sources and p["input_sha256"] == hashes
        assert p["recognition_input"] == relative_path(recognition) and p["ablation_input"] == relative_path(prior)
        original.preserved(p)
        return p
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Unrecognized nonempty output; historical results will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    members = [{k: e[k] for k in ["video_id", "source_video_id", "label", "recognition_role",
               "file_identity", "candidate_indices", "candidate_timestamps_seconds", "reference_directory",
               "reference_artifact_sha256"]} for e in manifest["videos"] if e["recognition_role"] != "train"]
    p = {"experiment": "refresh_target_decomposition_v1", "frozen_at_utc": now(),
         "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
         "previous_tracked_blob_identities": original.original_inventory(), "source_sha256": sources,
         "input_sha256": hashes, "recognition_input": relative_path(recognition), "ablation_input": relative_path(prior),
         "seed": seed, "classes": manifest["classes"], "class_to_index": {c: i for i, c in enumerate(manifest["classes"])},
         "source_membership": members, "cohorts": {"development": {"sources": 100, "opportunities": 900},
                                                  "exploratory": {"sources": 40, "opportunities": 360}},
         "classifier": {"path": relative_path(recognition/"classifiers/20/parameters.json"),
                        "sha256": sha256_file(recognition/"classifiers/20/parameters.json"),
                        "pooling": "mean of sixteen latest-cache outputs through the unchanged frozen affine head",
                        "arithmetic": "float64;existing logaddexp cross entropy;no normalization refit"},
         "opportunity": "candidate ordinals r=4,8,12;t=r-3,r-2,r-1;all nine per video retained",
         "schedules": {"A": "[0,4,8,12]", "B": "sorted(A union {t}),5 calls",
                       "C": "sorted((A minus {r}) union {t}),4 calls", "D": "sorted(A minus {r}),3 calls"},
         "forward": {"benefit": "LA-LB", "cost": "LC-LB", "net": "LA-LC=benefit-cost"},
         "reverse": {"cost": "LD-LA", "benefit": "LD-LC", "net": "benefit-cost"},
         "averaged": "mean of two benefits and mean of two costs;difference still LA-LC",
         "tolerances": {"sign": u.SIGN_TOL, "algebra_absolute": u.IDENTITY_ATOL, "algebra_relative": 0,
                        "replay_absolute": u.REPLAY_ATOL, "replay_relative": u.REPLAY_RTOL,
                        "provenance": "unchanged prior 1e-12 sign convention;existing verify_task_signals independent replay tolerance1e-10"},
         "relative_target_discrepancy": "abs(recomputed-original)/abs(original) when abs(original)>1e-12;otherwise null with reason;absolute discrepancy always reported",
         "primary": "each cohort separately:forward benefit>1e-12 AND net<-1e-12,count/all opportunities",
         "secondary": ["same count/harmful moves", "same count/beneficial additions",
                       "signed component means and mean absolute magnitudes", "reverse and averaged attribution",
                       "benefit/cost sign changes,forward-versus-reverse cancellation changes,robust overlap",
                       "same summaries among fixed C-nonlinear selections", "saved C-nonlinear realized utility reproduction"],
         "selection": "load saved development OOF/exploratory C-nonlinear;select iff prediction>1e-12;no refit,threshold search or alternate model",
         "continuous_aggregation": "mean within source first;selected-subset component means average only sources with selected opportunities,with count;nonselected-source means undefined,not zero",
         "conditional_rates": "pooled count numerator/denominator;recompute both inside source bootstrap;zero-denominator point estimate null;bootstrap invalid counts explicit",
         "bootstrap": {"resamples": resamples, "seed": seed, "rng": "PCG64", "percentile_method": "linear",
                       "unit": "whole source video,keep all nine opportunities;shared index matrix across scopes/orders/metrics within cohort",
                       "interpretation": "descriptive conditional diagnostics;exclude training/model-selection/repeated-research uncertainty"},
         "sensitivity_definition": "three-way benefit/cost signs at1e-12;forward versus reverse cancellation flags;no arbitrary robustness cutoff",
         "example_selection": ["largest positive net gain", "most negative net among forward cancellations",
                               "largest absolute order interaction among benefit-sign,cost-sign or cancellation-sensitive rows"],
         "example_ties": "cohort,source_video_id,r,t ascending;prefer distinct examples;report absent categories explicitly",
         "limits": ["both cohorts developmental,recognition regularization used development,exploratory previously inspected",
                    "B5 and D3 are unequal-budget diagnostic controls,not equal-cost deployment gains",
                    "retrospective whole-video nonlinear loss;neither order is a unique causal decomposition",
                    "early-addition benefit also uses whole-video context,not an observable online signal",
                    "unknown labels,representation,data support and future-content predictability unresolved",
                    "full-clip candidates and future-reference codec vectors:offline",
                    "no accuracy improvement,adaptive speedup,online controller or publication-ready claim",
                    "no completed fresh-confirmation cohort parsed"],
         "actual_RGB_encoder_executions": 0, "video_decodes": 0, "downloads": 0,
         "recognition_training_calls": 0, "predictor_training_calls": 0}
    strict_json(output/"protocol.json", p)
    strict_json(output/"environment.json", {"recorded_at_utc": now(), "python": platform.python_version(),
        "os": platform.platform(), "packages": {n: version(n) for n in ["numpy", "pandas", "torch", "matplotlib", "av"]},
        "device": "CPU saved-feature float64 arithmetic", "new_dependencies": [], "fitting": False})
    return p


def load_opportunities(prior: Path, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(prior/"counterfactual_targets.csv", float_precision="round_trip")
    assert table.groupby("cohort").size().to_dict() == {"development": 900, "exploratory": 360}
    assert table.example_id.nunique() == 1260 and table.groupby("source_video_id").size().eq(9).all()
    members = {e["video_id"]: e for e in manifest["videos"] if e["recognition_role"] != "train"}
    assert set(table.video_id) == set(members)
    for row in table.itertuples():
        e = members[row.video_id]
        assert row.source_video_id == e["source_video_id"] and row.label == e["label"]
        assert row.cohort == e["recognition_role"] and row.label_index == manifest["classes"].index(row.label)
        u.schedules(row.t, row.replaced_refresh)
    parts = []
    for filename, cohort in [("out_of_fold_predictions.csv", "development"), ("exploratory_predictions.csv", "exploratory")]:
        data_table = pd.read_csv(prior/filename, float_precision="round_trip")
        g = data_table[data_table.configuration == "C-nonlinear"].copy()
        expected_cohort = "development_oof" if cohort == "development" else cohort
        assert set(g.cohort) == {expected_cohort}
        expected = table[table.cohort == cohort].set_index("example_id").loc[g.example_id]
        assert len(g) == len(expected) == (900 if cohort == "development" else 360)
        assert not g.example_id.duplicated().any()
        for name in ["video_id", "source_video_id", "t", "last_refresh", "refresh_gain"]:
            np.testing.assert_array_equal(g[name], expected[name])
        assert (g.earlier_selected == (g.prediction > u.SIGN_TOL)).all()
        np.testing.assert_array_equal(g.realized_gain, g.refresh_gain*g.earlier_selected)
        parts.append(g)
    predictions = pd.concat(parts, ignore_index=True)
    assert predictions.example_id.nunique() == 1260
    return table, predictions


def run(args) -> None:
    recognition, prior, output = [(ROOT/p).resolve() for p in [args.recognition_input, args.ablation_input, args.output]]
    if args.audit_only and not (output/"verification.json").exists():
        raise FileNotFoundError("Audit requires evaluated outputs and verification.json")
    p = freeze(recognition, prior, output, args.seed, args.bootstrap)
    if args.audit_only or ((output/"status.json").exists() and data.read(output/"status.json")["status"] == "complete"):
        from verify_refresh_target_decomposition import audit
        print(json.dumps(audit(recognition, prior, output, p), indent=2))
        return
    strict_json(output/"status.json", {"status": "running", "started_at_utc": now()})
    manifest, head, _ = identities(recognition, prior)
    targets, predictions = load_opportunities(prior, manifest)
    parts = []
    with frozen_analysis_only():
        for role in ["development", "exploratory"]:
            videos = data.load_role(manifest["videos"], role)
            rows = u.evaluate(videos, targets[targets.cohort == role],
                              predictions[predictions.cohort == ("development_oof" if role == "development" else role)],
                              head, manifest["classes"])
            parts.append(rows)
            print(f"{role}: decomposed {len(rows)} opportunities from {len(videos)} sources", flush=True)
    table = pd.concat(parts, ignore_index=True)
    csv(output/"per_opportunity_decomposition.csv", table)
    video = u.video_summary(table)
    csv(output/"per_video_summary.csv", video)
    aggregate, intervals, indices = u.summarize(video, args.seed, args.bootstrap)
    strict_json(output/"aggregate_metrics.json", aggregate)
    strict_json(output/"bootstrap_intervals.json", intervals)
    np.savez_compressed(output/"bootstrap_resampling_indices.npz", **indices)
    strict_json(output/"metadata.json", {"evaluated_at_utc": now(), "sources": 140, "opportunities": 1260,
        "simulated_schedule_replays": 5040, "simulated_output_slots": 80640,
        "simulated_logical_refreshes_across_all_controls": 20160, "actual_RGB_encoder_executions": 0,
        "video_decodes": 0, "downloads": 0, "recognition_training_calls": 0, "predictor_training_calls": 0,
        "fresh_confirmation_parsed": 0, "B_and_D": "unequal-budget diagnostic controls"})
    from verify_refresh_target_decomposition import audit
    verification = audit(recognition, prior, output, p)
    test = subprocess.run([sys.executable, "-m", "unittest", "test_refresh_target_decomposition", "-v"],
                          cwd=ROOT, capture_output=True, text=True)
    assert test.returncode == 0, test.stdout+test.stderr
    release = subprocess.run([sys.executable, "validate_release.py"], cwd=ROOT, capture_output=True, text=True)
    assert release.returncode == 0, release.stdout+release.stderr
    verification.update(focused_tests={"returncode": 0, "output": test.stdout+test.stderr},
                        release_validator={"returncode": 0, "output": release.stdout+release.stderr})
    strict_json(output/"verification.json", verification)
    from refresh_target_decomposition_report import report
    report(output)
    strict_json(output/"status.json", {"status": "complete", "completed_at_utc": now(), "blocker": None})
    write_ledger(output)
    print(f"Complete: {relative_path(output)}", flush=True)


def write_ledger(output: Path) -> None:
    files = sorted(f for f in output.rglob("*") if f.is_file() and f.name != "SHA256SUMS.txt")
    (output/"SHA256SUMS.txt").write_text("".join(f"{sha256_file(f)}  {f.relative_to(output).as_posix()}\n" for f in files), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recognition-input", type=Path, default=Path("results/recognition_baseline_v2"))
    parser.add_argument("--ablation-input", type=Path, default=Path("results/task_signal_ablation_v2"))
    parser.add_argument("--output", type=Path, default=Path("results/refresh_target_decomposition_v1"))
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
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
