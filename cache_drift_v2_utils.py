"""Frozen v2 protocol, experiment preservation, and split/identity validation."""
from __future__ import annotations

import json
from pathlib import Path

from collect_cache_drift import AGE_INPUTS, MOTION_INPUTS, protected_inventory
from research_utils import ROOT, now, relative_path, sha256_file, write_json

SEED = 20261013
SOURCE = "https://github.com/cvdfoundation/kinetics-dataset"
BASE = "https://s3.amazonaws.com/kinetics/400/"
SPLIT_COUNTS = {"train": 120, "validation": 40, "test": 40}
COMPARISONS = [("age_plus_motion", "age_only"), ("motion_only", "constant"),
               ("motion_only", "age_only"), ("age_plus_motion", "constant"),
               ("age_plus_motion", "motion_only")]
PROTOCOL = {
    "experiment": "cache_drift_v2", "sampling_seed": SEED,
    "target": "200 distinct new source videos; 20 seeded classes, 10 clips per class",
    "split": {"per_class": {"train": 6, "validation": 2, "test": 2}, "totals": SPLIT_COUNTS},
    "eligibility": ["H.264", "duration 5..15 seconds", "at least 16 actually decoded frames",
                    "finite increasing candidate PTS timestamps", "successful finite CUDA feature extraction"],
    "motion_missing": "valid clips remain eligible when some or all motion side data is missing",
    "candidates": 16, "refresh_every": 4, "cache_ordinal": "4 * floor((t-1)/4), BEFORE refresh, t=1..15",
    "age_inputs": AGE_INPUTS, "motion_inputs": MOTION_INPUTS,
    "predictors": ["constant", "age_only", "motion_only", "age_plus_motion"],
    "ridge_alpha": 1.0, "standardization": "training-only mean and population std; constants use scale 1",
    "primary_predictions": "raw saved separately; clip to [0,2] for all primary metrics",
    "fit": "training videos only; validation never used for fitting or tuning; all four test models fixed",
    "validation_candidate_rule": "lowest equal-weight mean per-video validation MAE; alphabetical tie break",
    "metrics": ["MAE", "RMSE", "Spearman, average ranks; constant vectors undefined"],
    "aggregation": "per-video then equal weight across videos",
    "comparisons_candidate_vs_reference": [list(c) for c in COMPARISONS],
    "paired_improvement": "MAE(reference) - MAE(candidate); positive means improvement",
    "percentage_improvement": "100 * paired_improvement / MAE(reference); undefined when denominator is zero",
    "bootstrap": {"resamples": 10000, "seed": SEED, "percentiles": [2.5, 97.5], "method": "linear",
                  "unit": "whole source videos, paired methods together, fixed fitted models",
                  "interpretation": "uncertainty over held-out videos conditional on fitted models"},
    "concentration": "top two positive per-video absolute gains / positive net summed gain; undefined if net <=1e-12",
    "offline": "B-frames and future-reference codec vectors; no strict causal streaming claim",
    "out_of_scope": ["adaptive controller", "recognition accuracy", "new latency measurements"],
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def preservation_inventory() -> dict:
    inventory = protected_inventory()
    for name in ["collect_cache_drift.py", "analyze_cache_drift.py"]:
        inventory[name] = sha256_file(ROOT / name)
    for p in (ROOT / "results/cache_drift_v1").rglob("*"):
        if p.is_file():
            inventory[relative_path(p)] = sha256_file(p)
    return dict(sorted(inventory.items()))


def verify_preservation(expected: dict) -> None:
    if preservation_inventory() != expected:
        raise AssertionError("Original scripts or published experiment artifacts changed")


def freeze_protocol(output: Path) -> str:
    path = output / "protocol.json"
    if path.exists():
        assert read_json(path)["protocol"] == PROTOCOL, "Frozen analysis protocol differs"
    else:
        write_json(path, {"frozen_at_utc": now(), "protocol": PROTOCOL})
    return sha256_file(path)


def pilot_sources() -> set[str]:
    manifest = read_json(ROOT / "results/pilot_2026_10_10/manifest.json")
    return {e["video_id"] for e in manifest["videos"].values()}


def validate_split(manifest: dict, check_files: bool = True, complete: bool = True) -> dict:
    entries = manifest["videos"]
    ids = [e["source_video_id"] for e in entries]
    hashes = [e["file_identity"]["sha256"] for e in entries]
    assert len(ids) == len(set(ids)) and len(hashes) == len(set(hashes)), "Duplicate source or file"
    assert not set(ids) & pilot_sources(), "Original pilot source present"
    counts = {s: sum(e["experiment_split"] == s for e in entries) for s in SPLIT_COUNTS}
    for e in entries:
        n, indices = e["decoded_frame_count"], e["candidate_indices"]
        assert e["codec"] == "h264" and 5 <= e["duration_seconds"] <= 15 and n >= 16
        assert len(indices) == 16 and indices == sorted(set(indices)) and indices[0] == 0 and indices[-1] == n-1
        if check_files:
            p = ROOT / e["relative_path"]
            assert p.stat().st_size == e["file_identity"]["size_bytes"] and sha256_file(p) == e["file_identity"]["sha256"], f"Input identity mismatch: {e['video_id']}"
    if complete:
        assert counts == SPLIT_COUNTS and len(entries) == 200, f"Incomplete split: {counts}"
        labels = set(e["label"] for e in entries)
        assert len(labels) == 20
        for label in labels:
            assert {s: sum(e["label"] == label and e["experiment_split"] == s for e in entries) for s in counts} == {"train": 6, "validation": 2, "test": 2}
    return {"passed": True, "source_and_file_overlap": False, "pilot_sources_excluded": True, "counts": counts}
