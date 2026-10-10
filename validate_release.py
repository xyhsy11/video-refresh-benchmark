"""Validate the frozen public pilot without downloading data or using a GPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from summarize import per_video_summary, validate_trials

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "results" / "pilot_2026_10_10"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_links(path: Path) -> int:
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8"))
    checked = 0
    for target in links:
        if target.startswith(("https://", "http://", "#")):
            continue
        relative = target.split("#", 1)[0]
        if not (path.parent / relative).exists():
            raise AssertionError(f"Broken relative link in {path.name}: {target}")
        checked += 1
    return checked


def main() -> None:
    manifest = read_json(SNAPSHOT / "manifest.json")
    frame = pd.read_csv(SNAPSHOT / "pilot.csv")
    metadata = read_json(SNAPSHOT / "pilot.metadata.json")
    validation = validate_trials(frame, metadata, manifest)
    if digest(SNAPSHOT / "pilot.csv") != metadata["csv_sha256"]:
        raise AssertionError("Published raw trials changed")
    if validation["videos"] != 10 or validation["measured_rows"] != 150:
        raise AssertionError("Expected ten videos and 150 measured trials")
    published = pd.read_csv(SNAPSHOT / "summary.csv")
    regenerated = per_video_summary(frame, metadata, manifest)
    pd.testing.assert_frame_equal(published, regenerated, check_exact=False, rtol=1e-12, atol=1e-9)
    observed = {
        "median_reduction_pct": f"{regenerated.latency_reduction_pct.median():.2f}",
        "median_speedup": f"{regenerated.speedup.median():.3f}",
        "min_reduction_pct": f"{regenerated.latency_reduction_pct.min():.2f}",
        "max_reduction_pct": f"{regenerated.latency_reduction_pct.max():.2f}",
    }
    if observed != {"median_reduction_pct": "34.74", "median_speedup": "1.532", "min_reduction_pct": "17.52", "max_reduction_pct": "45.54"}:
        raise AssertionError(f"Headline statistics differ: {observed}")
    single = pd.read_csv(SNAPSHOT / "single_video.csv")
    single_meta = read_json(SNAPSHOT / "single_video.metadata.json")
    validate_trials(single, single_meta, manifest)
    if len(single) != 15 or digest(SNAPSHOT / "single_video.csv") != single_meta["csv_sha256"]:
        raise AssertionError("Initial single-video data changed")
    # Requirements were separated for release; the full original pins remain archived.
    for name, expected in metadata["source_code_sha256"].items():
        source = SNAPSHOT / "requirements-tested.txt" if name == "requirements.txt" else ROOT / name
        if digest(source) != expected:
            raise AssertionError(f"Measurement source bytes differ: {name}")
    snapshot_files = {p.name for p in SNAPSHOT.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"}
    listed = set()
    for line in (SNAPSHOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if name not in snapshot_files or digest(SNAPSHOT / name) != expected:
            raise AssertionError(f"Snapshot checksum mismatch: {name}")
        listed.add(name)
    if listed != snapshot_files:
        raise AssertionError("Snapshot checksum inventory is incomplete")
    private_path = re.compile(r"(?i)(?<![a-z])[a-z]:[\\/]|[\\/]users[\\/]|/home/")
    for path in [*ROOT.glob("*.py"), ROOT / "README.md", ROOT / "requirements.txt", ROOT / "requirements-cuda.txt", *SNAPSHOT.iterdir()]:
        # The validator contains the detection patterns themselves, not local paths.
        if path.is_file() and path.suffix != ".png" and path.resolve() != Path(__file__).resolve():
            if private_path.search(path.read_text(encoding="utf-8")):
                raise AssertionError(f"Personal/local absolute path in public file: {path.name}")
    for name in ("requirements.txt", "requirements-cuda.txt"):
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        if any("file://" in line for line in lines if not line.lstrip().startswith("#")):
            raise AssertionError(f"Machine-specific dependency in {name}")
    link_count = sum(validate_links(p) for p in [ROOT / "README.md", ROOT / "THIRD_PARTY_NOTICES.md", SNAPSHOT / "README.md"])
    print("Public snapshot PASS: ten videos, 150 pilot rows, 15 single-video rows; all counters valid.")
    print("CSV-derived headlines PASS: 34.74% median reduction, 1.532x median speedup, 17.52-45.54% range.")
    print(f"Original timing source hashes and {len(listed)} snapshot checksums PASS; {link_count} relative links PASS.")
    print("Personal-path and installation-requirement checks PASS; no GPU timing was rerun.")


if __name__ == "__main__":
    main()
