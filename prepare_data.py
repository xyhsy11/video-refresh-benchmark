"""Download, validate, preserve, and select the ten Kinetics tiny validation MP4s."""
from __future__ import annotations

import shutil
from pathlib import Path
import tempfile
from urllib.request import urlopen
import zipfile

from research_utils import ROOT, MANIFEST, load_manifest, now, relative_path, sha256_file, write_json

SOURCE_URL = "https://download.openmmlab.com/mmaction/kinetics400_tiny.zip"
FIRST_VIDEO = "y5Iu7XkTqV0.mp4"


def validate_archive(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Archive CRC failure: {bad}")
        members = sorted(name for name in archive.namelist() if name.startswith("kinetics400_tiny/val/") and name.lower().endswith(".mp4"))
        if len(members) != 10 or not any(Path(name).name == FIRST_VIDEO for name in members):
            raise ValueError(f"Expected ten validation MP4 files including {FIRST_VIDEO}; found {len(members)}")
        return members


def main() -> None:
    source = ROOT / "data" / "source"
    videos = ROOT / "data" / "videos"
    source.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)
    archive_path = source / "kinetics400_tiny.zip"
    reused = False
    if archive_path.exists():
        # Never replace an existing invalid file silently.
        validate_archive(archive_path)
        reused = True
        print(f"Reusing valid archive: {relative_path(archive_path)}")
    else:
        print(f"Downloading {SOURCE_URL}", flush=True)
        with tempfile.NamedTemporaryFile(dir=source, suffix=".download", delete=False) as target:
            temporary = Path(target.name)
            try:
                with urlopen(SOURCE_URL, timeout=90) as response:
                    shutil.copyfileobj(response, target, length=1024 * 1024)
            except Exception:
                target.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            validate_archive(temporary)
            temporary.replace(archive_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    members = validate_archive(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            destination = (source / item.filename).resolve()
            if not destination.is_relative_to(source.resolve()):
                raise ValueError(f"Unsafe archive member: {item.filename}")
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                # Validate an existing extracted file without destroying unrelated files.
                import hashlib
                archived_hash = hashlib.sha256(archive.read(item)).hexdigest()
                if sha256_file(destination) != archived_hash:
                    raise ValueError(f"Existing extracted file differs; preserved: {destination}")
            else:
                with archive.open(item) as incoming, destination.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    records = []
    for member in members:
        original = source / member
        destination = videos / original.name
        checksum = sha256_file(original)
        if destination.exists():
            if sha256_file(destination) != checksum:
                raise ValueError(f"Existing destination differs; preserved: {destination}")
        else:
            shutil.copy2(original, destination)
        records.append({"filename": original.name, "source_path": relative_path(original), "selected_path": relative_path(destination), "sha256": checksum})
    manifest = load_manifest()
    annotations = sorted(relative_path(p) for p in (source / "kinetics400_tiny").rglob("*") if p.is_file() and p.suffix.lower() != ".mp4")
    manifest["dataset"] = {
        "source_url": SOURCE_URL, "prepared_at_utc": now(),
        "archive_path": relative_path(archive_path), "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size, "reused_archive": reused,
        "selected_files": [record["selected_path"] for record in records],
        "selected_files_details": records, "preserved_annotation_files": annotations,
        "copy_method": "byte-for-byte shutil.copy2; encoding unchanged",
    }
    write_json(MANIFEST, manifest)
    write_json(ROOT / "results" / "dataset.json", manifest["dataset"])
    print(f"Prepared {len(records)} validation videos; preserved {len(annotations)} annotation/support files.")
    for record in records:
        print(f"  {record['selected_path']}")


if __name__ == "__main__":
    main()
