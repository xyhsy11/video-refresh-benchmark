"""Shared manifest, input identity, and fixed decoder settings."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "results" / "manifest.json"
DECODER_THREAD_COUNT = 1
DECODER_THREAD_TYPE = "NONE"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {"schema_version": 1, "videos": {}}
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("videos"), dict):
        raise ValueError("Unsupported or malformed manifest.json")
    return data


def selected_input_paths(input_path: Path, manifest: dict) -> list[Path]:
    input_path = input_path.resolve()
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input does not exist: {input_path}")
    selected = manifest.get("dataset", {}).get("selected_files", [])
    paths = [ROOT / name for name in selected if (ROOT / name).resolve().parent == input_path]
    if not paths:
        raise ValueError("Directory contains no selected dataset paths in the manifest. Run prepare_data.py first.")
    return paths


def configure_decoder(stream) -> None:
    stream.codec_context.thread_count = DECODER_THREAD_COUNT
    stream.codec_context.thread_type = DECODER_THREAD_TYPE
    stream.codec_context.skip_frame = "NONE"
