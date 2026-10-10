r"""Print and save dependency versions and a real CUDA correctness check.

Run with .\.venv\Scripts\python.exe .\check_env.py in PowerShell.
Exit status 0 means imports and both GPU checks passed; otherwise it is 1.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "av": "av",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "Pillow": "PIL",
}


def inspect_cpu() -> dict:
    details = {"name": platform.processor(), "logical_processors": os.cpu_count()}
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | ConvertTo-Json -Compress"],
                capture_output=True, text=True, timeout=20, check=True,
            )
            details["windows_processors"] = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            details["query_error"] = str(exc)
    return details


def inspect_nvidia() -> dict:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"gpus": [], "error": "nvidia-smi was not found on PATH."}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        gpus = [
            {"name": row[0].strip(), "driver_version": row[1].strip(), "memory_mib": int(row[2].strip())}
            for row in csv.reader(io.StringIO(result.stdout)) if row
        ]
        return {"gpus": gpus, "error": None}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return {"gpus": [], "error": f"{type(exc).__name__}: {exc}"}


def collect_report() -> dict:
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "full_version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "in_virtual_environment": sys.prefix != sys.base_prefix,
        },
        "os": platform.platform(),
        "cpu": inspect_cpu(),
        "packages": {},
        "ffmpeg_version": None,
        "ffmpeg_libraries": {},
        "nvidia_smi": inspect_nvidia(),
        "cuda": {
            "build_version": None,
            "available": False,
            "visible_devices_setting": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "devices": [],
            "tensor_test": {"passed": False, "operation": "A @ B on cuda:0", "result": None},
            "torchvision_test": {"passed": False, "operation": "torchvision.ops.nms on cuda:0", "result": None},
        },
        "errors": [],
        "diagnosis": [],
    }
    modules = {}
    for distribution, module_name in PACKAGES.items():
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = None
        try:
            modules[distribution] = importlib.import_module(module_name)
            report["packages"][distribution] = {"version": version, "import_ok": True}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            report["packages"][distribution] = {"version": version, "import_ok": False, "error": error}
            report["errors"].append(f"Cannot import {distribution}: {error}")

    if "av" in modules:
        av = modules["av"]
        report["ffmpeg_version"] = getattr(av, "ffmpeg_version_info", "not exposed by this PyAV build")
        report["ffmpeg_libraries"] = {
            name: ".".join(str(part) for part in version)
            for name, version in sorted(av.library_versions.items())
        }

    torch = modules.get("torch")
    if torch is None:
        report["diagnosis"].append("PyTorch could not be imported. Use the project .venv interpreter and reinstall the pinned wheels; see the import error above.")
    else:
        cuda = report["cuda"]
        cuda["build_version"] = torch.version.cuda
        try:
            cuda["available"] = torch.cuda.is_available()
            if cuda["available"]:
                for index in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(index)
                    cuda["devices"].append({
                        "index": index,
                        "name": properties.name,
                        "memory_bytes": properties.total_memory,
                        "compute_capability": f"{properties.major}.{properties.minor}",
                    })
                with torch.inference_mode():
                    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda:0")
                    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda:0")
                    product = left @ right
                    torch.cuda.synchronize(0)
                    actual = product.cpu().tolist()
                    expected = [[19.0, 22.0], [43.0, 50.0]]
                    cuda["tensor_test"].update({
                        "device": str(product.device), "input_a": left.cpu().tolist(),
                        "input_b": right.cpu().tolist(), "result": actual,
                        "expected": expected, "passed": actual == expected,
                    })
                    if actual != expected:
                        report["errors"].append("GPU matrix multiplication returned an unexpected result.")
                    if "torchvision" in modules:
                        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]], device="cuda:0")
                        scores = torch.tensor([0.9, 0.8, 0.7], device="cuda:0")
                        kept = modules["torchvision"].ops.nms(boxes, scores, 0.5)
                        torch.cuda.synchronize(0)
                        actual_indices = kept.cpu().tolist()
                        cuda["torchvision_test"].update({"result": actual_indices, "expected": [0, 2], "passed": actual_indices == [0, 2]})
                        if actual_indices != [0, 2]:
                            report["errors"].append("torchvision's GPU operation returned an unexpected result.")
            else:
                report["errors"].append("CUDA is unavailable; GPU benchmarks must wait until this is fixed.")
                if torch.version.cuda is None:
                    report["diagnosis"].append("This is a CPU-only PyTorch installation. Install the CUDA wheels using the README command.")
                else:
                    # Force initialization to reveal the driver's actual failure message.
                    try:
                        torch.cuda.init()
                    except Exception as exc:
                        report["diagnosis"].append(f"CUDA initialization error: {type(exc).__name__}: {exc}")
                    if not report["nvidia_smi"]["gpus"]:
                        report["diagnosis"].append("nvidia-smi could not see an NVIDIA GPU. Check the NVIDIA driver and Device Manager.")
                    if cuda["visible_devices_setting"] is not None:
                        report["diagnosis"].append("CUDA_VISIBLE_DEVICES is set. Check whether it hides or selects a nonexistent GPU.")
                    report["diagnosis"].append("Compare the NVIDIA driver with this PyTorch CUDA build; rerun from .venv and restart Windows after any driver update.")
        except Exception as exc:
            report["errors"].append(f"GPU check failed: {type(exc).__name__}: {exc}")
            report["diagnosis"].append("Use the error above to check driver compatibility and matching CUDA torch/torchvision wheels before benchmarking.")

    report["passed"] = (
        not report["errors"]
        and report["cuda"]["tensor_test"]["passed"]
        and report["cuda"]["torchvision_test"]["passed"]
    )
    return report


def format_report(report: dict) -> str:
    cuda = report["cuda"]
    lines = [
        "Video latency project environment check",
        f"Checked at (UTC): {report['checked_at_utc']}",
        f"OS: {report['os']}",
        f"CPU: {report['cpu']}",
        f"Python: {report['python']['full_version']}",
        f"Python executable: {report['python']['executable']}",
        f"Virtual environment: {report['python']['in_virtual_environment']}",
    ]
    for name, details in report["packages"].items():
        label = {"torch": "PyTorch", "av": "PyAV"}.get(name, name)
        lines.append(f"{label}: {details['version']} (import {'OK' if details['import_ok'] else 'FAILED'})")
    lines.append(f"FFmpeg release used by PyAV: {report['ffmpeg_version']}")
    lines.append("FFmpeg library versions used by PyAV:")
    lines.extend(f"  {name}: {version}" for name, version in report["ffmpeg_libraries"].items())
    for gpu in report["nvidia_smi"]["gpus"]:
        lines.append(f"GPU (nvidia-smi): {gpu['name']}; driver {gpu['driver_version']}; {gpu['memory_mib']} MiB")
    if report["nvidia_smi"]["error"]:
        lines.append(f"nvidia-smi: {report['nvidia_smi']['error']}")
    lines.append(f"torch.version.cuda: {cuda['build_version']}")
    lines.append(f"torch.cuda.is_available(): {cuda['available']}")
    for gpu in cuda["devices"]:
        lines.append(f"GPU (PyTorch cuda:{gpu['index']}): {gpu['name']}; compute capability {gpu['compute_capability']}")
    for key in ("tensor_test", "torchvision_test"):
        test = cuda[key]
        lines.append(f"{test['operation']}: {'PASS' if test['passed'] else 'FAIL / NOT RUN'}; result = {test['result']}")
    lines.extend(f"ERROR: {error}" for error in report["errors"])
    lines.extend(f"DIAGNOSIS: {diagnosis}" for diagnosis in report["diagnosis"])
    lines.append(f"Overall: {'PASS' if report['passed'] else 'FAIL'}")
    lines.append(
        "The GPU completed the calculation correctly. This is a setup check, not a latency measurement."
        if report["passed"] else "Resolve the errors above before running GPU benchmarks."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    report = collect_report()
    rendered = format_report(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "environment_check.json"
    text_path = args.output_dir / "environment_check.txt"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "environment.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    text_path.write_text(rendered, encoding="utf-8")
    # Preserve Unicode Windows paths even in redirected PowerShell output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(rendered, end="")
    print(f"Saved: {text_path}\nSaved: {json_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
