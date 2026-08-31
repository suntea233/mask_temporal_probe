from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import torch


def _command(args: list[str]) -> str:
    try:
        return subprocess.run(args, check=False, capture_output=True, text=True).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def capture(project: Path) -> dict:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_count": torch.cuda.device_count(),
        "gpu_models": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_available else [],
        "nvidia_smi": _command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "nvcc": _command(["nvcc", "--version"]),
        "official_repo_commit": _command(["git", "-C", str(project / "vendor/LLaDA"), "rev-parse", "HEAD"]),
    }


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    path = project / "results/environment.json"
    path.write_text(json.dumps(capture(project), indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
