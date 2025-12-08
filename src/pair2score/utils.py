from __future__ import annotations

import json
import os
import random
import socket
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch

_DEFAULT_CUBLAS_WORKSPACE = ":4096:8"


def _normalize_cuda_device(cuda_device: Any):
    if cuda_device is None or cuda_device == "all":
        return "all"
    if isinstance(cuda_device, torch.device):
        if cuda_device.type != "cuda":
            raise ValueError(f"Expected CUDA device, got {cuda_device}")
        return cuda_device.index
    if isinstance(cuda_device, str):
        if cuda_device.startswith("cuda:"):
            return int(cuda_device.split(":", 1)[1])
        return int(cuda_device)
    return int(cuda_device)


def set_global_seed(seed: int, *, cuda_device: Any = "all", enable_cuda_seed: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and enable_cuda_seed:
        target = _normalize_cuda_device(cuda_device)
        if target == "all":
            torch.cuda.manual_seed_all(seed)
        else:
            torch.cuda.set_device(target)
            torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_determinism(enabled: bool = False, *, warn_only: bool = False, allow_tf32: bool = True) -> None:
    """
    Optionally enable deterministic algorithms in PyTorch.

    Parameters
    ----------
    enabled:
        Whether to call ``torch.use_deterministic_algorithms(True)``.
    warn_only:
        Passed to ``torch.use_deterministic_algorithms`` to fall back with warnings instead of errors.
    allow_tf32:
        Controls ``torch.backends.cuda.matmul.allow_tf32`` / ``torch.backends.cudnn.allow_tf32`` to avoid TF32 kernels
        when seeking strict determinism.
    """
    if not enabled:
        return

    torch.use_deterministic_algorithms(True, warn_only=warn_only)

    if torch.cuda.is_available():
        if not warn_only:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _DEFAULT_CUBLAS_WORKSPACE)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = allow_tf32


def format_run_stamp() -> str:
    """Return a human-readable timestamp in Europe/Oslo time."""
    oslo = ZoneInfo("Europe/Oslo")
    return datetime.now(oslo).strftime("%Y-%m-%d %H:%M:%S %Z")


def current_hostname() -> str:
    return socket.gethostname()


def write_metadata_file(target_dir: Path, info: dict[str, Any], filename: str) -> Path:
    """Write a simple key/value metadata file."""
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename

    flattened = {}
    for key, value in info.items():
        if isinstance(value, (list, tuple)):
            flattened[key] = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            flattened[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            flattened[key] = str(value)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Pair2Score run metadata\n")
        for key in sorted(flattened):
            handle.write(f"{key}: {flattened[key]}\n")
    return path
