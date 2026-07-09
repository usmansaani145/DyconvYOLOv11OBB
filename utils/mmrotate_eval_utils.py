"""Environment helpers for MMRotate eval."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def mmrotate_env(mpl_dir: str) -> dict:
    env = os.environ.copy()
    repo = os.environ.get("SHIP_OBB_REPO", str(REPO_ROOT))
    env["PYTHONPATH"] = f"/opt/mmrotate:{repo}:" + env.get("PYTHONPATH", "")
    tmp = Path(repo) / "tmp"
    env.setdefault("MPLCONFIGDIR", mpl_dir)
    env.setdefault("HOME", str(tmp / "mmrotate_home"))
    env.setdefault("TORCH_HOME", str(tmp / "torch_cache"))
    env.setdefault("XDG_CACHE_HOME", str(tmp / "torch_cache"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    return env
