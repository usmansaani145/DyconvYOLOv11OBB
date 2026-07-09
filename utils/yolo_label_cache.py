"""Pre-build Ultralytics YOLO label caches safely for parallel training jobs."""

from __future__ import annotations

import fcntl
import glob
from pathlib import Path
from typing import Any

from shared.constants import DATA_ROOT, DATASET_YAML

_LOCK_PATH = DATA_ROOT / ".labels_cache.lock"
_PATCHED = False


def _patch_ultralytics_cache_save() -> None:
    """Make cache save tolerant of missing files (older Ultralytics builds)."""
    global _PATCHED
    if _PATCHED:
        return

    from ultralytics.data import utils as ul_utils

    if getattr(ul_utils, "_ship_obb_cache_patched", False):
        _PATCHED = True
        return

    original = ul_utils.save_dataset_cache_file

    def safe_save_dataset_cache_file(prefix: str, path: Path, x: dict, version: str) -> None:
        path = Path(path)
        x = dict(x)
        x["version"] = version
        if not ul_utils.is_dir_writeable(path.parent):
            ul_utils.LOGGER.warning(
                f"{prefix}Cache directory {path.parent} is not writable, cache not saved."
            )
            return
        if path.exists():
            path.unlink()
        try:
            import numpy as np

            with open(str(path), "wb") as file:
                np.save(file, x)
            ul_utils.LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as exc:
            path.unlink(missing_ok=True)
            ul_utils.LOGGER.warning(f"{prefix}WARNING Failed to save cache to {path}: {exc}")

    ul_utils.save_dataset_cache_file = safe_save_dataset_cache_file
    ul_utils._ship_obb_cache_patched = True
    _PATCHED = True


def _load_dataset_dict() -> dict[str, Any]:
    import yaml

    with open(DATASET_YAML, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data["path"] = str(DATA_ROOT)
    return data


def _cache_is_valid(cache_path: Path, img_dir: Path) -> bool:
    from ultralytics.data.dataset import DATASET_CACHE_VERSION
    from ultralytics.data.utils import get_hash, img2label_paths, load_dataset_cache_file

    im_files = sorted(glob.glob(str(img_dir / "*.*")))
    if not im_files:
        return False
    label_files = img2label_paths(im_files)
    cache = load_dataset_cache_file(cache_path)
    return cache["version"] == DATASET_CACHE_VERSION and cache["hash"] == get_hash(
        label_files + im_files
    )


def _build_split_cache(split: str, data: dict[str, Any]) -> None:
    from ultralytics.cfg import IterableSimpleNamespace
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils import colorstr

    img_path = str(DATA_ROOT / split / "images")
    YOLODataset(
        img_path=img_path,
        imgsz=1024,
        batch_size=1,
        augment=False,
        hyp=IterableSimpleNamespace(),
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix=colorstr(f"{split}: "),
        task="obb",
        classes=None,
        data=data,
        fraction=1.0,
    )


def ensure_yolo_label_caches() -> None:
    """Build train/val labels.cache once under an exclusive lock."""
    _patch_ultralytics_cache_save()
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _load_dataset_dict()

    with open(_LOCK_PATH, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        for split in ("train", "val"):
            cache_path = DATA_ROOT / split / "labels.cache"
            img_dir = DATA_ROOT / split / "images"
            if cache_path.is_file() and _cache_is_valid(cache_path, img_dir):
                print(f"YOLO label cache OK: {cache_path}")
                continue
            print(f"Building YOLO label cache: {cache_path}")
            _build_split_cache(split, data)
        print("YOLO label caches ready.")
