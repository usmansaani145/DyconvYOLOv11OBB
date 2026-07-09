#!/usr/bin/env python3
"""Download DOTA-ShipBench from Hugging Face."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import DATA_ROOT, HF_DATASET_REPO, HF_DATASET_URL

_REQUIRED = (
    "train/images",
    "val/images",
    "dataset.yaml",
    "gt_coco_filtered.json",
)


def is_dataset_ready(root: Path) -> bool:
    root = Path(root)
    if not all((root / part).exists() for part in _REQUIRED):
        return False
    train_img = root / "train" / "images"
    return train_img.is_dir() and any(train_img.iterdir())


def normalize_layout(root: Path) -> None:
    """Handle repos that nest everything under a dataset/ subfolder."""
    root = Path(root)
    nested = root / "dataset"
    if not nested.is_dir() or (root / "train").is_dir():
        return
    for item in nested.iterdir():
        dest = root / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))
    nested.rmdir()


def download_dataset(dest: Path | None = None, *, force: bool = False) -> Path:
    dest = Path(dest or DATA_ROOT)

    if is_dataset_ready(dest) and not force:
        print(f"Dataset already present at {dest}")
        return dest

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_DATASET_REPO}")
    print(f"  Source: {HF_DATASET_URL}")
    print(f"  Destination: {dest}")

    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    normalize_layout(dest)

    if not is_dataset_ready(dest):
        raise SystemExit(
            f"Download finished but dataset layout is incomplete at {dest}. "
            f"Expected train/images, val/images, dataset.yaml, gt_coco_filtered.json."
        )

    print(f"Dataset ready at {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DOTA-ShipBench from Hugging Face",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=f"Destination directory (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the dataset already exists",
    )
    args = parser.parse_args()
    download_dataset(args.dest, force=args.force)


if __name__ == "__main__":
    main()
