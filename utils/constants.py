"""Single source of truth for all ShipOBB paths (repo-relative, no cluster paths)."""

from __future__ import annotations

import os
from pathlib import Path

# Repository root = parent of utils/
REPO_ROOT = Path(__file__).resolve().parents[1]


def _root(env_key: str, default: Path) -> Path:
    val = os.environ.get(env_key)
    return Path(val) if val else default


DATA_ROOT = _root("SHIP_OBB_DATA_ROOT", REPO_ROOT / "dataset")
RUNS_ROOT = _root("SHIP_OBB_RUNS_ROOT", REPO_ROOT / "runs")
RESULTS_ROOT = _root("SHIP_OBB_RESULTS_ROOT", REPO_ROOT / "results")

HF_DATASET_REPO = "usmansaani145/DOTA-ShipBench"
HF_DATASET_URL = f"https://huggingface.co/datasets/{HF_DATASET_REPO}"

DOTA_TEST_ROOT = _root("DOTA_TEST_ROOT", DATA_ROOT / "dota_test")
HRSC2016_ROOT = _root("HRSC2016_ROOT", DATA_ROOT / "hrsc2016")
HRSC_IMAGES = HRSC2016_ROOT / "AllImages"
HRSC_ANNOTATIONS = HRSC2016_ROOT / "Annotations"

PRETRAINED_DIR = _root("PRETRAINED_DIR", REPO_ROOT / "pretrained")

DATASET_YAML = str(DATA_ROOT / "dataset.yaml")
GT_JSON = str(DATA_ROOT / "gt_coco_filtered.json")

YOLO_TRAIN_IMG = str(DATA_ROOT / "train" / "images")
YOLO_TRAIN_LBL = str(DATA_ROOT / "train" / "labels")
YOLO_VAL_IMG = str(DATA_ROOT / "val" / "images")
YOLO_VAL_LBL = str(DATA_ROOT / "val" / "labels")

MMROTATE_TRAIN_IMG = DATA_ROOT / "train" / "images"
MMROTATE_VAL_IMG = DATA_ROOT / "val" / "images"
MMROTATE_TRAIN_LBL = DATA_ROOT / "train" / "labelTxt"
MMROTATE_VAL_LBL = DATA_ROOT / "val" / "labelTxt"

PRETRAINED_YOLO11 = str(
    os.environ.get("PRETRAINED_YOLO11", PRETRAINED_DIR / "yolo11m-obb.pt")
)
PRETRAINED_YOLO8 = str(
    os.environ.get("PRETRAINED_YOLO8", PRETRAINED_DIR / "yolov8m-obb.pt")
)
PRETRAINED_YOLO26 = str(
    os.environ.get("PRETRAINED_YOLO26", PRETRAINED_DIR / "yolo26n-obb.pt")
)

TILE_SIZE = 1024
SEEDS = [42, 123, 456]
SEED_IDS = {42: 1, 123: 2, 456: 3}

MODELS_YOLO = ["yolov8_obb", "yolov11_obb", "yolov26_obb"]
MODELS_MMROTATE = ["orcnn", "s2anet"]
ALL_MODELS = MODELS_MMROTATE + MODELS_YOLO

REPO = REPO_ROOT
OUT = REPO_ROOT
RUNS_DIR = RUNS_ROOT
RESULTS_DIR = RESULTS_ROOT


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p

METRIC_KEYS = [
    "AP",
    "AP50",
    "AP75",
    "Precision",
    "Recall",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
]
