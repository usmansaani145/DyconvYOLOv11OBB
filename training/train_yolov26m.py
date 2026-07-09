#!/usr/bin/env python3
"""
YOLOv26-OBB MEDIUM baseline — corrected retraining.

Identical hyperparameters to original yolov26 run except:
  pretrained = yolo26m-obb.pt  (was yolo26n-obb.pt by mistake)
  output dir = yolov26_obb_medium (separate from original)
"""

from __future__ import annotations

import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

REPO = Path(".")
sys.path.insert(0, str(REPO))

from shared.obb_coco_eval import run_coco_eval
from shared.yolo_label_cache import ensure_yolo_label_caches

SEED = int(sys.argv[1])
MODEL_NAME = "yolov26_obb_medium"
PRETRAINED = "pretrained/yolo26m-obb.pt"
DATA_YAML = "dataset/dataset.yaml"
GT_JSON = "dataset/gt_coco_filtered.json"
OUTPUT_DIR = Path(f"runs/{MODEL_NAME}/{MODEL_NAME}_seed{SEED}")


def _metric_box(results):
    return getattr(results, "obb", None) or results.box


torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

assert torch.cuda.is_available(), "CUDA not available"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Model: {MODEL_NAME} | Seed: {SEED}")
print(f"Pretrained: {PRETRAINED}")

m_check = YOLO(PRETRAINED)
params_check = sum(x.numel() for x in m_check.model.parameters()) / 1e6
print(f"Pretrained params: {params_check:.1f}M")
assert params_check > 20, (
    f"Wrong variant loaded ({params_check:.1f}M) — expected medium (>20M)"
)
print("Medium variant confirmed")
del m_check

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ensure_yolo_label_caches()

model = YOLO(PRETRAINED)
model.train(
    data=DATA_YAML,
    imgsz=1024,
    epochs=100,
    batch=8,
    device=0,
    patience=0,
    project=str(OUTPUT_DIR),
    name="train",
    exist_ok=True,
    seed=SEED,
    deterministic=True,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=0.0,
    copy_paste=0.0,
    erasing=0.0,
    amp=True,
    workers=8,
    conf=0.25,
    iou=0.5,
    save_json=True,
    plots=True,
)
print("Training complete.")

run_dir = OUTPUT_DIR / "train"
best_pt = run_dir / "weights" / "best.pt"
last_pt = run_dir / "weights" / "last.pt"
weights = best_pt if best_pt.is_file() else last_pt
print(f"Loading: {weights}")

eval_model = YOLO(str(weights))

val_results = eval_model.val(
    data=DATA_YAML,
    imgsz=1024,
    batch=8,
    device=0,
    conf=0.25,
    iou=0.5,
    save_json=True,
    plots=True,
    project=str(OUTPUT_DIR),
    name="val",
    exist_ok=True,
    split="val",
)

box = _metric_box(val_results)
metrics = {
    "AP": float(getattr(box, "map", 0.0) or 0.0),
    "AP50": float(getattr(box, "map50", 0.0) or 0.0),
    "AP75": float(getattr(box, "map75", 0.0) or 0.0),
    "Precision": float(getattr(box, "mp", 0.0) or 0.0),
    "Recall": float(getattr(box, "mr", 0.0) or 0.0),
}

pred_jsons = sorted(
    glob.glob(str(OUTPUT_DIR / "val" / "**" / "predictions.json"), recursive=True)
    + glob.glob(str(OUTPUT_DIR / "val" / "*.json"))
)
if pred_jsons:
    size_metrics = run_coco_eval(GT_JSON, pred_jsons[0])
    metrics.update(size_metrics)
    print("COCOeval complete.")
else:
    print("WARNING: no predictions JSON found — COCOeval skipped")

metrics_path = OUTPUT_DIR / f"metrics_{MODEL_NAME}_seed{SEED}.json"
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"Metrics saved: {metrics_path}")

print(f"\n{'=' * 55}")
print(f"  {MODEL_NAME} | seed={SEED}")
print(f"{'=' * 55}")
original_nano_ap = 0.383
for k, v in metrics.items():
    flag = ""
    if k == "AP":
        diff = v - original_nano_ap
        flag = f"  (vs nano 0.383: {diff:+.3f})"
    print(f"  {k:<10}: {v:.4f}{flag}")
print(f"{'=' * 55}")
