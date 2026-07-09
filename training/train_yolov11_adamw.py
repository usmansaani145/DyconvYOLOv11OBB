#!/usr/bin/env python3
"""
YOLOv11-OBB control run with AdamW optimizer.
Purpose: isolate whether DyConv improvement comes from
architecture (DyConv) or optimizer change (SGD -> AdamW).

Identical to DyConv training EXCEPT no DyConv module injected.
IDENTICAL hyperparameters to DyConv-YOLOv11:
  optimizer     = AdamW   (NOT SGD like original baseline)
  lr0           = 0.001   (NOT 0.01 like original baseline)
  warmup_epochs = 5       (NOT 3 like original baseline)
  weight_decay  = 0.0005
  All other settings identical to original YOLOv11 baseline
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# PYDEPS removed — use pip install
REPO = "."

sys.path.insert(0, REPO)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

from ultralytics import YOLO  # noqa: E402

from shared.obb_coco_eval import run_coco_eval  # noqa: E402

SEED = int(sys.argv[1])
MODEL_NAME = "yolov11_obb_adamw"
PRETRAINED = "pretrained/yolo11m-obb.pt"
DATA_YAML = "dataset/dataset.yaml"
GT_JSON = "dataset/gt_coco_filtered.json"
OUTPUT_DIR = f"runs/{MODEL_NAME}/{MODEL_NAME}_seed{SEED}"

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

assert torch.cuda.is_available(), "CUDA not available"
print(f"GPU:  {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Seed: {SEED} | Model: {MODEL_NAME}")
print("Optimizer: AdamW lr0=0.001 (control for DyConv experiment)")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
model = YOLO(PRETRAINED)

model.train(
    data=DATA_YAML,
    imgsz=1024,
    epochs=100,
    batch=8,
    device=0,
    patience=0,
    project=OUTPUT_DIR,
    name="train",
    exist_ok=True,
    seed=SEED,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
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
    amp=True,
    save_period=10,
    workers=4,
    conf=0.25,
    iou=0.5,
    save_json=True,
    plots=True,
)
print("Training complete.")

run_dir = Path(OUTPUT_DIR) / "train"
best_pt = run_dir / "weights" / "best.pt"
last_pt = run_dir / "weights" / "last.pt"
weights = best_pt if best_pt.exists() else last_pt
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
    project=str(run_dir.parent),
    name="val",
    exist_ok=True,
    split="val",
)

metrics = {
    "AP": float(val_results.box.map),
    "AP50": float(val_results.box.map50),
    "AP75": float(val_results.box.map75),
    "Precision": float(val_results.box.mp),
    "Recall": float(val_results.box.mr),
}

pred_jsons = (
    glob.glob(str(run_dir.parent / "val" / "*.json"))
    + glob.glob(str(run_dir.parent / "val" / "**" / "*.json"), recursive=True)
)
if pred_jsons:
    size_metrics = run_coco_eval(GT_JSON, pred_jsons[0])
    metrics.update(size_metrics)
    print("COCOeval complete.")
else:
    print("WARNING: predictions JSON not found — COCOeval skipped")

metrics_path = Path(OUTPUT_DIR) / f"metrics_{MODEL_NAME}_seed{SEED}.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved: {metrics_path}")

ap = metrics.get("AP", 0.0)
print(f"\n{'=' * 65}")
print(f"  OPTIMIZER CONTROL EXPERIMENT RESULT | seed={SEED}")
print(f"{'=' * 65}")
print("  YOLOv11-OBB (SGD  baseline):  AP = 0.401 ± 0.004  [existing]")
print(f"  YOLOv11-OBB (AdamW control):  AP = {ap:.4f}         [THIS RUN]")
print("  DyConv-YOLOv11 (AdamW):       AP = 0.408 ± 0.001  [existing]")
print(f"{'=' * 65}")
adamw_gain = ap - 0.401
dyconv_gain = 0.408 - ap
print(f"  AdamW optimizer gain:          {adamw_gain:+.4f} vs SGD baseline")
print(f"  DyConv architectural gain:     {dyconv_gain:+.4f} vs AdamW control")
print()

if dyconv_gain > 0.003:
    print("  CONCLUSION: DyConv improvement is genuine architectural gain")
    print("  Paper claim: STRONG — optimizer change contributes minimally")
elif dyconv_gain > 0.001:
    print("  CONCLUSION: DyConv adds marginal gain beyond optimizer effect")
    print("  Paper claim: MODERATE — state both contributions separately")
else:
    print("  CONCLUSION: DyConv gain may be primarily from optimizer change")
    print("  Paper claim: WEAKEN — attribute improvement to AdamW + DyConv")
print(f"{'=' * 65}")
