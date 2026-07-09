#!/usr/bin/env python3
"""Combo 2 — DyConv + EMA. Seed via argv[1]."""

from __future__ import annotations

import glob
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from ultralytics import YOLO

COMBO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = COMBO_ROOT.parents[1]
sys.path.insert(0, str(COMBO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "utils"))
from utils.constants import DATASET_YAML, GT_JSON, RUNS_ROOT  # noqa: E402

from combo_train_utils import (  # noqa: E402
  apply_gamma_optimizer,
  check_activation_scale,
  make_gate_log_callback,
  make_grad_clip_callback,
  make_nan_guard_callback,
  rebuild_ema,
  sync_model_device,
)
from models.build_combo2 import apply_combo2_injections, build_combo2  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from shared.obb_coco_eval import run_coco_eval  # noqa: E402

COMBO_NAME = "combo2_dyconv_ema"
DATA_YAML = DATASET_YAML
BASE_LR = 0.001


def set_seed(seed: int) -> None:
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def eval_metrics(seed: int) -> Dict[str, Any]:
  output_dir = RUNS_ROOT / COMBO_NAME / f"{COMBO_NAME}_seed{seed}"
  run_dir = output_dir / "train"
  best_pt = run_dir / "weights" / "best.pt"
  if not best_pt.exists():
    best_pt = run_dir / "weights" / "last.pt"
  model_eval = YOLO(str(best_pt))
  val_results = model_eval.val(
      data=DATA_YAML, imgsz=1024, batch=8, device=0, conf=0.25, iou=0.5,
      save_json=True, plots=True, project=str(run_dir.parent), name="val",
      exist_ok=True, split="val",
  )
  metrics = {
      "AP": float(val_results.box.map),
      "AP50": float(val_results.box.map50),
      "AP75": float(val_results.box.map75),
      "Precision": float(val_results.box.mp),
      "Recall": float(val_results.box.mr),
  }
  pred_jsons = glob.glob(str(run_dir.parent / "val" / "*.json"))
  if pred_jsons:
    metrics.update(run_coco_eval(GT_JSON, pred_jsons[0]))
  path = output_dir / f"metrics_{COMBO_NAME}_seed{seed}.json"
  with open(path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
  print(f"Metrics saved: {path}")
  return metrics


def run_training(seed: int) -> Dict[str, Any]:
  assert torch.cuda.is_available()
  set_seed(seed)
  output_dir = RUNS_ROOT / COMBO_NAME / f"{COMBO_NAME}_seed{seed}"
  output_dir.mkdir(parents=True, exist_ok=True)
  print(f"Combo2 DyConv+EMA | seed={seed} | GPU={torch.cuda.get_device_name(0)}")

  model = build_combo2(verbose=True)
  if not check_activation_scale(model, "Combo2"):
    raise RuntimeError("Activation scale check failed — fix before training")

  def on_train_start(trainer):
    print("\n=== on_train_start: Combo2 injection + gamma optimizer ===")
    apply_combo2_injections(trainer.model, verbose=True)
    sync_model_device(trainer.model, trainer.device)
    rebuild_ema(trainer)
    apply_gamma_optimizer(trainer)

  model.add_callback("on_after_backward", make_grad_clip_callback(max_norm=10.0))
  model.add_callback("on_train_batch_end", make_nan_guard_callback())
  model.add_callback("on_train_start", on_train_start)
  model.add_callback("on_train_epoch_end", make_gate_log_callback())

  model.train(
      data=DATA_YAML, imgsz=1024, epochs=100, batch=8, device=0, patience=0,
      project=str(output_dir), name="train", exist_ok=True, seed=seed,
      pretrained=False, optimizer="AdamW", lr0=BASE_LR, lrf=0.01,
      weight_decay=0.0005, warmup_epochs=5, degrees=5.0, mosaic=0.0,
      copy_paste=0.0, amp=False, save_period=10, workers=4,
      conf=0.25, iou=0.5, save_json=True, plots=True,
  )
  return eval_metrics(seed)


def main() -> int:
  seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
  run_training(seed)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
