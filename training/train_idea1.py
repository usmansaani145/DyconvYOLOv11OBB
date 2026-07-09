#!/usr/bin/env python3
"""Idea 1 — LSKA in YOLOv11-OBB backbone. Seed via argv[1]."""

from __future__ import annotations

import csv
import glob
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "utils"))
from utils.constants import DATASET_YAML, GT_JSON, PRETRAINED_YOLO11, RUNS_ROOT

import numpy as np
import torch
from ultralytics import YOLO

IDEA_ROOT = REPO_ROOT / "ideas" / "idea1_lska"
sys.path.insert(0, str(IDEA_ROOT))


from models.inject_lska import build_idea1_model, inject_lska_module  # noqa: E402
from utils.idea_train_callbacks import (  # noqa: E402
    make_injection_start_callback,
    make_nan_stop_callback,
)

sys.path.insert(0, str(IDEA_ROOT))
from shared.obb_coco_eval import run_coco_eval  # noqa: E402

IDEA_NAME = "idea1_lska"
DATA_YAML = DATASET_YAML
GT_JSON = GT_JSON
WEIGHTS = PRETRAINED_YOLO11
RUNS_ROOT = RUNS_ROOT


def set_seed(seed: int) -> None:
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def _gate_param_names(model) -> List[str]:
  names = []
  for name, param in model.named_parameters():
    if "gamma" in name.lower():
      names.append(name)
  return names


def _make_gate_log_callback():
  def log_gate_callback(trainer):
    gates = {}
    for name, param in trainer.model.named_parameters():
      if "gamma" in name.lower():
        gates[name] = float(param.detach().cpu().item())
    if not gates:
      return
    epoch = int(trainer.epoch)
    total = int(trainer.epochs)
    if epoch % 5 == 0 or epoch >= total - 1:
      print(f"[epoch {epoch}] gate values: {gates}")
    save_dir = Path(getattr(trainer, "save_dir", "."))
    log_path = save_dir / "gate_log.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
      w = csv.writer(f)
      if write_header:
        w.writerow(["epoch"] + list(gates.keys()))
      w.writerow([epoch] + list(gates.values()))
  return log_gate_callback


def eval_metrics(seed: int) -> Dict[str, Any]:
  output_dir = RUNS_ROOT / IDEA_NAME / f"{IDEA_NAME}_seed{seed}"
  run_dir = output_dir / "train"
  best_pt = run_dir / "weights" / "best.pt"
  if not best_pt.exists():
    best_pt = run_dir / "weights" / "last.pt"
  if not best_pt.exists():
    raise FileNotFoundError(f"No checkpoint for {IDEA_NAME} seed {seed}")

  model_eval = YOLO(str(best_pt))
  val_results = model_eval.val(
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
  metrics: Dict[str, float] = {
      "AP": float(val_results.box.map),
      "AP50": float(val_results.box.map50),
      "AP75": float(val_results.box.map75),
      "Precision": float(val_results.box.mp),
      "Recall": float(val_results.box.mr),
  }
  pred_jsons = glob.glob(str(run_dir.parent / "val" / "*.json"))
  if pred_jsons:
    metrics.update(run_coco_eval(GT_JSON, pred_jsons[0]))

  metrics_path = output_dir / f"metrics_{IDEA_NAME}_seed{seed}.json"
  with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
  print(f"Metrics saved: {metrics_path}")
  for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
  return metrics


def run_training(seed: int) -> Dict[str, Any]:
  assert torch.cuda.is_available(), "CUDA required"
  set_seed(seed)
  output_dir = RUNS_ROOT / IDEA_NAME / f"{IDEA_NAME}_seed{seed}"
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"Idea1 LSKA | seed={seed} | GPU={torch.cuda.get_device_name(0)}")
  model = build_idea1_model(WEIGHTS, verbose=True)

  gate_cb = _make_gate_log_callback()

  def on_train_epoch_end(trainer):
    gate_cb(trainer)

  model.add_callback("on_train_batch_end", make_nan_stop_callback())
  model.add_callback(
      "on_train_start",
      make_injection_start_callback(inject_lska_module),
  )
  model.add_callback("on_train_epoch_end", on_train_epoch_end)

  model.train(
      data=DATA_YAML,
      imgsz=1024,
      epochs=100,
      batch=8,
      device=0,
      patience=0,
      project=str(output_dir),
      name="train",
      exist_ok=True,
      seed=seed,
      pretrained=False,
      optimizer="AdamW",
      lr0=0.001,
      lrf=0.01,
      weight_decay=0.0005,
      warmup_epochs=5,
      degrees=5.0,
      mosaic=0.0,
      copy_paste=0.0,
      amp=False,
      save_period=10,
      workers=4,
      conf=0.25,
      iou=0.5,
      save_json=True,
      plots=True,
  )
  return eval_metrics(seed)


def main() -> int:
  seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
  mode = sys.argv[2] if len(sys.argv) > 2 else "train"
  if mode == "eval":
    eval_metrics(seed)
  else:
    run_training(seed)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
