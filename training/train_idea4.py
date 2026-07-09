#!/usr/bin/env python3
"""Idea 4 — DynamicConv in YOLOv11-OBB mid-backbone. Seed via argv[1].

Gamma gate fix (GEM lesson): gamma init=0.1, wd=0, lr=10x base.
"""

from __future__ import annotations

import csv
import glob
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.optim as optim
from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "utils"))
sys.path.insert(0, str(REPO_ROOT / "ideas" / "idea4_dyconv"))

from utils.constants import DATASET_YAML, GT_JSON, PRETRAINED_YOLO11, RUNS_ROOT
from utils.idea_train_callbacks import (
    make_grad_clip_callback,
    make_injection_start_callback,
    make_nan_stop_callback,
    rebuild_ema,
    sync_model_device,
)
from models.inject_dyconv import build_idea4_model, inject_dyconv_module
from shared.obb_coco_eval import run_coco_eval

IDEA_NAME = "idea4_dyconv"
DATA_YAML = DATASET_YAML
WEIGHTS = PRETRAINED_YOLO11
BASE_LR = 0.001
GAMMA_LR = 0.01


def set_seed(seed: int) -> None:
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def build_gamma_param_groups(model, base_lr: float = BASE_LR, base_wd: float = 0.0005):
  decay_params: List = []
  no_decay_params: List = []
  gamma_params: List = []
  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    if name.endswith(".gamma") or ".gamma" in name:
      gamma_params.append(param)
    elif "bn" in name.lower() or name.endswith(".bias"):
      no_decay_params.append(param)
    else:
      decay_params.append(param)
  groups = [
      {"params": decay_params, "lr": base_lr, "weight_decay": base_wd, "initial_lr": base_lr},
      {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0, "initial_lr": base_lr},
      {"params": gamma_params, "lr": GAMMA_LR, "weight_decay": 0.0, "initial_lr": GAMMA_LR},
  ]
  return groups


def _apply_gamma_optimizer(trainer) -> None:
  if trainer.optimizer is None:
    return
  base_wd = getattr(trainer.args, "weight_decay", 0.0005)
  groups = build_gamma_param_groups(unwrap_model(trainer.model), BASE_LR, base_wd)
  trainer.optimizer = optim.AdamW(groups, lr=BASE_LR, weight_decay=base_wd)
  gamma_n = len(groups[2]["params"])
  print(f"DyConv gamma optimizer: gamma_params={gamma_n}, gamma_lr={GAMMA_LR}")


def _make_gate_log_callback():
  def log_gate_callback(trainer):
    gates = {}
    for name, param in trainer.model.named_parameters():
      if "gamma" in name.lower():
        gates[name] = float(param.detach().cpu().item())
    if not gates:
      return
    epoch = int(trainer.epoch)
    if epoch % 5 == 0 or epoch >= int(trainer.epochs) - 1:
      print(f"[epoch {epoch}] gamma values: {gates}")
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
  path = output_dir / f"metrics_{IDEA_NAME}_seed{seed}.json"
  with open(path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
  print(f"Metrics saved: {path}")
  return metrics


def _prepare_fresh_train(output_dir: Path) -> None:
  train_dir = output_dir / "train"
  if train_dir.is_dir():
    shutil.rmtree(train_dir)
    print(f"Cleared prior train artifacts: {train_dir}")


def run_training(seed: int, resume: bool = False) -> Dict[str, Any]:
  assert torch.cuda.is_available()
  set_seed(seed)
  output_dir = RUNS_ROOT / IDEA_NAME / f"{IDEA_NAME}_seed{seed}"
  output_dir.mkdir(parents=True, exist_ok=True)
  last_pt = output_dir / "train" / "weights" / "last.pt"
  train_resume = False
  if resume and last_pt.exists():
    print(f"Idea4 DyConv | seed={seed} | resuming from {last_pt}")
    model = YOLO(str(last_pt))
    train_resume = str(last_pt)
  else:
    if resume:
      print(f"WARNING: resume requested but {last_pt} not found — fresh training")
    _prepare_fresh_train(output_dir)
    print(f"Idea4 DyConv | seed={seed}")
    model = build_idea4_model(WEIGHTS, verbose=True)

  def on_train_start(trainer):
    print("\n=== on_train_start: DyConv injection + gamma optimizer ===")
    inject_dyconv_module(trainer.model, verbose=True)
    sync_model_device(trainer.model, trainer.device)
    rebuild_ema(trainer)
    _apply_gamma_optimizer(trainer)
    for name, param in trainer.model.named_parameters():
      if "gamma" in name.lower():
        print(f"  initial {name}: {param.item():.6f}")

  gate_cb = _make_gate_log_callback()
  model.add_callback("on_after_backward", make_grad_clip_callback(max_norm=10.0))
  model.add_callback("on_train_batch_end", make_nan_stop_callback())
  model.add_callback("on_train_start", on_train_start)
  model.add_callback("on_train_epoch_end", gate_cb)

  model.train(
      data=DATA_YAML, imgsz=1024, epochs=100, batch=8, device=0, patience=0,
      project=str(output_dir), name="train", exist_ok=True, seed=seed,
      pretrained=False, optimizer="AdamW", lr0=BASE_LR, lrf=0.01,
      weight_decay=0.0005, warmup_epochs=5, degrees=5.0, mosaic=0.0,
      copy_paste=0.0, amp=False, save_period=10, workers=4,
      conf=0.25, iou=0.5, save_json=True, plots=True,
      resume=train_resume,
  )
  return eval_metrics(seed)


def main() -> int:
  seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
  resume = len(sys.argv) > 2 and sys.argv[2] == "resume"
  run_training(seed, resume=resume)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
