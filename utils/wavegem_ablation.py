#!/usr/bin/env python3
"""Shared training + evaluation logic for WaveGEM-OBB ablation studies."""

from __future__ import annotations

import glob
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from ultralytics import YOLO

REPO = Path(".")
sys.path.insert(0, str(REPO))

from models.build_wavegem import build_wavegem_obb  # noqa: E402
from models.inject_gem import inject_gem_module  # noqa: E402
from models.inject_wpl import inject_wpl_module  # noqa: E402
from models.sync_injected import sync_injected_modules_device  # noqa: E402
from shared.obb_coco_eval import run_coco_eval  # noqa: E402

DATA_YAML = "dataset/dataset.yaml"
GT_JSON = "dataset/gt_coco_filtered.json"
WEIGHTS = "pretrained/yolo11m-obb.pt"
P2_YAML = str(REPO / "configs" / "yolov11_p2.yaml")


def set_seed(seed: int) -> None:
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def _gem_scales_for(p2_yaml: Optional[str]) -> list:
  return [1, 2] if p2_yaml else [0, 1]


def eval_ablation_metrics(
    seed: int,
    ablation: str,
    use_wpl: bool,
    use_gem: bool,
    p2_yaml: Optional[str] = None,
    weights_path: Optional[str] = None,
) -> Dict[str, Any]:
  """Run validation + COCO eval and write metrics JSON for a trained ablation."""
  output_dir = Path(f"runs/{ablation}/{ablation}_seed{seed}")
  run_dir = output_dir / "train"
  best_pt = run_dir / "weights" / "best.pt"
  last_pt = run_dir / "weights" / "last.pt"
  eval_weights = Path(weights_path) if weights_path else (best_pt if best_pt.exists() else last_pt)

  if not eval_weights.exists():
    raise FileNotFoundError(f"No checkpoint found for {ablation} seed {seed}: {eval_weights}")

  print(f"Evaluating {ablation} seed {seed} from {eval_weights}")
  gem_scales = _gem_scales_for(p2_yaml)

  model_eval = build_wavegem_obb(
      str(eval_weights),
      use_wpl=use_wpl,
      use_gem=use_gem,
      p2_yaml=p2_yaml,
      gem_scales=gem_scales,
  )

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
    size_metrics = run_coco_eval(GT_JSON, pred_jsons[0])
    metrics.update(size_metrics)

  metrics_path = output_dir / f"metrics_{ablation}_seed{seed}.json"
  with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
  print(f"\nMetrics saved: {metrics_path}")

  print(f"\n{'=' * 55}")
  print(f"  {ablation} | seed={seed}")
  print(f"{'=' * 55}")
  for k, v in metrics.items():
    flag = ""
    if k == "AP":
      flag = " ✅" if v > 0.401 else " (baseline=0.401)"
    if k == "APs":
      flag = " ✅" if v > 0.354 else " (baseline=0.354)"
    print(f"  {k:<10}: {v:.4f}{flag}")

  return metrics


def _extract_resume_model(ckpt: dict) -> torch.nn.Module:
  """Pull the trained nn.Module from an Ultralytics checkpoint."""
  for key in ("model", "ema"):
    obj = ckpt.get(key)
    if obj is None:
      continue
    if hasattr(obj, "ema"):
      obj = obj.ema
    if isinstance(obj, torch.nn.Module):
      return obj
  raise RuntimeError(f"No nn.Module found in checkpoint (keys: {list(ckpt.keys())})")


def _patch_trainer_setup_model_for_custom_resume() -> None:
  """Ensure setup_model uses injected WPL/GEM weights (Ultralytics resets model after callbacks)."""
  from ultralytics.engine.trainer import BaseTrainer

  if getattr(BaseTrainer.setup_model, "_wavegem_patched", False):
    return

  _original = BaseTrainer.setup_model

  def setup_model(self):
    resume_model = getattr(self, "_wavegem_resume_model", None)
    resume_ckpt = getattr(self, "_wavegem_resume_ckpt", None)
    if resume_model is not None:
      self.model = resume_model
      self._wavegem_resume_model = None
      self._wavegem_resume_ckpt = None
      return resume_ckpt
    return _original(self)

  setup_model._wavegem_patched = True
  BaseTrainer.setup_model = setup_model


def _patch_load_checkpoint_state_for_custom_resume() -> None:
  """Skip optimizer/scaler restore when WPL/GEM param groups differ from checkpoint."""
  from ultralytics.engine.trainer import BaseTrainer

  if getattr(BaseTrainer._load_checkpoint_state, "_wavegem_patched", False):
    return

  _original = BaseTrainer._load_checkpoint_state

  def _load_checkpoint_state(self, ckpt):
    if not getattr(self, "_wavegem_custom_resume", False):
      return _original(self, ckpt)
    if ckpt.get("optimizer") is not None:
      try:
        self.optimizer.load_state_dict(ckpt["optimizer"])
      except ValueError as exc:
        print(f"WARNING: skipping optimizer restore on custom resume: {exc}")
    if ckpt.get("scaler") is not None:
      try:
        self.scaler.load_state_dict(ckpt["scaler"])
      except (ValueError, KeyError) as exc:
        print(f"WARNING: skipping scaler restore on custom resume: {exc}")
    if self.ema and ckpt.get("ema"):
      from ultralytics.utils.torch_utils import ModelEMA

      self.ema = ModelEMA(self.model)
      self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
      self.ema.updates = ckpt["updates"]
      _sync_trainer_criterion_devices(self)
    self.best_fitness = ckpt.get("best_fitness", 0.0)

  _load_checkpoint_state._wavegem_patched = True
  BaseTrainer._load_checkpoint_state = _load_checkpoint_state


def _ensure_loss_criterion_device(model: torch.nn.Module, device: torch.device) -> None:
  """Move OBB loss helpers (e.g. proj) that are plain tensors, not nn.Module buffers."""
  from ultralytics.utils.torch_utils import unwrap_model

  inner = unwrap_model(model)
  criterion = getattr(inner, "criterion", None)
  if criterion is None:
    return
  proj = getattr(criterion, "proj", None)
  if isinstance(proj, torch.Tensor):
    criterion.proj = proj.to(device)


def _sync_trainer_criterion_devices(trainer) -> None:
  """Sync v8OBBLoss.proj on train + EMA models (proj is absent from checkpoint state_dict)."""
  device = trainer.device
  _ensure_loss_criterion_device(trainer.model, device)
  ema = getattr(trainer, "ema", None)
  if ema is not None and getattr(ema, "ema", None) is not None:
    _ensure_loss_criterion_device(ema.ema, device)


def _apply_wpl_param_lr(trainer, base_lr: float, wpl_lr_mult: float) -> None:
  """Rebuild optimizer with higher LR for WPL adapter parameters."""
  if wpl_lr_mult <= 1.0:
    return
  from modules.wpl_adapted import WPLYOLOAdapter
  from ultralytics.utils.torch_utils import unwrap_model

  wpl_lr = base_lr * wpl_lr_mult
  wpl_params = []
  other_params = []
  for module_name, module in unwrap_model(trainer.model).named_modules():
    if isinstance(module, WPLYOLOAdapter):
      wpl_params.extend(p for p in module.parameters() if p.requires_grad)

  wpl_param_ids = {id(p) for p in wpl_params}
  for p in trainer.model.parameters():
    if not p.requires_grad:
      continue
    if id(p) not in wpl_param_ids:
      other_params.append(p)

  if not wpl_params:
    print("WARNING: wpl_lr_mult>1 but no WPL parameters found — keeping default optimizer")
    return

  opt_name = type(trainer.optimizer).__name__ if trainer.optimizer else "AdamW"
  wd = getattr(trainer.args, "weight_decay", 0.0005)
  momentum = getattr(trainer.args, "momentum", 0.937)

  if opt_name == "AdamW":
    import torch.optim as optim

    groups = [
        {"params": other_params, "lr": base_lr, "initial_lr": base_lr},
        {"params": wpl_params, "lr": wpl_lr, "initial_lr": wpl_lr},
    ]
    trainer.optimizer = optim.AdamW(
        groups,
        lr=base_lr,
        weight_decay=wd,
    )
  else:
    import torch.optim as optim

    groups = [
        {"params": other_params, "lr": base_lr, "initial_lr": base_lr},
        {"params": wpl_params, "lr": wpl_lr, "initial_lr": wpl_lr},
    ]
    trainer.optimizer = optim.SGD(
        groups,
        lr=base_lr,
        momentum=momentum,
        weight_decay=wd,
        nesterov=True,
    )

  print(
      f"WPL per-param LR: base={base_lr:.4f}, WPL={wpl_lr:.4f} "
      f"({wpl_lr_mult:.1f}x) | WPL params={len(wpl_params)}, other={len(other_params)}"
  )


def _is_gem_gamma_param(name: str) -> bool:
  return "gamma" in name.lower() and "gem" in name.lower()


def build_gem_param_groups(model, base_lr: float = 0.001, base_wd: float = 0.0005):
  """Build AdamW param groups: decay / no-decay / GEM-gamma (wd=0, high LR)."""
  decay_params = []
  no_decay_params = []
  gamma_params = []

  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    if _is_gem_gamma_param(name):
      gamma_params.append(param)
    elif "bn" in name.lower() or name.endswith(".bias"):
      no_decay_params.append(param)
    else:
      decay_params.append(param)

  groups = [
      {"params": decay_params, "lr": base_lr, "weight_decay": base_wd},
      {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0},
      {"params": gamma_params, "lr": 0.01, "weight_decay": 0.0},
  ]
  for group in groups:
    group["initial_lr"] = group["lr"]
  return groups


def _print_optimizer_param_groups(optimizer) -> None:
  print("\n=== Optimizer param groups ===")
  for i, group in enumerate(optimizer.param_groups):
    n = len(group["params"])
    wd = group.get("weight_decay", "default")
    lr = group.get("lr", "default")
    print(f"Param group {i}: {n} params, wd={wd}, lr={lr}")
  print("=== end param groups ===\n")


def _apply_gem_gamma_optimizer(trainer, base_lr: float = 0.001, gamma_lr: float = 0.01) -> None:
  """Rebuild optimizer with GEM gamma excluded from weight decay and higher LR."""
  if trainer.optimizer is None:
    print("WARNING: gem_gamma_fix requested but optimizer is None")
    return

  import torch.optim as optim
  from ultralytics.utils.torch_utils import unwrap_model

  base_wd = getattr(trainer.args, "weight_decay", 0.0005)
  groups = build_gem_param_groups(unwrap_model(trainer.model), base_lr=base_lr, base_wd=base_wd)
  # Ensure gamma group uses requested LR (build_gem_param_groups hardcodes 0.01)
  if len(groups) >= 3 and gamma_lr != 0.01:
    groups[2]["lr"] = gamma_lr

  opt_name = type(trainer.optimizer).__name__
  if opt_name == "AdamW":
    trainer.optimizer = optim.AdamW(groups, lr=base_lr, weight_decay=base_wd)
  else:
    momentum = getattr(trainer.args, "momentum", 0.937)
    trainer.optimizer = optim.SGD(
        groups, lr=base_lr, momentum=momentum, weight_decay=base_wd, nesterov=True
    )

  gamma_n = len(groups[2]["params"]) if len(groups) > 2 else 0
  print(
      f"GEM gamma optimizer: base_lr={base_lr:.4f}, gamma_lr={gamma_lr:.4f}, "
      f"gamma_params={gamma_n}"
  )
  _print_optimizer_param_groups(trainer.optimizer)


def _make_log_gamma_callback():
  """Return on_train_epoch_end callback that logs GEM gamma trajectory to CSV."""

  def log_gamma_callback(trainer):
    gammas = {}
    for name, param in trainer.model.named_parameters():
      if _is_gem_gamma_param(name):
        gammas[name] = float(param.detach().cpu().item())

    if not gammas:
      return

    epoch = int(trainer.epoch)
    total = int(trainer.epochs)
    if epoch % 5 == 0 or epoch >= total - 1:
      print(f"[epoch {epoch}] gamma values: {gammas}")

    import csv

    save_dir = Path(getattr(trainer, "save_dir", "."))
    log_path = save_dir / "gamma_log.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
      w = csv.writer(f)
      if write_header:
        w.writerow(["epoch"] + list(gammas.keys()))
      w.writerow([epoch] + list(gammas.values()))

  return log_gamma_callback


def run_ablation(
    seed: int,
    ablation: str,
    use_wpl: bool,
    use_gem: bool,
    p2_yaml: Optional[str] = None,
    resume: bool = False,
    wpl_max_replace: int = 2,
    wpl_lr_mult: float = 1.0,
    gem_gamma_fix: bool = False,
    gem_gamma_init: float = 0.1,
    gem_gamma_lr: float = 0.01,
) -> Dict[str, Any]:
  assert torch.cuda.is_available(), "CUDA not available"
  print(f"GPU: {torch.cuda.get_device_name(0)}")
  print(
      f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
  )
  print(f"Ablation: {ablation} | Seed: {seed}")
  if use_wpl:
    print(f"WPL: max_replace={wpl_max_replace}, lr_mult={wpl_lr_mult}")
  if gem_gamma_fix:
    print(f"GEM gamma-fix: init={gem_gamma_init}, gamma_lr={gem_gamma_lr}, log=on")

  set_seed(seed)

  output_dir = Path(f"runs/{ablation}/{ablation}_seed{seed}")
  output_dir.mkdir(parents=True, exist_ok=True)

  gem_scales = _gem_scales_for(p2_yaml)

  last_pt = output_dir / "train" / "weights" / "last.pt"
  train_resume: bool | str = False
  custom_resume = False
  if resume and last_pt.exists():
    print(f"Resuming training from {last_pt}")
    train_resume = str(last_pt)
    if use_wpl or use_gem:
      # WPL/GEM changes layer names; load saved nn.Module before setup_model().
      custom_resume = True
      model = build_wavegem_obb(
          weights=WEIGHTS,
          use_wpl=use_wpl,
          use_gem=use_gem,
          p2_yaml=p2_yaml,
          gem_scales=gem_scales,
          wpl_max_replace=wpl_max_replace,
          gem_gamma_init=gem_gamma_init if gem_gamma_fix else 0.0,
      )
    else:
      model = YOLO(str(last_pt))
  else:
    if resume:
      print(f"WARNING: resume requested but {last_pt} not found — starting fresh")
    model = build_wavegem_obb(
        weights=WEIGHTS,
        use_wpl=use_wpl,
        use_gem=use_gem,
        p2_yaml=p2_yaml,
        gem_scales=gem_scales,
        wpl_max_replace=wpl_max_replace,
        gem_gamma_init=gem_gamma_init if gem_gamma_fix else 0.0,
    )

  def on_pretrain_routine_start(trainer):
    """Stash WPL/GEM checkpoint; setup_model patch applies it after Ultralytics init."""
    if not custom_resume:
      return
    ckpt = torch.load(str(last_pt), map_location="cpu", weights_only=False)
    trainer._wavegem_resume_model = _extract_resume_model(ckpt).float()
    trainer._wavegem_resume_ckpt = ckpt
    trainer._wavegem_custom_resume = True
    trainer.start_epoch = ckpt.get("epoch", -1) + 1
    print(
        f"Custom WPL/GEM resume: staged checkpoint from epoch "
        f"{ckpt.get('epoch', '?')} → continuing at epoch {trainer.start_epoch}"
    )

  def on_train_batch_end(trainer):
    if torch.isnan(trainer.loss) or torch.isinf(trainer.loss):
      print(
          f"NaN/Inf loss detected — epoch {trainer.epoch} "
          f"batch {trainer.batch_i} — stopping training"
      )
      trainer.stop = True

  def on_train_start(trainer):
    """Re-inject custom modules after Ultralytics builds the trainer model."""
    if not hasattr(trainer, "model") or trainer.model is None:
      print("WARNING: trainer.model not ready at on_train_start — skipping injection")
      return
    if custom_resume:
      sync_injected_modules_device(trainer.model, device=trainer.device)
      _sync_trainer_criterion_devices(trainer)
      print("Custom resume: using WPL/GEM model loaded from checkpoint")
    else:
      print("\n=== on_train_start: ensuring WPL/GEM on trainer.model ===")
      if use_wpl:
        inject_wpl_module(trainer.model, verbose=True, max_replace=wpl_max_replace)
      if use_gem:
        inject_gem_module(
            trainer.model,
            apply_scales=gem_scales,
            hook_container=trainer,
            verbose=True,
            gamma_init=gem_gamma_init if gem_gamma_fix else 0.0,
        )
      sync_injected_modules_device(trainer.model, device=trainer.device)

      # EMA is built before injection; rebuild so keys match WPL/GEM modules.
      if use_wpl or use_gem:
        try:
          from ultralytics.utils.torch_utils import ModelEMA

          if getattr(trainer, "ema", None) is not None:
            trainer.ema = ModelEMA(trainer.model)
            print("Rebuilt EMA after WPL/GEM injection")
        except Exception as exc:
          print(f"WARNING: could not rebuild EMA: {exc}")
        _sync_trainer_criterion_devices(trainer)

    if use_wpl and wpl_lr_mult > 1.0 and trainer.optimizer is not None:
      _apply_wpl_param_lr(trainer, base_lr=0.001, wpl_lr_mult=wpl_lr_mult)

    if gem_gamma_fix and use_gem and trainer.optimizer is not None:
      _apply_gem_gamma_optimizer(trainer, base_lr=0.001, gamma_lr=gem_gamma_lr)
      # Print initial gamma after optimizer rebuild
      for name, param in trainer.model.named_parameters():
        if _is_gem_gamma_param(name):
          print(f"  initial {name}: {param.item():.6f}")

  def on_train_epoch_end(trainer):
    """Validation uses EMA; ensure loss proj tensors are on GPU before each val pass."""
    if use_wpl or use_gem:
      _sync_trainer_criterion_devices(trainer)

  log_gamma_cb = _make_log_gamma_callback() if gem_gamma_fix else None

  def on_train_epoch_end_combined(trainer):
    on_train_epoch_end(trainer)
    if log_gamma_cb is not None:
      log_gamma_cb(trainer)

  model.add_callback("on_pretrain_routine_start", on_pretrain_routine_start)
  model.add_callback("on_train_batch_end", on_train_batch_end)
  model.add_callback("on_train_start", on_train_start)
  model.add_callback("on_train_epoch_end", on_train_epoch_end_combined)

  # WPL/GEM modules are injected after AMP setup and must stay fp32 (DWT + GAL).
  use_amp = not (use_wpl or use_gem)

  train_batch = 4 if (use_wpl or use_gem) else 8

  if custom_resume:
    _patch_trainer_setup_model_for_custom_resume()
    _patch_load_checkpoint_state_for_custom_resume()

  model.train(
      data=DATA_YAML,
      imgsz=1024,
      epochs=100,
      batch=train_batch,
      device=0,
      patience=0,
      project=str(output_dir),
      name="train",
      exist_ok=True,
      resume=train_resume,
      seed=seed,
      pretrained=False,
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
      amp=use_amp,
      save_period=10,
      workers=4,
      conf=0.25,
      iou=0.5,
      save_json=True,
      plots=True,
  )
  print("Training complete")

  return eval_ablation_metrics(
      seed=seed,
      ablation=ablation,
      use_wpl=use_wpl,
      use_gem=use_gem,
      p2_yaml=p2_yaml,
  )
