"""Shared training callbacks for ideas/ PBS jobs (infra only — not idea code)."""

from __future__ import annotations

import torch
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model


def sync_model_device(model: torch.nn.Module, device: torch.device) -> None:
  """Move full model including custom injected submodules to trainer device."""
  model.to(device)
  for module in unwrap_model(model).modules():
    for buf in module.buffers():
      buf.data = buf.data.to(device)
    for param in module.parameters(recurse=False):
      param.data = param.data.to(device)


def rebuild_ema(trainer) -> None:
  """Rebuild EMA after structural injection so state_dict keys match."""
  if getattr(trainer, "ema", None) is None:
    return
  trainer.ema = ModelEMA(trainer.model)
  print("Rebuilt EMA after custom module injection")


def make_nan_guard_callback():
  """Log non-finite train loss; do not stop (Ultralytics has its own recovery)."""

  def on_train_batch_end(trainer):
    loss = getattr(trainer, "loss", None)
    if loss is None or not torch.is_tensor(loss):
      return
    if not torch.isfinite(loss).all():
      batch = getattr(trainer, "batch_i", getattr(trainer, "ni", "?"))
      epoch = getattr(trainer, "epoch", "?")
      print(
          f"WARNING: NaN/Inf train loss at epoch {epoch} batch {batch} "
          "(continuing — custom modules should prevent val NaN)"
      )

  return on_train_batch_end


def make_nan_stop_callback():
  """Backward-compatible alias — logging only, no early stop."""
  return make_nan_guard_callback()


def make_grad_clip_callback(max_norm: float = 10.0):
  """Clip gradients after backward to limit weight explosion in custom modules."""

  def on_after_backward(trainer):
    model = unwrap_model(trainer.model)
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
      return
    torch.nn.utils.clip_grad_norm_(params, max_norm)

  return on_after_backward


def make_injection_start_callback(inject_fn, *, rebuild_ema_after: bool = True):
  """Call inject_fn(trainer.model) at train start, sync device, rebuild EMA."""

  def on_train_start(trainer):
    print("\n=== on_train_start: applying custom module injection ===")
    inject_fn(trainer.model, verbose=True)
    sync_model_device(trainer.model, trainer.device)
    if rebuild_ema_after:
      rebuild_ema(trainer)

  return on_train_start
