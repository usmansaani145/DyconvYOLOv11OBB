"""Sync injected WPL/GEM modules to the training device."""

from __future__ import annotations

from typing import Any, Union

import torch
import torch.nn as nn


def _resolve_device(device: Union[str, int, torch.device]) -> torch.device:
  if isinstance(device, torch.device):
    return device
  if isinstance(device, int):
    if device < 0 or not torch.cuda.is_available():
      return torch.device("cpu")
    return torch.device(f"cuda:{device}")
  if isinstance(device, str):
    return torch.device(device)
  return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sync_injected_modules_device(
    obb_model: nn.Module,
    device: Union[str, int, torch.device, None] = None,
) -> torch.device:
  """
  Move WPL/GEM adapters (and full model) to the training device.

  Must use trainer.device — at on_train_start, next(parameters()).device may
  still be CPU before the trainer finishes setup.
  """
  if device is None:
    try:
      device = next(obb_model.parameters()).device
    except StopIteration:
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  dev = _resolve_device(device)

  obb_model.to(dev)

  from modules.gem_adapted import GEMYOLOAdapter
  from modules.wpl_adapted import WPLYOLOAdapter

  for m in obb_model.modules():
    if isinstance(m, (WPLYOLOAdapter, GEMYOLOAdapter)):
      m.to(dev)
    if hasattr(m, "gem_modules") and m.gem_modules is not None:
      m.gem_modules.to(dev)

  from models.inject_gem import Cv3GEMWrapper

  for m in obb_model.modules():
    if isinstance(m, Cv3GEMWrapper):
      m.to(dev)

  print(f"Synced model + WPL/GEM modules to device: {dev}")
  return dev
