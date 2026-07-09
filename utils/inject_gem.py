"""Inject GEM (GAL adapter) into YOLOv11-OBB classification branch."""

from __future__ import annotations

from typing import Any, List, Optional

import torch.nn as nn


class Cv3GEMWrapper(nn.Module):
  """Wraps cv3 cls branch + GEM adapter (picklable — no forward hooks)."""

  def __init__(self, cv3: nn.Module, gem: nn.Module):
    super().__init__()
    self.cv3 = cv3
    self.gem = gem

  def forward(self, x):
    return self.gem(self.cv3(x))


def _find_detect_head(obb_model: nn.Module):
  for name, module in obb_model.named_modules():
    if type(module).__name__ in ("OBB", "Detect", "OBBHead"):
      return module, name
  return None, None


def _is_gem_wrapped(module: nn.Module) -> bool:
  return isinstance(module, Cv3GEMWrapper)


def inject_gem_module(
    obb_model: nn.Module,
    apply_scales: Optional[List[int]] = None,
    hook_container: Any = None,
    verbose: bool = True,
    gamma_init: float = 0.0,
) -> nn.Module:
  """
  Inject GEM by wrapping cv3 modules (picklable for checkpoint save).

  Replaces forward hooks which break torch.save() at epoch checkpoints.
  """
  from modules.gem_adapted import GEMYOLOAdapter

  if apply_scales is None:
    apply_scales = [0, 1]

  if verbose:
    print("=== YOLOv11-OBB head structure ===")
    for name, module in obb_model.named_modules():
      if "detect" in name.lower() or "obb" in name.lower():
        print(f"  {name}: {type(module).__name__}")

  detect_head, detect_head_name = _find_detect_head(obb_model)
  if detect_head is None:
    if verbose:
      print("All modules:")
      for name, _ in obb_model.named_modules():
        print(f"  {name}")
    raise ValueError("OBB head not found. Check module names above.")

  if not hasattr(detect_head, "cv3"):
    raise ValueError(f"Detection head {detect_head_name} has no cv3 (cls conv list).")

  num_scales = len(detect_head.cv3)
  if verbose:
    print(f"\nFound OBB head: {detect_head_name} ({type(detect_head).__name__})")
    print(f"Detection scales: {num_scales}")
    print(f"GEM apply_scales: {apply_scales}")

  gem_modules = nn.ModuleList()
  wrapped = 0
  for scale_idx in range(num_scales):
    if scale_idx in apply_scales:
      if _is_gem_wrapped(detect_head.cv3[scale_idx]):
        if verbose:
          print(f"   Scale {scale_idx} (P{3 + scale_idx}): already GEM-wrapped")
        gem_modules.append(detect_head.cv3[scale_idx].gem)
        continue

      cls_conv = detect_head.cv3[scale_idx]
      in_ch = None
      for layer in cls_conv.modules():
        if isinstance(layer, nn.Conv2d):
          in_ch = layer.out_channels
      if in_ch is None:
        in_ch = 256

      gem = GEMYOLOAdapter(
          in_channels=in_ch, reduction=8, sync_bn=False, gamma_init=gamma_init
      )
      detect_head.cv3[scale_idx] = Cv3GEMWrapper(cls_conv, gem)
      gem_modules.append(gem)
      wrapped += 1
      if verbose:
        print(f"✅ GEM wrapped cv3[{scale_idx}] (P{3 + scale_idx}), in_channels={in_ch}")
    else:
      gem_modules.append(nn.Identity())
      if verbose:
        print(f"   Scale {scale_idx} (P{3 + scale_idx}): identity (no GEM)")

  if hasattr(detect_head, "gem_modules"):
    del detect_head.gem_modules
  detect_head.add_module("gem_modules", gem_modules)

  if verbose:
    print(f"\nGEM injection complete: {wrapped} cv3 branches wrapped")
  return obb_model


def inject_gem(
    model: Any,
    apply_scales: Optional[List[int]] = None,
    verbose: bool = True,
    gamma_init: float = 0.0,
):
  """Inject GEM via Ultralytics YOLO wrapper."""
  inject_gem_module(
      model.model,
      apply_scales=apply_scales,
      verbose=verbose,
      gamma_init=gamma_init,
  )
  return model
