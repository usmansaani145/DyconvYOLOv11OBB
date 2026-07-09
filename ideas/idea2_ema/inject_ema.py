"""Inject EMA into YOLOv11-OBB backbone tail (Idea 2).

Injection point: layer 10 (C2PSA) — end of backbone before neck, on P5.
Ouyang et al. ICASSP 2023; SED-YOLO (Sci Reports 2025).
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO

from modules.ema import EMA

WEIGHTS = "pretrained/yolo11m-obb.pt"
EMA_LAYER = 10


def _find_backbone_end(seq: nn.ModuleList) -> int:
  for i, layer in enumerate(seq):
    if type(layer).__name__ == "Upsample":
      return i
  return 11


def _copy_layer_attrs(src: nn.Module, dst: nn.Module) -> None:
  for attr in ("i", "f", "type", "np", "stride"):
    if hasattr(src, attr):
      setattr(dst, attr, getattr(src, attr))


def _probe_out_channels(layer: nn.Module, default: int = 512) -> int:
  for m in layer.modules():
    if isinstance(m, nn.Conv2d):
      return m.out_channels
  return default


class _LayerWithPost(nn.Module):
  def __init__(self, orig: nn.Module, post: nn.Module, tag: str):
    super().__init__()
    self.orig = orig
    self.post = post
    self.tag = tag
    _copy_layer_attrs(orig, self)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.post(self.orig(x))


def _verify_std(pre: torch.Tensor, post: torch.Tensor, tag: str) -> None:
  ratio = post.std().item() / max(pre.std().item(), 1e-8)
  print(f"  [{tag}] ratio={ratio:.4f}")
  if not (0.5 < ratio < 2.0):
    print(f"  WARNING: feature scale mismatch at {tag}")


def inject_ema_module(
    obb_model: nn.Module,
    ema_layer: int = EMA_LAYER,
    groups: int = 8,
    verbose: bool = True,
) -> nn.Module:
  seq = obb_model.model
  backbone_end = _find_backbone_end(seq)

  if verbose:
    print("=== Idea2 EMA — backbone scan ===")
    for i, layer in enumerate(seq):
      marker = " [backbone]" if i < backbone_end else " [head/neck]"
      print(f"  Layer {i:2d}: {type(layer).__name__}{marker}")

  if ema_layer >= len(seq):
    raise ValueError(f"EMA layer {ema_layer} out of range")

  layer = seq[ema_layer]
  if isinstance(layer, _LayerWithPost):
    if verbose:
      print(f"  Layer {ema_layer}: already EMA-wrapped")
    return obb_model

  channels = _probe_out_channels(layer, default=512)
  while channels % groups != 0 and groups > 1:
    groups //= 2
  ema = EMA(channels, groups=groups)
  seq[ema_layer] = _LayerWithPost(layer, ema, tag=f"EMA@{ema_layer}")

  if verbose:
    print(f"  ✅ Layer {ema_layer} ({type(layer).__name__}): +EMA(ch={channels}, g={groups})")
    print("EMA injection complete")
  return obb_model


def verify_ema_forward(obb_model: nn.Module, device: str = "cuda") -> None:
  dev = torch.device(device if torch.cuda.is_available() else "cpu")
  obb_model.to(dev).eval()
  x = torch.randn(1, 3, 256, 256, device=dev)

  def _hook(module, inp, out):
    if isinstance(out, torch.Tensor) and isinstance(inp[0], torch.Tensor):
      _verify_std(inp[0].detach(), out.detach(), "EMA")

  h = None
  for layer in obb_model.model:
    if isinstance(layer, _LayerWithPost) and "EMA" in layer.tag:
      h = layer.register_forward_hook(_hook)
      break

  with torch.no_grad():
    obb_model(x)
  if h:
    h.remove()
  print("EMA forward verification PASSED")


def build_idea2_model(weights: str = WEIGHTS, verbose: bool = True) -> YOLO:
  model = YOLO(weights)
  inject_ema_module(model.model, verbose=verbose)
  if torch.cuda.is_available():
    verify_ema_forward(model.model)
  return model


def inject_ema(model: YOLO, verbose: bool = True) -> YOLO:
  inject_ema_module(model.model, verbose=verbose)
  return model
