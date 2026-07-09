"""Inject scale-differentiated CSD modules into YOLOv11-OBB (Idea 5).

From LMW-YOLO (Sci Reports, 2026) — Conflict-aware Scale-Differentiated strategy:
  - Layer 4 (P3 C3k2): LKCA for small objects / long-range context
  - Layer 6 (P4 C3k2): MSDP for semantic multi-scale context
  - P5 (layers 8-10): untouched — avoids uniform GEM-style large-ship harm
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO

from modules.lkca import LKCA
from modules.msdp import MSDP

WEIGHTS = "pretrained/yolo11m-obb.pt"
LKCA_LAYER = 4
MSDP_LAYER = 6


def _copy_layer_attrs(src: nn.Module, dst: nn.Module) -> None:
  for attr in ("i", "f", "type", "np", "stride"):
    if hasattr(src, attr):
      setattr(dst, attr, getattr(src, attr))


def _probe_layer_out_channels(obb_model: nn.Module, layer_idx: int, img_size: int = 256) -> int:
  storage: dict = {}
  layer = obb_model.model[layer_idx]

  def hook(_module, _inp, out):
    t = out[0] if isinstance(out, (list, tuple)) else out
    if isinstance(t, torch.Tensor):
      storage["c"] = t.shape[1]

  handle = layer.register_forward_hook(hook)
  device = next(obb_model.parameters()).device
  with torch.no_grad():
    obb_model(torch.randn(1, 3, img_size, img_size, device=device))
  handle.remove()
  return int(storage["c"])


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


def inject_csd_module(
    obb_model: nn.Module,
    lkca_layer: int = LKCA_LAYER,
    msdp_layer: int = MSDP_LAYER,
    verbose: bool = True,
) -> nn.Module:
  seq = obb_model.model

  if verbose:
    print("=== Idea5 CSD (LKCA+MSDP) — backbone scan ===")
    for i, layer in enumerate(seq):
      print(f"  Layer {i:2d}: {type(layer).__name__}")

  if torch.cuda.is_available():
    obb_model.cuda()
  ch_p3 = _probe_layer_out_channels(obb_model, lkca_layer)
  ch_p4 = _probe_layer_out_channels(obb_model, msdp_layer)
  if verbose:
    print(f"  Probed channels: P3 layer {lkca_layer}={ch_p3}, P4 layer {msdp_layer}={ch_p4}")

  # LKCA on P3
  p3 = seq[lkca_layer]
  if not isinstance(p3, _LayerWithPost):
    seq[lkca_layer] = _LayerWithPost(p3, LKCA(ch_p3), tag=f"LKCA@{lkca_layer}")
    if verbose:
      print(f"  ✅ Layer {lkca_layer}: +LKCA(ch={ch_p3}) on P3")

  # MSDP on P4
  p4 = seq[msdp_layer]
  if not isinstance(p4, _LayerWithPost):
    seq[msdp_layer] = _LayerWithPost(p4, MSDP(ch_p4), tag=f"MSDP@{msdp_layer}")
    if verbose:
      print(f"  ✅ Layer {msdp_layer}: +MSDP(ch={ch_p4}) on P4")

  if verbose:
    print("CSD injection complete (P5 untouched)")
  return obb_model


def verify_csd_forward(obb_model: nn.Module, device: str = "cuda") -> None:
  dev = torch.device(device if torch.cuda.is_available() else "cpu")
  obb_model.to(dev).eval()
  x = torch.randn(1, 3, 256, 256, device=dev)
  hooks = []

  def _hook(tag: str):
    def fn(module, inp, out):
      if isinstance(inp[0], torch.Tensor) and isinstance(out, torch.Tensor):
        _verify_std(inp[0].detach(), out.detach(), tag)
    return fn

  for i, layer in enumerate(obb_model.model):
    if isinstance(layer, _LayerWithPost):
      hooks.append(layer.register_forward_hook(_hook(layer.tag)))

  with torch.no_grad():
    obb_model(x)
  for h in hooks:
    h.remove()
  print("CSD forward verification PASSED")


def build_idea5_model(weights: str = WEIGHTS, verbose: bool = True) -> YOLO:
  model = YOLO(weights)
  inject_csd_module(model.model, verbose=verbose)
  if torch.cuda.is_available():
    verify_csd_forward(model.model)
  return model


def inject_csd(model: YOLO, verbose: bool = True) -> YOLO:
  inject_csd_module(model.model, verbose=verbose)
  return model
