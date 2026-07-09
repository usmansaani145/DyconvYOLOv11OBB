"""Inject LSKA into YOLOv11-OBB backbone (Idea 1).

Injection points (see ideas/yolov11_module_tree.txt):
  - Layer 9  (SPPF, P5 backbone tail): LSKA after SPPF
  - Layer 4  (C3k2, P3): LSKA after deepest P3 C3k2 block
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO

from modules.lska import LSKABlock

WEIGHTS = "pretrained/yolo11m-obb.pt"
LSKA_LAYERS = (4, 9)  # P3 C3k2, SPPF/P5


def _find_backbone_end(seq: nn.ModuleList) -> int:
  for i, layer in enumerate(seq):
    if type(layer).__name__ == "Upsample":
      return i
  return 11


def _copy_layer_attrs(src: nn.Module, dst: nn.Module) -> None:
  for attr in ("i", "f", "type", "np", "stride"):
    if hasattr(src, attr):
      setattr(dst, attr, getattr(src, attr))


def _probe_out_channels(layer: nn.Module, default: int = 256) -> int:
  for m in layer.modules():
    if isinstance(m, nn.Conv2d):
      return m.out_channels
  return default


def _probe_layer_out_channels(obb_model: nn.Module, layer_idx: int, img_size: int = 256) -> int:
  """Run a forward hook to read actual tensor channels at layer output."""
  storage: dict = {}
  layer = obb_model.model[layer_idx]

  def hook(_module, _inp, out):
    t = out
    if isinstance(out, (list, tuple)):
      t = out[0]
    if isinstance(t, torch.Tensor):
      storage["c"] = t.shape[1]

  handle = layer.register_forward_hook(hook)
  device = next(obb_model.parameters()).device
  with torch.no_grad():
    obb_model(torch.randn(1, 3, img_size, img_size, device=device))
  handle.remove()
  if "c" not in storage:
    raise RuntimeError(f"Could not probe channels at layer {layer_idx}")
  return int(storage["c"])


class _LayerWithPost(nn.Module):
  """Run original YOLO layer then apply post-module (LSKA)."""

  def __init__(self, orig: nn.Module, post: nn.Module, tag: str):
    super().__init__()
    self.orig = orig
    self.post = post
    self.tag = tag
    _copy_layer_attrs(orig, self)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.orig(x)
    return self.post(x)


def _verify_std(pre: torch.Tensor, post: torch.Tensor, tag: str) -> None:
  pre_std = pre.std().item()
  post_std = post.std().item()
  ratio = post_std / max(pre_std, 1e-8)
  print(f"  [{tag}] pre std={pre_std:.4f} post std={post_std:.4f} ratio={ratio:.4f}")
  if not (0.5 < ratio < 2.0):
    print(f"  WARNING: feature scale mismatch at {tag}: ratio={ratio:.4f}")


def inject_lska_module(
    obb_model: nn.Module,
    lska_layers: Tuple[int, ...] = LSKA_LAYERS,
    kernel_size: int = 23,
    dilation: int = 3,
    verbose: bool = True,
) -> nn.Module:
  seq = obb_model.model
  backbone_end = _find_backbone_end(seq)

  if verbose:
    print("=== Idea1 LSKA — backbone scan ===")
    for i, layer in enumerate(seq):
      marker = " [backbone]" if i < backbone_end else " [head/neck]"
      print(f"  Layer {i:2d}: {type(layer).__name__}{marker}")

  # Probe channels on unmodified layers before wrapping
  if torch.cuda.is_available():
    obb_model.cuda()
  channel_map = {
      idx: _probe_layer_out_channels(obb_model, idx)
      for idx in lska_layers
  }
  if verbose:
    print(f"  Probed channels: {channel_map}")

  injected = 0
  for idx in lska_layers:
    if idx >= len(seq):
      raise ValueError(f"Layer index {idx} out of range (len={len(seq)})")
    layer = seq[idx]
    if isinstance(layer, _LayerWithPost):
      if verbose:
        print(f"  Layer {idx}: already LSKA-wrapped — skip")
      continue
    channels = channel_map[idx]
    lska = LSKABlock(channels, kernel_size=kernel_size, dilation=dilation)
    wrapped = _LayerWithPost(layer, lska, tag=f"LSKA@{idx}")
    seq[idx] = wrapped
    injected += 1
    if verbose:
      print(f"  ✅ Layer {idx} ({type(layer).__name__}): +LSKA(ch={channels})")

  if injected == 0:
    raise RuntimeError("No LSKA modules injected")

  if verbose:
    print(f"LSKA injection complete: {injected} layers wrapped")
  return obb_model


def verify_lska_forward(obb_model: nn.Module, device: str = "cuda") -> None:
  dev = torch.device(device if torch.cuda.is_available() else "cpu")
  obb_model.to(dev).eval()
  x = torch.randn(1, 3, 256, 256, device=dev)
  hooks: List[Any] = []

  def _hook(tag: str):
    pre = {}

    def fn(module, inp, out):
      pre["x"] = inp[0].detach()
      post = out.detach() if isinstance(out, torch.Tensor) else out
      if isinstance(post, torch.Tensor):
        _verify_std(pre["x"], post, tag)

    return fn

  for i, layer in enumerate(obb_model.model):
    if isinstance(layer, _LayerWithPost):
      hooks.append(layer.register_forward_hook(_hook(f"layer{i}")))

  with torch.no_grad():
    obb_model(x)
  for h in hooks:
    h.remove()
  print("LSKA forward verification PASSED")


def build_idea1_model(weights: str = WEIGHTS, verbose: bool = True) -> YOLO:
  model = YOLO(weights)
  inject_lska_module(model.model, verbose=verbose)
  if torch.cuda.is_available():
    verify_lska_forward(model.model)
  return model


def inject_lska(model: YOLO, verbose: bool = True) -> YOLO:
  inject_lska_module(model.model, verbose=verbose)
  return model
