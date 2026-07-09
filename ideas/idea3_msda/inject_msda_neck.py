"""Inject MSDA + CSPStage into YOLOv11-OBB neck (Idea 3).

From LW-YOLO11 (Sensors 2025, 25(1), 65).

Neck injection points:
  - Layer 13: C3k2 after first upsample (P4 neck fusion)
  - Layer 16: C3k2 after second upsample (P3 neck fusion)
Each wrapped block: C3k2 -> MSDA -> CSPStage (residual on MSDA output).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO

from modules.cspstage import CSPStage
from modules.msda import MSDA

WEIGHTS = "pretrained/yolo11m-obb.pt"
NECK_LAYERS = (13, 16)


def _copy_layer_attrs(src: nn.Module, dst: nn.Module) -> None:
  for attr in ("i", "f", "type", "np", "stride"):
    if hasattr(src, attr):
      setattr(dst, attr, getattr(src, attr))


def _probe_out_channels(layer: nn.Module, default: int = 256) -> int:
  for m in layer.modules():
    if isinstance(m, nn.Conv2d):
      return m.out_channels
  return default


class _NeckMSDAWrapper(nn.Module):
  """C3k2 neck block followed by MSDA + CSPStage."""

  def __init__(self, c3k2: nn.Module, channels: int, tag: str):
    super().__init__()
    self.c3k2 = c3k2
    self.msda = MSDA(channels)
    self.csp = CSPStage(channels, n=1)
    self.tag = tag
    _copy_layer_attrs(c3k2, self)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.c3k2(x)
    x = self.msda(x)
    return self.csp(x)


def _verify_std(pre: torch.Tensor, post: torch.Tensor, tag: str) -> None:
  ratio = post.std().item() / max(pre.std().item(), 1e-8)
  print(f"  [{tag}] ratio={ratio:.4f}")
  if not (0.5 < ratio < 2.0):
    print(f"  WARNING: feature scale mismatch at {tag}")


def inject_msda_neck_module(
    obb_model: nn.Module,
    neck_layers: Tuple[int, ...] = NECK_LAYERS,
    verbose: bool = True,
) -> nn.Module:
  seq = obb_model.model

  if verbose:
    print("=== Idea3 MSDA+CSPStage — full layer scan ===")
    for i, layer in enumerate(seq):
      print(f"  Layer {i:2d}: {type(layer).__name__}")

  injected = 0
  defaults = {13: 512, 16: 256}
  for idx in neck_layers:
    if idx >= len(seq):
      raise ValueError(f"Layer {idx} out of range")
    layer = seq[idx]
    if isinstance(layer, _NeckMSDAWrapper):
      if verbose:
        print(f"  Layer {idx}: already MSDA-wrapped — skip")
      continue
    if type(layer).__name__ != "C3k2":
      raise ValueError(
          f"Expected C3k2 at layer {idx}, got {type(layer).__name__}. "
          "Re-check yolov11_module_tree.txt"
      )
    channels = _probe_out_channels(layer, default=defaults.get(idx, 256))
    seq[idx] = _NeckMSDAWrapper(layer, channels, tag=f"MSDA@{idx}")
    injected += 1
    if verbose:
      print(f"  ✅ Layer {idx}: C3k2 -> MSDA+CSPStage(ch={channels})")

  if injected == 0:
    raise RuntimeError("No MSDA neck modules injected")

  if verbose:
    print(f"MSDA neck injection complete: {injected} layers")
  return obb_model


def verify_msda_forward(obb_model: nn.Module, device: str = "cuda") -> None:
  dev = torch.device(device if torch.cuda.is_available() else "cpu")
  obb_model.to(dev).eval()
  x = torch.randn(1, 3, 256, 256, device=dev)
  hooks: List = []

  def _hook(tag: str):
    def fn(module, inp, out):
      if isinstance(inp[0], torch.Tensor) and isinstance(out, torch.Tensor):
        _verify_std(inp[0].detach(), out.detach(), tag)
    return fn

  for i, layer in enumerate(obb_model.model):
    if isinstance(layer, _NeckMSDAWrapper):
      hooks.append(layer.register_forward_hook(_hook(f"neck{i}")))

  with torch.no_grad():
    obb_model(x)
  for h in hooks:
    h.remove()
  print("MSDA neck forward verification PASSED")


def build_idea3_model(weights: str = WEIGHTS, verbose: bool = True) -> YOLO:
  model = YOLO(weights)
  inject_msda_neck_module(model.model, verbose=verbose)
  if torch.cuda.is_available():
    verify_msda_forward(model.model)
  return model


def inject_msda_neck(model: YOLO, verbose: bool = True) -> YOLO:
  inject_msda_neck_module(model.model, verbose=verbose)
  return model
