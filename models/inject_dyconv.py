"""Inject DynamicConv into YOLOv11-OBB mid-backbone (Idea 4).

Replaces mid-backbone stride-2 Conv layers (NOT stem layers 0/1):
  - Layer 3: Conv 256->256 (P3 path)
  - Layer 5: Conv 512->512 (P4 path)

Sci Reports 2025 — DyConv for remote sensing appearance diversity.
Gamma gate: init=0.1, wd=0, lr=10x (handled in train_idea4.py).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO

from modules.dyconv import DynamicConv

WEIGHTS = "pretrained/yolo11m-obb.pt"
DYCONV_LAYERS = (3, 5)


def _copy_layer_attrs(src: nn.Module, dst: nn.Module) -> None:
  for attr in ("i", "f", "type", "np", "stride"):
    if hasattr(src, attr):
      setattr(dst, attr, getattr(src, attr))


def _stride_is_2(conv_layer: nn.Module) -> bool:
  if not hasattr(conv_layer, "conv"):
    return False
  stride = conv_layer.conv.stride
  if hasattr(stride, "__iter__"):
    return tuple(stride) == (2, 2)
  return stride == 2


class _DyConvLayer(nn.Module):
  """Replace Ultralytics Conv with DynamicConv (preserves YOLO layer attrs)."""

  def __init__(self, orig: nn.Module):
    super().__init__()
    if not hasattr(orig, "conv"):
      raise ValueError(f"Layer {type(orig).__name__} has no .conv attribute")
    in_ch = orig.conv.in_channels
    out_ch = orig.conv.out_channels
    k = orig.conv.kernel_size[0] if hasattr(orig.conv.kernel_size, "__iter__") else orig.conv.kernel_size
    stride = orig.conv.stride[0] if hasattr(orig.conv.stride, "__iter__") else orig.conv.stride
    self.dyconv = DynamicConv(in_ch, out_ch, kernel_size=k, K=4, stride=int(stride))
    _copy_layer_attrs(orig, self)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.dyconv(x)


def inject_dyconv_module(
    obb_model: nn.Module,
    target_layers: Tuple[int, ...] = DYCONV_LAYERS,
    verbose: bool = True,
) -> nn.Module:
  seq = obb_model.model

  if verbose:
    print("=== Idea4 DyConv — backbone scan ===")
    for i, layer in enumerate(seq):
      stride_info = ""
      if hasattr(layer, "conv"):
        stride_info = f" stride={tuple(layer.conv.stride)}"
      print(f"  Layer {i:2d}: {type(layer).__name__}{stride_info}")

  replaced = 0
  for idx in target_layers:
    if idx in (0, 1):
      raise ValueError(f"Refusing to replace early layer {idx} (WPL lesson)")
    layer = seq[idx]
    if isinstance(layer, _DyConvLayer):
      if verbose:
        print(f"  Layer {idx}: already DyConv — skip")
      continue
    if type(layer).__name__ != "Conv" or not _stride_is_2(layer):
      raise ValueError(
          f"Layer {idx} is {type(layer).__name__}, expected stride-2 Conv"
      )
    in_ch = layer.conv.in_channels
    out_ch = layer.conv.out_channels
    seq[idx] = _DyConvLayer(layer)
    replaced += 1
    if verbose:
      print(f"  ✅ Layer {idx}: Conv({in_ch},{out_ch}) -> DynamicConv")

  if replaced == 0:
    raise RuntimeError("No DyConv layers injected")

  if verbose:
    print(f"DyConv injection complete: {replaced} layers")
  return obb_model


def verify_dyconv_forward(obb_model: nn.Module, device: str = "cuda") -> None:
  dev = torch.device(device if torch.cuda.is_available() else "cpu")
  obb_model.to(dev).eval()
  x = torch.randn(1, 3, 256, 256, device=dev)
  with torch.no_grad():
    out = obb_model(x)
  outputs = out if isinstance(out, (list, tuple)) else [out]
  for o in outputs:
    if isinstance(o, torch.Tensor):
      assert not torch.isnan(o).any(), "NaN in DyConv forward"
  print("DyConv forward verification PASSED")


def build_idea4_model(weights: str = WEIGHTS, verbose: bool = True) -> YOLO:
  model = YOLO(weights)
  inject_dyconv_module(model.model, verbose=verbose)
  if torch.cuda.is_available():
    verify_dyconv_forward(model.model)
  return model


def inject_dyconv(model: YOLO, verbose: bool = True) -> YOLO:
  inject_dyconv_module(model.model, verbose=verbose)
  return model
