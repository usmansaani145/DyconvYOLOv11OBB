"""Inject WPL (wad_module adapter) into YOLOv11-OBB backbone."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _find_backbone_end(seq) -> int:
  """Return index of first head layer (Upsample) in the YOLO sequential model."""
  for i, layer in enumerate(seq):
    if type(layer).__name__ == "Upsample":
      return i
  return 11


def _already_has_wpl(backbone) -> bool:
  from modules.wpl_adapted import WPLYOLOAdapter

  return any(isinstance(layer, WPLYOLOAdapter) for layer in backbone)


def _stride_is_2(conv) -> bool:
  stride = conv.stride
  if hasattr(stride, "__iter__"):
    return tuple(stride) == (2, 2)
  return stride == 2


def _calibrate_wpl_scale(wpl_layer, conv_layer, cal_size: int = 128) -> float:
  """Scale channel_proj weights so WPL output std matches replaced Conv on random input."""
  from modules.wpl_adapted import WPLYOLOAdapter

  if not isinstance(wpl_layer, WPLYOLOAdapter):
    return 1.0
  if isinstance(wpl_layer.channel_proj, nn.Identity):
    return 1.0

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  in_ch = conv_layer.conv.in_channels

  wpl_layer.to(device)
  conv_layer.to(device)
  wpl_layer.eval()
  conv_layer.eval()

  with torch.no_grad():
    x = torch.randn(2, in_ch, cal_size, cal_size, device=device)
    target_std = conv_layer(x).std().item()
    current_std = wpl_layer(x).std().item()

    if current_std < 1e-8 or target_std < 1e-8:
      return 1.0

    scale = target_std / current_std
    wpl_layer.channel_proj[0].weight.data.mul_(scale)
    return scale


def _init_wpl_from_conv(wpl_layer, conv_layer, calibrate: bool = True) -> None:
  """
  Initialize WPLYOLOAdapter.channel_proj from the replaced Ultralytics Conv.

  wad_module preserves channel count (in_ch). channel_proj is Conv2d(in_ch, out_ch, 1).
  Spatial stride-2 downsampling is handled by DWT; we approximate the original Conv's
  channel mixing by spatially averaging its kernel into the 1x1 projection weights.
  """
  from modules.wpl_adapted import WPLYOLOAdapter

  if not isinstance(wpl_layer, WPLYOLOAdapter):
    return
  if isinstance(wpl_layer.channel_proj, nn.Identity):
    return

  proj_conv = wpl_layer.channel_proj[0]
  if not isinstance(proj_conv, nn.Conv2d):
    return

  original_weight = conv_layer.conv.weight.data.clone()
  # (out_ch, in_ch, kH, kW) -> (out_ch, in_ch)
  avg_kernel = original_weight.mean(dim=(2, 3))

  with torch.no_grad():
    if proj_conv.weight.shape[:2] != avg_kernel.shape:
      raise ValueError(
          f"channel_proj shape {tuple(proj_conv.weight.shape[:2])} != "
          f"Conv avg kernel {tuple(avg_kernel.shape)}"
      )
    proj_conv.weight.data[:, :, 0, 0] = avg_kernel

    if hasattr(conv_layer, "bn") and len(wpl_layer.channel_proj) > 1:
      src_bn = conv_layer.bn
      dst_bn = wpl_layer.channel_proj[1]
      if (
          isinstance(dst_bn, nn.BatchNorm2d)
          and dst_bn.weight.shape == src_bn.weight.shape
      ):
        dst_bn.weight.data.copy_(src_bn.weight.data)
        dst_bn.bias.data.copy_(src_bn.bias.data)
        dst_bn.running_mean.data.copy_(src_bn.running_mean.data)
        dst_bn.running_var.data.copy_(src_bn.running_var.data)
        if hasattr(dst_bn, "num_batches_tracked") and hasattr(
            src_bn, "num_batches_tracked"
        ):
          dst_bn.num_batches_tracked.copy_(src_bn.num_batches_tracked)

  if calibrate:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wpl_layer.to(device)
    conv_layer.to(device)
    scale = _calibrate_wpl_scale(wpl_layer, conv_layer)
    wpl_layer._init_scale = scale  # noqa: SLF001 — debug attr


def inject_wpl_module(
    obb_model: nn.Module,
    verbose: bool = True,
    max_replace: int = 2,
) -> nn.Module:
  """
  Inject WPL into a raw Ultralytics OBBModel (trainer.model).

  Safe to call multiple times — skips if WPL is already present.
  max_replace: how many stride-2 Conv layers to replace (1 or 2).
  """
  from modules.wpl_adapted import WPLYOLOAdapter

  backbone = obb_model.model
  if _already_has_wpl(backbone):
    if verbose:
      print("WPL already present — skipping injection")
    return obb_model

  backbone_end = _find_backbone_end(backbone)

  if verbose:
    print("=== YOLOv11 backbone layer scan ===")
    for i, layer in enumerate(backbone):
      layer_type = type(layer).__name__
      stride_info = ""
      in_ch = out_ch = None
      if hasattr(layer, "conv"):
        c = layer.conv
        if hasattr(c, "stride"):
          stride_info = f"stride={tuple(c.stride)}"
        if hasattr(c, "in_channels"):
          in_ch = c.in_channels
          out_ch = c.out_channels
      marker = " [backbone]" if i < backbone_end else " [head]"
      print(
          f"  Layer {i:2d}: {layer_type} in={in_ch} out={out_ch} {stride_info}{marker}"
      )

  replaced = 0
  for i, layer in enumerate(backbone):
    if i >= backbone_end:
      break
    if replaced >= max_replace:
      break
    if hasattr(layer, "conv") and _stride_is_2(layer.conv):
      in_ch = layer.conv.in_channels
      out_ch = layer.conv.out_channels
      old_name = type(layer).__name__

      wpl_layer = WPLYOLOAdapter(in_ch, out_ch)
      _init_wpl_from_conv(wpl_layer, layer)

      for attr in ("i", "f", "type", "np", "stride"):
        if hasattr(layer, attr):
          setattr(wpl_layer, attr, getattr(layer, attr))

      backbone[i] = wpl_layer
      replaced += 1

      if verbose:
        scale_info = ""
        if hasattr(wpl_layer, "_init_scale"):
          scale_info = f", cal_scale={wpl_layer._init_scale:.3f}"
        print(
            f"✅ Replaced layer {i}: {old_name} Conv({in_ch},{out_ch}) "
            f"→ WPLYOLOAdapter (channel_proj initialized from Conv{scale_info})"
        )

  if replaced == 0:
    raise ValueError(
        "No stride-2 Conv layers found in YOLOv11 backbone.\n"
        "Backbone structure was printed above. "
        "Check which layer type contains the stride-2 conv."
    )

  if verbose:
    print(f"\nWPL injection complete: {replaced}/{max_replace} layers replaced")
  return obb_model


def inject_wpl(model: Any, verbose: bool = True, max_replace: int = 2):
  """
  Replace stride-2 Conv layers in YOLOv11 backbone with WPLYOLOAdapter.

  Wrapper around inject_wpl_module for Ultralytics YOLO objects.
  """
  inject_wpl_module(model.model, verbose=verbose, max_replace=max_replace)
  return model
