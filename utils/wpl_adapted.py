"""
WPL (Wavelet Pooling Layer) adapted from SmallTrack wad_module for YOLOv11-OBB.

Original: ./SmallTrack/siamban/models/backbone/DWT/wad_module.py
Usage in SmallTrack: replaces MaxPool2d(stride=2) in ResNet backbone after conv1+relu.

Dependencies (from SmallTrack wad_module.py):
  - torch, torch.nn
  - pywt (PyWavelets)          -> pip install PyWavelets
  - siamban.models.backbone.DWT.DWT_layer (DWT_2D)  -> requires SmallTrack on sys.path
  - matplotlib, PIL, cv2, imageio (imported by upstream wad_module; viz-only at runtime)

DWT_2D note: forward() moves input to CUDA internally (SmallTrack design).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.smalltrack_loader import import_wad_module_class

# Dependency check — report missing packages before adapter use
_MISSING = []
try:
  import pywt  # noqa: F401
except ImportError:
  _MISSING.append("PyWavelets (pip install PyWavelets)")

if _MISSING:
  import warnings

  warnings.warn(
      "WPL dependencies missing: " + "; ".join(_MISSING),
      ImportWarning,
      stacklevel=1,
  )

# Original wad_module — imported exactly as written in SmallTrack (not reimplemented)
wad_module = import_wad_module_class()


class WPLYOLOAdapter(nn.Module):
  """
  Adapts wad_module for YOLOv11 backbone stride-2 downsampling.

  SmallTrack wad_module replaces MaxPool2d(stride=2).
  YOLOv11 uses Conv(in_ch, out_ch, stride=2) for downsampling.

  This adapter:
    - Accepts (B, in_ch, H, W) — same as Conv(stride=2)
    - Outputs (B, out_ch, H//2, W//2) — same spatial reduction via DWT
    - Uses wad_module DWT core for the downsampling
    - Adds 1x1 channel projection if in_ch != out_ch
  """

  def __init__(self, in_channels: int, out_channels: int, wavename: str = "haar"):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    # Original wad_module — keep exactly as in SmallTrack
    self.wad = wad_module(wavename=wavename)

    # wad_module preserves channel count (DWT LL subband); project if needed
    wad_out_channels = in_channels
    if wad_out_channels != out_channels:
      self.channel_proj = nn.Sequential(
          nn.Conv2d(wad_out_channels, out_channels, 1, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.SiLU(inplace=True),
      )
    else:
      self.channel_proj = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # DWT_2D inside wad_module expects CUDA tensors (SmallTrack behaviour)
    if not x.is_cuda and torch.cuda.is_available():
      x = x.cuda()
    x = self.wad(x)
    x = self.channel_proj(x)
    return x
