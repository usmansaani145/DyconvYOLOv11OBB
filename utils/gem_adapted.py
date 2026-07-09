"""
GEM (Graph Enhancement Module) adapted from SmallTrack GAL for YOLOv11-OBB head.

Original: ./SmallTrack/siamban/models/gal/gal.py
Usage in SmallTrack: GAL(sync_bn=True, input_channels=2) on cls score map.

Dependencies (from gal.py):
  - torch, torch.nn, torch.nn.functional
  - siamban.models.gal.batchnorm (SynchronizedBatchNorm1d/2d) -> SmallTrack on sys.path

GAL uses dynamic spatial size (no fixed adjacency matrix) — works at any H×W.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.smalltrack_loader import import_gal_class

# Original GAL — imported exactly as written in SmallTrack (not reimplemented)
GAL = import_gal_class()

# Dependency check (after SmallTrack path is configured)
_MISSING = []
try:
  from siamban.models.gal.batchnorm import (  # noqa: F401
      SynchronizedBatchNorm1d,
      SynchronizedBatchNorm2d,
  )
except ImportError as exc:
  _MISSING.append(f"SmallTrack gal.batchnorm ({exc})")

if _MISSING:
  import warnings

  warnings.warn(
      "GEM dependencies missing: " + "; ".join(_MISSING),
      ImportWarning,
      stacklevel=1,
  )


class GEMYOLOAdapter(nn.Module):
  """
  Adapts GAL for YOLOv11-OBB classification branch.

  SmallTrack: GAL(sync_bn=True, input_channels=2) on 2-channel cls score map
  YOLOv11:    Classification branch has ~256 channels

  Strategy: channel reduction → GAL → channel expansion
  Residual gate gamma initialised to 0 by default (identity at start).
  Pass gamma_init=0.1 for gamma-fix ablations (non-zero head start).
  """

  def __init__(
      self,
      in_channels: int,
      reduction: int = 8,
      sync_bn: bool = False,
      gamma_init: float = 0.0,
  ):
    super().__init__()
    if torch.cuda.device_count() <= 1:
      sync_bn = False

    self.reduced_ch = max(in_channels // reduction, 16)

    self.pre_proj = nn.Sequential(
        nn.Conv2d(in_channels, self.reduced_ch, 1, bias=False),
        nn.BatchNorm2d(self.reduced_ch),
        nn.ReLU(inplace=True),
    )

    # Original GAL — keep exactly as SmallTrack
    self.gal = GAL(sync_bn=sync_bn, input_channels=self.reduced_ch)

    self.post_proj = nn.Sequential(
        nn.Conv2d(self.reduced_ch, in_channels, 1, bias=False),
        nn.BatchNorm2d(in_channels),
    )

    # Residual gate — 0.0 = identity at start; 0.1 = gamma-fix head start
    self.gamma = nn.Parameter(torch.full((1,), float(gamma_init)))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = self.pre_proj(x)
    x = self.gal(x)
    x = self.post_proj(x)
    return residual + self.gamma * x
