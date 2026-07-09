#!/usr/bin/env python3
"""
Diagnostic 2 — GEM / GAL spatial resolution and compute at P3 and P4.

Checks whether max_nodes downsampling exists (it does NOT in the current
SmallTrack GAL port) and reports actual feature-map sizes, channel reduction,
and memory footprint at imgsz=1024.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SmallTrack"))

# Typical YOLOv11m-OBB cls branch channel widths at P3 / P4 (after cv3[0], cv3[1])
P3_CHANNELS = 256
P4_CHANNELS = 512
IMGSZ = 1024
P3_HW = (IMGSZ // 8, IMGSZ // 8)   # 128 × 128
P4_HW = (IMGSZ // 16, IMGSZ // 16)  # 64 × 64


def _has_max_nodes(obj) -> bool:
  return hasattr(obj, "max_nodes")


def _report_scale(name: str, in_channels: int, h: int, w: int) -> None:
  from modules.gem_adapted import GEMYOLOAdapter

  n = h * w
  gem = GEMYOLOAdapter(in_channels=in_channels, reduction=8, sync_bn=False)
  gal = gem.gal

  print(f"\n{'=' * 60}")
  print(f"{name}  (in_channels={in_channels}, feature map {h}×{w})")
  print(f"{'=' * 60}")
  print(f"  Spatial nodes N = H×W = {n:,}")
  print(f"  GEM reduced_ch  = {gem.reduced_ch}  (in_ch // 8, min 16)")
  print(f"  GAL class       = {type(gal).__name__}")
  print(f"  GAL has max_nodes attr: {_has_max_nodes(gal)}")

  if _has_max_nodes(gal):
    cap = gal.max_nodes
    downsampled = n > cap
    print(f"  max_nodes cap   = {cap}")
    print(f"  downsampled     = {'YES' if downsampled else 'NO'}")
    if downsampled:
      scale = (cap / n) ** 0.5
      print(f"  -> effective grid ~{int(h * scale)}×{int(w * scale)}")
  else:
    print("  max_nodes cap   = NONE (full-resolution graph, no spatial pooling)")
    print("  downsampled     = NO — GAL runs at native H×W")

  # Forward pass + activation memory estimate
  x = torch.randn(2, in_channels, h, w)
  with torch.no_grad():
    out = gem(x)
  elem = 2 * in_channels * h * w * 4  # bytes, rough activations per tensor
  print(f"  Forward OK: {tuple(x.shape)} -> {tuple(out.shape)}")
  print(f"  gamma init    = {gem.gamma.item():.6f}  (residual gate, starts at 0)")
  print(f"  Rough activation mem per scale (batch=2, fp32): ~{elem / 1e6:.1f} MB / tensor")

  # GAL internal reshape sizes (dominant memory terms)
  reduced = gem.reduced_ch
  gal_in = torch.randn(2, reduced, h, w)
  with torch.no_grad():
    b, c, hh, ww = gal_in.shape
    edge = torch.stack(
        (
            torch.cat((gal_in[:, :, -1:], gal_in[:, :, :-1]), dim=2),
            torch.cat((gal_in[:, :, 1:], gal_in[:, :, :1]), dim=2),
            torch.cat((gal_in[:, :, :, -1:], gal_in[:, :, :, :-1]), dim=3),
            torch.cat((gal_in[:, :, :, 1:], gal_in[:, :, :, :1]), dim=3),
        ),
        dim=-1,
    )
  print(f"  GAL edge tensor shape: {tuple(edge.shape)}  "
        f"({edge.numel() * 4 / 1e6:.1f} MB fp32)")


def _report_injection_scales() -> None:
  from ultralytics import YOLO
  from models.inject_gem import inject_gem

  weights = "pretrained/yolo11m-obb.pt"
  print(f"\n{'=' * 60}")
  print("YOLOv11-OBB head — cv3 channel widths (default GEM scales [0,1])")
  print(f"{'=' * 60}")

  model = YOLO(weights)
  inject_gem(model, apply_scales=[0, 1], verbose=True)

  detect = None
  for m in model.model.modules():
    if type(m).__name__ in ("OBB", "Detect", "OBBHead"):
      detect = m
      break

  if detect is None:
    print("WARNING: OBB head not found")
    return

  for idx, cv3 in enumerate(detect.cv3):
    p_level = 3 + idx
    wrapped = type(cv3).__name__ == "Cv3GEMWrapper"
    in_ch = None
    for layer in cv3.modules() if hasattr(cv3, "modules") else []:
      if isinstance(layer, torch.nn.Conv2d):
        in_ch = layer.out_channels
    h = IMGSZ // (8 * (2 ** idx))
    w = h
    print(
        f"  cv3[{idx}] P{p_level}: wrapped={wrapped}, "
        f"est_map={h}×{w}, N={h*w:,}, in_ch≈{in_ch}"
    )


def main() -> int:
  print("GEM spatial / compute diagnostic (imgsz=1024)")
  print("NOTE: Current GAL port has NO max_nodes downsampling.\n")

  _report_scale("P3 (cv3[0])", P3_CHANNELS, P3_HW[0], P3_HW[1])
  _report_scale("P4 (cv3[1])", P4_CHANNELS, P4_HW[0], P4_HW[1])

  try:
    _report_injection_scales()
  except Exception as exc:
    print(f"\nSkipping live YOLO head scan: {exc}")

  print(f"\n{'=' * 60}")
  print("IMPLICATIONS")
  print(f"{'=' * 60}")
  print(
      "  • P3 GAL operates on 128×128 = 16,384 nodes (no cap) — highest cost.\n"
      "  • P4 GAL operates on 64×64  = 4,096 nodes (no cap).\n"
      "  • Hypothesis about 32×32 max_nodes downsampling does NOT apply to\n"
      "    the current codebase — if underperformance exists, causes are\n"
      "    elsewhere (gamma gate, cls-branch interference, optimization)."
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
