#!/usr/bin/env python3
"""Verify WPL channel_proj init matches baseline Conv activation scale at layer 0."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SmallTrack"))

WEIGHTS = "pretrained/yolo11m-obb.pt"


def main() -> int:
  from ultralytics import YOLO
  from models.inject_wpl import inject_wpl

  assert torch.cuda.is_available(), "CUDA required"

  baseline = YOLO(WEIGHTS)
  wpl_model = inject_wpl(YOLO(WEIGHTS))

  baseline.model.cuda().eval()
  wpl_model.model.cuda().eval()

  x = torch.randn(1, 3, 640, 640).cuda()

  base_acts: dict = {}
  wpl_acts: dict = {}

  def make_hook(store: dict, key: str):
    def hook(_module, _inp, out):
      store[key] = out[0] if isinstance(out, (list, tuple)) else out

    return hook

  baseline.model.model[0].register_forward_hook(make_hook(base_acts, "layer0"))
  wpl_model.model.model[0].register_forward_hook(make_hook(wpl_acts, "layer0"))

  with torch.no_grad():
    baseline.model(x)
    wpl_model.model(x)

  b_std = base_acts["layer0"].std().item()
  w_std = wpl_acts["layer0"].std().item()
  ratio = w_std / b_std

  print(f"Baseline layer 0 std: {b_std:.4f}")
  print(f"WPL     layer 0 std: {w_std:.4f}")
  print(f"Ratio (WPL/baseline): {ratio:.4f}")
  passed = ratio > 0.5
  print(f"Target: ratio > 0.5  {'✅ PASS' if passed else '❌ FAIL - init still wrong'}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
