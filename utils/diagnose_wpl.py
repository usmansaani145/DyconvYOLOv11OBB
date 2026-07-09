#!/usr/bin/env python3
"""WPL diagnostic: structure, activation stats, stripe preservation, vs baseline Conv."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SmallTrack"))

WEIGHTS = "pretrained/yolo11m-obb.pt"


def _stats(t: torch.Tensor, name: str) -> None:
  print(
      f"  {name}: shape={tuple(t.shape)} "
      f"mean={t.mean().item():.4f} std={t.std().item():.4f} "
      f"min={t.min().item():.4f} max={t.max().item():.4f}"
  )


def _hook_layer(model_seq, layer_idx: int, x: torch.Tensor, label: str) -> torch.Tensor:
  """Run forward through model_seq[0:layer_idx+1] and print stats."""
  out = x
  for i in range(layer_idx + 1):
    layer = model_seq[i]
    out = layer(out)
    if isinstance(out, (list, tuple)):
      out = out[0] if len(out) else out
  _stats(out, f"{label} after layer {layer_idx} ({type(model_seq[layer_idx]).__name__})")
  return out


def compare_baseline_vs_wpl() -> None:
  from ultralytics import YOLO
  from models.inject_wpl import inject_wpl_module
  from modules.wpl_adapted import WPLYOLOAdapter

  print("\n" + "=" * 70)
  print("1. MODEL STRUCTURE: baseline vs WPL-injected")
  print("=" * 70)

  baseline = YOLO(WEIGHTS)
  wpl_yolo = YOLO(WEIGHTS)
  inject_wpl_module(wpl_yolo.model, verbose=True)

  base_seq = baseline.model.model
  wpl_seq = wpl_yolo.model.model

  print("\n--- Layer type comparison (first 12 backbone layers) ---")
  for i in range(min(12, len(base_seq))):
    bt = type(base_seq[i]).__name__
    wt = type(wpl_seq[i]).__name__
    marker = " <-- WPL" if isinstance(wpl_seq[i], WPLYOLOAdapter) else ""
    changed = " *" if bt != wt else ""
    print(f"  Layer {i:2d}: baseline={bt:<20} wpl={wt:<20}{changed}{marker}")

  print("\n--- Early-layer activation stats (640x640 input) ---")
  device = torch.device("cuda")
  x = torch.randn(1, 3, 640, 640, device=device)
  baseline.model.model.to(device)
  wpl_yolo.model.model.to(device)

  for idx in [0, 1, 2]:
    print(f"\n  Layer index {idx}:")
    with torch.no_grad():
      _hook_layer(base_seq, idx, x, "baseline")
      _hook_layer(wpl_seq, idx, x, "wpl")

  # Compare stride-2 replacement layers directly
  wpl_indices = [i for i, m in enumerate(wpl_seq) if isinstance(m, WPLYOLOAdapter)]
  print(f"\n--- WPL adapter layers at indices: {wpl_indices} ---")
  for wi in wpl_indices:
    bi = wi
    print(f"\n  Comparing layer {wi}:")
    with torch.no_grad():
      # propagate to input of this layer
      inp_b = x
      inp_w = x
      for j in range(wi):
        inp_b = base_seq[j](inp_b)
        inp_w = wpl_seq[j](inp_w)
        if isinstance(inp_b, (list, tuple)):
          inp_b = inp_b[0]
        if isinstance(inp_w, (list, tuple)):
          inp_w = inp_w[0]
      _stats(inp_b, "input to layer (baseline path)")
      _stats(inp_w, "input to layer (wpl path)")
      out_b = base_seq[wi](inp_b)
      out_w = wpl_seq[wi](inp_w)
      if isinstance(out_b, (list, tuple)):
        out_b = out_b[0]
      if isinstance(out_w, (list, tuple)):
        out_w = out_w[0]
      _stats(out_b, "baseline Conv stride-2 output")
      _stats(out_w, "WPLYOLOAdapter output")
      ratio = out_w.std().item() / max(out_b.std().item(), 1e-8)
      print(f"  std ratio (wpl/baseline): {ratio:.3f}")


def check_wad_statistics() -> None:
  from modules.wpl_adapted import WPLYOLOAdapter, wad_module

  print("\n" + "=" * 70)
  print("2. WAD MODULE OUTPUT STATISTICS")
  print("=" * 70)

  device = torch.device("cuda")

  for in_ch, out_ch in [(3, 64), (64, 128)]:
    print(f"\n--- WPLYOLOAdapter({in_ch} -> {out_ch}) ---")
    wpl = WPLYOLOAdapter(in_ch, out_ch).to(device).eval()
    x = torch.randn(2, in_ch, 640, 640, device=device)
    with torch.no_grad():
      out = wpl(x)
    _stats(x, "Input")
    _stats(out, "Output")
    ratio = out.std().item() / max(x.std().item(), 1e-8)
    print(f"  std ratio (out/in): {ratio:.3f}")
    if ratio > 3.0:
      print("  WARNING: possible feature explosion (std ratio > 3)")
    elif ratio < 0.1:
      print("  WARNING: possible feature collapse (std ratio < 0.1)")

  print("\n--- Raw wad_module (no channel_proj) ---")
  raw = wad_module(wavename="haar").to(device).eval()
  for in_ch in [3, 64]:
    x = torch.randn(2, in_ch, 640, 640, device=device)
    with torch.no_grad():
      out = raw(x)
    _stats(x, f"Input ch={in_ch}")
    _stats(out, f"Output ch={in_ch}")
    print(f"  std ratio: {out.std().item() / max(x.std().item(), 1e-8):.3f}")


def check_stripe_preservation() -> None:
  from modules.wpl_adapted import WPLYOLOAdapter

  print("\n" + "=" * 70)
  print("3. SPATIAL INFORMATION (vertical stripe pattern)")
  print("=" * 70)

  device = torch.device("cuda")
  wpl = WPLYOLOAdapter(3, 64).to(device).eval()

  x_test = torch.zeros(1, 3, 64, 64, device=device)
  x_test[:, :, :, ::2] = 1.0

  with torch.no_grad():
    out = wpl(x_test)

  in_std = x_test.std().item()
  out_std = out.std().item()
  print(f"  Input stripe std:  {in_std:.4f}")
  print(f"  Output std:        {out_std:.4f}")
  print(f"  Output mean:       {out.mean().item():.4f}")

  # spatial variance across columns (should show structure if preserved)
  col_var = out[0].var(dim=(0, 1, 2)).mean().item()
  print(f"  Output mean col-var: {col_var:.6f}")

  if out_std < 0.01:
    print("  FAIL: stripe pattern likely collapsed")
  else:
    print("  OK: non-trivial spatial structure in output")


def compare_trained_vs_baseline_preds() -> None:
  """Compare Ultralytics vs COCO AP gap on trained B checkpoint if present."""
  import json

  print("\n" + "=" * 70)
  print("4. TRAINED Abl-B: Ultralytics val vs stored COCO metrics")
  print("=" * 70)

  metrics_path = Path(
      "runs/ablation_B_wpl/ablation_B_wpl_seed42/"
      "metrics_ablation_B_wpl_seed42.json"
  )
  rc_path = Path(
      "runs/ablation_B_wpl/ablation_B_wpl_seed42/"
      "train/results.csv"
  )
  if metrics_path.exists():
    m = json.loads(metrics_path.read_text())
    print(f"  Stored COCO AP:    {m.get('AP', 'N/A')}")
    print(f"  Stored COCO AP50:  {m.get('AP50', 'N/A')}")
    print(f"  Stored Recall:     {m.get('Recall', 'N/A')}")
  if rc_path.exists():
    last = rc_path.read_text().strip().split("\n")[-1].split(",")
    if len(last) >= 9:
      print(f"  Training val mAP50:     {last[7]}")
      print(f"  Training val mAP50-95:  {last[8]}")
      print(
          "\n  NOTE: Large gap between training Ultralytics OBB mAP and COCO axis-aligned"
          " AP suggests eval conversion or OBB quality issue, not necessarily WPL forward bug."
      )


def main() -> int:
  assert torch.cuda.is_available(), "CUDA required for WPL diagnostics (DWT uses CUDA)"
  print(f"GPU: {torch.cuda.get_device_name(0)}")
  compare_baseline_vs_wpl()
  check_wad_statistics()
  check_stripe_preservation()
  compare_trained_vs_baseline_preds()
  print("\n" + "=" * 70)
  print("DIAGNOSTICS COMPLETE")
  print("=" * 70)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
