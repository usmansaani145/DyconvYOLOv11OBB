#!/usr/bin/env python3
"""
Diagnostic 3 — Inspect trained GEM residual gate (gamma) values.

Usage:
  python3 inspect_gem_gamma.py
  python3 inspect_gem_gamma.py --checkpoint path/to/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SmallTrack"))

RUNS = Path("runs/ablation_C_gem")
SEEDS = [42, 123, 456]


def _interpret(gamma: float) -> str:
  if gamma < 0.05:
    return "barely used — model learned to bypass GEM"
  if gamma < 0.5:
    return "moderately used"
  return "heavily relied upon"


def _load_gamma_from_checkpoint(ckpt_path: Path) -> list:
  ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
  model = ckpt.get("ema") or ckpt.get("model")
  if model is None:
    raise RuntimeError(f"No model in {ckpt_path}")

  if hasattr(model, "ema"):
    model = model.ema

  rows = []
  for name, param in model.named_parameters():
    if "gamma" in name.lower():
      rows.append((name, float(param.detach().cpu().item())))
  return rows


def _load_gamma_via_yolo(ckpt_path: Path) -> list:
  from ultralytics import YOLO

  model = YOLO(str(ckpt_path))
  rows = []
  for name, param in model.model.named_parameters():
    if "gamma" in name.lower():
      rows.append((name, float(param.detach().cpu().item())))
  return rows


def inspect_checkpoint(ckpt_path: Path, label: str | None = None) -> None:
  if not ckpt_path.exists():
    print(f"checkpoint missing: {ckpt_path}")
    return

  metrics_guess = ckpt_path.parent.parent.parent / (
      f"metrics_{ckpt_path.parent.parent.parent.name}.json"
  )
  ap = None
  if metrics_guess.exists():
    ap = json.loads(metrics_guess.read_text()).get("AP")

  print(f"\n{'=' * 60}")
  print(label or str(ckpt_path))
  if ap is not None:
    print(f"COCO AP = {ap:.4f}")
  print(f"{'=' * 60}")

  try:
    rows = _load_gamma_via_yolo(ckpt_path)
  except Exception as exc:
    print(f"  YOLO load failed ({exc}), trying raw checkpoint...")
    rows = _load_gamma_from_checkpoint(ckpt_path)

  if not rows:
    print("  No gamma parameters found — is GEM wrapped in this checkpoint?")
    return

  for name, val in rows:
    print(f"  {name}: {val:.6f}  → {_interpret(val)}")

  vals = [v for _, v in rows]
  print(f"  mean gamma = {sum(vals) / len(vals):.6f}")


def inspect_seed(seed: int) -> None:
  run_dir = RUNS / f"ablation_C_gem_seed{seed}"
  best = run_dir / "train" / "weights" / "best.pt"
  inspect_checkpoint(best, label=f"Abl-C seed {seed}  |  {best}")


def main() -> int:
  parser = argparse.ArgumentParser(description="Inspect GEM gamma in trained checkpoints")
  parser.add_argument(
      "--checkpoint",
      type=Path,
      default=None,
      help="Path to best.pt (if omitted, inspect original Abl-C seeds 42/123/456)",
  )
  args = parser.parse_args()

  if args.checkpoint is not None:
    inspect_checkpoint(args.checkpoint.resolve())
  else:
    print("GEM gamma inspection (Abl-C trained checkpoints)")
    for seed in SEEDS:
      inspect_seed(seed)

  print(f"\n{'=' * 60}")
  print("INTERPRETATION GUIDE")
  print(f"{'=' * 60}")
  print("  gamma ≈ 0.00–0.05  → GEM barely used; AP gap ≈ optimization noise")
  print("  gamma ≈ 0.10–0.50  → GEM moderately contributing")
  print("  gamma > 0.50         → GEM heavily relied upon")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
