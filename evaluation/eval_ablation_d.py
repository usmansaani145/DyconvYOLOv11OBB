#!/usr/bin/env python3
"""Re-run post-training eval + metrics for ablation D (WPL+GEM, seed 42)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared.wavegem_ablation import eval_ablation_metrics

if __name__ == "__main__":
  seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
  ablation = "ablation_D_wpl_gem"
  print(f"\n{'#' * 60}\n# Evaluating {ablation} seed={seed}\n{'#' * 60}")
  eval_ablation_metrics(
      seed=seed,
      ablation=ablation,
      use_wpl=True,
      use_gem=True,
      p2_yaml=None,
  )
