#!/usr/bin/env python3
"""Re-run post-training eval + metrics for ablations B, C, and E (seed 42)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared.wavegem_ablation import P2_YAML, eval_ablation_metrics

CONFIGS = [
  ("ablation_B_wpl", True, False, None),
  ("ablation_C_gem", False, True, None),
  ("ablation_E_full", True, True, P2_YAML),
]

if __name__ == "__main__":
  seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
  for ablation, use_wpl, use_gem, p2_yaml in CONFIGS:
    print(f"\n{'#' * 60}\n# Evaluating {ablation} seed={seed}\n{'#' * 60}")
    eval_ablation_metrics(
        seed=seed,
        ablation=ablation,
        use_wpl=use_wpl,
        use_gem=use_gem,
        p2_yaml=p2_yaml,
    )
