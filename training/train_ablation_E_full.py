#!/usr/bin/env python3
"""Ablation E: P2 + WPL + GEM (full model)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared.wavegem_ablation import P2_YAML, run_ablation

if __name__ == "__main__":
  seed = int(sys.argv[1])
  resume = "--resume" in sys.argv[2:] or (len(sys.argv) > 2 and sys.argv[2] == "resume")
  run_ablation(
      seed=seed,
      ablation="ablation_E_full",
      use_wpl=True,
      use_gem=True,
      p2_yaml=P2_YAML,
      resume=resume,
  )
