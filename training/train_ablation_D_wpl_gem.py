#!/usr/bin/env python3
"""Ablation D: WPL + GEM (WaveGEM-OBB main model)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared.wavegem_ablation import run_ablation

if __name__ == "__main__":
  seed = int(sys.argv[1])
  resume = "--resume" in sys.argv[2:] or (len(sys.argv) > 2 and sys.argv[2] == "resume")
  run_ablation(
      seed=seed,
      ablation="ablation_D_wpl_gem",
      use_wpl=True,
      use_gem=True,
      resume=resume,
  )
