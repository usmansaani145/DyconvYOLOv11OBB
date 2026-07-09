#!/usr/bin/env python3
"""Ablation A: P2 head only (no WPL, no GEM)."""

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
      ablation="ablation_A_p2",
      use_wpl=False,
      use_gem=False,
      p2_yaml=P2_YAML,
      resume=resume,
  )
