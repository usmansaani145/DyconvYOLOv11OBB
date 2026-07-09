#!/usr/bin/env python3
"""
Ablation C-gammafix: GEM only with gamma fixes (seed 42 diagnostic re-run).

Fixes applied (all together):
  1. Exclude GEM gamma from weight decay (wd=0)
  2. Higher LR for gamma scalars (0.01 vs base 0.001)
  3. Initialize gamma=0.1 (not 0) via GEMYOLOAdapter(gamma_init=0.1)
  4. Per-epoch gamma logging to train/gamma_log.csv

Param-group builder and logging callback live in shared/wavegem_ablation.py:
  - build_gem_param_groups()
  - _apply_gem_gamma_optimizer()
  - _make_log_gamma_callback()
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared.wavegem_ablation import (  # noqa: E402
    build_gem_param_groups,
    run_ablation,
)

# Re-export for inspection / tests (Fix 1 implementation)
__all__ = ["build_gem_param_groups", "main"]


def main() -> int:
  seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
  resume = "--resume" in sys.argv[2:] or (len(sys.argv) > 2 and sys.argv[2] == "resume")

  run_ablation(
      seed=seed,
      ablation="ablation_C_gem_gammafix",
      use_wpl=False,
      use_gem=True,
      resume=resume,
      gem_gamma_fix=True,
      gem_gamma_init=0.1,
      gem_gamma_lr=0.01,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
