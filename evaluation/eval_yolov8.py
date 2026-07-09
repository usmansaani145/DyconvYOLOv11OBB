#!/usr/bin/env python3
"""Evaluate YOLOv8-OBB checkpoint (eval-only mode)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared.constants import RUNS_DIR, SEEDS
from shared.experiment_runner import ExperimentConfig, run_experiment


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else SEEDS[0]
    run_dir = RUNS_DIR / "yolov8_obb" / f"yolov8_obb_seed{seed}"
    cfg = ExperimentConfig(
        experiment_name=f"yolov8_obb_seed{seed}",
        output_dir=str(RUNS_DIR / "yolov8_obb"),
        report_title=f"yolov8_obb seed{seed} eval-only",
        metrics_json_name="metrics_yolov8.json",
        seed=seed,
        eval_only=True,
        train=False,
        weights_path=str(run_dir / "weights" / "best.pt"),
    )
    run_experiment(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
