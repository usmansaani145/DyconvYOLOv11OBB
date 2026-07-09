#!/usr/bin/env python3
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from shared.constants import RUNS_DIR, SEEDS
from shared.experiment_runner import ExperimentConfig, run_experiment

def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else SEEDS[0]
    if seed not in SEEDS:
        raise SystemExit(f"Seed must be one of {SEEDS}, got {seed}")
    cfg = ExperimentConfig(
        experiment_name=f"yolov26_obb_seed{seed}",
        output_dir=str(RUNS_DIR / "yolov26_obb"),
        report_title=f"yolov26_obb seed{seed} (ship_obb_v2)",
        metrics_json_name="metrics_yolov26.json",
        pretrained=PRETRAINED_YOLO26,
        lr0=0.01,
        degrees=5.0,
        seed=seed,
        epochs=100,
        resume=False,
    )
    run_experiment(cfg)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
