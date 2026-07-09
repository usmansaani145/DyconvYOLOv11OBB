#!/usr/bin/env python3
"""Evaluate S2ANET on ship_obb_v2 val set."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared.constants import GT_JSON, MMROTATE_VAL_LBL, RUNS_DIR
from shared.mmrotate_eval_utils import mmrotate_env
from shared.obb_coco_eval import convert_mmrotate_pkl_to_coco_dt, run_unified_eval

CONFIG = Path(__file__).resolve().parent / "config_s2anet_v2.py"
LAUNCHER = REPO / "shared" / "mmrotate_test_launcher.py"
ANGLE = "le135"


def find_checkpoint(work_dir: Path) -> Path:
    for name in ("best_mAP.pth", "best_bbox_mAP.pth", "epoch_100.pth", "latest.pth"):
        p = work_dir / name
        if p.is_file():
            return p
    epochs = sorted(work_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    if epochs:
        return epochs[-1]
    raise FileNotFoundError(f"No checkpoint in {work_dir}")


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    work_dir = RUNS_DIR / "s2anet" / f"s2anet_seed{seed}"
    pkl_path = work_dir / "results.pkl"
    ckpt = find_checkpoint(work_dir)
    print(f"Checkpoint: {ckpt}")

    force_infer = os.environ.get("FORCE_INFER", "").lower() in ("1", "true", "yes")
    if pkl_path.is_file() and not force_infer:
        print(f"Using existing predictions: {pkl_path}")
    else:
        cmd = [
            "python3", str(LAUNCHER),
            str(CONFIG),
            str(ckpt),
            "--gpu-id", "0",
            "--cfg-options", "data.workers_per_gpu=0", "data.samples_per_gpu=1",
            "--out", str(pkl_path),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(work_dir), check=True, env=mmrotate_env(f"./tmp/mpl_s2anet"))

    dt_coco = convert_mmrotate_pkl_to_coco_dt(
        str(pkl_path), str(GT_JSON), str(MMROTATE_VAL_LBL),
        angle_version=ANGLE,
        bbox_json_path=str(work_dir / "predictions_bbox.json"),
    )
    dt_path = work_dir / "predictions_coco_dt.json"
    dt_path.write_text(json.dumps(dt_coco), encoding="utf-8")
    run_unified_eval(
        str(dt_path), str(GT_JSON),
        f"S2ANET seed{seed} (ship_obb_v2)",
        str(work_dir / "metrics_s2anet.json"),
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
