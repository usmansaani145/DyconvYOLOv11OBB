#!/usr/bin/env python3
"""Build unified val GT COCO JSON from ship_obb_v2 val tiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared.constants import GT_JSON, TILE_SIZE, YOLO_VAL_IMG, YOLO_VAL_LBL
from shared.obb_coco_eval import build_gt_coco_json_tiled


def main() -> int:
    parser = argparse.ArgumentParser(description="Build gt_coco.json for ship_obb_v2")
    parser.add_argument("--lbl-dir", default=YOLO_VAL_LBL)
    parser.add_argument("--img-dir", default=YOLO_VAL_IMG)
    parser.add_argument("--output", default="dataset/gt_coco_filtered.json")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    args = parser.parse_args()

    build_gt_coco_json_tiled(
        img_dir=args.img_dir,
        lbl_dir=args.lbl_dir,
        output_path=args.output,
        tile_size=args.tile_size,
    )
    data = json.loads(Path(args.output).read_text(encoding="utf-8"))
    print(f"Saved: {args.output}")
    print(f"Val tiles: {len(data['images'])}")
    print(f"Total instances: {len(data['annotations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
