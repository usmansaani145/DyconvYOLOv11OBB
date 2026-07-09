#!/usr/bin/env python3
"""
Build ship_obb_v2 dataset.

Phase A — SAHI 1024×1024 tiling (train + val, overlap=0.2, min_area_ratio=0.1)
Phase B — Discrete rotation {90°,180°,270°} on ship-containing TRAIN tiles only
Phase C — Boundary violation fix (train labels)
Phase D — DOTA labelTxt export for MMRotate + dataset.yaml

Val set is NEVER augmented — tiling only.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2 if "pipeline" in str(__file__) else 1])
AUG_ROOT = REPO_ROOT / "dataset" / "pipeline"
sys.path.insert(0, str(AUG_ROOT))

# Headless OpenCV in Docker (system cv2 needs Qt)
os.environ.setdefault("PYDEPS", "")
os.environ.setdefault("SYSTEM_SITE", "/usr/local/lib/python3.10/dist-packages")
try:
    from utils.env_bootstrap import setup_import_paths

    setup_import_paths()
except Exception:
    pass

import cv2
import numpy as np
import yaml

from utils.common import list_image_label_pairs  # noqa: E402
from utils.obb_slicing import save_tile, slice_one_image  # noqa: E402
from utils.obb_utils import (  # noqa: E402
    LabelFormat,
    OBBInstance,
    load_label_file,
    order_points_clockwise,
    polygon_area,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path("./data")
SOURCE_TRAIN_IMG = Path("dataset/dota_raw/train/images")
SOURCE_TRAIN_LBL = Path("dataset/dota_raw/train/labels")
SOURCE_VAL_IMG = Path("dataset/dota_raw/val/images")
SOURCE_VAL_LBL = Path("dataset/dota_raw/val/labels")

TRAIN_IMG = DATA_ROOT / "train" / "images"
TRAIN_LBL = DATA_ROOT / "train" / "labels"
TRAIN_LABELTXT = DATA_ROOT / "train" / "labelTxt"
VAL_IMG = DATA_ROOT / "val" / "images"
VAL_LBL = DATA_ROOT / "val" / "labels"
VAL_LABELTXT = DATA_ROOT / "val" / "labelTxt"
DATASET_YAML = DATA_ROOT / "dataset.yaml"
BOUNDARY_REPORT = Path("./outputs/boundary_fix_report.txt")

TILE = 1024
OVERLAP = 0.2
MIN_AREA_RATIO = 0.1
ROTATIONS = [90, 180, 270]
SEED = 42
EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

SIZE_SMALL = 1024
SIZE_MEDIUM = 9216


# ---------------------------------------------------------------------------
# Rotation helpers (square 1024×1024 tiles)
# ---------------------------------------------------------------------------

def rotate_image(image: np.ndarray, angle_deg: int) -> np.ndarray:
    if angle_deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle_deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle_deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"Unsupported angle: {angle_deg}")


def rotate_polygon(pts: np.ndarray, angle_deg: int, size: int = TILE) -> np.ndarray:
    """Rotate polygon coordinates to match cv2.rotate for square tiles."""
    t = float(size - 1)
    x = pts[:, 0]
    y = pts[:, 1]
    if angle_deg == 90:
        new = np.column_stack([y, t - x])
    elif angle_deg == 180:
        new = np.column_stack([t - x, t - y])
    elif angle_deg == 270:
        new = np.column_stack([t - y, x])
    else:
        raise ValueError(f"Unsupported angle: {angle_deg}")
    return order_points_clockwise(new.astype(np.float64))


def rotate_instances(instances: Sequence[OBBInstance], angle_deg: int) -> List[OBBInstance]:
    out: List[OBBInstance] = []
    for inst in instances:
        poly = rotate_polygon(inst.polygon, angle_deg)
        clipped = poly
        out.append(
            OBBInstance(
                polygon=order_points_clockwise(clipped),
                class_id=inst.class_id,
                difficulty=inst.difficulty,
                class_name=inst.class_name,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

def slice_split(
    img_dir: Path,
    lbl_dir: Path,
    out_img: Path,
    out_lbl: Path,
) -> Tuple[int, int, int]:
    """Slice one split. Returns (n_source, n_tiles, n_instances)."""
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    pairs = list_image_label_pairs(img_dir, lbl_dir, EXTENSIONS)
    n_src = n_tiles = n_inst = 0

    for idx, (ip, lp) in enumerate(pairs):
        img = cv2.imread(str(ip))
        if img is None:
            print(f"  skip unreadable: {ip}", file=sys.stderr)
            continue
        ih, iw = img.shape[:2]
        insts = load_label_file(lp, iw, ih, LabelFormat.YOLO_POLYGON_NORM)

        tiles = slice_one_image(
            img,
            insts,
            slice_height=TILE,
            slice_width=TILE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            min_area_ratio=MIN_AREA_RATIO,
            pad_to_size=True,
        )

        for t_idx, (tile_bgr, tile_insts, _bbox) in enumerate(tiles):
            stem = f"{ip.stem}_slice{t_idx:03d}"
            save_tile(
                tile_bgr,
                tile_insts,
                out_img / f"{stem}.png",
                out_lbl / f"{stem}.txt",
                norm_w=TILE,
                norm_h=TILE,
            )
            n_tiles += 1
            n_inst += len(tile_insts)

        n_src += 1
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(pairs)} source -> {n_tiles} tiles")

    return n_src, n_tiles, n_inst


# ---------------------------------------------------------------------------
# Rotation augmentation (train only)
# ---------------------------------------------------------------------------

def apply_rotation_augmentation(
    tiled_img: Path,
    tiled_lbl: Path,
    final_img: Path,
    final_lbl: Path,
    rng: random.Random,
) -> Tuple[int, int, int, int]:
    """
    Copy tiled train set to final train dir; add one rotated copy per
    ship-containing tile.

    Returns (n_original, n_rotated, n_ship_tiles, n_empty_tiles).
    """
    final_img.mkdir(parents=True, exist_ok=True)
    final_lbl.mkdir(parents=True, exist_ok=True)

    lbl_files = sorted(tiled_lbl.glob("*.txt"))
    n_orig = n_rot = n_ship = n_empty = 0

    for lp in lbl_files:
        stem = lp.stem
        ip = tiled_img / f"{stem}.png"
        if not ip.is_file():
            continue

        lines = [ln.strip() for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        img = cv2.imread(str(ip))
        if img is None:
            continue

        # Copy original tile
        out_ip = final_img / f"{stem}.png"
        out_lp = final_lbl / f"{stem}.txt"
        cv2.imwrite(str(out_ip), img)
        out_lp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_orig += 1

        if not lines:
            n_empty += 1
            continue

        n_ship += 1
        insts = load_label_file(lp, TILE, TILE, LabelFormat.YOLO_POLYGON_NORM)
        angle = rng.choice(ROTATIONS)
        rot_img = rotate_image(img, angle)
        rot_insts = rotate_instances(insts, angle)

        rot_stem = f"{stem}_rot{angle}"
        rot_ip = final_img / f"{rot_stem}.png"
        rot_lp = final_lbl / f"{rot_stem}.txt"
        cv2.imwrite(str(rot_ip), rot_img)

        from utils.obb_utils import save_yolo_polygon_labels

        save_yolo_polygon_labels(rot_insts, rot_lp, TILE, TILE)
        n_rot += 1

    return n_orig, n_rot, n_ship, n_empty


# ---------------------------------------------------------------------------
# labelTxt + dataset.yaml
# ---------------------------------------------------------------------------

def yolo_labels_to_labeltxt(yolo_lbl_dir: Path, labeltxt_dir: Path) -> Tuple[int, int]:
    """Convert 9-col normalized YOLO polygon labels to DOTA labelTxt pixels."""
    labeltxt_dir.mkdir(parents=True, exist_ok=True)
    n_files = n_inst = 0

    for lp in sorted(yolo_lbl_dir.glob("*.txt")):
        lines_out: List[str] = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            coords = [float(v) for v in parts[1:9]]
            xs = [coords[i] * TILE for i in range(0, 8, 2)]
            ys = [coords[i] * TILE for i in range(1, 8, 2)]
            lines_out.append(
                f"{xs[0]:.2f} {ys[0]:.2f} {xs[1]:.2f} {ys[1]:.2f} "
                f"{xs[2]:.2f} {ys[2]:.2f} {xs[3]:.2f} {ys[3]:.2f} ship 0\n"
            )
            n_inst += 1

        (labeltxt_dir / lp.name).write_text("".join(lines_out), encoding="utf-8")
        n_files += 1

    return n_files, n_inst


def write_dataset_yaml() -> None:
    cfg = {
        "path": str(DATA_ROOT),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": {0: "ship"},
    }
    DATASET_YAML.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_YAML.open("w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Boundary fix
# ---------------------------------------------------------------------------

def run_boundary_fix() -> None:
    fix_script = AUG_ROOT / "fix_boundary_violations.py"
    if not fix_script.is_file():
        raise FileNotFoundError(fix_script)

    cmd = [
        sys.executable,
        str(fix_script),
        "--train-labels",
        str(TRAIN_LBL),
        "--val-labels",
        str(VAL_LBL),
        "--report",
        str(BOUNDARY_REPORT),
    ]
    print("\n=== Phase C: boundary violation fix ===")
    print("Running:", " ".join(cmd))

    env = os.environ.copy()
    env["PYDEPS"] = os.environ.get("PYDEPS", "")
    env["SYSTEM_SITE"] = os.environ.get(
        "SYSTEM_SITE", "/usr/local/lib/python3.10/dist-packages"
    )
    env["SHIP_AUG_BOOTSTRAP"] = "1"
    env["PYTHONSTARTUP"] = str(AUG_ROOT / "bootstrap_startup.py")
    env["PYTHONPATH"] = ":".join(
        [
            str(Path(__file__).resolve().parent),
            str(AUG_ROOT),
            env["PYDEPS"],
            env["SYSTEM_SITE"],
        ]
    )
    subprocess.run(cmd, check=True, env=env)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def count_instances(lbl_dir: Path) -> Tuple[int, int, Dict[str, int]]:
    """Return (n_files, n_instances, size_distribution)."""
    n_files = n_inst = 0
    size_dist = {"small": 0, "medium": 0, "large": 0}

    for lp in lbl_dir.glob("*.txt"):
        lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            n_files += 1
            continue
        n_files += 1
        insts = load_label_file(lp, TILE, TILE, LabelFormat.YOLO_POLYGON_NORM)
        n_inst += len(insts)
        for inst in insts:
            area = polygon_area(inst.polygon)
            if area < SIZE_SMALL:
                size_dist["small"] += 1
            elif area <= SIZE_MEDIUM:
                size_dist["medium"] += 1
            else:
                size_dist["large"] += 1

    return n_files, n_inst, size_dist


def print_size_distribution(title: str, size_dist: Dict[str, int]) -> None:
    total = sum(size_dist.values()) or 1
    print(f"   {title}")
    for key, label in [("small", "Small  (area < 1024)"), ("medium", "Medium (1024-9216)"), ("large", "Large  (> 9216)")]:
        n = size_dist[key]
        pct = 100.0 * n / total
        print(f"     {label}:  {n:5d} ({pct:5.1f}%)")


def print_statistics(
    n_train_src: int,
    n_val_src: int,
    train_tiles_after_tiling: int,
    val_tiles_after_tiling: int,
    ship_tiles: int,
    n_rotated: int,
) -> None:
    train_files, train_inst, _ = count_instances(TRAIN_LBL)
    val_files, val_inst, val_size = count_instances(VAL_LBL)

    print("\n=== Dataset v2 statistics ===")
    print(f"   Original train images    : {n_train_src}")
    print(f"   Original val images      : {n_val_src}")
    print(f"   Train tiles after tiling : {train_tiles_after_tiling}")
    print(f"   Val tiles after tiling   : {val_tiles_after_tiling}")
    print(f"   Ship-containing tiles    : {ship_tiles}")
    print(f"   After rotation aug (×2)  : {ship_tiles + n_rotated}  (+{n_rotated} rotated copies)")
    print(f"   Final train tiles        : {train_files}  (target: ~3,000)")
    print(f"   Final val tiles          : {val_files}  (target: ~800)")
    print(f"   Total train instances    : {train_inst}")
    print(f"   Total val instances      : {val_inst}")
    print("   Size distribution (val)  :")
    print_size_distribution("", val_size)
    print("   ==============================")


def verify_dataset() -> int:
    """Verify built dataset without rebuilding."""
    errors: List[str] = []

    for p in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL, DATASET_YAML]:
        if not p.exists():
            errors.append(f"Missing: {p}")

    # Val must not contain rotation-augmented tiles
    val_rot = list(VAL_LBL.glob("*_rot*.txt"))
    if val_rot:
        errors.append(f"Val contains rotated tiles (forbidden): {len(val_rot)}")

    train_files, train_inst, _ = count_instances(TRAIN_LBL)
    val_files, val_inst, val_size = count_instances(VAL_LBL)

    print("\n=== Dataset v2 verification ===")
    print(f"   Train tiles : {train_files}")
    print(f"   Val tiles   : {val_files}")
    print(f"   Train inst  : {train_inst}")
    print(f"   Val inst    : {val_inst}")
    print(f"   Val rotated : {len(val_rot)} (must be 0)")
    print("   Size distribution (val)  :")
    print_size_distribution("", val_size)

    if errors:
        print("\n   ERRORS:")
        for e in errors:
            print(f"     - {e}")
        return 1

    print("\n   Verification PASSED")
    return 0


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def finalize_dataset(
    n_train_src: int = 437,
    n_val_src: int = 145,
    train_tiled: int = 9632,
    val_tiled: int = 3191,
    ship_tiles: int = 3062,
    n_rot: int = 3062,
) -> int:
    """Run Phase C + D only (resume after tiling/rotation already on disk)."""
    run_boundary_fix()
    print("\n=== Phase D: labelTxt + dataset.yaml ===")
    yolo_labels_to_labeltxt(TRAIN_LBL, TRAIN_LABELTXT)
    yolo_labels_to_labeltxt(VAL_LBL, VAL_LABELTXT)
    write_dataset_yaml()
    print(f"  Saved: {DATASET_YAML}")
    print_statistics(n_train_src, n_val_src, train_tiled, val_tiled, ship_tiles, n_rot)
    return 0


def build_dataset() -> int:
    rng = random.Random(SEED)

    tmp_train_img = DATA_ROOT / "_tmp" / "train" / "images"
    tmp_train_lbl = DATA_ROOT / "_tmp" / "train" / "labels"

    print("=== Phase A: SAHI tiling ===")
    print("Train:", SOURCE_TRAIN_IMG)
    n_train_src, train_tiled, train_inst_tiled = slice_split(
        SOURCE_TRAIN_IMG, SOURCE_TRAIN_LBL, tmp_train_img, tmp_train_lbl
    )
    print(f"  Train: {n_train_src} images -> {train_tiled} tiles, {train_inst_tiled} instances")

    print("Val:", SOURCE_VAL_IMG)
    n_val_src, val_tiled, val_inst_tiled = slice_split(
        SOURCE_VAL_IMG, SOURCE_VAL_LBL, VAL_IMG, VAL_LBL
    )
    print(f"  Val: {n_val_src} images -> {val_tiled} tiles, {val_inst_tiled} instances")

    print("\n=== Phase B: discrete rotation augmentation (train only) ===")
    n_orig, n_rot, ship_tiles, n_empty = apply_rotation_augmentation(
        tmp_train_img, tmp_train_lbl, TRAIN_IMG, TRAIN_LBL, rng
    )
    print(f"  Original train tiles : {n_orig}")
    print(f"  Ship-containing      : {ship_tiles}")
    print(f"  Empty (no rotation)  : {n_empty}")
    print(f"  Rotated copies added : {n_rot}")

    return finalize_dataset(
        n_train_src, n_val_src, train_tiled, val_tiled, ship_tiles, n_rot
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ship_obb_v2 dataset")
    p.add_argument(
        "--verify_only",
        action="store_true",
        help="Verify existing dataset without rebuilding",
    )
    p.add_argument(
        "--finalize_only",
        action="store_true",
        help="Run boundary fix + labelTxt + dataset.yaml only (resume)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        return verify_dataset()
    if args.finalize_only:
        return finalize_dataset()
    return build_dataset()


if __name__ == "__main__":
    raise SystemExit(main())
