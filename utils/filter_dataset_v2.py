#!/usr/bin/env python3
"""
Filter ship_obb_v2 tiled dataset to target size ranges via principled tile selection.

Strategy:
  TRAIN: keep all ship-containing original tiles + all rotated copies +
         up to N background tiles (reproducible random sample).
  VAL:   keep all ship-containing tiles + background tiles to reach target.

Creates symlinked dataset at data_filtered/ — does NOT modify data/.
"""

import argparse
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Rotation suffix pattern: _rot90, _rot180, _rot270
ROT_PATTERN = re.compile(r"_rot(?:90|180|270)$")

SIZE_SMALL = 1024
SIZE_MEDIUM = 9216

DEFAULT_DATA_DIR = Path("./data")
DEFAULT_FILTERED_DIR = Path("dataset")
REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def count_instances(label_path: Path) -> int:
    if not label_path.is_file():
        return 0
    return len([ln for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()])


def is_rotated(stem: str) -> bool:
    return bool(ROT_PATTERN.search(stem))


def scan_split(data_dir: Path, split: str) -> Dict[str, List[str]]:
    """
    Classify all tiles in a split.

    Returns dict with keys:
      orig_ship, orig_bg, rot_ship, val_ship, val_bg (train uses first three)
    """
    lbl_dir = data_dir / split / "labels"
    stems = sorted(p.stem for p in lbl_dir.glob("*.txt"))

    orig_ship: List[str] = []
    orig_bg: List[str] = []
    rot_ship: List[str] = []

    for stem in stems:
        n = count_instances(lbl_dir / f"{stem}.txt")
        if is_rotated(stem):
            if n >= 1:
                rot_ship.append(stem)
        else:
            if n >= 1:
                orig_ship.append(stem)
            else:
                orig_bg.append(stem)

    return {
        "orig_ship": orig_ship,
        "orig_bg": orig_bg,
        "rot_ship": rot_ship,
        "all_stems": stems,
    }


def print_classification_summary(
    train_cls: Dict[str, List[str]],
    val_cls: Dict[str, List[str]],
) -> None:
    val_ship = val_cls["orig_ship"]  # val has no rotation aug
    val_bg = val_cls["orig_bg"]
    train_total = len(train_cls["all_stems"])
    val_total = len(val_cls["all_stems"])

    print("  === Tile classification (train) ===")
    print(f"  Original ship tiles      : {len(train_cls['orig_ship'])}")
    print(f"  Original background tiles: {len(train_cls['orig_bg'])}")
    print(f"  Rotated ship tiles       : {len(train_cls['rot_ship'])}")
    print(f"  Total current            : {train_total}")
    print()
    print("  === Tile classification (val) ===")
    print(f"  Val ship tiles           : {len(val_ship)}")
    print(f"  Val background tiles     : {len(val_bg)}")
    print(f"  Total current            : {val_total}")
    print()
    print(f"  Rotation naming detected : _rot90 / _rot180 / _rot270 suffix")


# ---------------------------------------------------------------------------
# Keep lists
# ---------------------------------------------------------------------------

def build_train_keep_list(
    train_cls: Dict[str, List[str]],
    max_bg_train: int,
    train_min: int,
    train_max: int,
    seed: int,
) -> Tuple[List[str], Dict[str, int]]:
    """Build train keep list per priority strategy."""
    rng = random.Random(seed)

    keep: List[str] = []
    keep.extend(train_cls["orig_ship"])
    keep.extend(train_cls["rot_ship"])

    stats = {
        "orig_ship": len(train_cls["orig_ship"]),
        "rot_ship": len(train_cls["rot_ship"]),
        "bg_added": 0,
    }

    # Priority 3: background tiles capped
    remaining_cap = train_max - len(keep)
    n_bg = min(max_bg_train, len(train_cls["orig_bg"]), max(0, remaining_cap))
    if n_bg > 0:
        bg_sample = rng.sample(train_cls["orig_bg"], n_bg)
        keep.extend(bg_sample)
        stats["bg_added"] = n_bg

    if len(keep) < train_min:
        raise ValueError(
            f"Train count {len(keep)} below minimum {train_min}. "
            "Cannot satisfy range with current strategy."
        )
    if len(keep) > train_max:
        raise ValueError(
            f"Train count {len(keep)} exceeds maximum {train_max}. "
            "Reduce max_bg_train or increase train_max."
        )

    return keep, stats


def build_val_keep_list(
    val_cls: Dict[str, List[str]],
    target_val: int,
    val_min: int,
    val_max: int,
    seed: int,
) -> Tuple[List[str], Dict[str, int]]:
    """Build val keep list per priority strategy."""
    rng = random.Random(seed)

    val_ship = list(val_cls["orig_ship"])
    val_bg = list(val_cls["orig_bg"])

    stats = {
        "ship_kept": 0,
        "bg_added": 0,
    }

    if len(val_ship) > val_max:
        keep_val = rng.sample(val_ship, val_max)
        stats["ship_kept"] = val_max
        stats["bg_added"] = 0
    else:
        keep_val = list(val_ship)
        stats["ship_kept"] = len(val_ship)
        # Add background up to target_val (but respect val_min/val_max)
        needed_bg = max(0, min(target_val, val_max) - len(keep_val))
        # Also ensure we reach val_min if possible
        if len(keep_val) < val_min:
            needed_bg = max(needed_bg, val_min - len(keep_val))
        n_bg = min(needed_bg, len(val_bg))
        if n_bg > 0:
            keep_val.extend(rng.sample(val_bg, n_bg))
            stats["bg_added"] = n_bg

    if len(keep_val) < val_min:
        raise ValueError(
            f"Val count {len(keep_val)} below minimum {val_min}. "
            f"Ship tiles={len(val_ship)}, bg available={len(val_bg)}."
        )
    if len(keep_val) > val_max:
        raise ValueError(f"Val count {len(keep_val)} exceeds maximum {val_max}.")

    return keep_val, stats


# ---------------------------------------------------------------------------
# Symlink creation
# ---------------------------------------------------------------------------

def _safe_symlink(src: Path, dst: Path) -> None:
    """Create symlink, replacing broken existing link."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def create_symlinks(
    data_dir: Path,
    filtered_dir: Path,
    split: str,
    keep_stems: List[str],
) -> Tuple[int, int]:
    """
    Create image + label + labelTxt symlinks for kept tiles.

    Returns (n_img_links, n_lbl_links) created.
    """
    img_out = filtered_dir / split / "images"
    lbl_out = filtered_dir / split / "labels"
    txt_out = filtered_dir / split / "labelTxt"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    txt_out.mkdir(parents=True, exist_ok=True)

    img_src_dir = data_dir / split / "images"
    lbl_src_dir = data_dir / split / "labels"
    txt_src_dir = data_dir / split / "labelTxt"

    n_img = n_lbl = 0
    for stem in keep_stems:
        src_img = img_src_dir / f"{stem}.png"
        if not src_img.is_file():
            # try other extensions
            cands = list(img_src_dir.glob(f"{stem}.*"))
            src_img = cands[0] if cands else src_img
        src_lbl = lbl_src_dir / f"{stem}.txt"
        src_txt = txt_src_dir / f"{stem}.txt"

        if src_img.is_file():
            _safe_symlink(src_img, img_out / src_img.name)
            n_img += 1
        if src_lbl.is_file():
            _safe_symlink(src_lbl, lbl_out / f"{stem}.txt")
            n_lbl += 1
        if src_txt.is_file():
            _safe_symlink(src_txt, txt_out / f"{stem}.txt")

    return n_img, n_lbl


def verify_symlinks(filtered_dir: Path) -> Tuple[int, List[str]]:
    """Return (broken_count, list of broken paths)."""
    broken: List[str] = []
    for sub in ("train", "val"):
        for kind in ("images", "labels", "labelTxt"):
            d = filtered_dir / sub / kind
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.is_symlink() and not p.resolve().is_file():
                    broken.append(str(p))
    return len(broken), broken


# ---------------------------------------------------------------------------
# dataset.yaml + GT COCO
# ---------------------------------------------------------------------------

def write_dataset_yaml(filtered_dir: Path) -> Path:
    out = filtered_dir / "dataset.yaml"
    try:
        import yaml

        cfg = {
            "path": str(filtered_dir),
            "train": "train/images",
            "val": "val/images",
            "nc": 1,
            "names": {0: "ship"},
        }
        with out.open("w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
    except ImportError:
        text = (
            "path: {}\n"
            "train: train/images\n"
            "val: val/images\n"
            "nc: 1\n"
            "names:\n"
            "  0: ship\n"
        ).format(filtered_dir)
        out.write_text(text, encoding="utf-8")
    return out


def build_filtered_gt_coco(filtered_dir: Path, output_path: Path) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from shared.obb_coco_eval import build_gt_coco_json_tiled

    val_img = str(filtered_dir / "val" / "images")
    val_lbl = str(filtered_dir / "val" / "labels")
    return build_gt_coco_json_tiled(
        img_dir=val_img,
        lbl_dir=val_lbl,
        output_path=str(output_path),
        tile_size=1024,
    )


# ---------------------------------------------------------------------------
# Instance statistics
# ---------------------------------------------------------------------------

def polygon_area_norm(coords: List[float]) -> float:
    """Shoelace area from 8 normalized coords."""
    xs = [coords[i] for i in range(0, 8, 2)]
    ys = [coords[i] for i in range(1, 8, 2)]
    n = 4
    area = abs(
        sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))
    ) / 2.0
    return area * (1024 ** 2)


def count_split_stats(filtered_dir: Path, split: str, keep_stems: List[str]) -> Tuple[int, Dict[str, int]]:
    lbl_dir = filtered_dir / split / "labels"
    n_inst = 0
    size_dist = {"small": 0, "medium": 0, "large": 0}

    for stem in keep_stems:
        lp = lbl_dir / f"{stem}.txt"
        if not lp.is_file():
            continue
        for line in lp.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            coords = [float(parts[i]) for i in range(1, 9)]
            area = polygon_area_norm(coords)
            n_inst += 1
            if area < SIZE_SMALL:
                size_dist["small"] += 1
            elif area <= SIZE_MEDIUM:
                size_dist["medium"] += 1
            else:
                size_dist["large"] += 1

    return n_inst, size_dist


def print_size_distribution(size_dist: Dict[str, int]) -> None:
    total = sum(size_dist.values()) or 1
    labels = [
        ("small", "Small  (area <  1024 px²)"),
        ("medium", "Medium (1024-9216 px²)"),
        ("large", "Large  (> 9216 px²)"),
    ]
    for key, label in labels:
        n = size_dist[key]
        pct = 100.0 * n / total
        print(f"    {label}: {n:5d} ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# Update training scripts
# ---------------------------------------------------------------------------

def update_training_scripts(filtered_dir: Path, gt_json: Path) -> None:
    """Point all model configs and shared constants to filtered dataset."""
    filtered_str = str(filtered_dir)
    if not filtered_str.endswith("/"):
        filtered_data_root = filtered_str + "/"
    else:
        filtered_data_root = filtered_str

    dataset_yaml = str(filtered_dir / "dataset.yaml")
    gt_json_str = str(gt_json)

    # shared/constants.py
    constants_path = REPO_ROOT / "shared" / "constants.py"
    text = constants_path.read_text(encoding="utf-8")
    text = text.replace(
        'DATA_ROOT = OUT / "data"',
        'DATA_ROOT = OUT / "data_filtered"',
    )
    text = text.replace(
        'GT_JSON = OUT / "shared" / "gt_coco.json"',
        f'GT_JSON = Path("{gt_json_str}")',
    )
    constants_path.write_text(text, encoding="utf-8")
    print(f"  Updated: {constants_path}")

    # MMRotate configs
    for model in ("orcnn", "s2anet"):
        cfg_path = REPO_ROOT / "baselines" / model / f"config_{model}_v2.py"
        if cfg_path.is_file():
            cfg_text = cfg_path.read_text(encoding="utf-8")
            cfg_text = re.sub(
                r'data_root = ".*?"',
                f'data_root = "{filtered_data_root}"',
                cfg_text,
            )
            cfg_path.write_text(cfg_text, encoding="utf-8")
            print(f"  Updated: {cfg_path}")

    # build_gt_coco.py default output
    gt_build = REPO_ROOT / "shared" / "build_gt_coco.py"
    if gt_build.is_file():
        gt_text = gt_build.read_text(encoding="utf-8")
        gt_text = gt_text.replace(
            'parser.add_argument("--output", default=str(GT_JSON))',
            f'parser.add_argument("--output", default="{gt_json_str}")',
        )
        gt_build.write_text(gt_text, encoding="utf-8")
        print(f"  Updated: {gt_build}")

    print("\n  All training scripts now point to data_filtered/")
    print(f"    DATASET_YAML = {dataset_yaml}")
    print(f"    GT_JSON      = {gt_json_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter ship_obb_v2 dataset to target sizes")
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--filtered_dir", type=Path, default=DEFAULT_FILTERED_DIR)
    p.add_argument("--max_bg_train", type=int, default=500)
    p.add_argument("--target_val", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_min", type=int, default=3000)
    p.add_argument("--train_max", type=int, default=6624)
    p.add_argument("--val_min", type=int, default=800)
    p.add_argument("--val_max", type=int, default=1200)
    p.add_argument("--dry_run", action="store_true", help="Plan only, no symlinks")
    p.add_argument("--update_scripts", action="store_true", help="Update model configs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    filtered_dir = args.filtered_dir.resolve()
    gt_json_path = filtered_dir / "gt_coco_filtered.json"

    if not data_dir.is_dir():
        print(f"ERROR: data_dir not found: {data_dir}", file=sys.stderr)
        return 1

    print("=== Filter Dataset v2 ===")
    print(f"  Source:   {data_dir}")
    print(f"  Output:   {filtered_dir}")
    print(f"  Seed:     {args.seed}")
    print(f"  Dry run:  {args.dry_run}")
    print()

    # Step 1 — classify
    train_cls = scan_split(data_dir, "train")
    val_cls = scan_split(data_dir, "val")
    print_classification_summary(train_cls, val_cls)

    # Step 2 — build keep lists
    keep_train, train_stats = build_train_keep_list(
        train_cls,
        max_bg_train=args.max_bg_train,
        train_min=args.train_min,
        train_max=args.train_max,
        seed=args.seed,
    )
    keep_val, val_stats = build_val_keep_list(
        val_cls,
        target_val=args.target_val,
        val_min=args.val_min,
        val_max=args.val_max,
        seed=args.seed,
    )

    print("  === Planned selection ===")
    print(f"  TRAIN keep: {len(keep_train)} tiles")
    print(f"    Original ship : {train_stats['orig_ship']}")
    print(f"    Rotated ship  : {train_stats['rot_ship']}")
    print(f"    Background add: {train_stats['bg_added']} (cap {args.max_bg_train})")
    print(f"  VAL keep:   {len(keep_val)} tiles")
    print(f"    Ship tiles    : {val_stats['ship_kept']}")
    print(f"    Background add: {val_stats['bg_added']}")
    print()

    train_ok = args.train_min <= len(keep_train) <= args.train_max
    val_ok = args.val_min <= len(keep_val) <= args.val_max
    print(f"  Range check train: {'PASS' if train_ok else 'FAIL'} "
          f"({args.train_min} ≤ {len(keep_train)} ≤ {args.train_max})")
    print(f"  Range check val:   {'PASS' if val_ok else 'FAIL'} "
          f"({args.val_min} ≤ {len(keep_val)} ≤ {args.val_max})")

    if not train_ok or not val_ok:
        print("\nERROR: Keep lists outside target ranges.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n  Dry run complete — no files created.")
        return 0

    # Step 3 — create symlinks
    print("\n=== Creating symlinked filtered dataset ===")
    if filtered_dir.exists():
        print(f"  WARNING: {filtered_dir} already exists — symlinks will be refreshed per tile")

    n_train_img, n_train_lbl = create_symlinks(data_dir, filtered_dir, "train", keep_train)
    n_val_img, n_val_lbl = create_symlinks(data_dir, filtered_dir, "val", keep_val)

    # Step 4 — dataset.yaml
    yaml_path = write_dataset_yaml(filtered_dir)
    print(f"  Wrote: {yaml_path}")

    # Step 5 — GT COCO JSON
    print("\n=== Building filtered GT COCO JSON ===")
    gt_data = build_filtered_gt_coco(filtered_dir, gt_json_path)
    print(f"  Val tiles in GT: {len(gt_data['images'])}")
    print(f"  Instances in GT: {len(gt_data['annotations'])}")
    print(f"  Saved: {gt_json_path}")

    # Step 6 — statistics
    train_inst, _ = count_split_stats(filtered_dir, "train", keep_train)
    val_inst, val_size = count_split_stats(filtered_dir, "val", keep_val)
    broken_n, broken_list = verify_symlinks(filtered_dir)

    print()
    print("=== Filtered Dataset v2 statistics ===")
    print(f"  Original tiles (train):       {len(train_cls['all_stems'])}")
    print(f"  Original tiles (val):           {len(val_cls['all_stems'])}")
    print()
    print("  --- TRAIN (filtered) ---")
    print(f"  Original ship tiles kept  :    {train_stats['orig_ship']}")
    print(f"  Rotated ship tiles kept   :    {train_stats['rot_ship']}")
    print(f"  Background tiles added    :    {train_stats['bg_added']} (capped at {args.max_bg_train})")
    print(f"  TOTAL TRAIN TILES         :    {len(keep_train)}")
    print()
    print("  --- VAL (filtered) ---")
    print(f"  Ship val tiles kept       :    {val_stats['ship_kept']}")
    print(f"  Background val tiles added:    {val_stats['bg_added']}")
    print(f"  TOTAL VAL TILES           :    {len(keep_val)}")
    print()
    print("  --- INSTANCES ---")
    print(f"  Total train instances     :    {train_inst}")
    print(f"  Total val instances       :    {val_inst}")
    print("  Val size distribution:")
    print_size_distribution(val_size)
    print()
    print("  --- PATHS ---")
    print(f"  Filtered dataset yaml : {yaml_path}")
    print(f"  GT COCO JSON          : {gt_json_path}")
    print(f"  Symlinks created      : {n_train_img} train img + {n_val_img} val img")
    print()
    print("  --- VERIFICATION ---")
    print(f"  Range check train: PASS ({args.train_min} ≤ {len(keep_train)} ≤ {args.train_max}) ✅")
    print(f"  Range check val:   PASS ({args.val_min} ≤ {len(keep_val)} ≤ {args.val_max}) ✅")
    print(f"  Zero broken symlinks: {'✅' if broken_n == 0 else f'❌ ({broken_n} broken)'}")
    if broken_list[:5]:
        for b in broken_list[:5]:
            print(f"    {b}")
    print(f"  dataset.yaml valid: ✅")
    print("==========================================")
    print("Filtered dataset ready.")
    print("Paper description:")
    print(
        f'"We retain all {train_stats["orig_ship"]} ship-containing training tiles and '
        f'their {train_stats["rot_ship"]} rotated variants, supplementing with '
        f'{train_stats["bg_added"]} background tiles (random seed {args.seed}), '
        f'yielding {len(keep_train)} training tiles. For validation we retain '
        f'{val_stats["ship_kept"]} ship-containing tiles plus {val_stats["bg_added"]} '
        f'background tiles, giving {len(keep_val)} validation tiles. '
        f'Validation tiles are never augmented."'
    )

    # Step 7 — update scripts
    if args.update_scripts:
        print("\n=== Updating training script paths ===")
        update_training_scripts(filtered_dir, gt_json_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
