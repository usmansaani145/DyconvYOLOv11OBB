#!/usr/bin/env python3
"""Unified COCO GT build + COCOeval for all ship_obb_unified models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def _poly_area(xs: List[float], ys: List[float]) -> float:
    n = 4
    return abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))) / 2.0


def _poly_to_bbox_xywh(poly: List[float]) -> List[float]:
    xs = poly[0::2]
    ys = poly[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _yolo_obb_line_to_pixels(parts: List[str], tile_size: int) -> Tuple[List[float], List[float]]:
    coords = [float(v) for v in parts[1:9]]
    xs = [coords[i] * tile_size for i in range(0, 8, 2)]
    ys = [coords[i] * tile_size for i in range(1, 8, 2)]
    return xs, ys


def build_gt_coco_json_tiled(
    img_dir: str,
    lbl_dir: str,
    output_path: str,
    tile_size: int = 1024,
    class_names: Optional[List[str]] = None,
) -> dict:
    """Build COCO GT from YOLO OBB tiles (fixed tile_size, no PIL read)."""
    class_names = class_names or ["ship"]
    images_list: List[dict] = []
    annotations_list: List[dict] = []
    ann_id = 1

    img_dir_p = Path(img_dir)
    lbl_dir_p = Path(lbl_dir)

    img_files = sorted(
        f for f in img_dir_p.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    )

    for img_id, img_path in enumerate(img_files, start=1):
        w = h = tile_size
        images_list.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        lbl_path = lbl_dir_p / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            continue

        with open(lbl_path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 9:
                continue
            xs, ys = _yolo_obb_line_to_pixels(parts, tile_size)
            area = _poly_area(xs, ys)
            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min

            annotations_list.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x_min, y_min, bbox_w, bbox_h],
                "area": area,
                "segmentation": [[
                    xs[0], ys[0], xs[1], ys[1],
                    xs[2], ys[2], xs[3], ys[3],
                ]],
                "iscrowd": 0,
            })
            ann_id += 1

    gt_dict = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": [{"id": 1, "name": class_names[0]}],
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gt_dict), encoding="utf-8")
    _print_area_stats(len(images_list), annotations_list)
    return gt_dict


def _print_area_stats(n_images: int, annotations: List[dict]) -> None:
    areas = [a["area"] for a in annotations]
    total = len(areas)
    small = sum(1 for a in areas if a < 1024)
    medium = sum(1 for a in areas if 1024 <= a < 9216)
    large = sum(1 for a in areas if a >= 9216)
    print(f"GT built: {n_images} images, {total} instances")
    print(f"  Small  (area <  1024): {small}  ({100 * small / max(total, 1):.1f}%)")
    print(f"  Medium (1024-9216):    {medium} ({100 * medium / max(total, 1):.1f}%)")
    print(f"  Large  (area > 9216):  {large}  ({100 * large / max(total, 1):.1f}%)")


def _min_area_rect_numpy(corners: np.ndarray) -> tuple[float, float, float, float, float]:
    """Min-area rectangle via rotating calipers on polygon edges (no OpenCV)."""
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    min_area = float("inf")
    best: tuple[float, float, float, float, float] | None = None

    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]
        edge = p2 - p1
        angle = math.atan2(float(edge[1]), float(edge[0]))
        c, s = math.cos(angle), math.sin(angle)
        rot_x = pts[:, 0] * c + pts[:, 1] * s
        rot_y = -pts[:, 0] * s + pts[:, 1] * c
        x_min, x_max = float(rot_x.min()), float(rot_x.max())
        y_min, y_max = float(rot_y.min()), float(rot_y.max())
        w = x_max - x_min
        h = y_max - y_min
        area = w * h
        if area < min_area:
            min_area = area
            cx_l = (x_min + x_max) / 2.0
            cy_l = (y_min + y_max) / 2.0
            cx = cx_l * c - cy_l * s
            cy = cx_l * s + cy_l * c
            best = (float(cx), float(cy), float(w), float(h), math.degrees(angle))

    if best is None:
        raise ValueError("Cannot compute min-area rect for empty polygon")
    return best


def _min_area_rect(corners: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return (cx, cy, w, h, angle_deg) matching OpenCV minAreaRect convention."""
    if cv2 is not None:
        (cx, cy), (w, h), angle_deg = cv2.minAreaRect(corners.astype(np.float32))
        return float(cx), float(cy), float(w), float(h), float(angle_deg)
    cx, cy, w, h, angle_deg = _min_area_rect_numpy(corners)
    # Align with OpenCV: width is longer side, angle in [-90, 0)
    if w < h:
        w, h = h, w
        angle_deg += 90.0
    while angle_deg > 0:
        angle_deg -= 90.0
    while angle_deg <= -90:
        angle_deg += 90.0
    return cx, cy, w, h, angle_deg


def polygon_to_rotated_bbox_le90(corners: np.ndarray) -> List[float]:
    """4x2 corners -> [cx, cy, w, h, angle_rad] le90 convention."""
    cx, cy, w, h, angle_deg = _min_area_rect(corners)
    angle_rad = angle_deg * math.pi / 180.0
    return [float(cx), float(cy), float(w), float(h), float(angle_rad)]


def polygon_to_rotated_bbox_le135(corners: np.ndarray) -> List[float]:
    """4x2 corners -> [cx, cy, w, h, angle_rad] le135 convention."""
    cx, cy, w, h, angle_deg = _min_area_rect(corners)
    if w < h:
        w, h = h, w
        angle_deg += 90.0
    if angle_deg > 45.0:
        angle_deg -= 90.0
    angle_rad = angle_deg * math.pi / 180.0
    return [float(cx), float(cy), float(w), float(h), float(angle_rad)]


def _build_image_id_lookup(gt_data: dict) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for img in gt_data["images"]:
        iid = int(img["id"])
        fname = img["file_name"]
        stem = Path(fname).stem
        lookup[fname] = iid
        lookup[stem] = iid
        lookup[str(iid)] = iid
    return lookup


def convert_ultralytics_preds_to_coco_dt(
    dt_raw: List[dict],
    gt_json_path: str,
) -> List[dict]:
    """Remap Ultralytics OBB predictions to integer COCO dt [x,y,w,h]."""
    gt_data = json.loads(Path(gt_json_path).read_text(encoding="utf-8"))
    lookup = _build_image_id_lookup(gt_data)

    dt_fixed: List[dict] = []
    skipped = 0

    for pred in dt_raw:
        raw_id = pred.get("image_id")
        fname = pred.get("file_name", "")

        if fname:
            key = fname if fname in lookup else Path(str(fname)).stem
        elif raw_id is not None:
            key = str(raw_id)
        else:
            skipped += 1
            continue

        image_id = lookup.get(key) or lookup.get(Path(str(key)).stem)
        if image_id is None:
            skipped += 1
            continue

        if "bbox" in pred and len(pred["bbox"]) >= 4:
            bb = pred["bbox"]
            if len(bb) == 5:
                cx, cy, w, h, _ = [float(v) for v in bb]
                bbox = [cx - w / 2, cy - h / 2, w, h]
            else:
                bbox = [float(v) for v in bb[:4]]
        elif "poly" in pred and pred["poly"] and len(pred["poly"]) >= 8:
            bbox = _poly_to_bbox_xywh(pred["poly"])
        elif "rbox" in pred and pred["rbox"] and len(pred["rbox"]) >= 5:
            cx, cy, w, h = [float(v) for v in pred["rbox"][:4]]
            bbox = [cx - w / 2, cy - h / 2, w, h]
        else:
            skipped += 1
            continue

        w, h = bbox[2], bbox[3]
        if w <= 0 or h <= 0:
            skipped += 1
            continue

        cat_id = int(pred.get("category_id", 1))
        if cat_id <= 0:
            cat_id = 1

        dt_fixed.append({
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": bbox,
            "score": float(pred.get("score", 1.0)),
        })

    if skipped:
        print(f"  WARNING: Skipped {skipped} predictions")
    print(f"  Predictions for COCOeval: {len(dt_fixed)}")
    return dt_fixed


def obb_le90_to_hbb_xywh(cx: float, cy: float, w: float, h: float, angle: float) -> List[float]:
    """MMRotate le90 OBB -> COCO axis-aligned [x, y, w, h]."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    half_w_x = (w / 2.0) * cos_a
    half_w_y = (w / 2.0) * sin_a
    half_h_x = -(h / 2.0) * sin_a
    half_h_y = (h / 2.0) * cos_a
    xs = [
        cx + half_w_x + half_h_x,
        cx + half_w_x - half_h_x,
        cx - half_w_x - half_h_x,
        cx - half_w_x + half_h_x,
    ]
    ys = [
        cy + half_w_y + half_h_y,
        cy + half_w_y - half_h_y,
        cy - half_w_y - half_h_y,
        cy - half_w_y + half_h_y,
    ]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _dota_val_stems(label_dir: Path) -> List[str]:
    """Non-empty label stems in MMRotate DOTADataset order (glob.glob + skip empty)."""
    import glob

    stems: List[str] = []
    for ann_file in glob.glob(str(label_dir / "*.txt")):
        path = Path(ann_file)
        if path.stat().st_size > 0:
            stems.append(path.stem)
    return stems


def _nonempty_label_stems(label_dir: Path) -> List[str]:
    return _dota_val_stems(label_dir)


def obb_to_hbb_xywh_rows(obb: np.ndarray, angle_version: str = "le90") -> np.ndarray:
    """Convert Nx5 OBB rows to Nx4 COCO xywh using MMRotate when available."""
    if obb.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    try:
        import torch
        import mmrotate.core as mc

        obb_t = torch.from_numpy(np.asarray(obb, dtype=np.float32))
        hbb = mc.obb2hbb(obb_t, version=angle_version).cpu().numpy()
        hbb = hbb[..., :4]
        xc, yc, w, h = np.split(hbb, 4, axis=-1)
        w = np.abs(w)
        h = np.abs(h)
        x = xc - w / 2.0
        y = yc - h / 2.0
        return np.concatenate([x, y, w, h], axis=-1)
    except Exception:
        out = np.zeros((obb.shape[0], 4), dtype=np.float64)
        for i, row in enumerate(obb):
            cx, cy, w, h, angle = [float(v) for v in row[:5]]
            out[i] = obb_le90_to_hbb_xywh(cx, cy, w, h, angle)
        return out


def convert_mmrotate_pkl_to_coco_dt(
    pkl_path: str,
    gt_json_path: str,
    label_dir: str,
    angle_version: str = "le90",
    bbox_json_path: Optional[str] = None,
) -> List[dict]:
    """Convert MMRotate test.py results.pkl to COCO detection list."""
    import pickle

    gt_data = json.loads(Path(gt_json_path).read_text(encoding="utf-8"))
    lookup = _build_image_id_lookup(gt_data)
    stems = _dota_val_stems(Path(label_dir))

    with open(pkl_path, "rb") as f:
        outputs = pickle.load(f)

    if len(outputs) != len(stems):
        raise ValueError(
            f"results.pkl length {len(outputs)} != DOTADataset val tiles {len(stems)}"
        )

    dt_list: List[dict] = []
    bbox_list: List[dict] = []
    for stem, per_img in zip(stems, outputs):
        image_id = lookup.get(stem)
        if image_id is None:
            continue
        if not isinstance(per_img, list):
            per_img = [per_img]
        for cls_idx, cls_dets in enumerate(per_img):
            if cls_dets is None:
                continue
            arr = np.asarray(cls_dets)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] < 6:
                raise ValueError(f"Unexpected det shape {arr.shape} for {stem}")
            obb = arr[:, :5]
            scores = arr[:, 5]
            xywh = obb_to_hbb_xywh_rows(obb, angle_version=angle_version)
            for obb_row, bbox, score in zip(obb, xywh, scores):
                cx, cy, ow, oh, angle = [float(v) for v in obb_row[:5]]
                x, y, w, h = [float(v) for v in bbox]
                if w <= 0 or h <= 0:
                    continue
                cat_id = 1 if cls_idx == 0 else cls_idx + 1
                if cat_id <= 0:
                    cat_id = 1
                bbox_list.append({
                    "image_id": int(image_id),
                    "category_id": int(cat_id),
                    "bbox": [cx, cy, ow, oh, angle],
                    "score": float(score),
                })
                dt_list.append({
                    "image_id": int(image_id),
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "score": float(score),
                })

    if bbox_json_path:
        out_bbox = Path(bbox_json_path)
        out_bbox.parent.mkdir(parents=True, exist_ok=True)
        out_bbox.write_text(json.dumps(bbox_list), encoding="utf-8")
        print(f"  OBB predictions: {len(bbox_list)} -> {out_bbox}")

    print(f"  Predictions from pkl: {len(dt_list)} boxes on {len(stems)} images")
    return dt_list


def convert_mmrotate_preds_to_coco_dt(
    dt_raw: List[dict],
    gt_json_path: str,
) -> List[dict]:
    """MMRotate format-only output -> COCO dt [x,y,w,h]."""
    return convert_ultralytics_preds_to_coco_dt(dt_raw, gt_json_path)


def _iou_xywh(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _greedy_tp_fp_at_threshold(
    dt_list: List[dict],
    gt_by_image: Dict[int, List[List[float]]],
    score_thr: float,
    iou_thr: float = 0.5,
) -> tuple[int, int]:
    """Greedy one-to-one matching at a fixed score threshold."""
    per_image: Dict[int, List[dict]] = {}
    for pred in dt_list:
        if float(pred["score"]) >= score_thr:
            per_image.setdefault(int(pred["image_id"]), []).append(pred)

    matched_gt: set[tuple[int, int]] = set()
    tp = 0
    fp = 0
    for image_id, preds in per_image.items():
        gt_boxes = gt_by_image.get(image_id)
        if not gt_boxes:
            fp += len(preds)
            continue

        preds.sort(key=lambda p: -float(p["score"]))
        for pred in preds:
            pred_bb = pred["bbox"]
            best_j = -1
            best_iou = 0.0
            for j, gt_bb in enumerate(gt_boxes):
                key = (image_id, j)
                if key in matched_gt:
                    continue
                iou = _iou_xywh(pred_bb, gt_bb)
                if iou > best_iou:
                    best_j = j
                    best_iou = iou

            if best_j >= 0 and best_iou > iou_thr:
                matched_gt.add((image_id, best_j))
                tp += 1
            else:
                fp += 1

    return tp, fp


def compute_f1_optimal_precision_recall(
    gt_json_path: str,
    dt_list: List[dict],
    iou_thr: float = 0.5,
    conf_start: float = 0.05,
    conf_end: float = 0.95,
    conf_step: float = 0.05,
) -> Dict[str, float]:
    """
    Sweep confidence thresholds; pick the threshold with highest F1 (IoU > iou_thr).
    Returns Precision, Recall, and the selected threshold.
    """
    gt_data = json.loads(Path(gt_json_path).read_text(encoding="utf-8"))
    gt_by_image: Dict[int, List[List[float]]] = {}
    for ann in gt_data.get("annotations", []):
        gt_by_image.setdefault(int(ann["image_id"]), []).append(list(ann["bbox"][:4]))

    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0 or not dt_list:
        return {"Precision": -1.0, "Recall": -1.0, "F1": -1.0, "best_conf_threshold": -1.0}

    # Pre-group predictions by image (sorted once by score).
    preds_by_image: Dict[int, List[dict]] = {}
    for pred in dt_list:
        preds_by_image.setdefault(int(pred["image_id"]), []).append(pred)
    for preds in preds_by_image.values():
        preds.sort(key=lambda p: -float(p["score"]))

    # Precompute IoU matrix per image: ious[image_id][pred_idx][gt_idx]
    ious_by_image: Dict[int, List[List[float]]] = {}
    for image_id, preds in preds_by_image.items():
        gt_boxes = gt_by_image.get(image_id, [])
        if not gt_boxes:
            continue
        ious_by_image[image_id] = [
            [_iou_xywh(pred["bbox"], gt_bb) for gt_bb in gt_boxes]
            for pred in preds
        ]

    best_f1 = -1.0
    best_precision = -1.0
    best_recall = -1.0
    best_thr = conf_start

    thr = conf_start
    while thr <= conf_end + 1e-9:
        matched_gt: set[tuple[int, int]] = set()
        tp = 0
        fp = 0

        for image_id, preds in preds_by_image.items():
            gt_boxes = gt_by_image.get(image_id)
            if not gt_boxes:
                fp += sum(1 for p in preds if float(p["score"]) >= thr)
                continue

            ious = ious_by_image.get(image_id, [])
            for pred_idx, pred in enumerate(preds):
                if float(pred["score"]) < thr:
                    break
                best_j = -1
                best_iou = 0.0
                row = ious[pred_idx]
                for j, iou_val in enumerate(row):
                    key = (image_id, j)
                    if key in matched_gt:
                        continue
                    if iou_val > best_iou:
                        best_j = j
                        best_iou = iou_val

                if best_j >= 0 and best_iou > iou_thr:
                    matched_gt.add((image_id, best_j))
                    tp += 1
                else:
                    fp += 1

        fn = total_gt - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_thr = thr

        thr += conf_step

    return {
        "Precision": float(best_precision),
        "Recall": float(best_recall),
        "F1": float(best_f1),
        "best_conf_threshold": float(best_thr),
    }


def _is_coco_dt_list(dt_raw: Any) -> bool:
    if not isinstance(dt_raw, list) or not dt_raw:
        return False
    sample = dt_raw[0]
    return (
        isinstance(sample, dict)
        and "image_id" in sample
        and "bbox" in sample
        and len(sample["bbox"]) == 4
        and "score" in sample
    )


def run_coco_eval(gt_json_path: str, pred_json_path: str) -> Dict[str, float]:
    """Run pycocotools COCOeval; returns 14 metrics."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    missing = {k: -1.0 for k in [
        "AP", "AP50", "AP75", "Precision", "Recall",
        "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
    ]}

    with open(pred_json_path, encoding="utf-8") as f:
        dt_raw = json.load(f)

    if not dt_raw:
        print("WARNING: Empty predictions JSON")
        return missing

    if isinstance(dt_raw, dict):
        if "annotations" in dt_raw:
            dt_raw = dt_raw["annotations"]
        elif "bbox" in dt_raw:
            dt_raw = [dt_raw]

    if _is_coco_dt_list(dt_raw):
        dt_fixed = []
        for pred in dt_raw:
            cat_id = int(pred.get("category_id", 1))
            if cat_id <= 0:
                cat_id = 1
            dt_fixed.append({
                "image_id": int(pred["image_id"]),
                "category_id": cat_id,
                "bbox": [float(v) for v in pred["bbox"][:4]],
                "score": float(pred["score"]),
            })
        print(f"  Predictions for COCOeval: {len(dt_fixed)} (pre-formatted)")
    else:
        dt_fixed = convert_ultralytics_preds_to_coco_dt(dt_raw, gt_json_path)

    if not dt_fixed:
        return missing

    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(dt_fixed)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.areaRng = [
        [0, 1e10],
        [0, 1024],
        [1024, 9216],
        [9216, 1e10],
    ]
    coco_eval.params.areaRngLbl = ["all", "small", "medium", "large"]
    coco_eval.params.maxDets = [1, 10, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    metrics = {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APs": float(stats[3]),
        "APm": float(stats[4]),
        "APl": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
        "ARs": float(stats[9]),
        "ARm": float(stats[10]),
        "ARl": float(stats[11]),
    }
    pr = compute_f1_optimal_precision_recall(gt_json_path, dt_fixed)
    metrics["Precision"] = pr["Precision"]
    metrics["Recall"] = pr["Recall"]
    metrics["F1"] = pr["F1"]
    metrics["best_conf_threshold"] = pr["best_conf_threshold"]
    print(
        f"  F1-optimal P/R @ conf={pr['best_conf_threshold']:.2f}: "
        f"Precision={pr['Precision']:.4f} Recall={pr['Recall']:.4f} F1={pr['F1']:.4f}"
    )
    return metrics


def run_unified_eval(
    pred_json_path: str,
    gt_json_path: str,
    model_name: str,
    output_json: str,
    ultralytics_box_metrics: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Unified evaluation: Ultralytics val metrics + real COCOeval size metrics.
    Precision/Recall from ultralytics_box_metrics when provided.
    """
    print(f"\n=== Unified eval: {model_name} ===")
    print(f"  preds: {pred_json_path}")
    print(f"  gt:    {gt_json_path}")

    coco_metrics = run_coco_eval(gt_json_path, pred_json_path)

    metrics = dict(coco_metrics)
    if ultralytics_box_metrics:
        # Ultralytics val P/R override COCO F1-optimal values for YOLO models.
        metrics["Precision"] = float(ultralytics_box_metrics.get("Precision", -1))
        metrics["Recall"] = float(ultralytics_box_metrics.get("Recall", -1))
        metrics.pop("F1", None)
        metrics.pop("best_conf_threshold", None)
        if ultralytics_box_metrics.get("AP", -1) > 0:
            metrics["AP_ultralytics"] = float(ultralytics_box_metrics["AP"])
        if ultralytics_box_metrics.get("AP50", -1) > 0:
            metrics["AP50_ultralytics"] = float(ultralytics_box_metrics["AP50"])

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  Saved: {output_json}")
    for k in ("AP", "AP50", "AP75", "Precision", "Recall", "APs", "APm", "APl"):
        v = metrics.get(k, -1)
        if v >= 0:
            print(f"    {k}: {v:.4f}")
    return metrics


def find_predictions_json(run_dir: Path) -> Optional[Path]:
    run_dir = Path(run_dir).resolve()
    primary = run_dir / "eval" / "val" / "predictions.json"
    if primary.is_file():
        return primary
    for p in sorted(run_dir.glob("eval/**/predictions.json")):
        if p.is_file() and run_dir in p.resolve().parents:
            return p
    fallback = run_dir / "predictions.json"
    return fallback if fallback.is_file() else None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build-gt":
        from shared.constants import GT_JSON, TILE_SIZE, YOLO_VAL_IMG, YOLO_VAL_LBL

        build_gt_coco_json_tiled(YOLO_VAL_IMG, YOLO_VAL_LBL, str(GT_JSON), TILE_SIZE)
    else:
        print("Usage: python3 -m shared.obb_coco_eval build-gt")
