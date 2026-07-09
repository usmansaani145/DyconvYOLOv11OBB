#!/usr/bin/env python3
"""Unified Ultralytics train + val + COCOeval for ship_obb_v2."""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.build_gt_coco import main as build_gt_main
from shared.constants import DATASET_YAML, GT_JSON, PRETRAINED_YOLO11, PRETRAINED_YOLO26, PRETRAINED_YOLO8
from shared.obb_coco_eval import find_predictions_json, run_unified_eval
from shared.yolo_label_cache import ensure_yolo_label_caches


@dataclass
class ExperimentConfig:
    experiment_name: str
    output_dir: str
    report_title: str
    metrics_json_name: str
    build_model: Optional[Callable[[], Any]] = None
    pretrained: str = PRETRAINED_YOLO11
    lr0: float = 0.01
    degrees: float = 5.0
    clip_grad: bool = False
    load_pretrained: bool = False
    seed: int = 42
    eval_only: bool = False
    weights_path: Optional[str] = None
    register_modules: Optional[Callable[[], None]] = None
    train: bool = True
    epochs: int = 100
    resume: bool = False


def ensure_gt_json() -> None:
    gt_path = Path(GT_JSON)
    if not gt_path.is_file():
        print("Building GT COCO json...")
        build_gt_main()
    else:
        print(f"GT json exists: {GT_JSON}")


def _resolve_pretrained(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return str(p)
    if "yolov8" in path.lower():
        raise FileNotFoundError(
            f"YOLOv8 pretrained weights not found: {path}. "
            f"Download yolov8m-obb.pt to {PRETRAINED_YOLO8} before training."
        )
    if "yolo26" in path.lower() and Path(PRETRAINED_YOLO11).is_file():
        print(f"WARNING: {path} missing — using {PRETRAINED_YOLO11}")
        return PRETRAINED_YOLO11
    raise FileNotFoundError(f"Pretrained weights not found: {path}")


def _clear_eval_outputs(run_dir: Path) -> None:
    eval_dir = run_dir / "eval"
    if eval_dir.is_dir():
        shutil.rmtree(eval_dir)
    for stale in run_dir.glob("predictions.json"):
        stale.unlink()


def _prepare_fresh_train(run_dir: Path) -> None:
    for sub in ("weights", "eval"):
        path = run_dir / sub
        if path.is_dir():
            shutil.rmtree(path)
    for metrics in run_dir.glob("metrics_*.json"):
        metrics.unlink()


def _metric_box(results: Any) -> Any:
    return getattr(results, "obb", None) or results.box


def _run_dir(cfg: ExperimentConfig) -> Path:
    return Path(cfg.output_dir) / cfg.experiment_name


def _train_kwargs(cfg: ExperimentConfig, *, resume: bool = False) -> dict:
    return dict(
        data=DATASET_YAML,
        imgsz=1024,
        epochs=cfg.epochs,
        batch=8,
        device=0,
        patience=0,
        resume=resume,
        project=cfg.output_dir,
        name=cfg.experiment_name,
        exist_ok=True,
        seed=cfg.seed,
        deterministic=True,
        lr0=cfg.lr0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=cfg.degrees,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.0,
        copy_paste=0.0,
        erasing=0.0,
        conf=0.25,
        iou=0.5,
        save_json=True,
        plots=True,
        amp=True,
    )


def run_experiment(cfg: ExperimentConfig) -> dict:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        raise RuntimeError("CUDA required")

    ensure_gt_json()
    ensure_yolo_label_caches()
    from ultralytics import YOLO

    run_dir = _run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.eval_only:
        _clear_eval_outputs(run_dir)
        weights = Path(cfg.weights_path or (run_dir / "weights" / "best.pt"))
        if not weights.is_file():
            weights = run_dir / "weights" / "last.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"No weights: {weights}")
        print(f"Eval-only: {weights}")
        if cfg.register_modules is not None:
            cfg.register_modules()
        model = YOLO(str(weights))
    else:
        if cfg.build_model is not None:
            model = cfg.build_model()
            pt = _resolve_pretrained(cfg.pretrained)
            if cfg.load_pretrained and Path(pt).is_file():
                lp = getattr(model, "_load_pretrained_partial", None)
                if callable(lp):
                    lp(pt)
        elif cfg.train:
            if os.environ.get("FRESH_TRAIN", "0") in ("1", "true", "yes"):
                print(f"FRESH_TRAIN=1 — clearing prior artifacts in {run_dir}")
                _prepare_fresh_train(run_dir)
            pretrained = _resolve_pretrained(cfg.pretrained)
            print(f"Pretrained weights: {pretrained}")
            model = YOLO(pretrained)
        else:
            model = YOLO(_resolve_pretrained(cfg.pretrained))

        if cfg.train:
            print(f"\nStarting training (seed={cfg.seed}, epochs={cfg.epochs})...")
            model.train(**_train_kwargs(cfg, resume=cfg.resume))
            print("Training complete.")

        tr = getattr(model, "trainer", None)
        if tr is not None and getattr(tr, "save_dir", None):
            run_dir = Path(tr.save_dir).resolve()

        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"
        weights = best_pt if best_pt.is_file() else last_pt
        print(f"Loading weights: {weights}")
        if cfg.register_modules is not None:
            cfg.register_modules()
        model = YOLO(str(weights))

    if not cfg.eval_only:
        _clear_eval_outputs(run_dir)

    print("\nRunning model.val()...")
    val_results = model.val(
        data=DATASET_YAML,
        imgsz=1024,
        batch=8,
        device=0,
        conf=0.25,
        iou=0.5,
        save_json=True,
        plots=True,
        project=str(run_dir / "eval"),
        name="val",
        exist_ok=True,
        split="val",
    )

    box = _metric_box(val_results)
    ul_metrics = {
        "AP": float(getattr(box, "map", 0.0) or 0.0),
        "AP50": float(getattr(box, "map50", 0.0) or 0.0),
        "AP75": float(getattr(box, "map75", 0.0) or 0.0),
        "Precision": float(getattr(box, "mp", 0.0) or 0.0),
        "Recall": float(getattr(box, "mr", 0.0) or 0.0),
    }

    pred_json = find_predictions_json(run_dir)
    if pred_json is None:
        found = sorted(glob.glob(str(run_dir / "eval" / "**" / "predictions.json"), recursive=True))
        pred_json = Path(found[0]) if found else None

    metrics_path = run_dir / cfg.metrics_json_name
    if pred_json is not None and pred_json.is_file():
        metrics = run_unified_eval(
            str(pred_json),
            str(GT_JSON),
            cfg.report_title,
            str(metrics_path),
            ultralytics_box_metrics=ul_metrics,
        )
    else:
        print("WARNING: No predictions.json — saving Ultralytics metrics only")
        metrics = ul_metrics
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
