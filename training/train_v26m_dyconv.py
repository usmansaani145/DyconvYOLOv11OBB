#!/usr/bin/env python3
"""
Idea 6: YOLOv26m-OBB + DyConv

Baseline YOLOv26m AP = 0.411 ± 0.001
DyConv on YOLOv11m  AP = 0.408 ± 0.001 (+0.007 vs v11 baseline)
Target: AP > 0.415 to clearly beat all models
"""

from __future__ import annotations

import csv
import glob
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

IDEA_DIR = Path("./ideas/idea6_v26m_dyconv")
REPO_DIR = Path(".")
# PYDEPS removed — use pip install

# IDEA_DIR must come first so models.inject_dyconv resolves
# to idea6_v26m_dyconv/models/ not ship_obb_v2/models/

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(IDEA_DIR))  # MUST be last insert = first in path

# Verify correct models package is loaded
spec = importlib.util.find_spec("models.inject_dyconv")
assert spec is not None, f"models.inject_dyconv not found — sys.path: {sys.path[:5]}"
assert "idea6_v26m_dyconv" in spec.origin, f"Wrong models package loaded: {spec.origin}"
print(f"models.inject_dyconv resolved to: {spec.origin}")

sys.path.insert(1, str(REPO_DIR / "ideas"))

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import torch
import torch.optim as optim
from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

from models.inject_dyconv import inject_dyconv, inject_dyconv_module  # noqa: E402
from idea_train_callbacks import (  # noqa: E402
    make_grad_clip_callback,
    make_nan_stop_callback,
    rebuild_ema,
    sync_model_device,
)
from shared.obb_coco_eval import run_coco_eval  # noqa: E402
from shared.yolo_label_cache import ensure_yolo_label_caches  # noqa: E402

SEED = int(sys.argv[1])
MODEL_NAME = "idea6_v26m_dyconv"
PRETRAINED = "pretrained/yolo26m-obb.pt"
DATA_YAML = "dataset/dataset.yaml"
GT_JSON = "dataset/gt_coco_filtered.json"
OUTPUT_DIR = Path(f"runs/{MODEL_NAME}/{MODEL_NAME}_seed{SEED}")
BASE_LR = 0.001
GAMMA_LR = 0.01


def _metric_box(results):
    return getattr(results, "obb", None) or results.box


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_gamma_param_groups(model, base_lr: float = BASE_LR, base_wd: float = 0.0005):
    decay_params: List = []
    no_decay_params: List = []
    gamma_params: List = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".gamma") or ".gamma" in name:
            gamma_params.append(param)
        elif "bn" in name.lower() or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "lr": base_lr, "weight_decay": base_wd, "initial_lr": base_lr},
        {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0, "initial_lr": base_lr},
        {"params": gamma_params, "lr": GAMMA_LR, "weight_decay": 0.0, "initial_lr": GAMMA_LR},
    ]


def _apply_gamma_optimizer(trainer) -> None:
    if trainer.optimizer is None:
        return
    base_wd = getattr(trainer.args, "weight_decay", 0.0005)
    groups = build_gamma_param_groups(unwrap_model(trainer.model), BASE_LR, base_wd)
    trainer.optimizer = optim.AdamW(groups, lr=BASE_LR, weight_decay=base_wd)
    print(f"DyConv gamma optimizer: gamma_params={len(groups[2]['params'])}, gamma_lr={GAMMA_LR}")


def _activation_scale_check(model: YOLO) -> None:
    # activation scale check (DyConv-injected layers only)
    DYCONV_TARGET_LAYERS = [3, 5]  # only layers where DyConv was injected

    model.model.cuda().eval()
    dummy = torch.randn(2, 3, 640, 640).cuda()
    acts = {}
    hooks = []
    for i, layer in enumerate(list(model.model.model.children())):
        def make_hook(idx):
            def h(m, inp, out):
                if isinstance(out, torch.Tensor):
                    acts[idx] = out.std().item()
            return h
        hooks.append(layer.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        model.model(dummy)
    for h in hooks:
        h.remove()

    # Print all layers for reference (info only, no assertion)
    print("\nActivation std — all layers (info only):")
    for k, v in sorted(acts.items()):
        tag = " <- DyConv" if k in DYCONV_TARGET_LAYERS else ""
        print(f"  layer {k}: std={v:.4f}{tag}")

    # Assert ONLY on DyConv-injected layers 3 and 5
    # Acceptable range: 0.1 to 100.0
    # Stem layers (0,1) always have high std on random input — not checked
    print("\nActivation scale check (DyConv layers 3 and 5 only):")
    all_ok = True
    for idx in DYCONV_TARGET_LAYERS:
        if idx in acts:
            v = acts[idx]
            ok = 0.1 < v < 100.0
            flag = "OK" if ok else "FAIL"
            if not ok:
                all_ok = False
            print(f"  layer {idx} (DyConv): std={v:.4f} {flag}")
        else:
            print(f"  layer {idx}: not captured in hooks")

    if not all_ok:
        raise RuntimeError(
            "DyConv-injected layers (3,5) failed activation scale check.\n"
            "Expected 0.1 < std < 100.0 for both layers.\n"
            "Check DyConv channel mapping for YOLOv26m backbone."
        )
    print("Activation scale check PASSED\n")
    model.model.cpu()
    torch.cuda.empty_cache()


def _make_gate_log_callback():
    def log_gate_callback(trainer):
        gates = {}
        for name, param in trainer.model.named_parameters():
            if "gamma" in name.lower():
                gates[name] = float(param.detach().cpu().item())
        if not gates:
            return
        epoch = int(trainer.epoch)
        if epoch % 5 == 0 or epoch >= int(trainer.epochs) - 1:
            print(f"[epoch {epoch}] gamma values: {gates}")
        save_dir = Path(getattr(trainer, "save_dir", "."))
        log_path = save_dir / "gate_log.csv"
        write_header = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["epoch"] + list(gates.keys()))
            w.writerow([epoch] + list(gates.values()))
    return log_gate_callback


def eval_metrics(seed: int) -> Dict[str, Any]:
    run_dir = OUTPUT_DIR / "train"
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        best_pt = run_dir / "weights" / "last.pt"
    print(f"Loading: {best_pt}")
    eval_model = YOLO(str(best_pt))

    val_results = eval_model.val(
        data=DATA_YAML,
        imgsz=1024,
        batch=8,
        device=0,
        conf=0.25,
        iou=0.5,
        save_json=True,
        plots=True,
        project=str(OUTPUT_DIR),
        name="val",
        exist_ok=True,
        split="val",
    )

    box = _metric_box(val_results)
    metrics = {
        "AP": float(getattr(box, "map", 0.0) or 0.0),
        "AP50": float(getattr(box, "map50", 0.0) or 0.0),
        "AP75": float(getattr(box, "map75", 0.0) or 0.0),
        "Precision": float(getattr(box, "mp", 0.0) or 0.0),
        "Recall": float(getattr(box, "mr", 0.0) or 0.0),
    }

    pred_jsons = sorted(
        glob.glob(str(OUTPUT_DIR / "val" / "**" / "predictions.json"), recursive=True)
        + glob.glob(str(OUTPUT_DIR / "val" / "*.json"))
    )
    if pred_jsons:
        metrics.update(run_coco_eval(GT_JSON, pred_jsons[0]))

    metrics_path = OUTPUT_DIR / f"metrics_{MODEL_NAME}_seed{seed}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved: {metrics_path}")
    return metrics


def main() -> int:
    assert torch.cuda.is_available(), "CUDA not available"
    set_seed(SEED)
    ensure_yolo_label_caches()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Seed: {SEED} | Model: {MODEL_NAME}")

    model = YOLO(PRETRAINED)
    params_before = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"Pretrained params (before injection): {params_before:.1f}M")
    assert params_before > 20, f"Wrong variant ({params_before:.1f}M) — expected medium >20M"

    model = inject_dyconv(model, verbose=True)
    params_after = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"Pretrained params (after injection): {params_after:.1f}M")
    _activation_scale_check(model)

    def on_train_start(trainer):
        print("\n=== on_train_start: re-inject DyConv + gamma optimizer ===")
        inject_dyconv_module(trainer.model, verbose=True)
        sync_model_device(trainer.model, trainer.device)
        rebuild_ema(trainer)
        _apply_gamma_optimizer(trainer)
        for name, param in trainer.model.named_parameters():
            if "gamma" in name.lower():
                print(f"  initial {name}: {param.item():.6f}")

    gate_cb = _make_gate_log_callback()
    model.add_callback("on_after_backward", make_grad_clip_callback(max_norm=10.0))
    model.add_callback("on_train_batch_end", make_nan_stop_callback())
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", gate_cb)

    model.train(
        data=DATA_YAML,
        imgsz=1024,
        epochs=100,
        batch=8,
        device=0,
        patience=0,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        seed=SEED,
        pretrained=False,
        optimizer="AdamW",
        lr0=BASE_LR,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.0,
        copy_paste=0.0,
        amp=True,
        save_period=10,
        workers=4,
        conf=0.25,
        iou=0.5,
        save_json=True,
        plots=True,
    )
    print("Training complete.")

    metrics = eval_metrics(SEED)

    comparisons = {
        "YOLOv11m baseline": 0.401,
        "YOLOv26m baseline": 0.411,
        "DyConv-YOLOv11m": 0.408,
        f"{MODEL_NAME} seed{SEED}": metrics.get("AP", 0),
    }
    print(f"\n{'=' * 60}")
    print(f"  Idea 6: YOLOv26m + DyConv | seed={SEED}")
    print(f"{'=' * 60}")
    for name, ap in comparisons.items():
        marker = " <- THIS RUN" if MODEL_NAME in name else ""
        print(f"  {name:<30}: AP = {ap:.4f}{marker}")
    print()

    ap = metrics.get("AP", 0)
    if ap > 0.415:
        print("  PROMOTE — beats all models (AP > 0.415)")
        print("     -> Submit seeds 123 and 456")
    elif ap > 0.411:
        print("  MARGINAL — above YOLOv26m baseline but small gain")
        print("     -> Discuss before running seeds 123/456")
    elif ap > 0.408:
        print("  NEUTRAL — between DyConv-v11 and YOLOv26m baseline")
        print("     -> DyConv does not add much to YOLOv26m")
    else:
        print("  DROP — below YOLOv26m baseline")
        print("     -> DyConv hurts YOLOv26m, do not proceed")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
