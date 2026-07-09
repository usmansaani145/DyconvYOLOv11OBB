#!/usr/bin/env python3
"""
Collect multi-seed results and print paper-ready comparison tables.

Usage:
  python3 results/collect_results.py              # aggregate all models
  python3 results/collect_results.py --print_tables
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared.constants import ALL_MODELS, METRIC_KEYS, RESULTS_DIR, RUNS_DIR, SEEDS
from shared.multi_seed_runner import aggregate_model

DISPLAY_NAMES = {
    "orcnn": "ORCNN",
    "s2anet": "S2ANet",
    "yolov8_obb": "v8-OBB",
    "yolov11_obb": "v11-OBB",
    "yolov26_obb": "v26-OBB",
}

TABLE1_METRICS = [
    ("AP", "AP"),
    ("AP50", "AP50"),
    ("AP75", "AP75"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("APs", "APs (small)"),
    ("APm", "APm (medium)"),
    ("APl", "APl (large)"),
    ("AR1", "AR1"),
    ("AR10", "AR10"),
    ("AR100", "AR100"),
    ("ARs", "ARs (small)"),
    ("ARm", "ARm (medium)"),
    ("ARl", "ARl (large)"),
]


def fmt_mean_std(mean: float, std: float) -> str:
    if mean < 0:
        return "N/A"
    return f"{mean:.3f}±{std:.3f}"


def load_or_aggregate(model: str) -> dict:
    stats_path = RESULTS_DIR / f"{model}_stats.json"
    if stats_path.is_file():
        return json.loads(stats_path.read_text(encoding="utf-8"))
    return aggregate_model(model)


def _seed_metrics_path(model: str, seed: int) -> Optional[Path]:
    run_dir = RUNS_DIR / model / f"{model}_seed{seed}"
    cands = sorted(run_dir.glob("metrics_*.json"))
    return cands[0] if cands else None


def build_table1(all_stats: Dict[str, dict]) -> str:
    cols = ALL_MODELS
    header = (
        "==============================================================================\n"
        "  Ship OBB Detection — DOTA 2.0 (mean ± std, 3 seeds: 42/123/456)\n"
        "  Dataset: ~3,000 train tiles / ~800 val tiles (1024×1024)\n"
        "  Protocol: model.val() conf=0.25 iou=0.5 + pycocotools COCOeval\n"
        "==============================================================================\n"
        f"{'Metric':<12}| " + " | ".join(f"{DISPLAY_NAMES[m]:<14}" for m in cols) + "\n"
        + "-" * 12 + "|" + ("|" + "-" * 16) * len(cols) + "\n"
    )
    rows = []
    for key, label in TABLE1_METRICS:
        cells = []
        for m in cols:
            ms = all_stats[m]["metrics"].get(key, {"mean": -1, "std": -1})
            cells.append(fmt_mean_std(ms["mean"], ms["std"]))
        rows.append(f"{label:<12}| " + " | ".join(f"{c:<14}" for c in cells))

    best_ap = max(cols, key=lambda m: all_stats[m]["metrics"]["AP"]["mean"])
    best_rec = max(cols, key=lambda m: all_stats[m]["metrics"]["Recall"]["mean"])
    best_apl = max(cols, key=lambda m: all_stats[m]["metrics"]["APl"]["mean"])

    ap = all_stats[best_ap]["metrics"]["AP"]
    rec = all_stats[best_rec]["metrics"]["Recall"]
    apl = all_stats[best_apl]["metrics"]["APl"]

    footer = (
        "==============================================================================\n"
        f"Best AP:       {DISPLAY_NAMES[best_ap]} ({ap['mean']:.3f} ± {ap['std']:.3f})\n"
        f"Best Recall:   {DISPLAY_NAMES[best_rec]} ({rec['mean']:.3f} ± {rec['std']:.3f})\n"
        f"Best APl:      {DISPLAY_NAMES[best_apl]} ({apl['mean']:.3f} ± {apl['std']:.3f})\n"
        "==============================================================================\n"
    )
    return header + "\n".join(rows) + "\n" + footer


def build_table2(speed_json: Path) -> str:
    if not speed_json.is_file():
        return "Inference speed JSON not found. Run results/inference_speed.py first.\n"

    data = json.loads(speed_json.read_text(encoding="utf-8"))
    type_map = {
        "orcnn": "2-stage",
        "s2anet": "1-stage",
        "yolov8_obb": "anchor-free",
        "yolov11_obb": "anchor-free",
        "yolov26_obb": "anchor-free",
    }
    lines = [
        "==============================================================================",
        "  Inference Speed Analysis — batch=1, imgsz=1024×1024, NVIDIA A40",
        "==============================================================================",
        f"{'Model':<12}| {'Params (M)':<10} | {'GFLOPs':<7} | {'Latency (ms)':<13} | {'FPS':<14} | Type",
        "-" * 12 + "|" + "-" * 11 + "|" + "-" * 9 + "|" + "-" * 15 + "|" + "-" * 16 + "|" + "-" * 10,
    ]
    for m in ALL_MODELS:
        row = data.get(m, {})
        name = DISPLAY_NAMES[m]
        params = row.get("params_M", -1)
        gflops = row.get("gflops", -1)
        lat_m = row.get("latency_ms_mean", -1)
        lat_s = row.get("latency_ms_std", -1)
        fps = row.get("fps", -1)
        lines.append(
            f"{name:<12}| {params:>8.1f}   | {gflops:>5.1f}   | "
            f"{lat_m:>4.1f} ± {lat_s:<4.1f}  | {fps:>12.1f}   | {type_map[m]}"
        )
    lines += [
        "==============================================================================",
        "Note: Latency = mean ± std over 200 forward passes after 50 warmup runs.",
        "      FLOPs computed via thop/fvcore on 1×3×1024×1024 input.",
        "==============================================================================",
    ]
    return "\n".join(lines) + "\n"


def write_csv(all_stats: Dict[str, dict]) -> None:
    csv_path = RESULTS_DIR / "all_results.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["model", "seed", "run_type"] + METRIC_KEYS
        writer.writerow(header)
        for model, stats in all_stats.items():
            for seed in SEEDS:
                sm = stats["per_seed"].get(f"seed_{seed}", {})
                writer.writerow([model, seed, "per_seed"] + [sm.get(k, "") for k in METRIC_KEYS])
            for k in METRIC_KEYS:
                pass
            mean_row = [model, "mean", "aggregate"] + [
                stats["metrics"][k]["mean"] for k in METRIC_KEYS
            ]
            std_row = [model, "std", "aggregate"] + [
                stats["metrics"][k]["std"] for k in METRIC_KEYS
            ]
            writer.writerow(mean_row)
            writer.writerow(std_row)
    print(f"Saved CSV: {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print_tables", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_stats: Dict[str, dict] = {}

    for model in ALL_MODELS:
        stats = load_or_aggregate(model)
        all_stats[model] = stats
        out = RESULTS_DIR / f"{model}_stats.json"
        out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        ap = stats["metrics"]["AP"]
        print(f"Aggregated {model}: AP={ap['mean']:.4f}±{ap['std']:.4f}")

    write_csv(all_stats)

    table1 = build_table1(all_stats)
    table2 = build_table2(RESULTS_DIR / "inference_speed.json")
    (RESULTS_DIR / "table1_detection.txt").write_text(table1, encoding="utf-8")
    (RESULTS_DIR / "table2_speed.txt").write_text(table2, encoding="utf-8")

    if args.print_tables:
        print("\n" + table1)
        print(table2)
    else:
        print(f"\nSaved: {RESULTS_DIR / 'table1_detection.txt'}")
        print(f"Saved: {RESULTS_DIR / 'table2_speed.txt'}")
        print("Run with --print_tables to display both tables.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
