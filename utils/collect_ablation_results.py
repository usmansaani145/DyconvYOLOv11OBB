#!/usr/bin/env python3
"""Collect WaveGEM-OBB ablation metrics and print comparison table."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RUNS_ROOT = Path("runs")
RESULTS_DIR = Path("results")
BASELINE_STATS = RESULTS_DIR / "all_models_stats.json"

SEEDS = [42, 123, 456]

ABLATIONS = [
    ("YOLOv11-OBB (baseline)", "yolov11_obb", "yolov11_obb", True),
    ("+ P2 head (Abl-A)", "ablation_A_p2", "ablation_A_p2", False),
    ("+ WPL only (Abl-B)", "ablation_B_wpl", "ablation_B_wpl", False),
    ("+ GEM only (Abl-C)", "ablation_C_gem", "ablation_C_gem", False),
    ("+ WPL+GEM / WaveGEM-OBB", "ablation_D_wpl_gem", "ablation_D_wpl_gem", False),
    ("+ P2+WPL+GEM (full)", "ablation_E_full", "ablation_E_full", False),
]

METRICS = ["AP", "APs", "APm", "APl", "Recall"]


def _load_metrics(path: Path) -> Optional[Dict[str, float]]:
  if not path.exists():
    return None
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def _mean_std(values: List[float]) -> Tuple[float, float]:
  if not values:
    return float("nan"), float("nan")
  mean = sum(values) / len(values)
  if len(values) == 1:
    return mean, 0.0
  var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
  return mean, math.sqrt(var)


def _collect_ablation(run_prefix: str, ablation_key: str) -> Dict[str, Tuple[float, float]]:
  stats: Dict[str, List[float]] = {m: [] for m in METRICS}
  for seed in SEEDS:
    metrics_path = (
        RUNS_ROOT
        / run_prefix
        / f"{ablation_key}_seed{seed}"
        / f"metrics_{ablation_key}_seed{seed}.json"
    )
    data = _load_metrics(metrics_path)
    if data is None:
      continue
    for m in METRICS:
      if m in data:
        stats[m].append(float(data[m]))
  return {m: _mean_std(stats[m]) for m in METRICS}


def _baseline_from_stats() -> Dict[str, Tuple[float, float]]:
  out: Dict[str, Tuple[float, float]] = {}
  if not BASELINE_STATS.exists():
    return out
  with open(BASELINE_STATS, encoding="utf-8") as f:
    all_stats = json.load(f)
  y11 = all_stats.get("yolov11_obb", {})
  for m in METRICS:
    if m in y11:
      out[m] = (y11[m]["mean"], y11[m]["std"])
  return out


def _fmt(mean: float, std: float) -> str:
  if math.isnan(mean):
    return "N/A"
  return f"{mean:.3f}±{std:.3f}"


def main() -> int:
  RESULTS_DIR.mkdir(parents=True, exist_ok=True)

  rows: List[Dict[str, str]] = []
  table_stats: Dict[str, Dict[str, Tuple[float, float]]] = {}

  for label, run_prefix, ablation_key, is_baseline in ABLATIONS:
    if is_baseline:
      stats = _baseline_from_stats()
    else:
      stats = _collect_ablation(run_prefix, ablation_key)
    table_stats[label] = stats
    row = {"Model": label}
    for m in METRICS:
      mean, std = stats.get(m, (float("nan"), float("nan")))
      row[m] = _fmt(mean, std)
    rows.append(row)

  header = (
      "===================================================================\n"
      "  WaveGEM-OBB Ablation Study — DOTA 2.0 Ship Detection\n"
      "  Dataset: 6,624 train / 1,006 val tiles (1024×1024)\n"
      "  Evaluation: mean ± std across seeds 42/123/456\n"
      "  Baseline: YOLOv11-OBB (already completed)\n"
      "===================================================================\n"
      f"{'Model':<24}|  {'AP':<12} {'APs':<12} {'APm':<12} {'APl':<12} {'Recall':<12}\n"
      "------------------------|----------------------------------------------------------"
  )

  lines = [header]
  for row in rows:
    lines.append(
        f"{row['Model']:<24}| {row['AP']:<12} {row['APs']:<12} "
        f"{row['APm']:<12} {row['APl']:<12} {row['Recall']:<12}"
    )

  baseline_ap = table_stats.get("YOLOv11-OBB (baseline)", {}).get("AP", (0.401, 0.003))
  baseline_aps = table_stats.get("YOLOv11-OBB (baseline)", {}).get("APs", (0.354, 0.004))

  best_ap_name, best_ap = "", (-1.0, 0.0)
  best_aps_name, best_aps = "", (-1.0, 0.0)
  for label, stats in table_stats.items():
    if label.startswith("YOLOv11"):
      continue
    ap = stats.get("AP", (float("nan"), 0.0))
    aps = stats.get("APs", (float("nan"), 0.0))
    if not math.isnan(ap[0]) and ap[0] > best_ap[0]:
      best_ap_name, best_ap = label, ap
    if not math.isnan(aps[0]) and aps[0] > best_aps[0]:
      best_aps_name, best_aps = label, aps

  lines.append("===================================================================")
  if best_ap_name:
    delta = best_ap[0] - baseline_ap[0]
    lines.append(
        f"Best model (AP):    {best_ap_name}  AP={_fmt(*best_ap)}  ({delta:+.3f} vs baseline)"
    )
  if best_aps_name:
    delta = best_aps[0] - baseline_aps[0]
    lines.append(
        f"Best model (APs):   {best_aps_name}  APs={_fmt(*best_aps)} ({delta:+.3f} vs baseline)"
    )
  lines.append("===================================================================")
  lines.append("")
  lines.append(
      'Paper table caption (auto-generated):\n'
      '"Ablation results on DOTA 2.0 ship-only validation set.\n'
      " Mean ± standard deviation reported across three random seeds.\n"
      ' Best values in bold. ↑ indicates improvement over baseline."'
  )

  text = "\n".join(lines)
  print(text)

  txt_path = RESULTS_DIR / "ablation_table.txt"
  csv_path = RESULTS_DIR / "ablation_table.csv"
  txt_path.write_text(text + "\n", encoding="utf-8")

  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Model"] + METRICS)
    writer.writeheader()
    writer.writerows(rows)

  print(f"\nSaved: {txt_path}")
  print(f"Saved: {csv_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
