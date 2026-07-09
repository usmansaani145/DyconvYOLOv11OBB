#!/usr/bin/env python3
"""Collect all baseline, ablation, and idea metrics into one comparison table."""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RUNS_ROOT = Path("runs")
RESULTS_DIR = Path("./results")
SEEDS = [42, 123, 456]
CORE_METRICS = ["AP", "APs", "APm", "APl"]
EXTRA_METRICS = ["AP50", "AP75", "Recall", "Precision"]

# (display label, run_prefix, run_key, metrics filename template)
MODELS = [
    ("YOLOv8-OBB (baseline)", "yolov8_obb", "yolov8_obb", "metrics_yolov8.json"),
    ("YOLOv11-OBB (baseline)", "yolov11_obb", "yolov11_obb", "metrics_yolov11.json"),
    ("YOLOv26-OBB (baseline)", "yolov26_obb", "yolov26_obb", "metrics_yolov26.json"),
    ("S2ANet (baseline)", "s2anet", "s2anet", "metrics_s2anet.json"),
    ("Oriented R-CNN (baseline)", "orcnn", "orcnn", "metrics_orcnn.json"),
    ("Ablation A — P2 detection head", "ablation_A_p2", "ablation_A_p2",
     "metrics_ablation_A_p2_seed{seed}.json"),
    ("Ablation B — WPL (Wavelet Pooling)", "ablation_B_wpl", "ablation_B_wpl",
     "metrics_ablation_B_wpl_seed{seed}.json"),
    ("Ablation C — GEM / GAL", "ablation_C_gem", "ablation_C_gem",
     "metrics_ablation_C_gem_seed{seed}.json"),
    ("Ablation D — WPL+GEM (combined)", "ablation_D_wpl_gem", "ablation_D_wpl_gem",
     "metrics_ablation_D_wpl_gem_seed{seed}.json"),
    ("Ablation E — P2+WPL+GEM (full)", "ablation_E_full", "ablation_E_full",
     "metrics_ablation_E_full_seed{seed}.json"),
    ("Idea 1 — LSKA", "idea1_lska", "idea1_lska", "metrics_idea1_lska_seed{seed}.json"),
    ("Idea 2 — EMA", "idea2_ema", "idea2_ema", "metrics_idea2_ema_seed{seed}.json"),
    ("Idea 3 — MSDA+CSPStage", "idea3_msda", "idea3_msda",
     "metrics_idea3_msda_seed{seed}.json"),
    ("Idea 4 — DyConv", "idea4_dyconv", "idea4_dyconv",
     "metrics_idea4_dyconv_seed{seed}.json"),
    ("Idea 5 — CSD (LKCA+MSDP)", "idea5_csd_lkca_msdp", "idea5_csd_lkca_msdp",
     "metrics_idea5_csd_lkca_msdp_seed{seed}.json"),
]


def _metrics_path(prefix: str, key: str, tmpl: str, seed: int) -> Path:
  if "{seed}" in tmpl:
    return RUNS_ROOT / prefix / f"{key}_seed{seed}" / tmpl.format(seed=seed)
  return RUNS_ROOT / prefix / f"{key}_seed{seed}" / tmpl


def _load(path: Path) -> Optional[Dict[str, float]]:
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


def _fmt_val(v: Optional[float]) -> str:
  if v is None or (isinstance(v, float) and math.isnan(v)):
    return "N/A"
  return f"{v:.3f}"


def _fmt_ms(mean: float, std: float) -> str:
  if math.isnan(mean):
    return "N/A"
  if std == 0.0:
    return f"{mean:.3f}"
  return f"{mean:.3f}±{std:.3f}"


def collect_all() -> Dict[str, Dict[int, Dict[str, float]]]:
  out: Dict[str, Dict[int, Dict[str, float]]] = {}
  for label, prefix, key, tmpl in MODELS:
    out[label] = {}
    for seed in SEEDS:
      data = _load(_metrics_path(prefix, key, tmpl, seed))
      if data is not None:
        out[label][seed] = {k: float(data[k]) for k in data if isinstance(data[k], (int, float))}
  return out


def _pad(s: str, w: int) -> str:
  return s[:w].ljust(w)


def build_table(data: Dict[str, Dict[int, Dict[str, float]]]) -> str:
  lines: List[str] = []
  sep = "=" * 120
  thin = "-" * 120

  lines += [
      sep,
      "  COMPLETE SHIP OBB DETECTION RESULTS — ALL MODELS, ALL SEEDS",
      "  Dataset: 6,624 train / 1,006 val tiles (1024×1024) | Seeds: 42, 123, 456",
      "  Evaluation: pycocotools COCOeval (AP @ IoU 0.50:0.95)",
      sep,
      "",
      "TABLE 1 — AP (mAP@0.50:0.95) per seed and mean±std",
      thin,
  ]

  h = (
      f"{'Model':<34}| {'s42':>8} | {'s123':>8} | {'s456':>8} | {'mean±std':>12} | {'n':>2}"
  )
  lines.append(h)
  lines.append(thin)

  for label, _, _, _ in MODELS:
    seeds_data = data.get(label, {})
    aps = []
    row_vals = []
    for seed in SEEDS:
      if seed in seeds_data and "AP" in seeds_data[seed]:
        v = seeds_data[seed]["AP"]
        aps.append(v)
        row_vals.append(_fmt_val(v))
      else:
        row_vals.append("N/A")
    mean, std = _mean_std(aps)
    lines.append(
        f"{_pad(label, 34)}| {row_vals[0]:>8} | {row_vals[1]:>8} | {row_vals[2]:>8} | "
        f"{_fmt_ms(mean, std):>12} | {len(aps):>2}"
    )

  for metric in CORE_METRICS:
    if metric == "AP":
      continue
    lines += ["", f"TABLE — {metric} per seed and mean±std", thin]
    lines.append(h.replace("AP (mAP@0.50:0.95)", metric))
    lines.append(thin)
    for label, _, _, _ in MODELS:
      seeds_data = data.get(label, {})
      vals = []
      row_vals = []
      for seed in SEEDS:
        if seed in seeds_data and metric in seeds_data[seed]:
          v = seeds_data[seed][metric]
          vals.append(v)
          row_vals.append(_fmt_val(v))
        else:
          row_vals.append("N/A")
      mean, std = _mean_std(vals)
      lines.append(
          f"{_pad(label, 34)}| {row_vals[0]:>8} | {row_vals[1]:>8} | {row_vals[2]:>8} | "
          f"{_fmt_ms(mean, std):>12} | {len(vals):>2}"
      )

  lines += [
      "",
      sep,
      "TABLE 2 — FULL PER-SEED DETAIL (AP, AP50, AP75, APs, APm, APl, Recall, Precision)",
      sep,
  ]

  all_metrics = CORE_METRICS + EXTRA_METRICS
  for label, _, _, _ in MODELS:
    seeds_data = data.get(label, {})
    if not seeds_data:
      lines += ["", f"{label}", "  (no metrics found)"]
      continue
    lines += ["", f"{label}", thin]
    hdr = f"{'Seed':>6} | " + " | ".join(f"{m:>10}" for m in all_metrics)
    lines.append(hdr)
    lines.append(thin)
    for seed in SEEDS:
      if seed not in seeds_data:
        lines.append(f"{seed:>6} | " + " | ".join(f"{'N/A':>10}" for _ in all_metrics))
        continue
      d = seeds_data[seed]
      cells = [_fmt_val(d.get(m)) for m in all_metrics]
      lines.append(f"{seed:>6} | " + " | ".join(f"{c:>10}" for c in cells))
    # mean row for available seeds
    mean_cells = []
    for m in all_metrics:
      vals = [seeds_data[s][m] for s in SEEDS if s in seeds_data and m in seeds_data[s]]
      mean, std = _mean_std(vals)
      mean_cells.append(_fmt_ms(mean, std) if vals else "N/A")
    lines.append(thin)
    lines.append("  mean | " + " | ".join(f"{c:>10}" for c in mean_cells))

  lines += [
      "",
      sep,
      "NOTES",
      "  • Primary baseline for architecture comparison: YOLOv11-OBB (AP=0.401±0.003 over 3 seeds)",
      "  • Ablation B, D, E: only seed 42 available (multi-seed runs not completed)",
      "  • S2ANet / Oriented R-CNN: MMRotate baselines on same tiled dataset",
      "  • Ideas 1–5: isolated architectural modules on YOLOv11m-OBB, 100 epochs, AdamW lr=0.001",
      sep,
  ]
  return "\n".join(lines)


def main() -> int:
  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
  data = collect_all()
  text = build_table(data)
  print(text)

  txt_path = RESULTS_DIR / "complete_comparison_table.txt"
  csv_path = RESULTS_DIR / "complete_comparison_table.csv"

  txt_path.write_text(text + "\n", encoding="utf-8")

  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["Model", "Metric", "seed42", "seed123", "seed456", "mean", "std", "n_seeds"]
    )
    for label, _, _, _ in MODELS:
      seeds_data = data.get(label, {})
      for metric in CORE_METRICS + EXTRA_METRICS:
        row_vals = []
        nums = []
        for seed in SEEDS:
          if seed in seeds_data and metric in seeds_data[seed]:
            v = seeds_data[seed][metric]
            row_vals.append(f"{v:.6f}")
            nums.append(v)
          else:
            row_vals.append("")
        mean, std = _mean_std(nums)
        writer.writerow([
            label, metric, *row_vals,
            f"{mean:.6f}" if nums else "",
            f"{std:.6f}" if len(nums) > 1 else ("0" if nums else ""),
            len(nums),
        ])

  print(f"\nSaved: {txt_path}")
  print(f"Saved: {csv_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
