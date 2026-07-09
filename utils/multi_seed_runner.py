#!/usr/bin/env python3
"""
Aggregate metrics across 3 seeds (42, 123, 456) for one model.

Reads per-seed metrics JSON and writes mean ± std summary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared.constants import METRIC_KEYS, RESULTS_DIR, RUNS_DIR, SEEDS


def _metrics_path(model: str, seed: int) -> Path:
    run_dir = RUNS_DIR / model / f"{model}_seed{seed}"
    cands = sorted(run_dir.glob("metrics_*.json"))
    if cands:
        return cands[0]
    raise FileNotFoundError(f"Missing metrics for {model} seed {seed} in {run_dir}")


def _load_seed_metrics(model: str, seed: int) -> Dict[str, float]:
    path = _metrics_path(model, seed)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(data.get(k, -1.0)) for k in METRIC_KEYS}


def aggregate_model(model: str) -> dict:
    per_seed: Dict[str, dict] = {}
    values: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}

    for seed in SEEDS:
        m = _load_seed_metrics(model, seed)
        per_seed[f"seed_{seed}"] = m
        for k in METRIC_KEYS:
            v = m.get(k, -1.0)
            if v >= 0:
                values[k].append(v)

    metrics_summary: Dict[str, dict] = {}
    for k in METRIC_KEYS:
        vals = values[k]
        if not vals:
            metrics_summary[k] = {"mean": -1.0, "std": -1.0}
        elif len(vals) == 1:
            metrics_summary[k] = {"mean": vals[0], "std": 0.0}
        else:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            metrics_summary[k] = {"mean": mean, "std": math.sqrt(var)}

    return {
        "model": model,
        "seeds": SEEDS,
        "metrics": metrics_summary,
        "per_seed": per_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed metrics")
    parser.add_argument("model", nargs="?", help="Model name (e.g. yolov11_obb)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Aggregate all models listed in constants",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        from shared.constants import ALL_MODELS

        models = ALL_MODELS
    elif args.model:
        models = [args.model]
    else:
        parser.error("Provide model name or --all")

    for model in models:
        stats = aggregate_model(model)
        out = RESULTS_DIR / f"{model}_stats.json"
        out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        ap = stats["metrics"]["AP"]
        print(f"{model}: AP = {ap['mean']:.4f} ± {ap['std']:.4f}  -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
