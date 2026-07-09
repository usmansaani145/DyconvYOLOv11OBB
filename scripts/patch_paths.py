#!/usr/bin/env python3
"""One-time patch: remove cluster-specific paths from all repo Python/JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PY_REPLACEMENTS = [
    ("/gdata1/ranausman/ship_augmentation/.pydeps", '""'),  # remove pydeps
    ("/gdata1/ranausman/ship_obb_v2/data_filtered", 'str(DATA_ROOT)'),
    ("/gdata1/ranausman/ship_obb_v2/runs", 'str(RUNS_ROOT)'),
    ("/gdata1/ranausman/ship_obb_v2/results/figures", 'str(RESULTS_ROOT)'),
    ("/gdata1/ranausman/ship_obb_v2/results", 'str(RESULTS_ROOT)'),
    ("/gdata1/ranausman/ship_obb_v2", 'str(REPO_ROOT)'),
    ("/ghome/ranausman/ship_obb_v2", 'str(REPO_ROOT)'),
    ("/ghome/ranausman/pretrained/yolo11m-obb.pt", 'PRETRAINED_YOLO11'),
    ("/ghome/ranausman/pretrained/yolov8m-obb.pt", 'PRETRAINED_YOLO8'),
    ("/ghome/ranausman/pretrained/yolo26n-obb.pt", 'PRETRAINED_YOLO26'),
    ("/ghome/ranausman/pretrained/", 'str(PRETRAINED_DIR / "")'),
    ("/ghome/ranausman/dota2.0_shipsonly/test", 'str(DOTA_TEST_ROOT)'),
    ("/ghome/ranausman/dota2.0_shipsonly", 'str(DATA_ROOT / "dota_raw")'),
    ("/gdata1/ranausman/HRSC2016/AllImages", 'str(HRSC_IMAGES)'),
    ("/gdata1/ranausman/HRSC2016/Annotations", 'str(HRSC_ANNOTATIONS)'),
    ("/gdata1/ranausman/HRSC2016", 'str(HRSC2016_ROOT)'),
    ('Path("/ghome/ranausman/ship_obb_v2")', 'REPO_ROOT'),
    ('Path("/gdata1/ranausman/ship_obb_v2/runs")', 'RUNS_ROOT'),
    ('Path("/gdata1/ranausman/ship_obb_v2/data_filtered")', 'DATA_ROOT'),
    ('Path("/gdata1/ranausman/ship_obb_v2/results/figures")', 'RESULTS_ROOT'),
    ('Path("/gdata1/ranausman/ship_obb_v2/results")', 'RESULTS_ROOT'),
    ("parents[2]", "parents[1]"),
    ('os.environ.setdefault("YOLO_CONFIG_DIR", "/gdata1/ranausman/ship_obb_v2/tmp/ultra_fig4")',
     'os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / "tmp" / "ultralytics"))'),
    ('os.environ.setdefault("YOLO_CONFIG_DIR", "/gdata1/ranausman/ship_obb_v2/tmp/ultra_fig5")',
     'os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / "tmp" / "ultralytics"))'),
]

PATH_SUBS_JSON = [
    ("/gdata1/ranausman/ship_obb_v2/data_filtered/val/images/", "dataset/val/images/"),
    ("/ghome/ranausman/dota2.0_shipsonly/test/", "dataset/dota_test/"),
    ("/gdata1/ranausman/HRSC2016/AllImages/", "dataset/hrsc2016/AllImages/"),
]

CONSTANTS_IMPORT = """from utils.constants import (
    DATA_ROOT,
    DOTA_TEST_ROOT,
    HRSC2016_ROOT,
    HRSC_ANNOTATIONS,
    HRSC_IMAGES,
    PRETRAINED_DIR,
    PRETRAINED_YOLO11,
    PRETRAINED_YOLO8,
    PRETRAINED_YOLO26,
    REPO_ROOT,
    RESULTS_ROOT,
    RUNS_ROOT,
)
"""


def patch_py(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    if "patch_paths.py" in str(path):
        return False
    for old, new in PY_REPLACEMENTS:
        text = text.replace(old, new)
    if "/gdata1/" in text or "/ghome/" in text:
        pass  # leave for manual review
    if orig != text and "REPO_ROOT" in text and "from utils.constants import" not in text:
        if "REPO = Path(__file__)" in text or "REPO_ROOT" in text:
            # insert import after first sys.path block if needed
            if "from utils.constants import" not in text:
                lines = text.splitlines()
                insert_at = 0
                for i, line in enumerate(lines[:30]):
                    if line.startswith("import ") or line.startswith("from "):
                        insert_at = i + 1
                lines.insert(insert_at, CONSTANTS_IMPORT.strip())
                text = "\n".join(lines) + "\n"
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_json(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    text = raw
    for old, new in PATH_SUBS_JSON:
        text = text.replace(old, new)
    if text != raw:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    n_py = n_json = 0
    for p in REPO.rglob("*.py"):
        if "scripts/patch_paths" in str(p):
            continue
        if patch_py(p):
            n_py += 1
    for p in REPO.rglob("*.json"):
        if patch_json(p):
            n_json += 1
    print(f"Patched {n_py} Python files, {n_json} JSON files")
    remaining = []
    for p in REPO.rglob("*"):
        if p.suffix in {".py", ".json", ".yaml", ".yml", ".md", ".sh"} and p.is_file():
            t = p.read_text(encoding="utf-8", errors="ignore")
            if "/gdata1/" in t or "/ghome/" in t:
                remaining.append(str(p.relative_to(REPO)))
    if remaining:
        print("Files still containing cluster paths:")
        for r in remaining[:30]:
            print(f"  {r}")


if __name__ == "__main__":
    main()
