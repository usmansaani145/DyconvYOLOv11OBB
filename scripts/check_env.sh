#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
echo "=== ShipOBB check ==="
python3 - <<PY
import sys
from pathlib import Path
root = Path("$ROOT")
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "utils"))
from constants import DATA_ROOT, RUNS_ROOT, DOTA_TEST_ROOT, HRSC2016_ROOT, HF_DATASET_URL, SEEDS
from download_dataset import is_dataset_ready
print(f"  DATA_ROOT={DATA_ROOT}")
print(f"  RUNS_ROOT={RUNS_ROOT}")
print(f"  DOTA_TEST={DOTA_TEST_ROOT}")
print(f"  HRSC2016={HRSC2016_ROOT}")
print(f"  SEEDS={SEEDS}")
print(f"  HF_DATASET={HF_DATASET_URL}")
if not is_dataset_ready(DATA_ROOT):
    print("")
    print("ERROR: Dataset not found.")
    print("Download DOTA-ShipBench from Hugging Face:")
    print("  bash scripts/download_dataset.sh")
    sys.exit(1)
PY
for d in dataset/train/images dataset/val/images dataset/dota_test dataset/hrsc2016/AllImages; do
  echo "  $d: $(find "$ROOT/$d" -type f 2>/dev/null | wc -l) files"
done
echo "=== OK — see README.md ==="
