#!/usr/bin/env bash
# Download DOTA-ShipBench from Hugging Face into dataset/ (or SHIP_OBB_DATA_ROOT).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
python3 "$ROOT/utils/download_dataset.py" "$@"
