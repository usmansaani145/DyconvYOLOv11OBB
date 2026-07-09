#!/usr/bin/env bash
# Set repo-relative paths for the current shell session.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SHIP_OBB_REPO="$ROOT"
export SHIP_OBB_DATA_ROOT="${SHIP_OBB_DATA_ROOT:-$ROOT/dataset}"
export SHIP_OBB_RUNS_ROOT="${SHIP_OBB_RUNS_ROOT:-$ROOT/runs}"
export SHIP_OBB_RESULTS_ROOT="${SHIP_OBB_RESULTS_ROOT:-$ROOT/results}"
export DOTA_TEST_ROOT="${DOTA_TEST_ROOT:-$ROOT/dataset/dota_test}"
export HRSC2016_ROOT="${HRSC2016_ROOT:-$ROOT/dataset/hrsc2016}"
export PRETRAINED_DIR="${PRETRAINED_DIR:-$ROOT/pretrained}"
export PYTHONPATH="${ROOT}:${ROOT}/utils:${ROOT}/ideas/idea4_dyconv:${PYTHONPATH:-}"
echo "ShipOBB environment configured for: $ROOT"
