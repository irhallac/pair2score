#!/usr/bin/env bash
set -euo pipefail

# Run the smoke config through three scenarios:
# 1) Stage 1 relative only
# 2) Stage 2 absolute only (fresh adapter)
# 3) Full pipeline (Stage 1 feeding Stage 2)

CONFIG="${1:-configs/repro_baseline_smoke_mini_short.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Stage 1: relative trainer ==="
bash "${SCRIPT_DIR}/run_relative.sh" "${CONFIG}"

echo
echo "=== Stage 2: absolute trainer (fresh adapter) ==="
PYTHONPATH="src:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m pair2score.absolute --config "${CONFIG}" --relative-off

echo
echo "=== Full pipeline: relative + absolute ==="
bash "${SCRIPT_DIR}/run_pipeline.sh" "${CONFIG}"
