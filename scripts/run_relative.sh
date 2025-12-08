#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/pipeline.yaml}"
STAMP=$(date '+%Y-%m-%d_%H-%M-%S')
RUN_META=$(PYTHONPATH="src:${PYTHONPATH:-}" python -c 'import sys; from pair2score.config import load_pipeline_config as load; cfg = load(sys.argv[1]); print(cfg.run_id); print(cfg.trait)' "${CONFIG}")
RUN_ID=$(echo "${RUN_META}" | sed -n '1p')
TRAIT=$(echo "${RUN_META}" | sed -n '2p')
LOG_DIR="outputs/run${RUN_ID}/${TRAIT}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/relative_${STAMP}.log"

echo "Logging to ${LOG_FILE}" >&2
PYTHONPATH="src:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 python -m pair2score.relative --config "${CONFIG}" 2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}

echo "Log stored at ${LOG_FILE}" >&2
exit ${STATUS}
