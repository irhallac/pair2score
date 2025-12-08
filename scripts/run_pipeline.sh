#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/pipeline.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP=$(TZ=Europe/Oslo date '+%Y-%m-%d_%H-%M-%S')

RUN_META=$(
    PYTHONPATH="src:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -c 'import sys; from pair2score.config import load_pipeline_config as load; cfg = load(sys.argv[1]); print(cfg.run_id); print(cfg.trait); print(cfg.get("paths", "output_dir"))' \
        "${CONFIG}" 2>/dev/null || true
)

if [[ -n "${RUN_META}" ]]; then
    RUN_ID=$(echo "${RUN_META}" | sed -n '1p')
    TRAIT=$(echo "${RUN_META}" | sed -n '2p')
    OUTPUT_DIR=$(echo "${RUN_META}" | sed -n '3p')
    if [[ -n "${OUTPUT_DIR}" && "${OUTPUT_DIR}" != "None" ]]; then
        LOG_DIR="${OUTPUT_DIR}"
    else
        LOG_DIR="outputs/run${RUN_ID}/${TRAIT}"
    fi
else
    LOG_DIR="${LOG_DIR:-logs}"
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_${STAMP}.log"
CONFIG_COPY="${LOG_DIR}/run_${STAMP}_config.yaml"

cp "${CONFIG}" "${CONFIG_COPY}"

echo "Logging to ${LOG_FILE}"
PYTHONPATH="src:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m pair2score.cli "${CONFIG}" 2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}

echo "Log stored at ${LOG_FILE}"
echo "Config snapshot at ${CONFIG_COPY}"
exit "${STATUS}"
