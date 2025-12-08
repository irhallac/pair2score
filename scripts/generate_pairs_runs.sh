#!/usr/bin/env bash
set -euo pipefail

# Generates large and small pair datasets for a set of traits and run ids.
# Defaults mirror the original run-1 specification (target usage 5, 80/10/10 split,
# min gap 1.0, seed 36) so that every run is comparable.
#
# Usage examples:
#   bash scripts/generate_pairs_runs.sh                    # grammar + vocabulary
#   bash scripts/generate_pairs_runs.sh grammar            # single trait
#   FORCE=1 bash scripts/generate_pairs_runs.sh vocabulary # overwrite existing files
#
# Environment overrides:
#   CSV=<path>           Source essay CSV (default: data/datasets/main/train_with_folds.csv)
#   TARGET_USAGE=<int>   Per-essay target usage (default: 5)
#   VAL_RATIO=<float>    Validation essay fraction (default: 0.1)
#   TEST_RATIO=<float>   Test essay fraction (default: 0.1)
#   MIN_GAP=<float>      Minimum score gap between paired essays (default: 1.0)
#   SEED=<int>           Base RNG seed (default: 36)
#   SMALL_FRACTION=<float> Split fraction for the “small” datasets (default: 0.5)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

CSV=${CSV:-data/datasets/main/train_with_folds.csv}
TARGET_USAGE=${TARGET_USAGE:-5}
VAL_RATIO=${VAL_RATIO:-0.1}
TEST_RATIO=${TEST_RATIO:-0.1}
MIN_GAP=${MIN_GAP:-1.0}
SEED=${SEED:-36}
SMALL_FRACTION=${SMALL_FRACTION:-0.5}
FORCE=${FORCE:-0}

if [[ ! -f "${CSV}" ]]; then
    echo "Essay CSV not found at ${CSV}" >&2
    exit 1
fi

# Default traits when none are supplied.
if [[ $# -eq 0 ]]; then
    set -- grammar vocabulary
fi

# Five-fold rotation: pair_folds | heldout_fold (stage-2 only).
declare -A RUN_ROTATIONS=(
    [1]="A B C D|E"
    [2]="B C D E|A"
    [3]="C D E A|B"
    [4]="D E A B|C"
    [5]="E A B C|D"
)

run_generate() {
    local trait="$1"
    local run_id="$2"
    local pair_dir="$3"
    local split_fraction="$4"
    local folds_and_holdout="${RUN_ROTATIONS[${run_id}]}"
    local pair_folds="${folds_and_holdout%%|*}"
    local heldout="${folds_and_holdout##*|}"

    local output_json="${pair_dir}/run${run_id}_${trait}.jsonl"
    local output_meta="${pair_dir}/run${run_id}_${trait}_meta.json"

    if [[ "${FORCE}" != "1" && -f "${output_json}" && -f "${output_meta}" ]]; then
        echo "[skip] ${output_json} already exists (set FORCE=1 to overwrite)"
        return
    fi

    echo "[generate] trait=${trait} run=${run_id} split_fraction=${split_fraction} -> ${output_json}"
    python data/generate_pairs.py \
        --trait "${trait}" \
        --run-id "${run_id}" \
        --target-usage "${TARGET_USAGE}" \
        --val-ratio "${VAL_RATIO}" \
        --test-ratio "${TEST_RATIO}" \
        --min-gap "${MIN_GAP}" \
        --seed "${SEED}" \
        --essays-csv "${CSV}" \
        --pair-folds ${pair_folds} \
        --heldout-fold "${heldout}" \
        --output-dir "${pair_dir}" \
        --split-fraction "${split_fraction}"
}

for trait in "$@"; do
    echo "=== Generating pairs for trait '${trait}' ==="
    mkdir -p data/pairs_large data/pairs_small
    for run_id in 1 2 3 4 5; do
        run_generate "${trait}" "${run_id}" "data/pairs_large" 1.0
        run_generate "${trait}" "${run_id}" "data/pairs_small" "${SMALL_FRACTION}"
    done
done

echo "All requested pair files processed."
