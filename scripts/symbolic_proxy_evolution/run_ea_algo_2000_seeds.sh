#!/usr/bin/env bash
set -euo pipefail

START_SEED="${START_SEED:-2027}"
END_SEED="${END_SEED:-2226}"
BACKBONE="${BACKBONE:-autoformer}"
DATASET="${DATASET:-ECL}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_GENERATIONS="${MAX_GENERATIONS:-200}"
VISUALIZE_TOP_K="${VISUALIZE_TOP_K:-10}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVOLVE_SCRIPT="${REPO_ROOT}/scripts/symbolic_proxy_evolution/evolve_symbolic_proxy.py"

if [[ ! -f "${EVOLVE_SCRIPT}" ]]; then
  echo "Evolution script not found: ${EVOLVE_SCRIPT}" >&2
  exit 1
fi

if (( END_SEED < START_SEED )); then
  echo "END_SEED must be >= START_SEED. Got START_SEED=${START_SEED}, END_SEED=${END_SEED}" >&2
  exit 1
fi

TOTAL_RUNS=$((END_SEED - START_SEED + 1))
echo "Running ${TOTAL_RUNS} sequential symbolic proxy evolution jobs."
echo "backbone=${BACKBONE} dataset=${DATASET} seeds=${START_SEED}..${END_SEED}"

for seed in $(seq "${START_SEED}" "${END_SEED}"); do
  run_index=$((seed - START_SEED + 1))
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] run ${run_index}/${TOTAL_RUNS}: seed=${seed}"
  "${PYTHON_BIN}" "${EVOLVE_SCRIPT}" \
    --backbone "${BACKBONE}" \
    --dataset "${DATASET}" \
    --seed "${seed}" \
    --num-workers "${NUM_WORKERS}" \
    --max-generations "${MAX_GENERATIONS}" \
    --visualize-archive \
    --visualize-top-k "${VISUALIZE_TOP_K}"
done

echo "Completed ${TOTAL_RUNS} runs."
