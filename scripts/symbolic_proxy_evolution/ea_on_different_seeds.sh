#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKBONE="${BACKBONE:-autoformer}"
GROUNDTRUTH_DIR="${GROUNDTRUTH_DIR:-${REPO_ROOT}/proxy_scores/monash_time}"

# Run the mandatory first 200 seeds, then extend each unfinished dataset one
# seed at a time until it has strictly more than UNIQUE_FORMULA_TARGET formulas
# or reaches its per-dataset seed cap.
SEED_START="${SEED_START:-2027}"
INITIAL_SEED_END="${INITIAL_SEED_END:-2226}"
UNIQUE_FORMULA_TARGET="${UNIQUE_FORMULA_TARGET:-1000}"
TOP_K="${TOP_K:-10}"
MAX_SEEDS_PER_DATASET="${MAX_SEEDS_PER_DATASET:-300}"

MAX_GENERATIONS="${MAX_GENERATIONS:-200}"
TARGET_METRIC="${TARGET_METRIC:-mse}"
TARGET_DIRECTION="${TARGET_DIRECTION:-minimize}"

# Two-level parallelism: MAX_PARALLEL EA runs, each with NUM_WORKERS formula
# evaluation workers. Their product should normally not exceed available CPUs.
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-32}"

SOFT_DIV="${SOFT_DIV:-false}"
DENOMINATOR="${DENOMINATOR:-1e-8}"
OVERWRITE="${OVERWRITE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

# A stable run name makes the entire workflow resumable across shell restarts.
# Change RUN_TAG when changing material EA hyperparameters.
RUN_TAG="${RUN_TAG:-monash_time_unique${UNIQUE_FORMULA_TARGET}}"
RUN_NAME="run_${RUN_TAG}"

EVOLVE_SCRIPT="${REPO_ROOT}/scripts/symbolic_proxy_evolution/evolve_symbolic_proxy.py"
COUNT_SCRIPT="${REPO_ROOT}/scripts/symbolic_proxy_evolution/count_unique_top_formulas.py"
ARCHIVE_ROOT="${REPO_ROOT}/archive/symbolic_proxy_evolution/${BACKBONE}"
SUMMARY_CSV="${SUMMARY_CSV:-${ARCHIVE_ROOT}/formula_collection_${RUN_TAG}.csv}"

EXTRA_ARGS=("$@")

require_positive_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "${name} must be a positive integer. Got: ${value}" >&2
    exit 1
  fi
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer. Got: ${value}" >&2
    exit 1
  fi
}

for required_file in "${EVOLVE_SCRIPT}" "${COUNT_SCRIPT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required script not found: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${GROUNDTRUTH_DIR}" ]]; then
  echo "GroundTruth CSV directory not found: ${GROUNDTRUTH_DIR}" >&2
  exit 1
fi
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 1
fi

require_positive_integer "SEED_START" "${SEED_START}"
require_positive_integer "INITIAL_SEED_END" "${INITIAL_SEED_END}"
require_positive_integer "UNIQUE_FORMULA_TARGET" "${UNIQUE_FORMULA_TARGET}"
require_positive_integer "TOP_K" "${TOP_K}"
require_positive_integer "MAX_SEEDS_PER_DATASET" "${MAX_SEEDS_PER_DATASET}"
require_nonnegative_integer "NUM_WORKERS" "${NUM_WORKERS}"
require_positive_integer "MAX_PARALLEL" "${MAX_PARALLEL}"
if (( INITIAL_SEED_END < SEED_START )); then
  echo "INITIAL_SEED_END must be >= SEED_START." >&2
  exit 1
fi
INITIAL_SEED_COUNT=$((INITIAL_SEED_END - SEED_START + 1))
if (( INITIAL_SEED_COUNT > MAX_SEEDS_PER_DATASET )); then
  echo "Initial seed count (${INITIAL_SEED_COUNT}) exceeds MAX_SEEDS_PER_DATASET=${MAX_SEEDS_PER_DATASET}." >&2
  exit 1
fi
MAX_SEED=$((SEED_START + MAX_SEEDS_PER_DATASET - 1))
if (( NUM_WORKERS == 0 && MAX_PARALLEL > 1 )); then
  echo "[warn] NUM_WORKERS=0 lets every EA auto-use up to all CPUs. With MAX_PARALLEL=${MAX_PARALLEL}, this can massively oversubscribe CPU and RAM." >&2
fi

declare -A CSV_BY_DATASET=()
CSV_PREFIX="${BACKBONE}_300_sl96_"
while IFS= read -r csv_path; do
  filename="$(basename "${csv_path}")"
  dataset="${filename#${CSV_PREFIX}}"
  dataset="${dataset%%_proxy_scores_*}"
  if [[ -z "${dataset}" || "${dataset}" == "${filename}" ]]; then
    echo "Could not parse dataset name from CSV: ${csv_path}" >&2
    exit 1
  fi
  if [[ -n "${CSV_BY_DATASET[${dataset}]:-}" ]]; then
    echo "Multiple proxy-score CSVs found for dataset=${dataset}" >&2
    exit 1
  fi
  CSV_BY_DATASET["${dataset}"]="${csv_path}"
done < <(find "${GROUNDTRUTH_DIR}" -maxdepth 1 -type f \
  -name "${CSV_PREFIX}*_proxy_scores_*.csv" -print | sort)

if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASET_LIST <<< "${DATASETS}"
else
  mapfile -t DATASET_LIST < <(printf '%s\n' "${!CSV_BY_DATASET[@]}" | sort)
  if (( ${#DATASET_LIST[@]} != 47 )); then
    echo "Expected 47 Monash+TIME datasets, found ${#DATASET_LIST[@]} under ${GROUNDTRUTH_DIR}." >&2
    exit 1
  fi
fi

for dataset in "${DATASET_LIST[@]}"; do
  if [[ -z "${CSV_BY_DATASET[${dataset}]:-}" ]]; then
    echo "No proxy-score/full-trained-result CSV found for dataset=${dataset}" >&2
    exit 1
  fi
done

COMMON_ARGS=(
  "${EVOLVE_SCRIPT}"
  --backbone "${BACKBONE}"
  --repo-root "${REPO_ROOT}"
  --max-generations "${MAX_GENERATIONS}"
  --target-metric "${TARGET_METRIC}"
  --target-direction "${TARGET_DIRECTION}"
  --num-workers "${NUM_WORKERS}"
  --soft_div "${SOFT_DIV}"
  --denominator "${DENOMINATOR}"
  --visualize-archive
  --visualize-top-k "${TOP_K}"
)

run_one() {
  local seed="$1"
  local dataset="$2"
  local csv_path="${CSV_BY_DATASET[${dataset}]}"
  local output_dir="${ARCHIVE_ROOT}/${dataset}/seed_${seed}/${RUN_NAME}"
  local log_path="${output_dir}/run.log"
  local complete_marker="${output_dir}/.ea_complete"
  local cmd=(
    "${PYTHON_BIN}"
    "${COMMON_ARGS[@]}"
    --dataset "${dataset}"
    --csv-path "${csv_path}"
    --seed "${seed}"
    --output-dir "${output_dir}"
    "${EXTRA_ARGS[@]}"
  )

  if [[ -f "${complete_marker}" && "${OVERWRITE}" != "1" ]]; then
    echo "[skip] dataset=${dataset} seed=${seed}: completed"
    return 0
  fi

  mkdir -p "${output_dir}"
  echo "[run] dataset=${dataset} seed=${seed} output=${output_dir} log=${log_path}"
  if "${cmd[@]}" > "${log_path}" 2>&1; then
    if [[ ! -s "${output_dir}/archive.csv" || ! -s "${output_dir}/visualizations/archive_latex.tex" ]]; then
      echo "[fail] dataset=${dataset} seed=${seed}: expected archive outputs are missing" >&2
      return 1
    fi
    touch "${complete_marker}"
    echo "[done] dataset=${dataset} seed=${seed}"
    return 0
  fi
  echo "[fail] dataset=${dataset} seed=${seed}; see ${log_path}" >&2
  return 1
}

ACTIVE_JOBS=0
PHASE_FAILED=0

wait_for_one() {
  if ! wait -n; then
    PHASE_FAILED=1
  fi
  ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
}

launch_one() {
  local seed="$1"
  local dataset="$2"
  if (( MAX_PARALLEL == 1 )); then
    if ! run_one "${seed}" "${dataset}"; then
      PHASE_FAILED=1
    fi
    return
  fi

  run_one "${seed}" "${dataset}" &
  ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
  if (( ACTIVE_JOBS >= MAX_PARALLEL )); then
    wait_for_one
  fi
}

finish_phase() {
  while (( ACTIVE_JOBS > 0 )); do
    wait_for_one
  done
  if (( PHASE_FAILED != 0 )); then
    echo "[abort] one or more EA runs failed; fix the logged error and rerun this script" >&2
    exit 1
  fi
}

formula_stats() {
  local dataset="$1"
  local seed_end="$2"
  "${PYTHON_BIN}" "${COUNT_SCRIPT}" \
    --dataset-root "${ARCHIVE_ROOT}/${dataset}" \
    --run-name "${RUN_NAME}" \
    --seed-start "${SEED_START}" \
    --seed-end "${seed_end}" \
    --top-k "${TOP_K}" \
    --strict
}

declare -A LAST_SEED=()
declare -A ARCHIVE_FILE_COUNT=()
declare -A TOTAL_FORMULA_ROWS=()
declare -A UNIQUE_FORMULA_COUNT=()

refresh_formula_stats() {
  local dataset="$1"
  local stats
  if ! stats="$(formula_stats "${dataset}" "${LAST_SEED[${dataset}]}")"; then
    echo "Failed to count formulas for dataset=${dataset}" >&2
    exit 1
  fi
  read -r ARCHIVE_FILE_COUNT["${dataset}"] TOTAL_FORMULA_ROWS["${dataset}"] UNIQUE_FORMULA_COUNT["${dataset}"] <<< "${stats}"
}

write_summary() {
  local summary_dir
  local temp_path
  summary_dir="$(dirname "${SUMMARY_CSV}")"
  mkdir -p "${summary_dir}"
  temp_path="$(mktemp "${summary_dir}/.formula_collection.XXXXXX")"
  {
    echo "dataset,csv_path,seed_start,last_seed,seed_count,max_seeds_per_dataset,archive_file_count,top_k,total_formula_rows,unique_formula_count,unique_formula_target,target_exceeded,status,run_name"
    for dataset in "${DATASET_LIST[@]}"; do
      local last_seed="${LAST_SEED[${dataset}]}"
      local seed_count=$((last_seed - SEED_START + 1))
      local exceeded="false"
      local status="collecting"
      if (( UNIQUE_FORMULA_COUNT[${dataset}] > UNIQUE_FORMULA_TARGET )); then
        exceeded="true"
        status="target_exceeded"
      elif (( seed_count >= MAX_SEEDS_PER_DATASET )); then
        status="seed_cap_reached"
      fi
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${dataset}" "${CSV_BY_DATASET[${dataset}]}" "${SEED_START}" "${last_seed}" \
        "${seed_count}" "${MAX_SEEDS_PER_DATASET}" "${ARCHIVE_FILE_COUNT[${dataset}]}" "${TOP_K}" \
        "${TOTAL_FORMULA_ROWS[${dataset}]}" "${UNIQUE_FORMULA_COUNT[${dataset}]}" \
        "${UNIQUE_FORMULA_TARGET}" "${exceeded}" "${status}" "${RUN_NAME}"
    done
  } > "${temp_path}"
  mv "${temp_path}" "${SUMMARY_CSV}"
}

echo "[config] datasets=${#DATASET_LIST[@]} initial_seeds=${SEED_START}..${INITIAL_SEED_END} (${INITIAL_SEED_COUNT})"
echo "[config] unique_target=>${UNIQUE_FORMULA_TARGET} top_k=${TOP_K} max_seeds_per_dataset=${MAX_SEEDS_PER_DATASET} max_seed=${MAX_SEED} run_name=${RUN_NAME}"
if (( NUM_WORKERS == 0 )); then
  echo "[config] max_parallel=${MAX_PARALLEL} num_workers=auto (nested worker count is host-dependent)"
else
  echo "[config] max_parallel=${MAX_PARALLEL} num_workers=${NUM_WORKERS} approximate_inner_worker_slots=$((MAX_PARALLEL * NUM_WORKERS))"
fi
echo "[config] groundtruth_dir=${GROUNDTRUTH_DIR} summary=${SUMMARY_CSV}"

if [[ "${PLAN_ONLY}" == "1" ]]; then
  first_dataset="${DATASET_LIST[0]}"
  echo "[plan-only] validated ${#DATASET_LIST[@]} dataset CSVs; no EA process was launched"
  echo "[plan-only] first dataset=${first_dataset} csv=${CSV_BY_DATASET[${first_dataset}]}"
  exit 0
fi

echo "[phase] launching mandatory initial seed range"
PHASE_FAILED=0
for seed in $(seq "${SEED_START}" "${INITIAL_SEED_END}"); do
  for dataset in "${DATASET_LIST[@]}"; do
    launch_one "${seed}" "${dataset}"
  done
done
finish_phase

for dataset in "${DATASET_LIST[@]}"; do
  LAST_SEED["${dataset}"]="${INITIAL_SEED_END}"
  refresh_formula_stats "${dataset}"
  echo "[count] dataset=${dataset} seeds=${INITIAL_SEED_COUNT} unique=${UNIQUE_FORMULA_COUNT[${dataset}]} target=>${UNIQUE_FORMULA_TARGET}"
done
write_summary

extra_round=0
while true; do
  pending_datasets=()
  for dataset in "${DATASET_LIST[@]}"; do
    completed_seed_count=$((LAST_SEED[${dataset}] - SEED_START + 1))
    if (( UNIQUE_FORMULA_COUNT[${dataset}] <= UNIQUE_FORMULA_TARGET && completed_seed_count < MAX_SEEDS_PER_DATASET )); then
      pending_datasets+=("${dataset}")
    fi
  done

  if (( ${#pending_datasets[@]} == 0 )); then
    break
  fi

  extra_round=$((extra_round + 1))
  echo "[adaptive] round=${extra_round} datasets_below_or_equal_target=${#pending_datasets[@]}"
  PHASE_FAILED=0
  for dataset in "${pending_datasets[@]}"; do
    next_seed=$((LAST_SEED[${dataset}] + 1))
    launch_one "${next_seed}" "${dataset}"
  done
  finish_phase

  for dataset in "${pending_datasets[@]}"; do
    LAST_SEED["${dataset}"]=$((LAST_SEED[${dataset}] + 1))
    refresh_formula_stats "${dataset}"
    echo "[count] dataset=${dataset} last_seed=${LAST_SEED[${dataset}]} unique=${UNIQUE_FORMULA_COUNT[${dataset}]} target=>${UNIQUE_FORMULA_TARGET}"
  done
  write_summary
done

write_summary
seed_cap_count=0
for dataset in "${DATASET_LIST[@]}"; do
  if (( UNIQUE_FORMULA_COUNT[${dataset}] <= UNIQUE_FORMULA_TARGET )); then
    seed_cap_count=$((seed_cap_count + 1))
    echo "[capped] dataset=${dataset} seeds=${MAX_SEEDS_PER_DATASET} last_seed=${LAST_SEED[${dataset}]} unique=${UNIQUE_FORMULA_COUNT[${dataset}]} target=>${UNIQUE_FORMULA_TARGET}"
  fi
done
if (( seed_cap_count == 0 )); then
  echo "[done] every dataset has more than ${UNIQUE_FORMULA_TARGET} unique top-${TOP_K} formulas"
else
  echo "[done] ${seed_cap_count} dataset(s) reached the ${MAX_SEEDS_PER_DATASET}-seed cap without exceeding ${UNIQUE_FORMULA_TARGET}; other datasets completed normally"
fi
echo "[done] summary=${SUMMARY_CSV}"
