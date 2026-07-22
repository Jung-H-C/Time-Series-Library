#!/usr/bin/env bash
set -uo pipefail

# Exhaustive fixed-split DCSPG hyperparameter search.
#
# By default four processes are run on each GPU. Increase/decrease
# JOBS_PER_GPU when host RAM or GPU memory requires it, for example:
#   JOBS_PER_GPU=2 bash scripts/run_dcspg_framework_hparam_search.sh
# GPU-specific concurrency can instead be selected with:
#   GPU_SLOT_COUNTS="2 1 3" bash scripts/run_dcspg_framework_hparam_search.sh
# To resume an interrupted search from its existing manifest:
#   RESUME_MANIFEST=path/to/hparam_manifest.csv \
#     bash scripts/run_dcspg_framework_hparam_search.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
if [[ -n "${CONDA_ENV_NAME}" ]]; then
    PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python)
else
    PYTHON_CMD=("${PYTHON_BIN}")
fi

SEARCH_STAMP="${SEARCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEARCH_NAME="${SEARCH_NAME:-dcspg_fixed_split_hparam_${SEARCH_STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-DCSPG/checkpoints/fixed_split_hparam_search/${SEARCH_NAME}}"
LOG_DIR="${LOG_DIR:-logs/${SEARCH_NAME}}"
RESUME_MANIFEST="${RESUME_MANIFEST:-}"
if [[ -n "${RESUME_MANIFEST}" ]]; then
    MANIFEST_CSV="${RESUME_MANIFEST}"
else
    MANIFEST_CSV="${OUTPUT_ROOT}/hparam_manifest.csv"
fi

read -r -a GPU_IDS <<< "${GPU_IDS:-0 1}"
read -r -a D_MODEL_VALUES <<< "${D_MODEL_VALUES:-32 64 128}"
read -r -a N_HEAD_VALUES <<< "${N_HEAD_VALUES:-1 2 4}"
read -r -a ENCODER_LAYER_VALUES <<< "${ENCODER_LAYER_VALUES:-1 2 3}"
read -r -a DECODER_LAYER_VALUES <<< "${DECODER_LAYER_VALUES:-1 2 3}"
read -r -a LEARNING_RATE_VALUES <<< "${LEARNING_RATE_VALUES:-5e-5 1e-4 2e-4}"

JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
GPU_SLOT_COUNTS_RAW="${GPU_SLOT_COUNTS:-}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_dcspg_framework_hparam_search.sh [TRAINING_ARGS...]

Runs all 243 combinations of:
  d_model={32,64,128}, n_heads={1,2,4},
  encoder_layers={1,2,3}, decoder_layers={1,2,3},
  dim_feedforward=2*d_model, learning_rate={5e-5,1e-4,2e-4}.

Any arguments supplied to this script are forwarded to train_dcspg_framework.py.
Useful environment variables:
  GPU_IDS="0 1 3"          GPUs used by the worker pool
  JOBS_PER_GPU=4           concurrent processes on every GPU (default: 4)
  GPU_SLOT_COUNTS="2 1 2" per-GPU concurrency; overrides JOBS_PER_GPU
  CONDA_ENV_NAME=name      run Python in this conda environment
  OUTPUT_ROOT=path         checkpoint/search output root
  LOG_DIR=path             per-run log directory
  RESUME_MANIFEST=path     use an existing manifest, skip completed runs, and
                           retry incomplete/missing runs in manifest order
  DRY_RUN=1                print commands without running or writing files

Example (two simultaneous training processes per GPU):
  JOBS_PER_GPU=2 CONDA_ENV_NAME=myenv \
    bash scripts/run_dcspg_framework_hparam_search.sh --max-epochs 500

Example (resume with four simultaneous processes per GPU):
  RESUME_MANIFEST=DCSPG/checkpoints/fixed_split_hparam_search/<search>/hparam_manifest.csv \
    JOBS_PER_GPU=4 bash scripts/run_dcspg_framework_hparam_search.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if (( ${#GPU_IDS[@]} == 0 )); then
    echo "GPU_IDS must contain at least one GPU id." >&2
    exit 2
fi

WORKER_GPUS=()
if [[ -n "${GPU_SLOT_COUNTS_RAW}" ]]; then
    read -r -a GPU_SLOT_COUNTS_ARRAY <<< "${GPU_SLOT_COUNTS_RAW}"
    if (( ${#GPU_SLOT_COUNTS_ARRAY[@]} != ${#GPU_IDS[@]} )); then
        echo "GPU_SLOT_COUNTS must have one value per GPU_IDS entry." >&2
        exit 2
    fi
    for gpu_index in "${!GPU_IDS[@]}"; do
        slot_count="${GPU_SLOT_COUNTS_ARRAY[gpu_index]}"
        if ! is_positive_integer "${slot_count}"; then
            echo "GPU slot counts must be positive integers; got '${slot_count}'." >&2
            exit 2
        fi
        for ((slot_index = 0; slot_index < slot_count; slot_index++)); do
            WORKER_GPUS+=("${GPU_IDS[gpu_index]}")
        done
    done
else
    if ! is_positive_integer "${JOBS_PER_GPU}"; then
        echo "JOBS_PER_GPU must be a positive integer; got '${JOBS_PER_GPU}'." >&2
        exit 2
    fi
    for gpu_id in "${GPU_IDS[@]}"; do
        for ((slot_index = 0; slot_index < JOBS_PER_GPU; slot_index++)); do
            WORKER_GPUS+=("${gpu_id}")
        done
    done
fi

format_cmd() {
    printf ' %q' "$@"
}

combo_name() {
    local d_model="$1"
    local n_heads="$2"
    local encoder_layers="$3"
    local decoder_layers="$4"
    local learning_rate="$5"
    local lr_label="${learning_rate//./p}"
    lr_label="${lr_label//-/m}"
    printf 'dm%s_nh%s_enc%s_dec%s_ff%s_lr%s' \
        "${d_model}" "${n_heads}" "${encoder_layers}" "${decoder_layers}" \
        "$((d_model * 2))" "${lr_label}"
}

COMBOS=()
RUN_NAMES=()
RUN_DIRS=()
LOG_FILES=()

if [[ -n "${RESUME_MANIFEST}" ]]; then
    if [[ ! -f "${MANIFEST_CSV}" ]]; then
        echo "Resume manifest does not exist: ${MANIFEST_CSV}" >&2
        exit 2
    fi

    header="$(head -n 1 "${MANIFEST_CSV}")"
    header="${header%$'\r'}"
    expected_header="run_name,d_model,n_heads,encoder_layers,decoder_layers,dim_feedforward,learning_rate,gpu_worker_assignment,output_dir,log_file"
    if [[ "${header}" != "${expected_header}" ]]; then
        echo "Unexpected manifest header in ${MANIFEST_CSV}" >&2
        exit 2
    fi

    while IFS=',' read -r run_name d_model n_heads encoder_layers decoder_layers \
        dim_feedforward learning_rate _gpu_worker_assignment run_dir log_file; do
        [[ "${run_name}" == "run_name" ]] && continue
        log_file="${log_file%$'\r'}"
        if [[ -z "${run_name}" || -z "${run_dir}" || -z "${log_file}" ]]; then
            echo "Malformed row in ${MANIFEST_CSV}: run_name/output_dir/log_file is empty." >&2
            exit 2
        fi
        if ! [[ "${d_model}" =~ ^[1-9][0-9]*$ \
            && "${n_heads}" =~ ^[1-9][0-9]*$ \
            && "${encoder_layers}" =~ ^[1-9][0-9]*$ \
            && "${decoder_layers}" =~ ^[1-9][0-9]*$ \
            && "${dim_feedforward}" =~ ^[1-9][0-9]*$ ]]; then
            echo "Invalid numeric hyperparameter in manifest row for ${run_name}." >&2
            exit 2
        fi
        if (( d_model % n_heads != 0 )); then
            echo "Invalid manifest row for ${run_name}: d_model=${d_model} is not divisible by n_heads=${n_heads}." >&2
            exit 2
        fi
        if [[ "${dim_feedforward}" != "$((d_model * 2))" ]]; then
            echo "Invalid manifest row for ${run_name}: dim_feedforward=${dim_feedforward}, expected $((d_model * 2))." >&2
            exit 2
        fi
        COMBOS+=("${d_model}:${n_heads}:${encoder_layers}:${decoder_layers}:${learning_rate}")
        RUN_NAMES+=("${run_name}")
        RUN_DIRS+=("${run_dir}")
        LOG_FILES+=("${log_file}")
    done < "${MANIFEST_CSV}"

    if (( ${#COMBOS[@]} == 0 )); then
        echo "Resume manifest contains no runs: ${MANIFEST_CSV}" >&2
        exit 2
    fi
else
    for d_model in "${D_MODEL_VALUES[@]}"; do
        for n_heads in "${N_HEAD_VALUES[@]}"; do
            if (( d_model % n_heads != 0 )); then
                echo "Invalid combination: d_model=${d_model} is not divisible by n_heads=${n_heads}." >&2
                exit 2
            fi
            for encoder_layers in "${ENCODER_LAYER_VALUES[@]}"; do
                for decoder_layers in "${DECODER_LAYER_VALUES[@]}"; do
                    for learning_rate in "${LEARNING_RATE_VALUES[@]}"; do
                        run_name="$(combo_name "${d_model}" "${n_heads}" "${encoder_layers}" "${decoder_layers}" "${learning_rate}")"
                        COMBOS+=("${d_model}:${n_heads}:${encoder_layers}:${decoder_layers}:${learning_rate}")
                        RUN_NAMES+=("${run_name}")
                        RUN_DIRS+=("${OUTPUT_ROOT}/${run_name}")
                        LOG_FILES+=("${LOG_DIR}/${run_name}.log")
                    done
                done
            done
        done
    done
fi

is_completed_run() {
    local run_dir="$1"
    local candidate
    # Retries use train_dcspg_framework.py's _2, _3, ... suffixes. Any one
    # fully finalized directory means this manifest setting is complete.
    for candidate in "${run_dir}" "${run_dir}"_[0-9]*; do
        [[ -d "${candidate}" ]] || continue
        if [[ -s "${candidate}/summary.json" \
            && -s "${candidate}/test_results_best_checkpoint.csv" \
            && -s "${candidate}/test_results_averaged_checkpoint.csv" ]]; then
            return 0
        fi
    done
    return 1
}

run_one_combo() {
    local gpu_id="$1"
    local job_index="$2"
    shift 2
    local combo="${COMBOS[job_index]}"
    local d_model n_heads encoder_layers decoder_layers learning_rate
    IFS=':' read -r d_model n_heads encoder_layers decoder_layers learning_rate <<< "${combo}"

    local dim_feedforward=$((d_model * 2))
    local run_name="${RUN_NAMES[job_index]}"
    local run_dir="${RUN_DIRS[job_index]}"
    local output_root_for_run
    output_root_for_run="$(dirname "${run_dir}")"
    local log_file="${LOG_FILES[job_index]}"
    if [[ -e "${run_dir}" ]]; then
        local resume_stamp
        resume_stamp="$(date +%Y%m%d_%H%M%S)"
        if [[ "${log_file}" == *.log ]]; then
            log_file="${log_file%.log}.resume_${resume_stamp}.log"
        else
            log_file="${log_file}.resume_${resume_stamp}"
        fi
    fi
    local -a command=(
        "${PYTHON_CMD[@]}"
        train_dcspg_framework.py
        "$@"
        --output-dir "${output_root_for_run}"
        --run-name "${run_name}"
        --device auto
        --gpu-id "${gpu_id}"
        --d-model "${d_model}"
        --n-heads "${n_heads}"
        --encoder-layers "${encoder_layers}"
        --decoder-layers "${decoder_layers}"
        --dim-feedforward "${dim_feedforward}"
        --learning-rate "${learning_rate}"
    )

    echo "[GPU ${gpu_id}] START ${run_name}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  command:$(format_cmd "${command[@]}")"
        return 0
    fi

    mkdir -p "$(dirname "${log_file}")"

    local exit_code
    if "${command[@]}" > "${log_file}" 2>&1; then
        echo "[GPU ${gpu_id}] DONE  ${run_name}"
        return 0
    else
        exit_code=$?
    fi

    echo "[GPU ${gpu_id}] FAIL  ${run_name} (exit=${exit_code}, log=${log_file})" >&2
    return "${exit_code}"
}

run_worker() {
    local worker_index="$1"
    shift
    local gpu_id="${WORKER_GPUS[worker_index]}"
    local pending_index job_index
    local worker_failed=0
    for ((pending_index = worker_index; pending_index < ${#PENDING_INDICES[@]}; pending_index += ${#WORKER_GPUS[@]})); do
        job_index="${PENDING_INDICES[pending_index]}"
        if ! run_one_combo "${gpu_id}" "${job_index}" "$@"; then
            worker_failed=1
        fi
    done
    return "${worker_failed}"
}

echo "DCSPG fixed-split exhaustive hyperparameter search"
echo "  combinations=${#COMBOS[@]}"
echo "  gpu_ids=${GPU_IDS[*]}"
echo "  worker_gpu_assignments=${WORKER_GPUS[*]}"
echo "  concurrent_jobs=${#WORKER_GPUS[@]}"
if [[ -n "${RESUME_MANIFEST}" ]]; then
    echo "  output_root=$(dirname "${RUN_DIRS[0]}") (from manifest)"
    echo "  log_dir=$(dirname "${LOG_FILES[0]}") (from manifest)"
else
    echo "  output_root=${OUTPUT_ROOT}"
    echo "  log_dir=${LOG_DIR}"
fi
echo "  manifest=${MANIFEST_CSV}"

if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ -z "${RESUME_MANIFEST}" ]]; then
        mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
        {
            echo "run_name,d_model,n_heads,encoder_layers,decoder_layers,dim_feedforward,learning_rate,gpu_worker_assignment,output_dir,log_file"
            for job_index in "${!COMBOS[@]}"; do
                IFS=':' read -r d_model n_heads encoder_layers decoder_layers learning_rate <<< "${COMBOS[job_index]}"
                gpu_id="${WORKER_GPUS[job_index % ${#WORKER_GPUS[@]}]}"
                echo "${RUN_NAMES[job_index]},${d_model},${n_heads},${encoder_layers},${decoder_layers},$((d_model * 2)),${learning_rate},${gpu_id},${RUN_DIRS[job_index]},${LOG_FILES[job_index]}"
            done
        } > "${MANIFEST_CSV}"
    fi
fi

PENDING_INDICES=()
completed_count=0
for job_index in "${!COMBOS[@]}"; do
    if is_completed_run "${RUN_DIRS[job_index]}"; then
        completed_count=$((completed_count + 1))
        echo "[SKIP completed] ${RUN_NAMES[job_index]}"
    else
        PENDING_INDICES+=("${job_index}")
    fi
done
echo "  completed=${completed_count}"
echo "  pending=${#PENDING_INDICES[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
    for pending_index in "${!PENDING_INDICES[@]}"; do
        job_index="${PENDING_INDICES[pending_index]}"
        worker_index=$((pending_index % ${#WORKER_GPUS[@]}))
        run_one_combo "${WORKER_GPUS[worker_index]}" "${job_index}" "$@"
    done
    echo "Dry run completed; no files were written."
    exit 0
fi

if (( ${#PENDING_INDICES[@]} == 0 )); then
    echo "All ${#COMBOS[@]} manifest runs are already complete."
    exit 0
fi

pids=()
for worker_index in "${!WORKER_GPUS[@]}"; do
    run_worker "${worker_index}" "$@" &
    pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failures=1
    fi
done

if (( failures != 0 )); then
    echo "Search finished with one or more failed runs. Inspect the manifest log paths (retries use a .resume_<timestamp> suffix)." >&2
    exit 1
fi

echo "All ${#PENDING_INDICES[@]} pending hyperparameter runs completed successfully (${completed_count} were already complete)."
