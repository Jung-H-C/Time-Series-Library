#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
if [[ -n "${CONDA_ENV_NAME}" ]]; then
    PYTHON_CMD=(conda run -n "${CONDA_ENV_NAME}" python)
else
    PYTHON_CMD=("${PYTHON_BIN}")
fi

SEARCH_STAMP="${SEARCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEARCH_NAME="${SEARCH_NAME:-dcspg_lodo_hparam_${SEARCH_STAMP}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-DCSPG/checkpoints/lodo_hparam_search/${SEARCH_NAME}}"
RESULT_DIR="${RESULT_DIR:-${CHECKPOINT_ROOT}/csv}"
LOG_DIR="${LOG_DIR:-logs/${SEARCH_NAME}}"

GPU_IDS=(${GPU_IDS:-0 1 2 3})
D_MODEL_VALUES=(${D_MODEL_VALUES:-64 128})
N_HEAD_VALUES=(${N_HEAD_VALUES:-2 4})
ENCODER_LAYER_VALUES_RAW="${ENCODER_LAYER_VALUES:-${ENCODER_LAYERS:-1 2 3}}"
DECODER_LAYER_VALUES_RAW="${DECODER_LAYER_VALUES:-${DECODER_LAYERS:-1 2 3}}"
ENCODER_LAYER_VALUES=(${ENCODER_LAYER_VALUES_RAW})
DECODER_LAYER_VALUES=(${DECODER_LAYER_VALUES_RAW})

TRAIN_BUDGET_STEPS="${TRAIN_BUDGET_STEPS:-2000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
CHECK_EVERY="${CHECK_EVERY:-100}"
LOSS_WINDOW="${LOSS_WINDOW:-100}"
LOSS_LOG_INTERVAL="${LOSS_LOG_INTERVAL:-100}"
LOG_EVERY="${LOG_EVERY:-100}"
TRAIN_SEED="${TRAIN_SEED:-2026}"
TEST_SEED="${TEST_SEED:-2026}"
TEST_SPLIT="${TEST_SPLIT:-proxy_test}"
DEVICE="${DEVICE:-auto}"
TARGET_SAMPLING_STRATEGY="${TARGET_SAMPLING_STRATEGY:-cycle}"

BATCH_SIZE="${BATCH_SIZE:-32}"
K_SAMPLES="${K_SAMPLES:-16}"
BASE_EPISODES_PER_DATASET="${BASE_EPISODES_PER_DATASET:-6}"
EXTRA_EPISODES="${EXTRA_EPISODES:-2}"

DROPOUT="${DROPOUT:-0.1}"
MAX_FORMULA_LEN="${MAX_FORMULA_LEN:-16}"
MAX_STACK_DEPTH="${MAX_STACK_DEPTH:-8}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

DATASETS="${DATASETS:-}"
TEST_MAX_LEN="${TEST_MAX_LEN:-}"
MAX_CHECKPOINTS_PER_DATASET="${MAX_CHECKPOINTS_PER_DATASET:-}"
MULTIPLE_SEEDS="${MULTIPLE_SEEDS:-0}"
SEED_COUNT="${SEED_COUNT:-10}"
TEST_SEEDS="${TEST_SEEDS:-}"
BEAM_SEARCH="${BEAM_SEARCH:-0}"
BEAM_SIZE="${BEAM_SIZE:-5}"
EVAL_RESUME="${EVAL_RESUME:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

DRY_RUN="${DRY_RUN:-0}"
ALLOW_INVALID_COMBOS="${ALLOW_INVALID_COMBOS:-0}"

if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
    echo "GPU_IDS must contain at least one GPU id." >&2
    exit 1
fi

format_cmd() {
    local -a cmd=("$@")
    printf ' %q' "${cmd[@]}"
}

combo_name() {
    local d_model="$1"
    local n_heads="$2"
    local dim_feedforward="$3"
    local encoder_layers="$4"
    local decoder_layers="$5"
    printf 'dmodel%s_heads%s_enc%s_dec%s_ff%s' \
        "${d_model}" "${n_heads}" "${encoder_layers}" "${decoder_layers}" "${dim_feedforward}"
}

build_combos() {
    local d_model
    local n_heads
    local dim_feedforward
    local encoder_layers
    local decoder_layers
    COMBOS=()
    INVALID_COMBOS=()
    for d_model in "${D_MODEL_VALUES[@]}"; do
        for n_heads in "${N_HEAD_VALUES[@]}"; do
            dim_feedforward=$((d_model * 4))
            for encoder_layers in "${ENCODER_LAYER_VALUES[@]}"; do
                for decoder_layers in "${DECODER_LAYER_VALUES[@]}"; do
                    COMBOS+=("${d_model}:${n_heads}:${dim_feedforward}:${encoder_layers}:${decoder_layers}")
                    if (( d_model % n_heads != 0 )); then
                        INVALID_COMBOS+=(
                            "${d_model}:${n_heads}:${dim_feedforward}:${encoder_layers}:${decoder_layers}"
                        )
                    fi
                done
            done
        done
    done
}

validate_combos() {
    if [[ "${#INVALID_COMBOS[@]}" -eq 0 || "${ALLOW_INVALID_COMBOS}" == "1" ]]; then
        return 0
    fi

    echo "Invalid Transformer hyperparameter combination(s):" >&2
    local combo
    local d_model
    local n_heads
    local dim_feedforward
    local encoder_layers
    local decoder_layers
    for combo in "${INVALID_COMBOS[@]}"; do
        IFS=':' read -r d_model n_heads dim_feedforward encoder_layers decoder_layers <<< "${combo}"
        echo "  d_model=${d_model}, n_heads=${n_heads}, encoder_layers=${encoder_layers}, decoder_layers=${decoder_layers}, dim_feedforward=${dim_feedforward}" >&2
    done
    echo "PyTorch Transformer requires d_model to be divisible by n_heads." >&2
    echo "Adjust D_MODEL_VALUES/N_HEAD_VALUES, or set ALLOW_INVALID_COMBOS=1 to submit them anyway." >&2
    exit 2
}

run_one_combo() {
    local gpu_id="$1"
    local combo="$2"
    local d_model
    local n_heads
    local dim_feedforward
    local encoder_layers
    local decoder_layers
    IFS=':' read -r d_model n_heads dim_feedforward encoder_layers decoder_layers <<< "${combo}"

    local run_name
    run_name="$(combo_name "${d_model}" "${n_heads}" "${dim_feedforward}" "${encoder_layers}" "${decoder_layers}")"
    local run_dir="${CHECKPOINT_ROOT}/${run_name}"
    local output_csv="${RESULT_DIR}/${run_name}_checkpoint_test_results.csv"
    local train_log="${LOG_DIR}/${run_name}.train.log"
    local eval_log="${LOG_DIR}/${run_name}.eval.log"

    local -a train_cmd=(
        "${PYTHON_CMD[@]}"
        train_dcspg_framework.py
        --output-dir "${CHECKPOINT_ROOT}"
        --run-name "${run_name}"
        --device "${DEVICE}"
        --gpu-id "${gpu_id}"
        --seed "${TRAIN_SEED}"
        --target-sampling-strategy "${TARGET_SAMPLING_STRATEGY}"
        --batch-size "${BATCH_SIZE}"
        --k-samples "${K_SAMPLES}"
        --base-episodes-per-dataset "${BASE_EPISODES_PER_DATASET}"
        --extra-episodes "${EXTRA_EPISODES}"
        --d-model "${d_model}"
        --n-heads "${n_heads}"
        --encoder-layers "${encoder_layers}"
        --decoder-layers "${decoder_layers}"
        --dim-feedforward "${dim_feedforward}"
        --dropout "${DROPOUT}"
        --max-formula-len "${MAX_FORMULA_LEN}"
        --max-stack-depth "${MAX_STACK_DEPTH}"
        --learning-rate "${LEARNING_RATE}"
        --weight-decay "${WEIGHT_DECAY}"
        --grad-clip "${GRAD_CLIP}"
        --train-budget-steps "${TRAIN_BUDGET_STEPS}"
        --check-every "${CHECK_EVERY}"
        --checkpoint-interval "${CHECKPOINT_INTERVAL}"
        --loss-window "${LOSS_WINDOW}"
        --loss-log-interval "${LOSS_LOG_INTERVAL}"
        --log-every "${LOG_EVERY}"
    )

    local -a eval_cmd=(
        "${PYTHON_CMD[@]}"
        evaluate_lodo_checkpoints.py
        --checkpoint-root "${run_dir}"
        --output-csv "${output_csv}"
        --device "${DEVICE}"
        --gpu-id "${gpu_id}"
        --k-samples "${K_SAMPLES}"
        --test-seed "${TEST_SEED}"
        --test-split "${TEST_SPLIT}"
    )

    if [[ -n "${DATASETS}" ]]; then
        eval_cmd+=(--datasets "${DATASETS}")
    fi
    if [[ -n "${TEST_MAX_LEN}" ]]; then
        eval_cmd+=(--test-max-len "${TEST_MAX_LEN}")
    fi
    if [[ -n "${MAX_CHECKPOINTS_PER_DATASET}" ]]; then
        eval_cmd+=(--max-checkpoints-per-dataset "${MAX_CHECKPOINTS_PER_DATASET}")
    fi
    if [[ "${MULTIPLE_SEEDS}" == "1" ]]; then
        eval_cmd+=(--multiple-seeds --seed-count "${SEED_COUNT}")
    fi
    if [[ -n "${TEST_SEEDS}" ]]; then
        eval_cmd+=(--test-seeds "${TEST_SEEDS}")
    fi
    if [[ "${BEAM_SEARCH}" == "1" ]]; then
        eval_cmd+=(--beam-search --beam-size "${BEAM_SIZE}")
    fi
    if [[ "${EVAL_RESUME}" == "1" ]]; then
        eval_cmd+=(--resume)
    fi
    if [[ "${FAIL_FAST}" == "1" ]]; then
        eval_cmd+=(--fail-fast)
    fi

    echo "[GPU ${gpu_id}] ${run_name}: train -> eval"
    echo "  run_dir=${run_dir}"
    echo "  output_csv=${output_csv}"
    echo "  train_log=${train_log}"
    echo "  eval_log=${eval_log}"
    echo "  train_cmd:$(format_cmd "${train_cmd[@]}")"
    echo "  eval_cmd:$(format_cmd "${eval_cmd[@]}")"

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    if [[ -e "${run_dir}" ]]; then
        echo "[GPU ${gpu_id}] ${run_name}: run directory already exists: ${run_dir}" >&2
        echo "Use a new SEARCH_STAMP/SEARCH_NAME/CHECKPOINT_ROOT, or remove the existing run directory." >&2
        return 1
    fi

    mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
    "${train_cmd[@]}" > "${train_log}" 2>&1
    "${eval_cmd[@]}" > "${eval_log}" 2>&1
    echo "[GPU ${gpu_id}] ${run_name}: completed; csv=${output_csv}"
}

run_worker() {
    local worker_index="$1"
    local gpu_id="${GPU_IDS[worker_index]}"
    local job_index
    for ((job_index = worker_index; job_index < ${#COMBOS[@]}; job_index += ${#GPU_IDS[@]})); do
        run_one_combo "${gpu_id}" "${COMBOS[job_index]}"
    done
}

MANIFEST_CSV="${RESULT_DIR}/hparam_manifest.csv"

build_combos
validate_combos

if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${CHECKPOINT_ROOT}"
    {
        echo "run_name,d_model,n_heads,encoder_layers,decoder_layers,dim_feedforward,checkpoint_root,result_csv"
        for combo in "${COMBOS[@]}"; do
            IFS=':' read -r d_model n_heads dim_feedforward encoder_layers decoder_layers <<< "${combo}"
            run_name="$(combo_name "${d_model}" "${n_heads}" "${dim_feedforward}" "${encoder_layers}" "${decoder_layers}")"
            echo "${run_name},${d_model},${n_heads},${encoder_layers},${decoder_layers},${dim_feedforward},${CHECKPOINT_ROOT}/${run_name},${RESULT_DIR}/${run_name}_checkpoint_test_results.csv"
        done
    } > "${MANIFEST_CSV}"
fi

echo "DCSPG LODO hyperparameter search"
echo "  combos=${#COMBOS[@]}"
echo "  gpu_ids=${GPU_IDS[*]}"
echo "  d_model_values=${D_MODEL_VALUES[*]}"
echo "  n_head_values=${N_HEAD_VALUES[*]}"
echo "  encoder_layer_values=${ENCODER_LAYER_VALUES[*]}"
echo "  decoder_layer_values=${DECODER_LAYER_VALUES[*]}"
echo "  checkpoint_root=${CHECKPOINT_ROOT}"
echo "  result_dir=${RESULT_DIR}"
echo "  log_dir=${LOG_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  manifest=${MANIFEST_CSV} (not written in dry run)"
else
    echo "  manifest=${MANIFEST_CSV}"
fi
echo "  train_budget_steps=${TRAIN_BUDGET_STEPS}"

if [[ "${DRY_RUN}" == "1" ]]; then
    gpu_index=0
    for job_index in "${!COMBOS[@]}"; do
        gpu_index=$((job_index % ${#GPU_IDS[@]}))
        run_one_combo "${GPU_IDS[gpu_index]}" "${COMBOS[job_index]}"
    done
    echo "Dry run completed; no files were written."
    exit 0
fi

pids=()
for worker_index in "${!GPU_IDS[@]}"; do
    (
        run_worker "${worker_index}"
    ) &
    pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failures=1
    fi
done

if [[ "${failures}" -ne 0 ]]; then
    echo "One or more hyperparameter jobs failed. See logs in ${LOG_DIR}." >&2
    exit 1
fi

echo "Completed DCSPG LODO hyperparameter search."
