#!/usr/bin/env bash

set -euo pipefail

DATASETS_CSV="${DATASETS_CSV:-Weather,Traffic,ECL,Etth1,Exchange,ILI}"
# DATASETS_CSV="${DATASETS_CSV:-Weather,Traffic,ECL,Etth1,Exchange,ILI,M4_Hourly,M4_Monthly,M4_Quarterly,M4_Daily,M4_Weekly,M4_Yearly}"
BENCHMARK_DIR="${BENCHMARK_DIR:-./benchmark}"
CANDIDATE_DIR="${CANDIDATE_DIR:-./candidates}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./meta_checkpoints/dspbuilder_meta_leave_one_out}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
TRAIN_ONLY="${TRAIN_ONLY:-1}"
TEST_DATASETS_CSV="${TEST_DATASETS_CSV:-}"
DRY_RUN="${DRY_RUN:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-${gpu_id:-0}}}"

# train_dspbuilder_meta.py hyperparameters.
# Override any value from the shell, e.g.
# EPOCHS=200 WEIGHT_HEAD_LAYERS=3 SAMPLE_ENCODER_NORM=1 SET_ENCODER_NORM=1 RAW_STAT_EMB=1 bash scripts/run_dspbuilder_meta_leave_one_out.sh
EPOCHS="${EPOCHS:-100}"
ITERATIONS_PER_EPOCH="${ITERATIONS_PER_EPOCH:-40}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
# Validation loss uses one fixed support set with the original fixed 5 non-overlapping query batches.
VAL_ITERATIONS_PER_DATASET="5"
EVAL_ITERATIONS_PER_DATASET="${EVAL_ITERATIONS_PER_DATASET:-5}"
SUPPORT_SIZE="${SUPPORT_SIZE:-5}"
TRAIN_QUERY_SIZE="${TRAIN_QUERY_SIZE:-20}"
VAL_QUERY_SIZE="${VAL_QUERY_SIZE:-20}"
TEST_QUERY_SIZE="${TEST_QUERY_SIZE:-10}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
WEIGHT_HEAD_LAYERS="${WEIGHT_HEAD_LAYERS:-1}"
MLP_NORM="${MLP_NORM:-0}"
ENCODER_HIDDEN_DIM="${ENCODER_HIDDEN_DIM:-64}"
NUMBER_OF_CONV1D_LAYER="${NUMBER_OF_CONV1D_LAYER:-1}"
SAMPLE_ENCODER_NORM="${SAMPLE_ENCODER_NORM:-0}"
NUMBER_OF_SETENCODER_MLP_LAYERS="${NUMBER_OF_SETENCODER_MLP_LAYERS:-1}"
SET_ENCODER_NORM="${SET_ENCODER_NORM:-0}"
RAW_STAT_EMB="${RAW_STAT_EMB:-0}"
DROPOUT="${DROPOUT:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-0}"
PROXY_SIGNATURE_REGRESSION="${PROXY_SIGNATURE_REGRESSION:-0}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda:0}"

IFS=',' read -r -a ALL_DATASETS <<< "${DATASETS_CSV}"
GPU_DEVICES=()

if [[ -n "${GPU_IDS}" ]]; then
    IFS=',' read -r -a RAW_GPU_IDS <<< "${GPU_IDS}"
    for raw_gpu_id in "${RAW_GPU_IDS[@]}"; do
        gpu_id="${raw_gpu_id//[[:space:]]/}"
        if [[ -z "${gpu_id}" ]]; then
            continue
        fi
        if [[ "${gpu_id}" == *":"* || "${gpu_id}" == "cpu" || "${gpu_id}" == "mps" ]]; then
            GPU_DEVICES+=("${gpu_id}")
        else
            GPU_DEVICES+=("cuda:${gpu_id}")
        fi
    done

    if [[ "${#GPU_DEVICES[@]}" -eq 0 ]]; then
        echo "GPU_IDS was set but no valid device ids were parsed: ${GPU_IDS}" >&2
        exit 1
    fi
fi

if [[ "${#ALL_DATASETS[@]}" -lt 2 ]]; then
    echo "Need at least two datasets in DATASETS_CSV." >&2
    exit 1
fi

if [[ "${TRAIN_ONLY}" != "1" && -z "${TEST_DATASETS_CSV}" ]]; then
    echo "Set TEST_DATASETS_CSV when TRAIN_ONLY is not 1." >&2
    exit 1
fi

join_by_comma() {
    local IFS=','
    echo "$*"
}

append_bool_flag() {
    local value="$1"
    local true_flag="$2"
    local false_flag="$3"

    case "${value}" in
        1|true|TRUE|yes|YES|on|ON)
            cmd+=("${true_flag}")
            ;;
        0|false|FALSE|no|NO|off|OFF)
            cmd+=("${false_flag}")
            ;;
        *)
            echo "Boolean value must be 0/1, true/false, yes/no, or on/off: ${value}" >&2
            exit 1
            ;;
    esac
}

run_fold() {
    local idx="$1"
    local run_device="$2"
    local val_dataset="${ALL_DATASETS[idx]}"
    local train_datasets=()
    local dataset
    local train_csv
    local run_name
    local run_output_dir
    local command_line
    local launch_message
    local -a cmd

    for dataset in "${ALL_DATASETS[@]}"; do
        if [[ "${dataset}" != "${val_dataset}" ]]; then
            train_datasets+=("${dataset}")
        fi
    done

    train_csv="$(join_by_comma "${train_datasets[@]}")"
    run_name="$(printf '%02d_val_%s' "$((idx + 1))" "${val_dataset}")"
    run_output_dir="${SUITE_DIR}/${run_name}"

    cmd=(
        "${PYTHON_CMD[@]}"
        train_dspbuilder_meta.py
        --benchmark-dir "${BENCHMARK_DIR}"
        --candidate-dir "${CANDIDATE_DIR}"
        --train-datasets "${train_csv}"
        --val-datasets "${val_dataset}"
        --epochs "${EPOCHS}"
        --iterations-per-epoch "${ITERATIONS_PER_EPOCH}"
        --train-batch-size "${TRAIN_BATCH_SIZE}"
        --val-iterations-per-dataset "${VAL_ITERATIONS_PER_DATASET}"
        --eval-iterations-per-dataset "${EVAL_ITERATIONS_PER_DATASET}"
        --support-size "${SUPPORT_SIZE}"
        --train-query-size "${TRAIN_QUERY_SIZE}"
        --val-query-size "${VAL_QUERY_SIZE}"
        --test-query-size "${TEST_QUERY_SIZE}"
        --hidden-dim "${HIDDEN_DIM}"
        --weight-head-layers "${WEIGHT_HEAD_LAYERS}"
        --encoder-hidden-dim "${ENCODER_HIDDEN_DIM}"
        --number-of-conv1d-layer "${NUMBER_OF_CONV1D_LAYER}"
        --dropout "${DROPOUT}"
        --learning-rate "${LEARNING_RATE}"
        --weight-decay "${WEIGHT_DECAY}"
        --cls-loss-weight "${CLS_LOSS_WEIGHT}"
        --patience "${PATIENCE}"
        --seed "${SEED}"
        --device "${run_device}"
        --output-dir "${run_output_dir}"
    )

    if [[ -n "${NUMBER_OF_SETENCODER_MLP_LAYERS}" ]]; then
        cmd+=(--number-of-setencoder-mlp-layers "${NUMBER_OF_SETENCODER_MLP_LAYERS}")
    fi

    append_bool_flag "${SAMPLE_ENCODER_NORM}" "--sample_encoder_norm" "--no-sample_encoder_norm"
    append_bool_flag "${SET_ENCODER_NORM}" "--set_encoder_norm" "--no-set_encoder_norm"
    append_bool_flag "${MLP_NORM}" "--mlp_norm" "--no-mlp_norm"
    append_bool_flag "${RAW_STAT_EMB}" "--raw_stat_emb" "--no-raw_stat_emb"
    append_bool_flag "${PROXY_SIGNATURE_REGRESSION}" "--proxy-signature-regression" "--no-proxy-signature-regression"

    if [[ "${TRAIN_ONLY}" == "1" ]]; then
        cmd+=(--train-only)
    else
        cmd+=(--test-datasets "${TEST_DATASETS_CSV}")
    fi

    if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    printf -v command_line ' %q' "${cmd[@]}"
    printf -v launch_message '\n[%02d/%02d] val=%s device=%s\nrun_name=%s\ntrain=%s\noutput_dir=%s\ncommand:%s\n' \
        "$((idx + 1))" \
        "${#ALL_DATASETS[@]}" \
        "${val_dataset}" \
        "${run_device}" \
        "${run_name}" \
        "${train_csv}" \
        "${run_output_dir}" \
        "${command_line}"
    printf '%s' "${launch_message}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    "${cmd[@]}"
}

run_worker() {
    local worker_index="$1"
    local run_device="$2"
    local idx

    for ((idx = worker_index; idx < ${#ALL_DATASETS[@]}; idx += ${#GPU_DEVICES[@]})); do
        run_fold "${idx}" "${run_device}"
    done
}

run_postprocess() {
    local loss_csv="${SUITE_DIR}/loss.csv"
    local spearman_csv="${SUITE_DIR}/spearman.csv"
    local loss_png="${SUITE_DIR}/loss.png"
    local spearman_png="${SUITE_DIR}/spearman.png"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "Dry-run: skipping validation CSV/plot export."
        return 0
    fi

    echo "Exporting validation summaries under ${SUITE_DIR}"
    "${PYTHON_CMD[@]}" scripts/export_dspbuilder_meta_validation_loss_csv.py \
        --base-dir "${SUITE_DIR}" \
        --output "${loss_csv}"
    "${PYTHON_CMD[@]}" scripts/export_dspbuilder_meta_validation_spearman_csv.py \
        --base-dir "${SUITE_DIR}" \
        --output "${spearman_csv}"

    echo "Plotting validation summaries under ${SUITE_DIR}"
    "${PYTHON_CMD[@]}" scripts/plot_dspbuilder_meta_validation_loss.py \
        --csv "${loss_csv}" \
        --output "${loss_png}"
    "${PYTHON_CMD[@]}" scripts/plot_dspbuilder_meta_validation_spearman.py \
        --csv "${spearman_csv}" \
        --output "${spearman_png}"

    echo "Post-processing outputs:"
    echo "  ${loss_csv}"
    echo "  ${spearman_csv}"
    echo "  ${loss_png}"
    echo "  ${spearman_png}"
}

if [[ -n "${CONDA_ENV_NAME}" ]]; then
    PYTHON_CMD=(conda run -n "${CONDA_ENV_NAME}" python)
else
    PYTHON_CMD=(python)
fi

SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="${OUTPUT_ROOT%/}/suite_${SUITE_STAMP}"
EXTRA_ARGS=("$@")

echo "Suite directory: ${SUITE_DIR}"
echo "Datasets: ${DATASETS_CSV}"
if [[ "${#GPU_DEVICES[@]}" -gt 0 ]]; then
    echo "Devices: ${GPU_DEVICES[*]} (parallel workers=${#GPU_DEVICES[@]})"
else
    echo "Device: ${DEVICE}"
fi
echo "Train-only: ${TRAIN_ONLY}"
echo "Hyperparameters:"
echo "  epochs=${EPOCHS} iterations_per_epoch=${ITERATIONS_PER_EPOCH} train_batch_size=${TRAIN_BATCH_SIZE}"
echo "  val_iterations_per_dataset=${VAL_ITERATIONS_PER_DATASET} eval_iterations_per_dataset=${EVAL_ITERATIONS_PER_DATASET}"
echo "  support_size=${SUPPORT_SIZE} train_query_size=${TRAIN_QUERY_SIZE} val_query_size=${VAL_QUERY_SIZE} test_query_size=${TEST_QUERY_SIZE}"
echo "  hidden_dim=${HIDDEN_DIM} weight_head_layers=${WEIGHT_HEAD_LAYERS} mlp_norm=${MLP_NORM} encoder_hidden_dim=${ENCODER_HIDDEN_DIM}"
echo "  number_of_conv1d_layer=${NUMBER_OF_CONV1D_LAYER} sample_encoder_norm=${SAMPLE_ENCODER_NORM} raw_stat_emb=${RAW_STAT_EMB}"
echo "  number_of_setencoder_mlp_layers=${NUMBER_OF_SETENCODER_MLP_LAYERS:-legacy} set_encoder_norm=${SET_ENCODER_NORM}"
echo "  dropout=${DROPOUT} learning_rate=${LEARNING_RATE} weight_decay=${WEIGHT_DECAY}"
echo "  cls_loss_weight=${CLS_LOSS_WEIGHT} proxy_signature_regression=${PROXY_SIGNATURE_REGRESSION} patience=${PATIENCE} seed=${SEED}"

if [[ "${#GPU_DEVICES[@]}" -gt 0 ]]; then
    mkdir -p "${SUITE_DIR}/worker_logs"

    pids=()
    for ((worker_index = 0; worker_index < ${#GPU_DEVICES[@]}; worker_index++)); do
        if [[ "${worker_index}" -eq 0 ]]; then
            run_worker "${worker_index}" "${GPU_DEVICES[worker_index]}" &
        else
            log_device="${GPU_DEVICES[worker_index]//:/_}"
            log_device="${log_device//\//_}"
            worker_log="${SUITE_DIR}/worker_logs/worker_${worker_index}_${log_device}.log"
            run_worker "${worker_index}" "${GPU_DEVICES[worker_index]}" >"${worker_log}" 2>&1 &
        fi
        pids+=("$!")
    done

    exit_code=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            exit_code=1
        fi
    done
    if [[ "${exit_code}" -eq 0 ]]; then
        echo "All GPU workers completed."
        run_postprocess
    else
        echo "One or more GPU workers failed. Check logs under ${SUITE_DIR}/worker_logs." >&2
    fi
    exit "${exit_code}"
fi

for ((idx = 0; idx < ${#ALL_DATASETS[@]}; idx++)); do
    run_fold "${idx}" "${DEVICE}"
done

run_postprocess
