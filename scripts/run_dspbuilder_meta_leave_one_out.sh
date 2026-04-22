#!/usr/bin/env bash

set -euo pipefail

DATASETS_CSV="${DATASETS_CSV:-Weather,Traffic,ECL,Etth1,Exchange,M4_Hourly,M4_Monthly,M4_Quarterly,M4_Weekly,ILI,M4_Daily,M4_Yearly}"
BENCHMARK_DIR="${BENCHMARK_DIR:-./benchmark}"
CANDIDATE_DIR="${CANDIDATE_DIR:-./candidates}"
DEVICE="${DEVICE:-cuda:0}"
CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./meta_checkpoints/dspbuilder_meta_leave_one_out}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
TRAIN_ONLY="${TRAIN_ONLY:-1}"
TEST_DATASETS_CSV="${TEST_DATASETS_CSV:-}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a ALL_DATASETS <<< "${DATASETS_CSV}"

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

if [[ -n "${CONDA_ENV_NAME}" ]]; then
    PYTHON_CMD=(conda run -n "${CONDA_ENV_NAME}" python)
else
    PYTHON_CMD=(python)
fi

SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="${OUTPUT_ROOT%/}/suite_${SUITE_STAMP}"

echo "Suite directory: ${SUITE_DIR}"
echo "Datasets: ${DATASETS_CSV}"
echo "Device: ${DEVICE}"
echo "Train-only: ${TRAIN_ONLY}"

for ((idx = 0; idx < ${#ALL_DATASETS[@]}; idx++)); do
    val_dataset="${ALL_DATASETS[idx]}"
    train_datasets=()

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
        --device "${DEVICE}"
        --cls-loss-weight "${CLS_LOSS_WEIGHT}"
        --output-dir "${run_output_dir}"
    )

    if [[ "${TRAIN_ONLY}" == "1" ]]; then
        cmd+=(--train-only)
    else
        cmd+=(--test-datasets "${TEST_DATASETS_CSV}")
    fi

    if [[ "$#" -gt 0 ]]; then
        cmd+=("$@")
    fi

    printf '\n[%02d/%02d] val=%s\n' "$((idx + 1))" "${#ALL_DATASETS[@]}" "${val_dataset}"
    echo "run_name=${run_name}"
    echo "train=${train_csv}"
    echo "output_dir=${run_output_dir}"
    printf 'command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if [[ "${DRY_RUN}" == "1" ]]; then
        continue
    fi

    "${cmd[@]}"
done
