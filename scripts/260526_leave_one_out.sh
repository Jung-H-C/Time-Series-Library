#!/usr/bin/env bash
set -euo pipefail

DATASETS_CSV="${DATASETS_CSV:-ECL,Etth1,Traffic,Weather,ILI,Exchange}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
DRY_RUN="${DRY_RUN:-0}"
SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-./logs/new_dspbuilder_meta_6dataset_leave_one_out_${SUITE_STAMP}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${CONDA_ENV_NAME}" ]]; then
    PYTHON_CMD=(conda run -n "${CONDA_ENV_NAME}" python)
else
    PYTHON_CMD=(python)
fi

IFS=',' read -r -a ALL_DATASETS <<< "${DATASETS_CSV}"
if [[ "${#ALL_DATASETS[@]}" -ne 6 ]]; then
    echo "Expected exactly 6 datasets in DATASETS_CSV, got ${#ALL_DATASETS[@]}: ${DATASETS_CSV}" >&2
    exit 1
fi

join_by_comma() {
    local IFS=','
    echo "$*"
}

run_fold() {
    local idx="$1"
    local test_dataset="${ALL_DATASETS[idx]}"
    local train_datasets=()
    local dataset
    local train_csv
    local -a cmd
    local command_line
    local log_file

    for dataset in "${ALL_DATASETS[@]}"; do
        if [[ "${dataset}" != "${test_dataset}" ]]; then
            train_datasets+=("${dataset}")
        fi
    done

    train_csv="$(join_by_comma "${train_datasets[@]}")"
    log_file="${LOG_DIR}/$(printf '%02d_test_%s.log' "$((idx + 1))" "${test_dataset}")"

    cmd=(
        "${PYTHON_CMD[@]}"
        train_new_dspbuilder_meta.py
        --train-datasets "${train_csv}"
        --test-datasets "${test_dataset}"
        --stratified
        --weight-head-layers 3
        --number-of-conv1d-layer 3
        --number-of-setencoder-mlp-layers 3
    )

    printf -v command_line ' %q' "${cmd[@]}"
    printf '\n[%02d/%02d] test=%s\ntrain=%s\nlog=%s\ncommand:%s\n' \
        "$((idx + 1))" \
        "${#ALL_DATASETS[@]}" \
        "${test_dataset}" \
        "${train_csv}" \
        "${log_file}" \
        "${command_line}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    mkdir -p "${LOG_DIR}"
    "${cmd[@]}" 2>&1 | tee "${log_file}"
}

echo "Datasets: ${DATASETS_CSV}"
echo "Log directory: ${LOG_DIR}"
echo "Fixed options: --stratified --weight-head-layers 3 --number-of-conv1d-layer 3 --number-of-setencoder-mlp-layers 3"
echo "Other train_new_dspbuilder_meta.py options are left at their parser defaults."

for ((idx = 0; idx < ${#ALL_DATASETS[@]}; idx++)); do
    run_fold "${idx}"
done

echo
echo "Completed 6-dataset leave-one-out run."
