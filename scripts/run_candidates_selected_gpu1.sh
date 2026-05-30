#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-1}"
LOG_DIR="${LOG_DIR:-./log}"

mkdir -p "${LOG_DIR}"

DATASETS=(
  "Electricity_Hourly"
  "Electricity_Weekly"
  "FRED_MD"
  "KDD_Cup_2018"
  "London_Smart_Meters"
  "M1_Monthly"
  "M1_Quarterly"
)

for dataset in "${DATASETS[@]}"; do
  candidate_file="candidates/DSPBuilder_Mamba_${dataset}_candidates.json"
  log_file="${LOG_DIR}/260504_${dataset}.out"

  if [[ ! -f "${candidate_file}" ]]; then
    echo "Missing candidate file: ${candidate_file}" >&2
    exit 1
  fi

  echo "===== Start: ${dataset} ====="
  echo "Candidate: ${candidate_file}"
  echo "Log: ${log_file}"

  "${PYTHON_BIN}" sample_candidates.py \
    --run-candidates-file "${candidate_file}" \
    --gpu-id "${GPU_ID}" \
    > "${log_file}" 2>&1

  echo "===== Done: ${dataset} ====="
done

echo "All selected candidate runs completed."
