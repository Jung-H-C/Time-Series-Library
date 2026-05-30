#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
NUM_BATCHES="${NUM_BATCHES:-5}"
PROXIES="${PROXIES:-synflow}"
EXTRA_ARGS=("$@")

DATASET_NAMES=(
  "etth1"
  "exchange"
  "ili"
  "m4_daily"
  "m4_hourly"
  "m4_monthly"
  "m4_quarterly"
  "m4_weekly"
  "m4_yearly"
  "traffic"
  "weather"
)

CANDIDATE_FILES=(
  "candidates/DSPBuilder_Mamba_etth1_candidates.json"
  "candidates/DSPBuilder_Mamba_exchange_candidates.json"
  "candidates/DSPBuilder_Mamba_ILI_candidates.json"
  "candidates/DSPBuilder_Mamba_M4_Daily_candidates.json"
  "candidates/DSPBuilder_Mamba_M4_Hourly_candidates.json"
  "candidates/DSPBuilder_Mamba_M4_Monthly_candidates.json"
  "candidates/DSPBuilder_Mamba_M4_Quarterly_candidates.json"
  "candidates/DSPBuilder_Mamba_M4_Weekly_candidates.json"
  "candidates/DSPBuilder_Mamba_M4_Yearly_candidates.json"
  "candidates/DSPBuilder_Mamba_Traffic_candidates.json"
  "candidates/DSPBuilder_Mamba_Weather_candidates.json"
)

cd "${PROJECT_ROOT}"

for idx in "${!DATASET_NAMES[@]}"; do
  dataset_name="${DATASET_NAMES[$idx]}"
  candidates_file="${CANDIDATE_FILES[$idx]}"

  if [[ ! -f "${candidates_file}" ]]; then
    echo "[ERROR] Missing candidates file: ${candidates_file}" >&2
    exit 1
  fi

  echo
  echo "[$((idx + 1))/${#DATASET_NAMES[@]}] Running synflow for ${dataset_name}"
  echo "  candidates-file: ${candidates_file}"

  python score_candidates.py \
    --candidates-file "${candidates_file}" \
    --num-batches "${NUM_BATCHES}" \
    --gpu-id "${GPU_ID}" \
    --proxies "${PROXIES}" \
    --deterministic \
    "${EXTRA_ARGS[@]}"
done
