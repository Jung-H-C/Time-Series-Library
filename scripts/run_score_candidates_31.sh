#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_BATCHES="${NUM_BATCHES:-5}"
GPU_IDS="${GPU_IDS:-1}"

CANDIDATE_FILES=(
  "candidates/DSPBuilder_Mamba_Aus_Electricity_Demand_candidates.json"
  "candidates/DSPBuilder_Mamba_Bitcoin_candidates.json"
  "candidates/DSPBuilder_Mamba_CIF_2016_candidates.json"
  "candidates/DSPBuilder_Mamba_Dominick_TSF_candidates.json"
  "candidates/DSPBuilder_Mamba_Electricity_Hourly_candidates.json"
  "candidates/DSPBuilder_Mamba_Electricity_Weekly_candidates.json"
  "candidates/DSPBuilder_Mamba_FRED_MD_candidates.json"
  "candidates/DSPBuilder_Mamba_KDD_Cup_2018_candidates.json"
  "candidates/DSPBuilder_Mamba_London_Smart_Meters_candidates.json"
  "candidates/DSPBuilder_Mamba_M1_Monthly_candidates.json"
  "candidates/DSPBuilder_Mamba_M1_Quarterly_candidates.json"
  "candidates/DSPBuilder_Mamba_M1_Yearly_candidates.json"
  "candidates/DSPBuilder_Mamba_M3_Monthly_candidates.json"
  "candidates/DSPBuilder_Mamba_M3_Quarterly_candidates.json"
  "candidates/DSPBuilder_Mamba_M3_Yearly_candidates.json"
  "candidates/DSPBuilder_Mamba_NN5_candidates.json"
  "candidates/DSPBuilder_Mamba_Pedestrian_Counts_candidates.json"
  "candidates/DSPBuilder_Mamba_Rideshare_candidates.json"
  "candidates/DSPBuilder_Mamba_San_Francisco_Traffic_Hourly_candidates.json"
  "candidates/DSPBuilder_Mamba_Saugeen_River_Flow_candidates.json"
  "candidates/DSPBuilder_Mamba_Solar_10min_candidates.json"
  "candidates/DSPBuilder_Mamba_Solar_Power_candidates.json"
  "candidates/DSPBuilder_Mamba_Sunspot_candidates.json"
  "candidates/DSPBuilder_Mamba_Temperature_Rain_candidates.json"
  "candidates/DSPBuilder_Mamba_Tourism_Monthly_candidates.json"
  "candidates/DSPBuilder_Mamba_US_Births_candidates.json"
  "candidates/DSPBuilder_Mamba_Vehicle_Trips_candidates.json"
  "candidates/DSPBuilder_Mamba_Weather_TSF_candidates.json"
  "candidates/DSPBuilder_Mamba_Web_Traffic_candidates.json"
  "candidates/DSPBuilder_Mamba_Wind_Farms_candidates.json"
  "candidates/DSPBuilder_Mamba_Wind_Power_candidates.json"
)

echo "Scoring ${#CANDIDATE_FILES[@]} candidate files with GPU_IDS=${GPU_IDS}, NUM_BATCHES=${NUM_BATCHES}"

for candidate_file in "${CANDIDATE_FILES[@]}"; do
  if [[ ! -f "${candidate_file}" ]]; then
    echo "Missing candidate file: ${candidate_file}" >&2
    exit 1
  fi

  echo "===== Start: ${candidate_file} ====="
  "${PYTHON_BIN}" score_candidates.py \
    --candidates-file "${candidate_file}" \
    --num-batches "${NUM_BATCHES}" \
    --gpu-id "${GPU_IDS}" \
    --deterministic
  echo "===== Done: ${candidate_file} ====="
done

echo "All candidate proxy scoring jobs completed."
