#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CANDIDATES="${CANDIDATES:-autoformer_300_sl96.json}"
GPU_IDS=(${GPU_IDS:-3})
DATASETS=(ECL ETTh1 Exchange ILI Traffic Weather)

mkdir -p proxy_scores

for dataset in "${DATASETS[@]}"; do
  csv_path="proxy_scores/autoformer_300_sl96_${dataset}_proxy_scores.csv"
  echo "Scoring ${dataset}; output prefix: ${csv_path}"
  "${PYTHON_BIN}" score_candidates.py \
    --candidates "${CANDIDATES}" \
    --dataset "${dataset}" \
    --csv-path "${csv_path}" \
    --gpu-id "${GPU_IDS[@]}" \
    --deterministic
done
