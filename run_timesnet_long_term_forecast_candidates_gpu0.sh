#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

GPU_ID="${1:-0}"

CANDIDATE_CONFIGS=(
  "TimesNet_long_term_forecast_ETTh2"
)

cd "${REPO_ROOT}"

for config_name in "${CANDIDATE_CONFIGS[@]}"; do
  printf '\n[%s] Running %s on GPU %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${config_name}" "${GPU_ID}"
  python sample_candidates.py \
    --run-candidates "${config_name}" \
    --gpu-id "${GPU_ID}"
done

printf '\n[%s] Finished all TimesNet long_term_forecast ETT candidate runs on GPU %s.\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" \
  "${GPU_ID}"
