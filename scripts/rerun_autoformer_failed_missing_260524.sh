#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-/home/gpuadmin/anaconda3/envs/tslib/bin/python}"
GPU="${GPU:-0}"
N_JOBS="${N_JOBS:-4}"
LOG_DIR="${LOG_DIR:-logs/multi_backbone_proxy_rerun_260524}"
MANIFEST="${MANIFEST:-logs/multi_backbone_proxy_rerun_260524_manifest.jsonl}"

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR" "$(dirname "$MANIFEST")"

run_subset() {
  "$PYTHON_CMD" scripts/multi_backbone_proxy/run_candidates.py \
    --candidates candidates/autoformer_100_sl96.json \
    --repo-root "$REPO_ROOT" \
    --backbones Autoformer \
    --pred-lens 96 \
    --dataset-pred-lens ILI=36 \
    --dataset-seq-len ILI=36 \
    --python-cmd "$PYTHON_CMD" \
    --gpu "$GPU" \
    --log-dir "$LOG_DIR" \
    --manifest "$MANIFEST" \
    --n_jobs "$N_JOBS" \
    --execute \
    --keep-going \
    "$@"
}

# Failed in the original run: 00557, 00575, 00581, 00587.
run_subset \
  --datasets Traffic \
  --candidate-ids Autoformer_092 Autoformer_095 Autoformer_096 Autoformer_097

# Failed in the original run: 00589, 00593, 00594.
run_subset \
  --datasets ECL Traffic Weather \
  --candidate-ids Autoformer_098

# Failed original job 00595 plus original jobs 00596-00600 that never started.
run_subset \
  --datasets ECL ETTh1 Exchange ILI Traffic Weather \
  --candidate-ids Autoformer_099
