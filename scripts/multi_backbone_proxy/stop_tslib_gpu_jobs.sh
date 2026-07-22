#!/usr/bin/env bash

set -uo pipefail

WAIT_SECONDS="${WAIT_SECONDS:-5}"
RUN_CANDIDATES_PATTERN='scripts/multi_backbone_proxy/run_candidates.py'
PYTHON_PATH_SUFFIX='/envs/tslib_nightly/bin/python'

log() {
    printf '[stop-tslib-jobs] %s\n' "$*"
}

is_pid() {
    [[ "$1" =~ ^[0-9]+$ ]] && (( 1 < 10#$1 ))
}

is_alive() {
    kill -0 "$1" 2>/dev/null
}

signal_pids() {
    local signal="$1"
    local label="$2"
    shift 2

    local pid
    for pid in "$@"; do
        if is_pid "$pid" && is_alive "$pid"; then
            log "Sending ${signal} to ${label} PID ${pid}"
            kill "-${signal}" "$pid" 2>/dev/null ||
                log "WARNING: could not send ${signal} to PID ${pid}"
        fi
    done
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log 'ERROR: nvidia-smi was not found.'
    exit 1
fi

if ! command -v pgrep >/dev/null 2>&1; then
    log 'ERROR: pgrep was not found.'
    exit 1
fi

# Stop the scheduler first so that it cannot launch replacement workers while
# the existing GPU processes are being terminated.
mapfile -t manager_pids < <(
    pgrep -f "$RUN_CANDIDATES_PATTERN" 2>/dev/null || true
)

if ((${#manager_pids[@]})); then
    log "Found run_candidates.py manager PID(s): ${manager_pids[*]}"
    signal_pids TERM 'manager' "${manager_pids[@]}"
else
    log 'No run_candidates.py manager process found.'
fi

# Only select processes that nvidia-smi reports as GPU compute applications and
# whose executable path ends with the requested Conda-environment Python path.
mapfile -t gpu_pids < <(
    nvidia-smi --query-compute-apps=pid,process_name \
        --format=csv,noheader,nounits 2>/dev/null |
        awk -F ', *' -v suffix="$PYTHON_PATH_SUFFIX" '
            index($2, suffix) == length($2) - length(suffix) + 1 { print $1 }
        ' |
        sort -nu
)

if ((${#gpu_pids[@]})); then
    log "Found matching GPU Python PID(s): ${gpu_pids[*]}"
else
    log 'No matching GPU Python process found.'
fi

parent_pids=()
if ((${#gpu_pids[@]})); then
    mapfile -t parent_pids < <(
        for pid in "${gpu_pids[@]}"; do
            ps -o ppid= -p "$pid" 2>/dev/null || true
        done |
            awk '{$1=$1; if ($1 ~ /^[0-9]+$/ && $1 > 1) print $1}' |
            sort -nu
    )
fi

if ((${#parent_pids[@]})); then
    log "Found immediate parent PID(s): ${parent_pids[*]}"
fi

# Ask parents and GPU workers to exit cleanly first.
signal_pids TERM 'GPU parent' "${parent_pids[@]}"
signal_pids TERM 'GPU worker' "${gpu_pids[@]}"

log "Waiting ${WAIT_SECONDS} second(s) for graceful termination..."
sleep "$WAIT_SECONDS"

# Force-kill only the exact PIDs collected above that are still alive.
signal_pids KILL 'manager' "${manager_pids[@]}"
signal_pids KILL 'GPU parent' "${parent_pids[@]}"
signal_pids KILL 'GPU worker' "${gpu_pids[@]}"

remaining=0
for pid in "${manager_pids[@]}" "${parent_pids[@]}" "${gpu_pids[@]}"; do
    if is_pid "$pid" && is_alive "$pid"; then
        log "WARNING: PID ${pid} is still alive."
        remaining=1
    fi
done

if ((remaining)); then
    exit 1
fi

log 'Targeted processes have been terminated.'
