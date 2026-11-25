#!/usr/bin/env bash
# Guarded root test runner for the shared RTX 5080 host (or local 5060).
# Usage (as root, via `sudo -S bash p3_guarded_run.sh <plan_file> <log_file>`):
#   Each non-empty, non-# line of <plan_file> is:  <label> | <timeout_s> | <shell command>
#   Before EVERY test the GPU is checked for other compute processes; if busy, wait up to
#   WAIT_BUSY_S seconds, then skip the test (logged as SKIPPED_GPU_BUSY). BAR0 pulses are
#   never fired while another user's job is resident on the GPU.
set -uo pipefail

PLAN="$1"
LOG="$2"
WAIT_BUSY_S="${WAIT_BUSY_S:-600}"
OWNER="${RUN_OWNER:-}"

gpu_busy() {
    local apps
    apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
    [[ -n "$apps" ]] && { echo "$apps"; return 0; }
    return 1
}

wait_idle() {
    [[ "${SKIP_GUARD:-0}" == "1" ]] && return 0   # single-user local desktop GPU only
    local waited=0 apps
    while apps="$(gpu_busy)"; do
        if (( waited >= WAIT_BUSY_S )); then
            echo "[GUARD] GPU still busy after ${waited}s: ${apps//$'\n'/; }" | tee -a "$LOG"
            return 1
        fi
        echo "[GUARD] GPU busy (${apps//$'\n'/; }), waiting..." | tee -a "$LOG"
        sleep 30
        waited=$((waited + 30))
    done
    return 0
}

echo "=== guarded run start $(date -Is) host=$(hostname) plan=$PLAN ===" | tee -a "$LOG"
nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader | tee -a "$LOG"

while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line// }" || "${line#\#}" != "$line" ]] && continue
    label="$(echo "$line" | cut -d'|' -f1 | xargs)"
    tmo="$(echo "$line" | cut -d'|' -f2 | xargs)"
    cmd="$(echo "$line" | cut -d'|' -f3-)"
    if ! wait_idle; then
        echo ">>> [$label] SKIPPED_GPU_BUSY" | tee -a "$LOG"
        continue
    fi
    echo ">>> [$label] $(date +%T)" | tee -a "$LOG"
    t0=$(date +%s.%N)
    timeout --kill-after=10 "$tmo" bash -c "$cmd" < /dev/null >> "$LOG" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    printf "<<< [%s] rc=%d wall=%.1fs\n" "$label" "$rc" "$(echo "$t1 - $t0" | bc)" | tee -a "$LOG"
done < "$PLAN"

echo "=== guarded run done $(date -Is) ===" | tee -a "$LOG"
[[ -n "$OWNER" ]] && chown -R "$OWNER" "$(dirname "$LOG")" 2>/dev/null
exit 0
