#!/usr/bin/env bash
# Guarded root test runner for the shared RTX 5080 host (or local 5060).
# Usage (as root, via `sudo -S bash p3_guarded_run.sh <plan_file> <log_file>`):
#   Each non-empty, non-# line of <plan_file> is:  <label> | <timeout_s> | <shell command>
#   Before EVERY test the GPU must have had no compute process for QUIET_S seconds (another
#   user's jobs often come in bursts of short runs); after WAIT_BUSY_S seconds of waiting the test
#   is skipped (SKIPPED_GPU_BUSY). While a test runs, the GPU is polled every second: if a compute
#   process that is not part of the test's own process tree appears, the test is stopped at once
#   (ABORTED_FOREIGN_GPU_JOB) - BAR0 pulses and trap-handler time are GPU-wide and a co-running
#   context of another user can be hit (Xid 109 CTX SWITCH TIMEOUT when a time slice falls on a
#   warp inside the trap handler).
set -uo pipefail

PLAN="$1"
LOG="$2"
WAIT_BUSY_S="${WAIT_BUSY_S:-1800}"
QUIET_S="${QUIET_S:-60}"
OWNER="${RUN_OWNER:-}"

# GUARD_IGNORE: extended regex of process names that never count as GPU jobs (desktop processes
# listed as compute apps on a local display GPU, e.g. 'nautilus|gnome-shell').
gpu_apps() {
    local apps
    apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
    [[ -n "${GUARD_IGNORE:-}" ]] && apps="$(echo "$apps" | grep -Ev "$GUARD_IGNORE" || true)"
    echo "$apps"
}

gpu_busy() {
    local apps
    apps="$(gpu_apps)"
    [[ -n "$apps" ]] && { echo "$apps"; return 0; }
    return 1
}

# 0 if pid descends from root_pid.
descends_from() {
    local pid="$1" root_pid="$2" ppid
    while [[ -n "$pid" && "$pid" != "0" && "$pid" != "1" ]]; do
        [[ "$pid" == "$root_pid" ]] && return 0
        ppid="$(awk '/^PPid:/ {print $2}' "/proc/$pid/status" 2>/dev/null)"
        [[ -z "$ppid" ]] && return 1
        pid="$ppid"
    done
    return 1
}

# Prints compute processes that are not descendants of $1 (empty: none).
foreign_gpu_apps() {
    local root_pid="$1" line pid
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        pid="$(echo "$line" | cut -d',' -f1 | xargs)"
        descends_from "$pid" "$root_pid" || echo "$line"
    done < <(gpu_apps)
}

wait_idle() {
    [[ "${SKIP_GUARD:-0}" == "1" ]] && return 0   # single-user local desktop GPU only
    local waited=0 quiet=0 apps last=""
    while (( quiet < QUIET_S )); do
        if apps="$(gpu_busy)"; then
            quiet=0
            if (( waited >= WAIT_BUSY_S )); then
                echo "[GUARD] GPU still busy after ${waited}s: ${apps//$'\n'/; }" | tee -a "$LOG"
                return 1
            fi
            [[ "$apps" != "$last" ]] && echo "[GUARD] GPU busy (${apps//$'\n'/; }), waiting..." | tee -a "$LOG"
            last="$apps"
        else
            quiet=$((quiet + 5))
        fi
        sleep 5
        waited=$((waited + 5))
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
    timeout --kill-after=10 "$tmo" bash -c "$cmd" < /dev/null >> "$LOG" 2>&1 &
    tpid=$!
    aborted=""
    while kill -0 "$tpid" 2>/dev/null; do
        if [[ "${SKIP_GUARD:-0}" != "1" ]]; then
            f="$(foreign_gpu_apps "$tpid")"
            if [[ -n "$f" ]]; then
                aborted="$f"
                echo "[GUARD] foreign GPU job appeared (${f//$'\n'/; }): stopping [$label]" | tee -a "$LOG"
                kill -TERM "$tpid" 2>/dev/null       # timeout relays it to the test's process group
                sleep 3
                kill -KILL -- "-$tpid" 2>/dev/null   # timeout is its own process-group leader
                break
            fi
        fi
        sleep 1
    done
    wait "$tpid"
    rc=$?
    [[ -n "$aborted" ]] && echo ">>> [$label] ABORTED_FOREIGN_GPU_JOB" | tee -a "$LOG"
    t1=$(date +%s.%N)
    printf "<<< [%s] rc=%d wall=%.1fs\n" "$label" "$rc" "$(echo "$t1 - $t0" | bc)" | tee -a "$LOG"
done < "$PLAN"

echo "=== guarded run done $(date -Is) ===" | tee -a "$LOG"
[[ -n "$OWNER" ]] && chown -R "$OWNER" "$(dirname "$LOG")" 2>/dev/null
exit 0
