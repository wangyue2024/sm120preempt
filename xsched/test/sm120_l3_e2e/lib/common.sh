#!/usr/bin/env bash
# ==============================================================================
# XSched Blackwell (SM120 / RTX 5060) E2E Test Common Library
# ==============================================================================

# Colors
if [[ -t 1 ]] || [[ "${FORCE_COLOR:-0}" == "1" ]]; then
    C_RESET="\033[0m"
    C_RED="\033[1;31m"
    C_GREEN="\033[1;32m"
    C_YELLOW="\033[1;33m"
    C_BLUE="\033[1;34m"
    C_CYAN="\033[1;36m"
    C_BOLD="\033[1m"
else
    C_RESET=""
    C_RED=""
    C_GREEN=""
    C_YELLOW=""
    C_BLUE=""
    C_CYAN=""
    C_BOLD=""
fi

# Workspace Root Paths
E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XSCHED_ROOT="$(cd "${E2E_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XSCHED_ROOT}/.." && pwd)"
BUILD_DIR="${XSCHED_ROOT}/build_sm120"
BIN_DIR="${BUILD_DIR}/bin"

# Expected Binaries
BIN_VERIFY_COMPENDIUM="${BIN_DIR}/sm120_verify_compendium"
BIN_TRAP_BENCH="${BIN_DIR}/sm120_l3_trap_bench"
BIN_RESUMABLE_BENCH="${BIN_DIR}/sm120_l3_resumable_bench"

# Global Test Metrics
declare -i TOTAL_TESTS=0
declare -i PASSED_TESTS=0
declare -i FAILED_TESTS=0
declare -i SKIPPED_TESTS=0

declare -a TEST_RECORDS=()

log_info() {
    echo -e "${C_BLUE}[INFO]${C_RESET} $*"
}

log_pass() {
    echo -e "${C_GREEN}[PASS]${C_RESET} $*"
}

log_fail() {
    echo -e "${C_RED}[FAIL]${C_RESET} $*"
}

log_skip() {
    echo -e "${C_YELLOW}[SKIP]${C_RESET} $*"
}

log_warn() {
    echo -e "${C_YELLOW}[WARN]${C_RESET} $*"
}

# Register test result: record_result <ID> <STATUS: PASS|FAIL|SKIP> <MESSAGE> [P50_US] [P99_US]
record_result() {
    local tid="$1"
    local status="$2"
    local msg="$3"
    local p50="${4:--1}"
    local p99="${5:--1}"

    TOTAL_TESTS+=1
    case "$status" in
        PASS)
            PASSED_TESTS+=1
            log_pass "[${tid}] ${msg}"
            ;;
        FAIL)
            FAILED_TESTS+=1
            log_fail "[${tid}] ${msg}"
            ;;
        SKIP)
            SKIPPED_TESTS+=1
            log_skip "[${tid}] ${msg}"
            ;;
        *)
            FAILED_TESTS+=1
            log_fail "[${tid}] (Unknown status ${status}) ${msg}"
            ;;
    esac

    # Store structured record: ID|STATUS|MSG|P50|P99
    TEST_RECORDS+=("${tid}|${status}|${msg}|${p50}|${p99}")
}

# Assertion: exit code equals expected
assert_exit_code() {
    local expected="$1"
    local actual="$2"
    local test_name="$3"
    if [[ "$expected" -ne "$actual" ]]; then
        echo "Expected exit code ${expected}, got ${actual} in ${test_name}"
        return 1
    fi
    return 0
}

# Assertion: string contains expected substring
assert_contains() {
    local haystack="$1"
    local needle="$2"
    local test_name="$3"
    if [[ "$haystack" != *"$needle"* ]]; then
        echo "Assertion failed in ${test_name}: output does not contain '${needle}'"
        return 1
    fi
    return 0
}

# Parse Latency: extracts P50_US and P99_US from benchmark output
parse_latencies() {
    local text="$1"
    local p50
    local p99

    p50=$(echo "$text" | grep -oE "P50(_US)?[[:space:]]*:[[:space:]]*[0-9]+" | grep -oE "[0-9]+" | head -n1 || echo "")
    p99=$(echo "$text" | grep -oE "P99(_US)?[[:space:]]*:[[:space:]]*[0-9]+" | grep -oE "[0-9]+" | head -n1 || echo "")

    if [[ -z "$p50" ]]; then
        p50=$(echo "$text" | grep -oE "latency_p50_us\":[[:space:]]*[0-9]+" | grep -oE "[0-9]+" | head -n1 || echo "-1")
    fi
    if [[ -z "$p99" ]]; then
        p99=$(echo "$text" | grep -oE "latency_p99_us\":[[:space:]]*[0-9]+" | grep -oE "[0-9]+" | head -n1 || echo "-1")
    fi

    echo "${p50:- -1} ${p99:- -1}"
}

# Assertion: checks mismatch == 0 and cudaSuccess in output
assert_mismatch_zero_and_cuda_success() {
    local text="$1"
    local test_name="$2"

    if [[ "$text" == *"cudaError"* || "$text" == *"CUDA_ERROR"* ]]; then
        if [[ "$text" != *"cudaSuccess"* ]]; then
            echo "Assertion failed in ${test_name}: detected CUDA error in output"
            return 1
        fi
    fi

    if [[ "$text" =~ mismatch[[:space:]]*==[[:space:]]*([1-9][0-9]*) ]]; then
        echo "Assertion failed in ${test_name}: non-zero mismatch count detected (${BASH_REMATCH[1]})"
        return 1
    fi

    return 0
}

# Check GPU Health: nvidia-smi status and temperature
check_gpu_health() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "nvidia-smi not found"
        return 1
    fi

    local smi_out
    smi_out=$(nvidia-smi --query-gpu=name,driver_version,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
    if [[ $? -ne 0 || -z "$smi_out" ]]; then
        echo "nvidia-smi query failed"
        return 1
    fi

    local temp
    temp=$(echo "$smi_out" | awk -F',' '{print $3}' | tr -d ' ')
    if [[ -n "$temp" && "$temp" -gt 85 ]]; then
        echo "GPU thermal threshold exceeded: ${temp}°C > 85°C"
        return 2
    fi

    return 0
}

# Check Boot Filesystem Integrity: verify /boot, /boot/efi exist and have expected files
check_boot_integrity() {
    if [[ ! -d "/boot" ]]; then
        echo "Missing /boot directory"
        return 1
    fi

    if [[ ! -d "/boot/efi" && ! -d "/boot/EFI" ]]; then
        echo "Missing /boot/efi directory"
        return 1
    fi

    local vmlinuz_count
    vmlinuz_count=$(ls -1 /boot/vmlinuz* 2>/dev/null | wc -l)
    if [[ "$vmlinuz_count" -lt 1 ]]; then
        echo "No vmlinuz kernel found in /boot"
        return 1
    fi

    return 0
}

# Output JSON report
generate_json_report() {
    local output_file="$1"
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    cat <<EOF > "${output_file}"
{
  "test_suite": "XSched SM120 Level 3 E2E Test Suite",
  "target_gpu": "NVIDIA GeForce RTX 5060 (SM120 / GB206)",
  "timestamp": "${timestamp}",
  "summary": {
    "total": ${TOTAL_TESTS},
    "passed": ${PASSED_TESTS},
    "failed": ${FAILED_TESTS},
    "skipped": ${SKIPPED_TESTS},
    "pass_rate_pct": $(awk "BEGIN {printf \"%.2f\", (${PASSED_TESTS} / (${TOTAL_TESTS} > 0 ? ${TOTAL_TESTS} : 1)) * 100}")
  },
  "results": [
EOF

    local first=1
    for rec in "${TEST_RECORDS[@]}"; do
        IFS='|' read -r tid status msg p50 p99 <<< "$rec"
        if [[ $first -eq 0 ]]; then
            echo "," >> "${output_file}"
        fi
        first=0
        cat <<EOF >> "${output_file}"
    {
      "test_id": "${tid}",
      "status": "${status}",
      "message": "${msg//\"/\\\"}",
      "latency_p50_us": ${p50},
      "latency_p99_us": ${p99}
    }
EOF
    done

    cat <<EOF >> "${output_file}"
  ]
}
EOF
    log_info "JSON report written to: ${output_file}"
}

# Output summary table
print_summary_table() {
    echo ""
    echo -e "${C_BOLD}================================================================================${C_RESET}"
    echo -e "${C_BOLD}                         E2E TEST EXECUTION SUMMARY                             ${C_RESET}"
    echo -e "${C_BOLD}================================================================================${C_RESET}"
    printf "%-14s | %-8s | %-12s | %-12s | %-26s\n" "TEST ID" "STATUS" "P50 (µs)" "P99 (µs)" "DESCRIPTION"
    echo "--------------------------------------------------------------------------------"

    for rec in "${TEST_RECORDS[@]}"; do
        IFS='|' read -r tid status msg p50 p99 <<< "$rec"
        local status_colored
        case "$status" in
            PASS) status_colored="${C_GREEN}PASS${C_RESET}    " ;;
            FAIL) status_colored="${C_RED}FAIL${C_RESET}    " ;;
            SKIP) status_colored="${C_YELLOW}SKIP${C_RESET}    " ;;
            *)    status_colored="${status}    " ;;
        esac

        local p50_str="-"
        local p99_str="-"
        if [[ "$p50" -ge 0 ]]; then p50_str="${p50}"; fi
        if [[ "$p99" -ge 0 ]]; then p99_str="${p99}"; fi

        # Shorten message for display
        local short_msg="${msg:0:26}"
        printf "%-14s | %b | %-12s | %-12s | %-26s\n" "${tid}" "${status_colored}" "${p50_str}" "${p99_str}" "${short_msg}"
    done

    echo "================================================================================"
    echo -e "Total: ${C_BOLD}${TOTAL_TESTS}${C_RESET} | Passed: ${C_GREEN}${PASSED_TESTS}${C_RESET} | Failed: ${C_RED}${FAILED_TESTS}${C_RESET} | Skipped: ${C_YELLOW}${SKIPPED_TESTS}${C_RESET}"
    echo -e "Pass Rate: ${C_BOLD}$(awk "BEGIN {printf \"%.2f\", (${PASSED_TESTS} / (${TOTAL_TESTS} > 0 ? ${TOTAL_TESTS} : 1)) * 100}")%${C_RESET}"
    echo "================================================================================"
}
