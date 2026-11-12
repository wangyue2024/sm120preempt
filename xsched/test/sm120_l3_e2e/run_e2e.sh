#!/usr/bin/env bash
# ==============================================================================
# XSched Blackwell (SM120 / RTX 5060) Level 3 End-to-End Test Suite Runner
# ==============================================================================
# Target: NVIDIA GeForce RTX 5060 (GB206 / SM120) with Driver 595.99.02
# Usage:
#   run_e2e.sh [--tier 1|2|3|4] [--all] [--test <TEST_ID>] [--json] [--output <path>]
# ==============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/tier1_feature_coverage.sh"
source "${SCRIPT_DIR}/tier2_boundary_corner.sh"
source "${SCRIPT_DIR}/tier3_cross_feature.sh"
source "${SCRIPT_DIR}/tier4_real_world.sh"

TIER_SELECTION="ALL"
TARGET_TEST=""
GENERATE_JSON=0
JSON_OUTPUT_PATH="${SCRIPT_DIR}/e2e_results.json"
VERBOSE=0

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --tier <1|2|3|4>    Execute only the specified test tier:
                        1 : Feature Coverage (25 tests across F1-F5)
                        2 : Boundary & Corner Cases (25 tests across F1-F5)
                        3 : Cross-Feature Combinations (6 tests)
                        4 : Real-World Application Scenarios (5 tests)
  --all               Execute all 4 tiers (61 tests total, default)
  --test <TEST_ID>    Execute a single specific test case (e.g. T1_F1_01)
  --json              Generate structured machine-readable JSON telemetry
  --output <file>     Custom file path for JSON output (default: ${JSON_OUTPUT_PATH})
  --verbose           Enable verbose output
  -h, --help          Show this help message
EOF
}

# Parse Command-Line Arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)
            if [[ -z "$2" || ! "$2" =~ ^[1-4]$ ]]; then
                echo "Error: --tier requires an argument between 1 and 4."
                exit 1
            fi
            TIER_SELECTION="$2"
            shift 2
            ;;
        --all)
            TIER_SELECTION="ALL"
            shift
            ;;
        --test)
            if [[ -z "$2" ]]; then
                echo "Error: --test requires a test ID."
                exit 1
            fi
            TARGET_TEST="$2"
            shift 2
            ;;
        --json)
            GENERATE_JSON=1
            shift
            ;;
        --output)
            if [[ -z "$2" ]]; then
                echo "Error: --output requires a file path."
                exit 1
            fi
            JSON_OUTPUT_PATH="$2"
            GENERATE_JSON=1
            shift 2
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

echo -e "${C_BOLD}================================================================================${C_RESET}"
echo -e "${C_BOLD}     XSCHED SM120 LEVEL 3 PREEMPTION END-TO-END TEST SUITE (RTX 5060)           ${C_RESET}"
echo -e "${C_BOLD}================================================================================${C_RESET}"
log_info "Workspace: ${WORKSPACE_ROOT}"
log_info "XSched Root: ${XSCHED_ROOT}"
log_info "Build Dir: ${BUILD_DIR}"
log_info "Selection: Tier ${TIER_SELECTION}${TARGET_TEST:+ (Filter: ${TARGET_TEST})}"
echo ""

# Pre-Flight System Integrity & GPU Status Check
log_info "Executing pre-flight health checks..."
if ! check_boot_integrity; then
    log_fail "Pre-flight check failed: /boot or /boot/efi integrity issue detected!"
    exit 2
fi

if ! check_gpu_health; then
    log_warn "Pre-flight check: GPU not ready or thermal threshold exceeded."
fi

# Run Single Test Case Mode
if [[ -n "$TARGET_TEST" ]]; then
    log_info "Executing single test case: ${TARGET_TEST}"
    
    # Run the containing tier and filter
    case "$TARGET_TEST" in
        T1_*) run_tier1 ;;
        T2_*) run_tier2 ;;
        T3_*) run_tier3 ;;
        T4_*) run_tier4 ;;
        *)
            log_fail "Unknown test ID format: ${TARGET_TEST}"
            exit 1
            ;;
    esac

    # Filter TEST_RECORDS for target test
    declare -a FILTERED_RECORDS=()
    TOTAL_TESTS=0
    PASSED_TESTS=0
    FAILED_TESTS=0
    SKIPPED_TESTS=0
    for rec in "${TEST_RECORDS[@]}"; do
        if [[ "$rec" == "${TARGET_TEST}|"* ]]; then
            FILTERED_RECORDS+=("$rec")
            TOTAL_TESTS+=1
            IFS='|' read -r _ status _ _ _ <<< "$rec"
            case "$status" in
                PASS) PASSED_TESTS+=1 ;;
                FAIL) FAILED_TESTS+=1 ;;
                SKIP) SKIPPED_TESTS+=1 ;;
            esac
        fi
    done
    TEST_RECORDS=("${FILTERED_RECORDS[@]}")

# Run Tier Selection Mode
else
    case "$TIER_SELECTION" in
        1)
            run_tier1
            ;;
        2)
            run_tier2
            ;;
        3)
            run_tier3
            ;;
        4)
            run_tier4
            ;;
        ALL)
            run_tier1
            echo ""
            run_tier2
            echo ""
            run_tier3
            echo ""
            run_tier4
            ;;
    esac
fi

# Post-Flight Safety Verification
log_info "Executing post-flight health and dual-boot safety checks..."
if ! check_boot_integrity; then
    log_fail "CRITICAL: System /boot or /boot/efi integrity modified during test!"
    exit 3
fi
log_pass "Post-flight safety check confirmed: dual-boot and system partitions intact."

# Print Summary Table
print_summary_table

# Generate JSON Report if Requested
if [[ $GENERATE_JSON -eq 1 ]]; then
    generate_json_report "${JSON_OUTPUT_PATH}"
fi

# Exit status code
if [[ $FAILED_TESTS -gt 0 ]]; then
    exit 1
fi
exit 0
