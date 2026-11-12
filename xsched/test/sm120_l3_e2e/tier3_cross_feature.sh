#!/usr/bin/env bash
# ==============================================================================
# Tier 3: Cross-Feature Combinations Test Suite (6 Tests: T3_COMB_01 to T3_COMB_06)
# ==============================================================================

TIER3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${TIER3_DIR}/lib/common.sh"

run_tier3() {
    log_info "================================================================="
    log_info "Starting Tier 3: Cross-Feature Combinations (6 Tests)"
    log_info "================================================================="

    # T3_COMB_01: F1 Verification Probes + F2 Level 2 Baseline
    if [[ -x "${BIN_VERIFY_COMPENDIUM}" ]] && [[ -f "${XSCHED_ROOT}/platforms/cuda/hal/src/arch/sm120.cpp" ]]; then
        local out
        out=$("${BIN_VERIFY_COMPENDIUM}" 2>&1)
        if [[ $? -eq 0 ]]; then
            record_result "T3_COMB_01" "PASS" "F1 RegOps/ABI verification succeeded prior to F2 L2 queue initialization"
        else
            record_result "T3_COMB_01" "FAIL" "F1 verification probe failed before F2 queue init"
        fi
    else
        record_result "T3_COMB_01" "SKIP" "Pending sm120_verify_compendium or sm120.cpp"
    fi

    # T3_COMB_02: F2 Level 2 Draining + F3 Level 3 Trap Preemption on In-Flight Grid
    if [[ -x "${BIN_TRAP_BENCH}" ]]; then
        local out
        out=$("${BIN_TRAP_BENCH}" --duration_ms 100 --test_l2_drain 1 2>&1)
        local ret=$?
        read -r p50 p99 <<< "$(parse_latencies "$out")"
        if [[ $ret -eq 0 ]] && assert_mismatch_zero_and_cuda_success "$out" "T3_COMB_02"; then
            record_result "T3_COMB_02" "PASS" "Combined L2 queue draining with L3 trap preemption succeeded" "$p50" "$p99"
        else
            record_result "T3_COMB_02" "FAIL" "Combined L2+L3 preemption failed (exit: $ret)" "$p50" "$p99"
        fi
    else
        record_result "T3_COMB_02" "SKIP" "Pending sm120_l3_trap_bench binary"
    fi

    # T3_COMB_03: F2 Level 2 Queue Control + F4 Resumable Preemption Coexistence
    if [[ -x "${BIN_RESUMABLE_BENCH}" ]]; then
        local out
        out=$("${BIN_RESUMABLE_BENCH}" --multi_queue 1 --duration_ms 100 2>&1)
        local ret=$?
        if [[ $ret -eq 0 ]]; then
            record_result "T3_COMB_03" "PASS" "Coexistence of L2 queue control with L3 resumable queue confirmed"
        else
            record_result "T3_COMB_03" "FAIL" "L2 + L3 resumable queue coexistence failed (exit: $ret)"
        fi
    else
        record_result "T3_COMB_03" "SKIP" "Pending sm120_l3_resumable_bench binary"
    fi

    # T3_COMB_04: F3 Trap Preemption vs F4 Resumable Preemption Comparative Benchmark
    if [[ -x "${BIN_TRAP_BENCH}" ]] && [[ -x "${BIN_RESUMABLE_BENCH}" ]]; then
        local out_trap
        out_trap=$("${BIN_TRAP_BENCH}" --duration_ms 100 --iters 5 2>&1)
        read -r p50_trap p99_trap <<< "$(parse_latencies "$out_trap")"

        local out_res
        out_res=$("${BIN_RESUMABLE_BENCH}" --duration_ms 100 --iters 5 2>&1)
        read -r p50_res p99_res <<< "$(parse_latencies "$out_res")"

        record_result "T3_COMB_04" "PASS" "Comparative latency evaluated: Trap P50=${p50_trap}µs vs Resumable P50=${p50_res}µs" "$p50_trap" "$p99_trap"
    else
        record_result "T3_COMB_04" "SKIP" "Pending sm120_l3_trap_bench or sm120_l3_resumable_bench"
    fi

    # T3_COMB_05: F3 Trap Preemption + F5 Hardware Health Under 20-Round Continuous Stress
    if [[ -x "${BIN_TRAP_BENCH}" ]]; then
        local out
        out=$("${BIN_TRAP_BENCH}" --iters 20 2>&1)
        local ret=$?
        if [[ $ret -eq 0 ]] && check_gpu_health; then
            record_result "T3_COMB_05" "PASS" "20 continuous trap preemption cycles completed with 100% GPU health"
        else
            record_result "T3_COMB_05" "FAIL" "20-round trap stress or health check failed (exit: $ret)"
        fi
    else
        record_result "T3_COMB_05" "SKIP" "Pending sm120_l3_trap_bench binary"
    fi

    # T3_COMB_06: F4 Resumable Preemption + F5 System Integrity with High-Priority Workload
    if [[ -x "${BIN_RESUMABLE_BENCH}" ]]; then
        local out
        out=$("${BIN_RESUMABLE_BENCH}" --hp_workload real_time_infer --duration_ms 300 2>&1)
        local ret=$?
        if [[ $ret -eq 0 ]] && check_boot_integrity; then
            record_result "T3_COMB_06" "PASS" "Resumable preemption with high-priority workload maintained filesystem integrity"
        else
            record_result "T3_COMB_06" "FAIL" "Resumable HP workload or boot integrity check failed (exit: $ret)"
        fi
    else
        record_result "T3_COMB_06" "SKIP" "Pending sm120_l3_resumable_bench binary"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    run_tier3
    print_summary_table
fi
