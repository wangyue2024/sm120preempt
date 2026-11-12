#!/usr/bin/env bash
# ==============================================================================
# Tier 4: Real-World Application Scenarios Test Suite (5 Tests: T4_SCEN_01 to T4_SCEN_05)
# ==============================================================================

TIER4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${TIER4_DIR}/lib/common.sh"

run_tier4() {
    log_info "================================================================="
    log_info "Starting Tier 4: Real-World Application Scenarios (5 Tests)"
    log_info "================================================================="

    # T4_SCEN_01: LLM / Transformer Attention Preemption Scenario
    if [[ -x "${BIN_RESUMABLE_BENCH}" ]]; then
        local out
        out=$("${BIN_RESUMABLE_BENCH}" --workload attention --hp_workload prompt_eval 2>&1)
        local ret=$?
        read -r p50 p99 <<< "$(parse_latencies "$out")"
        if [[ $ret -eq 0 ]] && assert_mismatch_zero_and_cuda_success "$out" "T4_SCEN_01"; then
            record_result "T4_SCEN_01" "PASS" "Transformer attention preemption satisfied interactive SLA" "$p50" "$p99"
        else
            record_result "T4_SCEN_01" "FAIL" "Transformer attention preemption failed (exit: $ret)" "$p50" "$p99"
        fi
    elif [[ -x "${BIN_TRAP_BENCH}" ]]; then
        local out
        out=$("${BIN_TRAP_BENCH}" --workload attention 2>&1)
        local ret=$?
        read -r p50 p99 <<< "$(parse_latencies "$out")"
        if [[ $ret -eq 0 ]] && assert_mismatch_zero_and_cuda_success "$out" "T4_SCEN_01"; then
            record_result "T4_SCEN_01" "PASS" "Transformer attention preempted and replayed with 100% numerical parity" "$p50" "$p99"
        else
            record_result "T4_SCEN_01" "FAIL" "Transformer attention trap test failed (exit: $ret)" "$p50" "$p99"
        fi
    else
        record_result "T4_SCEN_01" "SKIP" "Pending sm120_l3_resumable_bench or sm120_l3_trap_bench"
    fi

    # T4_SCEN_02: Deep Learning Dense GEMM (Matrix Multiplication) Scenario
    if [[ -x "${BIN_RESUMABLE_BENCH}" ]]; then
        local out
        out=$("${BIN_RESUMABLE_BENCH}" --workload gemm_4096 --check_exact_gemm 1 2>&1)
        local ret=$?
        read -r p50 p99 <<< "$(parse_latencies "$out")"
        if [[ $ret -eq 0 ]] && assert_mismatch_zero_and_cuda_success "$out" "T4_SCEN_02"; then
            record_result "T4_SCEN_02" "PASS" "Dense GEMM (4096x4096) preempted and resumed with exact numerical parity" "$p50" "$p99"
        else
            record_result "T4_SCEN_02" "FAIL" "Dense GEMM scenario failed (exit: $ret)" "$p50" "$p99"
        fi
    else
        record_result "T4_SCEN_02" "SKIP" "Pending sm120_l3_resumable_bench binary"
    fi

    # T4_SCEN_03: Scientific 3D Stencil PDE Solver Scenario
    if [[ -x "${BIN_RESUMABLE_BENCH}" ]]; then
        local out
        out=$("${BIN_RESUMABLE_BENCH}" --workload stencil_3d --steps 50 --pause_step 25 2>&1)
        local ret=$?
        if [[ $ret -eq 0 ]] && assert_mismatch_zero_and_cuda_success "$out" "T4_SCEN_03"; then
            record_result "T4_SCEN_03" "PASS" "Scientific 3D stencil PDE solver paused and resumed without precision loss"
        else
            record_result "T4_SCEN_03" "FAIL" "3D stencil PDE solver scenario failed (exit: $ret)"
        fi
    else
        record_result "T4_SCEN_03" "SKIP" "Pending sm120_l3_resumable_bench binary"
    fi

    # T4_SCEN_04: Multi-Tenant Cloud GPU Time-Sharing SLA Scenario
    if [[ -x "${BIN_TRAP_BENCH}" ]] || [[ -x "${BIN_RESUMABLE_BENCH}" ]]; then
        local target_bin="${BIN_RESUMABLE_BENCH}"
        if [[ ! -x "$target_bin" ]]; then target_bin="${BIN_TRAP_BENCH}"; fi
        local out
        out=$("$target_bin" --scenario multi_tenant --sla_us 2000 2>&1)
        local ret=$?
        if [[ $ret -eq 0 ]]; then
            record_result "T4_SCEN_04" "PASS" "Multi-tenant time-sharing preserved priority tenant SLA < 2ms"
        else
            record_result "T4_SCEN_04" "FAIL" "Multi-tenant SLA test failed (exit: $ret)"
        fi
    else
        record_result "T4_SCEN_04" "SKIP" "Pending L3 bench binaries"
    fi

    # T4_SCEN_05: 50-Iteration Heavy Preemption Stress & Reliability Workload
    if [[ -x "${BIN_TRAP_BENCH}" ]] || [[ -x "${BIN_RESUMABLE_BENCH}" ]]; then
        local target_bin="${BIN_RESUMABLE_BENCH}"
        if [[ ! -x "$target_bin" ]]; then target_bin="${BIN_TRAP_BENCH}"; fi
        local out
        out=$("$target_bin" --iters 50 --duration_ms 50 2>&1)
        local ret=$?
        if [[ $ret -eq 0 ]] && check_gpu_health; then
            record_result "T4_SCEN_05" "PASS" "50-iteration continuous preemption stress passed with zero driver fault"
        else
            record_result "T4_SCEN_05" "FAIL" "50-iteration preemption stress failed (exit: $ret)"
        fi
    else
        record_result "T4_SCEN_05" "SKIP" "Pending L3 bench binaries"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    run_tier4
    print_summary_table
fi
