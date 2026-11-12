# TEST_READY — XSched SM120 Level 3 Preemption E2E Test Suite

**Test Suite**: XSched Blackwell (RTX 5060 / SM120) Level 3 Preemption 4-Tier E2E Test Suite  
**Status**: **READY FOR EXECUTION**  
**Target Hardware**: NVIDIA GeForce RTX 5060 (`GB206`, Compute Capability `12.0` / `sm_120`, 30 SMs, 8151 MiB VRAM)  
**Host Environment**: Ubuntu 26.04.1 LTS (`Linux 7.0.0-38-generic`), NVIDIA Open Kernel Driver & GSP Firmware `595.99.02`, CUDA Driver Version `13.2`.

---

## 1. Quickstart & Runner Commands

The unified test runner is located at:  
`/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/test/sm120_l3_e2e/run_e2e.sh`

### Execute All 61 Tests (Tiers 1–4)
```bash
# Run entire test suite with interactive table and JSON output
xsched/test/sm120_l3_e2e/run_e2e.sh --all --json
```

### Execute by Tier
```bash
# Tier 1: Feature Coverage (25 tests across F1-F5)
xsched/test/sm120_l3_e2e/run_e2e.sh --tier 1

# Tier 2: Boundary & Corner Cases (25 tests across F1-F5)
xsched/test/sm120_l3_e2e/run_e2e.sh --tier 2

# Tier 3: Cross-Feature Combinations (6 tests)
xsched/test/sm120_l3_e2e/run_e2e.sh --tier 3

# Tier 4: Real-World Application Scenarios (5 tests)
xsched/test/sm120_l3_e2e/run_e2e.sh --tier 4
```

### Execute Single Test Case
```bash
# Example: Run GPU Health Check
xsched/test/sm120_l3_e2e/run_e2e.sh --test T1_F5_01

# Example: Run 56-Byte RegOp Alignment Test
xsched/test/sm120_l3_e2e/run_e2e.sh --test T1_F1_04
```

---

## 2. Test Coverage Matrix

The test suite covers all 5 features (`F1`–`F5`) defined in `PROJECT.md § Feature Inventory` across 4 progressive tiers:

| Feature | Feature Description | Tier 1 (Core) | Tier 2 (Boundary) | Tier 3 (Cross) | Tier 4 (Scenario) | Total Tests |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **F1** | Paper L3 Model & Critical Verification of `compendium` | 5 | 5 | 1 | - | **11** |
| **F2** | SM120 Level 2 Baseline Integration & Contracts | 5 | 5 | 2 | - | **12** |
| **F3** | Sub-Millisecond Non-Abort L3 Trap Preemption on RTX 5060 | 5 | 5 | 3 | 2 | **15** |
| **F4** | Superior Non-Abort Zero-Lost-Work Resumable Preemption | 5 | 5 | 3 | 2 | **15** |
| **F5** | End-to-End Hardware Health & Latency Benchmarking | 5 | 5 | 2 | 1 | **13** |
| **Total** | **All Features Fully Covered** | **25** | **25** | **6** | **5** | **61** |

---

## 3. Interface Entry Points & Binary Contracts

The E2E test runner interacts with the following interface entry points specified in `PROJECT.md § Interface Contracts`:

| Entry Point / Binary | Owning Milestone | CLI Invocations & Verification Assertions |
|---|---|---|
| `xsched/docs/sm120_level3_analysis_and_verification.md` | `M1` | Verified for completeness across L1/L2/L3 theory, compendium debunking, PRI errors, and trap layout. |
| `xsched/build_sm120/bin/sm120_verify_compendium` | `M1` | `--all`, `--probe gsp_client`, `--probe regops_ptimer`, `--probe reg_419e84_and_traphandler`. Asserts exit code `0` and `[VERIFY_PASS]`. |
| `xsched/build_sm120/bin/sm120_l3_trap_bench` | `M2` | `--duration_ms <N>`, `--iters <N>`, `--streams <N>`, `--rapid <0\|1>`, `--killable <0\|1>`. Asserts `cudaSuccess`, `mismatch == 0`, parses `P50_US < 1000 µs`. |
| `xsched/build_sm120/bin/sm120_l3_resumable_bench` | `M3` | `--duration_ms <N>`, `--iters <N>`, `--hp_workload <name>`, `--pause_pct <N>`, `--multi_pause <N>`. Asserts `cudaSuccess`, `mismatch == 0`, parses `SUSPEND_P50_US`, `RESUME_P50_US`, `TURNAROUND_P50_US`. |
| System Health Probes (`nvidia-smi`, `/proc/driver/nvidia/version`, `/boot` integrity) | `E2E/M4` | Verified live before and after test execution. Asserts 0 zombie GPU processes, driver 595.99.02, intact dual-boot /boot/efi. |

---

## 4. Progressive Testability & Execution Semantics

To support progressive testability during milestone implementation:
1. **Live Environment Probes**: Tests targeting live system properties (GPU health, driver version, dual-boot /boot integrity, struct alignment, and live kernel module definitions) execute immediately on the live RTX 5060 and pass.
2. **Graceful Binary Dependency Handling**: When milestone implementation binaries (`sm120_verify_compendium`, `sm120_l3_trap_bench`, `sm120_l3_resumable_bench`) are being built by M1, M2, or M3, the test harness clearly reports `SKIP (Pending binary)` without failing the test runner. Once the binary is compiled into `xsched/build_sm120/bin/`, the runner automatically invokes it with full verification assertions.
3. **Strict Zero-Toleration for Failures**: Any test that executes implementation logic or hardware probes must return exit code `0`, `cudaSuccess`, `mismatch == 0`, and adhere to sub-millisecond latency bounds (`P50_US < 1000 µs`).

---

## 5. Artifact Directory Layout

```
xsched/test/sm120_l3_e2e/
├── run_e2e.sh                      # Master executable runner (--tier 1|2|3|4|all, --test <ID>, --json)
├── tier1_feature_coverage.sh       # Tier 1 suite: 25 tests covering F1-F5
├── tier2_boundary_corner.sh        # Tier 2 suite: 25 tests covering boundary conditions
├── tier3_cross_feature.sh          # Tier 3 suite: 6 tests covering pairwise feature combinations
├── tier4_real_world.sh             # Tier 4 suite: 5 tests covering real-world GPU workloads
├── e2e_results.json                # Structured telemetry output from test execution
├── lib/
│   └── common.sh                   # Core test framework, color logging, assertions, latency parsers
└── scenarios/
    ├── Makefile                    # Build definition for standalone scenario workloads
    ├── workload_dense_gemm.cu      # 4096x4096 matrix multiplication workload
    ├── workload_3d_stencil.cu      # 3D heat diffusion finite-difference PDE solver
    ├── workload_transformer_attention.cu # Transformer attention dot-product emulation
    ├── workload_atomic_accum.cu    # Non-idempotent memory accumulator integrity test
    └── workload_multi_tenant.cu    # Multi-tenant priority scheduling SLA workload
```

---

## 6. Latest Test Run Evidence

Executing `run_e2e.sh --all --json` on live host:
- **Total Tests Defined**: 61
- **Live System Tests Passed**: 19
- **Failed Tests**: 0
- **Skipped (Awaiting M1/M2/M3 builds)**: 42
- **Pass Rate**: 31.15% (100% of currently runnable tests)
- **GPU Status**: `NVIDIA GeForce RTX 5060` (Temp: ~51°C, Driver: `595.99.02`, CUDA: `13.2`)
- **System Safety**: `/boot` and `/boot/efi` dual-boot partitions verified intact.
