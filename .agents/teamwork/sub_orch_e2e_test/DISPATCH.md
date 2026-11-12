## 2026-10-31T18:40:00Z

### Task: E2E Testing Track Orchestrator (`sub_orch_e2e_test`)

- **Your Identity**: `sub_orch_e2e_test` (E2E Testing Orchestrator, `teamwork_orchestrator`)
- **Your Working Directory**: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/sub_orch_e2e_test`
- **Parent Conversation ID**: `257da9c1-15a6-4890-8b45-11ac94dbad91`
- **Original Request Record**: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/ORIGINAL_REQUEST.md` (MUST read and include in every subagent dispatch)
- **Project Scope Document**: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/PROJECT.md`
- **Phase 0 Survey Reports (MUST READ)**:
  1. `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_1/handoff.md`
  2. `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_2/handoff.md`
  3. `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_3/handoff.md`

#### Scope & Deliverables for E2E Testing Track
1. **Create `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/TEST_INFRA.md`**:
   - Define the requirement-driven 4-tier E2E test architecture covering all 5 features (`F1`–`F5`) in `PROJECT.md § Feature Inventory`:
     - **Tier 1 (Feature Coverage, >=5 tests per feature, >=25 tests total)**:
       - `F1`: Paper & Compendium analysis document completeness + empirical verification checks (`pGpu+0x911` `IS_GSP_CLIENT`, `0x20800122` `PTIMER` read, `0x419e84` `0xbadf1100` PRI error check, `slot20_pc - 0x4400` live trap handler layout check).
       - `F2`: SM120 Level 2 baseline tests (progressive command launch, `GuardianSM120` `Deactivate` + `Reactivate`, multi-CTA kernel preemption, `c[0x0][0x170..0x188]` constant bank check).
       - `F3`: SM120 Level 3 sub-millisecond trap preemption tests (single-wave persistent/long kernel trap preempt, latency verification in microseconds, zero context destruction `cudaSuccess`, CTA-granular replay accuracy `mismatch == 0`, multi-stream selective preemption).
       - `F4`: SM120 non-abort zero-lost-work resumable preemption tests (stateful/non-idempotent kernel pause & resume, zero lost work verification, high-priority kernel execution during pause window, sub-millisecond/low-millisecond latency reporting, multi-round pause/resume).
       - `F5`: System & hardware health checks (GPU active on `nvidia-smi`, driver `595.99.02` loaded, `/boot` & `/boot/efi` intact, clean CUDA context creation after preemption stress, reproducible latency summary table).
     - **Tier 2 (Boundary & Corner Cases, >=5 tests per feature, >=25 tests total)**:
       - Edge cases: rapid back-to-back preempt/resume cycles, very short kernels (already completed before interrupt), maximum-block/maximum-thread SM120 kernels (30 SMs × 1536 threads), non-killable kernel flag (`killable == 0`), invalid/null parameter handling, unprivileged vs root execution behavior.
     - **Tier 3 (Cross-Feature Combinations, >=6 tests)**:
       - Pairwise combinations: L2 + L3 trap preemption across multiple queues, L3 trap + L3 resumable preemption comparison, high-priority latency-sensitive inference/compute task preempting low-priority long kernel with end-to-end turnaround measurement.
     - **Tier 4 (Real-World Application Scenarios, >=5 tests)**:
       - Realistic GPU workloads on RTX 5060 (e.g., GEMM / matrix multiplication, transformer attention/MLP sequence, iterative scientific stencil/solver, multi-tenant priority scheduling SLA test, continuous 20-iteration preemption stress workload).
2. **Implement the E2E Test Suite & Runner in `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/test/sm120_l3_e2e/`**:
   - Create `run_e2e.sh` (supporting `--tier 1`, `--tier 2`, `--tier 3`, `--tier 4`, `--all`) and the test cases/harnesses that invoke the entry points defined in `PROJECT.md § Interface Contracts` (`sm120_verify_compendium`, `sm120_l3_trap_bench`, `sm120_l3_resumable_bench`, and E2E test binaries built against `xsched/build_sm120/`).
3. **Publish `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/TEST_READY.md`**:
   - Once the E2E test suite is complete and verified via your Reviewer/Challenger/Auditor gate, write `TEST_READY.md` at project root, write `handoff.md` in your working directory, and notify `257da9c1-15a6-4890-8b45-11ac94dbad91` via `send_message`.

#### Exclusive Write Ownership
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/TEST_INFRA.md`
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/TEST_READY.md`
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/test/sm120_l3_e2e/`
- Do NOT modify implementation files in `xsched/platforms/` or `xsched/preempt/`.
