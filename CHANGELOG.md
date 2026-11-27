# Changelog — XSched Blackwell (`sm_120`) Level-3 Preemption

All notable changes to the Blackwell (`sm_120`) Level-3 transparent preemption and scheduling system are documented in this file.

## [Unreleased] - 2026-11-27 — Phase 3.5 round 2: transparent multi-process vLLM end to end

Doc: `doc/SM120_V2_PHASE3_5_TIER1_ZH.md` §8; ledger §10.13; environment variables in the principles doc §11.

### Added
- `bench_vllm_mp_p35.py`: two unmodified vLLM 0.34 processes under XSched (HP Qwen2.5-0.5B with Poisson short requests, LP Qwen2.5-7B-AWQ or 3B with long-document prefill batches compared to a reference), modes ref / solo / native / glb-solo / glb-lv1 / glb-lv3; only environment variables (`XSCHED_SCHEDULER=GLB`, `XSCHED_AUTO_XQUEUE=1`) and `xserver` HPF.
- Tests: `test_auto_legacy_p35.py` (default-stream PyTorch under automatic XQueues), `test_regpress255_p35.py` + `kernels/regpress255.cu` (register-file pressure), `test_lv3_stress_kernel_p35.py` (flash-attention / GEMM under frequent preemption), `test_multiqueue_lv3_p35.py` (several Level-3 queues in one process suspended independently), `kernels/ce_vs_sm_probe.cu`; Suite 9 `auto-lv1` / `auto-lv3` (no explicit XQueues); regression `SYN-regpress255`, `SYN-flash-stress`.
- `test_foreign_ctx_p35.py` (a native victim process next to a Level-3 aggressor).
- Diagnostics: `XCudaShimStats()` (commands on the legacy stream / on streams without an XQueue / on XQueues), `XSCHED_SM120_WATCHDOG_MS` + `XSm120DumpStreams()` (per-queue host stage, control header and CTA state histogram when a trap operation stalls), `rf_full` route statistic.
- `xserver` HPF: `XSCHED_HPF_RESUME_DELAY_US` (keep lower priorities suspended for a while after a higher priority was ready; default 0).
- `p3_guarded_run.sh`: requires `QUIET_S` (60 s) of an idle GPU before each entry and stops the running test as soon as a compute process outside the test's own process tree appears (`ABORTED_FOREIGN_GPU_JOB`); `GUARD_IGNORE` for desktop processes.

### Fixed
- **DEF-40 (P0 safety)**: a BAR0 pulse traps the warps of whatever context is resident on the GPU at that instant; a context without the XSched trap handler cannot return. A native GEMM process on the RTX 5080 was killed (CUDA error 719, Xid 109 on its channel) in 2 of 2 runs under Level-3 suspension pressure (control: unaffected); the desktop Xid 31 faults at address 0 on the RTX 5060 (OBS-8 / OBS-14) are the same mechanism. Now every sm_120 XQueue (any level) installs the handler for its context and registers the process in `/dev/shm/xsched-sm120/` (`XSCHED_SM120_TENANT_HANDLER`), and an NVML poller (`XSCHED_SM120_FOREIGN_POLL_MS`, 100 ms) withholds pulses while an unregistered process has a context on the GPU - suspension then has Level-1 semantics (`XSCHED_SM120_FOREIGN_CTX=block` default, `allow` to opt out).
- **DEF-41 (mitigated)**: two processes that both run Level-3 queues on one GPU are two time-sliced contexts; a context switch that arrives while one of them has warps in the trap handler times out (Xid 109, the DEF-27 mechanism): 14/14 runs of `test_multiqueue_lv3_p35.py --procs 2` with the host-wide pulse lock off and on. The ready gate only covers Level-1 peers that wait for the suspension. Processes running Level-2/3 kernels now add a `.lv3` mark to their tenant registration and a Level-3 process withholds pulses while another one is on the GPU (`XSCHED_SM120_LV3_PEERS=allow` to opt out). Supported: one Level-3 process per GPU (others Level-1 under the GLB ready gate), or all Level-3 work in one process.
- **DEF-39**: the gate-mirror write (`cuMemsetD2D32Async`) and the control-block reset (32 KB / 1 MB memsets) are executed by the driver as SM kernels and cannot start while every SM is occupied (`kernels/ce_vs_sm_probe.cu`): with stuck CTAs no pulse or recovery pulse could be issued, and normally every suspension waited for an SM to have room. They are now pinned host-to-device copies (pitched 2D for the mirror), the reset clears only the CTA range launched since the last reset and is skipped when no pulse / restore happened (`XSCHED_SM120_CE_CONTROL`, default on).
- **DEF-38 (GPU wedge)**: several Level-3 queues of one process suspended / resumed independently could wedge the GPU (4 queues, ~6,600 pulses/s: RTX 5080 5 of 8 runs hung, RTX 5060 Xid 109). The BAR0 pulse is GPU-wide and one-shot - a warp inside the trap handler is not trapped again - so another queue's pulse landing while a queue's warps were in the handler (restore wave in Cave 1, or a collateral no-op trap just before the queue's own cmd write and pulse) made warps miss their pulse or cut a restore, leaving CTAs at `BAR.SYNC.TRAP`. Pulses and restore set-up are now serialized per context and a pulse waits for other queues' Mode 5 restores (`XSCHED_SM120_CTX_PULSE_GATE`, default on); across processes the trap operations are serialized by a host-wide `flock` on `/dev/shm/xsched-sm120/<pci>.pulse` with a bounded wait. Gate on: 26/26 runs without a hang (gate off: 13/23 hung; serialize-only: 10/10).
- **DEF-37**: work on the legacy default stream always bypassed XSched (46% of the commands of a 3B vLLM LP; HP 46 ms vs 30.5 ms solo). Under `XSCHED_AUTO_XQUEUE` it now goes to one blocking proxy stream per context with an XQueue (`XSCHED_AUTO_XQUEUE_LEGACY`, default on); 57 synchronous memcpy / memset APIs first drain the blocking XQueues; legacy semantics fall back to the old path once the application creates a blocking stream of its own.
- **DEF-36**: automatic XQueues were created for every stream (128 per PyTorch process) and suspending an idle Level-3 queue still fired a GPU-wide pulse: transparent vLLM Level-3 corrupted an LP batch and made HP slower than native. Automatic XQueues are created on a stream's first submission (`XSCHED_AUTO_XQUEUE_LAZY`, default on) and idle queues no longer pulse.
- **DEF-35**: the loop detector missed uniform-predicate branches `BRA.U` (`0x547`).
- **Tier-1 item 4 (driver robustness)**: when the trap engine does not recognise the driver's trap-handler layout (it already validated export tables, ToolsMemory size, handler fingerprint and hook-site instructions), Level-2/3 queues now route kernels to the Level-1 boundary (`lv1_no_engine` route statistic, one warning with the driver version) instead of aborting in `PrepareLaunch`; a failed `PrepareLaunch` (e.g. allocation) also falls back for that kernel. `XSCHED_SM120_SIMULATE_UNSUPPORTED_DRIVER=1` simulates it (`SYN-no-engine`).

### Known limitation
- **DEF-34 (hardware)**: an SM whose register file is allocated to the last register does not take the trap (0 of 61 evictions vs 45 of 59 with 1,024 registers free); suspension then waits for the kernel (data stays correct). Detected by geometry (`rf_full`), optional `XSCHED_SM120_RF_FULL=lv1`. Affects e.g. the vLLM Marlin AWQ GEMM and cutlass INT8 GEMM.
- DEF-20b closed: ptxas never allocates R253 / R254 on sm_120, so R254 needs no saving.
- Across processes the DEF-38 gate only serializes (host-wide `flock`, `XSCHED_SM120_HOST_PULSE_GATE`); two concurrently running Level-3 processes are not supported (DEF-41: pulses are withheld while a Level-3 peer is present).
- DEF-40 detection polls every 100 ms: a foreign process that starts in between can still be hit until the next poll.

### Validation
- RTX 5080 (no display): final validation + full regression 297/297 (cross-process 4, foreign context 1, multi-queue 3, vLLM end to end 4, full regression 285), no Xid; final build targeted 34/34.
- RTX 5060 (drives the desktop): full regressions 285/285 (twice); final build 285/286 (`SYN-event-bridge-lv1`: the test's native reference was non-deterministic, OBS-20, not reproduced in 80 controlled runs). The local regression environment no longer opts in to pulses on the desktop GPU (DEF-40): with the desktop in use they faulted a browser (Xid 31).

### Results
- Two vLLM processes on an RTX 5080: HP E2E P50 29.9 ms with a 3B LP (solo 30.5, native 61.9), P99 30.3; with a 7B-AWQ LP 29.8-31.5 ms (native 67.4), P99 33.6-36.1 (Level-1: 56.8); LP keeps 88-92% of its solo throughput; HP 40/40 and every LP batch exact.

## [Unreleased] - 2026-11-26 — Phase 3.5 (tier 1): Level-3 value, CUDA Graph in-graph preemption, vLLM

Doc: `doc/SM120_V2_PHASE3_5_TIER1_ZH.md`; mechanisms in `doc/SM120_V2_PHASE3_PRINCIPLES_AND_IMPLEMENTATION_ZH.md` §3.5, §5, §5.1, §5.2, §7.3; ledger §10.12.

### Added
- **DEF-09 done - CUDA Graph flattening** (`platforms/cuda/shim/src/graph_flatten.cpp`): at `cuGraphInstantiate*` the graph is walked (Kahn topological order, child graphs inlined, kernel arguments copied byte-wise via `cuFuncGetParamInfo`, memset / memcpy nodes kept); on a Level-3 XQueue `cuGraphLaunch` submits the nodes one by one, so every kernel is trap-preemptible. Host / event / conditional / memory-alloc nodes, `extra` arguments, cluster and cooperative kernels keep the graph whole; any `cuGraphExecUpdate` / `cuGraphExec*SetParams` / `cuGraphNodeSetEnabled` invalidates the plan. `XSCHED_CUDA_GRAPH_FLATTEN` (0 off, 1 default: Level >= Interrupt queues, 2 all XQueues), `XSCHED_CUDA_GRAPH_FLATTEN_MIN_US` (flatten only graphs whose first, timed whole launch ran at least that long). `XCudaGraphFlattenStats()` counters. Whole-network-captured training step (fwd + bwd + capturable AdamW): bit-exact under 15 suspensions, suspend P99 407 ms -> 0.4-1.4 ms; `DNN-07` / `HW-04` preempt inside the graph.
- `bench_lv3_value_p35.py`: HP latency vs LP kernel duration (cuBLAS GEMM, Triton persistent GEMM partial / full occupancy, SDPA) x native / lv1 / lv1-sync / lv3 / lv3-strict x small / large HP, both GPUs; `test_graph_preempt_p35.py`, `test_graph_relaunch_p35.py`, `test_event_bridge_p35.py`, `test_vllm_lv3_p35.py` (vLLM 0.34 in-process engine on an XQueue). Regression: `SYN-graph-preempt`, `SYN-event-bridge-{lv3,lv1,modload,hazard}`, `SYN-graph-relaunch-{whole,flat}`.
- `XQueue::GetPreemptLevel()`, `XHintGetPriority()`.

### Fixed
- **DEF-32 (P0, silent data corruption)**: kernel nodes of a whole-graph launch use their function's Guardian parameter window *at launch time*; a window left by an earlier Mode 5 launch of the same function made graph kernels get evicted / drained as that launch and never restored (vLLM without flattening: wrong tokens after every preemption). Graphs now record all kernel functions at instantiation; whole-graph launches (Level-3 queue, and pass-through on ordinary streams when any window is set) clear those windows under the context `LaunchMutex` first.
- **DEF-31**: Level-3 `Reactivate()` relaunched non-killable commands (whole graphs, Level-1-routed kernels, copies) that had already run, using an `active_kernel_idx` only killable kernels update. Only killable kernels and idempotent commands are relaunched now.
- **DEF-29**: `cuStreamWaitEvent(ordinary stream, XQueue event)` blocked the caller until the event completed (vLLM `copy_stream.wait_stream(main_stream)`: 1.8 ms per decode step, Level-1 -22% tokens/s). Level-1 recording queues: wait until the record is launched, then the native GPU wait. Level-2/3: the waiting stream is promoted to an internal Level-1 XQueue whose worker waits (inherits the priority hint, drained and destroyed with the stream). `XSCHED_CUDA_ASYNC_EVENT_WAIT` (0 old blocking, 1 default, 2 unsafe test mode). A first design (GPU waits on a host flag) deadlocked against `cuModuleLoadData`, which holds a driver lock while waiting for the GPU to idle; rejected.
- **DEF-30**: the short-CTA router also requires a measured duration <= `XSCHED_SM120_SHORT_CTA_MAX_US` (default 200 us; first launch of each geometry stays Level-3 and is timed). RTX 5060 Level-3 suspend P99 under GEMM / SDPA LP 1.94 / 2.53 -> 0.22 / 0.25 ms.
- **PERF-3**: the router's occupancy query is cached per geometry instead of issued on every launch.
- **DEF-33**: at process exit the per-context trap engines were destroyed by static destruction and wrote the original trap handler back (`cuXtraMemcpyHtoD`) into contexts the CUDA runtime / cuFFT exit handlers may already have torn down: heap corruption inside libcuda (RTX 5080 `LIB-12` aborted at exit in 18/20 runs; the committed f29aab3 build too). The trap handler lives in each context's own ToolsMemory, so nothing needs restoring at exit: the trap-engine, Guardian-registry and launch-mutex singletons and the graph-plan table are now intentionally leaked. LIB-12 0/20.

### Observed
- OBS-20 (open): three processes in two RTX 5060 full regressions failed right at start (non-deterministic native golden twice, Xid 43 once), each right after a Level-3 process with preemptions; controlled reproduction 0/54. Possibly the same exit-time write as DEF-33; not seen again in the final-build regression.
- OBS-21 (open): one RTX 5080 run of `S8-prefill-lv3-slo` had 1 of 156 LP iterations differ from the golden (HP 20/20 exact, an unusual 50 ms TTFT P99 in that run); 10/10 reruns pass; ~1 in 2,700 LP iterations over the day.

### Validation
- Final build, full regression (295 entries incl. the 7 new SYN tests): RTX 5060 295/295, RTX 5080 294/295 (OBS-21); no new Xid on either GPU.

### Results
- Level-3 value: with an SM-filling LP kernel, native HP P99 grows with the LP kernel (RTX 5060 12.7 / 26.6 ms, RTX 5080 4.0-19.4 ms) while Level-3 stays at 0.8-2.6 ms; when the LP leaves SMs free, native already co-schedules the HP kernel (P99 0.7-1.8 ms) and Level-3 is 10-40% lower. Level-1 with synchronous suspension grows linearly with the LP kernel (up to 370 ms).
- vLLM 0.34 on RTX 5080 (Qwen2.5-0.5B / 3B / 7B-AWQ), 32/32 greedy outputs identical under preemption in every mode. 7B prefill HP P99: native 3.9, Level-1 9.6, Level-3 1.1 ms; 7B decode: 3.4 / 7.2 / 2.6 ms. Level-3 throughput cost (flattening CPU per node): decode -20% (7B) / -36% (0.5B), prefill -10% / -16%.

## [Unreleased] - 2026-11-26 — Documentation: principles and exploration review

- `doc/SM120_V2_PHASE3_PRINCIPLES_AND_IMPLEMENTATION_ZH.md`: every mechanism of the system as of Phase 3.4 - XQueue submission / launch / synchronization / suspend / resume, the BAR0 evict pulse, the 9 trap-handler caves, what Mode 5 saves (with the DEF-20 register-cascade correction), CTA states and the zero-pulse self-trap restore, the control header byte layout and the DebuggerParams window, the Entry Guardian slot map and its four optimizations with the SASS encodings they needed, routing rules, memory, launch-path optimizations, the BAR0 proxy, cross-process scheduling with the ready gate, scheduling bounds, a complete environment-variable reference, build / regression / deployment procedures.
- `doc/SM120_V2_PHASE3_EXPLORATION_REVIEW_ZH.md`: the development process - how each problem was found, the hypotheses and the experiments that confirmed or rejected them (including our own misdiagnoses such as DEF-25), 18 new hardware / driver / software facts with evidence, the T-1..T-24 methodology lessons, final results and open items.
- Observation IDs: the Phase 3.4-2 observations first logged as OBS-8 / OBS-9 collided with the existing OBS-8 (`SR_SMEMSZ` reserve) and OBS-9 (single control-header layout slot); they are now **OBS-14** (desktop graphics Xid 31 on a display GPU) and **OBS-15** (paused Level-1 worker launches after a log-full wait). The Level-3 display warning text follows. Ledger table extended with OBS-13..15. Word manual §11.5 lists the two new documents.

## [Unreleased] - 2026-11-26 — DEF-27 root fix: ready gate for the cross-process global scheduler

### Fixed
- **DEF-27**: under `XSCHED_SCHEDULER=GLB`, a high-priority process now holds the first launch after each Idle->Ready transition until `xserver` acknowledges that Ready event, and `xserver` sends those acknowledgements only after the lower-priority processes suspending a Level-3 (interrupt-level) queue have completed their suspensions (Level-3 suspension returns after the eviction; Level-1/2 suspensions are not waited for). So no other context waits for the GPU while warps are in the trap handler. Protocol: `SchedOperation::ReadyAck()` (`ready_ack`), `ProcessStatus::ready_seen`; client `ReadyGate` (`preempt/.../sched/ready_gate.{h,cpp}`); server `LocalScheduler::SetSuspendFirst()` (suspensions first, bounded wait on `OperationCompleteEvent`, then the rest). Default on under GLB for clients and `xserver` (`XSCHED_GLB_READY_GATE=0` disables; `XSCHED_GLB_READY_GATE_TIMEOUT_US`, default 10000; `XSCHED_GLB_READY_GATE_TRACE=1` prints statistics). With the gate on, SM120 kernels use Level-3 under GLB again; the Level-1 fallback applies only with the gate off.
- RTX 5080 `glb-lv3`: TPOT P99 3.1-3.5x without the gate (19-22 ms stalls) -> 1.10-1.29x with it (10 runs, 0 Xid). RTX 5060: 0 Xid 109 in 6 gated runs (4/4 runs had one without). High-priority gate wait 160-580 us mean (the eviction), 0 timeouts. Level-1 is unaffected: 8x8 interleaved `glb-lv1` A/B, TPOT P50 0.968x / 0.969x, overlapping P99.

### Observed
- OBS-15: a paused launch worker that waited for a free command-log slot launches that command without re-checking the pause (Level-1 work during a suspension; at Level-3 the Guardian drains it). Re-checking was tried and reverted: interleaved A/B showed no gain (request policy unchanged, `glb-lv1` slightly worse).

### Changed
- Level-3 warns once at start-up on a GPU that drives a display (`CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT`): OBS-14, two more Xid 31 events in desktop graphics processes (`Xwayland`, `gnome-shell`, both recovered) during gated runs.
- `bench_mp_serving_p34.py` refuses to start `xserver` whenever the host's IPC queue key 0x1 belongs to another user, also as root (a root server would delete another user's running server's queue and take its port).

### Validation
- RTX 5060 全量 288/288、最终构建 10/10；RTX 5080 全量 283/288（5 项为共享主机环境原因），最终构建复跑 15/15.

## [Unreleased] - 2026-11-26 — Phase 3.4-2: multi-process serving, DEF-26 / DEF-27 / DEF-28

Doc: `doc/SM120_V2_PHASE3_4_MULTIPROCESS_ZH.md`.

### Fixed
- **DEF-28**: `xserver` removed its IPC message queue (SysV key 0x1) only on SIGINT / SIGQUIT. After a root run ended with SIGTERM, a server started by another user could not replace the stale queue and clients blocked sending into it (cross-process runs hung). The server now handles SIGTERM; the multi-process harness stops it with SIGINT and refuses to start over a stale queue owned by another user.
- **DEF-26**: the local / global schedulers suspend and resume queues on their executor thread, which has no CUDA context; Level-3 `Reactivate()` failed (`sm120 PrepareLaunch failed`). `CudaQueueLv3Trap::{Deactivate, Interrupt, Reactivate}` push the queue's context when needed.

### Changed
- **DEF-27 mitigation**: with `XSCHED_SCHEDULER=GLB`, SM120 kernels use the Level-1 boundary (route `lv1_glb_sched`; `XSCHED_SM120_GLB_LV3=1` overrides). Under cross-process transparent scheduling the high-priority process keeps launching while the low-priority one is being trapped; a GPU time-slice context switch that arrives while warps are in the trap handler times out (Xid 109 `CTX SWITCH TIMEOUT`, about once per ~300 suspends) and the recovery fails both processes' contexts.

### Added
- `test/sm120_v2/eval/bench_mp_serving_p34.py`: HP and LP in separate processes (solo / native / req-lv1 / req-lv3 / glb-lv1 / glb-lv3). Request-level suspend: HP TPOT P50 1.00x of solo on both GPUs (single process: 1.02-1.07x, confirming the GIL explanation of OBS-7); transparent cross-process scheduling: LP 71-78%, TPOT P50 0.94-1.02x.
- `test_xqueue_idle_p34.py` (Idle within 50 ms after every synchronize; measured ~21 us); regression suites `SYN-xq-idle` and Suite 9 (`S9-*`, multi-process).

### Withdrawn
- DEF-25 (caller-side `WaitAll` "never reports Idle") was a measurement error: the launch worker makes the transition ~21 us after the synchronize returns. The caller-side transition added for it broke the command buffer's invariant (an Idle queue's last command is a completed `WaitAll`) and made `HW-05` assert; it was reverted before release.
- OBS-14 recorded: on the RTX 5060 that drives the desktop, two Level-3 stress runs coincided with Xid 31 MMU faults in desktop graphics processes (no crash).

### Validation
- 两卡全量回归（最终构建，含 Suite 9 与 SYN-xq-idle）288/288 + 288/288.

## [Unreleased] - 2026-11-26 — Phase 3.4-1: DEF-10 non-root Level-3 preemption

Doc: `doc/SM120_V2_PHASE3_4_DEF10_BAR0_PROXY_ZH.md`; deployment: `xsched/platforms/cuda/bar0d/README.md`.

### Added
- **`xsched-bar0d`** (`platforms/cuda/bar0d/`): proxy for the one Level-3 operation that needs root, the BAR0 evict pulse. Started as root, it maps only the two 4 KiB BAR0 pages holding the trap register (`0x424684`) and the flush register (`0x9400`), creates a `root:<group> 0660` `SOCK_SEQPACKET` socket (`/run/xsched/bar0-<bdf>.sock`), then drops to `nobody` (`setgroups(0)`, `no_new_privs`, not dumpable). It accepts `PULSE` and `HELLO` only, rate-limits each connection (token bucket, default 5000/s, burst 64), and serializes pulses from all processes. Systemd template unit `xsched-bar0d@.service` (instance = BDF).
- `Sm120Arena`: maps BAR0 directly when possible, otherwise connects to the proxy (`XSCHED_SM120_BAR0_PROXY`, `XSCHED_SM120_BAR0_FORCE_PROXY`, `XSCHED_SM120_BAR0_PROXY_TIMEOUT_MS`), checks the served PCI BDF with `HELLO`, reconnects after a broken connection; shared `FireBar0Pulse()` in `sm120_bar0_proxy.h` for both paths.
- `test/sm120_v2/eval/test_bar0_proxy_p34.py`: protocol checks that never touch hardware.

### Changed
- The "BAR0 pulse failed" warning is limited to one per second with a running count.

### Validation
- Non-root: RTX 5060 37/37, RTX 5080 50/50 (`LLM-08` 24/24, two processes preempting concurrently, multi-wave stress, library/hardware cases, 8 production ops bit-exact). Fault injection: no proxy, no permission, rate limit, wrong-GPU proxy, over-long path, proxy killed and restarted mid-run (in-process reconnect) all degrade to run-to-completion or recover as designed.
- Cost: socket round trip 4.5 us (RTX 5080 host) / 12 us (RTX 5060 host) P50; same-session A/B suspend latency +2-15 us versus direct mapping.
- 两卡全量回归（root 直接映射路径）265/265 + 265/265.

## [Unreleased] - 2026-11-26 — Phase 3.3 completion: short-CTA routing, per-call submission path, PERF-2, DEF-24

Final build, full regression (`regress_p3.py --suites full`, incl. the Phase 3.2 acceptance group and arena-trim-under-preemption runs): **RTX 5060 265/265, RTX 5080 265/265**.

Report: `doc/SM120_V2_PHASE3_2_3_3_REPORT_ZH.md` (§3.6, §4, §5, §6, §8); evidence: `doc/evidence/phase3_2_3_3_20261126/`.

### Performance
- **DEF-17 target met (< 10%)**: loop-free kernels (no backward / indirect branch, no calls; `BinaryIsLoopFree()`, BRA offset encoding fitted on 3,653 branches) with 2-1024 waves of CTAs are routed to the Level-1 boundary (`XSCHED_SM120_SHORT_CTA_LV1`, `_MIN_WAVES`, `_MAX_WAVES`). Final full regression with the BAR.RED Guardian and the gate mirror (router off = pure Mode 5 -> default): RTX 5080 `transpose_copy` +1.2% -> -0.0%, `rope_apply` +7.3% -> +2.2%, `sort_16m` +3.3%; RTX 5060 +1.9% -> +0.1%, +8.0% -> +1.1%, `sort_16m` +5.7%; Lv3 median +0.3% on both GPUs, worst op +5.7%. Trade-off: routed kernels preempt at their own end (RTX 5060 `transpose_copy` HP P50 1.09 -> 2.37 ms).
- **Per-call submission**: direct submit (the submitting thread launches CUDA kernels itself when the launch worker is asleep; `HwCommand::DirectLaunchable()`, `CommandBuffer` gate, `XSCHED_XQUEUE_DIRECT`), caller-side `WaitAll` (the waiting thread synchronizes the stream itself, falls back on suspension; `XSCHED_XQUEUE_CALLER_SYNC`), spin before background batch sync, control-header layout cache (`XSCHED_SM120_LAYOUT_CACHE`). C++ micro-benchmark (~300 us kernel + host gap): RTX 5060 Lv1/Lv3 +10.4/+11.7 -> +2.0/+3.1 us, RTX 5080 +6.5/+7.8 -> +1.9/+3.0 us; Lv3 launch 4.10 -> 2.1 us.

### Added (2026-11-27)
- **Gate mirror** (default on, `XSCHED_SM120_GATE_MIRROR=0` disables, `XSCHED_SM120_GATE_MIRROR_CHECK=1` verifies every mirror line against the header on each host read): the control allocation carries 256 x 128 B mirror lines `{cmd, is_restore_wave}` before the header; each Guardian CTA reads its SM's line (`S2R SR_VIRTUALSMID` + one `LD.E.64 [u_ctrl-0x8000+128*smid]`) instead of two serialized loads of the hot header line. The host writes mirrors exactly like the header (PreemptStream: cmd only; PrepareRestore, the 256-thread conditional clear kernel, reset, reallocation: both fields). RTX 5080 pure Mode 5: `sort_16m` +14.8% -> +3.3%, `topk_sampling` +9.9% -> +4.8%, `rope_apply` +54.1% -> +7.3%, `transpose_copy` +53.4% -> +1.1%. A stale-`is_restore_wave` bug in the first version (illegal instruction under multi-wave stress with mid-restore re-preemption) was found by the checker and fixed before enabling.

### Added (evening)
- `BAR.RED` norm_gate (default on, `XSCHED_SM120_GUARDIAN_BARRED=0` disables): the Guardian barrier becomes `BAR.RED.OR 0, P0` over `P0 = (tid == 0 && cmd != 0)` with `B2R.RESULT`, and the post-barrier `entry_dec` load runs only when the reduction is true. restore_path / replay_state_c move by one slot (runtime anchors shared with the trap engine); default-layout bytes unchanged. RTX 5060 `sort_16m` +14.8% -> +8.0%, `batchnorm_train` +7.0% -> +3.8%; RTX 5080 `sort_16m` +15.6% -> +14.8%, `batchnorm_train` +4.5% -> +2.1%. Validation before enabling: 5060 40/40, 5080 62/62 incl. LLM-08 36/36.

### Added (afternoon)
- Adaptive direct submit: a call whose predecessor (commands between two `WaitAll`s) had one command goes direct even while the launch worker is awake. Single-kernel calls: RTX 5080 Lv1/Lv3 +1.2/+2.4 us (was +3.8/+5.2), RTX 5060 +1.1/+2.2 us.
- **MEM-1** arena trimming (`XSCHED_SM120_ARENA_TRIM`, `XSCHED_SM120_ARENA_TRIM_MIN_MB`): after a stream synchronization with no launch in between and no pending eviction state, arenas retired by growth are freed and an arena that stayed > 2x the largest requirement (and above the threshold) for 64 synchronizations is released.
- Serving bench `--lp-sms` (LP on a CUDA green context SM partition) and `--switch-interval-ms`; spatial-sharing results in the report §5.3.

### Fixed
- **PERF-2**: every pass-through kernel launch (stream without an XQueue, e.g. the high-priority stream) resolved `CUkernel -> CUfunction` via `cuXtraKernelGetFunction` in the command constructor (8.4% of all CPU samples in a PyTorch decode run) and cleared the Guardian params with a driver call. Now resolved lazily (`CudaKernelCommand::FuncHandle()`), params cleared only for launch handles that may hold Mode 5 params (`Sm120StreamContext::MaybeParamsSet()`, still under the launch mutex; `XSCHED_SM120_ALWAYS_CLEAR_PARAMS=1` restores the old behaviour), and the `DirectLaunch` context/arch checks cached per thread. RTX 5060 in-process decode TPOT with XSched loaded: +16% -> +1.3-3.8% over no XSched.
- **DEF-24**: indirect calls (`CALL.REL.NOINC Rx`, 0x344) and callee code (`RET`, 0x950) are detected as device calls.
- `LaunchWorker::SyncAllWithLock` only completes commands launched before the synchronization (previously all in-flight commands).
- `LIB-06` test: LP workload extended to 4 in-place FFT passes (a single pass ended before the harness could preempt), bitwise comparison; 23 pageable H2D uploads in `test_libs_and_features.cu` now synchronize the device (T-11 race).

### Removed
- Unreferenced `CudaContextRegistry` (`context_registry.{h,cpp}`, kept in commit 84e10dc).

### Added
- `test_percall_p3.cu` per-call micro-benchmark; regression `acc` group (C1 x10 np=2/3, 12-kernel chain, HW-06 x20, dnn06 x5), LIB-08 x30, Suite 7 `lv3m5` (router off) column; mechanism suites run with the router off.

## [Unreleased] - 2026-11-25 (evening) — Phase 3.2 correctness close-out & Phase 3.3 performance

Report: `doc/SM120_V2_PHASE3_2_3_3_REPORT_ZH.md`; evidence: `doc/evidence/phase3_2_3_3_20261125/`.
Full regression after all changes (`regress_p3.py --suites full`): **RTX 5060 184/184, RTX 5080 184/184**
(synthetic 5, Suite 1 25, Suite 4 6, Suite 5 8, Suite 6 8, Suite 7 116 = 29 ops x 4 modes, Suite 8 16).

### Fixed
- **DEF-21 closed**: the synthetic "DEF-12 logic fails C1 on RTX 5060" result was a test-harness race (pageable `cudaMemcpy` H2D vs. a non-blocking stream, T-11). With `H2D()` (copy + device sync): 36 processes, 0 failures; stress (multi-wave grids, 4-6 queued kernels, 300 us-1 ms gaps, 8-10 preemptions per round) ~3,000 preemptions all bit-exact, about half hitting a restore wave.
- **DEF-22**: `BinaryHasDeviceCalls` also matches `CALL.ABS.NOINC` (`0x343`); `XSCHED_SM120_DETECT_CALL_ABS=0` disables it.
- **DEF-23 (new)**: the LMEM save area is static local size, plus `CU_LIMIT_STACK_SIZE` for kernels with device calls (was a fixed 4 KiB/thread); LayerNorm arena 231 -> 103 MiB. `XSCHED_SM120_LMEM_CAP_BYTES` overrides.
- **DEF-19 mitigated**: kernels with `ClusterSize > 1` route to Level-1 by default (`XSCHED_SM120_CLUSTER_MODE5=1` restores single-wave Mode 5).

### Performance
- **DEF-17 Entry Guardian**: the default guardian no longer re-reads `cmd` at slot 28 (hot control-header L2 line). RTX 5080 Lv3 batch overhead: `rope_apply` +101% -> +56.5%, `transpose_copy` +82% -> +53.9%, `embedding` +22.8% -> +1.9%, `gelu_tanh` +18.3% -> +0.3%, `sort_16m` +24.9% -> +15.7%; RTX 5060 `transpose_copy` +28.5% -> +18.1%. All anchors unchanged.
- **Experimental guardian v2** (`XSCHED_SM120_GUARDIAN_V2=1`, sub-switches `_V2_SERIAL_HDR` / `_V2_SERIAL_SREG`): asynchronous control-header and `S2R/LDC` issue on scoreboards SB2-SB5. Not default: LLM-08 (two Lv3 queues + HP) failed 11/61 vs. 0/78 with serialized issue on RTX 5080. `XSCHED_SM120_GUARDIAN_V1=1` restores the exact original guardian.
- **XQueue bounded spin** (`xqueue/spin.h`, `XSCHED_SPIN_US`, default 30, 0 disables): the worker and `XCommand::WaitUntil` spin before blocking on condition variables; Lv3 single-call latency -3 to -5 us.
- SASS helpers `with_ctl()` / `with_extra_wait()` in `sm120_sass.h`.

### Added
- Suite 8 scheduling policies `--policy request|prefill|token|slo`, `--tpot-slo-ms`; suspension time is counted in TTFT/TPOT.
- Regression pipeline: `regress_p3.py` (plan / summarize), `build_regress_bins.sh`, `regress_env.example.sh`; `test_convbar_p3.cu` stress knobs `GRID_MULT/CHAIN/GAP_US/TARGET_MS`, `NP_LIST`, `MISMATCH_DETAIL`.

## [Unreleased] - 2026-11-25 — Phase 3 production-operator & grounded real-model testing, DEF-20 fix

Summary & roadmap: `doc/SM120_V2_PHASE3_SUMMARY_AND_ROADMAP_ZH.md`.

### Added
- **Suite 7 `bench_prod_ops_p3.py`**: 29 production operators (cuBLAS BF16/FP8/INT8, MoE bmm, FlashAttention SDPA prefill/decode, RoPE, RMSNorm/LayerNorm/Softmax/SwiGLU/GELU/CE, Triton RMSNorm, cuDNN conv/BN, CUB reduce/scan/sort, topk, embedding, int32 atomics, dtype cast, transpose, fused AdamW, linear backward with grad accumulation) x {native, shim, lv1, lv3}, one process per (op, mode), per-call on-GPU bit-exact checks, golden SHA-256 digests, native high-priority-stream co-run baseline.
- **Suite 8 `bench_real_serving_p3.py`**: real `Qwen2.5-0.5B-Instruct` weights; online chat requests (Poisson, greedy decode + KV cache, TTFT/TPOT) co-located with offline perplexity / prefill evaluation on the same weights; exact-token and bit-exact LP checks.
- **`test_partial_exit_p3.cu`**: synthetic Mode 5 evict/restore of partially exited, divergent and reconverging warps (5/5 PASS on RTX 5060).
- **Diagnostics**: exported `XSm120GetRouteStats()` (adaptive-router counters); `XSCHED_SM120_TRACE_LAUNCH=1` (per-launch kernel name/geometry, reactivate telemetry, evicted-CTA trap PCs as original-SASS offsets via `Sm120StreamContext::DebugDumpEvicted`); `XSCHED_SM120_DUMP_DIR` (dump original SASS for `nvdisasm -b SM120`); kernel name + binary size in the instrumentation log.

### Fixed (working tree, not yet committed) — see `doc/SM120_V2_PHASE3_DEFECT_LEDGER_ZH.md`
- **DEF-20 (P0, root cause of DEF-14/15/16, LIB-08 and LLM-07 m5 intermittents)**: the Cave 2/5 register cascade used `SR_REGALLOC - 3`, but `SR_REGALLOC` equals the kernel's exact register count (measured with a binary-patched `S2R SR_REGALLOC` probe), so the top 1-3 live registers were never saved/restored (R14/R15 entirely for 14-16-register kernels). Fix: `SR_REGALLOC - 1` in `sm120_trap_engine.cpp` + round the instrumented register count up to even (`<255`) in `sm120_guardian.cpp` so 64-bit register pairs never straddle the allocation. Validation: Suite 7 Lv3 29/29 on both GPUs and both builds; RTX 5080 LIB-08 60/60, LLM-07 m5 10/10, Suites 1/4/5/6 all pass (with DEF-12 kept).
- Added `test_convbar_p3.cu` (nested-divergence, register-pressure and per-accumulator attribution cases) and `test/sm120_v2/eval/repro/` (SR_REGALLOC probe, minimal cast repro, in-process drift check).

### Still open
- **DEF-21 (P1)** preemption while a restore wave is in flight: HEAD logic loses CTAs (DNN-06, real-model serving LP), the working-tree DEF-12 logic fails the synthetic 2-preemption C1 case on RTX 5060.
- **DEF-19 (P2)** CTA cluster + DSMEM (HW-06) intermittent after the fix (WT 9/10, VAR 6/10).
- **DEF-22 (P2, latent)** `BinaryHasDeviceCalls` misses `CALL.ABS.NOINC` (opcode `0x343`).
- **DEF-18 revised**: the LIB-08 regression signal disappeared after the DEF-20 fix; keep the DEF-12 change.

### Found (open, see `doc/SM120_V2_PHASE3_PRODUCTION_OPS_GROUNDED_TESTS_ZH.md` §5)
- **DEF-14 (P0)** Mode 5 restore crash (700/717) on PyTorch `unrolled_elementwise_kernel<direct_copy_kernel_cuda>` (all dtype casts) after the first State-B eviction; both GPUs; also on committed HEAD.
- **DEF-15 (P0)** Restore crashes on CUB onesweep sort / CUB integer scan / `mbtopk` (and `LIB-08` NP=2, `LLM-07 m5` intermittently).
- **DEF-16 (P0, silent)** Triton fused RMSNorm returns wrong results under Lv3 preemption without any error.
- **DEF-17 (P1)** Entry Guardian per-CTA cost: up to +101% steady-state overhead for short-CTA memory-bound kernels (median +1.7%).
- **DEF-18 (P1)** The uncommitted DEF-12 restore-path change raises `LIB-08` NP=2 failure rate from 15% (HEAD, 6/40) to 38% (19/50), Fisher p = 0.019; keep HEAD behaviour until root-caused.

## [v2.2.1-sm120] - 2026-11-24

### Dual-GPU Closed-Loop Verification (`RTX 5080 GB203` + `RTX 5060 GB206`)
- **RTX 5080 (`GB203`, 84 SMs, 16 GiB, Driver `580.95.05`, CUDA `12.9`, PyTorch `2.11+cu128` / cuDNN `9.19` / Triton `3.6.0`) Full Verification**:
  - Zero-warning `C++20 -Wall -Wextra -Werror` build, `79/79` SASS disasm, `11 Caves / 1,703 slots`, **`11 / 11 E2E ALL_PASS`**, **`P50 = 420 µs` CILP** (`ZERO_LOST_WORK`).
  - **`sm120_v2_bench` VRAM Speedups (`v2.0.0` → `v2.2.1`, RTX 5080)**: Entry Guardian `30.78 µs → 0.99 µs` (**31.1×**); Mode 1 `512 CTAs` Evict `971.6 → 110.1 µs` (**8.8×**); Mode 5 `84 CTAs 0K/32K` Evict **`50.9 / 55.4 µs`** (**6.5×–6.7×**), Restore **`30.5 / 41.4 µs`** (**20.1×–25.0×**); Mode 5 `512 CTAs 48K` Evict **`119.9 µs`** (**10.5×**), Restore **`292.0 µs`** (**6.4×**); compact `u_arena` **`139.7×`** (`29.31 MiB` vs `4,096 MiB`).
  - **Suite 1 (`15 / 15 PASS`) + `DEF-03/06` (`PASS`)**: `cuBLAS` `80–137 µs`, `cuFFT` `140 µs`, `cuSPARSE` `113 µs`, `CUB` `55 µs`, `mma.sync` `34 µs`, `cp.async` `114 µs`, **single-wave Blackwell CTA Cluster (`blocks=60 <= sm=84`) dynamically routed to Mode 5 (`111 µs`, `0/7,680`)**, `DEF-03` `34 µs`.
  - **Suite 2**: LLM Serving `HP P50 = 0.270 ms`, `SUSP P50 = 32 µs`, `LP_ret = 102.4%`; 100% SM Saturated Heavy QoS `HP P50 = 0.060 ms` (**102.6×** lower than Lv1 `6.155 ms`), `SUSP P50 = 45 µs` (**136.5×** faster than Lv1), `SLO ≤ 2 ms = 100%`, `LP_ret = 103.4%` (`+27.3 pp` over Mode 1 `76.1%`); Cross-Context BAR0 Bystander **`0.011 ms → 0.011 ms` (`0.000 ms` overhead even at P99/Max `0.012 ms`)**.
  - **Suite 3 (`11 / 11 PyTorch PASS`)**: ResNet-50 `70 µs`, ResNet-18 AdamW training `31 µs` (`7.45e-9`), 4× Llama SDPA `23–991 µs` (`0.000e+00`), Triton 3.6.0 `126 µs` (`0.000e+00`), `PT-06` `empty_cache` `30 µs` (`0.000e+00`), `PT-05-QOS` `HP P50 = 0.290 ms` (`SLO ≤ 2 ms = 100%`, `SUSP P50 = 224 µs`, `LP_ret = 105.3%`).

### Additional 84-SM Edge-Case Fixes Discovered on RTX 5080
- **SMEM Hardware Quantum Fix (`sm120_trap_engine.cpp`)**: Compute `smem_area = AlignUp(total_user_smem + 0x400, 4096)` (when `total_user_smem > 0`) to account for the `0x400 B` driver-reserved region before Blackwell's 4 KiB SMEM allocation quantum (prevents 3 KiB tail-slot overrun on kernels like `CUB Onesweep` with `static_smem = 36,352 B -> SR_SMEMSZ = 40,960 B`).
- **Stream-Ordered `is_restore_wave` Clear (`sm120_trap_engine.cpp`)**: `sm120_cond_clear_hdr` now unconditionally clears `is_restore_wave` (`+0x1c`) upon completion of the restored kernel while keeping `ctrl_dev_[0..27]` conditional on `cmd == 6 || cmd == 2`.
- **Entry Guardian Drain `active_kernel_idx` Tracking (`sm120_guardian.cpp`)**: Replaced 3-instruction `active != 0` check with 1-instruction `lop3_or3_p0(10, 11, kRegZero)` (zero slot shift, `bar_idx` stays at exact slot 39 / `0x270`) and recorded `active_kernel_idx = my_idx` on Entry Guardian State C drains (`st_e_64(4, 0x08, 8, kP0)`).
- **Dynamic SM-Count Delay Scaling (`test_libs_and_features.cu`, `test_pytorch_llm_dnn.py`)**: Automatically scales mid-execution preemption trigger delays by `30 / max(30, sm_count)` across 30-SM RTX 5060 and 84-SM RTX 5080.

## [v2.2.0-sm120] - 2026-11-24

### Performance Breakthroughs (10× – 548× Speedup on RTX 5060 `GB206`)
- **DEF-01 & DEF-08 — 100% GDDR7 VRAM Control Plane + High-Priority Copy Engine DMA (`sm120_trap_engine.h/.cpp`)**:
  - Migrated `Sm120ControlHeader` + `Sm120CtaEntry[65536]` (`ctrl_dev_`) from PCIe Zero-Copy pinned host RAM into GPU GDDR7 VRAM, eliminating PCIe Gen5 round-trip latency (~1.6–2.0 µs/atomic) in the Entry Guardian and Trap Handler hot paths.
  - Added dedicated high-priority `ctrl_op_stream_` using pure Copy Engine `Driver::MemcpyHtoDAsync_v2` / `MemcpyDtoHAsync_v2` (never blocked by 100% SM saturation, unlike compute-engine `cuMemsetD32Async`).
  - Added GPU-native 1-thread PTX helper `sm120_cond_clear_hdr` (~2 µs, strictly ordered on `kStream`) replacing host callback `cuLaunchHostFunc` (`PostRestoreClearHostCb`).
  - **Verified Before/After Impact (`v2.1.0` → `v2.2.0`, RTX 5060)**:
    - Entry Guardian Delta: `43.25 µs` → **`0.43–0.97 µs` (45×–100× faster)**
    - Mode 1 Evict P50 (`512 CTAs, 48K SMEM`): `901.2 µs` → **`38.0 µs` (23.7× faster)**
    - Mode 5 Evict P50 (`512 CTAs, 48K SMEM`): `1,076.2 µs` → **`48.0 µs` (22.4× faster)**
    - `cuBLAS` / `cuSPARSE` Suspend: `66.1 ms` / `50.8 ms` → **`0.856 ms` / `0.109 ms` (77× / 466× faster)**
    - LLM Serving (`15 ms` QoS): `HP P50 32.16 ms → 0.475 ms (67.7×)`, `SUSP P50 31.32 ms → 0.057 ms (548×)`, `SLO ≤ 2 ms: 0% → 100%`, `LP_ret: 19.3% → 98.2%`
    - 100% SM Saturated Heavy QoS (`15 ms`): `HP P50 0.083 ms` (85× lower than Lv1 `7.091 ms`), `SUSP P50 0.066 ms`, `SLO ≤ 2 ms: 100%`, `LP_ret: 100.7%`
    - Real PyTorch `PT-05-QOS` (`m5`): `HP P50 73.76 ms → 0.552 ms (133.6×)`, `SUSP P50 73.45 ms → 0.454 ms (161.8×)`, `SLO ≤ 2 ms: 0% → 100%`, `LP_ret: 0.7% → 101.9%`

### Production Hardening & Universal Compatibility (`DEF-02` .. `DEF-07` + Race Fix)
- **DEF-02 & Phase 2 Adaptive Kernel/Mode Router (`sm120_guardian.h/.cpp`, `cuda_queue.h/.cpp`, `xcommand.h`)**:
  - Static SM120 SASS scanner (`BinaryHasHwMbarrier`) detects `SYNCS` hardware `mbarrier` opcodes (`op9 = 0x1a7 / 0x1b2`) during one-time instrumentation.
  - `CudaQueueLv3Trap::OnHwCommandSubmit` adaptively routes `SYNCS/mbarrier`, Cooperative kernels, multi-wave CTA Clusters (`ClusterSize > 1 && BlockCnt > sm_count`), and CUDA Graphs cleanly to Level-1 boundary (`killable = false`, `ClearProps(kCommandPropertyDeactivatable)`), bringing Suite 1 to **15 / 15 PASS**.
- **DEF-03 — Memory-Safe Deferred Free Queue (`xqueue.h/.cpp`, `async_xqueue.h`, `shim.cpp`)**:
  - `XMemFree_v2` detects any suspended `XQueue` via `XQueueManager::HasAnySuspended()` and defers freeing into `g_deferred_frees` instead of blocking on `WaitAllActive()` (eliminating mid-suspend `cudaFree` / `torch.cuda.empty_cache()` deadlock while guaranteeing suspended LP kernels' buffers stay live until resume). Verified in `test_def03_def06.cu` and PyTorch `PT-06`.
- **DEF-04 — Intercept `cuLaunchCooperativeKernel[_ptsz]` (`cuda_command.h`, `shim.h/.cpp`, `intercept.cpp`)**:
  - Added `CudaCooperativeKernelLaunchCommand` and `XLaunchCooperativeKernel[_ptsz]` interception so cooperative kernels enter XQueue tracking and cleanly route to Level-1 boundary (`HW-07`).
- **DEF-05 — PyTorch ResNet-18 Training `cuDNN` Async SMEM Race & `LaunchEx` Deep Copy (`cuda_command.h/.cpp`)**:
  - Fixed `CUDA_ERROR_INVALID_VALUE (1)` in `PT-02` caused by `cuDNN` updating `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` on the main thread before async `LaunchWorker` launched an earlier larger-SMEM BatchNorm kernel (`EnsureDynamicSmem` zero-hot-path retry).
  - Fixed `CudaKernelLaunchExCommand` attribute deep copy when `param_cnt_ == 0` and max-extent `buffer_size` copy for `params_`. Verified `PT-02` PASS (`max_weight_err = 5.472e-09`, `avg_susp = 0.023 ms`).
- **DEF-06 — Context-Free Worker Thread Automatic `StreamGetCtx` Fallback (`shim.h/.cpp`)**:
  - Added RAII `ScopedStreamContext` so threads with `cuCtxGetCurrent() == nullptr` seamlessly resolve and bind the stream's CUDA context. Verified in `test_def03_def06.cu`.
- **DEF-07 — `max_slots` Occupancy Slack & Restore-Wait Cap (`sm120_trap_engine.cpp`)**:
  - Raised `max_slots` slack to `std::max(64u, sm_count * 2)` and capped prior-restore wait target to `std::min(evicted_slots, max_slots)`.
- **Upstream `CudaQueueLv3Trap` Concurrency Fix (`cuda_queue.h/.cpp`)**:
  - Guarded `last_kernel_` with `last_kernel_mtx_`, removed unsynchronized `OnHwCommandSubmit` write, and prevented trailing `CudaEventRecordCommand`/memory commands from clobbering `last_kernel_`. Eliminated `PT-05-QOS` `launch_worker.cpp:98` crash.
- **Test Harness Hygiene (`test_pytorch_llm_dnn.py`, `test_def03_def06.cu`, `run_all_eval.sh`)**:
  - Added `test_def03_def06.cu` and PyTorch `PT-06`; added missing `torch.cuda.synchronize()` after default-stream tensor creation before non-blocking stream consumption in `test_pytorch_llm_dnn.py`, achieving **11 / 11 PyTorch PASS** (`0.000e+00` bit-exact across ResNet-50, 4 Llama SDPA backends, Triton 3.6.0, and `PT-06`).

## [v2.1.0-sm120] - 2026-11-24

### Added
- **Comprehensive 5-Dimension Evaluation Suite** (`xsched/test/sm120_v2/eval/`):
  - `test_libs_and_features.cu`: 8 closed-source NVIDIA CUDA libraries (`cuBLAS`, `cuBLAS GemmEx FP16/BF16`, `cuBLASLt`, `cuFFT`, `cuSPARSE`, `CUB`) + 7 Blackwell HW boundary tests (`mma.sync`, `cp.async`, `mbarrier`, `CUDA Graph`, Idle Suspend, CTA Cluster + DSMEM, Cooperative Groups).
  - `bench_dnn_llm_scheduling.cu` & `bench_heavy_qos.cu`: LLM Serving (LP Prefill vs HP Decode), Non-Idempotent DNN Training (in-place AdamW vs HP Inference), 100% SM-saturated QoS, and 5/10/20/50 ms frequency sweep across Native / Level-1 / Mode 1 / Mode 5.
  - `bench_bar0_cross_ctx.cu` & `bystander_pure.cu`: Cross-context BAR0 broadcast collateral interference benchmark.
  - `test_pytorch_llm_dnn.py`: Unmodified PyTorch 2.11 + `torchvision` ResNet-50/18 + `cuDNN 9.19` + 4 Llama SDPA backends (`math`/`flash`/`efficient`/`cudnn`) + `Triton 3.6.0` preemption & QoS suite.
  - `run_all_eval.sh`: One-command reproducible runner with built-in `nvidia-smi` GPU idle safety check and MPS isolation.
- **Authoritative Technical Documentation (`doc/`)**:
  - `doc/SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md`: Full real-world library, DNN/LLM, PyTorch/Triton evaluation & defect analysis.
  - `doc/SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md`: Milestone status report, 10-item P0/P1/P2 problem ledger, Phase 1–3 roadmap, and 5080 sync guide.
  - `doc/SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md`: Cross-chip (`GB206` vs `GB203`) and cross-driver (`595.99.02` vs `580.95.05`) adaptation & difference analysis.

### Fixed (6 Deep HAL Bugs Uncovered Under Closed-Source Libraries)
- **FIX-1 (`cuda_queue.cpp`)**: Prevent idle-queue `Suspend/Resume` from poisoning `ctrl_host_->cmd = 6` (`kCmdMode5Restore`) via `has_preempted` check and `ResetControlSync()`.
- **FIX-2 (`sm120_trap_engine.h/.cpp`)**: Replace unconditional `EnqueuePostRestoreClear` DMA zeroing with `PostRestoreClearState` (dedicated micro-mutex + monotonic `epoch`) `Driver::LaunchHostFunc`, eliminating stale wipe on double preemption (`NP >= 2`).
- **FIX-3 (`sm120_trap_engine.cpp`)**: Guard `EmitDynamicSmemLoop` with `SR_SMEMSZ >= 0x401` before subtracting `0x400` (fixes unsigned underflow `cuda error 717` on 0-SMEM kernels) and align `smem_area` to `4096 B`.
- **FIX-4 (`sm120_trap_engine.h/.cpp`)**: Track exact per-kernel arena VA via `kernel_arena_map_[kernel_idx]` (fixes multi-kernel growth `cuda error 700`), expand `layout_stage_` to 256 slots (16 KiB), and raise initial `ctrl_cta_cap_` to 65,536 CTAs with stream sync before growth.
- **FIX-5 (`sm120_guardian.h/.cpp`)**: Route Entry Guardian uniformly by wave-invariant `is_restore_wave` (`+0x1c`) instead of mutable `cmd`, eliminating same-CTA warp split deadlock on `cmd: 6 -> 5` transitions.
- **FIX-6 (`sm120_trap_engine.cpp`)**: Upgrade Cave 0 to 5-interval deadlock-free consensus distinguishing `entry_dec == 2` (Guardian-decided drain: bypass `bar_sync_trap()`) from `entry_dec == 4` (Trap-decided drain: safe `bar_sync_trap()`), shrinking Cave 0 to `139/152` slots (`1,045/1,390` total).

## [v2.0.0-sm120] - 2026-11-23

### Changed
- Completed Blackwell `sm_120` Level-3 v2 clean architecture refactor: simplified to Mode 1 (`kEarlyExit`) + Mode 5 (`kTrapFrame`) + decoupled Context CILP (`ResumableSm120Context`).
- Implemented static 9-Cave engine, permanent `0x500` B 3D Entry Guardian, user-space `BPT.TRAP 0x1` 0-pulse restore, compact `slot_id` pooled `u_arena` (160.6×–436.9× VRAM reduction), D10 `NV_PTIMER_TIME_0` BAR0 pulse safety, and `C++20 -Wall -Wextra -Werror` zero-warning build.
- Verified `11/11 ALL_PASS` × 3 runs on RTX 5080 (`580.95.05`), 79/79 `nvdisasm -b SM120` encoding checks, and cross-driver cave discovery on RTX 5060 (`595.99.02`).
- Removed legacy v1 drafts, vendored NVIDIA driver source trees, and temporary build artifacts.
