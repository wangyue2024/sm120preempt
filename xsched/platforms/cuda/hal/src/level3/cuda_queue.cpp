#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

#include "xsched/utils/env.h"
#include "xsched/utils/common.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/level3/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

namespace
{

/// Suspend / resume may be called from a thread without the queue's context: the scheduler
/// executor thread of the local / global schedulers (XSCHED_SCHEDULER=LCL / GLB) has none.
/// Reactivate() relaunches kernels and the trap engine allocates / writes control memory, so
/// make the queue's context current for the duration of the call (DEF-26).
struct CtxScope
{
    bool pushed = false;
    explicit CtxScope(CUcontext ctx)
    {
        CUcontext cur = nullptr;
        if (ctx != nullptr && (Driver::CtxGetCurrent(&cur) != CUDA_SUCCESS || cur != ctx)) {
            pushed = Driver::CtxPushCurrent_v2(ctx) == CUDA_SUCCESS;
        }
    }
    ~CtxScope()
    {
        CUcontext popped = nullptr;
        if (pushed) (void)Driver::CtxPopCurrent_v2(&popped);
    }
};
// Process-wide SM120 adaptive-router telemetry (relaxed counters, test/eval observability only).
// Counts every kernel/graph submitted to a Level>=2 SM120 trap queue by routing decision.
enum Sm120RouteStat : int {
    kRouteLv3Killable = 0,  // instrumented, in-kernel preemptible (Mode 1 / Mode 5)
    kRouteLv1Mbarrier,      // SYNCS / hardware mbarrier kernel -> Level-1 boundary
    kRouteLv1Cooperative,   // cooperative launch (grid.sync) -> Level-1 boundary
    kRouteLv1Cluster,       // multi-wave thread block cluster -> Level-1 boundary
    kRouteLv1Graph,         // CUDA graph launch -> Level-1 graph boundary
    kRouteLv1Uninstrumentable, // Entry Guardian instrumentation failed -> Level-1 boundary
    kRouteLv1ShortCta,      // DEF-17: loop-free multi-wave kernel (short CTAs) -> Level-1 boundary
    kRouteLv1GlobalSched,   // DEF-27: cross-process global scheduler (XSCHED_SCHEDULER=GLB) -> Level-1
    kRouteRfFull,           // DEF-34: killable kernel whose resident CTAs allocate the whole register file
    kRouteLv1RfFull,        // DEF-34: ... routed to the Level-1 boundary (XSCHED_SM120_RF_FULL=lv1)
    kRouteLv1NoEngine,      // tier-1 item 4: trap engine unavailable (unknown driver layout) -> Level-1
    kRouteStatCount,
};
std::atomic<uint64_t> g_sm120_route_stats[kRouteStatCount] = {};

// Duration-aware short-CTA routing (tier-1 item 2 follow-up). The wave-count window alone admitted
// kernels lasting 1-2 ms (large elementwise kernels on the 5060); a suspend that arrives while such
// Level-1-routed kernels are queued behind the trapped kernel waits for all of them (Lv3 HP P99 2-3
// ms vs 0.5-1 ms with the router off). The first launch of each candidate (function, CTA count)
// stays at Level-3 and is timed with CUDA events; later launches take the Level-1 route only if the
// measured duration is <= XSCHED_SM120_SHORT_CTA_MAX_US (default 200 us; 0 = wave window only).
// A measurement inflated by a preemption classifies the kernel as long (the safe side).
struct ShortCtaProbe
{
    int rf_full = -1;         // DEF-34: resident CTAs allocate the whole register file (-1 not computed)
    int window = -1;          // wave-window decision for this (function, geometry): -1 not computed
    int state = 0;            // 0 unknown, 1 short (route Level-1), 2 long (stay Level-3)
    bool pending = false;     // a timed launch was submitted
    bool recorded = false;    // ... and its events were recorded by Launch() (an event that was never
                              // recorded reads as complete, so it must not be queried before)
    CUevent beg = nullptr, end = nullptr;
};
std::mutex g_probe_mtx;
std::unordered_map<uint64_t, ShortCtaProbe> g_probes;

int64_t ShortCtaMaxUs()
{
    static const int64_t v = [] { int64_t x = 200; GetEnvInt64("XSCHED_SM120_SHORT_CTA_MAX_US", x); return x; }();
    return v;
}

inline void CountRoute(Sm120RouteStat s)
{
    g_sm120_route_stats[s].fetch_add(1, std::memory_order_relaxed);
}
} // namespace

/// @brief Read (and optionally reset) SM120 adaptive-router counters.
/// @param out   array of at least `n` uint64: [lv3_killable, lv1_mbarrier, lv1_cooperative,
///              lv1_cluster, lv1_graph, lv1_uninstrumentable, lv1_short_cta, lv1_glb_sched]
/// @param n     number of entries to copy (clamped to 7)
/// @param reset non-zero to atomically reset the copied counters to zero
/// @return number of entries written
EXPORT_C_FUNC int XSm120GetRouteStats(uint64_t *out, int n, int reset)
{
    if (out == nullptr || n <= 0) return 0;
    if (n > kRouteStatCount) n = kRouteStatCount;
    for (int i = 0; i < n; ++i) {
        out[i] = reset ? g_sm120_route_stats[i].exchange(0, std::memory_order_relaxed)
                       : g_sm120_route_stats[i].load(std::memory_order_relaxed);
    }
    return n;
}

bool CudaQueueLv3Trap::IsSm120Stream(CUstream stream)
{
    (void)stream;
    CUdevice dev = 0;
    if (Driver::CtxGetDevice(&dev) != CUDA_SUCCESS) return false;
    int major = 0, minor = 0;
    if (Driver::DeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS ||
        Driver::DeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS) {
        return false;
    }
    return (major * 10 + minor) == 120;
}

sm120::Sm120PreemptMode CudaQueueLv3Trap::SelectSm120Mode(const std::shared_ptr<HwCommand> &hw_cmd) const
{
    if (const char *env = getenv("XSCHED_SM120_FORCE_MODE1")) {
        if (env[0] == '1') return sm120::Sm120PreemptMode::kEarlyExit;
    }
    if (hw_cmd != nullptr && hw_cmd->GetProps(kCommandPropertyIdempotent) != 0) {
        return sm120::Sm120PreemptMode::kEarlyExit;
    }
    return sm120::Sm120PreemptMode::kTrapFrame;
}

CudaQueueLv3Trap::CudaQueueLv3Trap(CUstream stream)
    : CudaQueueLv2(stream, !IsSm120Stream(stream))
    , is_sm120_(instrument_manager_ == nullptr)
{
    if (is_sm120_) {
        Driver::DeviceGetAttribute(&sm_count_, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cudevice_);
        Driver::DeviceGetAttribute(&rf_regs_per_sm_, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, cudevice_);
        sm120_engine_ = sm120::Sm120TrapEngine::Instance(context_);
        sm120_stream_ = std::make_unique<sm120::Sm120StreamContext>(
            context_, cudevice_, sm120_engine_);
    } else {
        interrupt_context_ = InterruptContext::Instance(context_);
    }
}

void CudaQueueLv3Trap::Launch(std::shared_ptr<HwCommand> hw_cmd)
{
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
    if (is_sm120_) {
        if (kernel != nullptr) {
            {
                std::lock_guard<std::mutex> lk(last_kernel_mtx_);
                last_kernel_ = hw_cmd;
            }
            relaunched_ = true;
            const auto mode = SelectSm120Mode(hw_cmd);
            const uint64_t kidx = static_cast<uint64_t>(hw_cmd->GetIdx());
            static const bool trace_launch = GetEnvOption("XSCHED_SM120_TRACE_LAUNCH", false);
            if (trace_launch) {
                const char *fname = nullptr;
                if (Driver::FuncGetName(&fname, kernel->kFuncHandle) != CUDA_SUCCESS) fname = nullptr;
                XINFO("sm120_trace: idx=%llu func=%p name=%.160s ctas=%u thr=%u dyn_smem=%u "
                      "level=%d killable=%d mode=%d",
                      static_cast<unsigned long long>(kidx), static_cast<void *>(kernel->kFuncHandle),
                      fname ? fname : "?", kernel->BlockCnt(), kernel->ThreadCnt(),
                      kernel->DynSmemBytes(), static_cast<int>(level_), kernel->killable ? 1 : 0,
                      static_cast<int>(mode));
            }
            std::lock_guard<std::mutex> lock(sm120::Sm120TrapEngine::LaunchMutex(context_));
            if (level_ >= kPreemptLevelDeactivate && kernel->killable) {
                if (!sm120_stream_->PrepareLaunch(kernel->kFuncHandle, kernel->BlockCnt(),
                                                  kernel->ThreadCnt(), kernel->DynSmemBytes(),
                                                  kidx, mode, kStream, kernel->kFunc)) {
                    // e.g. control / arena allocation failed: run this kernel without the trap
                    // (Interrupt() then waits for it at the Level-1 boundary) instead of aborting.
                    static std::atomic<int> warned_prep{0};
                    if (warned_prep.fetch_add(1) < 5) {
                        XWARN("sm120 PrepareLaunch failed (idx=%llu): launching at the Level-1 boundary",
                              static_cast<unsigned long long>(kidx));
                    }
                    kernel->killable = false;
                    sm120::Sm120StreamContext::ClearDebuggerParams(kernel->kFuncHandle);
                }
            } else {
                sm120::Sm120StreamContext::ClearDebuggerParams(kernel->kFuncHandle);
            }
            if (kernel->route_probe_key != 0) {
                // Time this launch for duration-aware routing (see ShortCtaProbe).
                CUevent beg = nullptr, end = nullptr;
                {
                    std::lock_guard<std::mutex> lk(g_probe_mtx);
                    ShortCtaProbe &p = g_probes[kernel->route_probe_key];
                    if (p.beg == nullptr) Driver::EventCreate(&p.beg, CU_EVENT_DEFAULT);
                    if (p.end == nullptr) Driver::EventCreate(&p.end, CU_EVENT_DEFAULT);
                    beg = p.beg;
                    end = p.end;
                }
                Driver::EventRecord(beg, kStream);
                CUDA_ASSERT(kernel->LaunchWrapper(kStream));
                Driver::EventRecord(end, kStream);
                std::lock_guard<std::mutex> lk(g_probe_mtx);
                g_probes[kernel->route_probe_key].recorded = true;
                return;
            }
            CUDA_ASSERT(kernel->LaunchWrapper(kStream));
            return;
        }
        if (std::dynamic_pointer_cast<CudaGraphCommand>(hw_cmd) != nullptr) {
            std::lock_guard<std::mutex> lk(last_kernel_mtx_);
            last_kernel_ = hw_cmd;
        }
        auto graph = std::dynamic_pointer_cast<CudaGraphLaunchCommand>(hw_cmd);
        if (graph != nullptr && graph->graph_funcs != nullptr && sm120::Sm120StreamContext::AnyParamsSet()) {
            // DEF-32: a whole-graph launch is not trap-managed; its kernel nodes must not run with a
            // Mode 5 window left by an earlier killable launch of the same function (they would be
            // drained / evicted as that launch and never restored). Clear and launch under the
            // LaunchMutex so that no Mode 5 launch can set a window in between.
            std::lock_guard<std::mutex> lock(sm120::Sm120TrapEngine::LaunchMutex(context_));
            for (CUfunction f : *graph->graph_funcs) sm120::Sm120StreamContext::ClearDebuggerParams(f);
            CudaQueueLv1::Launch(hw_cmd);
            return;
        }
        CudaQueueLv1::Launch(hw_cmd);
        return;
    }

    if (kernel != nullptr) {
        {
            std::lock_guard<std::mutex> lk(last_kernel_mtx_);
            last_kernel_ = hw_cmd;
        }
        relaunched_ = true;
        XINFO("trap queue: remembered kernel cmd idx=%ld", (long)hw_cmd->GetIdx());
    }
    CudaQueueLv2::Launch(hw_cmd);
}

void CudaQueueLv3Trap::Synchronize()
{
    // Phase 3.3: the stream is idle right after a synchronization, the safe point to release arenas
    // retired by growth and oversized idle arenas (see Sm120StreamContext::TrimArenas()).
    const uint64_t seq = (is_sm120_ && sm120_stream_) ? sm120_stream_->LaunchSeq() : 0;
    CudaQueueLv1::Synchronize();
    if (is_sm120_ && sm120_stream_) sm120_stream_->TrimArenas(seq);
}

void CudaQueueLv3Trap::Deactivate()
{
    XASSERT(level_ >= kPreemptLevelDeactivate, "Deactivate() not supported on level-%d", level_);
    CtxScope ctx_scope(context_);
    if (is_sm120_) {
        // Suspend() invokes Deactivate() and then Interrupt(). Only trigger Mode 1 early-exit
        // directly when operating strictly at level-2; level-3 triggers once inside Interrupt().
        if (level_ == kPreemptLevelDeactivate) {
            std::shared_ptr<HwCommand> cur;
            {
                std::lock_guard<std::mutex> lk(last_kernel_mtx_);
                cur = last_kernel_;
            }
            auto cur_k = std::dynamic_pointer_cast<CudaKernelCommand>(cur);
            if (cur != nullptr && (cur_k == nullptr || !cur_k->killable)) {
                return;
            }
            last_preempt_mode_ = sm120::Sm120PreemptMode::kEarlyExit;
            sm120_stream_->PreemptStream(last_preempt_mode_);
        }
        return;
    }
    CudaQueueLv2::Deactivate();
}

void CudaQueueLv3Trap::Interrupt()
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Interrupt() not supported on level-%d", level_);
    CtxScope ctx_scope(context_);
    if (is_sm120_) {
        std::shared_ptr<HwCommand> cur;
        {
            std::lock_guard<std::mutex> lk(last_kernel_mtx_);
            cur = last_kernel_;
        }
        auto cur_k = std::dynamic_pointer_cast<CudaKernelCommand>(cur);
        if (cur != nullptr && (cur_k == nullptr || !cur_k->killable)) {
            // Non-trappable command (CUDA Graph / Cooperative / SM120 mbarrier / multi-wave cluster):
            // fall back cleanly to Level-1 boundary synchronization.
            return;
        }
        // DEF-36: no pulse for a queue with nothing on the GPU. The BAR0 pulse is GPU-wide: every
        // running warp of every context traps (and, if not this queue's, returns). The launch worker
        // is already paused here, so an idle stream stays idle until Reactivate(). Under the global
        // scheduler a vLLM process had 128 Level-3 queues (one per PyTorch pool stream), and each
        // high-priority request pulsed once per queue.
        if (cur == nullptr || Driver::StreamQuery(kStream) == CUDA_SUCCESS) {
            return;
        }
        last_preempt_mode_ = SelectSm120Mode(cur);
        sm120_stream_->PreemptStream(last_preempt_mode_);
        return;
    }
    // wait until the preempt flag is set
    XINFO("Interrupt: syncing op stream");
    CUDA_ASSERT(Driver::StreamSynchronize(instrument_manager_->OpStream()));
    XINFO("Interrupt: op stream idle, triggering");
    // FIXME: what if multiple threads call Interrupt()?
    interrupt_context_->Interrupt();
}

void CudaQueueLv3Trap::Restore(const CommandLog &)
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Restore() not supported on level-%d", level_);
    if (is_sm120_) {
        // SM120 v2 restores cleanly in Reactivate(log) via BPT.TRAP 0x1 with zero BAR0 pulses.
        return;
    }
}

// DEF-31 (Phase 3.5): on restore, only commands that the trap / Entry Guardian can have killed are
// relaunched: killable (trap-managed) kernels, and idempotent commands (event records) that must be
// re-issued after them. Non-killable commands - whole CUDA graphs, Level-1-routed kernels, memory
// copies, host functions - are never stopped by the gate or the trap (their Guardian parameters are
// cleared or they are not instrumented), so they always ran to completion; relaunching them executed
// them twice. The launch worker's barrier rule (a non-deactivatable, non-idempotent command waits for
// all in-flight deactivatable commands) keeps them from being queued behind a killed kernel, but the
// no-eviction path below compared with a stale active_kernel_idx (the last trap-managed kernel, which
// had finished) and relaunched e.g. a vLLM decode graph that was the last launched command when the
// suspension arrived (graph without flattening + non-blocking cross-stream waits: wrong tokens).
static bool RelaunchableAfterTrap(const std::shared_ptr<HwCommand> &cmd)
{
    if (auto k = std::dynamic_pointer_cast<CudaKernelCommand>(cmd)) return k->killable;
    return cmd->GetProps(kCommandPropertyIdempotent) != 0;
}

void CudaQueueLv3Trap::Reactivate(const CommandLog &log)
{
    CtxScope ctx_scope(context_);
    if (is_sm120_) {
        this->Synchronize();
        const auto t = sm120_stream_->Telemetry();
        const bool has_preempted =
            (t.evicted_slots > 0 || t.drained_cnt > 0 || t.mode1_exit_cnt > 0);
        static const bool trace_launch = GetEnvOption("XSCHED_SM120_TRACE_LAUNCH", false);
        if (trace_launch) {
            std::string kinds;
            for (const auto &c : log) {
                const char k = std::dynamic_pointer_cast<CudaKernelCommand>(c) != nullptr ? 'K'
                             : std::dynamic_pointer_cast<CudaMemoryCommand>(c) != nullptr ? 'M' : 'O';
                kinds += std::to_string(c->GetIdx()) + k + " ";
                if (kinds.size() > 400) { kinds += "..."; break; }
            }
            XINFO("sm120_trace: reactivate cmd=%u evicted=%u restored=%u drained=%u m1exit=%u ovf=%u "
                  "inflight=%u restore_wave=%u active_idx=%llu preempted=%d arena_bytes=%zu log=[%s]",
                  t.cmd, t.evicted_slots, t.restored_slots, t.drained_cnt, t.mode1_exit_cnt,
                  t.overflow_err, t.trap_inflight,
                  t.is_restore_wave, static_cast<unsigned long long>(t.active_kernel_idx),
                  has_preempted ? 1 : 0, t.arena_bytes, kinds.c_str());
            if (t.evicted_slots > 0) {
                for (const auto &c : log) {
                    if (static_cast<uint64_t>(c->GetIdx()) != t.active_kernel_idx) continue;
                    auto k = std::dynamic_pointer_cast<CudaKernelCommand>(c);
                    if (k == nullptr) break;
                    const auto *rec = sm120::Sm120GuardianRegistry::Instance(context_)->Find(
                        k->kFuncHandle);
                    sm120_stream_->DebugDumpEvicted(rec, k->BlockCnt(), k->ThreadCnt(),
                                                    k->DynSmemBytes(), 4);
                    break;
                }
            }
        }
        if (!has_preempted) {
            sm120_stream_->ResetControlSync();
            if (t.active_kernel_idx != 0) {
                for (const auto &cmd : log) {
                    if (static_cast<uint64_t>(cmd->GetIdx()) > t.active_kernel_idx &&
                        RelaunchableAfterTrap(cmd)) {
                        this->Launch(cmd);
                    }
                }
            }
            return;
        }
        const uint64_t active_idx = sm120_stream_->PrepareRestore(last_preempt_mode_);
        bool cleared_post_active = false;
        for (const auto &cmd : log) {
            if (active_idx != 0 && static_cast<uint64_t>(cmd->GetIdx()) < active_idx) continue;
            if (!RelaunchableAfterTrap(cmd)) continue;
            this->Launch(cmd);
            if (!cleared_post_active &&
                (active_idx == 0 || static_cast<uint64_t>(cmd->GetIdx()) == active_idx)) {
                sm120_stream_->EnqueuePostRestoreClear(kStream);
                cleared_post_active = true;
            }
        }
        if (!cleared_post_active && log.empty()) {
            std::shared_ptr<HwCommand> cur;
            {
                std::lock_guard<std::mutex> lk(last_kernel_mtx_);
                cur = last_kernel_;
            }
            if (cur != nullptr) {
                this->Launch(cur);
                sm120_stream_->EnqueuePostRestoreClear(kStream);
                cleared_post_active = true;
            }
        }
        if (!cleared_post_active) {
            sm120_stream_->ResetControlSync();
        }
        return;
    }

    relaunched_ = false;
    CudaQueueLv2::Reactivate(log);
    std::shared_ptr<HwCommand> cur;
    {
        std::lock_guard<std::mutex> lk(last_kernel_mtx_);
        cur = last_kernel_;
    }
    XINFO("trap reactivate: log=%zu last_kernel=%p relaunched=%d", log.size(),
          (void *)cur.get(), (int)relaunched_);
    if (relaunched_ || cur == nullptr) return;
    this->Launch(cur);
}

void CudaQueueLv3Trap::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    if (is_sm120_) {
        if (level >= kPreemptLevelDeactivate) {
            sm120_engine_->Init();
        }
        level_ = level;
        return;
    }
    if (level == kPreemptLevelInterrupt) {
        instrument_manager_->NotifyTrapInstrumented();
        interrupt_context_->InstrumentTrapHandler();
    }
    level_ = level;
}

void CudaQueueLv3Trap::OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    if (is_sm120_) {
        this->CudaQueueLv1::OnHwCommandSubmit(hw_cmd);
        if (level_ < kPreemptLevelDeactivate) return;

        if (std::dynamic_pointer_cast<CudaGraphCommand>(hw_cmd) != nullptr) {
            CountRoute(kRouteLv1Graph);
            static bool warned_graph = false;
            if (!warned_graph) {
                warned_graph = true;
                XWARN("CUDA graph cannot support trap-based level-3 preemption, "
                      "falling back to level-1");
            }
            return;
        }
        auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
        if (kernel != nullptr) {
            // Adaptive SM120 Preemption Policy Router:
            // Route hardware-incompatible kernels cleanly to Level-1 boundary instead of faulting.

            // DEF-27 (Phase 3.4-2): under the cross-process global scheduler, a high-priority
            // process that keeps launching while this process is being trapped can make the GPU's
            // time-slice context switch arrive while warps are in the trap handler; the switch
            // times out (Xid 109 CTX SWITCH TIMEOUT) and the driver's recovery fails every context.
            // The ready gate (XSCHED_GLB_READY_GATE, default on) prevents it: launches after a
            // Ready transition wait until the server has completed the lower-priority suspensions.
            // Only if the gate is disabled do kernels fall back to the Level-1 boundary under GLB
            // (unless XSCHED_SM120_GLB_LV3=1 forces Level-3).
            static const bool glb_sched = [] {
                const char *v = getenv("XSCHED_SCHEDULER");
                return v != nullptr && std::string(v) == "GLB";
            }();
            static const bool gate_off = [] {
                const char *v = getenv("XSCHED_GLB_READY_GATE");
                return v != nullptr && v[0] == '0';
            }();
            static const bool glb_lv3 = GetEnvOption("XSCHED_SM120_GLB_LV3", false);
            if (glb_sched && gate_off && !glb_lv3) {
                CountRoute(kRouteLv1GlobalSched);
                kernel->killable = false;
                kernel->ClearProps(kCommandPropertyDeactivatable);
                static bool warned_glb = false;
                if (!warned_glb) {
                    warned_glb = true;
                    XWARN("SM120 Level-3 under the global scheduler without the ready gate can stall "
                          "GPU context switches (DEF-27), using the level-1 boundary "
                          "(XSCHED_SM120_GLB_LV3=1 overrides)");
                }
                return;
            }
            if (kernel->IsCooperative()) {
                CountRoute(kRouteLv1Cooperative);
                kernel->killable = false;
                kernel->ClearProps(kCommandPropertyDeactivatable);
                static bool warned_coop = false;
                if (!warned_coop) {
                    warned_coop = true;
                    XWARN("Cooperative kernel requires whole-grid residency, "
                          "routing cleanly to level-1 boundary");
                }
                return;
            }
            // DEF-19: thread block clusters are routed to the Level-1 boundary. Cluster-scope state
            // (cluster barriers, DSMEM mapping, co-scheduling of the cluster's CTAs) is not part of
            // the Mode 5 trap frame; single-wave clusters under Mode 5 returned wrong results for 1-2
            // CTAs in ~10-40% of HW-06 runs on RTX 5080. XSCHED_SM120_CLUSTER_MODE5=1 re-enables the
            // experimental single-wave Mode 5 path (multi-wave clusters always use Level-1).
            static const bool cluster_mode5 = GetEnvOption("XSCHED_SM120_CLUSTER_MODE5", false);
            const unsigned int cluster_sz = kernel->ClusterSize();
            const bool multi_wave = sm_count_ > 0 &&
                                    kernel->BlockCnt() > static_cast<unsigned int>(sm_count_);
            if (cluster_sz > 1 && (multi_wave || !cluster_mode5)) {
                CountRoute(kRouteLv1Cluster);
                kernel->killable = false;
                kernel->ClearProps(kCommandPropertyDeactivatable);
                static bool warned_cluster = false;
                if (!warned_cluster) {
                    warned_cluster = true;
                    XWARN("Thread block cluster (cluster=%u, blocks=%u, sm=%d), "
                          "routing cleanly to level-1 boundary",
                          cluster_sz, kernel->BlockCnt(), sm_count_);
                }
                return;
            }
            // Tier-1 item 4 (driver robustness): the trap engine validates the driver's trap-handler
            // layout (export tables, ToolsMemory size, handler fingerprint, hook-site instructions)
            // and refuses to patch anything it does not recognise. Kernels then take the Level-1
            // boundary instead of aborting the process in PrepareLaunch().
            if (sm120_engine_ == nullptr || (!sm120_engine_->IsReady() && !sm120_engine_->Init())) {
                CountRoute(kRouteLv1NoEngine);
                kernel->killable = false;
                kernel->ClearProps(kCommandPropertyDeactivatable);
                static std::atomic<bool> warned_engine{false};
                if (!warned_engine.exchange(true)) {
                    int drv = 0;
                    Driver::DriverGetVersion(&drv);
                    XWARN("sm120: Level-3 trap engine unavailable with this driver (CUDA driver API %d): "
                          "kernels use the Level-1 boundary (preemption at kernel boundaries only)", drv);
                }
                return;
            }
            const auto *rec = sm120::Sm120GuardianRegistry::Instance(context_)->EnsureInstrumented(
                kernel->kFuncHandle);
            if (rec == nullptr || rec->has_hw_mbarrier) {
                CountRoute(rec == nullptr ? kRouteLv1Uninstrumentable : kRouteLv1Mbarrier);
                kernel->killable = false;
                kernel->ClearProps(kCommandPropertyDeactivatable);
                static bool warned_mbar = false;
                if (!warned_mbar) {
                    warned_mbar = true;
                    XWARN("SM120 hardware mbarrier (SYNCS) kernel detected, "
                          "routing cleanly to level-1 boundary");
                }
                return;
            }
            // DEF-17 (Phase 3.3): a loop-free kernel (no backward / indirect branches, no calls)
            // with many waves of CTAs pays the Entry Guardian's fixed per-CTA cost on every very
            // short CTA (up to +50-100% on RTX 5080), while a Level-1 boundary preempts it within
            // the kernel's own short duration. Bounded by a wave cap so long grids keep Mode 5.
            // XSCHED_SM120_SHORT_CTA_LV1=0 disables; _MIN_WAVES / _MAX_WAVES tune the window.
            static const bool short_cta_lv1 = GetEnvOption("XSCHED_SM120_SHORT_CTA_LV1", true);
            if (short_cta_lv1 && rec->loop_free && sm_count_ > 0) {
                static const int64_t min_waves = []() {
                    int64_t v = 2; GetEnvInt64("XSCHED_SM120_SHORT_CTA_MIN_WAVES", v); return v; }();
                static const int64_t max_waves = []() {
                    int64_t v = 1024; GetEnvInt64("XSCHED_SM120_SHORT_CTA_MAX_WAVES", v); return v; }();
                const int64_t ctas = static_cast<int64_t>(kernel->BlockCnt());
                uint64_t key = reinterpret_cast<uint64_t>(kernel->kFuncHandle);
                key = key * 1000003ULL ^ static_cast<uint64_t>(ctas);
                key = key * 1000003ULL ^ static_cast<uint64_t>(kernel->ThreadCnt());
                key = key * 1000003ULL ^ static_cast<uint64_t>(kernel->DynSmemBytes());
                std::lock_guard<std::mutex> lk(g_probe_mtx);
                ShortCtaProbe &p = g_probes[key];
                if (p.window < 0) {
                    // The occupancy query is a driver call: done once per (function, geometry), not
                    // on every launch (flattened CUDA graphs submit thousands of kernels per second).
                    int blocks_per_sm = 0;
                    if (Driver::OccupancyMaxActiveBlocksPerMultiprocessor(
                            &blocks_per_sm, kernel->kFuncHandle, static_cast<int>(kernel->ThreadCnt()),
                            kernel->DynSmemBytes()) != CUDA_SUCCESS || blocks_per_sm <= 0) {
                        blocks_per_sm = 1;
                    }
                    const int64_t capacity = static_cast<int64_t>(sm_count_) * blocks_per_sm;
                    p.window = (ctas >= min_waves * capacity && ctas <= max_waves * capacity) ? 1 : 0;
                }
                bool route = p.window == 1;
                if (route && ShortCtaMaxUs() > 0) {
                    if (p.state == 0 && p.pending && p.recorded && Driver::EventQuery(p.end) == CUDA_SUCCESS) {
                        float ms = 0.0f;
                        const bool ok = Driver::EventElapsedTime(&ms, p.beg, p.end) == CUDA_SUCCESS;
                        p.state = (ok && ms * 1000.0f <= static_cast<float>(ShortCtaMaxUs())) ? 1 : 2;
                        p.pending = false;
                    }
                    if (p.state != 1) {
                        route = false;
                        if (p.state == 0 && !p.pending) {
                            p.pending = true;
                            p.recorded = false;
                            kernel->route_probe_key = key;
                        }
                    }
                }
                if (route) {
                    CountRoute(kRouteLv1ShortCta);
                    kernel->killable = false;
                    kernel->ClearProps(kCommandPropertyDeactivatable);
                    return;
                }
            }
            // DEF-34: an SM whose register file is allocated to the last register does not take the
            // Level-3 trap (measured: 0 evictions with 0 free registers, normal evictions with 1024
            // free). Detect such geometries (occupancy x warps x per-warp allocation, granularity
            // 256 registers per warp); XSCHED_SM120_RF_FULL = count (default: only counted, see
            // DEF-34 in the ledger) | lv1 (route to the Level-1 boundary).
            if (rf_regs_per_sm_ > 0 && sm_count_ > 0) {
                const int64_t ctas = static_cast<int64_t>(kernel->BlockCnt());
                uint64_t key = reinterpret_cast<uint64_t>(kernel->kFuncHandle) ^ 0x5bd1e995ULL;
                key = key * 1000003ULL ^ static_cast<uint64_t>(ctas);
                key = key * 1000003ULL ^ static_cast<uint64_t>(kernel->ThreadCnt());
                key = key * 1000003ULL ^ static_cast<uint64_t>(kernel->DynSmemBytes());
                bool full = false;
                {
                    std::lock_guard<std::mutex> lk(g_probe_mtx);
                    ShortCtaProbe &p = g_probes[key];
                    if (p.rf_full < 0) {
                        int bps = 0, regs = 0;
                        Driver::OccupancyMaxActiveBlocksPerMultiprocessor(
                            &bps, kernel->kFuncHandle, static_cast<int>(kernel->ThreadCnt()), kernel->DynSmemBytes());
                        Driver::FuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel->kFuncHandle);
                        const int64_t warps = (static_cast<int64_t>(kernel->ThreadCnt()) + 31) / 32;
                        const int64_t per_warp = ((static_cast<int64_t>(regs) * 32 + 255) / 256) * 256;
                        const int64_t resident = std::min<int64_t>(bps, (ctas + sm_count_ - 1) / sm_count_);
                        p.rf_full = (bps > 0 && regs > 0 && resident * warps * per_warp >= rf_regs_per_sm_) ? 1 : 0;
                        static int logged = 0;
                        if (p.rf_full == 1 && logged < 64) {
                            ++logged;
                            const char *fname = nullptr;
                            if (Driver::FuncGetName(&fname, kernel->kFuncHandle) != CUDA_SUCCESS) fname = nullptr;
                            XINFO("sm120 DEF-34: register file full (%d regs x %lld warps x %lld CTAs/SM = %lld of %d): "
                                  "the Level-3 trap is not taken on these SMs: %.120s", regs, (long long)warps,
                                  (long long)resident, (long long)(resident * warps * per_warp), rf_regs_per_sm_,
                                  fname ? fname : "?");
                        }
                    }
                    full = p.rf_full == 1;
                }
                if (full) {
                    CountRoute(kRouteRfFull);
                    static const bool to_lv1 = [] {
                        const char *v = getenv("XSCHED_SM120_RF_FULL");
                        return v != nullptr && std::string(v) == "lv1";
                    }();
                    if (to_lv1) {
                        CountRoute(kRouteLv1RfFull);
                        kernel->killable = false;
                        kernel->ClearProps(kCommandPropertyDeactivatable);
                        return;
                    }
                }
            }
            CountRoute(kRouteLv3Killable);
            kernel->killable = true;
        }
        return;
    }

    this->CudaQueueLv2::OnHwCommandSubmit(hw_cmd);
    if (level_ < kPreemptLevelInterrupt) return;

    if (std::dynamic_pointer_cast<CudaGraphCommand>(hw_cmd) != nullptr) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            XWARN("CUDA graph cannot support trap-based level-3 preemption, "
                  "falling back to level-1");
        }
        return;
    }
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
    if (kernel != nullptr) {
        kernel->killable = true;
        {
            std::lock_guard<std::mutex> lk(last_kernel_mtx_);
            last_kernel_ = hw_cmd;
        }
        XINFO("trap queue: remembered kernel cmd idx=%ld", (long)hw_cmd->GetIdx());
    }
}

CudaQueueLv3Tsg::CudaQueueLv3Tsg(CUstream stream): CudaQueueLv1(stream)
{
    tsg_context_ = TsgContext::Instance(context_);
}

void CudaQueueLv3Tsg::Interrupt()
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Interrupt() not supported on level-%d", level_);
    tsg_context_->Interrupt();
}

void CudaQueueLv3Tsg::Restore(const CommandLog &)
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Restore() not supported on level-%d", level_);
    tsg_context_->Restore();
}

void CudaQueueLv3Tsg::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    level_ = level;
}

CudaQueueLv3Resumable::CudaQueueLv3Resumable(CUstream stream): CudaQueueLv1(stream)
{
    resumable_context_ = ResumableSm120Context::Instance(context_);
}

void CudaQueueLv3Resumable::Interrupt()
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Interrupt() not supported on level-%d", level_);
    resumable_context_->Interrupt();
}

void CudaQueueLv3Resumable::Restore(const CommandLog &)
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Restore() not supported on level-%d", level_);
    resumable_context_->Restore();
}

void CudaQueueLv3Resumable::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    level_ = level;
}
