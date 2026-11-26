#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "xsched/utils/log.h"
#include "xsched/protocol/def.h"
#include "xsched/preempt/sched/ready_gate.h"

namespace xsched::preempt
{

namespace
{
std::atomic<uint64_t> g_sent{0};
uint64_t g_acked = 0;
std::mutex g_mtx;
std::condition_variable g_cv;

bool EnvOn(const char *name)
{
    const char *v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

/// Optional statistics (XSCHED_GLB_READY_GATE_TRACE=1), printed at process exit.
struct GateStats
{
    std::atomic<uint64_t> gated{0}, waited{0}, timeouts{0}, wait_ns_sum{0}, wait_ns_max{0};
    ~GateStats()
    {
        if (gated.load() == 0 || !EnvOn("XSCHED_GLB_READY_GATE_TRACE")) return;
        const uint64_t w = waited.load();
        XINFO("ready gate stats: %llu gated launches, %llu waited (mean %.1f us, max %.1f us), "
              "%llu timeouts", (unsigned long long)gated.load(), (unsigned long long)w,
              w ? wait_ns_sum.load() / 1e3 / w : 0.0, wait_ns_max.load() / 1e3,
              (unsigned long long)timeouts.load());
    }
} g_stats;
} // namespace

bool ReadyGate::Enabled()
{
    static const bool on = [] {
        const char *s = std::getenv(XSCHED_SCHEDULER_ENV_NAME);
        // Default on under the global scheduler; XSCHED_GLB_READY_GATE=0 disables.
        const char *g = std::getenv("XSCHED_GLB_READY_GATE");
        const bool gate = g == nullptr || g[0] == '\0' || g[0] != '0';
        return s != nullptr && std::strcmp(s, XSCHED_SCHEDULER_NAME_GLB) == 0 && gate;
    }();
    return on;
}

uint64_t ReadyGate::NoteReadySent()
{
    return g_sent.fetch_add(1, std::memory_order_acq_rel) + 1;
}

void ReadyGate::Ack(uint64_t ready_ack)
{
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        if (ready_ack <= g_acked) return;
        g_acked = ready_ack;
    }
    g_cv.notify_all();
}

bool ReadyGate::Wait(uint64_t seq)
{
    static const auto timeout = std::chrono::microseconds([] {
        const char *v = std::getenv("XSCHED_GLB_READY_GATE_TIMEOUT_US");
        return v ? std::max(1L, std::atol(v)) : 10000L;
    }());
    g_stats.gated.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock<std::mutex> lock(g_mtx);
    if (g_acked >= seq) return true;
    g_stats.waited.fetch_add(1, std::memory_order_relaxed);
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = g_cv.wait_for(lock, timeout, [seq] { return g_acked >= seq; });
    const uint64_t ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count());
    g_stats.wait_ns_sum.fetch_add(ns, std::memory_order_relaxed);
    uint64_t prev = g_stats.wait_ns_max.load(std::memory_order_relaxed);
    while (ns > prev && !g_stats.wait_ns_max.compare_exchange_weak(prev, ns)) {}
    if (ok) return true;
    const uint64_t n = g_stats.timeouts.fetch_add(1) + 1;
    if ((n & (n - 1)) == 0) {   // 1, 2, 4, 8, ... : bounded log volume
        XWARN("ready gate: no acknowledgement of Ready event %llu within %lld us (%llu timeouts); "
              "launching anyway", static_cast<unsigned long long>(seq),
              static_cast<long long>(timeout.count()), static_cast<unsigned long long>(n));
    }
    return false;
}

} // namespace xsched::preempt
