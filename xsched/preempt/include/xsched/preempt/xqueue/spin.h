#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>

namespace xsched::preempt
{

/// @brief Bounded spin budget (microseconds) used before blocking on a condition variable in the
/// XQueue submit/launch/wait paths. A futex wake-up costs several microseconds per hand-off; a
/// short bounded spin removes it for back-to-back submissions and fast completions while idle
/// queues still go to sleep. Configure with XSCHED_SPIN_US (0 = never spin). Default: 30 us.
inline int64_t SpinBudgetUs()
{
    static const int64_t budget = []() -> int64_t {
        const char *env = std::getenv("XSCHED_SPIN_US");
        if (env == nullptr || env[0] == '\0') return 30;
        const long long v = std::atoll(env);
        return v < 0 ? 0 : (v > 100000 ? 100000 : v);
    }();
    return budget;
}

/// @brief Read a boolean XSched option ("0"/"1"), with a default when unset.
inline bool XQueueEnvFlag(const char *name, bool def)
{
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return def;
    return env[0] != '0';
}

/// @brief Direct submit (Phase 3.3): when nothing is queued ahead of a kernel and the launch worker
/// is idle and running, the submitting thread launches it itself instead of handing it to the
/// worker (removes one thread hand-off per call). XSCHED_XQUEUE_DIRECT=0 disables.
inline bool DirectSubmitEnabled()
{
    static const bool on = XQueueEnvFlag("XSCHED_XQUEUE_DIRECT", true);
    return on;
}

/// @brief Caller-side WaitAll (Phase 3.3): once every submitted command has been launched, the
/// waiting thread synchronizes the hardware queue itself instead of waiting for the worker to do
/// it and wake it up; falls back to the worker path if a suspension happened meanwhile.
/// XSCHED_XQUEUE_CALLER_SYNC=0 disables.
inline bool CallerSyncEnabled()
{
    static const bool on = XQueueEnvFlag("XSCHED_XQUEUE_CALLER_SYNC", true);
    return on;
}

inline void CpuRelax()
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

/// @brief Spin until pred() is true or the budget expires. Returns pred()'s final value.
template <typename Pred>
inline bool SpinUntil(Pred &&pred)
{
    const int64_t budget = SpinBudgetUs();
    if (budget <= 0) return pred();
    if (pred()) return true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(budget);
    uint32_t iter = 0;
    while (!pred()) {
        CpuRelax();
        if ((++iter & 63u) == 0 && std::chrono::steady_clock::now() >= deadline) return pred();
    }
    return true;
}

} // namespace xsched::preempt
