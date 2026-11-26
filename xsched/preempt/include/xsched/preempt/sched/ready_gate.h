#pragma once

#include <cstdint>

namespace xsched::preempt
{

/// @brief DEF-27 ready gate for the cross-process global scheduler.
///
/// With XSCHED_SCHEDULER=GLB (and unless XSCHED_GLB_READY_GATE=0), the first command after a queue turns
/// Ready is not launched until the server acknowledges that Ready event. The server (xserver, same
/// default and variable) sends the operations that suspend lower-priority queues first, waits for
/// them to complete, and only then sends this process the operation carrying the acknowledgement.
/// So a high-priority process never puts work on the GPU while a lower-priority process is still
/// being trapped (a time-slice context switch arriving during the trap times out: Xid 109).
class ReadyGate
{
public:
    /// @brief Whether launches wait for the acknowledgement (GLB scheduler and the env variable).
    static bool Enabled();
    /// @brief Count a Ready event this process is about to send; returns its sequence number.
    /// Must be called for every XQueueReadyEvent so the count matches the server's.
    static uint64_t NoteReadySent();
    /// @brief The scheduler acknowledged this many Ready events of this process.
    static void Ack(uint64_t ready_ack);
    /// @brief Wait (bounded by XSCHED_GLB_READY_GATE_TIMEOUT_US, default 10000) until Ready event
    /// seq is acknowledged. Returns false on timeout.
    static bool Wait(uint64_t seq);
};

} // namespace xsched::preempt
