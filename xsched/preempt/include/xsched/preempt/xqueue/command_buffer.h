#pragma once

#include <list>
#include <mutex>
#include <atomic>
#include <memory>
#include <condition_variable>

#include "xsched/types.h"
#include "xsched/preempt/xqueue/xcommand.h"

namespace xsched::preempt
{

class CommandBuffer
{
public:
    CommandBuffer(XQueueHandle xq_h);
    ~CommandBuffer() = default;

    void WaitForNextIdle();
    XQueueState GetXQueueState();
    void ForEachCommand(std::function<bool (std::shared_ptr<XCommand>)> func);

    /// @brief Dequeue an XCommand from the CommandBuffer.
    /// @return Pointer to the dequeued XCommand.
    std::shared_ptr<XCommand> Dequeue();

    /// @brief Enqueue an XCommand to the CommandBuffer.
    /// @param xcmd Pointer to the enqueued XCommand.
    void Enqueue(std::shared_ptr<XCommand> xcmd);

    void DropAll();
    std::shared_ptr<XQueueWaitAllCommand> EnqueueXQueueWaitAllCommand();

    /// @brief Direct submit gate. Succeeds only if no command is queued and the worker is not
    /// holding a dequeued hardware / host-function command that it has not finished processing,
    /// i.e., every previously submitted command has already been launched. While the gate is
    /// held the worker does not dequeue. On success the XQueue is marked ready (as Enqueue()).
    bool TryBeginDirect(std::shared_ptr<XCommand> xcmd, bool allow_awake = false);

    /// @brief Release the direct submit gate. If the command was launched (launched == true) it
    /// becomes the last enqueued command and the worker is woken so it keeps tracking completion
    /// (batch synchronization / idle detection) as for worker-launched commands. Otherwise the
    /// command is appended to the buffer (still in order, the gate excluded other enqueues of
    /// this submitter and the worker).
    void EndDirect(std::shared_ptr<XCommand> xcmd, bool launched);

    /// @brief Called by the worker when it has finished processing the last dequeued command.
    void WorkerDone();

    /// @brief True if no hardware / host-function command is queued or held by the worker, i.e.,
    /// everything submitted so far has been handed to the hardware queue (or dropped).
    bool AllOrderedLaunched();
    /// @brief Lock-free hint for AllOrderedLaunched() (used to spin before taking the lock).
    bool MaybeAllOrderedLaunched() const
    {
        return pending_.load(std::memory_order_acquire) == 0 &&
               !worker_holds_ordered_.load(std::memory_order_acquire);
    }

private:
    const XQueueHandle kXQueueHandle;

    std::mutex mtx_;
    std::condition_variable cv_;

    XQueueState xq_state_ = kQueueStateIdle;
    std::shared_ptr<XCommand> last_enqueued_cmd_ = nullptr;
    std::shared_ptr<XCommand> last_dequeued_cmd_ = nullptr;
    std::list<std::shared_ptr<XCommand>> cmds_;
    // Number of queued commands, readable without the lock (bounded spin in Dequeue()).
    std::atomic<size_t> pending_{0};
    // Direct submit state (protected by mtx_).
    std::atomic<bool> worker_holds_ordered_{false}; // worker holds a dequeued hw/host-func command
    bool direct_gate_ = false;          // a direct launch is in progress
    std::atomic<uint64_t> direct_seq_{0}; // number of completed direct launches
    bool worker_asleep_ = false;        // worker is blocked on cv_ waiting for work (idle)

    uint64_t idle_cnt_ = 0;
    std::condition_variable idle_cv_;
};

} // namespace xsched::preempt
