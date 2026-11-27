#include <atomic>
#include <cstdlib>
#include <mutex>
#include "xsched/protocol/def.h"
#include "xsched/utils/xassert.h"
#include "xsched/sched/scheduler/global.h"

using namespace xsched::sched;

// DEF-42 (Phase 3.5): once the process has started to exit, the worker drops operations from the
// server. They used to be executed during static destruction, after the CUDA runtime / driver had been
// torn down: a Level-3 Resume -> Reactivate -> cuStreamSynchronize failed with
// CUDA_ERROR_DEINITIALIZED, CUDA_ASSERT called exit() from the worker while the main thread was
// already inside exit() and joining that worker (deadlock), or the process crashed (SIGSEGV / SIGABRT).
// The atexit handler is registered when the worker starts, i.e. after the CUDA runtime registered its
// own exit handlers, so it runs before them.
static std::atomic<bool> g_process_exiting{false};

GlobalScheduler::GlobalScheduler(): Scheduler(kSchedulerGlobal)
{

}

GlobalScheduler::~GlobalScheduler()
{
    this->Stop();
}

void GlobalScheduler::Run()
{
    client_chn_key_ = (ipc::ChannelKey)GetProcessId();
    ipc::ChannelKey server_chn_key = (ipc::ChannelKey)XSCHED_SERVER_CHANNEL_KEY;
    recv_chan_ = std::make_unique<ipc::Node>(client_chn_key_, ipc::NodeType::kReceiver);
    send_chan_ = std::make_unique<ipc::Node>(server_chn_key, ipc::NodeType::kSender);
    static std::once_flag once;
    std::call_once(once, [] { atexit([] { g_process_exiting.store(true, std::memory_order_release); }); });
    thread_ = std::make_unique<std::thread>(&GlobalScheduler::Worker, this);
}

void GlobalScheduler::Stop()
{
    if (thread_) {
        auto op = std::make_unique<TerminateOperation>();
        auto self_chan = std::make_unique<ipc::Node>(client_chn_key_, ipc::NodeType::kSender);
        XASSERT(self_chan->Send(op->Data(), op->Size()),
                "cannot send TerminateOperation to worker thread");
        self_chan->Remove();
        thread_->join();
        thread_ = nullptr;
    }

    if (recv_chan_) {
        recv_chan_->Remove();
        recv_chan_ = nullptr;
    }

    if (send_chan_) {
        send_chan_->Remove();
        send_chan_ = nullptr;
    }
}

void GlobalScheduler::RecvEvent(std::shared_ptr<const Event> event)
{
    bool sent = send_chan_->Send(event->Data(), event->Size());
    if (LIKELY(sent)) return;
    this->Stop();
    XASSERT(false, "failed to send event to server");
}

void GlobalScheduler::Worker()
{
    while (true) {
        auto data = recv_chan_->Receive();
        if (data == nullptr) {
            XDEBG("recv_chan_ receive failed, exiting Worker thread");
            return;
        }
        auto op = Operation::CopyConstructor(data->Data());
        if (UNLIKELY(op->Type() == kOperationTerminate)) return;
        if (UNLIKELY(g_process_exiting.load(std::memory_order_acquire))) continue;   // DEF-42
        Execute(op);
    }
}
