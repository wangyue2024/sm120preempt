#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ipc/ipc.h"
#include "ipc/msgq/msgq.h"
#include "ipc/pipe/pipe.h"
#include "utils/assert.h"
#include "utils/log.h"

#ifndef _WIN32
#include <sys/ipc.h> // For IPC_PRIVATE
#endif

namespace ipc {

bool Channel::Remove(ChannelKey chn_key, ChannelType chn_type)
{
#ifdef _WIN32
    const std::string chn_name = "CPPIPC-" + std::to_string(chn_key);
    switch (chn_type) {
    case ChannelType::kMessageQueue:
        return msgq::MessageQueue::Remove(chn_name);
    case ChannelType::kNamedPipe:
        XWARN("removing named pipe channel is not supported on Windows.");
        return false;
    default:
        return msgq::MessageQueue::Remove(chn_name);
    }
#else
    key_t key = (key_t)chn_key;
    switch (chn_type) {
    case ChannelType::kMessageQueue:
        return msgq::MessageQueue::Remove(key);
    case ChannelType::kNamedPipe:
        XWARN("Named pipe channel is not supported on Linux.");
        return false;
    default:
        return msgq::MessageQueue::Remove(key);
    }
#endif
}

Node::Node(ChannelKey chn_key, NodeType node_type, ChannelType chn_type)
    : node_type_(node_type)
    , channel_key_(chn_key)
    , channel_name_("CPPIPC-" + std::to_string(chn_key))
{
    InitChannel(chn_type);
}

Node::~Node()
{
    this->Remove();
}

bool Node::Send(void const* data, size_t data_size)
{
    XASSERT_RET(channel_, false, "Channel not initialized");
    XASSERT_RET(node_type_ == NodeType::kSender, false, "Cannot Send data from a Receiver Node");

    return channel_->Send(data, data_size);
}

std::shared_ptr<Buffer> Node::Receive()
{
    XASSERT_RET(channel_, nullptr, "Channel not initialized");
    XASSERT_RET(node_type_ == NodeType::kReceiver, nullptr, "Cannot Receive data from a kSender Node");

    return channel_->Receive();
}

bool Node::Remove()
{
    if (channel_) {
        bool result = channel_->Remove();
        channel_.reset(); // Reset the shared pointer to release the channel
        return result;
    }
    return true; // No channel to disconnect
}

void Node::InitChannel(ChannelType chn_type)
{
#ifdef _WIN32
    switch (chn_type) {
    case ChannelType::kMessageQueue:
        channel_ = std::make_shared<msgq::MessageQueue>(channel_name_, node_type_);
        break;
    case ChannelType::kNamedPipe:
        channel_ = std::make_shared<pipe::NamedPipe>(channel_name_, node_type_);
        break;
    default:
        channel_ = std::make_shared<msgq::MessageQueue>(channel_name_, node_type_);
    }
#else
    key_t chn_key = (key_t)channel_key_;
    XASSERT(chn_key != IPC_PRIVATE, "channel key should not be IPC_PRIVATE(%d)", IPC_PRIVATE);
    switch (chn_type) {
    case ChannelType::kMessageQueue:
        channel_ = std::make_shared<msgq::MessageQueue>(channel_name_, node_type_, chn_key);
        break;
    case ChannelType::kNamedPipe:
        XERRO("Named pipe channel is not supported on Linux.");
    default:
        channel_ = std::make_shared<msgq::MessageQueue>(channel_name_, node_type_, chn_key);
    }
#endif
}

} // namespace ipc