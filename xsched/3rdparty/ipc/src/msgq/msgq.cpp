#include <cstddef>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ipc/msgq/msgq.h"
#include "utils/assert.h"
#include "utils/common.h"
#include "utils/log.h"

#ifdef _WIN32

#include "boost/interprocess/ipc/message_queue.hpp"

using namespace boost::interprocess;

namespace msgq {

MessageQueue::MessageQueue(std::string name, NodeType ntype)
    : msgq_name_(name)
    , node_type_(ntype)
{
    switch (ntype) {
    case NodeType::kReceiver:
        message_queue::remove(msgq_name_.c_str());
        try {
            message_queue_ = std::make_unique<message_queue>(
                create_only,
                msgq_name_.c_str(),
                max_msg_count_, // Max number of messages
                max_msg_size_   // Max message size
            );
        } catch (const interprocess_exception& e) {
            XERRO("Receiver %s create failed: %s", msgq_name_.c_str(), e.what());
        }
        XDEBG("kReceiver (MessageQueue) '%s' created successfully", msgq_name_.c_str());
        break;
    case NodeType::kSender:
        // Move connection establishment to Send method
        // Prevent errors caused by not creating a receiver during initialization
        break;
    default:
        XERRO("Unknown NodeType %d for node %s", static_cast<int>(ntype), msgq_name_.c_str());
        break;
    }
}

MessageQueue::~MessageQueue()
{
    MessageQueue::Remove();
}

bool MessageQueue::Send(const void* data, size_t data_size)
{
    if (!message_queue_)
        try {
            message_queue_ = std::make_unique<message_queue>(
                open_only,
                msgq_name_.c_str());
            XDEBG("kSender (MessageQueue) '%s' opened successfully", msgq_name_.c_str());
        } catch (const interprocess_exception& e) {
            XERRO("Sender %s create failed: %s", msgq_name_.c_str(), e.what());
        }

    XASSERT_RET(node_type_ != NodeType::kReceiver, false, "kReceiver can't send data");
    XASSERT_RET(data, false, "Data is null");
    XASSERT_RET(message_queue_, false, "Message queue is not initialized");

    try {
        message_queue_->send(data, data_size, 0);
        XDEBG("kSender '%s' sent %zu bytes", msgq_name_.c_str(), data_size);
        return true;
    } catch (const interprocess_exception& e) {
        XWARN("Sender %s Send failed: %s", msgq_name_.c_str(), e.what());
        return false;
    }
}

std::shared_ptr<Buffer> MessageQueue::Receive()
{
    XASSERT_RET(message_queue_, nullptr, "Message queue is not initialized");

    try {
        size_t received_size;
        unsigned int priority;
        auto buffer = std::make_shared<Buffer>(malloc(max_msg_size_), max_msg_size_);

        XDEBG("Receiver '%s' waiting for data...", msgq_name_.c_str());
        message_queue_->receive(buffer->Data(), buffer->Size(), received_size, priority);
        XDEBG("Receiver '%s' received %zu bytes", msgq_name_.c_str(), received_size);
        buffer->SetSize(received_size);

        return buffer;

    } catch (const interprocess_exception& e) {
        XWARN("Receiver %s Receive failed: %s", msgq_name_.c_str(), e.what());
        return nullptr;
    }
}

bool MessageQueue::Remove()
{
    try {
        if (node_type_ == NodeType::kReceiver) {
            XDEBG("Removing message queue '%s'", msgq_name_.c_str());
            return message_queue::remove(msgq_name_.c_str());
        } else
            return true;
    } catch (const interprocess_exception& e) {
        XWARN("Receiver %s Remove failed: %s", msgq_name_.c_str(), e.what())
        return false;
    }
}

bool MessageQueue::Remove(const std::string& name)
{
    try {
        return message_queue::remove(name.c_str());
    } catch (const interprocess_exception& e) {
        XWARN("fail to remove channel \"%s\": %s", name.c_str(), e.what())
        return false;
    }
}

} // namespace msgq

#else

#include <sys/ipc.h>
#include <sys/types.h>
#include <unistd.h>

namespace msgq {

MessageQueue::MessageQueue(std::string name, NodeType ntype, key_t key)
    : msgq_name_(name), node_type_(ntype), key_(key)
{
    switch (ntype) {
    case NodeType::kReceiver:
        // IPC_CREAT: create the message queue if it does not exist
        // IPC_EXCL: fail if key exists, must be used together with IPC_CREAT
        // Octal number determines the access permissions to the message queue
        // 6 (110 binary) represents read and write permissions (4 read + 2 write)
        // 0666 indicates that all users (owners, groups, and others) can read and write to this message queue
        msgid_ = msgget(key, IPC_EXCL | IPC_CREAT | 0666);
        if (msgid_ == -1) {
            // -1 indicates that msgq already exists
            // But only one receiver can exist for a message queue
            // Delete the existing message queue
            XWARN("MessageQueue %s (key: 0x%x) already exists, deleting", msgq_name_.c_str(), key);
            msgid_ = msgget(key, 0666);
            XASSERT(msgid_ >= 0, "failed to get existing MessageQueue (key: 0x%x)", key);
            // IPC_RMID: Remove the message queue
            // If the message queue is not empty, it will be deleted after all messages are read
            // If the message queue is empty, it will be deleted immediately
            XASSERT(msgctl(msgid_, IPC_RMID, nullptr) >= 0,
                    "failed to delete MessageQueue (key: 0x%x)", key);
            msgid_ = msgget(key, IPC_EXCL | IPC_CREAT | 0666);
            XASSERT(msgid_ >= 0, "failed to create MessageQueue (key: 0x%x)", key);
        }

        struct msqid_ds queue_info;
        XASSERT(msgctl(msgid_, IPC_STAT, &queue_info) >= 0,
                "failed to get MessageQueue info (key: 0x%x)", key);
        max_msg_size_ = queue_info.msg_qbytes;
        XDEBG("Receiver of MessageQueue \"%s\" (key: 0x%x) created with id %d",
              msgq_name_.c_str(), key, msgid_);
        break;
    case NodeType::kSender:
        // Move connection establishment to Send method
        // Prevent errors caused by not creating a receiver during initialization
        break;
    default:
        XERRO("Unknown NodeType %d for MessageQueue \"%s\" (key: 0x%x)",
              static_cast<int>(ntype), msgq_name_.c_str(), key);
    }
}

MessageQueue::~MessageQueue()
{
    MessageQueue::Remove();
}

bool MessageQueue::Send(const void* data, size_t data_size)
{
    if (UNLIKELY(msgid_ == -1)) {
        msgid_ = msgget(key_, 0666);
        XASSERT_RET(msgid_ >= 0, false, "MessageQueue %s (key: 0x%d) does not exist",
                    msgq_name_.c_str(), key_);
        XDEBG("Sender of MessageQueue \"%s\" (key: 0x%x) created with id %d",
              msgq_name_.c_str(), key_, msgid_);

        // Get the maximum message size for this queue
        struct msqid_ds queue_info;
        XASSERT_RET(msgctl(msgid_, IPC_STAT, &queue_info) >= 0, false,
                    "failed to get MessageQueue info (key: 0x%x)", key_);
        max_msg_size_ = queue_info.msg_qbytes;
    }

    XASSERT_RET(data, false, "empty data");
    size_t total_size = sizeof(Message) + data_size;
    XASSERT_RET(total_size <= max_msg_size_, false,
                "message size (%zu) exceeds maximum (%zu)",
                data_size, max_msg_size_);

    Message *message = static_cast<Message *>(malloc(total_size));
    XASSERT_RET(message, false, "fail to malloc message");

    message->mtype = MESSAGE_TYPE;
    message->size = data_size;
    memcpy(message->data, data, data_size);

    int ret = msgsnd(msgid_, message, total_size, 0);
    free(message);

    // Fail reasons:
    // 1. kReceiver restart makes the msgid_ invalid
    XASSERT_RET(ret >= 0, false,
                "failed to send message size %zu to MessageQueue \"%s\" (key: 0x%x)",
                data_size, msgq_name_.c_str(), key_);
    XDEBG("Sent %zu bytes to MessageQueue \"%s\" (key: 0x%x)",
          data_size, msgq_name_.c_str(), key_);
    return true;
}


// std::shared_ptr<Buffer> MessageQueue::Receive()
// {
//     std::unique_ptr<char[]> buffer(new char[max_msg_size_]);
//     XASSERT_RET(buffer, nullptr, "malloc fail");

//     XDEBG("Receiver '%s' waiting for data...", msgq_name_.c_str());
//     ssize_t received = msgrcv(msgid_, buffer.get(), max_msg_size_, 0, 0);
//     XASSERT_RET(received != -1, nullptr, "msgrcv fail");
//     XDEBG("Receiver '%s' received %ld bytes", msgq_name_.c_str(), received);

//     Message* message = reinterpret_cast<Message*>(buffer.get());
//     XASSERT_RET(static_cast<size_t>(received) == sizeof(Message) + message->size, nullptr,
//         "Received size %ld does not match expected size %zu", received, sizeof(Message) + message->size);

//     auto result = std::make_shared<Buffer>(malloc(message->size), message->size);
//     XASSERT_RET(result, nullptr, "malloc fail");
//     memcpy(result->Data(), message->data, message->size);
//     return result;
// }

std::shared_ptr<Buffer> MessageQueue::Receive()
{
    void *buf = malloc(max_msg_size_);
    XASSERT_RET(buf, nullptr, "fail to malloc buffer");

    XDEBG("Waiting for data from MessageQueue \"%s\" (key: 0x%x)",
          msgq_name_.c_str(), key_);
    ssize_t received = msgrcv(msgid_, buf, max_msg_size_, 0, 0);
    if (UNLIKELY(received < 0)) {
        XWARN("Fail to receive from MessageQueue \"%s\" (key: 0x%x)", msgq_name_.c_str(), key_);
        free(buf);
        return nullptr;
    }
    XDEBG("Received %ld bytes from MessageQueue \"%s\" (key: 0x%x)",
          received, msgq_name_.c_str(), key_);

    Message *message = reinterpret_cast<Message *>(buf);
    const size_t data_size = message->size;
    if (UNLIKELY((size_t)received != sizeof(Message) + data_size)) {
        XWARN("Received size %ld does not match expected size %zu",
              received, sizeof(Message) + data_size);
        free(buf);
        return nullptr;
    }

    void *src_data = message->data;
    memmove(buf, src_data, data_size);
    void *shrinked_buf = realloc(buf, data_size);
    if (LIKELY(shrinked_buf)) buf = shrinked_buf;

    return std::make_shared<Buffer>(buf, data_size);
}

bool MessageQueue::Remove()
{
    if (node_type_ == NodeType::kSender) return true;

    XDEBG("Removing MessageQueue \"%s\" (key: 0x%x)", msgq_name_.c_str(), key_);
    // The destructors of Node and msgq will call Remove() multiple times
    // msgctl will return -1 and set errno to EINVAL if Remove repeatedly
    int ret = msgctl(msgid_, IPC_RMID, nullptr);
    if (LIKELY(ret >= 0) || errno == EINVAL) return true;
    XWARN("Failed to remove MessageQueue \"%s\" (key: 0x%x)", msgq_name_.c_str(), key_);
    return false;
}

bool MessageQueue::Remove(key_t key)
{
    int msgid = msgget(key, 0666);
    if (msgid == -1) return false;
    int ret = msgctl(msgid, IPC_RMID, nullptr);
    if (LIKELY(ret >= 0) || errno == EINVAL) return true;
    XWARN("Failed to remove MessageQueue (key: 0x%x)", key);
    return false;
}

} // namespace msgq

#endif // _WIN32
