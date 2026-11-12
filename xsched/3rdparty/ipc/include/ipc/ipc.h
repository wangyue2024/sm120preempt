#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ipc {

enum class NodeType
{
    kSender,
    kReceiver
};

typedef uint32_t ChannelKey;
enum class ChannelType
{
    kDefault,
    kMessageQueue,
    kNamedPipe
};

class Buffer
{

public:
    Buffer(void* data, size_t size): data_(data), size_(size) {}
    ~Buffer() { free(data_); }

    size_t Size() const { return size_; }
    void SetSize(size_t size) { size_ = size; }
    void *Data() { return data_; }

private:
    void *data_ = nullptr;
    size_t size_ = 0;
};

class Channel
{
public:
    Channel() = default;
    virtual ~Channel() = default;

    // Disable copy constructor and assignment operator
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    virtual bool Send(const void* data, size_t data_size) = 0;
    virtual std::shared_ptr<Buffer> Receive() = 0;
    virtual bool Remove() = 0;

    // Remove the channel by key
    static bool Remove(ChannelKey chn_key, ChannelType chn_type = ChannelType::kDefault);
};

class Node {
public:
    Node(ChannelKey chn_key, NodeType node_type, ChannelType chn_type = ChannelType::kDefault);
    ~Node();

    // Disable copy constructor and assignment operator
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    bool Send(const void* data, size_t data_size);
    std::shared_ptr<Buffer> Receive();
    bool Remove();

private:
    void InitChannel(ChannelType chn_type);

    std::shared_ptr<Channel> channel_; // Pointer to the underlying IPC channel
    const NodeType node_type_;         // Type of the Node (kSender or kReceiver)
    const ChannelKey channel_key_;     // Key of the IPC Channel
    const std::string channel_name_;   // Name of the IPC Channel
};

} // namespace ipc
