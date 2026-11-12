#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include "ipc/ipc.h"

const int NUM_ITERATIONS = 1000;
const int WARMUP_ITERATIONS = 100;
const int REQUEST_CHANNEL_KEY = 1;
const int RESPONSE_CHANNEL_KEY = 2;

int main()
{
    // Create separate channels for receiving and sending
    ipc::Node receiver(REQUEST_CHANNEL_KEY, ipc::NodeType::kReceiver);
    ipc::Node sender(RESPONSE_CHANNEL_KEY, ipc::NodeType::kSender);

    std::cout << "IPC echo server is running" << std::endl;
    std::cout << "Receiving on channel: ipc-latency-request" << std::endl;
    std::cout << "Sending on channel: ipc-latency-response" << std::endl;

    int cnt = 0; 
    while (cnt++ < (WARMUP_ITERATIONS + NUM_ITERATIONS)) {
        // Wait for incoming message
        auto request = receiver.Receive();

        if (!request) {
            std::cout << "Error receiving message" << std::endl;
            break;
        }
        // Process the received message
        char* msg = static_cast<char*>(request.get()->Data());
        // Echo the message back with the same content
        sender.Send(msg, strlen(msg) + 1);
    }

    return 0;
}
