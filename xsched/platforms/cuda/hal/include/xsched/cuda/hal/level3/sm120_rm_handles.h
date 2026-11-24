#pragma once

#include <cstdint>

namespace xsched::cuda
{

/// @brief Captured NVIDIA Resource Manager (RM) ioctl handles for a CUDA context.
/// Isolated from TrapSm120/v2 so Context-level CILP (ResumableSm120Context) can
/// remain a standalone fallback module without depending on Mode 5 internals.
struct Sm120RmCtxHandles
{
    int      ctl_fd;
    uint32_t hClient;
    uint32_t hDevice;
    uint32_t hSubDevice;
    uint32_t hTsg;
    uint32_t hChannel;
    uint32_t hCompute;
    uint32_t hDebugger;
    uint32_t hDebuggerClient;
    uint32_t hShareClient;
};

class Sm120RmHandleStore
{
public:
    static bool GetLatest(Sm120RmCtxHandles *out);
    static int  GetCount();
    static bool GetByIndex(int idx, Sm120RmCtxHandles *out);
    static void SetPinDebuggerHandle(bool pin);
};

} // namespace xsched::cuda
