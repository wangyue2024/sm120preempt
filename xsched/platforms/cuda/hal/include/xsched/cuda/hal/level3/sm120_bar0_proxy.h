#pragma once

// DEF-10: non-root Level-3 trap pulses through a narrow privileged proxy (xsched-bar0d).
//
// The only BAR0 access Level-3 needs is the evict pulse below. Root processes keep mapping
// BAR0 directly; other processes connect to the proxy's Unix socket. The proxy maps only the
// two 4 KiB pages holding the trap register and the flush register, drops all privileges, and
// accepts a single operation (PULSE). It never reads or writes any other register, so access
// to its socket (group-restricted by the socket file mode) grants the ability to preempt, not
// general GPU register access.

#include <cstdint>

namespace xsched::cuda::sm120
{

inline constexpr uint32_t kBar0PtimerLoOff = 0x00009400u; // NV_PTIMER_TIME_0 (non-PRI flush read)
inline constexpr uint32_t kBar0TrapRegOff  = 0x00424684u; // trap STOP_TRIGGER register
inline constexpr uint32_t kBar0PulseOn     = 0x80001c01u;
inline constexpr uint32_t kBar0PulseOff    = 0x00001c01u;

/// @brief The evict pulse. Hardware rule (D10): never read the PGRAPH broadcast PRI register
/// while STOP_TRIGGER (bit 31) is asserted; flush the posted write with a non-PRI read of
/// NV_PTIMER_TIME_0 so the pulse window is clean (~1 us) before deasserting.
inline void FireBar0Pulse(volatile uint32_t *trap_reg, volatile uint32_t *ptimer_lo)
{
    *trap_reg = kBar0PulseOn;
    (void)*ptimer_lo;
    *trap_reg = kBar0PulseOff;
}

namespace bar0proxy
{

inline constexpr uint32_t kMagic = 0x50304258u; // "XB0P"
inline constexpr uint16_t kVersion = 1;
inline constexpr const char *kDefaultDir = "/run/xsched";

enum Op : uint16_t
{
    kOpHello = 1, // returns the device BDF the proxy serves; no hardware access
    kOpPulse = 2, // fire one evict pulse
};

enum Status : int32_t
{
    kOk = 0,
    kBadRequest = -1,
    kRateLimited = -2,
    kNotServed = -3,
};

struct Request
{
    uint32_t magic;
    uint16_t version;
    uint16_t op;
    uint64_t seq;
};

struct Reply
{
    uint32_t magic;
    int32_t status;
    uint64_t seq;
    uint64_t pulses; // pulses this connection has fired so far
    char bdf[16];    // "dddd:bb:ss.f" of the served GPU
};

static_assert(sizeof(Request) == 16, "wire format");
static_assert(sizeof(Reply) == 40, "wire format");

} // namespace bar0proxy
} // namespace xsched::cuda::sm120
