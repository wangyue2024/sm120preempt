#include "../bar0.h"
#include "xsched/cuda/hal/level3/trap_sm120.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
using U32 = unsigned;
using U64 = unsigned long long;
using Clock = std::chrono::steady_clock;

constexpr int kThreads = 256;
constexpr U32 kSteps = 16384;
constexpr int kPollSteps = 4;
constexpr auto kTimeout = std::chrono::seconds(2);

U64 checkpoint_ptimer()
{
    uint32_t hi, lo, again;
    do {
        hi = rd32(0x9410);
        lo = rd32(0x9400);
        again = rd32(0x9410);
    } while (hi != again);
    return (U64{hi} << 32) | lo;
}

void require(bool ok, const char *message)
{
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        fflush(stderr);
        std::exit(1);
    }
}

#define CUDA(expr) do { \
    cudaError_t status = (expr); \
    if (status != cudaSuccess) { \
        fprintf(stderr, "FAIL: %s: %s\n", #expr, cudaGetErrorString(status)); \
        fflush(stderr); \
        std::exit(1); \
    } \
} while (0)

struct Observation {
    U64 working;
    U64 parked;
    U32 sm;
    U32 step_at_entry;
};

struct State {
    U32 *request;             // Device memory, set by trap or cleared by the launch stream
    U64 *request_seen;        // First safe point that acknowledges the request
    U32 *next;                // Next uncommitted step, keyed by logical CTA
    U32 *image;               // [slot][live register words, shared memory words]
    U32 *slots;               // Logical CTA -> its saved image slot
    U32 *owned;               // Atomic ownership per image slot
    U32 *effects;             // Non-idempotent counter per CTA
    U32 *output;
    Observation *observe;     // Mapped host memory, benchmark telemetry only
    U32 *unpark;              // Mapped host memory, resident control experiment only
    U32 shared_words;
    U32 slot_count;
    U32 short_prefix;         // CTAs in the prefix finish early in the mixed-wave test
    bool keep_resident;
};

__device__ __forceinline__ U64 gpu_time()
{
    U64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ U32 gpu_read(const U32 *word)
{
    U32 value;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
                 : "=r"(value) : "l"(word) : "memory");
    return value;
}

__device__ __forceinline__ U32 sys_read(const U32 *word)
{
    U32 value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(value) : "l"(word) : "memory");
    return value;
}

__device__ __forceinline__ void stamp(U64 *word)
{
    asm volatile("st.release.sys.global.u64 [%0], %1;"
                 :: "l"(word), "l"(gpu_time()) : "memory");
}

__device__ __forceinline__ U32 mix(U32 x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    return x ^ (x >> 15);
}

__device__ __forceinline__ U32 claim_slot(State state, U32 block)
{
    U32 slot = block % state.slot_count;
    for (U32 i = 0; i < state.slot_count; ++i) {
        if (atomicCAS(state.owned + slot, 0U, 1U) == 0) return slot;
        if (++slot == state.slot_count) slot = 0;
    }
    return state.slot_count;
}

__global__ void warp_probe(U64 *high_start)
{
    if (threadIdx.x == 0) stamp(high_start + blockIdx.x);
}

template<int Tag>
__global__ void request_probe(const U32 *request, const U32 *stop, U64 *heartbeat)
{
    const U64 deadline = gpu_time() + 100000000;
    U32 leave;
    do {
        if (threadIdx.x == 0) {
            if (blockIdx.x == 0) stamp(heartbeat);
            leave = gpu_read(request) | sys_read(stop) | (gpu_time() >= deadline);
        }
        const U64 tick = gpu_time();
        while (gpu_time() - tick < 256 + Tag) {}
    } while (!__shfl_sync(0xffffffffU, leave, 0));
}

template<int Registers, int MinBlocks>
__global__ __launch_bounds__(kThreads, MinBlocks)
void workload(State state, U64 *high_start)
{
    const U32 lane = threadIdx.x;
    const U32 block = blockIdx.x;
    if (high_start) {
        // The probe requests the same resources as a low-priority CTA.
        if (lane == 0) stamp(high_start + block);
        return;
    }

    const U32 first_step = state.next[block];
    if (first_step == kSteps || __syncthreads_or(lane == 0 && gpu_read(state.request)))
        return;

    extern __shared__ U32 shared[];
    __shared__ U32 save_slot;
    const U32 tid = block * kThreads + lane;
    const U32 step_limit = block < state.short_prefix ? kPollSteps : kSteps;
    const U32 image_words = Registers * kThreads + state.shared_words;
    const U32 restore_slot = first_step ? state.slots[block] : 0;
    const U32 *const restore = state.image + restore_slot * image_words;
    U32 reg[Registers];
    #pragma unroll
    for (int r = 0; r < Registers; ++r)
        reg[r] = first_step ? restore[r * kThreads + lane] : mix(tid + r * 0x9e3779b9U);

    for (U32 i = lane; i < state.shared_words; i += kThreads)
        shared[i] = first_step ? restore[Registers * kThreads + i] : mix(block + i * 0x85ebca6bU);
    __syncthreads();
    if (first_step && lane == 0) atomicExch(state.owned + restore_slot, 0U);

    for (U32 step = first_step; step < step_limit;) {
        const U32 stripe = (step & (state.shared_words / kThreads - 1)) * kThreads;
        const U32 other = stripe + ((lane + 1) & (kThreads - 1));
        const U32 from_peer = shared[other];
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < Registers; ++r)
            reg[r] = reg[r] * 1664525U + from_peer + step + r;
        shared[stripe + lane] = from_peer ^ reg[step % Registers == 0 ? 0 : Registers - 1];
        ++step;
        __syncthreads();

        if (lane == 0) {
            atomicAdd(state.effects + block, 1U);
            if (step == first_step + 1) {
                U32 sm;
                asm("mov.u32 %0, %%smid;" : "=r"(sm));
                state.observe[block].sm = sm;
                state.observe[block].step_at_entry = first_step;
                stamp(&state.observe[block].working);
            }
        }
        if (step % kPollSteps || step == step_limit)
            continue;
        if (!__syncthreads_or(lane == 0 && gpu_read(state.request)))
            continue;

        if (lane == 0 && gpu_read(state.request) == 1 &&
            atomicCAS(state.request, 1U, 3U) == 1) {
            stamp(state.request_seen);
        }

        if (state.keep_resident) {
            if (lane == 0) {
                stamp(&state.observe[block].parked);
                const U64 deadline = gpu_time() + 20000000;
                while (!sys_read(state.unpark) && gpu_time() < deadline)
                    __nanosleep(256);
            }
            __syncthreads();
            return;
        }

        if (lane == 0) save_slot = claim_slot(state, block);
        __syncthreads();
        // A re-eviction before all old images have been consumed can fill the
        // pool. Keep doing useful work and retry at the next safe point.
        if (save_slot == state.slot_count) continue;
        U32 *const image = state.image + save_slot * image_words;
        #pragma unroll
        for (int r = 0; r < Registers; ++r)
            image[r * kThreads + lane] = reg[r];
        for (U32 i = lane; i < state.shared_words; i += kThreads)
            image[Registers * kThreads + i] = shared[i];
        __syncthreads();
        if (lane == 0) {
            state.slots[block] = save_slot;
            state.next[block] = step;
        }
        return; // All warps EXIT. The stream must retire before the image is reused.
    }

    U32 digest = 0;
    #pragma unroll
    for (int r = 0; r < Registers; ++r)
        digest = mix(digest ^ reg[r]);
    for (U32 i = lane; i < state.shared_words; i += kThreads)
        digest = mix(digest ^ shared[i]);
    state.output[tid] = digest;
    __syncthreads();
    if (lane == 0) state.next[block] = kSteps;
}

template<class T>
T *device_buffer(size_t count)
{
    T *ptr = nullptr;
    CUDA(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template<class T>
T *mapped_buffer(T **gpu, size_t count)
{
    T *ptr = nullptr;
    CUDA(cudaHostAlloc(&ptr, count * sizeof(T), cudaHostAllocMapped));
    CUDA(cudaHostGetDevicePointer(gpu, ptr, 0));
    memset(ptr, 0, count * sizeof(T));
    return ptr;
}

template<class T>
T observe(const T *word)
{
    return __atomic_load_n(word, __ATOMIC_ACQUIRE);
}

template<class Ready>
void wait_until(Ready ready, const char *message)
{
    auto deadline = Clock::now() + kTimeout;
    while (!ready()) {
        require(Clock::now() < deadline, message);
        std::this_thread::yield();
    }
}

void await_stream(cudaStream_t stream)
{
    wait_until([&] {
        const auto result = cudaStreamQuery(stream);
        if (result == cudaErrorNotReady) return false;
        CUDA(result);
        return true;
    }, "CUDA stream did not retire");
}

int request_checkpoint(xsched::cuda::TrapSm120Context &trap, U32 *control, U64 *request_seen)
{
    // A posted BAR0 pulse has no delivery acknowledgement. Retry only until a
    // CTA acknowledges at a safe point, without restarting the latency clock.
    for (int pulse = 1; pulse <= 4; ++pulse) {
        if (observe(request_seen)) return pulse - 1;
        require(trap.RequestStreamCheckpoint(control), "could not request checkpoint");
        const auto retry_at = Clock::now() + std::chrono::microseconds(100);
        do {
            if (observe(request_seen)) return pulse;
            std::this_thread::yield();
        } while (Clock::now() < retry_at);
    }
    require(observe(request_seen), "no CTA acknowledged four checkpoint pulses");
    return 4;
}

double delta_us(U64 from, U64 to)
{
    require(to >= from, "GPU timestamp predates its trigger");
    return (to - from) / 1000.0;
}

void stats(const char *name, std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());
    const size_t p50 = (samples.size() - 1) / 2;
    const size_t p95 = (samples.size() * 95 + 99) / 100 - 1;
    printf("  %-26s n=%zu p50=%7.2f us p95=%7.2f us max=%7.2f us\n",
           name, samples.size(), samples[p50], samples[p95], samples.back());
}

template<class RunTarget>
void verify_targeting(xsched::cuda::TrapSm120Context &trap, RunTarget run_target)
{
    U64 *gpu_heartbeat;
    auto *heartbeat = mapped_buffer(&gpu_heartbeat, 2);
    U32 *gpu_stop;
    auto *stop = mapped_buffer(&gpu_stop, 1);
    U32 *gpu_control;
    auto *control = mapped_buffer(&gpu_control, 4);
    U32 *requests = device_buffer<U32>(2);
    CUDA(cudaMemset(requests, 0, 2 * sizeof(U32)));
    CUDA(cudaDeviceSynchronize());

    cudaStream_t peer, unbound;
    CUDA(cudaStreamCreateWithFlags(&peer, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&unbound, cudaStreamNonBlocking));

    CUfunction peer_func, unbound_func;
    CUDA(cudaGetFuncBySymbol(&peer_func, reinterpret_cast<const void *>(request_probe<0>)));
    CUDA(cudaGetFuncBySymbol(&unbound_func, reinterpret_cast<const void *>(request_probe<1>)));
    CUDA(cudaFuncSetAttribute(request_probe<0>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA(cudaFuncSetAttribute(request_probe<1>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    auto launch = [&] {
        request_probe<0><<<1, 32, 1024, peer>>>(requests, gpu_stop, gpu_heartbeat);
        request_probe<1><<<1, 32, 1024, unbound>>>(requests + 1, gpu_stop, gpu_heartbeat + 1);
        CUDA(cudaGetLastError());
    };
    // First launches may synchronize while the driver loads kernel code.
    __atomic_store_n(stop, 1, __ATOMIC_RELEASE);
    launch();
    await_stream(peer);
    await_stream(unbound);
    memset(heartbeat, 0, 2 * sizeof(U64));
    __atomic_store_n(stop, 0, __ATOMIC_RELEASE);
    require(trap.BindKernelCheckpointRequest(peer_func, reinterpret_cast<uint64_t>(gpu_control),
            reinterpret_cast<uint64_t>(requests)), "could not bind isolation probe");

    launch();
    wait_until([&] { return observe(heartbeat) && observe(heartbeat + 1); },
               "isolation probes did not start");
    run_target();
    const U64 peer_before = observe(heartbeat);
    const U64 unbound_before = observe(heartbeat + 1);
    wait_until([&] {
        return observe(heartbeat) > peer_before && observe(heartbeat + 1) > unbound_before;
    }, "a non-target stream stopped on broadcast");
    __atomic_store_n(stop, 1, __ATOMIC_RELEASE);
    await_stream(peer);
    await_stream(unbound);
    U32 got[2];
    CUDA(cudaMemcpy(got, requests, sizeof(got), cudaMemcpyDeviceToHost));
    require(got[0] == 0 && got[1] == 0, "broadcast signaled a non-target stream");
    require(trap.BindKernelPreemptBuffer(peer_func, 0, 0), "could not unbind isolation probe");
    CUDA(cudaStreamDestroy(peer));
    CUDA(cudaStreamDestroy(unbound));
    CUDA(cudaFree(requests));
    CUDA(cudaFreeHost(control));
    CUDA(cudaFreeHost(stop));
    CUDA(cudaFreeHost(heartbeat));
    puts("  PASS: armed, unarmed, and unbound streams isolated");
}

template<int Registers, int MinBlocks>
void run(xsched::cuda::TrapSm120Context &trap, const char *name, const cudaDeviceProp &gpu,
         U32 shared_bytes, int rounds)
{
    auto kernel = workload<Registers, MinBlocks>;
    CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    cudaFuncAttributes attr{};
    CUDA(cudaFuncGetAttributes(&attr, kernel));
    cudaFuncAttributes probe_attr = attr;
    if constexpr (MinBlocks == 6)
        CUDA(cudaFuncGetAttributes(&probe_attr, warp_probe));
    int resident_per_sm = 0;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_per_sm, kernel,
                                                       kThreads, shared_bytes));
    const int resident = resident_per_sm * gpu.multiProcessorCount;
    // Three waves let us test never-started CTAs as well as saved CTAs.
    const int grid = resident * 3;
    const size_t cta_image_bytes = Registers * kThreads * sizeof(U32) + shared_bytes;
    const size_t regs_per_cta = ((attr.numRegs + 7) / 8 * 8) * kThreads;
    const size_t probe_regs = ((probe_attr.numRegs + 7) / 8 * 8) * kThreads;
    const size_t smem_per_cta = shared_bytes + attr.sharedSizeBytes + gpu.reservedSharedMemPerBlock;
    const size_t probe_smem = shared_bytes + probe_attr.sharedSizeBytes + gpu.reservedSharedMemPerBlock;
    const bool regs = resident_per_sm * regs_per_cta + probe_regs > size_t(gpu.regsPerMultiprocessor);
    const bool smem = resident_per_sm * smem_per_cta + probe_smem > gpu.sharedMemPerMultiprocessor;
    const bool warps = (resident_per_sm + 1) * kThreads > gpu.maxThreadsPerMultiProcessor;
    printf("\n[%s] regs/thread=%d (probe=%d) local/thread=%zu shared/CTA=%zu resident=%d (%d/SM) grid=%d"
           " blocker=%s%s%s arena=%zu KiB\n",
           name, attr.numRegs, probe_attr.numRegs, attr.localSizeBytes, smem_per_cta, resident, resident_per_sm,
           grid, regs ? "registers " : "", smem ? "shared " : "", warps ? "warp-slots" : "",
           resident * cta_image_bytes / 1024);
    require(attr.localSizeBytes == 0 && regs == (Registers > 100) &&
            smem == (shared_bytes >= 65536) && warps == (MinBlocks == 6) &&
            resident_per_sm == MinBlocks, "benchmark does not isolate the requested resource");

    State state{};
    state.request = device_buffer<U32>(1);
    state.next = device_buffer<U32>(grid);
    state.image = device_buffer<U32>(resident * cta_image_bytes / sizeof(U32));
    state.slots = device_buffer<U32>(grid);
    state.owned = device_buffer<U32>(resident);
    state.effects = device_buffer<U32>(grid);
    state.output = device_buffer<U32>(grid * kThreads);
    state.shared_words = shared_bytes / sizeof(U32);
    state.slot_count = resident;
    auto *request_seen = mapped_buffer(&state.request_seen, 1);
    auto *telemetry = mapped_buffer(&state.observe, grid);
    auto *unpark = mapped_buffer(&state.unpark, 1);
    U32 *gpu_control;
    auto *control = mapped_buffer(&gpu_control, 4 + 2 * grid);
    U64 *gpu_high_start;
    auto *high_start = mapped_buffer(&gpu_high_start, gpu.multiProcessorCount);

    int low_priority, high_priority;
    CUDA(cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority));
    require(high_priority < low_priority, "stream priority unavailable");
    cudaStream_t low, high;
    CUDA(cudaStreamCreateWithPriority(&low, cudaStreamNonBlocking, low_priority));
    CUDA(cudaStreamCreateWithPriority(&high, cudaStreamNonBlocking, high_priority));

    CUfunction func = nullptr;
    CUDA(cudaGetFuncBySymbol(&func, reinterpret_cast<const void *>(kernel)));
    require(trap.BindKernelCheckpointRequest(func, reinterpret_cast<uint64_t>(gpu_control),
                                             reinterpret_cast<uint64_t>(state.request)),
            "could not bind the checkpoint request");

    auto reset = [&] {
        __atomic_store_n(control, 0, __ATOMIC_RELEASE);
        __atomic_store_n(request_seen, 0, __ATOMIC_RELEASE);
        CUDA(cudaMemsetAsync(state.request, 0, sizeof(U32), low));
        CUDA(cudaMemsetAsync(state.next, 0, grid * sizeof(U32), low));
        CUDA(cudaMemsetAsync(state.owned, 0, resident * sizeof(U32), low));
        CUDA(cudaMemsetAsync(state.effects, 0, grid * sizeof(U32), low));
        CUDA(cudaMemsetAsync(state.output, 0, grid * kThreads * sizeof(U32), low));
        memset(telemetry, 0, grid * sizeof(Observation));
        memset(high_start, 0, gpu.multiProcessorCount * sizeof(U64));
        __atomic_store_n(unpark, 0, __ATOMIC_RELEASE);
    };
    auto launch_low = [&] {
        workload<Registers, MinBlocks><<<grid, kThreads, shared_bytes, low>>>(state, nullptr);
        CUDA(cudaGetLastError());
    };
    auto launch_high = [&] {
        if constexpr (MinBlocks == 6) {
            warp_probe<<<gpu.multiProcessorCount, kThreads, shared_bytes, high>>>(gpu_high_start);
        } else {
            workload<Registers, MinBlocks><<<gpu.multiProcessorCount, kThreads, shared_bytes, high>>>(
                State{}, gpu_high_start);
        }
        CUDA(cudaGetLastError());
    };
    auto read = [](U32 *src, std::vector<U32> &dst) {
        CUDA(cudaMemcpy(dst.data(), src, dst.size() * sizeof(U32), cudaMemcpyDeviceToHost));
    };

    std::vector<U32> reference(grid * kThreads), output(reference.size()), effects(grid), next(grid);
    std::vector<U32> owners(resident);
    reset();
    launch_low();
    await_stream(low);
    read(state.output, reference);

    auto verify = [&] {
        read(state.output, output);
        read(state.effects, effects);
        read(state.next, next);
        require(output == reference, "final register/shared digest differs from uninterrupted run");
        for (int i = 0; i < grid; ++i) {
            const U32 steps = U32(i) < state.short_prefix ? kPollSteps : kSteps;
            require(effects[i] == steps && next[i] == kSteps, "missing or repeated CTA work");
        }
        read(state.owned, owners);
        require(std::all_of(owners.begin(), owners.end(), [](U32 v) { return v == 0; }),
                "completed job still owns checkpoint slots");
    };
    verify();

    auto verify_residency = [&] {
        wait_until([&] {
            int count = 0;
            for (int b = state.short_prefix; b < grid; ++b)
                count += observe(&telemetry[b].working) != 0;
            return count >= resident;
        }, "low grid did not saturate the GPU");
        std::array<int, 1024> seen{};
        for (int b = state.short_prefix; b < grid; ++b) {
            if (!observe(&telemetry[b].working)) continue;
            const auto sm = observe(&telemetry[b].sm);
            require(sm < seen.size(), "unexpected SM ID");
            ++seen[sm];
        }
        require(std::count(seen.begin(), seen.end(), resident_per_sm) == gpu.multiProcessorCount,
                "low grid was not resident on every SM");
    };
    auto first_high = [&] {
        U64 earliest = ~0ULL;
        for (int b = 0; b < gpu.multiProcessorCount; ++b) {
            const U64 stamp = observe(high_start + b);
            if (stamp) earliest = std::min(stamp, earliest);
        }
        return earliest;
    };

    if constexpr (Registers == 4) {
        reset();
        await_stream(low);
        verify_targeting(trap, [&] {
            launch_low();
            verify_residency();
            request_checkpoint(trap, control, request_seen);
            await_stream(low);
        });
        read(state.next, next);
        require(std::count_if(next.begin(), next.end(), [](U32 step) {
            return step > 0 && step < kSteps;
        }) == resident, "target stream did not checkpoint during isolation probe");
    }

    state.keep_resident = true;
    reset();
    launch_low();
    verify_residency();
    launch_high();
    const U64 hold = checkpoint_ptimer();
    request_checkpoint(trap, control, request_seen);
    wait_until([&] {
        int count = 0;
        for (int b = 0; b < grid; ++b) count += observe(&telemetry[b].parked) != 0;
        return count == resident;
    }, "trap did not park the resident CTAs");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const U64 release = checkpoint_ptimer();
    __atomic_store_n(unpark, 1, __ATOMIC_RELEASE);
    await_stream(high);
    // After unpark, existing CTAs exit and pending CTAs skip at entry.
    await_stream(low);
    require(first_high() >= release && first_high() != ~0ULL,
            "high-priority CTA admitted during resident hold");
    printf("  resident control           blocked during %.2f us request-to-unpark window\n",
           delta_us(hold, release));

    state.keep_resident = false;
    std::vector<double> signal, drain, admission, high_wait, resume_first, resume_all;
    int retried = 0, max_pulses = 0;
    for (int i = 0; i <= rounds; ++i) {
        if (i == rounds) {
            state.short_prefix = resident;
            reset();
            launch_low();
            await_stream(low);
            read(state.output, reference);
            verify();
        }
        reset();
        launch_low();
        for (int cycle = 0; cycle < 2; ++cycle) {
            verify_residency();
            memset(high_start, 0, gpu.multiProcessorCount * sizeof(U64));
            const U64 queued = checkpoint_ptimer();
            launch_high();
            const U64 request = checkpoint_ptimer();
            const int pulses = request_checkpoint(trap, control, request_seen);
            await_stream(low);
            const U64 retired = checkpoint_ptimer();
            await_stream(high);
            require(first_high() != ~0ULL, "high probe published no timestamp");
            if (i < rounds) {
                signal.push_back(delta_us(request, observe(request_seen)));
                drain.push_back(delta_us(request, retired));
                admission.push_back(delta_us(request, first_high()));
                high_wait.push_back(delta_us(queued, first_high()));
                retried += pulses > 1;
                max_pulses = std::max(max_pulses, pulses);
            }

            read(state.next, next);
            read(state.effects, effects);
            read(state.owned, owners);
            int pending = 0, saved = 0, done = 0;
            for (int b = 0; b < grid; ++b) {
                const U32 step = next[b];
                const U32 expected = step == kSteps && U32(b) < state.short_prefix ? kPollSteps : step;
                require(step <= kSteps && step % kPollSteps == 0, "invalid checkpoint step");
                require(effects[b] == expected, "checkpoint progress disagrees with side effects");
                if (step == 0) ++pending;
                else if (step == kSteps) ++done;
                else ++saved;
            }
            const int owned = std::count(owners.begin(), owners.end(), 1U);
            if (saved != resident || pending == 0 || U32(done) != state.short_prefix ||
                owned != saved) {
                fprintf(stderr, "job=%d cycle=%d saved=%d pending=%d done=%d owned=%d expected_saved=%d expected_done=%u\n",
                        i + 1, cycle + 1, saved, pending, done, owned, resident, state.short_prefix);
                require(false, "did not checkpoint a resident wave");
            }
            memset(telemetry, 0, grid * sizeof(Observation));
            const U64 resume = checkpoint_ptimer();
            __atomic_store_n(control, 0, __ATOMIC_RELEASE);
            __atomic_store_n(request_seen, 0, __ATOMIC_RELEASE);
            CUDA(cudaMemsetAsync(state.request, 0, sizeof(U32), low));
            launch_low();

            U64 first = ~0ULL, last = 0;
            wait_until([&] {
                for (int b = 0; b < grid; ++b)
                    if (next[b] && next[b] != kSteps && !observe(&telemetry[b].working))
                        return false;
                return true;
            }, "saved CTA did not resume");
            for (int b = 0; b < grid; ++b) {
                if (next[b] == 0 || next[b] == kSteps) continue;
                const U64 time = observe(&telemetry[b].working);
                require(observe(&telemetry[b].step_at_entry) == next[b], "wrong restart step");
                first = std::min(first, time);
                last = std::max(last, time);
            }
            if (i < rounds) {
                resume_first.push_back(delta_us(resume, first));
                resume_all.push_back(delta_us(resume, last));
            }
            if (i == 0 || i == rounds)
                printf("  %s cycle %d: saved=%d pending=%d done=%d checkpoint bytes=%zu\n",
                       i == rounds ? "mixed" : "full", cycle + 1, saved, pending, done,
                       saved * cta_image_bytes);
        }
        await_stream(low);
        verify();
    }
    stats("trap to first safe point", signal);
    stats("trap to low stream retired", drain);
    stats("trap to first high CTA", admission);
    stats("enqueue to first high CTA", high_wait);
    stats("relaunch to first restored", resume_first);
    stats("relaunch to all restored", resume_all);
    printf("  pulse retries              %d/%zu requests, max %d pulses\n",
           retried, signal.size(), max_pulses);
    printf("  PASS: exact-once effects and final digest (%d jobs, %d evictions)\n",
           rounds + 1, 2 * (rounds + 1));

    require(trap.BindKernelPreemptBuffer(func, 0, 0), "could not unbind checkpoint request");
    CUDA(cudaStreamDestroy(high));
    CUDA(cudaStreamDestroy(low));
    CUDA(cudaFreeHost(high_start));
    CUDA(cudaFreeHost(unpark));
    CUDA(cudaFreeHost(control));
    CUDA(cudaFreeHost(request_seen));
    CUDA(cudaFreeHost(telemetry));
    CUDA(cudaFree(state.output));
    CUDA(cudaFree(state.effects));
    CUDA(cudaFree(state.owned));
    CUDA(cudaFree(state.slots));
    CUDA(cudaFree(state.image));
    CUDA(cudaFree(state.next));
    CUDA(cudaFree(state.request));
}

// ============================================================================
// Mode 5 (Path A): Zero-Source-Modification Arbitrary-PC Trap-Frame + 64 KB SMEM Evict & RTT Restore
// ============================================================================
constexpr int kPathAThreads = 32;
constexpr U32 kPathASharedWords = 16384; // 64 KB per CTA (1 CTA/SM on SM120)
constexpr U32 kPathASteps = 1U << 20;

// 100% Unmodified User Kernel:
// Mode 5 (Path A) Uncooperative Kernel:
// 100% pure C++ CUDA kernel: zero inline PTX, zero C++ preemption prologues,
// zero C++ eviction epilogues, and zero safe-point polling.
__global__ __launch_bounds__(kPathAThreads, 1)
void path_a_kernel(U32 *output, U64 *working, U64 *high_start)
{
    extern __shared__ U32 shared[];
    volatile U32 *smem = shared;
    U32 tid = threadIdx.x;
    U32 bid = blockIdx.x;
    if (high_start) {
        if (tid == 0) stamp(high_start + bid);
        return;
    }
    #pragma unroll 1
    for (U32 i = tid; i < kPathASharedWords; i += kPathAThreads) {
        smem[i] = (bid + 1U) * 0x9e3779b9U ^ (i * 0x85ebca6bU);
    }
    __syncthreads();
    if (tid == 0 && working) stamp(working + bid);

    // Uncooperative inner loop: NO request polling inside the loop!
    // Interrupted at an arbitrary PC, evicted inside the trap handler (ext_code),
    // and resumed directly to TRAP_RETURN_PC via hardware RTT.FALLTHROUGH.
    U32 acc = tid ^ (bid * 1664525U);
    U32 limit = kPathASteps | (blockIdx.y & tid);
    #pragma unroll 1
    for (U32 s = 0; s < limit; ++s) {
        U32 idx = (((s * 32U) + tid) & (kPathASharedWords - 1U));
        U32 v = smem[idx];
        acc = mix(acc + v) ^ s;
    }
    U32 final_sh = smem[(acc & 0x1ffU) * 32U + tid];
    output[bid * kPathAThreads + tid] = mix(acc ^ final_sh);
}

constexpr U32 kUnivThreadsX = 32;
constexpr U32 kUnivThreadsY = 4;
constexpr U32 kUnivThreads = kUnivThreadsX * kUnivThreadsY; // 128 threads = 4 warps/CTA
constexpr U32 kUnivSteps = 1U << 15;

// Universal Mode 5 Uncooperative Kernel (Steps 1-4):
// - Zero source modification (no polling, no prologue/epilogue, no inline PTX)
// - High GPR pressure (R14..R63+) + Uniform Registers (UR4..UR15)
// - Multi-warp 2D CTA (dim3(32, 4) = 128 threads = 4 warps/CTA) + 64 KiB Shared Memory
// - Intra-warp SIMT branch divergence (if/else) + inter-warp __syncthreads() (BAR.SYNC)
__global__ __launch_bounds__(kUnivThreads, 2)
void universal_trapframe_kernel(U32 *output, U64 *working, U64 *high_start)
{
    extern __shared__ U32 shared[];
    volatile U32 *smem = shared;
    const U32 tid = threadIdx.y * blockDim.x + threadIdx.x;
    const U32 bid = blockIdx.x;
    if (high_start) {
        if (tid == 0) stamp(high_start + bid);
        return;
    }

    #pragma unroll 1
    for (U32 i = tid; i < kPathASharedWords; i += kUnivThreads) {
        smem[i] = (bid + 1U) * 0x9e3779b9U ^ (i * 0x85ebca6bU);
    }
    __syncthreads();

    // 20 live per-thread accumulators + loop state -> ~34-42 hardware GPRs + URs
    U32 r0  = mix(tid + bid * 17U + 0x101U);
    U32 r1  = mix(tid + bid * 31U + 0x202U);
    U32 r2  = mix(tid + bid * 47U + 0x303U);
    U32 r3  = mix(tid + bid * 61U + 0x404U);
    U32 r4  = mix(tid + bid * 73U + 0x505U);
    U32 r5  = mix(tid + bid * 89U + 0x606U);
    U32 r6  = mix(tid + bid * 97U + 0x707U);
    U32 r7  = mix(tid + bid * 103U + 0x808U);
    U32 r8  = mix(tid + bid * 109U + 0x909U);
    U32 r9  = mix(tid + bid * 113U + 0xa0aU);
    U32 r10 = mix(tid + bid * 127U + 0xb0bU);
    U32 r11 = mix(tid + bid * 131U + 0xc0cU);
    U32 r12 = mix(tid + bid * 137U + 0xd0dU);
    U32 r13 = mix(tid + bid * 139U + 0xe0eU);
    U32 r14 = mix(tid + bid * 149U + 0xf0fU);
    U32 r15 = mix(tid + bid * 151U + 0x111U);
    U32 r16 = mix(tid + bid * 157U + 0x222U);
    U32 r17 = mix(tid + bid * 163U + 0x333U);
    U32 r18 = mix(tid + bid * 167U + 0x444U);

    // In multi-wave launches (gridDim.x > 30), CTAs 0..9 run a short 64-step prefix and exit
    // naturally before preemption (State A: w==0, ack==0), CTAs 10..39 occupy all 30 SMs
    // mid-execution (State B: w==4, ack==4), and CTAs 40..59 wait in the CWD queue (State C: w==0, ack==4).
    const bool short_cta = (gridDim.x > 30U && bid < 10U);
    U32 limit = (short_cta ? 64U : kUnivSteps) | (blockIdx.y & tid);
    #pragma unroll 1
    for (U32 s = 0; s < limit; ++s) {
        if (s == 128U && tid == 0 && working) stamp(working + bid);
        U32 idx = ((s * kUnivThreads) + tid) & (kPathASharedWords - 1U);
        U32 peer_idx = ((s * kUnivThreads) + ((tid + 37U) & (kUnivThreads - 1U))) & (kPathASharedWords - 1U);
        U32 v = smem[peer_idx];
        // Intra-warp SIMT branch divergence (odd vs even lanes diverge each step)
        if (((tid ^ s) & 1U) != 0U) {
            r0  = mix(r0 + v + s);
            r1  = r1 ^ (r0 * 1664525U);
            r2  = r2 + (r1 ^ idx);
            r3  = mix(r3 ^ r2);
            r4  = r4 + (r3 * 3U);
            r5  = r5 ^ r4;
            r12 = r12 + (r5 ^ v);
            r13 = mix(r13 ^ r12);
            r14 = r14 + (r13 * 7U);
            r15 = r15 ^ r14;
        } else {
            r6  = mix(r6 ^ (v + s));
            r7  = r7 + (r6 * 22695477U);
            r8  = r8 ^ (r7 + idx);
            r9  = mix(r9 + r8);
            r10 = r10 ^ (r9 * 5U);
            r11 = r11 + r10;
            r16 = r16 ^ (r11 + v);
            r17 = mix(r17 + r16);
            r18 = r18 ^ (r17 * 11U);
        }
        __syncthreads();
        if ((s & 7U) == 0U) {
            smem[idx] = v ^ r0 ^ r6 ^ r15 ^ r18;
        }
        __syncthreads();
    }

    U32 digest = r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7 ^ r8 ^ r9 ^ r10 ^ r11 ^
                 r12 ^ r13 ^ r14 ^ r15 ^ r16 ^ r17 ^ r18;
    U32 final_sh = smem[((digest & 0x7fU) * kUnivThreads) + tid];
    // Non-idempotent += accumulation: if a pre-completed State A CTA re-executed on Restore,
    // output[...] would be doubled and fail verification!
    output[bid * kUnivThreads + tid] += mix(digest ^ final_sh);
}

__global__ __launch_bounds__(kUnivThreads, 1)
void universal_highreg_kernel(U32 *output, U64 *working, U64 *high_start)
{
    extern __shared__ U32 shared[];
    volatile U32 *smem = shared;
    const U32 tid = threadIdx.y * blockDim.x + threadIdx.x;
    const U32 bid = blockIdx.x;
    if (high_start) {
        if (tid == 0) stamp(high_start + bid);
        return;
    }

    #pragma unroll 1
    for (U32 i = tid; i < kPathASharedWords; i += kUnivThreads) {
        smem[i] = (bid + 3U) * 0x9e3779b9U ^ (i * 0x85ebca6bU);
    }
    __syncthreads();
    if (tid == 0 && working) stamp(working + bid);

    // 242 live per-thread accumulators -> 255 hardware GPRs (R0..R252 architectural max!) + URs
    constexpr int kAccRegs = 242;
    U32 acc[kAccRegs];
    #pragma unroll
    for (int k = 0; k < kAccRegs; ++k) {
        acc[k] = mix(tid + bid * U32(19 + 4 * k) + U32(0x13579bdfU * (k + 1)));
    }

    U32 limit = (kUnivSteps / 2) | (blockIdx.y & tid);
    #pragma unroll 1
    for (U32 s = 0; s < limit; ++s) {
        U32 idx = ((s * kUnivThreads) + tid) & (kPathASharedWords - 1U);
        U32 peer_idx = ((s * kUnivThreads) + ((tid + 53U) & (kUnivThreads - 1U))) & (kPathASharedWords - 1U);
        U32 v = smem[peer_idx];
        if (((tid ^ s) & 1U) != 0U) {
            acc[0] = mix(acc[0] + v + s);
            #pragma unroll
            for (int k = 1; k < kAccRegs / 2; ++k) {
                acc[k] = (acc[k] ^ (acc[k - 1] * U32(2 * k + 1))) + idx;
            }
        } else {
            acc[kAccRegs / 2] = mix(acc[kAccRegs / 2] ^ (v + s));
            #pragma unroll
            for (int k = kAccRegs / 2 + 1; k < kAccRegs; ++k) {
                acc[k] = (acc[k] + (acc[k - 1] ^ idx)) ^ U32(0x9e3779b9U + k);
            }
        }
        __syncthreads();
        if ((s & 7U) == 0U) {
            smem[idx] = v ^ acc[0] ^ acc[kAccRegs / 2 - 1] ^ acc[kAccRegs / 2] ^ acc[kAccRegs - 1];
        }
        __syncthreads();
    }

    U32 digest = 0;
    #pragma unroll
    for (int k = 0; k < kAccRegs; ++k) {
        digest ^= acc[k];
    }
    U32 final_sh = smem[((digest & 0x7fU) * kUnivThreads) + tid];
    output[bid * kUnivThreads + tid] += mix(digest ^ final_sh);
}

void run_trapframe(xsched::cuda::TrapSm120Context &trap, const cudaDeviceProp &gpu, int rounds)
{
    const U32 shared_bytes = kPathASharedWords * sizeof(U32);
    CUDA(cudaFuncSetAttribute(path_a_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    CUDA(cudaFuncSetAttribute(path_a_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    cudaFuncAttributes attr{};
    CUDA(cudaFuncGetAttributes(&attr, path_a_kernel));
    const int resident = gpu.multiProcessorCount;
    const size_t arena_words = (0xe00000 / sizeof(U32)) + size_t(resident) * kPathASharedWords;
    printf("\n[trapframe] regs/thread=%d local/thread=%zu shared/CTA=%zu resident=%d (1/SM)"
           " blocker=shared (zero-source-mod trap-frame + 64KB SMEM + RTT restore) arena=%zu KiB\n",
           attr.numRegs, attr.localSizeBytes, shared_bytes + attr.sharedSizeBytes + gpu.reservedSharedMemPerBlock,
           resident, arena_words * sizeof(U32) / 1024);
    require(attr.localSizeBytes == 0 && attr.numRegs <= 80,
            "path_a_kernel register count exceeds trap-frame window");

    U32 *d_arena = device_buffer<U32>(arena_words);
    U32 *d_output = device_buffer<U32>(resident * kPathAThreads);
    U64 *gpu_working;
    auto *working = mapped_buffer(&gpu_working, resident);
    U64 *gpu_high_start;
    auto *high_start = mapped_buffer(&gpu_high_start, resident);

    U32 *gpu_control;
    auto *control = mapped_buffer(&gpu_control, 4 + 2 * resident);
    reinterpret_cast<U64 *>(control)[1] = reinterpret_cast<U64>(d_arena);

    int low_priority, high_priority;
    CUDA(cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority));
    cudaStream_t low, high;
    CUDA(cudaStreamCreateWithPriority(&low, cudaStreamNonBlocking, low_priority));
    CUDA(cudaStreamCreateWithPriority(&high, cudaStreamNonBlocking, high_priority));

    CUfunction func = nullptr;
    CUDA(cudaGetFuncBySymbol(&func, reinterpret_cast<const void *>(path_a_kernel)));
    require(trap.BindKernelTrapFrameEvict(func, reinterpret_cast<uint64_t>(gpu_control),
                                          reinterpret_cast<uint64_t>(d_arena)),
            "could not bind Mode 5 trap-frame evict handler");

    auto reset = [&] {
        memset(const_cast<U32 *>(control), 0, (4 + 2 * resident) * sizeof(U32));
        reinterpret_cast<U64 *>(control)[1] = reinterpret_cast<U64>(d_arena);
        memset(working, 0, resident * sizeof(U64));
        memset(high_start, 0, resident * sizeof(U64));
        CUDA(cudaMemsetAsync(d_arena, 0, 0xe00000, low));
        CUDA(cudaMemsetAsync(d_output, 0, resident * kPathAThreads * sizeof(U32), low));
    };

    std::vector<U32> reference(resident * kPathAThreads), output(resident * kPathAThreads);
    reset();
    path_a_kernel<<<resident, kPathAThreads, shared_bytes, low>>>(d_output, nullptr, nullptr);
    await_stream(low);
    CUDA(cudaMemcpy(reference.data(), d_output, reference.size() * sizeof(U32), cudaMemcpyDeviceToHost));

    std::vector<double> evict_signal, evict_drain, high_admission, rtt_restore;
    for (int r = 0; r < rounds; ++r) {
        reset();
        await_stream(low);
        path_a_kernel<<<resident, kPathAThreads, shared_bytes, low>>>(d_output, gpu_working, nullptr);
        wait_until([&] {
            int count = 0;
            for (int b = 0; b < resident; ++b) count += observe(working + b) != 0;
            return count == resident;
        }, "path_a_kernel did not saturate all SMs");

        path_a_kernel<<<resident, kPathAThreads, shared_bytes, high>>>(nullptr, nullptr, gpu_high_start);
        const U64 req_t = checkpoint_ptimer();
        require(trap.RequestStreamTrapFrameEvict(control, resident),
                "could not trigger Mode 5 trap-frame evict");
        const U64 evict_done_t = checkpoint_ptimer();
        await_stream(low);
        const U64 drain_t = checkpoint_ptimer();
        await_stream(high);

        U64 earliest_high = ~0ULL;
        for (int b = 0; b < resident; ++b) {
            if (observe(high_start + b)) earliest_high = std::min(earliest_high, observe(high_start + b));
        }
        require(earliest_high != ~0ULL, "Mode 5 high CTA did not start");
        evict_signal.push_back(delta_us(req_t, evict_done_t));
        evict_drain.push_back(delta_us(req_t, drain_t));
        high_admission.push_back(delta_us(req_t, earliest_high));

        // Patch 2-instruction Entry Spin Stub at func_pc + 0x00, relaunch unmodified kernel,
        // and pulse BAR0 trap so ext_code restores 64 KB SMEM + 112B Trap Frame + R14:R15
        // and teleports all CTAs back to TRAP_RETURN_PC via RTT.FALLTHROUGH.
        require(trap.PrepareKernelTrapFrameRestore(func),
                "could not patch Mode 5 entry spin stub");
        path_a_kernel<<<resident, kPathAThreads, shared_bytes, low>>>(d_output, gpu_working, nullptr);
        const U64 rtt_pulse_t = checkpoint_ptimer();
        require(trap.RequestStreamTrapFrameRestore(control, resident),
                "could not trigger Mode 5 RTT restore pulse");
        const U64 rtt_done_t = checkpoint_ptimer();
        require(trap.FinishKernelTrapFrameRestore(),
                "could not restore original kernel entry instructions");
        await_stream(low);

        rtt_restore.push_back(delta_us(rtt_pulse_t, rtt_done_t));

        CUDA(cudaMemcpy(output.data(), d_output, output.size() * sizeof(U32), cudaMemcpyDeviceToHost));
        require(output == reference, "Mode 5 zero-source-mod RTT restored output differs from uninterrupted run");
    }
    stats("trap to all 30 CTAs saved", evict_signal);
    stats("trap to low stream retired", evict_drain);
    stats("trap to first high CTA", high_admission);
    stats("RTT pulse to all 30 restored", rtt_restore);
    printf("  PASS: zero-source-mod arbitrary-PC 112B trap-frame + 64KB SMEM evict & hardware RTT restore (%d rounds, 0 errors)\n",
           rounds);

    require(trap.BindKernelPreemptBuffer(func, 0, 0), "could not unbind Mode 5 kernel");
    CUDA(cudaStreamDestroy(high));
    CUDA(cudaStreamDestroy(low));
    CUDA(cudaFreeHost(control));
    CUDA(cudaFreeHost(high_start));
    CUDA(cudaFreeHost(working));
    CUDA(cudaFree(d_output));
    CUDA(cudaFree(d_arena));

    // ========================================================================
    // Universal Mode 5 Verification (Steps 1-4: High GPRs + URs + Multi-Warp
    // 2D CTA + 64KB SMEM + SIMT Divergence + __syncthreads() + Multi-Wave Grid
    // + 3rd Code Cave ext3_code up to 192 regs/thread)
    // ========================================================================
    const dim3 ublock(kUnivThreadsX, kUnivThreadsY, 1);
    struct UnivCase {
        const char *tag;
        void (*kern)(U32 *, U64 *, U64 *);
        int waves;
    };
    const UnivCase ucases[] = {
        {"trapframe-univ-1w", universal_trapframe_kernel, 1},
        {"trapframe-univ-2w", universal_trapframe_kernel, 2},
        {"trapframe-univ-hireg", universal_highreg_kernel, 1},
    };

    for (const auto &uc : ucases) {
        CUDA(cudaFuncSetAttribute(uc.kern, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        CUDA(cudaFuncSetAttribute(uc.kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        cudaFuncAttributes uattr{};
        CUDA(cudaFuncGetAttributes(&uattr, uc.kern));
        const int waves = uc.waves;
        const int ugrid = resident * waves;
        const size_t uarena_words = (0xe00000 / sizeof(U32)) + size_t(ugrid) * kPathASharedWords;
        printf("\n[%s] regs/thread=%d local/thread=%zu shared/CTA=%zu block=(%u,%u) resident=%d grid=%d (%d waves)"
               " arena=%zu KiB\n",
               uc.tag, uattr.numRegs, uattr.localSizeBytes,
               shared_bytes + uattr.sharedSizeBytes + gpu.reservedSharedMemPerBlock,
               ublock.x, ublock.y, resident, ugrid, waves, uarena_words * sizeof(U32) / 1024);
        require(uattr.localSizeBytes == 0 && uattr.numRegs > 16 && uattr.numRegs <= 255,
                "universal kernel register count outside (16, 255]");

        U32 *u_arena = device_buffer<U32>(uarena_words);
        U32 *u_output = device_buffer<U32>(ugrid * kUnivThreads);
        U64 *u_gpu_working;
        auto *u_working = mapped_buffer(&u_gpu_working, ugrid);
        U64 *u_gpu_high;
        auto *u_high_start = mapped_buffer(&u_gpu_high, resident);
        U32 *u_gpu_control;
        auto *u_control = mapped_buffer(&u_gpu_control, 4 + 2 * ugrid);

        cudaStream_t ulow, uhigh;
        CUDA(cudaStreamCreateWithPriority(&ulow, cudaStreamNonBlocking, low_priority));
        CUDA(cudaStreamCreateWithPriority(&uhigh, cudaStreamNonBlocking, high_priority));

        CUfunction ufunc = nullptr;
        CUDA(cudaGetFuncBySymbol(&ufunc, reinterpret_cast<const void *>(uc.kern)));
        require(trap.BindKernelTrapFrameEvict(ufunc, reinterpret_cast<uint64_t>(u_gpu_control),
                                              reinterpret_cast<uint64_t>(u_arena)),
                "could not bind universal Mode 5 kernel");

        auto ureset = [&] {
            memset(const_cast<U32 *>(u_control), 0, (4 + 2 * ugrid) * sizeof(U32));
            reinterpret_cast<U64 *>(u_control)[1] = reinterpret_cast<U64>(u_arena);
            memset(u_working, 0, ugrid * sizeof(U64));
            memset(u_high_start, 0, resident * sizeof(U64));
            CUDA(cudaMemsetAsync(u_arena, 0, 0xe00000, ulow));
            CUDA(cudaMemsetAsync(u_output, 0, ugrid * kUnivThreads * sizeof(U32), ulow));
        };

        std::vector<U32> uref(ugrid * kUnivThreads), uout(ugrid * kUnivThreads);
        ureset();
        uc.kern<<<ugrid, ublock, shared_bytes, ulow>>>(u_output, nullptr, nullptr);
        await_stream(ulow);
        CUDA(cudaMemcpy(uref.data(), u_output, uref.size() * sizeof(U32), cudaMemcpyDeviceToHost));

        std::vector<double> u_evict, u_high, u_restore;
        for (int r = 0; r < rounds; ++r) {
            ureset();
            await_stream(ulow);
            uc.kern<<<ugrid, ublock, shared_bytes, ulow>>>(u_output, u_gpu_working, nullptr);
            wait_until([&] {
                int count = 0;
                for (int b = 0; b < ugrid; ++b) count += observe(u_working + b) != 0;
                return count >= resident;
            }, "universal kernel did not saturate all SMs");

            path_a_kernel<<<resident, kPathAThreads, shared_bytes, uhigh>>>(nullptr, nullptr, u_gpu_high);
            const U64 req_t = checkpoint_ptimer();
            require(trap.RequestStreamTrapFrameEvict(u_control, ugrid),
                    "could not trigger universal Mode 5 evict");
            const U64 evict_t = checkpoint_ptimer();
            await_stream(ulow);
            await_stream(uhigh);

            U64 earliest_high = ~0ULL;
            for (int b = 0; b < resident; ++b) {
                if (observe(u_high_start + b)) earliest_high = std::min(earliest_high, observe(u_high_start + b));
            }
            require(earliest_high != ~0ULL, "universal Mode 5 high CTA did not start");
            u_evict.push_back(delta_us(req_t, evict_t));
            u_high.push_back(delta_us(req_t, earliest_high));

            require(trap.PrepareKernelTrapFrameRestore(ufunc),
                    "could not patch universal Mode 5 entry spin stub");
            uc.kern<<<ugrid, ublock, shared_bytes, ulow>>>(u_output, u_gpu_working, nullptr);
            const U64 rtt_t0 = checkpoint_ptimer();
            bool rtt_ok = trap.RequestStreamTrapFrameRestore(u_control, ugrid);
            const U64 rtt_t1 = checkpoint_ptimer();
            require(trap.FinishKernelTrapFrameRestore(),
                    "could not restore universal Mode 5 entry instructions");
            require(rtt_ok, "could not trigger universal Mode 5 RTT restore");
            await_stream(ulow);
            u_restore.push_back(delta_us(rtt_t0, rtt_t1));

            CUDA(cudaMemcpy(uout.data(), u_output, uout.size() * sizeof(U32), cudaMemcpyDeviceToHost));
            require(uout == uref, "universal Mode 5 restored output differs from uninterrupted reference");
        }
        stats("trap to all CTAs evicted", u_evict);
        stats("trap to first high CTA", u_high);
        stats("RTT pulse to all restored", u_restore);
        printf("  PASS: universal Mode 5 (%d waves, %d CTAs, 128 threads/CTA, %d regs/thread, SIMT divergence + __syncthreads, %d rounds, 0 errors)\n",
               waves, ugrid, uattr.numRegs, rounds);

        require(trap.BindKernelPreemptBuffer(ufunc, 0, 0), "could not unbind universal Mode 5 kernel");
        CUDA(cudaStreamDestroy(uhigh));
        CUDA(cudaStreamDestroy(ulow));
        CUDA(cudaFreeHost(u_control));
        CUDA(cudaFreeHost(u_high_start));
        CUDA(cudaFreeHost(u_working));
        CUDA(cudaFree(u_output));
        CUDA(cudaFree(u_arena));
    }
}

// ============================================================================
// Real 2D Tiled SGEMM (64 KB SMEM/CTA) Spatial Preemption + O1 Overhead
// ============================================================================
constexpr int kSgemmTile = 32;
constexpr int kSgemmThreads = 256; // 32x8 threads; each thread computes 4 rows in a column
constexpr U32 kSgemmKTiles = 512;  // K = 512 * 32 = 16384
constexpr U32 kSgemmPollTiles = 4;

struct SgemmState {
    const float *a;
    const float *b;
    float *c;
    U32 *request;
    U64 *request_seen;
    U32 *next_tile;
    U32 *image;               // [slot][4 * 256 FP32 accumulators]
    U32 *slots;
    U32 *owned;
    Observation *observe;
    U32 slot_count;
};

__global__ __launch_bounds__(kSgemmThreads, 1)
void tiled_sgemm_checkpoint_kernel(SgemmState st, U64 *high_start)
{
    const U32 tid = threadIdx.x;
    const U32 bid = blockIdx.x;
    if (high_start) {
        if (tid == 0) stamp(high_start + bid);
        return;
    }

    const U32 first_tile = st.next_tile[bid];
    if (first_tile == kSgemmKTiles || __syncthreads_or(tid == 0 && gpu_read(st.request)))
        return;

    extern __shared__ float smem[];
    float *tile_a = smem;                       // 32x32 floats
    float *tile_b = smem + kSgemmTile * kSgemmTile; // 32x32 floats
    __shared__ U32 save_slot;

    const U32 tx = tid & 31U;
    const U32 ty = tid >> 5; // 0..7
    const U32 image_words = 4 * kSgemmThreads;
    const U32 restore_slot = first_tile ? st.slots[bid] : 0U;
    const U32 *restore = st.image + restore_slot * image_words;

    float acc[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        acc[r] = first_tile ? __uint_as_float(restore[r * kSgemmThreads + tid]) : 0.0f;
    }
    __syncthreads();
    if (first_tile && tid == 0) atomicExch(st.owned + restore_slot, 0U);

    for (U32 kt = first_tile; kt < kSgemmKTiles;) {
        // Load 32x32 tiles of A and B into 64 KB Shared Memory
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            U32 row = ty + r * 8;
            tile_a[row * kSgemmTile + tx] = st.a[(bid * kSgemmTile + row) * 1024 + ((kt * kSgemmTile + tx) & 1023U)];
            tile_b[row * kSgemmTile + tx] = st.b[((kt * kSgemmTile + row) & 1023U) * 1024 + tx];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kSgemmTile; ++k) {
            float b_val = tile_b[k * kSgemmTile + tx];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[r] = fmaf(tile_a[(ty + r * 8) * kSgemmTile + k], b_val, acc[r]);
            }
        }
        ++kt;
        __syncthreads();

        if (tid == 0 && kt == first_tile + 1) {
            st.observe[bid].step_at_entry = first_tile;
            stamp(&st.observe[bid].working);
        }
        if (kt % kSgemmPollTiles || kt == kSgemmKTiles)
            continue;
        if (!__syncthreads_or(tid == 0 && gpu_read(st.request)))
            continue;

        if (tid == 0 && gpu_read(st.request) == 1U && atomicCAS(st.request, 1U, 3U) == 1U) {
            stamp(st.request_seen);
        }
        if (tid == 0) {
            U32 slot = bid % st.slot_count;
            save_slot = st.slot_count;
            for (U32 i = 0; i < st.slot_count; ++i) {
                if (atomicCAS(st.owned + slot, 0U, 1U) == 0U) { save_slot = slot; break; }
                if (++slot == st.slot_count) slot = 0U;
            }
        }
        __syncthreads();
        if (save_slot == st.slot_count) continue;

        U32 *dst = st.image + save_slot * image_words;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            dst[r * kSgemmThreads + tid] = __float_as_uint(acc[r]);
        }
        __syncthreads();
        if (tid == 0) {
            st.slots[bid] = save_slot;
            st.next_tile[bid] = kt;
        }
        return;
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        st.c[(bid * kSgemmTile + ty + r * 8) * kSgemmTile + tx] = acc[r];
    }
    __syncthreads();
    if (tid == 0) st.next_tile[bid] = kSgemmKTiles;
}

void run_sgemm(xsched::cuda::TrapSm120Context &trap, CUcontext ctx, const cudaDeviceProp &gpu, int rounds)
{
    const U32 shared_bytes = 65536;
    CUDA(cudaFuncSetAttribute(tiled_sgemm_checkpoint_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    CUDA(cudaFuncSetAttribute(tiled_sgemm_checkpoint_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    cudaFuncAttributes attr{};
    CUDA(cudaFuncGetAttributes(&attr, tiled_sgemm_checkpoint_kernel));
    const int resident = gpu.multiProcessorCount; // 1 CTA/SM (64 KB SMEM/CTA)
    const int grid = resident * 3;                // 90 CTAs (3 waves)
    const size_t cta_image_bytes = 4 * kSgemmThreads * sizeof(U32);
    printf("\n[sgemm] regs/thread=%d local/thread=%zu shared/CTA=%zu resident=%d (1/SM) grid=%d"
           " blocker=shared (2D Tiled FP32 SGEMM) arena=%zu KiB\n",
           attr.numRegs, attr.localSizeBytes, shared_bytes + attr.sharedSizeBytes + gpu.reservedSharedMemPerBlock,
           resident, grid, resident * cta_image_bytes / 1024);

    const size_t a_elems = size_t(grid) * kSgemmTile * 1024;
    const size_t b_elems = 1024 * 1024;
    const size_t c_elems = size_t(grid) * kSgemmTile * kSgemmTile;
    std::vector<float> h_a(a_elems), h_b(b_elems), h_ref(c_elems), h_out(c_elems);
    for (size_t i = 0; i < a_elems; ++i) h_a[i] = float((i * 17 + 3) % 97) * 0.01f;
    for (size_t i = 0; i < b_elems; ++i) h_b[i] = float((i * 31 + 7) % 89) * 0.01f;

    SgemmState st{};
    st.a = device_buffer<float>(a_elems);
    st.b = device_buffer<float>(b_elems);
    st.c = device_buffer<float>(c_elems);
    st.request = device_buffer<U32>(1);
    st.next_tile = device_buffer<U32>(grid);
    st.image = device_buffer<U32>(resident * 4 * kSgemmThreads);
    st.slots = device_buffer<U32>(grid);
    st.owned = device_buffer<U32>(resident);
    st.slot_count = resident;
    CUDA(cudaMemcpy(const_cast<float *>(st.a), h_a.data(), a_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(const_cast<float *>(st.b), h_b.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice));

    auto *request_seen = mapped_buffer(&st.request_seen, 1);
    auto *telemetry = mapped_buffer(&st.observe, grid);
    U32 *gpu_control;
    auto *control = mapped_buffer(&gpu_control, 4 + 2 * grid);
    U64 *gpu_high_start;
    auto *high_start = mapped_buffer(&gpu_high_start, resident);

    int low_priority, high_priority;
    CUDA(cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority));
    cudaStream_t low, high;
    CUDA(cudaStreamCreateWithPriority(&low, cudaStreamNonBlocking, low_priority));
    CUDA(cudaStreamCreateWithPriority(&high, cudaStreamNonBlocking, high_priority));

    auto reset = [&] {
        __atomic_store_n(control, 0, __ATOMIC_RELEASE);
        __atomic_store_n(request_seen, 0, __ATOMIC_RELEASE);
        memset(telemetry, 0, grid * sizeof(Observation));
        memset(high_start, 0, resident * sizeof(U64));
        CUDA(cudaMemsetAsync(st.request, 0, sizeof(U32), low));
        CUDA(cudaMemsetAsync(st.next_tile, 0, grid * sizeof(U32), low));
        CUDA(cudaMemsetAsync(st.owned, 0, resident * sizeof(U32), low));
        CUDA(cudaMemsetAsync(st.c, 0, c_elems * sizeof(float), low));
    };

    // Measure O1 normal-execution overhead: unhooked vs hooked without preemption
    require(trap.RestoreOriginalTrapHandler(), "could not restore trap handler for O1 baseline");
    reset();
    tiled_sgemm_checkpoint_kernel<<<grid, kSgemmThreads, shared_bytes, low>>>(st, nullptr);
    await_stream(low);
    CUDA(cudaMemcpy(h_ref.data(), st.c, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<double> unhooked_times, hooked_times;
    for (int r = 0; r < rounds; ++r) {
        reset();
        await_stream(low);
        const U64 t0 = checkpoint_ptimer();
        tiled_sgemm_checkpoint_kernel<<<grid, kSgemmThreads, shared_bytes, low>>>(st, nullptr);
        await_stream(low);
        unhooked_times.push_back(delta_us(t0, checkpoint_ptimer()));
    }

    require(trap.InstallFineGrainedTrapHandler(ctx), "could not reinstall trap handler");
    CUfunction func = nullptr;
    CUDA(cudaGetFuncBySymbol(&func, reinterpret_cast<const void *>(tiled_sgemm_checkpoint_kernel)));
    require(trap.BindKernelCheckpointRequest(func, reinterpret_cast<uint64_t>(gpu_control),
                                             reinterpret_cast<uint64_t>(st.request)),
            "could not bind SGEMM checkpoint request");

    for (int r = 0; r < rounds; ++r) {
        reset();
        await_stream(low);
        const U64 t0 = checkpoint_ptimer();
        tiled_sgemm_checkpoint_kernel<<<grid, kSgemmThreads, shared_bytes, low>>>(st, nullptr);
        await_stream(low);
        hooked_times.push_back(delta_us(t0, checkpoint_ptimer()));
    }
    std::sort(unhooked_times.begin(), unhooked_times.end());
    std::sort(hooked_times.begin(), hooked_times.end());
    const double u_p50 = unhooked_times[(unhooked_times.size() - 1) / 2];
    const double h_p50 = hooked_times[(hooked_times.size() - 1) / 2];
    printf("  O1 normal-exec overhead    unhooked p50=%.2f us, hooked p50=%.2f us (delta=%+.2f%%)\n",
           u_p50, h_p50, (h_p50 - u_p50) * 100.0 / u_p50);

    std::vector<double> signal, drain, admission, resume_all;
    for (int r = 0; r < rounds; ++r) {
        reset();
        await_stream(low);
        tiled_sgemm_checkpoint_kernel<<<grid, kSgemmThreads, shared_bytes, low>>>(st, nullptr);
        for (int cycle = 0; cycle < 2; ++cycle) {
            wait_until([&] {
                int count = 0;
                for (int b = 0; b < grid; ++b) count += observe(&telemetry[b].working) != 0;
                return count >= resident;
            }, "SGEMM did not saturate all SMs");
            memset(high_start, 0, resident * sizeof(U64));
            tiled_sgemm_checkpoint_kernel<<<resident, kSgemmThreads, shared_bytes, high>>>(SgemmState{}, gpu_high_start);
            const U64 req_t = checkpoint_ptimer();
            request_checkpoint(trap, control, request_seen);
            await_stream(low);
            const U64 ret_t = checkpoint_ptimer();
            await_stream(high);

            U64 earliest_high = ~0ULL;
            for (int b = 0; b < resident; ++b) {
                if (observe(high_start + b)) earliest_high = std::min(earliest_high, observe(high_start + b));
            }
            signal.push_back(delta_us(req_t, observe(request_seen)));
            drain.push_back(delta_us(req_t, ret_t));
            admission.push_back(delta_us(req_t, earliest_high));

            memset(telemetry, 0, grid * sizeof(Observation));
            const U64 res_t = checkpoint_ptimer();
            __atomic_store_n(control, 0, __ATOMIC_RELEASE);
            __atomic_store_n(request_seen, 0, __ATOMIC_RELEASE);
            CUDA(cudaMemsetAsync(st.request, 0, sizeof(U32), low));
            tiled_sgemm_checkpoint_kernel<<<grid, kSgemmThreads, shared_bytes, low>>>(st, nullptr);

            std::vector<U32> next_vec(grid);
            CUDA(cudaMemcpy(next_vec.data(), st.next_tile, grid * sizeof(U32), cudaMemcpyDeviceToHost));
            wait_until([&] {
                for (int b = 0; b < grid; ++b)
                    if (next_vec[b] && next_vec[b] != kSgemmKTiles && !observe(&telemetry[b].working))
                        return false;
                return true;
            }, "saved SGEMM CTA did not resume");
            U64 latest_res = 0;
            for (int b = 0; b < grid; ++b) {
                if (next_vec[b] && next_vec[b] != kSgemmKTiles)
                    latest_res = std::max(latest_res, observe(&telemetry[b].working));
            }
            resume_all.push_back(delta_us(res_t, latest_res));
        }
        await_stream(low);
        CUDA(cudaMemcpy(h_out.data(), st.c, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
        require(h_out == h_ref, "preempted SGEMM output differs from uninterrupted reference");
    }
    stats("trap to first safe point", signal);
    stats("trap to low stream retired", drain);
    stats("trap to first high CTA", admission);
    stats("relaunch to all restored", resume_all);
    printf("  PASS: bit-exact 2D Tiled SGEMM across %d jobs (%d spatial evictions, 0 FP32 bit errors)\n",
           rounds, 2 * rounds);

    require(trap.BindKernelPreemptBuffer(func, 0, 0), "could not unbind SGEMM kernel");
    CUDA(cudaStreamDestroy(high));
    CUDA(cudaStreamDestroy(low));
    CUDA(cudaFreeHost(high_start));
    CUDA(cudaFreeHost(control));
    CUDA(cudaFreeHost(telemetry));
    CUDA(cudaFreeHost(request_seen));
    CUDA(cudaFree(st.owned));
    CUDA(cudaFree(st.slots));
    CUDA(cudaFree(st.image));
    CUDA(cudaFree(st.next_tile));
    CUDA(cudaFree(st.request));
    CUDA(cudaFree(st.c));
    CUDA(cudaFree(const_cast<float *>(st.b)));
    CUDA(cudaFree(const_cast<float *>(st.a)));
}
} // namespace

int main(int argc, char **argv)
{
    setbuf(stdout, nullptr);
    const std::string profile = argc > 1 ? argv[1] : "all";
    const int rounds = argc > 2 ? atoi(argv[2]) : 5;
    require(argc <= 3 && rounds > 0 && rounds <= 200 &&
            (profile == "all" || profile == "registers" || profile == "shared" ||
             profile == "warps" || profile == "trapframe" || profile == "sgemm"),
            "usage: checkpoint_bench [all|registers|shared|warps|trapframe|sgemm] [rounds 1..200]");

    cudaDeviceProp gpu{};
    CUDA(cudaGetDeviceProperties(&gpu, 0));
    char bus_id[32]{};
    CUDA(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), 0));
    require(gpu.major == 12 && gpu.minor == 0 && strstr(gpu.name, "RTX 5060") &&
            std::string(bus_id) == "0000:01:00.0", "unexpected GPU or BAR0 PCI address");
    CUDA(cudaSetDevice(0));
    CUDA(cudaFree(nullptr));

    require(bar0_open() == 0, "unable to map BAR0");
    CUcontext ctx = nullptr;
    require(cuCtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx, "no active CUDA context");
    auto &trap = xsched::cuda::TrapSm120Context::Instance();
    require(trap.InstallFineGrainedTrapHandler(ctx), "could not install the SM120 trap hook");
    if (profile == "all" || profile == "registers") run<144, 1>(trap, "registers", gpu, 1024, rounds);
    if (profile == "all" || profile == "shared") run<4, 1>(trap, "shared", gpu, 65536, rounds);
    if (profile == "all" || profile == "warps") run<1, 6>(trap, "warps", gpu, 1024, rounds);
    if (profile == "all" || profile == "trapframe") run_trapframe(trap, gpu, rounds);
    if (profile == "all" || profile == "sgemm") run_sgemm(trap, ctx, gpu, rounds);

    require(trap.RestoreOriginalTrapHandler(), "could not restore the original SM120 trap handler");
    CUDA(cudaDeviceReset());
}
