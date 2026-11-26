#!/usr/bin/env python3
"""XQueue idle-state check (Phase 3.4-2): after a stream synchronize the queue must report Idle soon.

The local / global schedulers learn that a queue went idle from the Ready -> Idle transition (and
XQueueIdleEvent) made by the launch worker. With the caller-side WaitAll fast path (PERF-1) the
synchronize returns on the caller thread and the worker makes the transition shortly afterwards,
after its own background synchronization (measured ~20 us). This test checks that every
synchronize is followed by Idle within IDLE_LIMIT_MS, and that new work makes the queue Ready again:
  single   one kernel per call + synchronize (direct submit + caller-side WaitAll)
  burst    8 kernels per call + synchronize
  repeat   200 single-kernel calls
for Level-1 and Level-3 queues. Prints the time-to-Idle distribution.

usage: test_xqueue_idle_p34.py
"""
import ctypes
import sys
import time

import torch

IDLE, READY = 1, 2
IDLE_LIMIT_MS = 50.0


def main():
    pre = ctypes.CDLL("libpreempt.so", mode=ctypes.RTLD_GLOBAL)
    hal = ctypes.CDLL("libhalcuda.so", mode=ctypes.RTLD_GLOBAL)
    hal.CudaQueueCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p]
    pre.XQueueCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.c_int, ctypes.c_int64]
    pre.XQueueQuery.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_int)]
    pre.XQueueDestroy.argtypes = [ctypes.c_uint64]
    pre.HwQueueDestroy.argtypes = [ctypes.c_uint64]
    torch.cuda.init()
    x = torch.randn(1 << 20, device="cuda")
    ok = True
    for level in (1, 3):
        s = torch.cuda.Stream()
        hwq, xq = ctypes.c_uint64(), ctypes.c_uint64()
        assert hal.CudaQueueCreate(ctypes.byref(hwq), ctypes.c_void_p(s.cuda_stream)) == 0
        assert pre.XQueueCreate(ctypes.byref(xq), hwq, level, 0) == 0

        def state():
            st = ctypes.c_int(0)
            assert pre.XQueueQuery(xq, ctypes.byref(st)) == 0
            return st.value

        def us_to_idle():
            """Microseconds until the queue reports Idle (None if not within IDLE_LIMIT_MS)."""
            t0 = time.perf_counter()
            while state() != IDLE:
                if (time.perf_counter() - t0) * 1e3 > IDLE_LIMIT_MS:
                    return None
            return (time.perf_counter() - t0) * 1e6

        results, lat = {}, []
        with torch.cuda.stream(s):
            for _ in range(3):                     # warm-up (instrumentation, first launches)
                y = x * 2 + 1
                s.synchronize()
            y = x * 2 + 1
            s.synchronize()
            t = us_to_idle()
            results["single"] = t is not None
            for _ in range(8):
                y = y * 2 + 1
            s.synchronize()
            t = us_to_idle()
            results["burst"] = t is not None
            for _ in range(200):
                y = x * 2 + 1
                s.synchronize()
                lat.append(us_to_idle())
            never = sum(v is None for v in lat)
            results["repeat"] = never == 0
            # A new long-running submission must make the queue Ready again, then Idle after sync.
            big = torch.randn(4096, 4096, device="cuda")
            torch.cuda.synchronize()
            z = big @ big
            results["ready-again"] = state() == READY
            s.synchronize()
            results["idle-again"] = us_to_idle() is not None
        done = sorted(v for v in lat if v is not None)
        print(f"[XQ-IDLE lv{level} latency] idle within {IDLE_LIMIT_MS:.0f} ms after {200 - never}/200 synchronizes; "
              f"time to idle p50 {done[len(done) // 2] if done else float('nan'):.1f} us, "
              f"max {done[-1] if done else float('nan'):.1f} us")
        for k, v in results.items():
            print(f"[XQ-IDLE lv{level} {k}] {'PASS' if v else 'FAIL'}")
            ok &= v
        pre.XQueueDestroy(xq.value)
        pre.HwQueueDestroy(hwq.value)
    print("[XQ-IDLE all]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
