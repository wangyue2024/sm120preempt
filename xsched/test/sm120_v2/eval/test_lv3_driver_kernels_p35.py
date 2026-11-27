#!/usr/bin/env python3
"""OBS-21 investigation: are driver-internal kernels (large memsets / device-to-device copies) safe on
a Level-3 XQueue under frequent preemption?

The driver runs large / strided memsets and device-to-device copies as SM kernels (ce_vs_sm_probe).
They are not instrumented: if a BAR0 pulse traps them, the trap handler must return through its
unmanaged-kernel path; if it evicted them instead, the operation would be cut short and - memory
commands are never relaunched after a trap - silently incomplete.

Each iteration: cuMemsetD8Async(B, v, N) then cuMemcpyDtoDAsync(C, B, N) on the Level-3 queue's
stream, then a full check of B and C on the GPU. A preemptor thread suspends / resumes the queue.

usage: test_lv3_driver_kernels_p35.py [--mib 1024] [--iters 200] [--gap-us 400] [--level 3]
"""
import argparse, ctypes, os, random, sys, threading, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--gap-us", type=float, default=400.0)
    ap.add_argument("--hold-us", type=float, default=150.0)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--ops", default="memset,copy", help="driver operations per iteration")
    a = ap.parse_args()
    torch.cuda.set_device(0)
    n = a.mib << 20
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        b = torch.empty(n, dtype=torch.uint8, device="cuda")
        c = torch.empty(n, dtype=torch.uint8, device="cuda")
        bad_b = torch.zeros(1, dtype=torch.int64, device="cuda")
        bad_c = torch.zeros(1, dtype=torch.int64, device="cuda")
    s.synchronize()
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuMemsetD8Async.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t, ctypes.c_void_p]
    cu.cuMemcpyDtoDAsync_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]

    ops = a.ops.split(",")

    def one(v):
        if "memset" in ops:
            assert cu.cuMemsetD8Async(b.data_ptr(), v, n, s.cuda_stream) == 0
        else:
            with torch.cuda.stream(s):
                b.fill_(v)
        if "copy" in ops:
            assert cu.cuMemcpyDtoDAsync_v2(c.data_ptr(), b.data_ptr(), n, s.cuda_stream) == 0
        else:
            with torch.cuda.stream(s):
                torch.add(b, 0, out=c)          # a kernel (copy_ would be a driver copy)
        with torch.cuda.stream(s):
            bad_b.add_(torch.count_nonzero(b != v))
            bad_c.add_(torch.count_nonzero(c != v))

    for v in (1, 2):                                   # native check first (no XQueue yet)
        one(v)
    s.synchronize()
    native_ok = int(bad_b.item()) == 0 and int(bad_c.item()) == 0
    bad_b.zero_(); bad_c.zero_(); torch.cuda.synchronize()

    from bench_mp_serving_p34 import Bridge
    xs = Bridge()
    hwq, xq = xs.create(s, a.level)
    stop, n_pre = threading.Event(), [0]

    def preemptor():
        rng = random.Random(11)
        while not stop.is_set():
            time.sleep(rng.uniform(0.3, 1.7) * a.gap_us / 1e6)
            xs.pre.XQueueSuspend(xq, 1)
            time.sleep(a.hold_us / 1e6)
            xs.pre.XQueueResume(xq, 0)
            n_pre[0] += 1

    th = threading.Thread(target=preemptor, daemon=True)
    th.start()
    t0 = time.perf_counter()
    for it in range(a.iters):
        one(3 + it % 250)
    s.synchronize()
    wall = time.perf_counter() - t0
    stop.set()
    th.join()
    bb, bc = int(bad_b.item()), int(bad_c.item())
    xs.destroy(hwq, xq)
    try:
        f = ctypes.CDLL("libhalcuda.so").XSm120Pulses
        f.restype = ctypes.c_uint64
        pulses = int(f())
    except Exception:
        pulses = -1
    ok = native_ok and bb == 0 and bc == 0
    print(f"[LV3-DRIVER-KERNELS lv{a.level} ops={a.ops}] {'PASS' if ok else 'FAIL'} | iters {a.iters} x {a.mib} MiB memset+copy | "
          f"wrong bytes memset {bb} copy {bc} | suspensions {n_pre[0]} pulses {pulses} | wall {wall:.2f}s | "
          f"native ok {native_ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
