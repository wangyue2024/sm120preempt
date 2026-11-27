#!/usr/bin/env python3
"""Phase 3.5: cross-stream waits on XQueue events under Level-3 preemption.

An LP stream bound to an XQueue computes Y (zeroed first, then a chain of matmuls), records an
event E; an ordinary (non-XQueue) copy stream waits on E and copies Y to Z. A preemptor suspends the
XQueue at a random point of the chain (Level-3 trap), holds it, and resumes. Z must equal the golden
result in every iteration: an event that signals when the trapped kernel exits early (before the
XQueue relaunches the remaining work) would let the copy read a partial Y.

XSCHED_CUDA_ASYNC_EVENT_WAIT: 0 = blocking host wait (old), 1 = default (Level-1: GPU wait after the
record is launched; Level-2/3: the waiting stream is promoted to an internal Level-1 XQueue whose
worker waits), 2 = UNSAFE plain GPU wait (shows the hazard).
Also reports how long the caller of wait_event is blocked. --module-load loads a new CUDA module
(a fresh jiterator kernel -> cuModuleLoadData) right after each wait: the driver holds a lock while
it waits for the GPU to go idle, which deadlocked a design in which the GPU waited on host progress.

usage: test_event_bridge_p35.py [--level 3] [--iters 40] [--expect-fail]
"""
import argparse, ctypes, os, random, sys, threading, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def stats():
    try:
        f = ctypes.CDLL("libcuda.so.1").XCudaEventWaitStats
    except (OSError, AttributeError):
        return None
    f.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int]
    b = (ctypes.c_uint64 * 4)()
    f(b, 4, 0)
    return {"lv1_gpu_wait": b[0], "promoted_waits": b[1], "blocking": b[2], "promotions": b[3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--chain", type=int, default=3)
    ap.add_argument("--expect-fail", action="store_true", help="mode 2: pass if corruption is observed")
    ap.add_argument("--no-preempt", action="store_true", help="control run without suspensions")
    ap.add_argument("--module-load", action="store_true", help="load a new CUDA module after each wait")
    a = ap.parse_args()
    mode = os.environ.get("XSCHED_CUDA_ASYNC_EVENT_WAIT", "1")
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    A = torch.randn(a.n, a.n, device="cuda") / a.n ** 0.5
    X = torch.randn(a.n, a.n, device="cuda") / a.n ** 0.5
    Y = torch.empty_like(X)
    Z = torch.empty_like(X)
    lp, cp, side = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    small = torch.zeros(16, device="cuda")
    # T-26: the inputs are produced on the legacy default stream and PyTorch's pool streams are
    # non-blocking: without this the first native run on `lp` can race with randn (seen under GPU
    # contention as a non-deterministic "gold", i.e. every later comparison failing).
    torch.cuda.synchronize()

    def lp_work():
        Y.zero_()
        t = X
        for i in range(a.chain):
            t = torch.mm(A, t) if i < a.chain - 1 else torch.mm(A, t, out=Y)

    with torch.cuda.stream(lp):
        lp_work()
    torch.cuda.synchronize()
    gold = Y.clone()
    # T-29 (OBS-20 root cause): the clone runs on the legacy default stream; without this the next
    # Y.zero_() on the non-blocking LP stream can overtake it under GPU contention (all-zero golden,
    # 20/20 under a background GEMM load, 0/20 with the synchronize).
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(lp):
        lp_work()
    torch.cuda.synchronize()
    work_ms = (time.perf_counter() - t0) * 1e3
    native_det = torch.equal(Y, gold)

    from bench_mp_serving_p34 import Bridge
    xs = Bridge()
    hwq, xq = xs.create(lp, a.level)
    with torch.cuda.stream(lp):
        lp_work()          # first launches (instrumentation) outside the measured iterations
    torch.cuda.synchronize()

    rng = random.Random(3)
    bad, block_us, diag = 0, [], None
    for it in range(a.iters):
        Z.fill_(float("nan"))
        torch.cuda.synchronize()
        delay = rng.uniform(0.1, 0.9) * work_ms / 1e3
        done = threading.Event()

        def preemptor():
            time.sleep(delay)
            if done.is_set():
                return
            xs.pre.XQueueSuspend(xq, 1)
            time.sleep(0.004)
            xs.pre.XQueueResume(xq, 0)

        th = threading.Thread(target=preemptor)
        if a.no_preempt:
            done.set()
        th.start()
        with torch.cuda.stream(lp):
            lp_work()
            ev = torch.cuda.Event()
            ev.record(lp)
        t1 = time.perf_counter()
        cp.wait_event(ev)
        block_us.append((time.perf_counter() - t1) * 1e6)
        if a.module_load:
            fn = torch.cuda.jiterator._create_jit_fn(
                "template <typename T> T jit_add_%d(T x) { return x + T(%d); }" % (it, it))
            with torch.cuda.stream(side):
                r = fn(small)
            side.synchronize()
            assert float(r[0]) == float(it)
        with torch.cuda.stream(cp):
            Z.copy_(Y, non_blocking=True)
        cp.synchronize()
        done.set()
        th.join()
        torch.cuda.synchronize()
        if not torch.equal(Z, gold):
            bad += 1
            if diag is None:
                d = (Z - gold).abs()
                diag = (f"first bad iter {it}: {int((Z != gold).sum())}/{Z.numel()} elements differ, "
                        f"max |diff| {float(d.nan_to_num(float('inf')).max()):.3g}, nan {int(Z.isnan().sum())}, "
                        f"Y==gold now {torch.equal(Y, gold)}")
    xs.destroy(hwq, xq)
    block_us.sort()
    st = stats()
    pulses = -1
    try:
        f = ctypes.CDLL("libhalcuda.so").XSm120Pulses
        f.restype = ctypes.c_uint64
        pulses = int(f())
    except Exception:
        pass
    skipped = False
    if a.expect_fail:
        # The hazard needs real evictions. With pulses withheld (DEF-40: e.g. the desktop's contexts on
        # a display GPU) the queue behaves like Level-1 and the hazard cannot be demonstrated.
        skipped = bad == 0 and pulses == 0 and a.level >= 3
        ok = bad > 0 or skipped
    else:
        ok = bad == 0
    print(f"[EVENT-BRIDGE lv{a.level} mode={mode}{' no-preempt' if a.no_preempt else ''}{' module-load' if a.module_load else ''}] {'SKIP (no pulses fired)' if skipped else ('PASS' if ok else 'FAIL')} | iters {a.iters} corrupted {bad} | pulses {pulses} | "
          f"work {work_ms:.1f} ms | wait_event caller blocked p50 {block_us[len(block_us)//2]:.0f} us "
          f"max {block_us[-1]:.0f} us | native deterministic {native_det} | {st}", flush=True)
    if diag:
        print("  " + diag, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
