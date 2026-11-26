#!/usr/bin/env python3
"""DEF-31: a non-killable command (whole CUDA graph) must never be executed twice by a Level-3 restore.

Each iteration launches, on a Level-3 XQueue, a killable kernel (sets the trap engine's active kernel
index) and then a CUDA graph that runs a few milliseconds of matmuls and increments a counter. The
host then pauses briefly, so the graph is often the last launched command when the preemptor's
suspension arrives (Interrupt() correctly falls back to the graph boundary). After N iterations the
counter must be exactly N; a restore that relaunches the graph increments it twice.

Run with XSCHED_CUDA_GRAPH_FLATTEN=0 (graph launched whole: the DEF-31 path) and =1 (flattened).

--stale-params (DEF-32): the graph's GEMM kernel is also launched eagerly (killable, Mode 5 Guardian
parameters) on the Level-3 XQueue, before the capture and right before every replay, and every
iteration queues a killable kernel behind the graph, so suspensions pulse while the graph runs. The graph's
matmul result is compared with a native reference: captured nodes that kept a stale Mode 5 parameter
window would be trapped / drained like that old launch and never restored.
usage: test_graph_relaunch_p35.py [--iters 60] [--n 2048] [--stale-params]
"""
import argparse, os, random, sys, threading, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--chain", type=int, default=8)
    ap.add_argument("--stale-params", action="store_true")
    a = ap.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    A = torch.randn(a.n, a.n, device="cuda") / a.n ** 0.5
    X = torch.randn(a.n, a.n, device="cuda") / a.n ** 0.5
    cnt = torch.zeros(1, device="cuda", dtype=torch.int64)
    Yg = torch.empty_like(X)
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        t = X
        for _ in range(a.chain):
            t = torch.mm(A, t)
        ref = t.clone()
    torch.cuda.synchronize()
    lp = torch.cuda.Stream()
    xs = hwq = xq = None
    if a.stale_params:
        from bench_mp_serving_p34 import Bridge
        xs = Bridge()
        hwq, xq = xs.create(lp, 3)
        with torch.cuda.stream(lp):      # Mode 5 launches of the same kernels before the capture
            for _ in range(3):
                t = X
                for _ in range(a.chain):
                    t = torch.mm(A, t)
        lp.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            t = X
            for _ in range(a.chain):
                t = torch.mm(A, t)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        t = X
        for _ in range(a.chain - 1):
            t = torch.mm(A, t)
        torch.mm(A, t, out=Yg)
        cnt.add_(1)
    torch.cuda.synchronize()
    cnt.zero_()
    t0 = time.perf_counter()
    with torch.cuda.stream(lp):
        g.replay()
    lp.synchronize()
    g_ms = (time.perf_counter() - t0) * 1e3
    cnt.zero_()
    torch.cuda.synchronize()

    if xs is None:
        from bench_mp_serving_p34 import Bridge
        xs = Bridge()
        hwq, xq = xs.create(lp, 3)
    Y = torch.empty(512, 512, device="cuda")
    Y2 = torch.empty_like(X)
    B = torch.randn(512, 512, device="cuda")
    torch.cuda.synchronize()          # T-26: inputs from the legacy stream before use on `lp`
    stop = threading.Event()
    n_susp = [0]

    def preemptor():
        rng = random.Random(5)
        while not stop.is_set():
            time.sleep(rng.uniform(0.2, 1.2) * g_ms / 1e3)
            xs.pre.XQueueSuspend(xq, 1)
            time.sleep(0.002)
            xs.pre.XQueueResume(xq, 0)
            n_susp[0] += 1

    th = threading.Thread(target=preemptor, daemon=True)
    th.start()
    rng = random.Random(9)
    bad_out = 0
    for _ in range(a.iters):
        with torch.cuda.stream(lp):
            Yg.zero_()
            if a.stale_params:
                torch.mm(A, X, out=Y2)     # same GEMM kernel as in the graph, launched killable
            torch.mm(B, B, out=Y)          # killable kernel
            g.replay()                     # whole graph (flatten=0): non-killable, not idempotent
            if a.stale_params:
                torch.mm(B, B, out=Y)      # killable kernel queued behind the running graph
        if a.stale_params:
            lp.synchronize()
            if not torch.equal(Yg, ref):
                bad_out += 1
        else:
            time.sleep(rng.uniform(0.0, 1.0) * g_ms / 1e3)
    lp.synchronize()
    stop.set()
    th.join()
    torch.cuda.synchronize()
    got = int(cnt.item())
    xs.destroy(hwq, xq)
    ok = got == a.iters and bad_out == 0
    tag = " stale-params" if a.stale_params else ""
    print(f"[GRAPH-RELAUNCH flatten={os.environ.get('XSCHED_CUDA_GRAPH_FLATTEN', '1')}{tag}] {'PASS' if ok else 'FAIL'} | "
          f"graph {g_ms:.1f} ms | iters {a.iters} counter {got} | wrong graph outputs {bad_out} | "
          f"suspensions {n_susp[0]}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
