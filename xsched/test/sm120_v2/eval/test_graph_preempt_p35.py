#!/usr/bin/env python3
"""DEF-09 (tier-1 item 1): preemption inside CUDA graphs.

A whole training step (MLP forward + backward + capturable AdamW, i.e. non-idempotent in-place
updates) is captured into one CUDA graph (PyTorch "whole network capture"). The graph is replayed
N times on a low-priority stream bound to an XQueue while a preemptor thread repeatedly suspends
the queue, runs a high-priority request on a priority stream and resumes. The final parameters and
optimizer state must be bit-identical to N native replays from the same initial state.

Run once with XSCHED_CUDA_GRAPH_FLATTEN=1 (graph replays are flattened into their kernels and each
kernel is Level-3 preemptible) and once with =0 (the graph is launched as a whole: Level-1 graph
boundary) to compare suspend / HP latency. Also reports the replay time without preemption (cost of
flattening) and the flattening counters.

usage: test_graph_preempt_p35.py [--level 3] [--replays 60] [--preempts 20] [--layers 12] [--dim 4096]
"""
import argparse
import ctypes
import json
import os
import random
import statistics
import sys
import threading
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))] if s else None


def flatten_stats(reset=False):
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        f = lib.XCudaGraphFlattenStats
    except (OSError, AttributeError):
        return None
    f.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int]
    buf = (ctypes.c_uint64 * 5)()
    f(buf, 5, 1 if reset else 0)
    return {"flat_launches": buf[0], "whole_launches": buf[1], "flat_cmds": buf[2], "unflattenable": buf[3],
            "kept_whole_short": buf[4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--replays", type=int, default=60)
    ap.add_argument("--preempts", type=int, default=20)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    flatten = os.environ.get("XSCHED_CUDA_GRAPH_FLATTEN", "1")
    if os.environ.get("XSCHED_CUDA_GRAPH_FLATTEN_MIN_US"):
        flatten += f"/min{os.environ['XSCHED_CUDA_GRAPH_FLATTEN_MIN_US']}us"

    layers = []
    for _ in range(a.layers):
        layers += [torch.nn.Linear(a.dim, a.dim), torch.nn.GELU()]
    model = torch.nn.Sequential(*layers).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, capturable=True)
    x = torch.randn(a.batch, a.dim, device="cuda") * 0.1

    # Warm up on a side stream (required before capture), then capture one training step.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            opt.zero_grad(set_to_none=False)
            model(x).pow(2).mean().backward()
            opt.step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    # PyTorch whole-network-capture recipe: grads set to None before capture, so the captured
    # backward writes fresh gradients into graph-owned buffers on every replay (no accumulation).
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    state = ([p for p in model.parameters()] + [p.grad for p in model.parameters() if p.grad is not None]
             + [t for st in opt.state.values() for t in st.values() if torch.is_tensor(t)])
    init = [t.detach().clone() for t in state]

    def restore():
        with torch.no_grad():
            for t, s0 in zip(state, init):
                t.copy_(s0)
        torch.cuda.synchronize()

    lp = torch.cuda.Stream()
    hp = torch.cuda.Stream(priority=-1)
    w_hp = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
    x_hp = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)

    def hp_fn():
        return torch.mm(torch.mm(x_hp, w_hp), w_hp)

    with torch.cuda.stream(hp):
        hp_gold = hp_fn().clone()
    hp.synchronize()

    def run_replays(n):
        with torch.cuda.stream(lp):
            for _ in range(n):
                g.replay()
        lp.synchronize()

    # Native golden (no XQueue) and its determinism, plus native replay time.
    restore()
    t0 = time.perf_counter(); run_replays(a.replays); native_ms = (time.perf_counter() - t0) * 1e3 / a.replays
    gold = [t.detach().clone() for t in state]
    restore()
    run_replays(a.replays)
    deterministic = all(torch.equal(t, g0) for t, g0 in zip(state, gold))

    res = {"level": a.level, "flatten": flatten, "replays": a.replays, "layers": a.layers, "dim": a.dim,
           "batch": a.batch, "native_replay_ms": native_ms, "deterministic": deterministic, "error": None}
    from bench_mp_serving_p34 import Bridge
    xs = Bridge()
    hwq, xq = xs.create(lp, a.level)
    flatten_stats(reset=True)

    # Replay time on the XQueue without preemption (cost of flattening / XQueue submission).
    restore()
    run_replays(3)   # first launches pay instrumentation
    restore()
    t0 = time.perf_counter(); run_replays(a.replays); xq_ms = (time.perf_counter() - t0) * 1e3 / a.replays
    res["xq_replay_ms"] = xq_ms
    res["no_preempt_exact"] = all(torch.equal(t, g0) for t, g0 in zip(state, gold))

    # Replays with preemptions.
    restore()
    susp, hp_lat, hp_bad = [], [], 0
    done = threading.Event()
    rng = random.Random(7)

    def preemptor():
        nonlocal hp_bad
        torch.cuda.set_device(0)
        for _ in range(a.preempts):
            time.sleep(rng.uniform(0.3, 1.7) * xq_ms / 1e3)
            if done.is_set():
                break
            t0 = time.perf_counter()
            xs.pre.XQueueSuspend(xq, 1)
            t1 = time.perf_counter()
            with torch.cuda.stream(hp):
                y = hp_fn()
            hp.synchronize()
            t2 = time.perf_counter()
            xs.pre.XQueueResume(xq, 0)
            susp.append((t1 - t0) * 1e3)
            hp_lat.append((t2 - t0) * 1e3)
            if not torch.equal(y, hp_gold):
                hp_bad += 1

    th = threading.Thread(target=preemptor, daemon=True)
    t0 = time.perf_counter()
    th.start()
    run_replays(a.replays)
    done.set()
    th.join(timeout=60)
    res["preempted_run_ms"] = (time.perf_counter() - t0) * 1e3
    exact = all(torch.equal(t, g0) for t, g0 in zip(state, gold))
    xs.destroy(hwq, xq)
    res.update({"preempts": len(susp), "susp_p50_ms": pct(susp, 0.5), "susp_p99_ms": pct(susp, 0.99),
                "susp_max_ms": max(susp) if susp else None, "hp_p50_ms": pct(hp_lat, 0.5), "hp_p99_ms": pct(hp_lat, 0.99),
                "hp_exact": len(hp_lat) - hp_bad, "exact": exact, "flatten_stats": flatten_stats()})
    ok = deterministic and exact and res["no_preempt_exact"] and hp_bad == 0
    f = lambda k: "-" if res.get(k) is None else f"{res[k]:.3f}"
    print(f"[GRAPH-PREEMPT lv{a.level} flatten={flatten}] {'PASS' if ok else 'FAIL'} | replay native {f('native_replay_ms')} "
          f"xq {f('xq_replay_ms')} ms | susp p50 {f('susp_p50_ms')} p99 {f('susp_p99_ms')} max {f('susp_max_ms')} ms | "
          f"HP p50 {f('hp_p50_ms')} p99 {f('hp_p99_ms')} ms | preempts {res['preempts']} | exact {exact} "
          f"(no-preempt {res['no_preempt_exact']}, deterministic {deterministic}) | {res['flatten_stats']}", flush=True)
    if a.out:
        with open(a.out, "a") as fo:
            fo.write(json.dumps(res) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
