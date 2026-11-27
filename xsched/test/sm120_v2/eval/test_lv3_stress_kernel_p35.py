#!/usr/bin/env python3
"""Phase 3.5 (OBS-21 / DEF-20b): Level-3 stress on one production kernel under frequent preemption.

Runs the kernel in a loop on an XQueue (Level 3 by default) while a preemptor thread suspends and
resumes the queue every few hundred microseconds; every output must be bit-identical to a golden
computed before any XQueue exists. Reports the number of wrong iterations and preemptions, and the
registers per thread of the kernel (from the instrumentation log, if XSched printed it).

  --kernel flash     PyTorch SDPA flash-attention forward, bf16, head dim 64, causal (the 255-register
                     pytorch_flash::flash_fwd_kernel used by HF Qwen2 prefill)
  --kernel flash128  same with head dim 128
  --kernel gemm      cuBLAS BF16 GEMM (control: < 255 registers)

usage: test_lv3_stress_kernel_p35.py [--kernel flash] [--level 3] [--iters 3000]
"""
import argparse, os, random, sys, threading, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def make(kind, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind.startswith("flash"):
        d = 128 if kind == "flash128" else 64
        q = torch.randn(4, 14, 1024, d, device="cuda", dtype=torch.bfloat16, generator=g)
        k = torch.randn(4, 14, 1024, d, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(4, 14, 1024, d, device="cuda", dtype=torch.bfloat16, generator=g)
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def run():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return run
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16, generator=g)
    return lambda: torch.mm(a, b)



def recovery_pulses():
    """Recovery pulses fired by the SM120 trap engine so far (OBS-21 investigation)."""
    try:
        import ctypes
        f = ctypes.CDLL("libhalcuda.so").XSm120RecoveryPulses
        f.restype = ctypes.c_uint64
        return int(f())
    except Exception:
        return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="flash", choices=["flash", "flash128", "gemm"])
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--gap-us", type=float, default=600.0)
    ap.add_argument("--hold-us", type=float, default=200.0)
    ap.add_argument("--chain", type=int, default=8, help="kernel calls per iteration (no sync in between)")
    a = ap.parse_args()
    torch.cuda.set_device(0)
    lp = torch.cuda.Stream()
    with torch.cuda.stream(lp):
        fn = make(a.kernel)
        gold = fn().clone()
        gold2 = fn().clone()
    lp.synchronize()
    det = torch.equal(gold, gold2)
    t0 = time.perf_counter()
    with torch.cuda.stream(lp):
        for _ in range(20):
            fn()
    lp.synchronize()
    k_ms = (time.perf_counter() - t0) * 1e3 / 20

    from bench_mp_serving_p34 import Bridge
    xs = Bridge()
    hwq, xq = xs.create(lp, a.level)
    with torch.cuda.stream(lp):
        fn()
    lp.synchronize()

    stop = threading.Event()
    n_pre = [0]

    def preemptor():
        rng = random.Random(3)
        while not stop.is_set():
            time.sleep(rng.uniform(0.3, 1.7) * a.gap_us / 1e6)
            xs.pre.XQueueSuspend(xq, 1)
            time.sleep(a.hold_us / 1e6)
            xs.pre.XQueueResume(xq, 0)
            n_pre[0] += 1

    th = threading.Thread(target=preemptor, daemon=True)
    th.start()
    bad, first = 0, None
    for it in range(a.iters):
        with torch.cuda.stream(lp):
            ys = [fn() for _ in range(a.chain)]
        lp.synchronize()
        for y in ys:
            if not torch.equal(y, gold):
                bad += 1
                if first is None:
                    first = (it, int((y != gold).sum()), y.numel())
    stop.set()
    th.join()
    xs.destroy(hwq, xq)
    ok = bad == 0 and det
    extra = f" first bad iter {first[0]} ({first[1]}/{first[2]} elements)" if first else ""
    print(f"[LV3-STRESS {a.kernel} lv{a.level}] {'PASS' if ok else 'FAIL'} | calls {a.iters * a.chain} wrong {bad} | "
          f"preemptions {n_pre[0]} | recovery pulses {recovery_pulses()} | kernel {k_ms:.2f} ms | native deterministic {det}{extra}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
