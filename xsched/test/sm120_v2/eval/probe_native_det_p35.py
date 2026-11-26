#!/usr/bin/env python3
"""OBS-20 probe: is a freshly started process's GPU work correct right after another process that
used Level-3 preemption has exited?

Computes a deterministic workload (64M-element cumsum and a 2048^2 GEMM chain) repeatedly during
the first --window seconds after start (inputs synchronized before use, T-26), then once more after
--settle seconds; every result must equal the late reference. Prints the number of mismatching
repetitions and when the first one happened. Runs without XSched by default (plain driver), so
anything it sees comes from outside the process.
"""
import argparse, sys, time
t_start = time.perf_counter()
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--settle", type=float, default=3.0)
    a = ap.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    # int32 input: an FP32 CUB scan is not bitwise run-to-run deterministic (look-back order)
    x = torch.randint(-1000, 1000, (64 * 1024 * 1024,), device="cuda", dtype=torch.int32)
    A = torch.randn(2048, 2048, device="cuda") / 45.0
    torch.cuda.synchronize()
    s = torch.cuda.Stream()

    w = None

    def fp(t):
        # Exact fingerprint of the bit pattern (two int64 checksums; the second is position-weighted).
        nonlocal w
        b = t.reshape(-1)
        b = b.view(torch.int32).to(torch.int64) if b.dtype == torch.float32 else b.to(torch.int64)
        if w is None or w.numel() != b.numel():
            w = torch.arange(1, b.numel() + 1, device=b.device, dtype=torch.int64) % 1000003
        return (int(b.sum()), int((b * w[:b.numel()]).sum()))

    def work():
        with torch.cuda.stream(s):
            c = torch.cumsum(x, 0)
            t = A
            for _ in range(4):
                t = torch.mm(A, t)
        s.synchronize()
        return c, t

    results = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.window:
        c, t = work()
        results.append((time.perf_counter() - t_start, fp(c), fp(t)))
    time.sleep(a.settle)
    c_ref, t_ref = work()
    c2, t2 = work()
    late_det = torch.equal(c_ref, c2) and torch.equal(t_ref, t2)
    fc, ft = fp(c_ref), fp(t_ref)
    bad = [(ts, c != fc, t != ft) for ts, c, t in results if c != fc or t != ft]
    first = f"first at {bad[0][0]:.2f}s (cumsum wrong {bad[0][1]}, gemm wrong {bad[0][2]})" if bad else ""
    print(f"[NATIVE-DET] {'PASS' if not bad and late_det else 'FAIL'} | reps {len(results)} bad {len(bad)} {first} | "
          f"late deterministic {late_det}", flush=True)
    return 0 if not bad and late_det else 1


if __name__ == "__main__":
    sys.exit(main())
