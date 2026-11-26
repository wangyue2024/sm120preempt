#!/usr/bin/env python3
"""
Tier-1 item 2 -- where does Level-3 matter? HP latency as a function of the LP kernel duration.

A low-priority (LP) stream runs back-to-back kernels of one family and size; a latency-critical
high-priority (HP) request (a decode-like chain of small GEMMs) arrives at random times. For each
arrival we measure the HP latency from arrival to completion. The LP kernel duration is swept from
~0.05 ms to ~100 ms.

  families: gemm        cuBLAS BF16 GEMM N x N x N (many CTAs per wave, each CTA one tile)
            persistent  Triton persistent BF16 GEMM, grid = #SMs (every CTA lives for the whole kernel)
            sdpa        causal flash attention prefill, B=1, H=16, D=128, sequence length L
  modes:    native      no XSched; HP on a high-priority stream (hardware CTA-granular priority)
            lv1         XSched Level-1: LP stops launching; HP (priority stream) competes with the LP
                        work already on the GPU (no hardware-queue sync)
            lv1-sync    Level-1 with SyncHwQueue: HP starts after the LP hardware queue drained
            lv3         XSched Level-3 trap preemption (Mode 5); HP after the eviction. Short loop-free
                        kernels (elementwise etc.) are routed to the Level-1 boundary (default router)
            lv3-strict  Level-3 with the short-CTA router off: every kernel is trap-preemptible

Correctness: every LP kernel output is compared to its golden on the GPU (cast-free mismatch flag),
every HP result is compared to its golden. LP retention = LP work done / (wall time - HP busy time),
relative to solo (1.0 = preemption costs LP nothing beyond the time HP occupies).

  python bench_lv3_value_p35.py run --mode lv3 --family gemm --out value.jsonl
  python bench_lv3_value_p35.py report value.jsonl [--md out.md] [--png out.png]
"""
import argparse
import json
import os
import random
import socket
import statistics
import sys
import threading
import time

try:
    import torch
except ImportError:      # `report` only needs the standard library (+ matplotlib for --png)
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SIZES = {
    "gemm": [1024, 2048, 4096, 6144, 8192, 12288],
    "persistent": [1024, 2048, 4096, 6144, 8192],
    "persistent-full": [1024, 2048, 4096, 6144, 8192],
    "sdpa": [2048, 4096, 8192, 16384, 32768],
}


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


# ---------------------------------------------------------------------------------------- LP kernels
def make_persistent(n_sms, full=False):
    import triton
    import triton.language as tl

    @triton.jit
    def _persistent_mm(a_ptr, b_ptr, c_ptr, M, N, K, NUM_SMS: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_n = N // BN
        num_tiles = (M // BM) * num_pid_n
        for tile in range(pid, num_tiles, NUM_SMS):
            pm = tile // num_pid_n
            pn = tile % num_pid_n
            offs_m = pm * BM + tl.arange(0, BM)
            offs_n = pn * BN + tl.arange(0, BN)
            offs_k = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k in range(0, K, BK):
                a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
                b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
                acc += tl.dot(a, b)
            tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))

    def run(a, b, c):
        n = a.shape[0]
        if full:   # ~96 KB shared memory per CTA: one CTA fills an SM, nothing can run beside it
            _persistent_mm[(n_sms,)](a, b, c, n, n, n, NUM_SMS=n_sms, BM=128, BN=128, BK=64,
                                     num_warps=8, num_stages=3)
        else:
            _persistent_mm[(n_sms,)](a, b, c, n, n, n, NUM_SMS=n_sms, BM=128, BN=128, BK=32,
                                     num_warps=8, num_stages=2)
        return c
    return run


def build_lp(family, size, n_sms):
    """Returns (run() -> output tensor, description). run() launches exactly one LP kernel
    (plus nothing else) on the current stream."""
    g = torch.Generator(device="cuda").manual_seed(1234)
    if family == "gemm":
        a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
        b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
        c = torch.empty(size, size, device="cuda", dtype=torch.bfloat16)
        return (lambda: torch.mm(a, b, out=c)), f"cuBLAS BF16 GEMM {size}^3"
    if family in ("persistent", "persistent-full"):
        a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
        b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
        c = torch.empty(size, size, device="cuda", dtype=torch.bfloat16)
        mm = make_persistent(n_sms, full=(family == "persistent-full"))
        tag = ", ~96 KB SMEM/CTA" if family == "persistent-full" else ""
        return (lambda: mm(a, b, c)), f"Triton persistent BF16 GEMM {size}^3, grid={n_sms}{tag}"
    if family == "sdpa":
        q = torch.randn(1, 16, size, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, 16, size, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, 16, size, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        f = torch.nn.functional.scaled_dot_product_attention
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def run():
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                return f(q, k, v, is_causal=True)
        return run, f"causal flash-attention prefill L={size} (B=1, H=16, D=128)"
    raise ValueError(family)


# ---------------------------------------------------------------------------------------- one config
def measure(family, size, mode, args, xs, hp_stream, n_sms):
    lp = torch.cuda.Stream()
    with torch.cuda.stream(lp):
        run, desc = build_lp(family, size, n_sms)
    torch.cuda.synchronize()

    # HP request (fixed, deterministic): "small" = decode-like chain of 4 GEMMs with 8 rows;
    # "large" = prefill-chunk-like chain with 512 rows (needs the whole GPU to be fast)
    gh = torch.Generator(device="cuda").manual_seed(99)
    rows = 8 if args.hp == "small" else 512
    x_hp = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16, generator=gh) * 0.05
    w_hp = [torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16, generator=gh) * 0.02 for _ in range(4)]

    def hp_fn():
        y = x_hp
        for w in w_hp:
            y = torch.mm(y, w)
        return y

    # Goldens and solo timings (before any XQueue exists)
    with torch.cuda.stream(lp):
        gold = run().clone()
        gold2 = run().clone()
    lp.synchronize()
    deterministic = bool(torch.equal(gold, gold2))
    with torch.cuda.stream(hp_stream):
        hp_gold = hp_fn().clone()
    hp_stream.synchronize()
    with torch.cuda.stream(lp):
        for _ in range(2):
            run()
    lp.synchronize()
    reps = 5
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(lp):
        e0.record()
        for _ in range(reps):
            run()
        e1.record()
    e1.synchronize()
    kernel_ms = e0.elapsed_time(e1) / reps
    hp_solo = []
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.cuda.stream(hp_stream):
            hp_fn()
        hp_stream.synchronize()
        hp_solo.append((time.perf_counter() - t0) * 1e3)
    hp_solo_ms = statistics.median(hp_solo[3:])

    res = {"family": family, "size": size, "desc": desc, "mode": mode, "hp": args.hp, "kernel_ms": kernel_ms,
           "hp_solo_ms": hp_solo_ms, "lp_deterministic": deterministic, "error": None}

    hwq = xq = None
    if mode != "native":
        hwq, xq = xs.create(lp, 1 if mode.startswith("lv1") else 3)
    if xs is not None:
        xs.route_stats(reset=True)

    mism = torch.zeros((), device="cuda", dtype=torch.bool)
    stop = threading.Event()
    lp_iters = [0]
    lp_exc = []

    def lp_loop():
        try:
            torch.cuda.set_device(0)
            pending = []
            with torch.cuda.stream(lp):
                while not stop.is_set():
                    o = run()
                    if deterministic:
                        torch.logical_or(mism, torch.ne(o, gold).any(), out=mism)
                    ev = torch.cuda.Event()
                    ev.record(lp)
                    pending.append(ev)
                    lp_iters[0] += 1
                    if len(pending) > 3:           # keep the GPU busy, bound the queue depth
                        pending.pop(0).synchronize()
            lp.synchronize()
        except Exception as e:  # noqa
            lp_exc.append(f"{type(e).__name__}: {e}")

    flags_sync = 1 if mode in ("lv1-sync", "lv3") else 0
    lat, susp, hp_busy, hp_bad = [], [], 0.0, 0
    rng = random.Random(size * 7 + len(mode))
    th = threading.Thread(target=lp_loop, daemon=True)
    th.start()
    time.sleep(max(0.05, 3 * kernel_ms / 1e3))
    t_start = None
    iters_start = 0
    try:
        for i in range(args.arrivals + args.warmup):
            if i == args.warmup:
                t_start = time.perf_counter()
                iters_start = lp_iters[0]
            # random arrival: uniformly over one-to-two LP kernel durations plus a small gap
            time.sleep(rng.uniform(0.5, 1.5) * kernel_ms / 1e3 + rng.uniform(0.002, 0.008) + args.gap_ms / 1e3)
            t0 = time.perf_counter()
            if xq is not None:
                xs.pre.XQueueSuspend(xq, flags_sync)
            t1 = time.perf_counter()
            with torch.cuda.stream(hp_stream):
                y = hp_fn()
            hp_stream.synchronize()
            t2 = time.perf_counter()
            if xq is not None:
                xs.pre.XQueueResume(xq, 0)
            if i < args.warmup:                 # first preemptions pay one-time costs; not measured
                continue
            lat.append((t2 - t0) * 1e3)
            susp.append((t1 - t0) * 1e3)
            hp_busy += t2 - t0
            if not torch.equal(y, hp_gold):
                hp_bad += 1
        wall = time.perf_counter() - t_start
        iters = lp_iters[0] - iters_start
    finally:
        stop.set()
        th.join(timeout=120)
    if xs is not None:
        res["route"] = xs.route_stats()
    if xq is not None:
        xs.destroy(hwq, xq)
    lp_work_ms = iters * kernel_ms
    res.update({
        "arrivals": len(lat), "hp_lat_p50_ms": pct(lat, 0.5), "hp_lat_p90_ms": pct(lat, 0.9),
        "hp_lat_p99_ms": pct(lat, 0.99), "hp_lat_max_ms": max(lat) if lat else None,
        "susp_p50_ms": pct(susp, 0.5), "susp_p99_ms": pct(susp, 0.99),
        "hp_exact": len(lat) - hp_bad, "lp_mismatch": bool(mism.item()) if deterministic else None,
        "lp_iters": iters, "wall_s": wall,
        # LP throughput over the whole measured window as a fraction of solo (any mode), and the
        # same excluding the time HP requests were in flight (meaningful when LP is suspended then)
        "lp_tput_frac": lp_work_ms / 1e3 / max(1e-9, wall),
        "hp_busy_frac": hp_busy / max(1e-9, wall),
        "lp_retention": lp_work_ms / 1e3 / max(1e-9, wall - hp_busy),
        "error": "LP: " + lp_exc[0] if lp_exc else None,
    })
    return res


def run_cmd(args):
    if args.mode == "lv3-strict":      # read by the HAL on first use, so set before any launch
        os.environ["XSCHED_SM120_SHORT_CTA_LV1"] = "0"
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    n_sms = prop.multi_processor_count
    xs = None
    if args.mode != "native":
        from bench_mp_serving_p34 import Bridge
        xs = Bridge()
        import ctypes
        try:
            xs.hal.XSm120GetRouteStats.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int]
            names = ["lv3_killable", "lv1_mbarrier", "lv1_cooperative", "lv1_cluster", "lv1_graph",
                     "lv1_uninstrumentable", "lv1_short_cta", "lv1_glb_sched"]

            def route_stats(reset=False):
                buf = (ctypes.c_uint64 * len(names))()
                n = xs.hal.XSm120GetRouteStats(buf, len(names), 1 if reset else 0)
                return {k: int(buf[i]) for i, k in enumerate(names[:max(0, n)])}
            xs.route_stats = route_stats
        except AttributeError:
            xs.route_stats = lambda reset=False: {}
    hp_stream = torch.cuda.Stream(priority=-1)
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else SIZES[args.family]
    for size in sizes:
        try:
            res = measure(args.family, size, args.mode, args, xs, hp_stream, n_sms)
        except Exception as e:  # noqa
            res = {"family": args.family, "size": size, "mode": args.mode, "error": f"{type(e).__name__}: {e}"}
        res.update({"gpu": prop.name, "sm": n_sms, "host": socket.gethostname()})
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        ok = (res.get("error") is None and res.get("hp_exact") == res.get("arrivals") and not res.get("lp_mismatch"))
        g = lambda k, s="{:.3f}": "-" if res.get(k) is None else s.format(res[k])
        print(f"[LV3-VALUE {args.family}:{size} {args.mode}] {'PASS' if ok else 'FAIL'} | kernel {g('kernel_ms')} ms | "
              f"HP p50 {g('hp_lat_p50_ms')} p99 {g('hp_lat_p99_ms')} max {g('hp_lat_max_ms')} ms (solo {g('hp_solo_ms')}) | "
              f"susp p50 {g('susp_p50_ms')} | LP tput {g('lp_tput_frac', '{:.2f}')} (HP busy {g('hp_busy_frac', '{:.2f}')}) | exact HP {res.get('hp_exact')}/"
              f"{res.get('arrivals')} LP mism {res.get('lp_mismatch')} | route {res.get('route', '-')} | {res.get('error') or ''}",
              flush=True)
    return 0


def report(paths, md=None, png=None):
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    rows = [r for r in rows if r.get("error") is None]
    modes = ["native", "lv1", "lv1-sync", "lv3", "lv3-strict"]
    out = []
    fams_all = ("gemm", "persistent", "persistent-full", "sdpa")
    for gpu in sorted({r["gpu"] for r in rows}):
      for hpk in ("small", "large"):
        for fam in fams_all:
            fr = [r for r in rows if r["gpu"] == gpu and r["family"] == fam and r.get("hp", "small") == hpk]
            if not fr:
                continue
            hp_solo = statistics.median(r["hp_solo_ms"] for r in fr)
            out.append(f"\n#### {gpu.replace('NVIDIA GeForce ', '')} — LP {fam}，HP {hpk}（独占 {hp_solo:.2f} ms）\n")
            out.append("| 规模 | LP 单核 (ms) | " + " | ".join(f"{m} HP P50 / P99 (ms)" for m in modes) + " | Lv3 LP 吞吐（占独占） | 正确性 |")
            out.append("|---|---:|" + "---|" * len(modes) + "---:|---|")
            for size in sorted({r["size"] for r in fr}):
                cells, kms, ret, ok = [], None, "-", True
                for m in modes:
                    r = next((x for x in fr if x["size"] == size and x["mode"] == m), None)
                    if r is None:
                        cells.append("-")
                        continue
                    kms = r["kernel_ms"] if kms is None else kms
                    cells.append(f"{r['hp_lat_p50_ms']:.2f} / {r['hp_lat_p99_ms']:.2f}")
                    ok &= r["hp_exact"] == r["arrivals"] and not r.get("lp_mismatch")
                    if m == "lv3":
                        ret = f"{r['lp_tput_frac']:.2f}"
                out.append(f"| {size} | {kms:.2f} | " + " | ".join(cells) + f" | {ret} | {'全部一致' if ok else '有误'} |")
    text = "\n".join(out)
    print(text)
    if md:
        open(md, "w").write(text + "\n")
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gpus = sorted({r["gpu"] for r in rows})
        fams = [f for f in ("gemm", "persistent", "persistent-full", "sdpa") if any(r["family"] == f for r in rows)]
        panels = [(gpu, hpk) for gpu in gpus for hpk in ("small", "large") if any(r["gpu"] == gpu and r.get("hp", "small") == hpk for r in rows)]
        fig, axes = plt.subplots(len(panels), len(fams), figsize=(4.0 * len(fams), 3.2 * len(panels)), squeeze=False)
        for i, (gpu, hpk) in enumerate(panels):
            for j, fam in enumerate(fams):
                ax = axes[i][j]
                for m, st in zip(modes, ["o-", "s--", "^-.", "D-", "v-"]):
                    pts = sorted((r["kernel_ms"], r["hp_lat_p99_ms"]) for r in rows
                                 if r["gpu"] == gpu and r["family"] == fam and r["mode"] == m and r.get("hp", "small") == hpk)
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], st, label=m, markersize=4)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"{gpu.replace('NVIDIA GeForce ', '')} LP={fam} HP={hpk}", fontsize=9)
                ax.set_xlabel("LP kernel duration (ms)", fontsize=8)
                ax.set_ylabel("HP latency P99 (ms)", fontsize=8)
                ax.grid(True, which="both", alpha=0.3)
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(png, dpi=130)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", required=True, choices=["native", "lv1", "lv1-sync", "lv3", "lv3-strict"])
    r.add_argument("--family", required=True, choices=list(SIZES))
    r.add_argument("--sizes", default="")
    r.add_argument("--arrivals", type=int, default=24)
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--hp", default="small", choices=["small", "large"])
    r.add_argument("--gap-ms", type=float, default=0.0, help="extra spacing between HP arrivals")
    r.add_argument("--out", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("paths", nargs="+")
    rp.add_argument("--md")
    rp.add_argument("--png")
    a = p.parse_args()
    return run_cmd(a) if a.cmd == "run" else report(a.paths, a.md, a.png)


if __name__ == "__main__":
    sys.exit(main() or 0)
