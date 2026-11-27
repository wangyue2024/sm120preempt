#!/usr/bin/env python3
"""Level-3 restore of a maximum-register kernel (ptxas reports 255; R0..R252 live at every trap).

kernels/regpress255.cu keeps 250 accumulators live in registers through a long loop; compiled with
-maxrregcount=255 it uses exactly 255 registers and no spills, so a trap at any point of the loop
finds R0..R254 live. The kernel runs repeatedly on a Level-3 XQueue while a preemptor thread
suspends it, runs a high-priority "register scrubber" (the same kernel with other inputs, filling the
register files of all SMs, as HP work would) and resumes it; every output must be bit-identical to
the native golden.


The kernel library is built on first use with $CUDA_HOME_LOCAL/bin/nvcc (host compiler $CXX or
g++-13) into $XSCHED_TEST_CACHE (default ~/.cache/xsched_tests).
"""
import argparse, ctypes, os, random, subprocess, sys, threading, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def build(nacc=250, maxrreg=255):
    src = os.path.join(HERE, "kernels", "regpress255.cu")
    cache = os.environ.get("XSCHED_TEST_CACHE", os.path.expanduser("~/.cache/xsched_tests"))
    os.makedirs(cache, exist_ok=True)
    so = os.path.join(cache, f"libregpress255_n{nacc}_r{maxrreg}.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        nvcc = os.path.join(os.environ.get("CUDA_HOME_LOCAL", "/usr/local/cuda"), "bin", "nvcc")
        cxx = os.environ.get("CXX") or ("/usr/bin/g++-13" if os.path.exists("/usr/bin/g++-13") else "g++")
        out = subprocess.run([nvcc, "-ccbin", cxx, "-arch=sm_120", "-O3", f"-maxrregcount={maxrreg}", "-Xptxas", "-v",
                              f"-DNACC={nacc}", f"-DLB={1 if maxrreg == 255 else 0}", "-shared", "-Xcompiler", "-fPIC", "-o", so, src], capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit("nvcc failed:\n" + out.stderr[-2000:])
        info = [l for l in out.stderr.splitlines() if "Used" in l or "spill" in l]
        if (nacc == 250 and maxrreg == 255 and not any("Used 255 registers" in l for l in info)) or \
                (maxrreg == 255 and any(" 0 bytes spill" not in l for l in info if "spill" in l)):
            raise SystemExit("kernel is not a 255-register, spill-free kernel: " + " | ".join(info))
    return so


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--calls", type=int, default=600)
    ap.add_argument("--iters", type=int, default=4000, help="loop iterations inside the kernel")
    ap.add_argument("--gap-us", type=float, default=1500.0)
    ap.add_argument("--hold-us", type=float, default=300.0)
    ap.add_argument("--expect-fail", action="store_true")
    ap.add_argument("--nacc", type=int, default=250, help="live accumulators (250 -> 255 registers)")
    ap.add_argument("--blocks-per-sm", type=int, default=2)
    ap.add_argument("--maxrreg", type=int, default=255)
    ap.add_argument("--no-scrub", dest="scrub", action="store_false", help="no HP work during the suspension")
    a = ap.parse_args()
    lib = ctypes.CDLL(build(a.nacc, a.maxrreg))
    lib.launch_regpress255.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    torch.cuda.set_device(0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    blocks = a.blocks_per_sm * sms
    lp = torch.cuda.Stream()
    x = torch.rand(4096, device="cuda") + 0.5
    out = torch.empty(blocks * 128, device="cuda")
    hp = torch.cuda.Stream(priority=-1)
    x_hp = torch.rand(4096, device="cuda") * 3.0 + 7.0
    out_hp = torch.empty(blocks * 128, device="cuda")
    torch.cuda.synchronize()

    def call():
        rc = lib.launch_regpress255(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(out.data_ptr()), a.iters, blocks,
                                    ctypes.c_void_p(lp.cuda_stream))
        assert rc == 0, rc

    call(); lp.synchronize(); gold = out.clone()
    call(); lp.synchronize(); det = torch.equal(out, gold)
    t0 = time.perf_counter(); call(); lp.synchronize(); k_ms = (time.perf_counter() - t0) * 1e3

    from bench_mp_serving_p34 import Bridge
    xs = Bridge()
    hwq, xq = xs.create(lp, a.level)
    call(); lp.synchronize()
    stop, n_pre = threading.Event(), [0]

    def preemptor():
        rng = random.Random(5)
        while not stop.is_set():
            time.sleep(rng.uniform(0.3, 1.7) * a.gap_us / 1e6)
            xs.pre.XQueueSuspend(xq, 1)
            if a.scrub:
                lib.launch_regpress255(ctypes.c_void_p(x_hp.data_ptr()), ctypes.c_void_p(out_hp.data_ptr()), 200,
                                       blocks, ctypes.c_void_p(hp.cuda_stream))
                hp.synchronize()
            time.sleep(a.hold_us / 1e6)
            xs.pre.XQueueResume(xq, 0)
            n_pre[0] += 1

    th = threading.Thread(target=preemptor, daemon=True)
    th.start()
    bad, first = 0, None
    for i in range(a.calls):
        with torch.cuda.stream(lp):    # T-26: on the kernel's stream (the legacy stream would race)
            out.zero_()
        call()
        lp.synchronize()
        if not torch.equal(out, gold):
            bad += 1
            if first is None:
                first = (i, int((out != gold).sum()))
    stop.set(); th.join()
    xs.destroy(hwq, xq)
    ok = (bad > 0) if a.expect_fail else (bad == 0 and det)
    tag = f"nacc={a.nacc} ctas/sm={a.blocks_per_sm}"
    extra = f" first bad call {first[0]} ({first[1]}/{out.numel()} threads)" if first else ""
    print(f"[REGPRESS255 lv{a.level} {tag}] {'PASS' if ok else 'FAIL'} | calls {a.calls} wrong {bad} | preemptions "
          f"{n_pre[0]} | kernel {k_ms:.2f} ms | native deterministic {det}{extra}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
