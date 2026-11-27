#!/usr/bin/env python3
"""DEF-37: default-stream (legacy stream) PyTorch work under XSCHED_AUTO_XQUEUE.

A plain PyTorch workload that never selects a stream (GEMM chains, integer cumsum, .cpu(), .item(),
non-blocking pinned copies, a side stream ordered with wait_stream on the default stream,
synchronous H2D copies via torch.tensor(...).cuda(), events). Under XSCHED_AUTO_XQUEUE the legacy
default stream is proxied to an XQueue; a preemptor thread suspends / resumes that XQueue
(Level-3 when XSCHED_AUTO_XQUEUE_LEVEL=3) during the run. The digest of every step must equal a
native run's (MODE=ref writes them). Reports how many commands went to the legacy stream itself.

  python test_auto_legacy_p35.py --ref ref.json --write          # native (no XSched) reference
  XSCHED_AUTO_XQUEUE=1 XSCHED_AUTO_XQUEUE_LEVEL=3 python test_auto_legacy_p35.py --ref ref.json
"""
import argparse, ctypes, hashlib, json, random, sys, threading, time
import torch


def digest(t):
    return hashlib.sha1(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def workload(steps):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.randn(1024, 1024, device="cuda") / 32
    x = torch.randn(1024, 1024, device="cuda")
    side = torch.cuda.Stream()
    pinned = torch.empty(1024, 1024, pin_memory=True)
    out = []
    for i in range(steps):
        for _ in range(6):
            x = torch.tanh(a @ x)                                    # default stream
        c = torch.cumsum((x * 1000).to(torch.int32), 0)              # CUB scan, integer: deterministic
        s = float(x.sum().item())                                    # D2H + sync on the default stream
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            y = (x * 2).sum(0)
        torch.cuda.current_stream().wait_stream(side)
        pinned.copy_(x, non_blocking=True)                          # D2H into pinned memory
        ev = torch.cuda.Event()
        ev.record()
        ev.synchronize()
        h = torch.tensor([float(i), s], dtype=torch.float32).cuda()  # synchronous H2D
        out.append({"x": digest(x), "c": digest(c), "y": digest(y), "pinned": digest(pinned), "h": digest(h)})
    torch.cuda.synchronize()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    torch.cuda.set_device(0)
    if a.write:
        json.dump(workload(a.steps), open(a.ref, "w"))
        print("[AUTO-LEGACY ref] written")
        return 0
    cu = ctypes.CDLL("libcuda.so.1")
    cu.XCudaLegacyProxyXQueue.restype = ctypes.c_uint64
    torch.zeros(1, device="cuda")                                   # create the context
    xq = cu.XCudaLegacyProxyXQueue()
    pre = ctypes.CDLL("libpreempt.so")
    for f in ("XQueueSuspend", "XQueueResume"):
        getattr(pre, f).argtypes = [ctypes.c_uint64, ctypes.c_int64]
    stop, n = threading.Event(), [0]

    def preemptor():
        rng = random.Random(4)
        while not stop.is_set():
            time.sleep(rng.uniform(0.5, 2.0) / 1e3)
            pre.XQueueSuspend(xq, 1)
            time.sleep(0.0005)
            pre.XQueueResume(xq, 0)
            n[0] += 1

    th = threading.Thread(target=preemptor, daemon=True)
    if xq:
        th.start()
    got = workload(a.steps)
    stop.set()
    if xq:
        th.join()
    ref = json.load(open(a.ref))
    bad = [i for i, (g, r) in enumerate(zip(got, ref)) if g != r]
    st = (ctypes.c_uint64 * 3)()
    cu.XCudaShimStats.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int]
    cu.XCudaShimStats(st, 3, 0)
    ok = xq != 0 and not bad
    print(f"[AUTO-LEGACY] {'PASS' if ok else 'FAIL'} | proxy xqueue {'yes' if xq else 'NO'} | steps {a.steps} wrong {len(bad)}"
          f"{(' first ' + str(bad[0])) if bad else ''} | suspensions {n[0]} | commands legacy {st[0]} unmanaged {st[1]} "
          f"xqueue {st[2]}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
