#!/usr/bin/env python3
"""Phase 3.5: several busy Level-3 XQueues in one process, each suspended / resumed independently.

The BAR0 pulse is GPU-wide, so a suspension of queue A traps the warps of queue B's kernels as well,
including while B is in the middle of its own restore wave. Each queue runs a chain of
non-idempotent in-place updates (t.addmm_(a, t) -- a re-executed or half-restored kernel changes the
result) and its own preemptor thread suspends and resumes it at random times. The final tensors must
be bit-identical to a native run of the same sequence.

usage: test_multiqueue_lv3_p35.py [--queues 3] [--steps 300] [--n 1024]
"""
import argparse, ctypes, os, random, subprocess, sys, threading, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queues", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--gap-us", type=float, default=1500.0)
    ap.add_argument("--hold-us", type=float, default=300.0)
    ap.add_argument("--allow-display", action="store_true", help="keep the pulse rate on a display GPU")
    ap.add_argument("--watchdog-s", type=float, default=20.0, help="dump queue state if the run takes longer")
    ap.add_argument("--procs", type=int, default=1, help="run this many processes, each with --queues queues "
                                                          "(cross-process pulses; DEF-38 host-wide lock)")
    ap.add_argument("--rank", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--sync-dir", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.procs > 1 and a.rank < 0:
        import tempfile
        sync = tempfile.mkdtemp(prefix="multiq_")
        args = [x for x in sys.argv[1:]]
        kids = [subprocess.Popen([sys.executable, os.path.abspath(__file__)] + args + ["--rank", str(r), "--sync-dir", sync])
                for r in range(a.procs)]
        rcs = [k.wait() for k in kids]
        mode = os.environ.get("XSCHED_SM120_HOST_PULSE_GATE", "1")
        print(f"[MULTIQ-LV3 procs={a.procs} host_gate={mode}] {'PASS' if all(r == 0 for r in rcs) else 'FAIL'} | "
              f"process exit codes {rcs}", flush=True)
        return 0 if all(r == 0 for r in rcs) else 1
    torch.cuda.set_device(0)
    # OBS-14: on a GPU that drives a display, frequent pulses make the desktop's time-slice context
    # switches hit warps inside the trap handler (Xid 109). Use a 4x lower rate there.
    # (`display_active` reports Disabled on a PRIME / Wayland desktop GPU; graphics processes do not lie.)
    try:
        smi = subprocess.run(["nvidia-smi", "-i", "0"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        smi = ""
    graphics = any(f.split()[5:6] in (["G"], ["C+G"]) for f in smi.splitlines() if f.startswith("|") and "MiB" in f)
    if graphics and not a.allow_display:
        a.gap_us, a.hold_us = a.gap_us * 4, a.hold_us * 2
        print(f"[MULTIQ-LV3] display GPU: gap {a.gap_us:.0f} us, hold {a.hold_us:.0f} us", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    streams = [torch.cuda.Stream() for _ in range(a.queues)]
    A = [torch.randn(a.n, a.n, device="cuda") / a.n for _ in range(a.queues)]
    T0 = [torch.randn(a.n, a.n, device="cuda") for _ in range(a.queues)]
    torch.cuda.synchronize()                      # T-26: inputs from the legacy stream

    def run_all(ts):
        for _ in range(a.steps):
            for q in range(a.queues):
                with torch.cuda.stream(streams[q]):
                    ts[q].addmm_(A[q], ts[q], beta=0.5, alpha=0.5)   # t = 0.5 t + 0.5 A t (in place)
        for s in streams:
            s.synchronize()

    gold = [t.clone() for t in T0]
    run_all(gold)
    gold2 = [t.clone() for t in T0]
    run_all(gold2)
    det = all(torch.equal(x, y) for x, y in zip(gold, gold2))

    from bench_mp_serving_p34 import Bridge
    xs = Bridge()
    qs = [xs.create(s, 3) for s in streams]
    warm = [t.clone() for t in T0]
    run_all(warm)                                  # first launches (instrumentation)
    stop = threading.Event()
    n_pre = [0] * a.queues

    def preemptor(q):
        rng = random.Random(100 + q)
        xq = qs[q][1]
        while not stop.is_set():
            time.sleep(rng.uniform(0.3, 1.7) * a.gap_us / 1e6)
            xs.pre.XQueueSuspend(xq, 1)
            time.sleep(a.hold_us / 1e6)
            xs.pre.XQueueResume(xq, 0)
            n_pre[q] += 1

    ths = [threading.Thread(target=preemptor, args=(q,), daemon=True) for q in range(a.queues)]
    for th in ths:
        th.start()
    got = [t.clone() for t in T0]
    torch.cuda.synchronize()
    finished = threading.Event()

    def watchdog():
        # Diagnostics for a stall: dump every Level-3 queue's host stage, control header and CTA
        # states (XSm120DumpStreams needs no CUDA context and takes no locks) and the GPU load.
        for k in range(3):
            if finished.wait(a.watchdog_s if k == 0 else 10.0):
                return
            print(f"[MULTIQ-LV3] watchdog: run not finished after {a.watchdog_s + 10 * k:.0f} s", flush=True)
            try:
                ctypes.CDLL("libhalcuda.so").XSm120DumpStreams()
                u = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu,power.draw",
                                    "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
                print(f"[MULTIQ-LV3] watchdog: GPU {u.strip()}", flush=True)
            except Exception as e:
                print(f"[MULTIQ-LV3] watchdog: dump failed: {e}", flush=True)

    threading.Thread(target=watchdog, daemon=True).start()
    if a.rank >= 0:                                 # start the stress phase of all processes together
        open(os.path.join(a.sync_dir, f"ready.{a.rank}"), "w").close()
        t_wait = time.time() + 120
        while len([f for f in os.listdir(a.sync_dir) if f.startswith("ready.")]) < a.procs and time.time() < t_wait:
            time.sleep(0.01)
    t0 = time.perf_counter()
    run_all(got)
    wall = time.perf_counter() - t0
    finished.set()
    stop.set()
    for th in ths:
        th.join()
    for hwq, xq in qs:
        xs.destroy(hwq, xq)
    bad = [q for q in range(a.queues) if not torch.equal(got[q], gold[q])]
    ok = det and not bad
    mode = os.environ.get("XSCHED_SM120_CTX_PULSE_GATE", "1")
    tag = f" rank={a.rank}" if a.rank >= 0 else ""
    print(f"[MULTIQ-LV3 q={a.queues}{tag} pulse_gate={mode}] {'PASS' if ok else 'FAIL'} | steps {a.steps} | "
          f"wrong queues {bad} | preemptions {n_pre} | wall {wall:.2f}s | native deterministic {det}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
