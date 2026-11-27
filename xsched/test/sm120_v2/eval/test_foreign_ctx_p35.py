#!/usr/bin/env python3
"""Phase 3.5 (OBS-8 / OBS-22): does a Level-3 BAR0 pulse hit a *foreign* CUDA context?

The BAR0 STOP_TRIGGER pulse is GPU-wide. Without MPS, contexts time-share the graphics engine; a
pulse fired while another context is resident traps that context's warps, which have no XSched
trap handler installed. This test runs, on the same GPU:

  victim     a native process (no XSched libraries) that loops a GEMM chain and checks every result
             against its own golden (reports wrong results, CUDA errors, exit status);
  aggressor  this process: a Level-3 XQueue running GEMMs that a preemptor thread suspends and
             resumes every few hundred microseconds.

Both are the caller's own processes. Run it only on a GPU with no other tenant (guarded runner).

  python test_foreign_ctx_p35.py [--seconds 20] [--gap-us 600] [--victim-only]
Environment: NATIVE_LD_LIBRARY_PATH for the victim (default /usr/lib/x86_64-linux-gnu).
"""
import argparse, json, os, random, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

VICTIM = r'''
import json, sys, time, torch
torch.backends.cuda.matmul.allow_tf32 = False
secs = float(sys.argv[1])
torch.manual_seed(1)
a = torch.randn(2048, 2048, device="cuda") / 64
x0 = torch.randn(2048, 2048, device="cuda")
def chain():
    x = x0
    for _ in range(8):
        x = torch.tanh(a @ x)
    return x
gold = chain().clone(); torch.cuda.synchronize()
n = bad = 0; err = ""
t_end = time.time() + secs
try:
    while time.time() < t_end:
        y = chain(); torch.cuda.synchronize()
        n += 1
        if not torch.equal(y, gold):
            bad += 1
except Exception as e:
    err = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
print("VICTIM " + json.dumps({"iters": n, "wrong": bad, "error": err}), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--gap-us", type=float, default=600.0)
    ap.add_argument("--hold-us", type=float, default=200.0)
    ap.add_argument("--victim-only", action="store_true", help="control run: no aggressor")
    a = ap.parse_args()
    env = {k: v for k, v in os.environ.items() if not k.startswith("XSCHED_")}
    env["LD_LIBRARY_PATH"] = os.environ.get("NATIVE_LD_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu")
    env.pop("LD_PRELOAD", None)
    victim = subprocess.Popen([sys.executable, "-c", VICTIM, str(a.seconds + 4)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(4)                                   # victim context created and running
    n_pre, agg_err = 0, ""
    if not a.victim_only:
        import torch
        torch.cuda.set_device(0)
        lp = torch.cuda.Stream()
        with torch.cuda.stream(lp):
            m = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        lp.synchronize()
        from bench_mp_serving_p34 import Bridge
        xs = Bridge()
        hwq, xq = xs.create(lp, 3)
        stop = threading.Event()

        def preemptor():
            nonlocal n_pre
            rng = random.Random(5)
            while not stop.is_set():
                time.sleep(rng.uniform(0.3, 1.7) * a.gap_us / 1e6)
                xs.pre.XQueueSuspend(xq, 1)
                time.sleep(a.hold_us / 1e6)
                xs.pre.XQueueResume(xq, 0)
                n_pre += 1

        th = threading.Thread(target=preemptor, daemon=True)
        th.start()
        t_end = time.time() + a.seconds
        try:
            while time.time() < t_end:
                with torch.cuda.stream(lp):
                    for _ in range(4):
                        torch.mm(m, m)
                lp.synchronize()
        except Exception as e:
            agg_err = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        stop.set()
        th.join()
        xs.destroy(hwq, xq)
    out, _ = victim.communicate(timeout=a.seconds + 120)
    v = {"iters": 0, "wrong": 0, "error": "no report"}
    for line in out.splitlines():
        if line.startswith("VICTIM "):
            v = json.loads(line[7:])
    hit = v["wrong"] > 0 or v["error"] != "" or victim.returncode != 0
    mode = "victim-only" if a.victim_only else f"aggressor lv3 gap {a.gap_us:.0f} us"
    print(f"[FOREIGN-CTX {mode}] victim {'HIT' if hit else 'unaffected'} | victim iters {v['iters']} wrong {v['wrong']} "
          f"rc {victim.returncode} error '{v['error']}' | aggressor suspensions {n_pre} error '{agg_err}'", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
