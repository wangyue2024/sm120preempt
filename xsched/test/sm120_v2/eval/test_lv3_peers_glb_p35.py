#!/usr/bin/env python3
"""DEF-41: two Level-3 LP processes plus a Level-1 HP process under the global scheduler.

Two processes that both run Level-3 queues are two time-sliced contexts; a context switch that arrives
while one of them has warps in the trap handler times out (Xid 109). The fix: `xserver`'s HPF policy
lets only one Level-3 process at a time run at the top priority (rotation every
XSCHED_HPF_LV3_QUANTUM_US), and a Level-3 process may pulse while its Level-3 peers are quiescent
(published in the DEF-40 tenant registry).

  ref   (native, no XSched)    writes per-checkpoint digests of a non-idempotent in-place chain
  lp x2 (GLB, auto XQueue Lv3) run the same chain continuously, compare every checkpoint
  hp    (GLB, auto XQueue Lv1) Poisson bursts of small GEMMs, compared to its own first result

usage: test_lv3_peers_glb_p35.py [--seconds 20] [--exclusive 1] [--port 50010]
Run as root (BAR0) on a GPU without other tenants. Reports pulses fired, pulses withheld, and new
Xid events (dmesg).
"""
import argparse, ctypes, hashlib, json, os, random, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def chain_setup():
    import torch
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        a = torch.randn(1024, 1024, device="cuda") * (1.5 / 32)   # bounded chaotic map (T-30)
        t0 = torch.randn(1024, 1024, device="cuda") * 0.5
    s.synchronize()
    return a, t0, s


def digest(t):
    import torch
    return hashlib.sha1(t.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def chain_round(a, t0, s, steps, every, check=None, t_end=None):
    """u = tanh(0.5 u + 0.5 A t) in place (two non-idempotent kernels, no aliasing), swap t <-> u.
    Returns the checkpoint digests (or the number of mismatches against `check`)."""
    import torch
    out, bad = [], 0
    with torch.cuda.stream(s):
        t, u = t0.clone(), t0 * 0.25
        for i in range(1, steps + 1):
            u.addmm_(a, t, beta=0.5, alpha=0.5)
            u.tanh_()
            t, u = u, t
            if i % every == 0:
                s.synchronize()
                d = digest(t)
                if check is None:
                    out.append((d, float(t.abs().mean())))
                elif d != check[i // every - 1]:
                    bad += 1
                if t_end is not None and time.time() > t_end:
                    break
    return out if check is None else bad


def role_ref(args):
    out = chain_round(*chain_setup(), args.steps, args.every)
    if min(m for _, m in out) < 0.05:
        raise SystemExit("reference chain degenerated")
    json.dump([d for d, _ in out], open(args.ref, "w"))
    print("REF_OK", len(out), flush=True)


def lib_counter(name):
    try:
        f = getattr(ctypes.CDLL("libhalcuda.so"), name)
        f.restype = ctypes.c_uint64
        return int(f())
    except Exception:
        return -1


def role_lp(args):
    import torch
    torch.cuda.set_device(0)
    ref = json.load(open(args.ref))
    a, t0, s = chain_setup()
    rounds = bad = 0
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        bad += chain_round(a, t0, s, args.steps, args.every, check=ref, t_end=t_end)
        rounds += 1
    print("LP_RESULT " + json.dumps({"rounds": rounds, "bad_checkpoints": bad, "pulses": lib_counter("XSm120Pulses"),
                                     "recovery": lib_counter("XSm120RecoveryPulses")}), flush=True)


def role_hp(args):
    import torch
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x = torch.randn(512, 1024, device="cuda") / 32
        w = torch.randn(1024, 1024, device="cuda") / 32
        gold = (x @ w).clone()
    s.synchronize()
    rng = random.Random(3)
    n = bad = 0
    lat = []
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        time.sleep(rng.expovariate(1.0 / (args.hp_gap_ms / 1e3)))
        t1 = time.perf_counter()
        with torch.cuda.stream(s):
            y = x
            for _ in range(4):
                y = x @ w
        s.synchronize()
        lat.append((time.perf_counter() - t1) * 1e3)
        n += 1
        if not torch.equal(y, gold):
            bad += 1
    lat.sort()
    print("HP_RESULT " + json.dumps({"requests": n, "bad": bad, "p50_ms": lat[len(lat) // 2] if lat else None,
                                     "p99_ms": lat[int(len(lat) * 0.99)] if lat else None}), flush=True)


def xid_count():
    try:
        return sum(1 for l in subprocess.run(["dmesg"], capture_output=True, text=True, errors="replace").stdout.splitlines()
                   if "NVRM: Xid" in l)
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="run")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--hp-gap-ms", type=float, default=8.0)
    ap.add_argument("--exclusive", default="1")
    ap.add_argument("--port", type=int, default=50010)
    ap.add_argument("--ref", default="/tmp/lv3_peers_ref.json")
    a = ap.parse_args()
    if a.role == "ref":
        return role_ref(a)
    if a.role == "lp":
        return role_lp(a)
    if a.role == "hp":
        return role_hp(a)
    sys.path.insert(0, HERE)
    from bench_vllm_mp_p35 import check_ipc_owner
    # Scheduler settings come from this harness; SM120 engine settings (XSCHED_SM120_*) are passed on.
    base = {k: v for k, v in os.environ.items() if not k.startswith("XSCHED_") or k.startswith("XSCHED_SM120_")}
    nat = dict(base, LD_LIBRARY_PATH=os.environ.get("NATIVE_LD_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu"))
    me = [sys.executable, os.path.abspath(__file__), "--seconds", str(a.seconds), "--steps", str(a.steps),
          "--every", str(a.every), "--hp-gap-ms", str(a.hp_gap_ms), "--ref", a.ref]
    print("[LV3-PEERS-GLB] reference ...", flush=True)
    r = subprocess.run(me + ["--role", "ref"], env=nat, capture_output=True, text=True, errors="replace", timeout=600)
    if "REF_OK" not in r.stdout:
        raise SystemExit("reference failed: " + r.stdout[-300:] + r.stderr[-300:])
    check_ipc_owner()
    x0 = xid_count()
    glb = dict(base, XSCHED_SCHEDULER="GLB", XSCHED_AUTO_XQUEUE="1")
    xserver = subprocess.Popen([os.path.join(os.environ["XSCHED"], "bin", "xserver"), "HPF", str(a.port)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env=dict(glb, XSCHED_HPF_LV3_EXCLUSIVE=a.exclusive))
    time.sleep(1.0)
    procs = []
    try:
        for _ in range(2):
            procs.append(("lp", subprocess.Popen(me + ["--role", "lp"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                 text=True, errors="replace", env=dict(glb, XSCHED_AUTO_XQUEUE_LEVEL="3",
                                                                     XSCHED_AUTO_XQUEUE_PRIORITY="0"))))
        time.sleep(3.0)
        procs.append(("hp", subprocess.Popen(me + ["--role", "hp"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             text=True, errors="replace", env=dict(glb, XSCHED_AUTO_XQUEUE_LEVEL="1",
                                                                 XSCHED_AUTO_XQUEUE_PRIORITY="1"))))
        print(f"[LV3-PEERS-GLB] running: {[(r, p.pid) for r, p in procs]} xserver {xserver.pid}", flush=True)
        outs = [(role, p.communicate(timeout=a.seconds + 120), p.returncode) for role, p in procs]
    finally:
        xserver.send_signal(signal.SIGINT)
        try:
            xserver.wait(timeout=30)
        except subprocess.TimeoutExpired:
            xserver.kill()
    res, ok = [], True
    withheld = 0
    for role, (out, err), rc in outs:
        withheld += err.count("Level-3 pulses withheld")
        line = next((l for l in out.splitlines() if l.startswith(("LP_RESULT", "HP_RESULT"))), None)
        d = json.loads(line.split(" ", 1)[1]) if line else {"error": f"rc={rc}", "tail": err[-200:]}
        d["role"], d["rc"] = role, rc
        if rc != 0:
            d["stderr_tail"] = [l for l in err.splitlines() if "instrumented func" not in l][-6:]
        res.append(d)
        if rc != 0 or line is None or d.get("bad_checkpoints", d.get("bad", 1)) != 0:
            ok = False
    xid = xid_count() - x0 if x0 >= 0 else -1
    ok = ok and xid == 0
    lps = [d for d in res if d["role"] == "lp"]
    hp = next((d for d in res if d["role"] == "hp"), {})
    print(f"[LV3-PEERS-GLB exclusive={a.exclusive}] {'PASS' if ok else 'FAIL'} | "
          f"LP rounds {[d.get('rounds') for d in lps]} bad {[d.get('bad_checkpoints') for d in lps]} "
          f"pulses {[d.get('pulses') for d in lps]} | HP requests {hp.get('requests')} bad {hp.get('bad')} "
          f"P50/P99 {hp.get('p50_ms') and round(hp['p50_ms'], 2)}/{hp.get('p99_ms') and round(hp['p99_ms'], 2)} ms | "
          f"withheld warnings {withheld} | new Xid {xid}"
          + ("" if ok else f" | {[d for d in res if d.get('rc') != 0 or 'error' in d]}"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
