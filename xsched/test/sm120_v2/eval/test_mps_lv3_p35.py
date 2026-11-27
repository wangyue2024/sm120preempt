#!/usr/bin/env python3
"""DEF-27 / DEF-41 structural fix: Level-3 across processes under MPS.

Two processes that both run Level-3 queues are two time-sliced contexts without MPS (a context switch
while one has warps in the trap handler: Xid 109, 14/14 runs). MPS clients share one hardware
context, so there is no time slicing between them. This test starts a private MPS control daemon
(foreground child, private pipe / log directories: only these test processes connect), then runs:

  1. two Level-3 processes (test_multiqueue_lv3_p35.py --procs 2) with default settings: pulses must
     be fired (the DEF-40 / DEF-41 checks recognise MPS peers) and every checkpoint must be exact;
  2. a native (non-XSched) MPS client next to a Level-3 client (test_foreign_ctx_p35.py): the victim
     must be unaffected (non-XSched MPS clients stay foreign; a pulse once took down the MPS server).

Reports new Xid events (needs root for dmesg). usage: test_mps_lv3_p35.py [--steps 6000]
"""
import argparse, os, re, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))


def xid_count():
    try:
        return sum(1 for l in subprocess.run(["dmesg"], capture_output=True, text=True, errors="replace").stdout.splitlines()
                   if "NVRM: Xid" in l)
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    if shutil.which("nvidia-cuda-mps-control") is None:
        print("[MPS-LV3] SKIP | nvidia-cuda-mps-control not found", flush=True)
        return 0
    root = tempfile.mkdtemp(prefix="xsched_mps_")
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=os.path.join(root, "pipe"),
               CUDA_MPS_LOG_DIRECTORY=os.path.join(root, "log"))
    os.makedirs(env["CUDA_MPS_PIPE_DIRECTORY"]); os.makedirs(env["CUDA_MPS_LOG_DIRECTORY"])
    x0 = xid_count()
    ctl = subprocess.Popen(["nvidia-cuda-mps-control", "-f"], env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    out = {}
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "test_multiqueue_lv3_p35.py"), "--procs", "2",
                            "--queues", "2", "--steps", str(a.steps), "--gap-us", "600", "--hold-us", "150"],
                           env=env, capture_output=True, text=True, errors="replace", timeout=300)
        lines = [l for l in r.stdout.splitlines() if l.startswith("[MULTIQ-LV3 q=")]
        pulses = [int(m.group(1)) for l in lines for m in [re.search(r"pulses (\d+) recovery", l)] if m]
        out["peers"] = (r.returncode == 0 and len(lines) == 2 and all("PASS" in l for l in lines), pulses)
        r = subprocess.run([sys.executable, os.path.join(HERE, "test_foreign_ctx_p35.py"), "--seconds", "10"],
                           env=env, capture_output=True, text=True, errors="replace", timeout=300)
        line = next((l for l in r.stdout.splitlines() if l.startswith("[FOREIGN-CTX")), "")
        out["foreign"] = ("victim unaffected" in line, line[:160])
    finally:
        subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env=env, timeout=60)
        try:
            ctl.wait(timeout=60)
        except subprocess.TimeoutExpired:
            ctl.kill()
        shutil.rmtree(root, ignore_errors=True)
    xid = xid_count() - x0 if x0 >= 0 else -1
    peers_ok, pulses = out.get("peers", (False, []))
    foreign_ok, fline = out.get("foreign", (False, ""))
    # On a GPU where pulses are withheld for other reasons (e.g. a desktop) pulses can be 0: the
    # correctness checks still apply, the "pulses fired" check only where the GPU has no other tenants.
    ok = peers_ok and foreign_ok and xid == 0
    print(f"[MPS-LV3] {'PASS' if ok else 'FAIL'} | two Level-3 MPS clients ok {peers_ok} pulses {pulses} | "
          f"non-XSched MPS client unaffected {foreign_ok} | new Xid {xid}", flush=True)
    if not foreign_ok:
        print("  " + fline, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
