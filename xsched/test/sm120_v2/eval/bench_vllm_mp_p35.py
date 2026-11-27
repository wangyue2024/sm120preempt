#!/usr/bin/env python3
"""Phase 3.5: end-to-end application test -- two unmodified vLLM engines in two processes on one GPU,
scheduled transparently by XSched.

  HP process: a small model (default Qwen2.5-0.5B-Instruct) serving short chat requests that arrive
              with Poisson gaps (greedy, --hp-tokens new tokens each); per-request latency is measured.
  LP process: a larger model (default Qwen2.5-7B-Instruct-AWQ) running prefill-heavy batches in a loop
              (long documents, --lp-tokens new tokens); every batch's outputs must equal the reference.

  modes: solo     HP process alone                          -> latency floor
         glb-solo HP process alone, but under XSched (xserver + automatic XQueue): XSched's own cost
         native   HP + LP, no XSched (the GPU time-slices the two contexts)
         glb-lv1  xserver (HPF) + XSCHED_SCHEDULER=GLB + automatic XQueues for every stream:
         glb-lv3  HP queues priority 1 / Level-1, LP queues priority 0 / Level-1 or Level-3.
                  No application code in the scheduling path: vLLM is unmodified.
  ref      writes the reference outputs of both models (native, separately) to --ref-dir.

Every HP output must equal the solo reference and every LP batch must equal the LP reference.

  python bench_vllm_mp_p35.py run --mode ref --out vmp.jsonl
  python bench_vllm_mp_p35.py run --mode glb-lv3 --out vmp.jsonl
"""
import argparse
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = ("The history of computing spans centuries, from mechanical calculators to programmable machines, "
       "transistors, integrated circuits and the accelerators that train and serve large models today. ")
HP_PROMPTS = ["What is the capital of France?", "Name three prime numbers.", "Explain what a GPU is in one sentence.",
              "Translate 'good morning' into German.", "What is 17 times 23?", "Give a synonym for 'fast'.",
              "Who wrote Hamlet?", "What does CPU stand for?"]


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))] if s else None


def build_llm(model, util, max_len, eager):
    from vllm import LLM
    return LLM(model=model, gpu_memory_utilization=util, max_model_len=max_len, enforce_eager=eager, seed=0)


def xsched_stats():
    """Route / graph-flattening counters of this process (empty without XSched)."""
    import ctypes
    out = {}
    names = ["lv3_killable", "lv1_mbarrier", "lv1_coop", "lv1_cluster", "lv1_graph", "lv1_uninstr",
             "lv1_short_cta", "lv1_glb", "rf_full", "lv1_rf_full", "lv1_no_engine"]
    try:
        f = ctypes.CDLL("libhalcuda.so").XSm120GetRouteStats
        f.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int]
        buf = (ctypes.c_uint64 * len(names))()
        n = f(buf, len(names), 0)
        out["route"] = {k: buf[i] for i, k in enumerate(names[:n]) if buf[i]}
        g = ctypes.CDLL("libcuda.so.1").XCudaGraphFlattenStats
        g.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int]
        gb = (ctypes.c_uint64 * 5)()
        g(gb, 5, 0)
        out["graph"] = dict(zip(["flat", "whole_unflattenable", "flat_cmds", "unflattenable", "kept_whole"], list(gb)))
        h = ctypes.CDLL("libcuda.so.1").XCudaShimStats
        h.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int]
        hb = (ctypes.c_uint64 * 3)()
        h(hb, 3, 0)
        out["cmds"] = dict(zip(["legacy_stream", "unmanaged_stream", "xqueue"], list(hb)))
    except (OSError, AttributeError):
        pass
    return out


# ------------------------------------------------------------------------------------------- roles
def role_lp(a):
    from vllm import SamplingParams
    llm = build_llm(a.lp_model, a.lp_util, 4096, a.eager)
    prompts = [f"Document {i}: " + DOC * (a.lp_doc_reps + i % 4) + "\nSummarize the document in one line:"
               for i in range(a.lp_batch)]
    sp = SamplingParams(temperature=0, max_tokens=a.lp_tokens)
    texts = lambda outs: [o.outputs[0].text for o in outs]
    llm.generate(prompts, sp, use_tqdm=False)                 # warm-up
    ref_path = os.path.join(a.ref_dir, "lp_ref.json")
    if a.mode == "ref":
        json.dump(texts(llm.generate(prompts, sp, use_tqdm=False)), open(ref_path, "w"))
        print("LP_REF_WRITTEN", flush=True)
        return
    ref = json.load(open(ref_path))
    t0 = time.perf_counter()
    for _ in range(2):
        llm.generate(prompts, sp, use_tqdm=False)
    solo_s = (time.perf_counter() - t0) / 2
    stop_file = a.ctrl + ".stop"
    print("LP_READY", flush=True)
    iters = bad = 0
    t0 = time.perf_counter()
    while not os.path.exists(stop_file):
        out = texts(llm.generate(prompts, sp, use_tqdm=False))
        iters += 1
        if out != ref:
            bad += 1
    wall = time.perf_counter() - t0
    res = {"lp_iters": iters, "lp_bad_iters": bad, "lp_wall_s": wall, "lp_solo_batch_s": solo_s,
           "lp_tput_pct_of_solo": (100.0 * iters * solo_s / wall) if wall > 0 else None}
    res.update(xsched_stats())
    print("LP_RESULT " + json.dumps(res), flush=True)


def role_hp(a):
    from vllm import SamplingParams
    llm = build_llm(a.hp_model, a.hp_util, 2048, a.eager)
    sp = SamplingParams(temperature=0, max_tokens=a.hp_tokens)
    sp1 = SamplingParams(temperature=0, max_tokens=1)
    for p in HP_PROMPTS:                                       # warm-up
        llm.generate([p], sp, use_tqdm=False)
    ref_path = os.path.join(a.ref_dir, "hp_ref.json")
    if a.mode == "ref":
        json.dump([llm.generate([p], sp, use_tqdm=False)[0].outputs[0].text for p in HP_PROMPTS], open(ref_path, "w"))
        print("HP_REF_WRITTEN", flush=True)
        return
    ref = json.load(open(ref_path))
    rng = random.Random(11)
    gaps = [rng.expovariate(1.0 / a.mean_gap_ms) / 1e3 for _ in range(a.n_req)]
    e2e, ttft, ok = [], [], 0
    t_start = time.perf_counter()
    nxt = t_start
    for i in range(a.n_req):
        nxt += gaps[i]
        while time.perf_counter() < nxt:
            time.sleep(0.0005)
        p = HP_PROMPTS[i % len(HP_PROMPTS)]
        if i % 2 == 0:                                         # first-token latency on every other request
            t0 = time.perf_counter()
            llm.generate([p], sp1, use_tqdm=False)
            ttft.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        text = llm.generate([p], sp, use_tqdm=False)[0].outputs[0].text
        e2e.append((time.perf_counter() - t0) * 1e3)
        ok += int(text == ref[i % len(HP_PROMPTS)])
    wall = time.perf_counter() - t_start
    busy = (sum(e2e) + sum(ttft)) / 1e3
    res = {"n_req": a.n_req, "hp_exact": ok, "e2e_ms_p50": pct(e2e, 0.5), "e2e_ms_p99": pct(e2e, 0.99),
           "e2e_ms_max": max(e2e), "ttft_ms_p50": pct(ttft, 0.5), "ttft_ms_p99": pct(ttft, 0.99),
           "ms_per_token_p50": pct([x / a.hp_tokens for x in e2e], 0.5), "hp_wall_s": wall,
           "hp_duty_pct": 100.0 * busy / wall}
    print("HP_RESULT " + json.dumps(res), flush=True)


# -------------------------------------------------------------------------------------------- run
def check_ipc_owner():
    import pwd
    me = pwd.getpwuid(os.getuid()).pw_name
    for line in subprocess.run(["ipcs", "-q"], capture_output=True, text=True).stdout.splitlines():
        f = line.split()
        if len(f) >= 3 and f[0] == "0x00000001" and f[2] != me:
            raise SystemExit(f"xserver queue key 0x1 belongs to another user ({f[2]}): not starting a server")


def run(a):
    base = dict(os.environ)
    base["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" if a.vllm_mp else "0"
    for k in [k for k in base if k.startswith("XSCHED_")]:
        base.pop(k)
    if a.mode in ("ref", "solo", "native"):
        base["LD_LIBRARY_PATH"] = a.native_ld_path         # plain driver, XSched not loaded
    os.makedirs(a.ref_dir, exist_ok=True)
    common = [sys.executable, os.path.abspath(__file__), "role", "--mode", a.mode, "--ctrl", a.ctrl,
              "--ref-dir", a.ref_dir, "--hp-model", a.hp_model, "--lp-model", a.lp_model,
              "--hp-util", str(a.hp_util), "--lp-util", str(a.lp_util), "--n-req", str(a.n_req),
              "--mean-gap-ms", str(a.mean_gap_ms), "--hp-tokens", str(a.hp_tokens), "--lp-tokens", str(a.lp_tokens),
              "--lp-batch", str(a.lp_batch), "--lp-doc-reps", str(a.lp_doc_reps)] + (["--eager"] if a.eager else [])
    if a.mode == "ref":
        for role in ("lp", "hp"):
            r = subprocess.run(common + ["--role", role], env=base, timeout=a.timeout)
            if r.returncode != 0:
                raise SystemExit(f"reference run of {role} failed rc={r.returncode}")
        print("[VLLM-MP ref] references written", flush=True)
        return 0

    xserver = None
    hp_env, lp_env = dict(base), dict(base)
    if a.mode.startswith("glb"):
        check_ipc_owner()
        glb = {"XSCHED_SCHEDULER": "GLB", "XSCHED_AUTO_XQUEUE": "1"}
        hp_env.update(glb, XSCHED_AUTO_XQUEUE_LEVEL="1", XSCHED_AUTO_XQUEUE_PRIORITY="1")
        lp_env.update(glb, XSCHED_AUTO_XQUEUE_LEVEL="3" if a.mode in ("glb-lv3", "glb-solo") else "1",
                      XSCHED_AUTO_XQUEUE_PRIORITY="0")
        for kv in a.lp_env:
            k, _, v = kv.partition("=")
            lp_env[k] = v
        for kv in a.env:
            k, _, v = kv.partition("=")
            lp_env[k] = v
            hp_env[k] = v
        xlog = open(a.out + ".xserver.log", "a")
        xlog.write(f"=== {time.strftime('%F %T')} mode={a.mode}\n")
        xlog.flush()
        xserver = subprocess.Popen([os.path.join(os.environ["XSCHED"], "bin", "xserver"), "HPF", str(a.port)],
                                   stdout=xlog, stderr=subprocess.STDOUT,
                                   env=dict(base, XSCHED_SCHEDULER="GLB", XSCHED_HPF_RESUME_DELAY_US=str(a.hpf_resume_delay_us),
                                            **dict(kv.partition("=")[::2] for kv in a.env)))
        time.sleep(1.0)
    stop_file = a.ctrl + ".stop"
    if os.path.exists(stop_file):
        os.remove(stop_file)
    res = {"mode": a.mode, "hp_model": a.hp_model, "lp_model": a.lp_model, "mean_gap_ms": a.mean_gap_ms,
           "hpf_resume_delay_us": a.hpf_resume_delay_us,
           "vllm_mp": a.vllm_mp, "error": None}
    lp = None
    lp_lines, lp_ready = [], threading.Event()

    def drain(pipe):                       # keep the LP's stdout pipe empty while the HP runs
        for line in pipe:
            lp_lines.append(line)
            if line.startswith("LP_READY"):
                lp_ready.set()
        lp_ready.set()

    reader = None
    try:
        if a.mode not in ("solo", "glb-solo"):
            lp = subprocess.Popen(common + ["--role", "lp"], stdout=subprocess.PIPE, text=True, env=lp_env)
            reader = threading.Thread(target=drain, args=(lp.stdout,), daemon=True)
            reader.start()
            lp_ready.wait(timeout=a.timeout)
            if not any(l.startswith("LP_READY") for l in lp_lines):
                raise RuntimeError(f"LP process exited before it was ready (rc={lp.poll()})")
        hp = subprocess.run(common + ["--role", "hp"], stdout=subprocess.PIPE, text=True, env=hp_env, timeout=a.timeout)
        for line in hp.stdout.splitlines():
            if line.startswith("HP_RESULT "):
                res.update(json.loads(line[10:]))
        if "hp_exact" not in res:
            res["error"] = f"HP process rc={hp.returncode}, no result"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        open(stop_file, "w").close()
        if lp is not None:
            try:
                lp.wait(timeout=300)
                if reader is not None:
                    reader.join(timeout=30)
                for line in lp_lines:
                    if line.startswith("LP_RESULT "):
                        res.update(json.loads(line[10:]))
                if lp.returncode not in (0, None) and "lp_iters" not in res:
                    res["error"] = (res["error"] or "") + f" LP rc={lp.returncode}"
            except subprocess.TimeoutExpired:
                lp.kill()
                res["error"] = (res["error"] or "") + " LP did not stop"
        os.remove(stop_file)
        if xserver is not None:
            xserver.send_signal(signal.SIGINT)
            try:
                xserver.wait(timeout=30)
            except subprocess.TimeoutExpired:
                xserver.kill()
    with open(a.out, "a") as fo:
        fo.write(json.dumps(res) + "\n")
    ok = (res["error"] is None and res.get("hp_exact") == res.get("n_req") and not res.get("lp_bad_iters"))
    f = lambda k, s="{:.1f}": "-" if res.get(k) is None else s.format(res[k])
    print(f"[VLLM-MP {a.mode:7s} delay={a.hpf_resume_delay_us}us] {'PASS' if ok else 'FAIL'} | HP E2E p50 {f('e2e_ms_p50')} p99 {f('e2e_ms_p99')} "
          f"max {f('e2e_ms_max')} ms, TTFT p50 {f('ttft_ms_p50')} p99 {f('ttft_ms_p99')} ms, "
          f"exact {res.get('hp_exact')}/{res.get('n_req')} | LP iters {res.get('lp_iters', '-')} bad "
          f"{res.get('lp_bad_iters', '-')} tput {f('lp_tput_pct_of_solo')}% of solo | {res['error'] or ''}", flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "role"):
        p = sp.add_parser(name)
        p.add_argument("--mode", required=True, choices=["ref", "solo", "native", "glb-solo", "glb-lv1", "glb-lv3"])
        p.add_argument("--ref-dir", default="vllm_mp_ref")
        p.add_argument("--hp-model", default="Qwen/Qwen2.5-0.5B-Instruct")
        p.add_argument("--lp-model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
        p.add_argument("--hp-util", type=float, default=0.22)
        p.add_argument("--lp-util", type=float, default=0.55)
        p.add_argument("--n-req", type=int, default=40)
        p.add_argument("--mean-gap-ms", type=float, default=300.0)
        p.add_argument("--hp-tokens", type=int, default=16)
        p.add_argument("--lp-tokens", type=int, default=8)
        p.add_argument("--lp-batch", type=int, default=16)
        p.add_argument("--lp-doc-reps", type=int, default=40)
        p.add_argument("--eager", action="store_true")
        p.add_argument("--ctrl", default=f"/tmp/xsched-vllm-mp-{os.getpid()}")
        if name == "run":
            p.add_argument("--out", required=True)
            p.add_argument("--port", type=int, default=50001)
            p.add_argument("--timeout", type=float, default=1800)
            p.add_argument("--vllm-mp", action="store_true", help="vLLM's default multiprocess engine")
            p.add_argument("--lp-env", action="append", default=[])
            p.add_argument("--env", action="append", default=[], help="K=V for both processes and xserver")
            p.add_argument("--hpf-resume-delay-us", type=int, default=0, help="xserver HPF resume delay")
            p.add_argument("--native-ld-path", default=os.environ.get("NATIVE_LD_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu"))
        else:
            p.add_argument("--role", required=True, choices=["hp", "lp"])
    a = ap.parse_args()
    if a.cmd == "run":
        sys.exit(run(a))
    (role_hp if a.role == "hp" else role_lp)(a)


if __name__ == "__main__":
    main()
