#!/usr/bin/env python3
"""
Phase 3.4-2 -- multi-process HP/LP serving on one GPU (removes the in-process GIL interference of
Suite 8, OBS-7). Same workload as Suite 8 (real Qwen2.5-0.5B weights, chat requests with Poisson
arrivals, greedy decode; LP perplexity / prefill eval loop), but HP and LP are separate processes,
i.e. separate CUDA contexts that the GPU time-slices.

  modes: solo     HP process alone                                   -> latency floor
         native   HP + LP processes, no XSched
         req-lv1  LP on an XQueue (Level-1); before each HP request the HP process asks the LP process
         req-lv3  over a Unix socket to suspend its queue, and resumes it after the request
                  (application-level cross-process 'request' policy; same as Suite 8 'request')
         glb-lv1  XSched global scheduler: xserver (HPF) + XSCHED_SCHEDULER=GLB in both processes;
         glb-lv3  the HP stream is a Level-1 XQueue with priority 1, the LP queue has priority 0;
                  no application code in the scheduling path (fully transparent)

  Every HP token sequence must equal the solo golden; every LP iteration must be bit-exact.

  python bench_mp_serving_p34.py run --mode req-lv3 --lp ppl --out mp.jsonl
  python bench_mp_serving_p34.py report mp.jsonl [--md out.md]
"""
import argparse
import ctypes
import json
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


class Bridge:
    """ctypes access to the XSched C API (XQueue on a torch stream, suspend/resume, priority hint)."""

    def __init__(self):
        self.pre = ctypes.CDLL("libpreempt.so", mode=ctypes.RTLD_GLOBAL)
        self.hal = ctypes.CDLL("libhalcuda.so", mode=ctypes.RTLD_GLOBAL)
        self.hal.CudaQueueCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p]
        self.pre.XQueueCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.c_int, ctypes.c_int64]
        for f in ("XQueueSuspend", "XQueueResume"):
            getattr(self.pre, f).argtypes = [ctypes.c_uint64, ctypes.c_int64]
        self.pre.XHintPriority.argtypes = [ctypes.c_uint64, ctypes.c_int32]
        self.pre.XQueueDestroy.argtypes = [ctypes.c_uint64]
        self.pre.HwQueueDestroy.argtypes = [ctypes.c_uint64]

    def create(self, stream, level, prio=None):
        hwq, xq = ctypes.c_uint64(), ctypes.c_uint64()
        assert self.hal.CudaQueueCreate(ctypes.byref(hwq), ctypes.c_void_p(stream.cuda_stream)) == 0
        assert self.pre.XQueueCreate(ctypes.byref(xq), hwq, level, 0) == 0
        if prio is not None:
            assert self.pre.XHintPriority(xq, prio) == 0
        return hwq.value, xq.value

    def destroy(self, hwq, xq):
        self.pre.XQueueDestroy(xq)
        self.pre.HwQueueDestroy(hwq)


def load(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    return torch, tok, model


def send(conn, msg):
    conn.sendall((json.dumps(msg) + "\n").encode())


def recv(f):
    line = f.readline()
    if not line:
        raise RuntimeError("peer closed the control connection")
    return json.loads(line)


# ------------------------------------------------------------------------------------------------ LP
def lp_main(args):
    from bench_real_serving_p3 import DOC
    sys.setswitchinterval(0.0002)  # bound the GIL wait of the control thread while the LP loop runs
    torch, tok, model = load(args)
    lp = torch.cuda.Stream()
    doc_ids = tok(DOC, return_tensors="pt").input_ids[0]
    base = doc_ids.repeat(args.lp_seq // doc_ids.numel() + 1)[: args.lp_seq]
    lp_ids = torch.stack([torch.roll(base, shifts=97 * b) for b in range(args.lp_batch)]).cuda()

    @torch.no_grad()
    def lp_step():
        with torch.cuda.stream(lp):
            if args.lp == "ppl":
                return model(input_ids=lp_ids, labels=lp_ids).loss.detach().reshape(1).clone()
            return model(input_ids=lp_ids, use_cache=False, logits_to_keep=1).logits[:, -1, :].detach().clone()

    gold = lp_step(); lp.synchronize()
    gold2 = lp_step(); lp.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        lp_step()
    lp.synchronize()
    res = {"lp_solo_ms": (time.perf_counter() - t0) * 1e3 / 3, "lp_native_deterministic": bool(torch.equal(gold, gold2)),
           "lp_tokens_per_step": args.lp_batch * args.lp_seq}

    xs = hwq = xq = None
    if args.mode != "native":
        xs = Bridge()
        level = 1 if args.mode.endswith("lv1") else 3
        hwq, xq = xs.create(lp, level, prio=0 if args.mode.startswith("glb") else None)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind("\0" + args.ctrl)
    srv.listen(1)
    print("LP_READY", flush=True)
    conn, _ = srv.accept()
    f = conn.makefile("r")
    assert recv(f)["op"] == "start"

    stop = threading.Event()
    counters = {"iters": 0, "bad": 0, "exc": None}

    def loop():
        try:
            torch.cuda.set_device(0)
            while not stop.is_set():
                y = lp_step()
                lp.synchronize()
                if not torch.equal(y, gold):
                    counters["bad"] += 1
                counters["iters"] += 1
        except Exception as e:  # noqa
            counters["exc"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=loop, daemon=True)
    t_start = time.perf_counter()
    th.start()
    send(conn, {"op": "started"})
    iters_at = {}
    while True:
        m = recv(f)
        if m["op"] == "suspend":
            ts = time.perf_counter()
            assert xs.pre.XQueueSuspend(xq, 1) == 0
            send(conn, {"op": "ack", "lp_side_ms": (time.perf_counter() - ts) * 1e3})
        elif m["op"] == "resume":
            assert xs.pre.XQueueResume(xq, 0) == 0
            send(conn, {"op": "ack"})
        elif m["op"] == "window":          # HP marks the measured window (after warm-up / before drain)
            iters_at[m["mark"]] = (counters["iters"], time.perf_counter())
            send(conn, {"op": "ack"})
        elif m["op"] == "stop":
            break
    stop.set()
    th.join(timeout=120)
    (i0, t0w), (i1, t1w) = iters_at["begin"], iters_at["end"]
    res.update({"lp_iters": i1 - i0, "lp_window_s": t1w - t0w, "lp_bad_iters": counters["bad"], "lp_error": counters["exc"],
                "lp_total_iters": counters["iters"], "lp_total_s": time.perf_counter() - t_start})
    res["lp_tput_pct_of_solo"] = 100.0 * res["lp_iters"] * res["lp_solo_ms"] / 1000.0 / max(1e-9, res["lp_window_s"])
    if xq is not None:
        xs.destroy(hwq, xq)
    send(conn, {"op": "result", "res": res})
    conn.close()


# ------------------------------------------------------------------------------------------------ HP
def hp_main(args):
    from bench_real_serving_p3 import PROMPTS
    torch, tok, model = load(args)
    prop = torch.cuda.get_device_properties(0)
    res = {"mode": args.mode, "lp": args.lp, "model": args.model, "gpu": prop.name, "sm": prop.multi_processor_count,
           "host": socket.gethostname(), "processes": 1 if args.mode == "solo" else 2, "error": None}
    hp = torch.cuda.Stream(priority=-1)
    rng = random.Random(1234)
    reqs = []
    for i in range(args.n_req):
        text = tok.apply_chat_template([{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], tokenize=False,
                                       add_generation_prompt=True)
        reqs.append(tok(text, return_tensors="pt").input_ids.cuda())
    arrivals, t = [], 0.0
    for _ in range(args.n_req):
        t += rng.expovariate(1.0 / args.mean_gap_ms)
        arrivals.append(t / 1000.0)

    @torch.no_grad()
    def serve(ids):
        out, tpots = [], []
        item_ms = [0.0]

        def item(t):
            t0 = time.perf_counter()
            v = int(t.item())
            item_ms[0] += (time.perf_counter() - t0) * 1e3
            return v

        with torch.cuda.stream(hp):
            t0 = time.perf_counter()
            o = model(input_ids=ids, use_cache=True)
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            hp.synchronize()
            ttft = (time.perf_counter() - t0) * 1e3
            out.append(item(nxt))
            past = o.past_key_values
            for _ in range(args.new_tokens - 1):
                t1 = time.perf_counter()
                o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
                hp.synchronize()
                tpots.append((time.perf_counter() - t1) * 1e3)
                out.append(item(nxt))
        serve.item_ms = item_ms[0]
        return out, ttft, tpots

    n_gold = min(len(reqs), len(PROMPTS))
    [serve(ids) for ids in reqs[:n_gold]]                       # warm-up
    gold_runs = [serve(ids) for ids in reqs[:n_gold]]           # LP process idle (or absent)
    gold = [g[0] for g in gold_runs]
    res["solo_tpot_p50_ms"] = statistics.median([x for g in gold_runs for x in g[2]])
    res["solo_ttft_p50_ms"] = statistics.median([g[1] for g in gold_runs])

    xs = hwq = xq = None
    if args.mode.startswith("glb"):
        xs = Bridge()
        hwq, xq = xs.create(hp, 1, prio=1)    # the scheduler sees HP activity through this queue

    conn = f = None
    if args.mode != "solo":
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect("\0" + args.ctrl)
        f = conn.makefile("r")
        send(conn, {"op": "start"})
        assert recv(f)["op"] == "started"
        time.sleep(0.3)

    def ctl(op, **kw):
        t0 = time.perf_counter()
        send(conn, dict(op=op, **kw))
        m = recv(f)
        assert m["op"] == "ack", m
        return (time.perf_counter() - t0) * 1e3

    ttft, tpot, e2e, susp, hp_ok, items = [], [], [], [], 0, []
    try:
        if conn:
            ctl("window", mark="begin")
        t_start = time.perf_counter()
        for i, ids in enumerate(reqs):
            target = t_start + arrivals[i]
            while time.perf_counter() < target:
                time.sleep(0.0005)
            ta = time.perf_counter()
            if args.mode.startswith("req"):
                susp.append(ctl("suspend"))
            toks, tt, tp = serve(ids)
            if args.mode.startswith("req"):
                ctl("resume")
                tt += susp[-1]
            e2e.append((time.perf_counter() - ta) * 1e3)
            items.append(serve.item_ms)
            ttft.append(tt)
            tpot += tp
            hp_ok += int(toks == gold[i % n_gold])
        wall_s = time.perf_counter() - t_start
        if conn:
            ctl("window", mark="end")
    except Exception as e:  # noqa
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-1200:]
        wall_s = None
    if conn:
        send(conn, {"op": "stop"})
        m = recv(f)
        res.update(m["res"])
        conn.close()
        if res.get("lp_error"):
            res["error"] = (res["error"] or "") + " LP: " + res["lp_error"]
    if xq is not None:
        xs.destroy(hwq, xq)
    res.update({"n_req": len(reqs), "new_tokens": args.new_tokens, "mean_gap_ms": args.mean_gap_ms, "hp_exact": hp_ok,
                "tpot_ms_p50": pct(tpot, 0.5), "tpot_ms_p99": pct(tpot, 0.99), "ttft_ms_p50": pct(ttft, 0.5),
                "ttft_ms_p99": pct(ttft, 0.99), "e2e_ms_p50": pct(e2e, 0.5), "e2e_ms_p99": pct(e2e, 0.99),
                "susp_ms_p50": pct(susp, 0.5), "susp_ms_p99": pct(susp, 0.99), "wall_s": wall_s,
                "item_ms_per_req_p50": pct(items, 0.5), "e2e_ms_list": [round(x, 2) for x in e2e]})
    if wall_s:
        res["hp_duty_pct"] = 100.0 * sum(e2e) / 1000.0 / wall_s
    print("RESULT " + json.dumps(res), flush=True)


# ------------------------------------------------------------------------------------------------ run
def run(args):
    env = dict(os.environ)
    xserver = None
    if args.mode.startswith("glb"):
        env["XSCHED_SCHEDULER"] = "GLB"
        xbin = os.path.join(os.environ.get("XSCHED", ""), "bin", "xserver")
        # xserver's IPC channel is SysV message queue key 1. A stale one owned by another user (e.g.
        # left by a root run whose server was killed) cannot be replaced, and clients would block
        # sending into it: refuse to start instead of hanging.
        import pwd
        me = os.getuid()
        for line in subprocess.run(["ipcs", "-q"], capture_output=True, text=True).stdout.splitlines():
            f = line.split()
            if len(f) >= 3 and f[0] == "0x00000001":
                owner = f[2]
                if owner != pwd.getpwuid(me).pw_name and me != 0:
                    raise SystemExit(f"stale xserver message queue key 0x1 (msqid {f[1]}) owned by {owner}: "
                                     f"remove it with 'ipcrm -q {f[1]}' as its owner or root")
        xlog = open(args.out + ".xserver.log", "a")
        xlog.write(f"=== {time.strftime('%F %T')} mode={args.mode} lp={args.lp}\n")
        xlog.flush()
        xserver = subprocess.Popen([xbin, "HPF", str(args.port)], stdout=xlog, stderr=subprocess.STDOUT, env=env)
        time.sleep(1.0)
    else:
        env.pop("XSCHED_SCHEDULER", None)
    ctrl = f"xsched-mp-{os.getpid()}"
    common = [sys.executable, os.path.abspath(__file__), "role", "--mode", args.mode, "--lp", args.lp, "--model", args.model,
              "--ctrl", ctrl, "--n-req", str(args.n_req), "--new-tokens", str(args.new_tokens), "--mean-gap-ms",
              str(args.mean_gap_ms), "--lp-batch", str(args.lp_batch), "--lp-seq", str(args.lp_seq)]
    lp = None
    res = None
    try:
        if args.mode != "solo":
            lp = subprocess.Popen(common + ["--role", "lp"], stdout=subprocess.PIPE, text=True, env=env)
            for line in lp.stdout:
                if line.startswith("LP_READY"):
                    break
            else:
                raise RuntimeError("LP process exited before it was ready")
        hp = subprocess.run(common + ["--role", "hp"], stdout=subprocess.PIPE, text=True, env=env, timeout=args.timeout)
        for line in hp.stdout.splitlines():
            if line.startswith("RESULT "):
                res = json.loads(line[7:])
        if res is None:
            res = {"mode": args.mode, "lp": args.lp, "error": f"HP process rc={hp.returncode}, no result"}
    finally:
        if lp is not None:
            try:
                lp.wait(timeout=120)
            except subprocess.TimeoutExpired:
                lp.kill()
        if xserver is not None:
            xserver.send_signal(signal.SIGINT)   # SIGINT removes the IPC queues (older servers ignore SIGTERM)
            try:
                xserver.wait(timeout=30)
            except subprocess.TimeoutExpired:
                xserver.kill()
    with open(args.out, "a") as fo:
        fo.write(json.dumps(res) + "\n")
    ok = res.get("error") is None and res.get("hp_exact") == res.get("n_req") and not res.get("lp_bad_iters")
    f = lambda k, s="{:.2f}": "-" if res.get(k) is None else s.format(res[k])
    print(f"[S8-MP {args.mode:7s} {args.lp}] {'PASS' if ok else 'FAIL'} | TPOT p50 {f('tpot_ms_p50')} p99 {f('tpot_ms_p99')} "
          f"(solo {f('solo_tpot_p50_ms')}) ms | TTFT p50 {f('ttft_ms_p50')} | susp p50 {f('susp_ms_p50', '{:.3f}')} ms | "
          f"LP {f('lp_tput_pct_of_solo', '{:.0f}')}% | HP duty {f('hp_duty_pct', '{:.0f}')}% | exact HP "
          f"{res.get('hp_exact')}/{res.get('n_req')} LP bad {res.get('lp_bad_iters')} | {res.get('error') or ''}")
    return 0 if ok else 1


def report(paths, md=None):
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    out = ["| GPU | LP | 模式 | HP 占空比 | LP 吞吐（% solo） | TPOT P50 / P99 (ms) | 相对 solo P50 · P99 | TTFT P50 (ms) | 挂起 P50 (ms) | 正确性 |",
           "|---|---|---|---:|---:|---|---|---:|---:|---|"]
    g = lambda r, k, s="{:.2f}": "-" if r.get(k) is None else s.format(r[k])
    for r in rows:
        solo = r.get("solo_tpot_p50_ms") or 0
        rel = "-" if not solo or r.get("tpot_ms_p50") is None else f"{r['tpot_ms_p50'] / solo:.2f}× · {r['tpot_ms_p99'] / solo:.2f}×"
        gpu = (r.get("gpu") or "").replace("NVIDIA GeForce ", "")
        ok = f"{r.get('hp_exact')}/{r.get('n_req')}" + ("" if not r.get("lp_bad_iters") else f", LP 错 {r['lp_bad_iters']}")
        out.append(f"| {gpu} | {r.get('lp')} | {r.get('mode')} | {g(r, 'hp_duty_pct', '{:.0f}%')} | {g(r, 'lp_tput_pct_of_solo', '{:.0f}%')} | "
                   f"{g(r, 'tpot_ms_p50')} / {g(r, 'tpot_ms_p99')} | {rel} | {g(r, 'ttft_ms_p50')} | {g(r, 'susp_ms_p50', '{:.3f}')} | {ok} |")
    text = "\n".join(out)
    print(text)
    if md:
        open(md, "w").write(text + "\n")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "role"):
        r = sub.add_parser(name)
        r.add_argument("--mode", required=True, choices=["solo", "native", "req-lv1", "req-lv3", "glb-lv1", "glb-lv3"])
        r.add_argument("--lp", default="ppl", choices=["ppl", "prefill"])
        r.add_argument("--model", default=os.environ.get("REAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
        r.add_argument("--n-req", type=int, default=20)
        r.add_argument("--new-tokens", type=int, default=16)
        r.add_argument("--mean-gap-ms", type=float, default=500.0)
        r.add_argument("--lp-batch", type=int, default=None, help="default: 4 (prefill) / 2 (ppl), as Suite 8")
        r.add_argument("--lp-seq", type=int, default=None, help="default: 1024 (prefill) / 512 (ppl), as Suite 8")
        if name == "run":
            r.add_argument("--out", required=True)
            r.add_argument("--port", type=int, default=50000)
            r.add_argument("--timeout", type=int, default=600)
        else:
            r.add_argument("--role", required=True, choices=["lp", "hp"])
            r.add_argument("--ctrl", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("paths", nargs="+")
    rp.add_argument("--md")
    a = p.parse_args()
    if a.cmd == "report":
        return report(a.paths, a.md)
    if a.lp_batch is None:
        a.lp_batch = 4 if a.lp == "prefill" else 2
    if a.lp_seq is None:
        a.lp_seq = 1024 if a.lp == "prefill" else 512
    if a.cmd == "run":
        return run(a)
    return lp_main(a) if a.role == "lp" else hp_main(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
