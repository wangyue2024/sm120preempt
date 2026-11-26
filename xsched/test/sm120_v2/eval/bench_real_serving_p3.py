#!/usr/bin/env python3
"""
Suite 8 -- Grounded real-model serving co-location benchmark (RTX 5060 / RTX 5080, sm_120).

One process per mode, unmodified HuggingFace stack with REAL pretrained weights
(default Qwen/Qwen2.5-0.5B-Instruct, loaded offline from the HF cache):

  HP (online, latency-critical): chat requests with Poisson arrivals; chat template, prefill +
      greedy decode with KV cache (manual loop so TTFT / per-token TPOT are measured exactly).
      Every generated token sequence must EXACTLY match a solo golden run.
  LP (offline, throughput):      continuous batch evaluation on the SAME weights:
      --lp ppl     : perplexity eval (model(input_ids, labels=...) -> HF upcasts logits to FP32,
                     i.e. large dtype-cast kernels, exactly as real eval pipelines do)
      --lp prefill : prefill-only scoring of the last position (cast-light variant)
      Loss / logits must be bit-exact vs a solo golden on EVERY iteration.

  modes: solo   (HP alone, no LP)              -> latency floor
         native (HP + LP co-run, priority stream only, no scheduler)
         lv1    (LP on XQueue Level-1; suspend LP on HP arrival, resume after the request)
         lv3    (LP on XQueue Level-3 SM120 trap engine, same policy)

  python bench_real_serving_p3.py run --mode lv3 --lp ppl --out serving.jsonl
  python bench_real_serving_p3.py report serving.jsonl [--md out.md]
"""
import argparse
import ctypes
import json
import math
import os
import random
import socket
import statistics
import threading
import time
import traceback

import torch

PROMPTS = [
    "Explain the difference between a process and a thread in an operating system.",
    "Write a short Python function that checks whether a string is a palindrome.",
    "What are the main causes of tail latency in large-scale online services?",
    "Summarize the idea of preemptive scheduling on GPUs in three sentences.",
    "Give me three tips for writing clear technical documentation.",
    "How does a KV cache speed up autoregressive decoding in transformers?",
    "Translate to French: The experiment finished successfully and the results are reproducible.",
    "What is the difference between throughput and latency? Give an example.",
    "List the steps to safely update a production database schema.",
    "Why do GPUs use SIMT execution, and what is warp divergence?",
]

DOC = (
    "Modern accelerators are shared by many workloads with very different service level objectives. "
    "Interactive applications such as chat assistants, code completion and search ranking require "
    "responses within tens of milliseconds, while offline jobs such as batch evaluation, embedding "
    "generation and fine-tuning care mostly about aggregate throughput. When both kinds of jobs are "
    "placed on the same device, the latency-critical requests can be blocked behind long running "
    "kernels of the throughput-oriented jobs. Operating systems solved a similar problem for CPUs "
    "decades ago with preemptive multitasking: the scheduler can interrupt a running task, save its "
    "context, run a more urgent task, and later restore the interrupted one exactly where it stopped. "
    "Graphics processors historically lacked fine grained preemption for general compute work, so "
    "schedulers had to wait for kernel boundaries or kill and re-execute idempotent kernels. "
    "Non-idempotent kernels, which update their inputs in place or accumulate results with atomic "
    "operations, cannot simply be re-executed, because replaying them would apply the same update "
    "twice. A practical preemption mechanism therefore has to capture the complete architectural "
    "state of every interrupted thread block, including registers, shared memory, barriers and "
    "predicates, and restore it transparently without changing the application or the libraries. "
)


class XSchedBridge:
    def __init__(self):
        self.pre = ctypes.CDLL("libpreempt.so", mode=ctypes.RTLD_GLOBAL)
        self.hal = ctypes.CDLL("libhalcuda.so", mode=ctypes.RTLD_GLOBAL)
        self.hal.CudaQueueCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p]
        self.pre.XQueueCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
                                          ctypes.c_int, ctypes.c_int]
        for f in ("XQueueSuspend", "XQueueResume"):
            getattr(self.pre, f).argtypes = [ctypes.c_uint64, ctypes.c_int]
        self.pre.XQueueDestroy.argtypes = [ctypes.c_uint64]
        self.pre.HwQueueDestroy.argtypes = [ctypes.c_uint64]

    def create(self, stream, level):
        hwq, xq = ctypes.c_uint64(), ctypes.c_uint64()
        assert self.hal.CudaQueueCreate(ctypes.byref(hwq), ctypes.c_void_p(stream.cuda_stream)) == 0
        assert self.pre.XQueueCreate(ctypes.byref(xq), hwq, level, 0) == 0
        return hwq.value, xq.value

    def suspend(self, xq):
        assert self.pre.XQueueSuspend(xq, 1) == 0

    def resume(self, xq):
        assert self.pre.XQueueResume(xq, 0) == 0

    def destroy(self, hwq, xq):
        self.pre.XQueueDestroy(xq)
        self.pre.HwQueueDestroy(hwq)


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))]


class _CUdevSmResource(ctypes.Structure):
    _fields_ = [("smCount", ctypes.c_uint)]


class _CUdevResource(ctypes.Structure):
    # CUdevResource: int type; 92 B internal padding; union { CUdevSmResource sm; 48 B }.
    _fields_ = [("type", ctypes.c_int), ("_pad", ctypes.c_ubyte * 92), ("sm", _CUdevSmResource),
                ("_tail", ctypes.c_ubyte * 48)]


def make_green_stream(n_sms):
    """Create a CUDA green context with `n_sms` SMs (spatial partition) and a non-blocking stream in it.
    Returns (granted_sms, total_sms, stream_ptr). Kernels launched on the stream run only on those SMs."""
    cu = ctypes.CDLL("libcuda.so.1")

    def chk(r, what):
        if r != 0:
            raise RuntimeError(f"{what} failed: {r}")

    dev = ctypes.c_int(torch.cuda.current_device())
    res = _CUdevResource()
    chk(cu.cuDeviceGetDevResource(dev, ctypes.byref(res), 1), "cuDeviceGetDevResource")  # 1 = SM
    groups = (_CUdevResource * 1)()
    nb = ctypes.c_uint(1)
    rem = _CUdevResource()
    chk(cu.cuDevSmResourceSplitByCount(groups, ctypes.byref(nb), ctypes.byref(res), ctypes.byref(rem), 0,
                                       ctypes.c_uint(n_sms)), "cuDevSmResourceSplitByCount")
    desc = ctypes.c_void_p()
    chk(cu.cuDevResourceGenerateDesc(ctypes.byref(desc), groups, 1), "cuDevResourceGenerateDesc")
    gctx = ctypes.c_void_p()
    chk(cu.cuGreenCtxCreate(ctypes.byref(gctx), desc, dev, 1), "cuGreenCtxCreate")  # 1 = default stream
    stream = ctypes.c_void_p()
    chk(cu.cuGreenCtxStreamCreate(ctypes.byref(stream), gctx, 1, 0), "cuGreenCtxStreamCreate")  # non-blocking
    make_green_stream.keep = getattr(make_green_stream, "keep", []) + [gctx]
    return groups[0].sm.smCount, res.sm.smCount, stream.value


def run(args):
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.switch_interval_ms > 0:
        sys.setswitchinterval(args.switch_interval_ms / 1000.0)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    prop = torch.cuda.get_device_properties(0)
    res = {"mode": args.mode, "policy": args.policy if args.mode in ("lv1", "lv3") else "-", "lp": args.lp, "model": args.model, "gpu": prop.name,
           "sm": prop.multi_processor_count, "host": socket.gethostname(), "error": None}

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    lp = torch.cuda.Stream()
    hp = torch.cuda.Stream(priority=-1)

    # ---------------- HP requests (real chat prompts) ----------------
    rng = random.Random(1234)
    reqs = []
    for i in range(args.n_req):
        msgs = [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        reqs.append(ids)
    arrivals, t = [], 0.0
    for _ in range(args.n_req):
        t += rng.expovariate(1.0 / args.mean_gap_ms)
        arrivals.append(t / 1000.0)

    @torch.no_grad()
    def serve(ids, pre_prefill=None, post_prefill=None, pre_step=None, post_step=None):
        """Greedy decode on the HP stream; returns (tokens, ttft_ms, [tpot_ms]).
        Optional hooks run around the prefill / each decode step (scheduling policies);
        timers start BEFORE the pre-hook so suspend latency is part of TTFT / TPOT."""
        out = []
        with torch.cuda.stream(hp):
            t0 = time.perf_counter()
            if pre_prefill: pre_prefill()
            o = model(input_ids=ids, use_cache=True)
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            hp.synchronize()
            ttft = (time.perf_counter() - t0) * 1e3
            if post_prefill: post_prefill()
            out.append(int(nxt.item()))
            past = o.past_key_values
            tpots = []
            for _ in range(args.new_tokens - 1):
                t1 = time.perf_counter()
                if pre_step: pre_step()
                o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
                hp.synchronize()
                tpots.append((time.perf_counter() - t1) * 1e3)
                if post_step: post_step(tpots[-1])
                out.append(int(nxt.item()))
        return out, ttft, tpots

    # ---------------- LP batch (real document text) ----------------
    doc_ids = tok(DOC, return_tensors="pt").input_ids[0]
    reps = (args.lp_seq // doc_ids.numel()) + 1
    base = doc_ids.repeat(reps)[: args.lp_seq]
    lp_ids = torch.stack([torch.roll(base, shifts=97 * b) for b in range(args.lp_batch)]).cuda()

    @torch.no_grad()
    def lp_step():
        with torch.cuda.stream(lp):
            if args.lp == "ppl":
                o = model(input_ids=lp_ids, labels=lp_ids)
                return o.loss.detach().reshape(1).clone()
            o = model(input_ids=lp_ids, use_cache=False, logits_to_keep=1)
            return o.logits[:, -1, :].detach().clone()

    # Goldens (solo, before any XQueue exists)
    hp_gold = [serve(ids)[0] for ids in reqs[: min(len(reqs), len(PROMPTS))]]  # warm + golden
    gold_runs = [serve(ids) for ids in reqs[: min(len(reqs), len(PROMPTS))]]
    hp_gold = [g[0] for g in gold_runs]
    solo_tpot_p50 = statistics.median([t for g in gold_runs for t in g[2]])
    slo_ms = args.tpot_slo_ms if args.tpot_slo_ms > 0 else 1.15 * solo_tpot_p50
    res["solo_tpot_p50_ms"] = solo_tpot_p50
    res["tpot_slo_ms"] = slo_ms
    lp_gold = lp_step(); lp.synchronize()
    lp_gold2 = lp_step(); lp.synchronize()
    res["lp_native_deterministic"] = bool(torch.equal(lp_gold, lp_gold2))
    t0 = time.perf_counter()
    for _ in range(3):
        lp_step()
    lp.synchronize()
    lp_solo_ms = (time.perf_counter() - t0) * 1e3 / 3
    res["lp_solo_ms"] = lp_solo_ms
    res["lp_tokens_per_step"] = args.lp_batch * args.lp_seq

    # Spatial sharing (Phase 3.3): after goldens and the LP solo reference were measured on the full
    # GPU, move LP to a green context holding --lp-sms SMs; HP keeps the remaining SMs to itself.
    res["lp_sms"] = 0
    res["switch_interval_ms"] = args.switch_interval_ms or 5.0
    if args.lp_sms > 0:
        if args.mode in ("lv1", "lv3"):
            raise SystemExit("--lp-sms is supported with --mode native / solo only")
        got, total, sptr = make_green_stream(args.lp_sms)
        lp = torch.cuda.ExternalStream(sptr)
        res["lp_sms"], res["total_sms"] = got, total

    xs = hwq = xq = None
    if args.mode in ("lv1", "lv3"):
        xs = XSchedBridge()
        hwq, xq = xs.create(lp, 1 if args.mode == "lv1" else 3)

    stop = threading.Event()
    lp_iters, lp_bad, lp_exc = [0], [0], []

    def lp_worker():
        try:
            torch.cuda.set_device(0)
            while not stop.is_set():
                y = lp_step()
                lp.synchronize()
                if not torch.equal(y, lp_gold):
                    lp_bad[0] += 1
                lp_iters[0] += 1
        except Exception as e:  # noqa
            lp_exc.append(f"{type(e).__name__}: {e}")

    th = None
    if args.mode != "solo":
        th = threading.Thread(target=lp_worker, daemon=True)
        th.start()
        time.sleep(0.2)

    ttft, tpot, e2e, susp, hp_ok = [], [], [], [], 0
    try:
        t_start = time.perf_counter()
        for i, ids in enumerate(reqs):
            target = t_start + arrivals[i]
            while time.perf_counter() < target:
                time.sleep(0.0005)
            ta = time.perf_counter()

            def do_suspend():
                t_s = time.perf_counter()
                xs.suspend(xq)
                susp.append((time.perf_counter() - t_s) * 1e3)

            def do_resume(*_):
                xs.resume(xq)

            slo_state = {"suspended": False}

            def slo_post_step(step_ms):
                # SLO-aware: co-run during decode; once a step misses the TPOT SLO, keep the LP
                # suspended for the rest of this request.
                if not slo_state["suspended"] and step_ms > slo_ms:
                    do_suspend()
                    slo_state["suspended"] = True

            if xq is None:
                toks, tt, tp = serve(ids)
            elif args.policy == "request":      # LP suspended for the whole HP request
                do_suspend()
                toks, tt, tp = serve(ids)
                do_resume()
                tt += susp[-1]
            elif args.policy == "prefill":      # suspend only for the compute-heavy prefill
                toks, tt, tp = serve(ids, pre_prefill=do_suspend, post_prefill=do_resume)
            elif args.policy == "token":        # suspend around every GPU step
                toks, tt, tp = serve(ids, pre_prefill=do_suspend, post_prefill=do_resume,
                                     pre_step=do_suspend, post_step=do_resume)
            else:                               # "slo": prefill suspended; decode co-runs until an SLO miss
                toks, tt, tp = serve(ids, pre_prefill=do_suspend, post_prefill=do_resume,
                                     post_step=slo_post_step)
                if slo_state["suspended"]:
                    do_resume()
            e2e.append((time.perf_counter() - ta) * 1e3)
            ttft.append(tt)
            tpot += tp
            if i < len(hp_gold) and toks == hp_gold[i]:
                hp_ok += 1
            elif i >= len(hp_gold) and toks == hp_gold[i % len(hp_gold)]:
                hp_ok += 1
        wall_s = time.perf_counter() - t_start
    except Exception as e:  # noqa
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-1200:]
        wall_s = None
    finally:
        stop.set()
        if th is not None:
            th.join(timeout=60)
        if xq is not None:
            xs.destroy(hwq, xq)

    if lp_exc:
        res["error"] = (res["error"] or "") + " LP: " + lp_exc[0]
    res.update({
        "n_req": len(reqs), "new_tokens": args.new_tokens, "mean_gap_ms": args.mean_gap_ms, "hp_exact": hp_ok,
        "ttft_ms": ttft, "tpot_ms_p50": pct(tpot, 0.5), "tpot_ms_p99": pct(tpot, 0.99),
        "ttft_ms_p50": pct(ttft, 0.5), "ttft_ms_p99": pct(ttft, 0.99),
        "e2e_ms_p50": pct(e2e, 0.5), "e2e_ms_p99": pct(e2e, 0.99),
        "susp_ms_p50": pct(susp, 0.5), "susp_ms_p99": pct(susp, 0.99),
        "lp_iters": lp_iters[0], "lp_bad_iters": lp_bad[0], "wall_s": wall_s,
    })
    if wall_s and args.mode != "solo":
        busy_hp_s = sum(e2e) / 1000.0
        res["hp_duty_pct"] = 100.0 * busy_hp_s / wall_s
        res["lp_tok_per_s"] = lp_iters[0] * args.lp_batch * args.lp_seq / wall_s
        res["lp_solo_tok_per_s"] = args.lp_batch * args.lp_seq / (lp_solo_ms / 1000.0)
        # LP throughput over the whole run as % of LP-alone throughput (upper bound 100%).
        res["lp_tput_pct_of_solo"] = 100.0 * res["lp_tok_per_s"] / res["lp_solo_tok_per_s"]
        # LP throughput relative to the GPU time NOT used by HP requests (100% = no waste).
        idle_s = max(1e-9, wall_s - busy_hp_s)
        res["lp_retention_pct"] = 100.0 * lp_iters[0] * lp_solo_ms / 1000.0 / idle_s
    return res


def fmt(v, f="{:.2f}"):
    return "-" if v is None else f.format(v)


def report(paths, md=None):
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    lines = []
    for gpu in sorted({r["gpu"] for r in rows}):
        for lpk in sorted({r["lp"] for r in rows if r["gpu"] == gpu}):
            sub = [r for r in rows if r["gpu"] == gpu and r["lp"] == lpk]
            lines.append(f"\n#### {gpu} — LP = `{lpk}` ({sub[0]['model']})\n")
            lines.append("| mode | TTFT P50/P99 ms | TPOT P50/P99 ms | E2E P50/P99 ms | suspend P50/P99 ms | HP exact | LP iters (bad) | LP tput % of solo | LP use of HP-idle time | status |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            keys = [("solo", "-"), ("native", "-")] + [(m, pol) for m in ("lv1", "lv3")
                                                        for pol in ("request", "prefill", "token", "slo")]
            for m, pol in keys:
                r = next((x for x in sub if x["mode"] == m and x.get("policy", "request" if m in ("lv1", "lv3") else "-") == pol), None)
                if r is None:
                    if pol in ("-", "request"):
                        lines.append(f"| {m} | | | | | | | | | CRASH / no result |")
                    continue
                m = m if pol == "-" else f"{m}/{pol}"
                st = "ERR: " + r["error"][:60] if r.get("error") else (
                    "PASS" if r["hp_exact"] == r["n_req"] and r.get("lp_bad_iters", 0) == 0 else "FAIL")
                lines.append(
                    f"| {m} | {fmt(r.get('ttft_ms_p50'))} / {fmt(r.get('ttft_ms_p99'))} | "
                    f"{fmt(r.get('tpot_ms_p50'))} / {fmt(r.get('tpot_ms_p99'))} | "
                    f"{fmt(r.get('e2e_ms_p50'))} / {fmt(r.get('e2e_ms_p99'))} | "
                    f"{fmt(r.get('susp_ms_p50'), '{:.3f}')} / {fmt(r.get('susp_ms_p99'), '{:.3f}')} | "
                    f"{r['hp_exact']}/{r['n_req']} | {r.get('lp_iters', '-')} ({r.get('lp_bad_iters', '-')}) | "
                    f"{fmt(r.get('lp_tput_pct_of_solo'), '{:.1f}%')} | {fmt(r.get('lp_retention_pct'), '{:.1f}%')} | {st} |")
    text = "\n".join(lines)
    print(text)
    if md:
        with open(md, "w") as fh:
            fh.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--mode", required=True, choices=["solo", "native", "lv1", "lv3"])
    r.add_argument("--lp", default="ppl", choices=["ppl", "prefill"])
    r.add_argument("--policy", default="request", choices=["request", "prefill", "token", "slo"],
                   help="lv1/lv3 suspension policy: whole request / prefill only / every GPU step / "
                        "SLO-aware (prefill suspended, decode co-runs until a step misses --tpot-slo-ms)")
    r.add_argument("--tpot-slo-ms", type=float, default=0.0,
                   help="TPOT SLO for --policy slo (default: 1.15 x measured solo TPOT P50)")
    r.add_argument("--model", default=os.environ.get("REAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    r.add_argument("--n-req", type=int, default=20)
    r.add_argument("--new-tokens", type=int, default=16)
    r.add_argument("--mean-gap-ms", type=float, default=500.0)
    r.add_argument("--lp-batch", type=int, default=None, help="default: 4 (prefill) / 2 (ppl)")
    r.add_argument("--lp-seq", type=int, default=None, help="default: 1024 (prefill) / 512 (ppl)")
    r.add_argument("--switch-interval-ms", type=float, default=0.0,
                   help="Python GIL switch interval (HP and LP share one process in this bench); 0 = default 5 ms")
    r.add_argument("--lp-sms", type=int, default=0,
                   help="run LP on a CUDA green context with this many SMs (spatial sharing; native mode)")
    r.add_argument("--out", required=True)
    rp = sp.add_parser("report")
    rp.add_argument("paths", nargs="+")
    rp.add_argument("--md")
    a = ap.parse_args()
    if a.cmd == "run":
        a.lp_batch = a.lp_batch or (2 if a.lp == "ppl" else 4)
        a.lp_seq = a.lp_seq or (512 if a.lp == "ppl" else 1024)
    if a.cmd == "report":
        report(a.paths, a.md)
        return
    res = run(a)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    st = "ERR " + res["error"] if res.get("error") else (
        "PASS" if res["hp_exact"] == res["n_req"] and res.get("lp_bad_iters", 0) == 0 else "FAIL")
    print(f"[SERVE mode={a.mode} policy={res.get('policy')} lp={a.lp}{(' lp_sms=' + str(res.get('lp_sms'))) if res.get('lp_sms') else ''}] {st} TTFT p50/p99={fmt(res.get('ttft_ms_p50'))}/{fmt(res.get('ttft_ms_p99'))}ms "
          f"TPOT p50/p99={fmt(res.get('tpot_ms_p50'))}/{fmt(res.get('tpot_ms_p99'))}ms hp_exact={res['hp_exact']}/{res['n_req']} "
          f"lp_iters={res.get('lp_iters')} bad={res.get('lp_bad_iters')} lp_tput={fmt(res.get('lp_tput_pct_of_solo'), '{:.1f}%')} "
          f"hp_duty={fmt(res.get('hp_duty_pct'), '{:.0f}%')}",
          flush=True)


if __name__ == "__main__":
    main()
