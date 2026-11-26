#!/usr/bin/env python3
"""Phase 3.5 (tier-1 item 3): vLLM (in-process engine) on an XSched XQueue, preempted by a
high-priority kernel during generation.

Requires a vLLM installation (tested: vLLM 0.34.0, torch 2.14+cu130, RTX 5080) and the XSched shim
on LD_LIBRARY_PATH (libcuda.so.1 -> XSched). Example (as root for Level-3, or with xsched-bar0d):
  MODE=native REF=ref.json python test_vllm_lv3_p35.py        # writes the greedy reference
  MODE=lv3    REF=ref.json python test_vllm_lv3_p35.py        # Level-3, graphs flattened (default)
  MODE=lv3 XSCHED_CUDA_GRAPH_FLATTEN=0 REF=ref.json python test_vllm_lv3_p35.py
  MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ GPU_UTIL=0.75 PROMPT_REPEAT=300 TOKENS=16 ...   (prefill-heavy)


The engine runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0). vLLM executes and captures its
CUDA graphs on vllm.utils.current_stream(), which follows torch.cuda.set_stream, so the main thread's
current stream is set to a non-default stream bound to an XQueue before the engine is built (the
legacy default stream always bypasses XSched; graph capture on the XQueue stream is handled by the
shim's CaptureBegin/End, which drains the XQueue and switches to direct launch while capturing).

Phases (same process, after a full warm-up):
  1. generate without preemption           -> wall_nopre_s (XQueue / flattening overhead vs native)
  2. generate while a preemptor thread repeatedly: XQueueSuspend(xq, sync) -> HP matmul chain on a
     priority stream -> XQueueResume                -> wall_pre_s, suspend / HP latency
Greedy outputs of both phases must equal a native run's outputs (MODE=native writes REF).

env: MODE native|lv1|lv3  MODEL  TOKENS(128)  GPU_UTIL(0.35)  EAGER(0)  REF  NPROMPTS(32)  HP_GAP_MS(20)
"""
import ctypes, json, os, random, statistics, sys, threading, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch

import vllm.utils.torch_utils  # noqa: F401  (patches torch.cuda.set_stream; must precede set_stream)

mode = os.environ.get("MODE", "native")
level = {"lv1": 1, "lv3": 3}.get(mode)
torch.cuda.set_device(0)
s = torch.cuda.Stream()
# vLLM keeps its own thread-local "current stream" (vllm.utils.torch_utils.current_stream()); if it is
# unset at the first call, vLLM creates a new stream and switches to it (seen with the 7B model: the
# whole steady state then bypassed the XQueue). Setting the stream through vLLM's patched set_stream
# makes the XQueue stream vLLM's current stream.
torch.cuda.set_stream(s)
pre = hal = None
xq = None
if level:
    pre = ctypes.CDLL("libpreempt.so", mode=ctypes.RTLD_GLOBAL)
    hal = ctypes.CDLL("libhalcuda.so", mode=ctypes.RTLD_GLOBAL)
    hal.CudaQueueCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p]
    pre.XQueueCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.c_int, ctypes.c_int64]
    for f in ("XQueueSuspend", "XQueueResume"):
        getattr(pre, f).argtypes = [ctypes.c_uint64, ctypes.c_int64]
    hwq, xqh = ctypes.c_uint64(), ctypes.c_uint64()
    assert hal.CudaQueueCreate(ctypes.byref(hwq), ctypes.c_void_p(s.cuda_stream)) == 0
    assert pre.XQueueCreate(ctypes.byref(xqh), hwq, level, 0) == 0
    xq = xqh.value
    if os.environ.get("THRESH"):
        pre.XQueueSetLaunchConfig.argtypes = [ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64]
        assert pre.XQueueSetLaunchConfig(xq, int(os.environ["THRESH"]), int(os.environ.get("BATCH", "8"))) == 0

from vllm import LLM, SamplingParams

model = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
t_load = time.time()
llm = LLM(model=model, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.35")),
          max_model_len=int(os.environ.get("MAX_LEN", "4096")),
          enforce_eager=os.environ.get("EAGER", "0") == "1", seed=0)
t_load = time.time() - t_load
n = int(os.environ.get("NPROMPTS", "32"))
rep = int(os.environ.get("PROMPT_REPEAT", "0"))      # >0: long prompts (prefill-heavy)
prompts = [f"Write a detailed paragraph about topic number {i}: " + "The history of computing. " * (rep + i % 8 + 1)
           for i in range(n)]
sp = SamplingParams(temperature=0, max_tokens=int(os.environ.get("TOKENS", "128")))
llm.generate(prompts, sp, use_tqdm=False)       # full warm-up: every batch shape / kernel seen once


def gen():
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, [o.outputs[0].text for o in outs], sum(len(o.outputs[0].token_ids) for o in outs)


wall_nopre, texts_nopre, ntok = gen()
_mr = getattr(llm.llm_engine.engine_core, "engine_core", llm.llm_engine.engine_core).model_executor.driver_worker.worker.model_runner
assert _mr.main_stream.cuda_stream == s.cuda_stream, "vLLM runs on a stream other than the XQueue stream"
if os.environ.get("PROFILE") == "1":
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    gen()
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(22)
    print("PROFILE-" + mode, buf.getvalue(), flush=True)

hp = torch.cuda.Stream(priority=-1)
w_hp = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
x_hp = torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16)


def hp_fn():
    y = x_hp
    for _ in range(4):
        y = torch.mm(y, w_hp)
    return y


with torch.cuda.stream(hp):
    hp_gold = hp_fn().clone()
    hp.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        hp_fn()
    hp.synchronize()
    hp_alone_ms = (time.perf_counter() - t0) * 1e3 / 20

susp, hp_lat, hp_bad = [], [], 0
stop = threading.Event()
gap = float(os.environ.get("HP_GAP_MS", "20")) / 1e3


def preemptor():
    global hp_bad
    torch.cuda.set_device(0)
    rng = random.Random(1)
    while not stop.is_set():
        time.sleep(rng.uniform(0.5, 1.5) * gap)
        if stop.is_set():
            break
        t0 = time.perf_counter()
        if xq:
            pre.XQueueSuspend(xq, 1)
        t1 = time.perf_counter()
        with torch.cuda.stream(hp):
            y = hp_fn()
        hp.synchronize()
        t2 = time.perf_counter()
        if xq:
            pre.XQueueResume(xq, 0)
        susp.append((t1 - t0) * 1e3)
        hp_lat.append((t2 - t0) * 1e3)
        if not torch.equal(y, hp_gold):
            hp_bad += 1


th = threading.Thread(target=preemptor, daemon=True)
th.start()
wall_pre, texts_pre, _ = gen()
stop.set()
th.join()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else None


res = {"mode": mode, "model": model, "thresh": os.environ.get("THRESH", "16"), "knobs": os.environ.get("KNOBS", ""), "eager": os.environ.get("EAGER", "0") == "1",
       "flatten": os.environ.get("XSCHED_CUDA_GRAPH_FLATTEN", "1"), "flat_min_us": os.environ.get("XSCHED_CUDA_GRAPH_FLATTEN_MIN_US", "0"),
       "async_wait": os.environ.get("XSCHED_CUDA_ASYNC_EVENT_WAIT", "1"), "prompts": n, "tokens_out": ntok,
       "load_s": t_load, "wall_nopre_s": wall_nopre, "wall_pre_s": wall_pre, "tok_s_nopre": ntok / wall_nopre,
       "hp_alone_ms": hp_alone_ms, "hp_n": len(hp_lat), "hp_exact": len(hp_lat) - hp_bad,
       "susp_p50_ms": pct(susp, 0.5), "susp_p99_ms": pct(susp, 0.99), "susp_max_ms": max(susp) if susp else None,
       "hp_p50_ms": pct(hp_lat, 0.5), "hp_p99_ms": pct(hp_lat, 0.99), "hp_max_ms": max(hp_lat) if hp_lat else None}
ref_path = os.environ.get("REF", "vllm_ref.json")
if mode == "native":
    json.dump(texts_nopre, open(ref_path, "w"))
    res["exact_nopre"] = res["exact_pre"] = sum(a == b for a, b in zip(texts_pre, texts_nopre))
else:
    ref = json.load(open(ref_path))
    res["exact_nopre"] = sum(a == b for a, b in zip(texts_nopre, ref))
    res["exact_pre"] = sum(a == b for a, b in zip(texts_pre, ref))
if hal is not None:
    try:
        f = hal.XSm120GetRouteStats
        f.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int]
        buf = (ctypes.c_uint64 * 8)()
        f(buf, 8, 0)
        res["route"] = {k: v for k, v in zip(["lv3_killable", "lv1_mbarrier", "lv1_coop", "lv1_cluster", "lv1_graph",
                                              "lv1_uninstr", "lv1_short_cta", "lv1_glb"], list(buf)) if v}
    except Exception as e:  # noqa: BLE001
        res["route"] = str(e)
try:
    g = ctypes.CDLL("libcuda.so.1").XCudaGraphFlattenStats
    g.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int]
    buf = (ctypes.c_uint64 * 4)()
    g(buf, 4, 0)
    res["graph"] = dict(zip(["flat_launches", "whole_launches", "flat_cmds", "unflattenable"], list(buf)))
except Exception as e:  # noqa: BLE001
    res["graph"] = str(e)
try:
    g = ctypes.CDLL("libcuda.so.1").XCudaEventWaitStats
    g.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int]
    buf = (ctypes.c_uint64 * 4)()
    g(buf, 4, 0)
    res["event_wait"] = dict(zip(["lv1_gpu_wait", "promoted_waits", "blocking", "promotions"], list(buf)))
except Exception as e:  # noqa: BLE001
    res["event_wait"] = str(e)
res["prompt_repeat"] = rep
print("VLLM-LV3", json.dumps(res), flush=True)
if os.environ.get("OUT"):
    with open(os.environ["OUT"], "a") as fo:
        fo.write(json.dumps(res) + "\n")
