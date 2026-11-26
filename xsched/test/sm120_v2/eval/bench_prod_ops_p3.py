#!/usr/bin/env python3
"""
Suite 7 -- Production Operator Performance & Preemption Benchmark (RTX 5060 / RTX 5080, sm_120).

Measures, for ~27 hot operators of production LLM / DNN inference & training stacks
(cuBLAS / cuBLASLt FP8 / FlashAttention SDPA / cuDNN / CUB / Triton / fused optimizers):

  * steady-state per-call latency   (batch of back-to-back calls, wall clock incl. host path)
  * single-call latency             (submit 1 + sync: launch-path overhead, matters for decode)
  * Level-3 routing coverage        (kernels per call: Lv3-killable vs routed to Lv1 boundary)
  * preemption behaviour            (LP job ~60 ms of the op, mid-job suspend -> HP probe -> resume):
        suspend latency, HP end-to-end latency, LP job inflation
  * correctness                     (every LP call checked bit-exactly on-GPU for out-of-place ops;
                                     final state bit-exact for in-place non-idempotent ops; golden
                                     SHA-256 digest recorded for cross-mode / cross-process checks)

Each (op, mode) runs in its own process so a fatal CUDA error cannot poison other measurements:

  python bench_prod_ops_p3.py run  --op gemm_bf16_4k --mode lv3 --out results.jsonl
  python bench_prod_ops_p3.py list
  python bench_prod_ops_p3.py report results.jsonl [--md out.md]

Modes:
  native : XSched libraries NOT loaded (caller must also keep the shim off LD_LIBRARY_PATH)
  shim   : XSched shim intercepts the driver API, but no XQueue (pure interception overhead)
  lv1    : XQueue Level-1 (block) on the LP stream
  lv3    : XQueue Level-3 (SM120 trap engine, Mode 5 default + adaptive router)
In native/shim modes the preemption test degrades to "co-run": the HP probe is issued on a
high-priority stream while the LP job keeps running (the baseline the scheduler must beat).
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
import socket
import statistics
import sys
import threading
import time
import traceback

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

ROUTE_NAMES = ["lv3_killable", "lv1_mbarrier", "lv1_cooperative", "lv1_cluster", "lv1_graph",
               "lv1_uninstrumentable", "lv1_short_cta"]


# =====================================================================================
# XSched bridge (ctypes)
# =====================================================================================
class XSchedBridge:
    kPreemptLevelBlock = 1
    kPreemptLevelInterrupt = 3
    kQueueSuspendFlagSyncHwQueue = 1

    def __init__(self):
        self.libpreempt = ctypes.CDLL("libpreempt.so", mode=ctypes.RTLD_GLOBAL)
        self.libhal = ctypes.CDLL("libhalcuda.so", mode=ctypes.RTLD_GLOBAL)
        self.libhal.CudaQueueCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p]
        self.libhal.CudaQueueCreate.restype = ctypes.c_int
        self.libpreempt.HwQueueDestroy.argtypes = [ctypes.c_uint64]
        self.libpreempt.XQueueCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
                                                 ctypes.c_int, ctypes.c_int]
        self.libpreempt.XQueueCreate.restype = ctypes.c_int
        self.libpreempt.XQueueDestroy.argtypes = [ctypes.c_uint64]
        self.libpreempt.XQueueSuspend.argtypes = [ctypes.c_uint64, ctypes.c_int]
        self.libpreempt.XQueueSuspend.restype = ctypes.c_int
        self.libpreempt.XQueueResume.argtypes = [ctypes.c_uint64, ctypes.c_int]
        self.libpreempt.XQueueResume.restype = ctypes.c_int
        self.route_fn = getattr(self.libhal, "XSm120GetRouteStats", None)
        if self.route_fn is not None:
            self.route_fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int]
            self.route_fn.restype = ctypes.c_int

    def create_xqueue(self, stream, level):
        hwq, xq = ctypes.c_uint64(0), ctypes.c_uint64(0)
        if self.libhal.CudaQueueCreate(ctypes.byref(hwq), ctypes.c_void_p(stream.cuda_stream)) != 0:
            raise RuntimeError("CudaQueueCreate failed")
        if self.libpreempt.XQueueCreate(ctypes.byref(xq), hwq, level, 0) != 0:
            raise RuntimeError("XQueueCreate failed")
        return hwq.value, xq.value

    def suspend(self, xq):
        if self.libpreempt.XQueueSuspend(xq, self.kQueueSuspendFlagSyncHwQueue) != 0:
            raise RuntimeError("XQueueSuspend failed")

    def resume(self, xq):
        if self.libpreempt.XQueueResume(xq, 0) != 0:
            raise RuntimeError("XQueueResume failed")

    def destroy(self, hwq, xq):
        self.libpreempt.XQueueDestroy(xq)
        self.libpreempt.HwQueueDestroy(hwq)

    def route_stats(self, reset=False):
        if self.route_fn is None:
            return None
        buf = (ctypes.c_uint64 * len(ROUTE_NAMES))()
        n = self.route_fn(buf, len(ROUTE_NAMES), 1 if reset else 0)
        return {ROUTE_NAMES[i]: int(buf[i]) for i in range(n)}


# =====================================================================================
# Operator zoo
# =====================================================================================
class Op:
    """An operator instance.  run() performs ONE production call on the current stream.
    Out-of-place ops: run() returns the output tensor (checked against golden every call).
    In-place ops (inplace=True): run() mutates state; state() returns the tensors to compare,
    reset() restores the initial state."""
    inplace = False

    def run(self):
        raise NotImplementedError

    def state(self):
        return []

    def reset(self):
        pass


OPS = {}


def register(name, cat, desc):
    def deco(builder):
        OPS[name] = (cat, desc, builder)
        return builder
    return deco


def _rand(*shape, dtype=torch.bfloat16, scale=1.0):
    return (torch.randn(*shape, device="cuda", dtype=torch.float32) * scale).to(dtype)


class Simple(Op):
    def __init__(self, fn):
        self.fn = fn

    def run(self):
        return self.fn()


# ---------------- GEMM / Linear (cuBLAS, cuBLASLt) ----------------
@register("gemm_bf16_4k", "gemm", "cuBLAS BF16 GEMM 4096x4096x4096 (prefill / training)")
def _b():
    a, b = _rand(4096, 4096, scale=0.05), _rand(4096, 4096, scale=0.05)
    return Simple(lambda: torch.mm(a, b))


@register("gemm_bf16_decode", "gemm", "cuBLAS BF16 skinny GEMM 16x4096 @ 4096x14336 (decode FFN)")
def _b():
    a, w = _rand(16, 4096, scale=0.05), _rand(4096, 14336, scale=0.05)
    return Simple(lambda: torch.mm(a, w))


@register("linear_bias_ffn", "gemm", "F.linear+bias BF16 [2048x4096] x [11008x4096]^T (FFN up-proj)")
def _b():
    x, w, bias = _rand(2048, 4096, scale=0.05), _rand(11008, 4096, scale=0.05), _rand(11008)
    return Simple(lambda: F.linear(x, w, bias))


@register("gemm_fp8_e4m3", "gemm", "cuBLASLt FP8-E4M3 _scaled_mm 4096^3 -> BF16 (Blackwell FP8 TC)")
def _b():
    a = _rand(4096, 4096, dtype=torch.float32, scale=0.5).to(torch.float8_e4m3fn)
    w = _rand(4096, 4096, dtype=torch.float32, scale=0.5).to(torch.float8_e4m3fn)
    sa = torch.tensor(1.0, device="cuda")
    sb = torch.tensor(1.0, device="cuda")
    return Simple(lambda: torch._scaled_mm(a, w.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))


@register("moe_bmm_8x", "gemm", "MoE expert batched GEMM bmm [8,512,4096] x [8,4096,1408]")
def _b():
    x, w = _rand(8, 512, 4096, scale=0.05), _rand(8, 4096, 1408, scale=0.05)
    return Simple(lambda: torch.bmm(x, w))


@register("int8_mm", "gemm", "INT8 _int_mm 2048x4096x4096 -> INT32 (W8A8 serving)")
def _b():
    a = torch.randint(-64, 64, (2048, 4096), device="cuda", dtype=torch.int8)
    b = torch.randint(-64, 64, (4096, 4096), device="cuda", dtype=torch.int8)
    return Simple(lambda: torch._int_mm(a, b))


# ---------------- Attention ----------------
@register("sdpa_prefill_causal", "attention", "SDPA (Flash) causal B=1 H=32 S=2048 D=128 BF16")
def _b():
    q, k, v = (_rand(1, 32, 2048, 128) for _ in range(3))
    return Simple(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))


@register("sdpa_decode_kv4k", "attention", "SDPA decode q_len=1, B=8 H=32 KV=4096 D=128 BF16")
def _b():
    q = _rand(8, 32, 1, 128)
    k, v = _rand(8, 32, 4096, 128), _rand(8, 32, 4096, 128)
    return Simple(lambda: F.scaled_dot_product_attention(q, k, v))


@register("rope_apply", "attention", "RoPE rotate_half apply (HF style) [1,32,4096,128] BF16")
def _b():
    q = _rand(1, 32, 4096, 128)
    pos = torch.arange(4096, device="cuda", dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, 128, 2, device="cuda", dtype=torch.float32) / 128))
    fr = torch.outer(pos, inv)
    emb = torch.cat((fr, fr), dim=-1)
    cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    def rot(x):
        x1, x2 = x[..., :64], x[..., 64:]
        return torch.cat((-x2, x1), dim=-1)
    return Simple(lambda: q * cos + rot(q) * sin)


# ---------------- Normalization / activation (memory-bound) ----------------
@register("rmsnorm", "norm_act", "F.rms_norm [8192x4096] BF16")
def _b():
    x, w = _rand(8192, 4096), _rand(4096)
    return Simple(lambda: F.rms_norm(x, (4096,), w, 1e-6))


@register("layernorm", "norm_act", "F.layer_norm [8192x4096] BF16 (+affine)")
def _b():
    x, w, b = _rand(8192, 4096), _rand(4096), _rand(4096)
    return Simple(lambda: F.layer_norm(x, (4096,), w, b, 1e-5))


@register("softmax", "norm_act", "softmax(dim=-1) [16,2048,2048] BF16 (attention probs)")
def _b():
    x = _rand(16, 2048, 2048)
    return Simple(lambda: torch.softmax(x, dim=-1))


@register("swiglu", "norm_act", "SwiGLU silu(a)*b [8192x11008] BF16")
def _b():
    a, b = _rand(8192, 11008), _rand(8192, 11008)
    return Simple(lambda: F.silu(a) * b)


@register("gelu_tanh", "norm_act", "GELU(tanh) [8192x11008] BF16")
def _b():
    a = _rand(8192, 11008)
    return Simple(lambda: F.gelu(a, approximate="tanh"))


@register("cross_entropy", "norm_act", "cross_entropy logits [4096x32000] FP32 (LM loss)")
def _b():
    logits = _rand(4096, 32000, dtype=torch.float32)
    tgt = torch.randint(0, 32000, (4096,), device="cuda")
    return Simple(lambda: F.cross_entropy(logits, tgt))


if HAVE_TRITON:
    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        r = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + row * N + offs, (x * r * w).to(tl.bfloat16), mask=mask)

    @register("triton_rmsnorm", "norm_act", "Triton fused RMSNorm [8192x4096] BF16 (custom kernel)")
    def _b():
        x, w = _rand(8192, 4096), _rand(4096)

        def fn():
            y = torch.empty_like(x)
            _rmsnorm_kernel[(8192,)](x, w, y, 4096, 1e-6, BLOCK=4096, num_warps=8)
            return y
        return Simple(fn)


# ---------------- Convolution / cuDNN ----------------
@register("conv3x3_nhwc", "conv", "cuDNN conv3x3 256->256 [32,256,56,56] channels_last BF16")
def _b():
    x = _rand(32, 256, 56, 56, scale=0.5).to(memory_format=torch.channels_last)
    w = _rand(256, 256, 3, 3, scale=0.05).to(memory_format=torch.channels_last)
    return Simple(lambda: F.conv2d(x, w, padding=1))


@register("conv1x1_nchw_fp32", "conv", "cuDNN conv1x1 512->2048 [16,512,28,28] FP32/TF32 NCHW")
def _b():
    x = _rand(16, 512, 28, 28, dtype=torch.float32)
    w = _rand(2048, 512, 1, 1, dtype=torch.float32, scale=0.05)
    return Simple(lambda: F.conv2d(x, w))


class BatchNormTrain(Op):
    """BN training forward: updates running_mean/var IN PLACE every call (non-idempotent)."""
    inplace = True

    def __init__(self):
        self.x = _rand(32, 256, 56, 56, dtype=torch.float32)
        self.rm0 = torch.zeros(256, device="cuda")
        self.rv0 = torch.ones(256, device="cuda")
        self.rm, self.rv = self.rm0.clone(), self.rv0.clone()

    def run(self):
        F.batch_norm(self.x, self.rm, self.rv, training=True, momentum=0.1)
        return None

    def state(self):
        return [self.rm, self.rv]

    def reset(self):
        self.rm.copy_(self.rm0)
        self.rv.copy_(self.rv0)


register("batchnorm_train", "conv", "BatchNorm2d train fwd [32,256,56,56] FP32, in-place running stats")(BatchNormTrain)


# ---------------- Reduction / scan / sort / select (CUB) ----------------
@register("reduce_sum", "reduce_sort", "sum(dim=-1) [16384x4096] FP32")
def _b():
    x = _rand(16384, 4096, dtype=torch.float32)
    return Simple(lambda: x.sum(dim=-1))


@register("cumsum_64m", "reduce_sort", "cumsum 64M INT32->INT64 (CUB decoupled look-back scan)")
def _b():
    # Integer dtype: FP32 CUB scan is NOT bitwise run-to-run deterministic (look-back combines
    # predecessor aggregates in timing-dependent order), so bit-exact checks need integers.
    x = torch.randint(-1000, 1000, (64 * 1024 * 1024,), device="cuda", dtype=torch.int32)
    return Simple(lambda: torch.cumsum(x, 0))


@register("sort_16m", "reduce_sort", "sort 16M FP32 stable (CUB onesweep radix sort)")
def _b():
    x = _rand(16 * 1024 * 1024, dtype=torch.float32)
    return Simple(lambda: torch.sort(x, stable=True)[0])


@register("topk_sampling", "reduce_sort", "topk k=50 over [256x152064] FP32 (LLM sampling)")
def _b():
    x = _rand(256, 152064, dtype=torch.float32)
    return Simple(lambda: torch.topk(x, 50, dim=-1)[0])


# ---------------- Gather / scatter (embedding, MoE) ----------------
@register("embedding", "gather_scatter", "embedding gather 8192 ids from [64000x4096] BF16")
def _b():
    table = _rand(64000, 4096)
    ids = torch.randint(0, 64000, (8192,), device="cuda")
    return Simple(lambda: F.embedding(ids, table))


class IndexAddInt(Op):
    """MoE-combine style scatter-add with atomics, INT32 (deterministic), IN PLACE accumulate."""
    inplace = True

    def __init__(self):
        self.src = torch.randint(-8, 8, (262144, 1024), device="cuda", dtype=torch.int32)
        self.idx = torch.randint(0, 8192, (262144,), device="cuda")
        self.acc = torch.zeros(8192, 1024, device="cuda", dtype=torch.int32)

    def run(self):
        self.acc.index_add_(0, self.idx, self.src)
        return None

    def state(self):
        return [self.acc]

    def reset(self):
        self.acc.zero_()


register("index_add_int32", "gather_scatter", "index_add_ (atomics) INT32 262144x1024 -> 8192 rows, in-place")(IndexAddInt)


@register("cast_bool_to_i64", "gather_scatter", "dtype cast bool->int64 [8192x4096] (unrolled direct_copy kernel)")
def _b():
    x = torch.rand(8192, 4096, device="cuda") > 0.5
    return Simple(lambda: x.to(torch.int64))


@register("transpose_copy", "gather_scatter", "transpose().contiguous() [8192x8192] BF16")
def _b():
    x = _rand(8192, 8192)
    return Simple(lambda: x.t().contiguous())


# ---------------- Training (non-idempotent, in place) ----------------
class AdamWFused(Op):
    """torch.optim.AdamW(fused=True).step() on 32M params: in-place p/m/v/step update."""
    inplace = True

    def __init__(self):
        n = 32 * 1024 * 1024
        self.p0 = _rand(n, dtype=torch.float32, scale=0.02)
        self.g = _rand(n, dtype=torch.float32, scale=0.001)
        self.p = torch.nn.Parameter(self.p0.clone())
        self.p.grad = self.g
        self.opt = torch.optim.AdamW([self.p], lr=1e-3, weight_decay=0.01, fused=True)
        self.opt.step()  # materialize state

    def run(self):
        self.opt.step()
        return None

    def state(self):
        st = self.opt.state[self.p]
        return [self.p.data, st["exp_avg"], st["exp_avg_sq"]]

    def reset(self):
        with torch.no_grad():
            self.p.data.copy_(self.p0)
            st = self.opt.state[self.p]
            st["exp_avg"].zero_()
            st["exp_avg_sq"].zero_()
            st["step"].zero_()


register("adamw_fused_32m", "training", "AdamW(fused=True).step() 32M FP32 params, in-place")(AdamWFused)


class LinearBackwardAccum(Op):
    """Linear fwd+bwd with gradient ACCUMULATION into W.grad (non-idempotent, in place)."""
    inplace = True

    def __init__(self):
        self.x = _rand(2048, 4096, scale=0.05)
        self.w = torch.nn.Parameter(_rand(4096, 4096, scale=0.05))
        self.gy = _rand(2048, 4096, scale=0.05)
        self.w.grad = torch.zeros_like(self.w)

    def run(self):
        y = F.linear(self.x, self.w)
        y.backward(self.gy)
        return None

    def state(self):
        return [self.w.grad]

    def reset(self):
        self.w.grad.zero_()


register("linear_bwd_accum", "training", "Linear fwd+bwd BF16 2048x4096x4096, grad accumulation in-place")(LinearBackwardAccum)


# =====================================================================================
# Measurement helpers
# =====================================================================================
def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
    return s[i]


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def out_tensors(op, out):
    if op.inplace:
        return [t.detach().clone() for t in op.state()]
    if isinstance(out, torch.Tensor):
        return [out.detach().clone()]
    return [t.detach().clone() for t in out]


def bitwise_equal(a_list, b_list):
    return all(torch.equal(a, b) for a, b in zip(a_list, b_list))


def max_abs_err(a_list, b_list):
    m = 0.0
    for a, b in zip(a_list, b_list):
        if a.dtype.is_floating_point:
            m = max(m, (a.float() - b.float()).abs().max().item())
        else:
            m = max(m, float((a.long() - b.long()).abs().max().item()))
    return m


def shim_loaded():
    try:
        with open("/proc/self/maps") as f:
            return "libshimcuda" in f.read()
    except OSError:
        return None


# =====================================================================================
# Core benchmark for one (op, mode)
# =====================================================================================
def bench_one(op_name, mode, args):
    cat, desc, builder = OPS[op_name]
    torch.manual_seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    prop = torch.cuda.get_device_properties(0)

    xs = XSchedBridge() if mode != "native" else None
    lp = torch.cuda.Stream()
    hp = torch.cuda.Stream(priority=-1)

    res = {"op": op_name, "cat": cat, "desc": desc, "mode": mode, "gpu": prop.name,
           "sm": prop.multi_processor_count, "host": socket.gethostname(),
           "shim_loaded": shim_loaded(), "torch": torch.__version__, "error": None}

    # Build operator + HP probe on the LP stream (allocations bound to LP stream)
    with torch.cuda.stream(lp):
        op = builder()
    a_hp, b_hp = _rand(512, 1024, scale=0.05), _rand(1024, 1024, scale=0.05)
    torch.cuda.synchronize()

    def hp_fn():
        return torch.mm(a_hp, b_hp)

    # Native golden (computed BEFORE any XQueue exists) + native determinism check
    def golden_once():
        with torch.cuda.stream(lp):
            op.reset()
            out = op.run()
        lp.synchronize()
        return out_tensors(op, out)

    g1 = golden_once()
    g2 = golden_once()
    res["native_deterministic"] = bitwise_equal(g1, g2)
    golden = g1
    res["golden_digest"] = digest(golden)
    with torch.cuda.stream(hp):
        hp_gold = hp_fn().clone()
    hp.synchronize()
    del g2

    hwq = xq = None
    if mode in ("lv1", "lv3"):
        lvl = XSchedBridge.kPreemptLevelBlock if mode == "lv1" else XSchedBridge.kPreemptLevelInterrupt
        hwq, xq = xs.create_xqueue(lp, lvl)

    try:
        # ---- warmup (Lv3: first-launch Entry Guardian instrumentation happens here) ----
        if xs is not None:
            xs.route_stats(reset=True)
        with torch.cuda.stream(lp):
            op.reset()
            out = op.run()
        lp.synchronize()
        if xs is not None and mode == "lv3":
            rs = xs.route_stats(reset=True)
            res["route_per_call"] = rs
            tot = sum(rs.values()) if rs else 0
            res["kernels_per_call"] = tot
            res["lv3_coverage"] = (rs["lv3_killable"] / tot) if tot else None
        # Correctness through the queue path WITHOUT preemption
        res["xq_nopreempt_bitexact"] = bitwise_equal(out_tensors(op, out), golden)
        with torch.cuda.stream(lp):
            for _ in range(3):
                op.run()
        lp.synchronize()

        # ---- single-call latency ----
        singles = []
        for _ in range(args.single_iters):
            t0 = time.perf_counter()
            with torch.cuda.stream(lp):
                op.run()
            lp.synchronize()
            singles.append((time.perf_counter() - t0) * 1e6)
        res["lat_single_us_p50"] = statistics.median(singles)

        # ---- batch (throughput) latency ----
        est_us = max(1.0, statistics.median(singles))
        k = int(min(2000, max(3, (args.batch_ms * 1000.0) / est_us)))
        per_call = []
        for _ in range(args.batch_trials):
            t0 = time.perf_counter()
            with torch.cuda.stream(lp):
                for _ in range(k):
                    op.run()
            lp.synchronize()
            per_call.append((time.perf_counter() - t0) * 1e6 / k)
        res["batch_k"] = k
        res["lat_batch_us_p50"] = statistics.median(per_call)
        res["lat_batch_us_min"] = min(per_call)

        # ---- HP probe solo latency ----
        hp_solo = []
        for _ in range(20):
            t0 = time.perf_counter()
            with torch.cuda.stream(hp):
                hp_fn()
            hp.synchronize()
            hp_solo.append((time.perf_counter() - t0) * 1e3)
        res["hp_solo_ms_p50"] = statistics.median(hp_solo)

        # ---- preemption / co-run test ----
        if args.job_ms > 0:
            preempt_test(op, golden, hp_fn, hp_gold, xs, xq, lp, hp, res, args)
    except Exception as e:  # report instead of crash: JSON line still written
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-1500:]
    finally:
        if xq is not None:
            try:
                xs.destroy(hwq, xq)
            except Exception:
                pass
    return res


def preempt_test(op, golden, hp_fn, hp_gold, xs, xq, lp, hp, res, args):
    reps = int(min(20000, max(3, round(args.job_ms * 1000.0 / max(1.0, res["lat_batch_us_p50"])))))
    res["job_reps"] = reps
    # Out-of-place ops: every call is compared to golden ON-GPU (mismatch counter on LP stream)
    gold_t = golden[0] if not op.inplace else None
    # Cast-free check (bool any() + logical_or): avoids injecting PyTorch's
    # unrolled_elementwise_kernel<direct_copy> (bool->int64 cast) into every LP job; that
    # cast kernel is benchmarked separately as `cast_bool_to_i64`.
    mism = torch.zeros((), device="cuda", dtype=torch.bool)

    def lp_job():
        with torch.cuda.stream(lp):
            for _ in range(reps):
                out = op.run()
                if gold_t is not None:
                    o = out if isinstance(out, torch.Tensor) else out[0]
                    torch.logical_or(mism, torch.ne(o, gold_t).any(), out=mism)

    # In-place golden for `reps` calls (from the same reset state), computed solo
    def reset_all():
        with torch.cuda.stream(lp):
            op.reset()
            mism.zero_()
        lp.synchronize()

    reset_all()
    lp_job()          # untimed warm-up job (allocator / mismatch-kernel first-use effects)
    lp.synchronize()
    solo = []
    for _ in range(2):
        reset_all()
        t0 = time.perf_counter()
        lp_job()
        lp.synchronize()
        solo.append((time.perf_counter() - t0) * 1e3)
    solo_ms = min(solo)
    gold_state = out_tensors(op, None) if op.inplace else None
    res["job_solo_ms"] = solo_ms
    res["job_solo_mism"] = int(mism.item())
    if op.inplace:
        res["job_golden_digest"] = digest(gold_state)

    susp_ms, hp_e2e_ms, hp_exec_ms, job_ms = [], [], [], []
    hp_bad = 0
    ok_trials = 0
    trial_errs = []
    fractions = [0.2, 0.45, 0.7][: args.preempts]
    for trial in range(args.trials):
        reset_all()
        bg_err = []
        started = threading.Event()

        def preemptor():
            try:
                torch.cuda.set_device(0)
                started.wait()
                t_start = time.perf_counter()
                for fr in fractions:
                    target = t_start + fr * solo_ms / 1e3
                    while time.perf_counter() < target:
                        time.sleep(0.0002)
                    t0 = time.perf_counter()
                    if xq is not None:
                        xs.suspend(xq)
                    t1 = time.perf_counter()
                    with torch.cuda.stream(hp):
                        y = hp_fn()
                    hp.synchronize()
                    t2 = time.perf_counter()
                    if xq is not None:
                        xs.resume(xq)
                    if xq is not None:
                        susp_ms.append((t1 - t0) * 1e3)
                    hp_exec_ms.append((t2 - t1) * 1e3)
                    hp_e2e_ms.append((t2 - t0) * 1e3)
                    if not torch.equal(y, hp_gold):
                        bg_err.append("hp_mismatch")
            except Exception as e:
                bg_err.append(f"{type(e).__name__}: {e}")

        th = threading.Thread(target=preemptor)
        th.start()
        t0 = time.perf_counter()
        started.set()
        lp_job()
        lp.synchronize()
        job_ms.append((time.perf_counter() - t0) * 1e3)
        th.join()
        if any(e == "hp_mismatch" for e in bg_err):
            hp_bad += 1
        bg_other = [e for e in bg_err if e != "hp_mismatch"]
        if bg_other:
            trial_errs.extend(bg_other)
        if op.inplace:
            ok = bitwise_equal(out_tensors(op, None), gold_state)
            if not ok:
                trial_errs.append(f"trial{trial}: state max_err="
                                  f"{max_abs_err(out_tensors(op, None), gold_state):.3e}")
        else:
            m = bool(mism.item())
            ok = not m
            if not ok:
                trial_errs.append(f"trial{trial}: some LP call output != golden")
        ok_trials += int(ok)

    res["trials"] = args.trials
    res["preempts_per_trial"] = len(fractions)
    res["ok_trials"] = ok_trials
    res["bitexact"] = (ok_trials == args.trials) and hp_bad == 0
    res["hp_bad_trials"] = hp_bad
    res["trial_errors"] = trial_errs[:8]
    res["susp_ms"] = susp_ms
    res["hp_e2e_ms"] = hp_e2e_ms
    res["hp_exec_ms"] = hp_exec_ms
    res["job_ms"] = job_ms
    res["susp_ms_p50"] = pct(susp_ms, 0.5)
    res["susp_ms_max"] = max(susp_ms) if susp_ms else None
    res["hp_e2e_ms_p50"] = pct(hp_e2e_ms, 0.5)
    res["hp_e2e_ms_p99"] = pct(hp_e2e_ms, 0.99)
    res["job_inflation_pct"] = 100.0 * (statistics.median(job_ms) - solo_ms) / solo_ms


# =====================================================================================
# Report
# =====================================================================================
MODES = ["native", "shim", "lv1", "lv3"]


def fmt(v, f="{:.1f}", na="-"):
    return na if v is None else f.format(v)


def report(paths, md_out=None):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    by = {}
    gpus = sorted({r["gpu"] for r in rows})
    lines = []
    for gpu in gpus:
        sub = [r for r in rows if r["gpu"] == gpu]
        for r in sub:
            by[(r["op"], r["mode"])] = r
        ops = [o for o in OPS if any(r["op"] == o for r in sub)]
        ran_modes = {r["mode"] for r in sub}
        sm = sub[0]["sm"]
        lines.append(f"\n### {gpu} ({sm} SMs, host `{sub[0]['host']}`)\n")
        lines.append("#### A. Steady-state overhead (no preemption) and Level-3 routing coverage\n")
        lines.append("| op | category | native batch µs | shim Δ% | Lv1 Δ% | Lv3 Δ% | native single µs | Lv3 single µs | kernels/call | Lv3 coverage | routed→Lv1 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for o in ops:
            n = by.get((o, "native"), {})
            base = n.get("lat_batch_us_p50")

            def d(mode):
                r = by.get((o, mode))
                if r is None:
                    return "CRASH" if mode in ran_modes else "-"
                v = r.get("lat_batch_us_p50")
                if r.get("error"):
                    return "ERR"
                return fmt(100.0 * (v - base) / base, "{:+.1f}") if (v and base) else "-"
            l3 = by.get((o, "lv3"), {})
            rs = l3.get("route_per_call") or {}
            routed = ", ".join(f"{k[4:]}={v}" for k, v in rs.items() if k != "lv3_killable" and v)
            lines.append(f"| `{o}` | {OPS[o][0]} | {fmt(base)} | {d('shim')} | {d('lv1')} | {d('lv3')} | "
                         f"{fmt(n.get('lat_single_us_p50'))} | {fmt(l3.get('lat_single_us_p50'))} | "
                         f"{fmt(l3.get('kernels_per_call'), '{:d}')} | "
                         f"{fmt(l3.get('lv3_coverage'), '{:.0%}')} | {routed or '-'} |")
        lines.append("\n#### B. Preemption: HP probe latency, suspend latency, LP correctness\n")
        lines.append("| op | LP job (reps) | HP solo ms | co-run HP P50/P99 ms (native) | Lv1 susp P50/max ms | Lv1 HP P50/P99 ms | Lv3 susp P50/max ms | Lv3 HP P50/P99 ms | Lv3 job infl. | bit-exact native/Lv1/Lv3 | golden digest match |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for o in ops:
            n, l1, l3 = (by.get((o, m), {}) for m in ("native", "lv1", "lv3"))

            def be(r):
                if not r:
                    return "CRASH"
                if r.get("error"):
                    return "ERR"
                b = r.get("bitexact")
                return "✅" if b else ("❌" if b is not None else "-")
            dm = "-"
            for key in ("golden_digest", "job_golden_digest"):
                digs = {m: by.get((o, m), {}).get(key) for m in MODES}
                vals = [v for v in digs.values() if v]
                if not vals:
                    continue
                if len(set(vals)) == 1:
                    dm = "✅" if dm in ("-", "✅") else dm
                else:
                    dm = f"❌ {key}={digs}"
            if by.get((o, "native"), {}).get("native_deterministic") is False:
                dm += " (native nondeterministic)"
            lines.append(
                f"| `{o}` | {fmt(n.get('job_solo_ms'))} ms ({n.get('job_reps', '-')}) | {fmt(n.get('hp_solo_ms_p50'), '{:.3f}')} | "
                f"{fmt(n.get('hp_e2e_ms_p50'), '{:.2f}')} / {fmt(n.get('hp_e2e_ms_p99'), '{:.2f}')} | "
                f"{fmt(l1.get('susp_ms_p50'), '{:.3f}')} / {fmt(l1.get('susp_ms_max'), '{:.3f}')} | "
                f"{fmt(l1.get('hp_e2e_ms_p50'), '{:.2f}')} / {fmt(l1.get('hp_e2e_ms_p99'), '{:.2f}')} | "
                f"{fmt(l3.get('susp_ms_p50'), '{:.3f}')} / {fmt(l3.get('susp_ms_max'), '{:.3f}')} | "
                f"{fmt(l3.get('hp_e2e_ms_p50'), '{:.2f}')} / {fmt(l3.get('hp_e2e_ms_p99'), '{:.2f}')} | "
                f"{fmt(l3.get('job_inflation_pct'), '{:+.1f}%')} | {be(n)} / {be(l1)} / {be(l3)} | {dm} |")
        errs = [r for r in sub if r.get("error") or r.get("trial_errors")]
        if errs:
            lines.append("\n#### Errors / non-bit-exact trials\n")
            for r in errs:
                lines.append(f"- `{r['op']}` [{r['mode']}]: {r.get('error') or ''} {r.get('trial_errors') or ''}")
    text = "\n".join(lines)
    print(text)
    if md_out:
        with open(md_out, "w") as f:
            f.write(text + "\n")


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--op", required=True)
    r.add_argument("--mode", required=True, choices=MODES)
    r.add_argument("--out", required=True)
    r.add_argument("--job-ms", type=float, default=60.0)
    r.add_argument("--trials", type=int, default=5)
    r.add_argument("--preempts", type=int, default=3)
    r.add_argument("--batch-ms", type=float, default=40.0)
    r.add_argument("--batch-trials", type=int, default=5)
    r.add_argument("--single-iters", type=int, default=30)
    sub.add_parser("list")
    rp = sub.add_parser("report")
    rp.add_argument("paths", nargs="+")
    rp.add_argument("--md")
    args = ap.parse_args()

    if args.cmd == "list":
        for k, (c, d, _) in OPS.items():
            print(f"{k:22s} {c:15s} {d}")
        return
    if args.cmd == "report":
        report(args.paths, args.md)
        return
    if args.op not in OPS:
        sys.exit(f"unknown op {args.op}")
    if args.mode == "native" and shim_loaded():
        sys.exit("native mode requested but libshimcuda is loaded: remove XSched from LD_LIBRARY_PATH")
    res = bench_one(args.op, args.mode, args)
    line = json.dumps(res)
    with open(args.out, "a") as f:
        f.write(line + "\n")
    tag = "ERR" if res.get("error") else ("PASS" if res.get("bitexact", True) else "FAIL")
    print(f"[OPS {args.op} mode={args.mode}] {tag} batch={fmt(res.get('lat_batch_us_p50'))}us "
          f"single={fmt(res.get('lat_single_us_p50'))}us susp_p50={fmt(res.get('susp_ms_p50'), '{:.3f}')}ms "
          f"hp_p50={fmt(res.get('hp_e2e_ms_p50'), '{:.2f}')}ms cov={fmt(res.get('lv3_coverage'), '{:.0%}')} "
          f"ok={res.get('ok_trials')}/{res.get('trials')} {res.get('error') or ''}", flush=True)


if __name__ == "__main__":
    main()
