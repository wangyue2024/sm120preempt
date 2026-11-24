import ctypes
import os
import sys
import time
import threading
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import torchvision.models as models
import triton
import triton.language as tl


# =========================================================================
# Minimal ctypes bridge to XSched HAL & Preempt C API
# =========================================================================
class XSchedBridge:
    kPreemptLevelBlock = 1
    kPreemptLevelDeactivate = 2
    kPreemptLevelInterrupt = 3
    kQueueCreateFlagNone = 0
    kQueueSuspendFlagSyncHwQueue = 1
    kQueueResumeFlagNone = 0

    def __init__(self):
        self.libpreempt = ctypes.CDLL("libpreempt.so", mode=ctypes.RTLD_GLOBAL)
        self.libhal = ctypes.CDLL("libhalcuda.so", mode=ctypes.RTLD_GLOBAL)

        self.libhal.CudaQueueCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p]
        self.libhal.CudaQueueCreate.restype = ctypes.c_int

        self.libpreempt.HwQueueDestroy.argtypes = [ctypes.c_uint64]
        self.libpreempt.HwQueueDestroy.restype = ctypes.c_int

        self.libpreempt.XQueueCreate.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.c_int, ctypes.c_int
        ]
        self.libpreempt.XQueueCreate.restype = ctypes.c_int

        self.libpreempt.XQueueDestroy.argtypes = [ctypes.c_uint64]
        self.libpreempt.XQueueDestroy.restype = ctypes.c_int

        self.libpreempt.XQueueSuspend.argtypes = [ctypes.c_uint64, ctypes.c_int]
        self.libpreempt.XQueueSuspend.restype = ctypes.c_int

        self.libpreempt.XQueueResume.argtypes = [ctypes.c_uint64, ctypes.c_int]
        self.libpreempt.XQueueResume.restype = ctypes.c_int

    def create_xqueue(self, stream: torch.cuda.Stream, level: int):
        hwq = ctypes.c_uint64(0)
        xq = ctypes.c_uint64(0)
        r1 = self.libhal.CudaQueueCreate(ctypes.byref(hwq), ctypes.c_void_p(stream.cuda_stream))
        if r1 != 0:
            raise RuntimeError(f"CudaQueueCreate failed: {r1}")
        r2 = self.libpreempt.XQueueCreate(ctypes.byref(xq), hwq, level, self.kQueueCreateFlagNone)
        if r2 != 0:
            raise RuntimeError(f"XQueueCreate failed: {r2}")
        return hwq, xq

    def suspend(self, xq):
        r = self.libpreempt.XQueueSuspend(xq, self.kQueueSuspendFlagSyncHwQueue)
        if r != 0:
            raise RuntimeError(f"XQueueSuspend failed: {r}")

    def resume(self, xq):
        r = self.libpreempt.XQueueResume(xq, self.kQueueResumeFlagNone)
        if r != 0:
            raise RuntimeError(f"XQueueResume failed: {r}")

    def destroy(self, hwq, xq):
        self.libpreempt.XQueueDestroy(xq)
        self.libpreempt.HwQueueDestroy(hwq)


# =========================================================================
# Custom Triton 3.6.0 Blackwell Kernel (Non-Idempotent In-Place Fused Update)
# =========================================================================
@triton.jit
def triton_fused_nonidemp_kernel(x_ptr, y_ptr, n_elements, iters, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    for _ in range(iters):
        x = x * 0.9999 + tl.sigmoid(y) * 0.001
        y = y * 0.9999 + tl.sin(x) * 0.001
    tl.store(x_ptr + offsets, x, mask=mask)
    tl.store(y_ptr + offsets, y, mask=mask)


def pct(vals, p):
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil(p * len(s))) - 1))
    return s[idx]


def run_with_mid_preempt(xs, xq, lp_stream, hp_stream, fn, delay_ms=3.0, num_preempts=2, hp_fn=None):
    done = threading.Event()
    err = []

    def worker():
        try:
            torch.cuda.set_device(0)
            with torch.cuda.stream(lp_stream):
                fn()
            lp_stream.synchronize()
        except Exception as e:
            err.append(e)
        finally:
            done.set()

    t = threading.Thread(target=worker)
    t.start()

    susp_ms = []
    hp_ms = []
    for _ in range(num_preempts):
        time.sleep(delay_ms / 1000.0)
        if done.is_set():
            break
        t0 = time.perf_counter()
        xs.suspend(xq)
        t1 = time.perf_counter()
        susp_ms.append((t1 - t0) * 1000.0)
        if hp_fn is not None:
            th0 = time.perf_counter()
            with torch.cuda.stream(hp_stream):
                hp_fn()
            hp_stream.synchronize()
            hp_ms.append((time.perf_counter() - th0) * 1000.0)
        xs.resume(xq)

    t.join()
    if err:
        raise err[0]
    return susp_ms, hp_ms


class MiniLlamaBlock(nn.Module):
    def __init__(self, d_model=1024, n_heads=16, d_ff=2816):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x, sdpa_backend):
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        with sdpa_kernel(sdpa_backend):
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(attn)
        h2 = self.norm2(x)
        x = x + self.w2(F.silu(self.w1(h2)) * self.w3(h2))
        return x


def test_pt01_resnet50_inference(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(0)
    model = models.resnet50(weights=None).cuda().eval().half()
    x = torch.randn(32, 3, 224, 224, device="cuda", dtype=torch.float16)
    hp_a = torch.randn(256, 512, device="cuda", dtype=torch.float16)
    hp_b = torch.randn(512, 512, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        with torch.cuda.stream(hp_stream):
            ref = model(x).clone()
        hp_stream.synchronize()
        # Warmup instrumentation on lp_stream
        with torch.cuda.stream(lp_stream):
            _ = model(x)
        lp_stream.synchronize()

        out_box = [None]
        def run_lp():
            out_box[0] = model(x)
        def run_hp():
            _ = hp_a @ hp_b

        susp, hp = run_with_mid_preempt(xs, xq, lp_stream, hp_stream, run_lp, delay_ms=4.0, num_preempts=2, hp_fn=run_hp)
        max_err = (out_box[0] - ref).abs().max().item()
    passed = (len(susp) >= 1) and (max_err <= 1e-2)
    print(f"[PT-01] PyTorch ResNet-50 FP16 Inference (cuDNN 9.19): preempts={len(susp)}/2 "
          f"avg_susp={sum(susp)/max(1,len(susp)):.3f}ms avg_hp={sum(hp)/max(1,len(hp)):.3f}ms "
          f"max_err={max_err:.3e} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return passed


def test_pt02_resnet50_train(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(42)
    m_ref = models.resnet18(weights=None).cuda().train()
    m_pre = models.resnet18(weights=None).cuda().train()
    m_pre.load_state_dict(m_ref.state_dict())

    x = torch.randn(32, 3, 224, 224, device="cuda")
    y = torch.randint(0, 1000, (32,), device="cuda")
    opt_ref = torch.optim.AdamW(m_ref.parameters(), lr=1e-3)
    opt_pre = torch.optim.AdamW(m_pre.parameters(), lr=1e-3)

    # Reference 1 training step on hp_stream (non-idempotent AdamW update!)
    with torch.cuda.stream(hp_stream):
        opt_ref.zero_grad()
        loss_ref = F.cross_entropy(m_ref(x), y)
        loss_ref.backward()
        opt_ref.step()
    hp_stream.synchronize()

    # Warmup instrumentation on a throwaway copy first
    m_warm = models.resnet18(weights=None).cuda().train()
    opt_warm = torch.optim.AdamW(m_warm.parameters(), lr=1e-3)
    with torch.cuda.stream(lp_stream):
        opt_warm.zero_grad()
        F.cross_entropy(m_warm(x), y).backward()
        opt_warm.step()
    lp_stream.synchronize()
    del m_warm, opt_warm

    hp_a = torch.randn(256, 512, device="cuda", dtype=torch.float16)
    hp_b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    def run_lp():
        opt_pre.zero_grad()
        loss = F.cross_entropy(m_pre(x), y)
        loss.backward()
        opt_pre.step()

    susp, hp = run_with_mid_preempt(xs, xq, lp_stream, hp_stream, run_lp, delay_ms=4.0, num_preempts=2,
                                    hp_fn=lambda: hp_a @ hp_b)
    max_w_err = 0.0
    for p1, p2 in zip(m_ref.parameters(), m_pre.parameters()):
        max_w_err = max(max_w_err, (p1 - p2).abs().max().item())
    passed = (len(susp) >= 1) and (max_w_err <= 1e-4)
    print(f"[PT-02] PyTorch ResNet-18 Non-Idemp Training (Fwd+Bwd+AdamW): preempts={len(susp)}/2 "
          f"avg_susp={sum(susp)/max(1,len(susp)):.3f}ms avg_hp={sum(hp)/max(1,len(hp)):.3f}ms "
          f"max_weight_err={max_w_err:.3e} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return passed


def test_pt03_llm_sdpa(xs, lp_stream, hp_stream, xq, backend_name):
    backend_map = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    be = backend_map[backend_name]
    torch.manual_seed(7)
    layers = nn.ModuleList([MiniLlamaBlock(1024, 16, 2816) for _ in range(4)]).cuda().half().eval()
    x = torch.randn(4, 512, 1024, device="cuda", dtype=torch.float16)
    hp_a = torch.randn(256, 512, device="cuda", dtype=torch.float16)
    hp_b = torch.randn(512, 512, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        with torch.cuda.stream(hp_stream):
            h = x
            for blk in layers:
                h = blk(h, be)
            ref = h.clone()
        hp_stream.synchronize()

        with torch.cuda.stream(lp_stream):
            h = x
            for blk in layers:
                h = blk(h, be)
        lp_stream.synchronize()

        out_box = [None]
        def run_lp():
            h = x
            for blk in layers:
                h = blk(h, be)
            out_box[0] = h

        susp, hp = run_with_mid_preempt(xs, xq, lp_stream, hp_stream, run_lp, delay_ms=2.0, num_preempts=2,
                                        hp_fn=lambda: hp_a @ hp_b)
        max_err = (out_box[0] - ref).abs().max().item()
    passed = (len(susp) >= 1) and (max_err <= 1e-2)
    print(f"[PT-03-{backend_name.upper():9s}] PyTorch 4L Llama Prefill (SDPA={backend_name}): preempts={len(susp)}/2 "
          f"avg_susp={sum(susp)/max(1,len(susp)):.3f}ms avg_hp={sum(hp)/max(1,len(hp)):.3f}ms "
          f"max_err={max_err:.3e} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return passed


def test_pt04_triton(xs, lp_stream, hp_stream, xq):
    N = 128 * 256
    x0 = torch.linspace(-1.0, 1.0, N, device="cuda", dtype=torch.float32)
    y0 = torch.linspace(0.5, -0.5, N, device="cuda", dtype=torch.float32)

    xr, yr = x0.clone(), y0.clone()
    with torch.cuda.stream(hp_stream):
        triton_fused_nonidemp_kernel[(128,)](xr, yr, N, iters=35000, BLOCK_SIZE=256)
    hp_stream.synchronize()

    # Warmup on lp_stream
    xw, yw = x0.clone(), y0.clone()
    with torch.cuda.stream(lp_stream):
        triton_fused_nonidemp_kernel[(128,)](xw, yw, N, iters=35000, BLOCK_SIZE=256)
    lp_stream.synchronize()

    xp, yp = x0.clone(), y0.clone()
    hp_a = torch.randn(256, 512, device="cuda", dtype=torch.float16)
    hp_b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    def run_lp():
        triton_fused_nonidemp_kernel[(128,)](xp, yp, N, iters=35000, BLOCK_SIZE=256)

    susp, hp = run_with_mid_preempt(xs, xq, lp_stream, hp_stream, run_lp, delay_ms=2.0, num_preempts=2,
                                    hp_fn=lambda: hp_a @ hp_b)
    err_x = (xp - xr).abs().max().item()
    err_y = (yp - yr).abs().max().item()
    passed = (len(susp) >= 1) and (err_x == 0.0) and (err_y == 0.0)
    print(f"[PT-04] Triton 3.6.0 Non-Idemp Kernel (sm_120): preempts={len(susp)}/2 "
          f"avg_susp={sum(susp)/max(1,len(susp)):.3f}ms avg_hp={sum(hp)/max(1,len(hp)):.3f}ms "
          f"err_x={err_x:.3e} err_y={err_y:.3e} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return passed


def test_pt05_qos(mode: str):
    torch.manual_seed(0)
    xs = None if mode == "native" else XSchedBridge()
    lp_stream = torch.cuda.Stream(priority=0)
    hp_stream = torch.cuda.Stream(priority=-1)
    hwq = xq = None
    if mode != "native":
        lv = XSchedBridge.kPreemptLevelBlock if mode == "lv1" else XSchedBridge.kPreemptLevelInterrupt
        hwq, xq = xs.create_xqueue(lp_stream, lv)

    # LP: Heavy Triton + PyTorch Transformer Prefill (~15ms/step)
    N = 256 * 256
    x0 = torch.linspace(-1.0, 1.0, N, device="cuda", dtype=torch.float32)
    y0 = torch.linspace(0.5, -0.5, N, device="cuda", dtype=torch.float32)
    layers = nn.ModuleList([MiniLlamaBlock(1024, 16, 2816) for _ in range(3)]).cuda().half().eval()
    tx = torch.randn(4, 512, 1024, device="cuda", dtype=torch.float16)

    # HP: Fast latency-critical PyTorch FP16 MLP inference (~0.25ms native)
    hp_w1 = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    hp_w2 = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    hp_x = torch.randn(32, 1024, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        # Warmup & standalone baseline
        with torch.cuda.stream(hp_stream):
            for _ in range(5):
                _ = F.gelu(hp_x @ hp_w1) @ hp_w2
        hp_stream.synchronize()
        th0 = time.perf_counter()
        for _ in range(10):
            with torch.cuda.stream(hp_stream):
                _ = F.gelu(hp_x @ hp_w1) @ hp_w2
            hp_stream.synchronize()
        hp_base = (time.perf_counter() - th0) * 1000.0 / 10.0

        def lp_step(s):
            with torch.cuda.stream(s):
                h = tx
                for blk in layers:
                    h = blk(h, SDPBackend.MATH)
                triton_fused_nonidemp_kernel[(256,)](x0, y0, N, iters=2500, BLOCK_SIZE=256)

        lp_step(hp_stream)
        hp_stream.synchronize()
        tl0 = time.perf_counter()
        for _ in range(3):
            lp_step(hp_stream)
        hp_stream.synchronize()
        lp_base = (time.perf_counter() - tl0) * 1000.0 / 3.0

        # Warmup LP instrumentation
        lp_step(lp_stream)
        lp_stream.synchronize()

        stop = threading.Event()
        lp_cnt = [0]
        def lp_worker():
            torch.cuda.set_device(0)
            with torch.no_grad():
                while not stop.is_set():
                    lp_step(lp_stream)
                    lp_stream.synchronize()
                    lp_cnt[0] += 1

        thr = threading.Thread(target=lp_worker)
        thr.start()
        time.sleep(0.02)

        hp_lats, susp_lats = [], []
        tb0 = time.perf_counter()
        for _ in range(20):
            time.sleep(0.02)
            t0 = time.perf_counter()
            if mode != "native":
                xs.suspend(xq)
            t1 = time.perf_counter()
            with torch.cuda.stream(hp_stream):
                _ = F.gelu(hp_x @ hp_w1) @ hp_w2
            hp_stream.synchronize()
            t2 = time.perf_counter()
            if mode != "native":
                xs.resume(xq)
            susp_lats.append((t1 - t0) * 1000.0)
            hp_lats.append((t2 - t0) * 1000.0)
        tb1 = time.perf_counter()
        stop.set()
        thr.join()

    lp_ret = 100.0 * (lp_cnt[0] / (tb1 - tb0)) / (1000.0 / lp_base)
    slo2 = 100.0 * sum(1 for v in hp_lats if v <= 2.0) / len(hp_lats)
    slo5 = 100.0 * sum(1 for v in hp_lats if v <= 5.0) / len(hp_lats)
    print(f"[PT-05-QOS] mode={mode:6s} | base_lp={lp_base:.2f}ms base_hp={hp_base:.3f}ms | "
          f"HP_ms(P50={pct(hp_lats,0.50):.3f} P95={pct(hp_lats,0.95):.3f} P99={pct(hp_lats,0.99):.3f} Max={max(hp_lats):.3f}) "
          f"SLO2ms={slo2:.1f}% SLO5ms={slo5:.1f}% | SUSP_P50={pct(susp_lats,0.50):.3f}ms | "
          f"LP_steps={lp_cnt[0]} LP_ret={lp_ret:.1f}%", flush=True)
    if mode != "native":
        xs.destroy(hwq, xq)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"[PYTORCH_SM120_SUITE] torch={torch.__version__} cuda={torch.version.cuda} "
          f"cudnn={torch.backends.cudnn.version()} triton={triton.__version__} "
          f"gpu={torch.cuda.get_device_name(0)} case={case}", flush=True)

    if case.startswith("qos:"):
        test_pt05_qos(case.split(":")[1])
        return

    xs = XSchedBridge()
    lp_stream = torch.cuda.Stream(priority=0)
    hp_stream = torch.cuda.Stream(priority=-1)
    hwq, xq = xs.create_xqueue(lp_stream, XSchedBridge.kPreemptLevelInterrupt)

    try:
        if case in ("all", "pt01"):
            test_pt01_resnet50_inference(xs, lp_stream, hp_stream, xq)
        if case in ("all", "pt02"):
            test_pt02_resnet50_train(xs, lp_stream, hp_stream, xq)
        if case in ("all", "pt03_math"):
            test_pt03_llm_sdpa(xs, lp_stream, hp_stream, xq, "math")
        if case in ("all", "pt03_flash"):
            test_pt03_llm_sdpa(xs, lp_stream, hp_stream, xq, "flash")
        if case in ("all", "pt03_efficient"):
            test_pt03_llm_sdpa(xs, lp_stream, hp_stream, xq, "efficient")
        if case in ("all", "pt03_cudnn"):
            test_pt03_llm_sdpa(xs, lp_stream, hp_stream, xq, "cudnn")
        if case in ("all", "pt04"):
            test_pt04_triton(xs, lp_stream, hp_stream, xq)
    finally:
        xs.destroy(hwq, xq)


if __name__ == "__main__":
    main()
