import copy
import ctypes
import math
import os
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache
from peft import LoraConfig, get_peft_model


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
        if self.libhal.CudaQueueCreate(ctypes.byref(hwq), ctypes.c_void_p(stream.cuda_stream)) != 0:
            raise RuntimeError("CudaQueueCreate failed")
        if self.libpreempt.XQueueCreate(ctypes.byref(xq), hwq, level, self.kQueueCreateFlagNone) != 0:
            raise RuntimeError("XQueueCreate failed")
        return hwq, xq

    def suspend(self, xq):
        if self.libpreempt.XQueueSuspend(xq, self.kQueueSuspendFlagSyncHwQueue) != 0:
            raise RuntimeError("XQueueSuspend failed")

    def resume(self, xq):
        if self.libpreempt.XQueueResume(xq, self.kQueueResumeFlagNone) != 0:
            raise RuntimeError("XQueueResume failed")

    def destroy(self, hwq, xq):
        self.libpreempt.XQueueDestroy(xq)
        self.libpreempt.HwQueueDestroy(hwq)


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil(p * len(s))) - 1))
    return s[idx]


def run_with_mid_preempt(xs, xq, lp_stream, hp_stream, fn, delay_ms=2.5, num_preempts=3, hp_fn=None):
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

    sms = max(30, torch.cuda.get_device_properties(0).multi_processor_count)
    eff_delay_s = max(0.0005, (delay_ms / 1000.0) * (30.0 / float(sms)))
    susp_ms = []
    hp_ms = []
    for _ in range(num_preempts):
        time.sleep(eff_delay_s)
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


def make_qwen2_model(num_layers=12, hidden=1536, inter=8960, heads=12, kv_heads=2, vocab=32000, dtype=torch.bfloat16):
    # Exact Qwen2.5-1.5B architecture dimensions (GQA 12:2, RoPE, SwiGLU, RMSNorm)
    cfg = Qwen2Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=8192,
        use_cache=True,
    )
    cfg._attn_implementation = "sdpa"
    m = Qwen2ForCausalLM(cfg).cuda().to(dtype)
    return m


def extract_kv_tensors(past_kv):
    # Compatible with transformers v4 & v5 DynamicCache API
    if hasattr(past_kv, "layers") and len(past_kv.layers) > 0:
        return [(lay.keys, lay.values) for lay in past_kv.layers]
    if hasattr(past_kv, "key_cache"):
        return list(zip(past_kv.key_cache, past_kv.value_cache))
    return [(k, v) for k, v in past_kv]


def make_hp_gemm():
    a = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    return lambda: torch.mm(a, b)


# =========================================================================
# Custom Triton W4A16 (INT4 Packed Weight-Only Dequant-GEMM, AWQ/Marlin-style)
# =========================================================================
@triton.jit
def triton_w4a16_gemm_kernel(
    a_ptr, qweight_ptr, scales_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_qk, stride_qn,
    stride_sk, stride_sn,
    stride_cm, stride_cn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # Packed 2x INT4 per uint8 along K dimension: qweight has shape [K//2, N]
        qk = offs_k // 2
        qshift = (offs_k % 2) * 4
        q_ptrs = qweight_ptr + qk[:, None] * stride_qk + offs_n[None, :] * stride_qn
        qbyte = tl.load(q_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0)
        qval = ((qbyte >> qshift[:, None]) & 0xF).to(tl.float16) - 8.0

        sg = offs_k // GROUP_SIZE
        s_ptrs = scales_ptr + sg[:, None] * stride_sk + offs_n[None, :] * stride_sn
        scale = tl.load(s_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=1.0)
        b = (qval * scale).to(tl.float16)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# =========================================================================
# Custom Triton vLLM-Style PagedAttention (Indirect Block-Table KV-Cache)
# =========================================================================
@triton.jit
def triton_paged_attention_v1(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr, block_tables_ptr, ctx_lens_ptr,
    scale,
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kh, stride_kb, stride_kd,
    stride_bt_b, stride_bt_i,
    NUM_KV_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // GQA_RATIO

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + seq_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd).to(tl.float32)

    ctx_len = tl.load(ctx_lens_ptr + seq_idx)
    num_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    m_i = -1e20
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    offs_tok = tl.arange(0, BLOCK_SIZE)

    for b in range(0, MAX_BLOCKS):
        if b < num_blocks:
            phys_blk = tl.load(block_tables_ptr + seq_idx * stride_bt_b + b * stride_bt_i)
            tok_pos = b * BLOCK_SIZE + offs_tok
            tok_mask = tok_pos < ctx_len

            k_ptrs = (k_cache_ptr + phys_blk * stride_kp + kv_head_idx * stride_kh +
                      offs_tok[:, None] * stride_kb + offs_d[None, :] * stride_kd)
            v_ptrs = (v_cache_ptr + phys_blk * stride_kp + kv_head_idx * stride_kh +
                      offs_tok[:, None] * stride_kb + offs_d[None, :] * stride_kd)

            k = tl.load(k_ptrs, mask=tok_mask[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=tok_mask[:, None], other=0.0).to(tl.float32)

            qk = tl.sum(k * q[None, :], axis=1) * scale
            qk = tl.where(tok_mask, qk, -1e20)

            m_new = tl.maximum(m_i, tl.max(qk, axis=0))
            p = tl.exp(qk - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

    out = (acc / l_i).to(tl.float16)
    tl.store(out_ptr + seq_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd, out)


# =========================================================================
# LLM-01: Real HuggingFace Qwen2ForCausalLM Long-Context Prefill + KV-Cache
# =========================================================================
def test_llm01_qwen2_prefill(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(101)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    n_layers = 16 if vram_gb >= 12.0 else 8
    seq_len = 2048 if vram_gb >= 12.0 else 1024
    model = make_qwen2_model(num_layers=n_layers).eval()
    input_ids = torch.randint(0, 32000, (1, seq_len), device="cuda")
    hp_fn = make_hp_gemm()

    with torch.no_grad():
        with torch.cuda.stream(lp_stream):
            _ = model(input_ids=input_ids, use_cache=True)
        lp_stream.synchronize()

        with torch.cuda.stream(lp_stream):
            ref_out = model(input_ids=input_ids, use_cache=True)
            ref_logits = ref_out.logits[:, -1, :].clone()
            ref_kv = [(k.clone(), v.clone()) for k, v in extract_kv_tensors(ref_out.past_key_values)]
        lp_stream.synchronize()

        box = {}
        def run_lp():
            out = model(input_ids=input_ids, use_cache=True)
            box["logits"] = out.logits[:, -1, :].clone()
            box["kv"] = [(k.clone(), v.clone()) for k, v in extract_kv_tensors(out.past_key_values)]

        susp, hp = run_with_mid_preempt(xs, xq, lp_stream, hp_stream, run_lp, delay_ms=3.0, num_preempts=4, hp_fn=hp_fn)

    logit_err = (box["logits"].float() - ref_logits.float()).abs().max().item()
    kv_err = max(
        max((k1.float() - k2.float()).abs().max().item(), (v1.float() - v2.float()).abs().max().item())
        for (k1, v1), (k2, v2) in zip(box["kv"], ref_kv)
    )
    ok = len(susp) >= 1 and logit_err == 0.0 and kv_err == 0.0
    print(f"[LLM-01 Qwen2-{n_layers}L Prefill T={seq_len} BF16] {'PASS' if ok else 'FAIL'} | preempts={len(susp)} "
          f"susp={[round(v, 3) for v in susp]} logit_err={logit_err:.3e} kv_err={kv_err:.3e}", flush=True)
    return ok


# =========================================================================
# LLM-02: Real Autoregressive 32-Token Generation with DynamicCache KV-Cache
# =========================================================================
def test_llm02_autoregressive_decode(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(102)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    n_layers = 12 if vram_gb >= 12.0 else 6
    model = make_qwen2_model(num_layers=n_layers).eval()
    prompt_ids = torch.randint(0, 32000, (1, 256), device="cuda")
    hp_fn = make_hp_gemm()

    def run_decode_32():
        cache = DynamicCache()
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        tokens = [next_tok]
        last_logits = out.logits[:, -1, :]
        for _ in range(31):
            out = model(input_ids=next_tok, past_key_values=cache, use_cache=True)
            last_logits = out.logits[:, -1, :]
            next_tok = last_logits.argmax(dim=-1, keepdim=True)
            tokens.append(next_tok)
        return torch.cat(tokens, dim=1), last_logits

    with torch.no_grad():
        with torch.cuda.stream(lp_stream):
            _ = run_decode_32()
        lp_stream.synchronize()

        with torch.cuda.stream(lp_stream):
            ref_toks, ref_log = run_decode_32()
            ref_toks, ref_log = ref_toks.clone(), ref_log.clone()
        lp_stream.synchronize()

        box = {}
        susp, _ = run_with_mid_preempt(xs, xq, lp_stream, hp_stream,
                                       lambda: box.__setitem__("r", run_decode_32()),
                                       delay_ms=2.5, num_preempts=6, hp_fn=hp_fn)

    got_toks, got_log = box["r"]
    exact_toks = int((got_toks == ref_toks).sum().item())
    logit_err = (got_log.float() - ref_log.float()).abs().max().item()
    ok = len(susp) >= 2 and exact_toks == 32 and logit_err == 0.0
    print(f"[LLM-02 Qwen2-{n_layers}L 32-Token Decode+KVCache] {'PASS' if ok else 'FAIL'} | preempts={len(susp)} "
          f"susp_p50={pct(susp, 0.5):.3f} ms exact_toks={exact_toks}/32 logit_err={logit_err:.3e}", flush=True)
    return ok


# =========================================================================
# LLM-03: Real PEFT LoRA SFT Fine-Tuning (Qwen2 + LoraConfig + Fused AdamW)
# =========================================================================
def test_llm03_peft_lora_sft(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(103)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    n_layers = 8 if vram_gb >= 12.0 else 4
    seq_len = 512
    hp_fn = make_hp_gemm()

    def build_lora_model(base_sd=None, lora_sd=None):
        base = make_qwen2_model(num_layers=n_layers, hidden=1024, inter=2816, heads=8, kv_heads=2, dtype=torch.bfloat16)
        if base_sd is not None:
            base.load_state_dict(base_sd)
        lcfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        peft_m = get_peft_model(base, lcfg).cuda().train()
        if lora_sd is not None:
            peft_m.load_state_dict(lora_sd)
        return peft_m

    init_m = build_lora_model()
    init_sd = copy.deepcopy(init_m.state_dict())
    base_init = make_qwen2_model(num_layers=n_layers, hidden=1024, inter=2816, heads=8, kv_heads=2, dtype=torch.bfloat16)
    del base_init

    batches = [torch.randint(0, 32000, (2, seq_len), device="cuda") for _ in range(3)]

    def run_lora_steps(model, opt):
        losses = []
        # Use deterministic MATH SDPA backward for bit-exact golden verification
        # (FlashAttention backward uses non-deterministic atomicAdd ~4e-4 even solo).
        with sdpa_kernel(SDPBackend.MATH):
            for ids in batches:
                opt.zero_grad(set_to_none=True)
                out = model(input_ids=ids, labels=ids, use_cache=False)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.detach().float())
        return losses

    # Warmup
    mw = build_lora_model(lora_sd=init_sd)
    ow = torch.optim.AdamW([p for p in mw.parameters() if p.requires_grad], lr=2e-4, fused=True)
    with torch.cuda.stream(lp_stream):
        run_lora_steps(mw, ow)
    lp_stream.synchronize()
    del mw, ow

    # Golden
    m_ref = build_lora_model(lora_sd=init_sd)
    o_ref = torch.optim.AdamW([p for p in m_ref.parameters() if p.requires_grad], lr=2e-4, fused=True)
    with torch.cuda.stream(lp_stream):
        ref_losses = run_lora_steps(m_ref, o_ref)
    lp_stream.synchronize()

    # Preempted
    m_got = build_lora_model(lora_sd=init_sd)
    o_got = torch.optim.AdamW([p for p in m_got.parameters() if p.requires_grad], lr=2e-4, fused=True)
    box = {}
    susp, _ = run_with_mid_preempt(xs, xq, lp_stream, hp_stream,
                                   lambda: box.__setitem__("l", run_lora_steps(m_got, o_got)),
                                   delay_ms=3.0, num_preempts=6, hp_fn=hp_fn)

    max_loss_err = max((a - b).abs().item() for a, b in zip(box["l"], ref_losses))
    max_lora_err = max(
        (p1.float() - p2.float()).abs().max().item()
        for p1, p2 in zip(m_got.parameters(), m_ref.parameters()) if p1.requires_grad
    )
    ok = len(susp) >= 2 and max_loss_err < 1e-4 and max_lora_err < 1e-4
    print(f"[LLM-03 PEFT LoRA SFT Qwen2-{n_layers}L (r=16)] {'PASS' if ok else 'FAIL'} | preempts={len(susp)} "
          f"susp_p50={pct(susp, 0.5):.3f} ms loss_err={max_loss_err:.3e} lora_w_err={max_lora_err:.3e}", flush=True)
    return ok


# =========================================================================
# LLM-04: Blackwell Quantized LLM Kernels (FP8 _scaled_mm + INT8 _int_mm + Triton W4A16)
# =========================================================================
def test_llm04_quantized_fp8_int8_w4a16(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(104)
    hp_fn = make_hp_gemm()

    # 1) FP8 E4M3 Tensor Core GEMM via torch._scaled_mm
    M, N, K = 4096, 4096, 4096
    a_fp8 = (torch.randn(M, K, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    b_fp8 = (torch.randn(N, K, device="cuda") * 0.25).to(torch.float8_e4m3fn).t()  # column-major
    sa = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    sb = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    # 2) INT8 Tensor Core GEMM via torch._int_mm
    a_i8 = torch.randint(-64, 64, (2048, 2048), device="cuda", dtype=torch.int8)
    b_i8 = torch.randint(-64, 64, (2048, 2048), device="cuda", dtype=torch.int8)

    # 3) Triton W4A16 (INT4 packed weight-only dequant GEMM)
    Mw, Nw, Kw = 1024, 4096, 4096
    GROUP = 128
    a_w4 = torch.randn(Mw, Kw, device="cuda", dtype=torch.float16) * 0.1
    q_w4 = torch.randint(0, 256, (Kw // 2, Nw), device="cuda", dtype=torch.uint8)
    s_w4 = (torch.rand(Kw // GROUP, Nw, device="cuda", dtype=torch.float16) * 0.05 + 0.01)
    c_w4 = torch.empty(Mw, Nw, device="cuda", dtype=torch.float16)

    def run_quant():
        y_fp8 = None
        for _ in range(4):
            y_fp8 = torch._scaled_mm(a_fp8, b_fp8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        y_i8 = None
        for _ in range(4):
            y_i8 = torch._int_mm(a_i8, b_i8)
        grid = (triton.cdiv(Mw, 64), triton.cdiv(Nw, 64))
        for _ in range(4):
            triton_w4a16_gemm_kernel[grid](
                a_w4, q_w4, s_w4, c_w4, Mw, Nw, Kw,
                a_w4.stride(0), a_w4.stride(1),
                q_w4.stride(0), q_w4.stride(1),
                s_w4.stride(0), s_w4.stride(1),
                c_w4.stride(0), c_w4.stride(1),
                GROUP_SIZE=GROUP, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            )
        return y_fp8, y_i8, c_w4.clone()

    with torch.no_grad():
        with torch.cuda.stream(lp_stream):
            _ = run_quant()
        lp_stream.synchronize()
        with torch.cuda.stream(lp_stream):
            ref_fp8, ref_i8, ref_w4 = run_quant()
            ref_fp8, ref_i8, ref_w4 = ref_fp8.clone(), ref_i8.clone(), ref_w4.clone()
        lp_stream.synchronize()

        box = {}
        susp, _ = run_with_mid_preempt(xs, xq, lp_stream, hp_stream,
                                       lambda: box.__setitem__("r", run_quant()),
                                       delay_ms=2.0, num_preempts=4, hp_fn=hp_fn)

    g_fp8, g_i8, g_w4 = box["r"]
    e_fp8 = (g_fp8.float() - ref_fp8.float()).abs().max().item()
    e_i8 = (g_i8 - ref_i8).abs().max().item()
    e_w4 = (g_w4.float() - ref_w4.float()).abs().max().item()
    ok = len(susp) >= 1 and e_fp8 == 0.0 and e_i8 == 0 and e_w4 == 0.0
    print(f"[LLM-04 Quantized FP8+INT8+Triton-W4A16] {'PASS' if ok else 'FAIL'} | preempts={len(susp)} "
          f"susp={[round(v, 3) for v in susp]} e_fp8={e_fp8:.3e} e_int8={e_i8} e_w4a16={e_w4:.3e}", flush=True)
    return ok


# =========================================================================
# LLM-05: Real vLLM-Style Triton PagedAttention + Dynamic Seq-Len Bucketing
# =========================================================================
def test_llm05_paged_attn_and_dynamic_shapes(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(105)
    hp_fn = make_hp_gemm()

    num_seqs = 32
    num_heads = 16
    num_kv_heads = 4
    head_dim = 128
    block_size = 16
    max_blocks = 128  # up to 2048 context
    total_phys_blocks = num_seqs * max_blocks

    q = torch.randn(num_seqs, num_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1
    k_cache = torch.randn(total_phys_blocks, num_kv_heads, block_size, head_dim, device="cuda", dtype=torch.float16) * 0.1
    v_cache = torch.randn(total_phys_blocks, num_kv_heads, block_size, head_dim, device="cuda", dtype=torch.float16) * 0.1
    # Shuffled non-contiguous physical block tables (real vLLM page allocation!)
    perm = torch.randperm(total_phys_blocks, device="cuda", dtype=torch.int32)
    block_tables = perm.view(num_seqs, max_blocks).contiguous()
    ctx_lens = torch.randint(512, max_blocks * block_size, (num_seqs,), device="cuda", dtype=torch.int32)
    out_pa = torch.empty_like(q)
    scale = 1.0 / math.sqrt(head_dim)

    # Also test dynamic sequence length bucketing on Qwen2-4L (stresses EnsureArenaPool / layout_stage_)
    m_dyn = make_qwen2_model(num_layers=4, hidden=1024, inter=2816, heads=8, kv_heads=2).eval()
    dyn_inputs = [torch.randint(0, 32000, (1, L), device="cuda") for L in (128, 512, 1024, 256, 1536, 384)]

    def run_all():
        for _ in range(10):
            triton_paged_attention_v1[(num_seqs, num_heads)](
                out_pa, q, k_cache, v_cache, block_tables, ctx_lens, scale,
                q.stride(0), q.stride(1), q.stride(2),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
                block_tables.stride(0), block_tables.stride(1),
                NUM_KV_HEADS=num_kv_heads, GQA_RATIO=num_heads // num_kv_heads,
                BLOCK_SIZE=block_size, HEAD_DIM=head_dim, MAX_BLOCKS=max_blocks,
            )
        outs = [m_dyn(input_ids=inp, use_cache=False).logits[:, -1, :] for inp in dyn_inputs]
        return out_pa.clone(), outs

    with torch.no_grad():
        with torch.cuda.stream(lp_stream):
            _ = run_all()
        lp_stream.synchronize()
        with torch.cuda.stream(lp_stream):
            ref_pa, ref_dyn = run_all()
            ref_pa = ref_pa.clone()
            ref_dyn = [t.clone() for t in ref_dyn]
        lp_stream.synchronize()

        box = {}
        susp, _ = run_with_mid_preempt(xs, xq, lp_stream, hp_stream,
                                       lambda: box.__setitem__("r", run_all()),
                                       delay_ms=2.0, num_preempts=5, hp_fn=hp_fn)

    g_pa, g_dyn = box["r"]
    e_pa = (g_pa.float() - ref_pa.float()).abs().max().item()
    e_dyn = max((a.float() - b.float()).abs().max().item() for a, b in zip(g_dyn, ref_dyn))
    ok = len(susp) >= 2 and e_pa == 0.0 and e_dyn == 0.0
    print(f"[LLM-05 Triton PagedAttention + DynamicBuckets] {'PASS' if ok else 'FAIL'} | preempts={len(susp)} "
          f"susp_p50={pct(susp, 0.5):.3f} ms e_paged={e_pa:.3e} e_dyn={e_dyn:.3e}", flush=True)
    return ok


# =========================================================================
# LLM-06: Production Co-Location QoS #1: Long-Context Prefill (LP) + Real-Time Decode (HP)
# =========================================================================
def test_llm06_qos_prefill_decode(mode: str):
    torch.manual_seed(106)
    xs = XSchedBridge()
    lp_stream = torch.cuda.Stream()
    hp_stream = torch.cuda.Stream(priority=-1)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    n_layers = 12 if vram_gb >= 12.0 else 6
    prefill_len = 2048 if vram_gb >= 12.0 else 1024
    model = make_qwen2_model(num_layers=n_layers).eval()

    lp_ids = torch.randint(0, 32000, (1, prefill_len), device="cuda")
    hp_prompt = torch.randint(0, 32000, (1, 128), device="cuda")
    hp_tok = torch.randint(0, 32000, (1, 1), device="cuda")

    # Build HP KV-cache once outside timing
    with torch.no_grad():
        hp_cache_base = DynamicCache()
        _ = model(input_ids=hp_prompt, past_key_values=hp_cache_base, use_cache=True)
        torch.cuda.synchronize()
        # Standalone HP single-token decode latency baseline
        with torch.cuda.stream(hp_stream):
            _ = model(input_ids=hp_tok, past_key_values=copy.deepcopy(hp_cache_base), use_cache=True)
        hp_stream.synchronize()

    hwq, xq = (None, None)
    if mode == "lv1":
        hwq, xq = xs.create_xqueue(lp_stream, XSchedBridge.kPreemptLevelBlock)
    elif mode == "m5":
        hwq, xq = xs.create_xqueue(lp_stream, XSchedBridge.kPreemptLevelInterrupt)

    # Golden LP prefill + LP standalone throughput
    with torch.no_grad():
        with torch.cuda.stream(lp_stream):
            _ = model(input_ids=lp_ids, use_cache=False)
        lp_stream.synchronize()
        t_s0 = time.perf_counter()
        with torch.cuda.stream(lp_stream):
            for _ in range(5):
                ref_lp = model(input_ids=lp_ids, use_cache=False).logits[:, -1, :]
        lp_stream.synchronize()
        lp_solo_s = (time.perf_counter() - t_s0) / 5.0
        with torch.cuda.stream(lp_stream):      # T-29: clone on the stream that owns ref_lp
            ref_lp = ref_lp.clone()
        lp_stream.synchronize()

        # Golden HP decode output (clone KV cache on hp_stream + sync!)
        with torch.cuda.stream(hp_stream):
            c_ref = copy.deepcopy(hp_cache_base)
            ref_hp = model(input_ids=hp_tok, past_key_values=c_ref, use_cache=True).logits[:, -1, :].clone()
        hp_stream.synchronize()

        n_req = 15
        hp_caches = [copy.deepcopy(hp_cache_base) for _ in range(n_req)]
        torch.cuda.synchronize()

    stop_ev = threading.Event()
    lp_iters = [0]
    lp_max_err = [0.0]
    lp_err_exc = []

    def lp_worker():
        try:
            torch.cuda.set_device(0)
            with torch.no_grad():
                while not stop_ev.is_set():
                    with torch.cuda.stream(lp_stream):
                        y = model(input_ids=lp_ids, use_cache=False).logits[:, -1, :]
                    lp_stream.synchronize()
                    e = (y.float() - ref_lp.float()).abs().max().item()
                    if e > lp_max_err[0]:
                        lp_max_err[0] = e
                    lp_iters[0] += 1
        except Exception as exc:
            lp_err_exc.append(exc)

    t = threading.Thread(target=lp_worker)
    t.start()
    time.sleep(0.05)

    hp_tpot_ms = []
    susp_ms = []
    hp_max_err = 0.0
    t_bench0 = time.perf_counter()
    with torch.no_grad():
        for req_i in range(n_req):
            time.sleep(0.025)
            c_step = hp_caches[req_i]
            t0 = time.perf_counter()
            if xq is not None:
                xs.suspend(xq)
                t_s = time.perf_counter()
                susp_ms.append((t_s - t0) * 1000.0)
            with torch.cuda.stream(hp_stream):
                yh = model(input_ids=hp_tok, past_key_values=c_step, use_cache=True).logits[:, -1, :]
            hp_stream.synchronize()
            t1 = time.perf_counter()
            if xq is not None:
                xs.resume(xq)
            hp_tpot_ms.append((t1 - t0) * 1000.0)
            he = (yh.float() - ref_hp.float()).abs().max().item()
            if he > hp_max_err:
                hp_max_err = he

    total_s = time.perf_counter() - t_bench0
    stop_ev.set()
    t.join()
    if xq is not None:
        xs.destroy(hwq, xq)
    if lp_err_exc:
        raise lp_err_exc[0]

    hp_total_s = sum(hp_tpot_ms) / 1000.0
    expected_lp = max(0.01, (total_s - hp_total_s) / lp_solo_s)
    lp_ret = min(130.0, 100.0 * lp_iters[0] / expected_lp)
    slo_10 = 100.0 * sum(1 for v in hp_tpot_ms if v <= 10.0) / len(hp_tpot_ms)
    ok = lp_max_err[0] == 0.0 and hp_max_err == 0.0 and lp_iters[0] >= 1
    print(f"[LLM-06 QoS Prefill+Decode mode={mode}] {'PASS' if ok else 'FAIL'} | "
          f"HP_TPOT P50={pct(hp_tpot_ms, 0.5):.3f}ms P99={pct(hp_tpot_ms, 0.99):.3f}ms SLO<=10ms={slo_10:.1f}% | "
          f"SUSP_P50={pct(susp_ms, 0.5):.3f}ms | LP_iters={lp_iters[0]} LP_ret={lp_ret:.1f}% | "
          f"lp_err={lp_max_err[0]:.3e} hp_err={hp_max_err:.3e}", flush=True)
    return ok


# =========================================================================
# LLM-07: Production Co-Location QoS #2: Real PEFT LoRA Fine-Tuning (LP) + Online Decode (HP)
# =========================================================================
def test_llm07_qos_lora_serve(mode: str):
    torch.manual_seed(107)
    xs = XSchedBridge()
    lp_stream = torch.cuda.Stream()
    hp_stream = torch.cuda.Stream(priority=-1)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    n_layers = 8 if vram_gb >= 12.0 else 4
    base_lp = make_qwen2_model(num_layers=n_layers, hidden=1024, inter=2816, heads=8, kv_heads=2, dtype=torch.bfloat16)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    lp_model = get_peft_model(base_lp, lcfg).cuda().train()
    lp_opt = torch.optim.AdamW([p for p in lp_model.parameters() if p.requires_grad], lr=2e-4, fused=True)
    lp_ids = torch.randint(0, 32000, (2, 512), device="cuda")

    hp_model = make_qwen2_model(num_layers=n_layers, hidden=1024, inter=2816, heads=8, kv_heads=2, dtype=torch.bfloat16).eval()
    hp_tok = torch.randint(0, 32000, (1, 1), device="cuda")
    hp_prompt = torch.randint(0, 32000, (1, 128), device="cuda")
    with torch.no_grad():
        hp_cache_base = DynamicCache()
        _ = hp_model(input_ids=hp_prompt, past_key_values=hp_cache_base, use_cache=True)
        torch.cuda.synchronize()
        with torch.cuda.stream(hp_stream):
            ref_hp = hp_model(input_ids=hp_tok, past_key_values=copy.deepcopy(hp_cache_base), use_cache=True).logits[:, -1, :].clone()
        hp_stream.synchronize()
        n_req = 15
        hp_caches = [copy.deepcopy(hp_cache_base) for _ in range(n_req)]
        torch.cuda.synchronize()

    hwq, xq = (None, None)
    if mode == "lv1":
        hwq, xq = xs.create_xqueue(lp_stream, XSchedBridge.kPreemptLevelBlock)
    elif mode == "m5":
        hwq, xq = xs.create_xqueue(lp_stream, XSchedBridge.kPreemptLevelInterrupt)

    # Warmup & standalone LP step time
    with torch.cuda.stream(lp_stream):
        lp_opt.zero_grad(set_to_none=True)
        lp_model(input_ids=lp_ids, labels=lp_ids, use_cache=False).loss.backward()
        lp_opt.step()
    lp_stream.synchronize()
    t0s = time.perf_counter()
    with torch.cuda.stream(lp_stream):
        for _ in range(4):
            lp_opt.zero_grad(set_to_none=True)
            lp_model(input_ids=lp_ids, labels=lp_ids, use_cache=False).loss.backward()
            lp_opt.step()
    lp_stream.synchronize()
    lp_solo_s = (time.perf_counter() - t0s) / 4.0

    stop_ev = threading.Event()
    lp_steps = [0]
    lp_nan = [False]
    lp_exc = []

    def lp_worker():
        try:
            torch.cuda.set_device(0)
            while not stop_ev.is_set():
                with torch.cuda.stream(lp_stream):
                    lp_opt.zero_grad(set_to_none=True)
                    loss = lp_model(input_ids=lp_ids, labels=lp_ids, use_cache=False).loss
                    loss.backward()
                    lp_opt.step()
                lp_stream.synchronize()
                if not torch.isfinite(loss).item():
                    lp_nan[0] = True
                lp_steps[0] += 1
        except Exception as exc:
            lp_exc.append(exc)

    t = threading.Thread(target=lp_worker)
    t.start()
    time.sleep(0.05)

    hp_tpot_ms = []
    susp_ms = []
    hp_max_err = 0.0
    t_b0 = time.perf_counter()
    with torch.no_grad():
        for req_i in range(n_req):
            time.sleep(0.025)
            c_step = hp_caches[req_i]
            t0 = time.perf_counter()
            if xq is not None:
                xs.suspend(xq)
                susp_ms.append((time.perf_counter() - t0) * 1000.0)
            with torch.cuda.stream(hp_stream):
                yh = hp_model(input_ids=hp_tok, past_key_values=c_step, use_cache=True).logits[:, -1, :]
            hp_stream.synchronize()
            t1 = time.perf_counter()
            if xq is not None:
                xs.resume(xq)
            hp_tpot_ms.append((t1 - t0) * 1000.0)
            he = (yh.float() - ref_hp.float()).abs().max().item()
            if he > hp_max_err:
                hp_max_err = he

    total_s = time.perf_counter() - t_b0
    stop_ev.set()
    t.join()
    if xq is not None:
        xs.destroy(hwq, xq)
    if lp_exc:
        raise lp_exc[0]

    hp_total_s = sum(hp_tpot_ms) / 1000.0
    expected_steps = max(0.01, (total_s - hp_total_s) / lp_solo_s)
    lp_ret = min(130.0, 100.0 * lp_steps[0] / expected_steps)
    slo_8 = 100.0 * sum(1 for v in hp_tpot_ms if v <= 8.0) / len(hp_tpot_ms)
    ok = (not lp_nan[0]) and hp_max_err == 0.0 and lp_steps[0] >= 1
    print(f"[LLM-07 QoS LoRA-Train+Decode mode={mode}] {'PASS' if ok else 'FAIL'} | "
          f"HP_TPOT P50={pct(hp_tpot_ms, 0.5):.3f}ms P99={pct(hp_tpot_ms, 0.99):.3f}ms SLO<=8ms={slo_8:.1f}% | "
          f"SUSP_P50={pct(susp_ms, 0.5):.3f}ms | LP_steps={lp_steps[0]} LP_ret={lp_ret:.1f}% | hp_err={hp_max_err:.3e}", flush=True)
    return ok


# =========================================================================
# LLM-08: Multi-Stream Concurrent Production Isolation (2x LP XQueues + 1x HP)
# =========================================================================
def test_llm08_multi_stream_concurrent():
    torch.manual_seed(108)
    xs = XSchedBridge()
    lp1_s = torch.cuda.Stream()
    lp2_s = torch.cuda.Stream()
    hp_s = torch.cuda.Stream(priority=-1)
    lvl = int(os.environ.get("XSCHED_TEST_LEVEL", str(XSchedBridge.kPreemptLevelInterrupt)))
    hwq1, xq1 = xs.create_xqueue(lp1_s, lvl)
    hwq2, xq2 = xs.create_xqueue(lp2_s, lvl)

    m1 = make_qwen2_model(num_layers=6, hidden=1024, inter=2816, heads=8, kv_heads=2).eval()
    m2 = make_qwen2_model(num_layers=6, hidden=1024, inter=2816, heads=8, kv_heads=2).eval()
    x1 = torch.randint(0, 32000, (1, 1024), device="cuda")
    x2 = torch.randint(0, 32000, (1, 1024), device="cuda")
    hp_fn = make_hp_gemm()

    with torch.no_grad():
        with torch.cuda.stream(lp1_s):
            _ = m1(input_ids=x1, use_cache=False)
            ref1 = m1(input_ids=x1, use_cache=False).logits[:, -1, :].clone()
        lp1_s.synchronize()
        with torch.cuda.stream(lp2_s):
            _ = m2(input_ids=x2, use_cache=False)
            ref2 = m2(input_ids=x2, use_cache=False).logits[:, -1, :].clone()
        lp2_s.synchronize()

        done1, done2 = threading.Event(), threading.Event()
        res, errs = {}, []

        def w1():
            try:
                torch.cuda.set_device(0)
                with torch.cuda.stream(lp1_s):
                    for _ in range(3):
                        y = m1(input_ids=x1, use_cache=False).logits[:, -1, :]
                    res["y1"] = y.clone()          # T-29: clone on the owning stream
                lp1_s.synchronize()
            except Exception as e:
                errs.append(e)
            finally:
                done1.set()

        def w2():
            try:
                torch.cuda.set_device(0)
                with torch.cuda.stream(lp2_s):
                    for _ in range(3):
                        y = m2(input_ids=x2, use_cache=False).logits[:, -1, :]
                    res["y2"] = y.clone()          # T-29: clone on the owning stream
                lp2_s.synchronize()
            except Exception as e:
                errs.append(e)
            finally:
                done2.set()

        t1 = threading.Thread(target=w1)
        t2 = threading.Thread(target=w2)
        t1.start(); t2.start()

        susp1, susp2 = [], []
        for step in range(4):
            time.sleep(0.004)
            if done1.is_set() and done2.is_set():
                break
            if step % 2 == 0 and not done1.is_set():
                ts0 = time.perf_counter(); xs.suspend(xq1); susp1.append((time.perf_counter() - ts0) * 1000.0)
                with torch.cuda.stream(hp_s): hp_fn()
                hp_s.synchronize()
                xs.resume(xq1)
            elif not done2.is_set():
                ts0 = time.perf_counter(); xs.suspend(xq2); susp2.append((time.perf_counter() - ts0) * 1000.0)
                with torch.cuda.stream(hp_s): hp_fn()
                hp_s.synchronize()
                xs.resume(xq2)

        t1.join(); t2.join()

    xs.destroy(hwq1, xq1)
    xs.destroy(hwq2, xq2)
    if errs:
        raise errs[0]

    e1 = (res["y1"].float() - ref1.float()).abs().max().item()
    e2 = (res["y2"].float() - ref2.float()).abs().max().item()
    ok = (len(susp1) + len(susp2)) >= 2 and e1 == 0.0 and e2 == 0.0
    print(f"[LLM-08 Multi-Stream 2xLP+1xHP Concurrent] {'PASS' if ok else 'FAIL'} | "
          f"p1={len(susp1)} susp1={[round(v, 3) for v in susp1]} | p2={len(susp2)} susp2={[round(v, 3) for v in susp2]} | "
          f"e1={e1:.3e} e2={e2:.3e}", flush=True)
    return ok


# =========================================================================
# LLM-09: Real Pretrained Open-Weights LLM (Qwen/Qwen2.5-0.5B-Instruct)
# =========================================================================
def test_llm09_real_pretrained_qwen25(xs, lp_stream, hp_stream, xq):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("REAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda().eval()
    prompt = "Explain why fine-grained GPU preemption improves tail latency for LLM serving in two sentences:"
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    hp_fn = make_hp_gemm()

    def run_gen():
        out = model.generate(
            **inputs, max_new_tokens=32, do_sample=False, use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
        return out

    with torch.no_grad():
        with torch.cuda.stream(lp_stream):
            _ = run_gen()
        lp_stream.synchronize()
        with torch.cuda.stream(lp_stream):
            ref_ids = run_gen().clone()
        lp_stream.synchronize()

        box = {}
        susp, _ = run_with_mid_preempt(xs, xq, lp_stream, hp_stream,
                                       lambda: box.__setitem__("ids", run_gen()),
                                       delay_ms=4.0, num_preempts=6, hp_fn=hp_fn)

    got_ids = box["ids"]
    exact = torch.equal(got_ids, ref_ids)
    text_got = tok.decode(got_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).replace("\n", " ")
    ok = len(susp) >= 2 and exact
    print(f"[LLM-09 Real Pretrained {model_id}] {'PASS' if ok else 'FAIL'} | preempts={len(susp)} "
          f"susp_p50={pct(susp, 0.5):.3f} ms exact_match={exact} | text=\"{text_got[:90]}...\"", flush=True)
    return ok


def main():
    target = sys.argv[1] if len(sys.argv) >= 2 else "all"

    if target.startswith("llm06:"):
        ok = test_llm06_qos_prefill_decode(target.split(":", 1)[1])
        sys.exit(0 if ok else 1)
    if target.startswith("llm07:"):
        ok = test_llm07_qos_lora_serve(target.split(":", 1)[1])
        sys.exit(0 if ok else 1)
    if target == "llm08":
        ok = test_llm08_multi_stream_concurrent()
        sys.exit(0 if ok else 1)

    xs = XSchedBridge()
    lp_stream = torch.cuda.Stream()
    hp_stream = torch.cuda.Stream(priority=-1)
    lvl = int(os.environ.get("XSCHED_TEST_LEVEL", str(XSchedBridge.kPreemptLevelInterrupt)))
    hwq, xq = xs.create_xqueue(lp_stream, lvl)

    tests = [
        ("llm01", test_llm01_qwen2_prefill),
        ("llm02", test_llm02_autoregressive_decode),
        ("llm03", test_llm03_peft_lora_sft),
        ("llm04", test_llm04_quantized_fp8_int8_w4a16),
        ("llm05", test_llm05_paged_attn_and_dynamic_shapes),
        ("llm09", test_llm09_real_pretrained_qwen25),
    ]

    all_ok = True
    try:
        for name, fn in tests:
            if target == "all" or target == name:
                if target == "all" and name == "llm09" and not os.environ.get("ENABLE_REAL_WEIGHTS"):
                    continue
                if not fn(xs, lp_stream, hp_stream, xq):
                    all_ok = False
    finally:
        xs.destroy(hwq, xq)

    if target == "all":
        if not test_llm08_multi_stream_concurrent():
            all_ok = False
        if all_ok:
            print("PHASE3_LLM_PROD_ALL_PASS", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
