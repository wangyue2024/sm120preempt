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
import torchvision.models as models
import torchvision.ops as tv_ops


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


def run_on_main_with_bg_preempt(xs, xq, lp_stream, hp_stream, fn, delay_ms=2.5, num_preempts=3, hp_fn=None):
    """Run LP on calling thread (preserving PyTorch Inductor/CUDAGraph TLS & per-thread cuBLASLt),
    while a background thread fires mid-kernel preemptions + HP workload."""
    done = threading.Event()
    susp_ms = []
    hp_ms = []
    bg_err = []

    sms = max(30, torch.cuda.get_device_properties(0).multi_processor_count)
    eff_delay_s = max(0.0005, (delay_ms / 1000.0) * (30.0 / float(sms)))

    def preemptor():
        try:
            torch.cuda.set_device(0)
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
        except Exception as e:
            bg_err.append(e)

    t = threading.Thread(target=preemptor)
    t.start()
    try:
        with torch.cuda.stream(lp_stream):
            res = fn()
        lp_stream.synchronize()
    finally:
        done.set()
        t.join()
    if bg_err:
        raise bg_err[0]
    return res, susp_ms, hp_ms


def make_hp_gemm():
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    return lambda: torch.mm(a, b)


# =========================================================================
# DNN-01: Vision Transformer ViT-B/16 (86M params, BF16 Inference)
# =========================================================================
def test_dnn01_vit_b16(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(42)
    model = models.vit_b_16(weights=None).cuda().eval().to(torch.bfloat16)
    x = torch.randn(16, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    hp_fn = make_hp_gemm()
    torch.cuda.synchronize()

    with torch.no_grad():
        # Warmup
        with torch.cuda.stream(lp_stream):
            _ = model(x)
        lp_stream.synchronize()
        # Golden
        with torch.cuda.stream(lp_stream):
            ref = model(x).clone()
        lp_stream.synchronize()

        out_box = {}
        def run_lp():
            out_box["y"] = model(x)

        susp, hp = run_with_mid_preempt(xs, xq, lp_stream, hp_stream, run_lp, delay_ms=3.0, num_preempts=3, hp_fn=hp_fn)

    max_err = (out_box["y"].float() - ref.float()).abs().max().item()
    ok = len(susp) >= 1 and max_err == 0.0
    print(f"[DNN-01 ViT-B/16 BF16 B=16] {'PASS' if ok else 'FAIL'} | preempts={len(susp)} "
          f"susp_ms={[round(v, 3) for v in susp]} hp_ms={[round(v, 3) for v in hp]} max_err={max_err:.3e}", flush=True)
    return ok


# =========================================================================
# DNN-02: Swin-B (Shifted Window Roll + RelPos) & ConvNeXt-Base (Channels-Last NHWC)
# =========================================================================
def test_dnn02_swin_convnext(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(43)
    hp_fn = make_hp_gemm()

    # Part A: Swin-B
    swin = models.swin_b(weights=None).cuda().eval().to(torch.bfloat16)
    x_swin = torch.randn(8, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        with torch.cuda.stream(lp_stream):
            _ = swin(x_swin)
        lp_stream.synchronize()
        with torch.cuda.stream(lp_stream):
            ref_swin = swin(x_swin).clone()
        lp_stream.synchronize()

        box_s = {}
        susp_s, _ = run_with_mid_preempt(xs, xq, lp_stream, hp_stream, lambda: box_s.__setitem__("y", swin(x_swin)),
                                         delay_ms=2.5, num_preempts=3, hp_fn=hp_fn)
    err_s = (box_s["y"].float() - ref_swin.float()).abs().max().item()
    del swin, x_swin, ref_swin, box_s
    torch.cuda.empty_cache()

    # Part B: ConvNeXt-Base Channels-Last (NHWC 7x7 depthwise conv)
    cnxt = models.convnext_base(weights=None).cuda().eval().to(torch.bfloat16, memory_format=torch.channels_last)
    x_c = torch.randn(16, 3, 224, 224, device="cuda", dtype=torch.bfloat16).to(memory_format=torch.channels_last)
    with torch.no_grad():
        with torch.cuda.stream(lp_stream):
            _ = cnxt(x_c)
        lp_stream.synchronize()
        with torch.cuda.stream(lp_stream):
            ref_c = cnxt(x_c).clone()
        lp_stream.synchronize()

        box_c = {}
        susp_c, _ = run_with_mid_preempt(xs, xq, lp_stream, hp_stream, lambda: box_c.__setitem__("y", cnxt(x_c)),
                                         delay_ms=2.5, num_preempts=3, hp_fn=hp_fn)
    err_c = (box_c["y"].float() - ref_c.float()).abs().max().item()

    ok = len(susp_s) >= 1 and len(susp_c) >= 1 and err_s == 0.0 and err_c == 0.0
    print(f"[DNN-02 Swin-B + ConvNeXt-NHWC] {'PASS' if ok else 'FAIL'} | "
          f"swin_preempts={len(susp_s)} susp={[round(v, 3) for v in susp_s]} err={err_s:.3e} | "
          f"cnxt_preempts={len(susp_c)} susp={[round(v, 3) for v in susp_c]} err={err_c:.3e}", flush=True)
    return ok


# =========================================================================
# DNN-03: DeepLabV3-ResNet50 (Dilated Conv + Upsample) + RoIAlign + NMS
# =========================================================================
def test_dnn03_seg_det_ops(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(44)
    hp_fn = make_hp_gemm()
    seg = models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None).cuda().eval().half()
    x = torch.randn(4, 3, 384, 384, device="cuda", dtype=torch.float16)

    # Detection ops: RoIAlign + NMS on irregular boxes
    feats = torch.randn(4, 256, 48, 48, device="cuda", dtype=torch.float32)
    b_idx = torch.randint(0, 4, (512, 1), device="cuda", dtype=torch.float32)
    xy1 = torch.rand(512, 2, device="cuda", dtype=torch.float32) * 200.0
    xy2 = xy1 + torch.rand(512, 2, device="cuda", dtype=torch.float32) * 150.0 + 10.0
    rois = torch.cat([b_idx, xy1, xy2], dim=1)
    scores = torch.rand(512, device="cuda", dtype=torch.float32)

    def run_all():
        out_seg = seg(x)["out"]
        pooled = tv_ops.roi_align(feats, rois, output_size=(7, 7), spatial_scale=1.0 / 8.0, aligned=True)
        keep = tv_ops.nms(rois[:, 1:], scores, iou_threshold=0.5)
        return out_seg, pooled, keep

    with torch.no_grad():
        with torch.cuda.stream(lp_stream):
            _ = run_all()
        lp_stream.synchronize()
        with torch.cuda.stream(lp_stream):
            ref_seg, ref_roi, ref_keep = run_all()
            ref_seg, ref_roi, ref_keep = ref_seg.clone(), ref_roi.clone(), ref_keep.clone()
        lp_stream.synchronize()

        box = {}
        susp, _ = run_with_mid_preempt(xs, xq, lp_stream, hp_stream, lambda: box.__setitem__("r", run_all()),
                                       delay_ms=2.5, num_preempts=3, hp_fn=hp_fn)

    got_seg, got_roi, got_keep = box["r"]
    err_seg = (got_seg.float() - ref_seg.float()).abs().max().item()
    err_roi = (got_roi - ref_roi).abs().max().item()
    nms_exact = torch.equal(got_keep, ref_keep)
    ok = len(susp) >= 1 and err_seg == 0.0 and err_roi == 0.0 and nms_exact
    print(f"[DNN-03 DeepLabV3 + RoIAlign + NMS] {'PASS' if ok else 'FAIL'} | preempts={len(susp)} "
          f"susp={[round(v, 3) for v in susp]} err_seg={err_seg:.3e} err_roi={err_roi:.3e} nms_exact={nms_exact}", flush=True)
    return ok


# =========================================================================
# DNN-04: Production AMP BF16 Training + 4x GradAccum + clip_grad_norm_ + AdamW(fused=True)
# =========================================================================
def test_dnn04_amp_fused_adamw(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(45)
    hp_fn = make_hp_gemm()
    base = models.resnet34(weights=None).cuda().train()
    init_sd = copy.deepcopy(base.state_dict())

    micro_xs = [torch.randn(16, 3, 128, 128, device="cuda") for _ in range(4)]
    micro_ys = [torch.randint(0, 1000, (16,), device="cuda") for _ in range(4)]

    def run_train_step(model, opt):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for mx, my in zip(micro_xs, micro_ys):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(mx)
                loss = F.cross_entropy(logits, my) / 4.0
            loss.backward()
            total_loss = total_loss + loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        return total_loss

    # Warmup kernels on lp_stream
    m_warm = models.resnet34(weights=None).cuda().train()
    m_warm.load_state_dict(init_sd)
    opt_warm = torch.optim.AdamW(m_warm.parameters(), lr=3e-4, fused=True)
    with torch.cuda.stream(lp_stream):
        run_train_step(m_warm, opt_warm)
    lp_stream.synchronize()
    del m_warm, opt_warm

    # Golden
    m_ref = models.resnet34(weights=None).cuda().train()
    m_ref.load_state_dict(init_sd)
    opt_ref = torch.optim.AdamW(m_ref.parameters(), lr=3e-4, fused=True)
    with torch.cuda.stream(lp_stream):
        loss_ref = run_train_step(m_ref, opt_ref)
    lp_stream.synchronize()

    # Preempted
    m_got = models.resnet34(weights=None).cuda().train()
    m_got.load_state_dict(init_sd)
    opt_got = torch.optim.AdamW(m_got.parameters(), lr=3e-4, fused=True)
    box = {}
    susp, _ = run_with_mid_preempt(xs, xq, lp_stream, hp_stream,
                                   lambda: box.__setitem__("loss", run_train_step(m_got, opt_got)),
                                   delay_ms=2.5, num_preempts=4, hp_fn=hp_fn)

    loss_err = (box["loss"] - loss_ref).abs().item()
    max_w_err = max((p1 - p2).abs().max().item() for p1, p2 in zip(m_got.parameters(), m_ref.parameters()))
    ok = len(susp) >= 1 and loss_err < 1e-5 and max_w_err < 1e-5
    print(f"[DNN-04 AMP-BF16 + 4xGradAccum + FusedAdamW] {'PASS' if ok else 'FAIL'} | preempts={len(susp)} "
          f"susp={[round(v, 3) for v in susp]} loss_err={loss_err:.3e} max_w_err={max_w_err:.3e}", flush=True)
    return ok


# =========================================================================
# DNN-05: Long-Horizon 20-Step Non-Idempotent Training Convergence Under Continuous Preemption
# =========================================================================
def test_dnn05_long_horizon_convergence(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(46)
    hp_fn = make_hp_gemm()

    # Deterministic transformer-like MLP + Conv hybrid (no non-deterministic atomicAdd backward)
    class DetNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1024, 2048)
            self.ln1 = nn.LayerNorm(2048)
            self.fc2 = nn.Linear(2048, 2048)
            self.ln2 = nn.LayerNorm(2048)
            self.fc3 = nn.Linear(2048, 128)

        def forward(self, x):
            h = F.gelu(self.ln1(self.fc1(x)))
            h = h + F.gelu(self.ln2(self.fc2(h)))
            return self.fc3(h)

    init_m = DetNet().cuda()
    init_sd = copy.deepcopy(init_m.state_dict())
    steps = 20
    xs_data = [torch.randn(512, 1024, device="cuda") for _ in range(steps)]
    ys_data = [torch.randint(0, 128, (512,), device="cuda") for _ in range(steps)]

    def run_20_steps(model, opt):
        losses = []
        for s in range(steps):
            opt.zero_grad(set_to_none=True)
            out = model(xs_data[s])
            loss = F.cross_entropy(out, ys_data[s])
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        return losses

    # Warmup
    mw = DetNet().cuda(); mw.load_state_dict(init_sd)
    ow = torch.optim.AdamW(mw.parameters(), lr=1e-3, fused=True)
    with torch.cuda.stream(lp_stream):
        run_20_steps(mw, ow)
    lp_stream.synchronize()
    del mw, ow

    # Golden
    m_ref = DetNet().cuda(); m_ref.load_state_dict(init_sd)
    o_ref = torch.optim.AdamW(m_ref.parameters(), lr=1e-3, fused=True)
    with torch.cuda.stream(lp_stream):
        losses_ref = run_20_steps(m_ref, o_ref)
    lp_stream.synchronize()

    # Preempted continuously every 2 ms across all 20 steps
    m_got = DetNet().cuda(); m_got.load_state_dict(init_sd)
    o_got = torch.optim.AdamW(m_got.parameters(), lr=1e-3, fused=True)
    box = {}
    susp, hp = run_with_mid_preempt(xs, xq, lp_stream, hp_stream,
                                    lambda: box.__setitem__("l", run_20_steps(m_got, o_got)),
                                    delay_ms=2.0, num_preempts=40, hp_fn=hp_fn)

    max_loss_diff = max((a - b).abs().item() for a, b in zip(box["l"], losses_ref))
    max_w_diff = max((p1 - p2).abs().max().item() for p1, p2 in zip(m_got.parameters(), m_ref.parameters()))
    susp_med = sorted(susp)[len(susp) // 2] if susp else 0.0
    ok = len(susp) >= 5 and max_loss_diff < 1e-5 and max_w_diff < 1e-5
    print(f"[DNN-05 20-Step Convergence ({len(susp)} preempts)] {'PASS' if ok else 'FAIL'} | "
          f"susp_p50={susp_med:.3f} ms max_loss_diff={max_loss_diff:.3e} max_w_diff={max_w_diff:.3e}", flush=True)
    return ok


# =========================================================================
# DNN-06: torch.compile(backend="inductor", mode="default") (Triton Fused Kernels)
# =========================================================================
def test_dnn06_torch_compile_default(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(47)
    hp_fn = make_hp_gemm()

    class FusedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(nn.RMSNorm(2048), nn.Linear(2048, 4096), nn.SiLU(), nn.Linear(4096, 2048))
                for _ in range(4)
            ])

        def forward(self, x):
            for blk in self.layers:
                x = x + blk(x)
            return x

    model = FusedMLP().cuda().eval().to(torch.bfloat16)
    compiled = torch.compile(model, backend="inductor", mode="default")
    x = torch.randn(64, 256, 2048, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        # Compile & warmup outside preemption
        with torch.cuda.stream(lp_stream):
            _ = compiled(x)
        lp_stream.synchronize()

        with torch.cuda.stream(lp_stream):
            ref = compiled(x).clone()
        lp_stream.synchronize()

        got, susp, _ = run_on_main_with_bg_preempt(xs, xq, lp_stream, hp_stream, lambda: compiled(x),
                                                   delay_ms=2.0, num_preempts=3, hp_fn=hp_fn)

    max_err = (got.float() - ref.float()).abs().max().item()
    ok = len(susp) >= 1 and max_err == 0.0
    print(f"[DNN-06 torch.compile(inductor, default)] {'PASS' if ok else 'FAIL'} | preempts={len(susp)} "
          f"susp={[round(v, 3) for v in susp]} max_err={max_err:.3e}", flush=True)
    return ok


# =========================================================================
# DNN-07: torch.compile(backend="inductor", mode="reduce-overhead") (Inductor + CUDA Graph)
# =========================================================================
def test_dnn07_torch_compile_cudagraph(xs, lp_stream, hp_stream, xq):
    torch.manual_seed(48)
    hp_fn = make_hp_gemm()

    class GraphMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(2048, 4096, bias=False)
            self.w2 = nn.Linear(4096, 2048, bias=False)

        def forward(self, x):
            for _ in range(4):
                x = x + self.w2(F.silu(self.w1(x)))
            return x

    model = GraphMLP().cuda().eval().to(torch.bfloat16)
    compiled = torch.compile(model, backend="inductor", mode="reduce-overhead")
    x = torch.randn(32, 256, 2048, device="cuda", dtype=torch.bfloat16)

    def run_3x():
        y = x
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            y = compiled(y)
        return y.clone()

    with torch.no_grad():
        # Warmup twice to record & replay CUDA Graph cleanly
        with torch.cuda.stream(lp_stream):
            torch.compiler.cudagraph_mark_step_begin()
            _ = compiled(x)
            torch.compiler.cudagraph_mark_step_begin()
            _ = compiled(x)
        lp_stream.synchronize()

        with torch.cuda.stream(lp_stream):
            ref3 = run_3x()
        lp_stream.synchronize()

        got3, susp, _ = run_on_main_with_bg_preempt(xs, xq, lp_stream, hp_stream, run_3x,
                                                    delay_ms=2.0, num_preempts=2, hp_fn=hp_fn)

    max_err = (got3.float() - ref3.float()).abs().max().item()
    ok = max_err == 0.0
    print(f"[DNN-07 torch.compile(reduce-overhead/CUDAGraph)] {'PASS' if ok else 'FAIL'} | preempts={len(susp)} "
          f"susp={[round(v, 3) for v in susp]} max_err={max_err:.3e}", flush=True)
    return ok


# =========================================================================
# DNN-08: cuDNN Multi-Layer LSTM (RNN) + HuggingFace diffusers UNet2DModel (10-step Denoising)
# =========================================================================
def test_dnn08_lstm_and_diffusion(xs, lp_stream, hp_stream, xq):
    from diffusers import UNet2DModel

    torch.manual_seed(49)
    hp_fn = make_hp_gemm()

    # Part A: cuDNN Fused 4-layer Bidirectional LSTM
    lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=4, batch_first=True, bidirectional=True).cuda().eval().half()
    x_seq = torch.randn(32, 256, 512, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        with torch.cuda.stream(lp_stream):
            _ = lstm(x_seq)
        lp_stream.synchronize()
        with torch.cuda.stream(lp_stream):
            ref_lstm, _ = lstm(x_seq)
            ref_lstm = ref_lstm.clone()
        lp_stream.synchronize()
        box_l = {}
        susp_l, _ = run_with_mid_preempt(xs, xq, lp_stream, hp_stream,
                                         lambda: box_l.__setitem__("y", lstm(x_seq)[0]),
                                         delay_ms=2.0, num_preempts=3, hp_fn=hp_fn)
    err_l = (box_l["y"].float() - ref_lstm.float()).abs().max().item()

    # Part B: diffusers UNet2DModel 10-step non-idempotent iterative denoising
    unet = UNet2DModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).cuda().eval().to(torch.bfloat16)
    latent0 = torch.randn(8, 4, 64, 64, device="cuda", dtype=torch.bfloat16)

    def run_denoise():
        lat = latent0.clone()
        for t_step in range(10, 0, -1):
            eps = unet(lat, t_step).sample
            lat = lat - 0.05 * eps  # sequential non-idempotent denoising update
        return lat

    with torch.no_grad():
        with torch.cuda.stream(lp_stream):
            _ = run_denoise()
        lp_stream.synchronize()
        with torch.cuda.stream(lp_stream):
            ref_d = run_denoise().clone()
        lp_stream.synchronize()
        box_d = {}
        susp_d, _ = run_with_mid_preempt(xs, xq, lp_stream, hp_stream,
                                         lambda: box_d.__setitem__("y", run_denoise()),
                                         delay_ms=3.0, num_preempts=4, hp_fn=hp_fn)
    err_d = (box_d["y"].float() - ref_d.float()).abs().max().item()

    ok = len(susp_l) >= 1 and len(susp_d) >= 1 and err_l == 0.0 and err_d == 0.0
    print(f"[DNN-08 cuDNN-LSTM + Diffusers-UNet2D 10-Step] {'PASS' if ok else 'FAIL'} | "
          f"lstm_p={len(susp_l)} susp={[round(v, 3) for v in susp_l]} err={err_l:.3e} | "
          f"diff_p={len(susp_d)} susp={[round(v, 3) for v in susp_d]} err={err_d:.3e}", flush=True)
    return ok


def main():
    target = sys.argv[1] if len(sys.argv) >= 2 else "all"
    xs = XSchedBridge()
    lp_stream = torch.cuda.Stream()
    hp_stream = torch.cuda.Stream(priority=-1)
    lvl = int(os.environ.get("XSCHED_TEST_LEVEL", str(XSchedBridge.kPreemptLevelInterrupt)))
    hwq, xq = xs.create_xqueue(lp_stream, lvl)

    tests = [
        ("dnn01", test_dnn01_vit_b16),
        ("dnn02", test_dnn02_swin_convnext),
        ("dnn03", test_dnn03_seg_det_ops),
        ("dnn04", test_dnn04_amp_fused_adamw),
        ("dnn05", test_dnn05_long_horizon_convergence),
        ("dnn06", test_dnn06_torch_compile_default),
        ("dnn07", test_dnn07_torch_compile_cudagraph),
        ("dnn08", test_dnn08_lstm_and_diffusion),
    ]

    all_ok = True
    try:
        for name, fn in tests:
            if target == "all" or target == name:
                if not fn(xs, lp_stream, hp_stream, xq):
                    all_ok = False
    finally:
        xs.destroy(hwq, xq)

    if all_ok and target == "all":
        print("PHASE3_DNN_PROD_ALL_PASS", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
