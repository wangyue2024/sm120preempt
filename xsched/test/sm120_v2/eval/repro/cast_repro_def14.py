"""Minimal Lv3 repro: repeated elementwise cast kernel on LP XQueue + mid-job suspend/resume."""
import ctypes, os, sys, threading, time, torch
kind, use_hp, n_trials = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3])
pre = ctypes.CDLL("libpreempt.so", mode=ctypes.RTLD_GLOBAL); hal = ctypes.CDLL("libhalcuda.so", mode=ctypes.RTLD_GLOBAL)
lp, hp = torch.cuda.Stream(), torch.cuda.Stream(priority=-1)
N = 8192 * 4096
with torch.cuda.stream(lp):
    if kind == "bool2i64": x = torch.rand(N, device="cuda") > 0.5; f = lambda: x.to(torch.int64)
    elif kind == "f2bf16": x = torch.randn(N, device="cuda"); f = lambda: x.to(torch.bfloat16)
    elif kind == "i8toi64": x = torch.randint(-100, 100, (N,), device="cuda", dtype=torch.int8); f = lambda: x.to(torch.int64)
    elif kind == "add_f32": x = torch.randn(N, device="cuda"); f = lambda: x + 1.0
    ref = f().clone()
lp.synchronize()
a, b = torch.randn(512, 1024, device="cuda"), torch.randn(1024, 1024, device="cuda"); torch.cuda.synchronize()
hwq, xq = ctypes.c_uint64(), ctypes.c_uint64()
assert hal.CudaQueueCreate(ctypes.byref(hwq), ctypes.c_void_p(lp.cuda_stream)) == 0
assert pre.XQueueCreate(ctypes.byref(xq), hwq, 3, 0) == 0
bad = 0
for t in range(n_trials):
    outs = []
    def preemptor():
        for k in range(3):
            time.sleep(0.004)
            pre.XQueueSuspend(xq, 1)
            if use_hp:
                with torch.cuda.stream(hp): torch.mm(a, b)
                hp.synchronize()
            pre.XQueueResume(xq, 0)
    th = threading.Thread(target=preemptor); th.start()
    with torch.cuda.stream(lp):
        for i in range(40): outs.append(f())
    lp.synchronize(); th.join()
    bad += sum(0 if torch.equal(o, ref) else 1 for o in outs)
print(f"RESULT kind={kind} hp={use_hp} trials={n_trials} bad_outputs={bad}", flush=True)
