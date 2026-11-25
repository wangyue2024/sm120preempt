import ctypes, os, sys, torch
n = int(float(sys.argv[1])); stable = sys.argv[2] == "1"; dt = getattr(torch, sys.argv[3]); level = int(sys.argv[4])
pre = ctypes.CDLL("libpreempt.so", mode=ctypes.RTLD_GLOBAL); hal = ctypes.CDLL("libhalcuda.so", mode=ctypes.RTLD_GLOBAL)
s = torch.cuda.Stream()
x = (torch.randn(n, device="cuda") * 1000).to(dt); torch.cuda.synchronize()
ref_v, ref_i = torch.sort(x, stable=stable); torch.cuda.synchronize()
hwq, xq = ctypes.c_uint64(), ctypes.c_uint64()
assert hal.CudaQueueCreate(ctypes.byref(hwq), ctypes.c_void_p(s.cuda_stream)) == 0
assert pre.XQueueCreate(ctypes.byref(xq), hwq, level, 0) == 0
reps = int(os.environ.get("REPS", "1"))
with torch.cuda.stream(s):
    for _ in range(reps):
        v, i = torch.sort(x, stable=stable)
s.synchronize()
print(f"RESULT n={n} stable={stable} dt={dt} level={level} values_eq={torch.equal(v, ref_v)} idx_eq={torch.equal(i, ref_i)}", flush=True)
