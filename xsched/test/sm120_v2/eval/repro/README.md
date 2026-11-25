# Phase 3 (2026-11-25) reproduction & probe tools

| File | Purpose |
|---|---|
| `cast_repro_def14.py` | Minimal DEF-14 repro: repeated `bool/int8 -> int64` casts on a Level-3 XQueue with 3 mid-job suspend/resume. `python -I cast_repro_def14.py bool2i64 0 10` |
| `sort_repro.py` | `torch.sort` through a Level-3 XQueue without preemption (control: passes) |
| `lv3_drift_obs1.py` | OBS-1: in-process A/B of an in-place op (native vs Lv3 queue), finds the first divergent call |
| `sr_regalloc_probe_kernels.cu` + `patch_s2r_regalloc.py` + `sr_regalloc_probe_host.cu` | DEF-20: measure the hardware `SR_REGALLOC` value vs `CU_FUNC_ATTRIBUTE_NUM_REGS` (plain context and under a Level-3 XQueue) |
| `smem_size_probe.cu` | Measures `%total_smem_size` for given dynamic SMEM (ruled-out SMEM accounting hypothesis) |
| `../p3_guarded_run.sh` | Guarded root test runner for the shared RTX 5080 (GPU-idle check before every test) |

SR_REGALLOC probe:
```bash
nvcc -arch=sm_120 -cubin -O3 sr_regalloc_probe_kernels.cu -o probe.cubin
python3 patch_s2r_regalloc.py probe.cubin probe_patched.cubin
nvcc -arch=sm_120 -I$XSCHED/include sr_regalloc_probe_host.cu -o probe_host -L$XSCHED/lib -lpreempt -lhalcuda -lcuda
./probe_host 0                      # plain context (reads probe_patched.cubin from cwd)
sudo env LD_LIBRARY_PATH=$XSCHED/lib:... ./probe_host 3   # under a Level-3 XQueue
```
