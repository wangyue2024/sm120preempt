# ==============================================================================
# Portable Standalone Makefile for Blackwell SM120 (RTX 5060 / 5090 & Other GPUs)
# Hardware Trap Preemption, Universal Trap-Frame Evict/Restore & Benchmark Suite
# ==============================================================================

CUDA_PATH ?= $(shell for cand in "$$CUDA_PATH" "$$(which nvcc 2>/dev/null | sed 's|/bin/nvcc||')" /home/wy/cuda-12.9 /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-13.0; do \
                  if [ -x "$$cand/bin/nvcc" ]; then \
                      if "$$cand/bin/nvcc" --help 2>&1 | grep -q "sm_120"; then \
                          echo "$$cand"; break; \
                      fi; \
                  fi; \
              done)
NVCC      := $(CUDA_PATH)/bin/nvcc
CXX       := $(shell which g++-13 2>/dev/null || which g++)
CC        := $(shell which gcc-13 2>/dev/null || which gcc)

# Auto-detect compute capability of active GPU (fallback to sm_120)
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
GPU_ARCH    ?= $(if $(DETECTED_CC),sm_$(DETECTED_CC),sm_120)

NVCCFLAGS := -O2 -std=c++17 -ccbin $(CXX) -arch=$(GPU_ARCH) -I. -I./include
CUXTRA_A  := ./lib/libcuxtra_linux_x86_64.a

# Use DT_RPATH (--disable-new-dtags) with $ORIGIN/lib so all binaries and their
# transitive shared library dependencies (libhalcuda.so -> libpreempt.so) resolve
# from ./lib automatically on any machine without needing LD_LIBRARY_PATH!
HAL_LIB   := -L./lib \
             -Xlinker --disable-new-dtags \
             -Xlinker -rpath,'$$ORIGIN/lib' \
             -Xlinker -rpath,'$$ORIGIN/../lib' \
             -lhalcuda -lpreempt -lcuda

TARGETS := cave_probe \
           checkpoint_bench \
           sm120_l3_resumable_bench \
           hal_fine_grained_test \
           preempt_rigorous_bench \
           stream_preempt_test \
           stream_preempt_test_commented \
           trap_latency_test \
           trapprobe \
           bar0tool \
           spin \
           smstop

.PHONY: all clean

all: $(TARGETS)

cave_probe: cave_probe.cu arena.h
	$(NVCC) $(NVCCFLAGS) cave_probe.cu $(CUXTRA_A) -lcuda -o cave_probe

checkpoint_bench: checkpoint_bench.cu bar0.h
	$(NVCC) $(NVCCFLAGS) checkpoint_bench.cu $(HAL_LIB) -o checkpoint_bench

sm120_l3_resumable_bench: sm120_l3_resumable_bench.cu
	$(NVCC) $(NVCCFLAGS) sm120_l3_resumable_bench.cu $(HAL_LIB) -o sm120_l3_resumable_bench

hal_fine_grained_test: hal_fine_grained_test.cu
	$(NVCC) $(NVCCFLAGS) hal_fine_grained_test.cu $(HAL_LIB) -o hal_fine_grained_test

preempt_rigorous_bench: preempt_rigorous_bench.cu bar0.h sass120.h livepatch.h arena.h
	$(NVCC) $(NVCCFLAGS) preempt_rigorous_bench.cu $(CUXTRA_A) -lcuda -o preempt_rigorous_bench

stream_preempt_test: stream_preempt_test.cu bar0.h sass120.h arena.h
	$(NVCC) $(NVCCFLAGS) stream_preempt_test.cu $(CUXTRA_A) -lcuda -o stream_preempt_test

stream_preempt_test_commented: stream_preempt_test_commented.cu bar0.h sass120.h arena.h
	$(NVCC) $(NVCCFLAGS) stream_preempt_test_commented.cu $(CUXTRA_A) -lcuda -o stream_preempt_test_commented

trap_latency_test: trap_latency_test.cu bar0.h sass120.h arena.h
	$(NVCC) $(NVCCFLAGS) trap_latency_test.cu $(CUXTRA_A) -lcuda -o trap_latency_test

trapprobe: trapprobe.cu bar0.h livepatch.h sass120.h arena.h
	$(NVCC) $(NVCCFLAGS) trapprobe.cu $(CUXTRA_A) -lcuda -o trapprobe

bar0tool: bar0tool.c bar0.h
	$(CC) -O2 -Wall bar0tool.c -o bar0tool

spin: spin.cu
	$(NVCC) $(NVCCFLAGS) spin.cu -o spin

smstop: smstop.cu bar0.h
	$(NVCC) $(NVCCFLAGS) smstop.cu -o smstop

clean:
	rm -f $(TARGETS)
