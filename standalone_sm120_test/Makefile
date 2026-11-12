# Portable Standalone Makefile for Blackwell SM120 (RTX 5060 / 5090) Preemption Suite

CUDA_PATH ?= $(shell for cand in "$$CUDA_PATH" "$$(which nvcc 2>/dev/null | sed 's|/bin/nvcc||')" /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-13.0 /home/wy/cuda-12.9; do \
                  if [ -x "$$cand/bin/nvcc" ]; then \
                      if "$$cand/bin/nvcc" --help 2>&1 | grep -q "sm_120"; then \
                          echo "$$cand"; break; \
                      fi; \
                  fi; \
              done)
NVCC      := $(CUDA_PATH)/bin/nvcc
CXX       := $(shell which g++-13 2>/dev/null || which g++)
CC        := $(shell which gcc-13 2>/dev/null || which gcc)

NVCCFLAGS := -O2 -std=c++17 -ccbin $(CXX) -arch=sm_120 -I. -I./include
CUXTRA_A  := ./lib/libcuxtra_linux_x86_64.a

# Set rpath to $ORIGIN/lib so binaries can find libhalcuda.so, libpreempt.so, libshimcuda.so regardless of where the folder is moved!
HAL_LIB   := -L./lib \
             -Xlinker -rpath,'$$ORIGIN/lib' \
             -Xlinker -rpath,'$$ORIGIN/../lib' \
             -lhalcuda -lpreempt -lcuda

TARGETS := bar0tool spin smstop trapprobe trap_latency_test stream_preempt_test stream_preempt_test_commented preempt_rigorous_bench hal_fine_grained_test

all: $(TARGETS)

bar0tool: bar0tool.c bar0.h
	$(CC) -O2 -Wall bar0tool.c -o bar0tool

spin: spin.cu
	$(NVCC) $(NVCCFLAGS) spin.cu -o spin

smstop: smstop.cu bar0.h
	$(NVCC) $(NVCCFLAGS) smstop.cu -o smstop

trapprobe: trapprobe.cu bar0.h livepatch.h sass120.h arena.h
	$(NVCC) $(NVCCFLAGS) trapprobe.cu $(CUXTRA_A) -lcuda -o trapprobe

trap_latency_test: trap_latency_test.cu bar0.h sass120.h arena.h
	$(NVCC) $(NVCCFLAGS) trap_latency_test.cu $(CUXTRA_A) -lcuda -o trap_latency_test

stream_preempt_test: stream_preempt_test.cu bar0.h sass120.h arena.h
	$(NVCC) $(NVCCFLAGS) stream_preempt_test.cu $(CUXTRA_A) -lcuda -o stream_preempt_test

stream_preempt_test_commented: stream_preempt_test_commented.cu bar0.h sass120.h arena.h
	$(NVCC) $(NVCCFLAGS) stream_preempt_test_commented.cu $(CUXTRA_A) -lcuda -o stream_preempt_test_commented

preempt_rigorous_bench: preempt_rigorous_bench.cu bar0.h sass120.h livepatch.h arena.h
	$(NVCC) $(NVCCFLAGS) preempt_rigorous_bench.cu $(CUXTRA_A) -lcuda -o preempt_rigorous_bench

hal_fine_grained_test: hal_fine_grained_test.cu
	$(NVCC) $(NVCCFLAGS) hal_fine_grained_test.cu $(HAL_LIB) -o hal_fine_grained_test

clean:
	rm -f $(TARGETS)
