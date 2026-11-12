NVCC      := /home/wy/cuda-12.9/bin/nvcc
CXX       := g++-13
CC        := gcc-13
NVCCFLAGS := -O2 -std=c++17 -ccbin $(CXX) -arch=sm_120
PROBE_INC := -I../xsched/test/sm120_l3_trap_v2/probe -I../xsched/3rdparty/cuxtra/include
CUXTRA_A  := ../xsched/3rdparty/cuxtra/lib/libcuxtra_linux_x86_64.a
HAL_INC   := -I../xsched/platforms/cuda/hal/include -I../xsched/preempt/include -I../xsched/utils/include
HAL_LIB   := -L../xsched/output/lib \
             -Xlinker --disable-new-dtags \
             -Xlinker -rpath,/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/output/lib \
             -lhalcuda -lcuda

TARGETS := bar0tool spin smstop trapprobe trap_latency_test stream_preempt_test preempt_rigorous_bench hal_fine_grained_test
CHECKPOINT_BENCH := spill/checkpoint_bench
CHECKPOINT_BUILD_DIR ?= ../xsched/build_sm120
CHECKPOINT_HAL_DIR := $(CHECKPOINT_BUILD_DIR)/platforms/cuda
CHECKPOINT_PREEMPT_DIR := $(CHECKPOINT_BUILD_DIR)/preempt
CHECKPOINT_HAL_LIB := -L$(CHECKPOINT_HAL_DIR) -L$(CHECKPOINT_PREEMPT_DIR) -L../xsched/output/lib \
                      -Xlinker --disable-new-dtags \
                      -Xlinker -rpath,$(abspath $(CHECKPOINT_HAL_DIR)):$(abspath $(CHECKPOINT_PREEMPT_DIR)):$(abspath ../xsched/output/lib) \
                      -lhalcuda -lcuda -Xlinker --no-as-needed,-lpreempt,--as-needed

all: $(TARGETS)

bar0tool: bar0tool.c bar0.h
	$(CC) -O2 -Wall bar0tool.c -o bar0tool

spin: spin.cu
	$(NVCC) $(NVCCFLAGS) spin.cu -o spin

smstop: smstop.cu bar0.h
	$(NVCC) $(NVCCFLAGS) smstop.cu -o smstop

trapprobe: trapprobe.cu bar0.h livepatch.h sass120.h
	$(NVCC) $(NVCCFLAGS) $(PROBE_INC) trapprobe.cu $(CUXTRA_A) -lcuda -o trapprobe

trap_latency_test: trap_latency_test.cu bar0.h sass120.h
	$(NVCC) $(NVCCFLAGS) $(PROBE_INC) trap_latency_test.cu $(CUXTRA_A) -lcuda -o trap_latency_test

stream_preempt_test: stream_preempt_test.cu bar0.h sass120.h
	$(NVCC) $(NVCCFLAGS) $(PROBE_INC) stream_preempt_test.cu $(CUXTRA_A) -lcuda -o stream_preempt_test

preempt_rigorous_bench: preempt_rigorous_bench.cu bar0.h sass120.h
	$(NVCC) $(NVCCFLAGS) $(PROBE_INC) preempt_rigorous_bench.cu $(CUXTRA_A) -lcuda -o preempt_rigorous_bench

$(CHECKPOINT_BENCH): spill/checkpoint_bench.cu bar0.h Makefile $(CHECKPOINT_HAL_DIR)/libhalcuda.so $(CHECKPOINT_HAL_DIR)/libshimcuda.so $(CHECKPOINT_PREEMPT_DIR)/libpreempt.so
	$(NVCC) $(NVCCFLAGS) $(HAL_INC) -Xptxas=-v $< $(CHECKPOINT_HAL_LIB) -o $@

hal_fine_grained_test: hal_fine_grained_test.cu
	$(NVCC) $(NVCCFLAGS) $(HAL_INC) hal_fine_grained_test.cu $(HAL_LIB) -o hal_fine_grained_test

clean:
	rm -f $(TARGETS) $(CHECKPOINT_BENCH)
