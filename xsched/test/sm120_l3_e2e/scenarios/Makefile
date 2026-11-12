NVCC ?= nvcc
NVCC_FLAGS ?= -O3 -arch=sm_120

TARGETS = workload_dense_gemm workload_3d_stencil workload_transformer_attention workload_atomic_accum workload_multi_tenant

all: $(TARGETS)

workload_dense_gemm: workload_dense_gemm.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

workload_3d_stencil: workload_3d_stencil.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

workload_transformer_attention: workload_transformer_attention.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

workload_atomic_accum: workload_atomic_accum.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

workload_multi_tenant: workload_multi_tenant.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
