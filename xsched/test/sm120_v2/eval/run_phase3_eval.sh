#!/usr/bin/env bash
# Phase 3 Production AI & Real Computing Libraries Verification Runner (RTX 5080 / RTX 5060)
# Covers:
#   - Suite 4: Extended Closed-Source CUDA Libraries (cuSOLVER LU/Cholesky, cuRAND Stateful,
#              cuFFT 2D R2C+C2R, CUB Decoupled Look-Back Scan/Reduce/Select, cuBLASLt FP8 E4M3)
#   - Suite 5: Production DNNs & Training (ViT-B/16, Swin-B, ConvNeXt-NHWC, DeepLabV3+RoIAlign+NMS,
#              AMP-BF16+GradAccum+FusedAdamW, 20-step Convergence, torch.compile default & CUDAGraph,
#              cuDNN 4-layer BiLSTM + Diffusers UNet2D 10-step iterative denoising)
#   - Suite 6: Production LLMs (HuggingFace Qwen2ForCausalLM Prefill+KVCache, 32-Token Autoregressive
#              Decode, PEFT LoRA SFT, Quantized FP8/INT8/Triton-W4A16, Triton PagedAttention + Dynamic
#              Buckets, Prefill+Decode & LoRA+Decode Co-Location QoS, Multi-Stream 2xLP+1xHP)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XSCHED_PREFIX="${XSCHED_PREFIX:-/tmp/xsched_install_5060}"
NVCC="${NVCC:-/home/wy/cuda-12.9/bin/nvcc}"
CXX="${CXX:-/usr/bin/g++-13}"
CUDA_LIB="${CUDA_LIB:-/home/wy/cuda-12.9/lib64}"
PY_BIN="${PY_BIN:-/home/wy/.venv_sm120/bin/python}"
OUT_DIR="${OUT_DIR:-/tmp/sm120_p3_bin}"
ALLOW_DESKTOP_PROCS="${ALLOW_DESKTOP_PROCS:-0}"
mkdir -p "$OUT_DIR"

# Safety check: never fire BAR0 while another user's compute workload is active
if [[ "$ALLOW_DESKTOP_PROCS" != "1" ]]; then
    ACTIVE="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "$ACTIVE" ]]; then
        echo "[SAFETY ABORT] Active GPU compute processes detected:"
        echo "$ACTIVE"
        echo "Refusing to trigger BAR0 MMIO while another GPU job is running."
        exit 1
    fi
fi

export CUDA_MPS_PIPE_DIRECTORY="/tmp/no_mps_${USER:-root}"

EXTRA_INC=""
EXTRA_LIB=""
EXTRA_RPATH=""
NV_SITE="$("$PY_BIN" -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))' 2>/dev/null || true)"
if [[ -n "$NV_SITE" && -d "$NV_SITE/cusolver/include" ]]; then
    mkdir -p "$OUT_DIR/nvlibs"
    for f in "$NV_SITE"/*/lib/lib*.so*; do
        [[ -e "$f" ]] || continue
        base="$(basename "$f")"
        ln -sf "$f" "$OUT_DIR/nvlibs/$base"
        short="${base%%.so.*}.so"
        ln -sf "$f" "$OUT_DIR/nvlibs/$short"
    done
    EXTRA_INC="-I$NV_SITE/cublas/include -I$NV_SITE/cufft/include -I$NV_SITE/curand/include -I$NV_SITE/cusolver/include -I$NV_SITE/cusparse/include"
    EXTRA_LIB="-L$OUT_DIR/nvlibs"
    EXTRA_RPATH="-Xlinker -rpath-link=$OUT_DIR/nvlibs"
fi

echo "[1/4] Building Suite 4 (test_real_cuda_libs_p3) ..."
# shellcheck disable=SC2086
"$NVCC" -ccbin "$CXX" -arch=sm_120 -O2 -I"$XSCHED_PREFIX/include" $EXTRA_INC \
    "$SCRIPT_DIR/test_real_cuda_libs_p3.cu" -o "$OUT_DIR/test_real_cuda_libs_p3" \
    -L"$XSCHED_PREFIX/lib" $EXTRA_LIB $EXTRA_RPATH \
    -lpreempt -lhalcuda -lcublas -lcublasLt -lcufft -lcurand -lcusolver -lcusparse -lcudart

export LD_LIBRARY_PATH="$XSCHED_PREFIX/lib:${OUT_DIR}/nvlibs:$CUDA_LIB:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

echo "[2/4] Running Suite 4: Extended Closed-Source CUDA Libraries (LIB-09..14, Mode 5, NP=2) ..."
for c in LIB-09 LIB-10 LIB-11 LIB-12 LIB-13 LIB-14; do
    TEST_NP=2 timeout 45 "$OUT_DIR/test_real_cuda_libs_p3" "$c" 2>&1 | grep -E "^\[LIB-|WARN|ERRO|FAIL" || true
done

echo "[3/4] Running Suite 5: Real Production DNNs, Fused Training, torch.compile, LSTM & Diffusion ..."
for c in dnn01 dnn02 dnn03 dnn04 dnn05 dnn06 dnn07 dnn08; do
    timeout 120 "$PY_BIN" "$SCRIPT_DIR/test_real_dnn_prod_p3.py" "$c" 2>&1 | grep -E "^\[DNN-|WARN|ERRO|Traceback|FAIL" || true
done

echo "[4/4] Running Suite 6: Real Production LLMs (HuggingFace Qwen2, PEFT LoRA, FP8/INT8/W4A16, PagedAttn, Co-Location QoS) ..."
for c in llm01 llm02 llm03 llm04 llm05 llm08 llm06:native llm06:lv1 llm06:m5 llm07:native llm07:lv1 llm07:m5; do
    timeout 150 "$PY_BIN" "$SCRIPT_DIR/test_real_llm_prod_p3.py" "$c" 2>&1 | grep -E "^\[LLM-|WARN|ERRO|Traceback|FAIL" || true
done

if [[ "${ENABLE_REAL_WEIGHTS:-0}" == "1" ]]; then
    echo "[Bonus] Running LLM-09 Real Pretrained Weights ..."
    timeout 180 "$PY_BIN" "$SCRIPT_DIR/test_real_llm_prod_p3.py" llm09 2>&1 | grep -E "^\[LLM-|WARN|ERRO|Traceback|FAIL" || true
fi

echo "PHASE3_ALL_EVAL_DONE"
