#!/usr/bin/env bash
# Reproducible SM120 v2 Comprehensive Evaluation Runner (RTX 5060 / RTX 5080)
# Usage:
#   XSCHED_PREFIX=/tmp/xsched_install NVCC=/home/wangyue/.local/cuda-12.9/bin/nvcc \
#   CXX=/usr/bin/g++-13 CUDA_LIB=/home/wangyue/.local/cuda-12.9/lib64 \
#   PY_BIN=/path/to/python bash run_all_eval.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XSCHED_PREFIX="${XSCHED_PREFIX:-/tmp/xsched_install_5060}"
NVCC="${NVCC:-/home/wy/cuda-12.9/bin/nvcc}"
CXX="${CXX:-/usr/bin/g++-13}"
CUDA_LIB="${CUDA_LIB:-/home/wy/cuda-12.9/lib64}"
OUT_DIR="${OUT_DIR:-/tmp/sm120_eval_bin}"
mkdir -p "$OUT_DIR"

# Safety check: never fire BAR0 while another user's GPU compute process is active
ACTIVE="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
if [[ -n "$ACTIVE" ]]; then
    echo "[SAFETY ABORT] Active GPU compute processes detected:"
    echo "$ACTIVE"
    echo "Refusing to trigger BAR0 MMIO while another GPU job is running."
    exit 1
fi

export CUDA_MPS_PIPE_DIRECTORY="/tmp/no_mps_${USER:-root}"

echo "[1/4] Building evaluation binaries into $OUT_DIR ..."
"$NVCC" -ccbin "$CXX" -arch=sm_120 -O2 -I"$XSCHED_PREFIX/include" \
    "$SCRIPT_DIR/test_libs_and_features.cu" -o "$OUT_DIR/test_libs_and_features" \
    -L"$XSCHED_PREFIX/lib" -lpreempt -lhalcuda -lcublas -lcublasLt -lcufft -lcusparse -lcudart
"$NVCC" -ccbin "$CXX" -arch=sm_120 -O2 -I"$XSCHED_PREFIX/include" \
    "$SCRIPT_DIR/bench_dnn_llm_scheduling.cu" -o "$OUT_DIR/bench_dnn_llm_scheduling" \
    -L"$XSCHED_PREFIX/lib" -lpreempt -lhalcuda -lcublas -lcudart
"$NVCC" -ccbin "$CXX" -arch=sm_120 -O2 -I"$XSCHED_PREFIX/include" \
    "$SCRIPT_DIR/bench_heavy_qos.cu" -o "$OUT_DIR/bench_heavy_qos" \
    -L"$XSCHED_PREFIX/lib" -lpreempt -lhalcuda -lcudart
"$NVCC" -ccbin "$CXX" -arch=sm_120 -O2 -I"$XSCHED_PREFIX/include" \
    "$SCRIPT_DIR/bench_bar0_cross_ctx.cu" -o "$OUT_DIR/bench_bar0_cross_ctx" \
    -L"$XSCHED_PREFIX/lib" -lpreempt -lhalcuda -lcudart
"$NVCC" -ccbin "$CXX" -arch=sm_120 -O2 \
    "$SCRIPT_DIR/bystander_pure.cu" -o "$OUT_DIR/bystander_pure"

export LD_LIBRARY_PATH="$XSCHED_PREFIX/lib:$CUDA_LIB:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

echo "[2/4] Running Suite 1: 8 Closed-Source CUDA Libraries + 7 Blackwell HW Features (NP=2) ..."
for c in LIB-01 LIB-02 LIB-03 LIB-04 LIB-05 LIB-06 LIB-07 LIB-08 HW-01 HW-02 HW-03 HW-04 HW-05 HW-06 HW-07; do
    echo ">>> $c <<<"
    TEST_NP=2 timeout 30 "$OUT_DIR/test_libs_and_features" "$c" 2>&1 | grep -E "^\[|preempts=|WARN|ERRO" || true
done

echo "[3/4] Running Suite 2: LLM Serving + Non-Idempotent DNN Training + Frequency Sweep ..."
for m in native lv1 m1 m5; do
    if [[ "$m" == "m1" ]]; then
        XSCHED_SM120_FORCE_MODE1=1 timeout 30 "$OUT_DIR/bench_dnn_llm_scheduling" llm m1 15 20 2>&1 | grep -E "^\[LLM_SERVING|WARN|ERRO" || true
    else
        timeout 30 "$OUT_DIR/bench_dnn_llm_scheduling" llm "$m" 15 20 2>&1 | grep -E "^\[LLM_SERVING|WARN|ERRO" || true
    fi
done
for m in native lv1 m1 m5; do
    if [[ "$m" == "m1" ]]; then
        XSCHED_SM120_FORCE_MODE1=1 timeout 30 "$OUT_DIR/bench_heavy_qos" m1 25 15 90000 2>&1 | grep -E "^\[HEAVY_QOS|WARN|ERRO" || true
    else
        timeout 30 "$OUT_DIR/bench_heavy_qos" "$m" 25 15 90000 2>&1 | grep -E "^\[HEAVY_QOS|WARN|ERRO" || true
    fi
done

echo "[4/4] Running Cross-Context BAR0 Collateral Interference ..."
LD_LIBRARY_PATH="$CUDA_LIB:/usr/lib/x86_64-linux-gnu" timeout 15 "$OUT_DIR/bystander_pure"
timeout 15 "$OUT_DIR/bench_bar0_cross_ctx" preemptor > "$OUT_DIR/preemptor.log" 2>&1 &
PID_P=$!
sleep 0.25
LD_LIBRARY_PATH="$CUDA_LIB:/usr/lib/x86_64-linux-gnu" timeout 15 "$OUT_DIR/bystander_pure"
wait "$PID_P" || true

if [[ -n "${PY_BIN:-}" && -x "${PY_BIN:-}" ]]; then
    echo "[Bonus] Running Suite 3: Real PyTorch + cuDNN + FlashAttention + Triton ..."
    for c in pt01 pt02 pt03_math pt03_flash pt03_efficient pt03_cudnn pt04 qos:native qos:lv1 qos:m5; do
        timeout 45 "$PY_BIN" "$SCRIPT_DIR/test_pytorch_llm_dnn.py" "$c" 2>&1 | grep -E "^\[PT-|WARN|ERRO" || true
    done
fi
echo "ALL_EVAL_DONE"
