#!/usr/bin/env bash
# Build the native SM120 regression binaries into $OUT (no root needed).
# Usage: bash build_regress_bins.sh <env_file>   (see regress_env.example.sh)
set -euo pipefail
source "$1"
NVCC="${NVCC:-$CUDA_HOME_LOCAL/bin/nvcc}"; CXX="${CXX:-/usr/bin/g++-13}"
mkdir -p "$OUT/nvlibs"
NV_SITE="$("$PY" -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))' 2>/dev/null || true)"
INC=""; LIBP=""
if [[ -n "$NV_SITE" && -d "$NV_SITE/cublas/include" ]]; then
  for f in "$NV_SITE"/*/lib/lib*.so*; do
    [[ -e "$f" ]] || continue; b="$(basename "$f")"; ln -sf "$f" "$OUT/nvlibs/$b"; ln -sf "$f" "$OUT/nvlibs/${b%%.so.*}.so"
  done
  INC="-I$NV_SITE/cublas/include -I$NV_SITE/cufft/include -I$NV_SITE/curand/include -I$NV_SITE/cusolver/include -I$NV_SITE/cusparse/include"
  LIBP="-L$OUT/nvlibs -Xlinker -rpath-link=$OUT/nvlibs"
fi
X="$XSCHED"
B() { "$NVCC" -ccbin "$CXX" -arch=sm_120 -O3 -I"$X/include" $INC "$EVAL/$1.cu" -o "$OUT/$1" -L"$X/lib" $LIBP "${@:2}"; }
B test_libs_and_features -lpreempt -lhalcuda -lcublas -lcublasLt -lcufft -lcusparse -lcudart
B test_real_cuda_libs_p3 -lpreempt -lhalcuda -lcublas -lcublasLt -lcufft -lcurand -lcusolver -lcusparse -lcudart
B test_partial_exit_p3 -lpreempt -lhalcuda -lcuda
B test_convbar_p3 -lpreempt -lhalcuda -lcuda
echo "BUILD_REGRESS_BINS_OK -> $OUT"
