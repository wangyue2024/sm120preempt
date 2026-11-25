# Example environment for regress_p3.py / build_regress_bins.sh / p3_guarded_run.sh plans.
export XSCHED=/path/to/xsched_install            # cmake --install prefix
export OUT=/path/to/regress_bin                    # native test binaries
export EVAL=/path/to/xsched/test/sm120_v2/eval
export PY=/path/to/venv/bin/python                 # torch 2.11+cu128, transformers, peft, triton
export CUDA_HOME_LOCAL=/path/to/cuda-12.9
export LD_LIBRARY_PATH=$XSCHED/lib:$OUT/nvlibs:$CUDA_HOME_LOCAL/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no_mps_$(id -u)
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/xsched_regress_cache/triton TORCHINDUCTOR_CACHE_DIR=/tmp/xsched_regress_cache/inductor
export PYTHONDONTWRITEBYTECODE=1
