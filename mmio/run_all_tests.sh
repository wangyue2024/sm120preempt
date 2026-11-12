#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo " [1/5] Building all SM120 Fine-Grained Preemption Tools & Benchmarks..."
echo "========================================================================"
make all

echo ""
echo "========================================================================"
echo " [2/5] Running Full-Chip Broadcast Latency Probe (trap_latency_test 6)"
echo "       Testing BAR0 0x00424684 broadcast across all 30/30 SMs..."
echo "========================================================================"
sudo ./trap_latency_test 6

echo ""
echo "========================================================================"
echo " [3/5] Running Standalone 3-Granularity Fine-Grained Suite"
echo "       (stream_preempt_test: Mode 2 Pause/Resume, Mode 3 Per-CTA, Mode 1 Early-Exit)"
echo "========================================================================"
sudo ./stream_preempt_test

echo ""
echo "========================================================================"
echo " [4/5] Running XSched libhalcuda.so C++ API Integration Suite"
echo "       (hal_fine_grained_test: TrapSm120Context Fine-Grained Methods)"
echo "========================================================================"
sudo env LD_LIBRARY_PATH="$SCRIPT_DIR/../xsched/output/lib" ./hal_fine_grained_test 2>/dev/null

echo ""
echo "========================================================================"
echo " [5/5] Running Full-Chip 30-SM Rigorous Contention & Overhead Benchmark"
echo "       (preempt_rigorous_bench: BENCH 1..4, 100% Slot Saturation & 50-Cycle Stress)"
echo "========================================================================"
sudo ./preempt_rigorous_bench

echo ""
echo "========================================================================"
echo " ALL FINE-GRAINED SM120 PREEMPTION TESTS PASSED SUCCESSFULLY!"
echo "========================================================================"
