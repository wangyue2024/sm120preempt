#!/usr/bin/env bash
# ==============================================================================
# Blackwell SM120 (RTX 5060 / 5090) Level 3 Hardware Preemption All-in-One Suite
# ==============================================================================
set -euo pipefail

# Ensure running with root privileges (required for MMIO sysfs resource0 mapping)
if [ "$(id -u)" -ne 0 ]; then
    echo "[*] Elevating to root privileges..."
    exec sudo -E "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-detect CUDA bin directory if nvcc is not in PATH
if ! command -v nvcc &>/dev/null; then
    for cand in /usr/local/cuda/bin /usr/local/cuda-12.8/bin /usr/local/cuda-12.9/bin /usr/local/cuda-13.0/bin /home/wy/cuda-12.9/bin; do
        if [ -x "$cand/nvcc" ]; then
            export PATH="$cand:$PATH"
            break
        fi
    done
fi

echo "========================================================================"
echo " [Setup] Auto-Detecting NVIDIA GPU PCI Bus ID & BAR0 Resource..."
echo "========================================================================"

DETECTED_BAR0=""
if [ -n "${BAR0_PATH:-}" ] && [ -e "$BAR0_PATH" ]; then
    DETECTED_BAR0="$BAR0_PATH"
elif [ -e "/sys/bus/pci/devices/0000:01:00.0/resource0" ]; then
    DETECTED_BAR0="/sys/bus/pci/devices/0000:01:00.0/resource0"
else
    # Scan /sys/bus/pci/devices for NVIDIA GPU (Vendor 0x10de, Class 0x03*)
    for dev in /sys/bus/pci/devices/*; do
        if [ -f "$dev/vendor" ]; then
            ven="$(cat "$dev/vendor" 2>/dev/null || true)"
            if [[ "$ven" =~ 0x10de|10de ]]; then
                cls="$(cat "$dev/class" 2>/dev/null || true)"
                if [[ "$cls" =~ ^0x03 ]]; then
                    if [ -e "$dev/resource0" ]; then
                        DETECTED_BAR0="$dev/resource0"
                        break
                    fi
                fi
            fi
        fi
    done
fi

if [ -z "$DETECTED_BAR0" ]; then
    echo "[-] WARNING: Could not auto-detect NVIDIA BAR0 resource. Defaulting to 0000:01:00.0/resource0"
    DETECTED_BAR0="/sys/bus/pci/devices/0000:01:00.0/resource0"
fi

echo "[+] Target BAR0 Resource : $DETECTED_BAR0"
if [ -e "$DETECTED_BAR0" ] && [ ! -w "$DETECTED_BAR0" ]; then
    echo "[-] WARNING: $DETECTED_BAR0 is not writable!"
    if grep -q "sysfs.*ro," /proc/mounts 2>/dev/null; then
        echo "[-] ERROR: /sys is mounted READ-ONLY (ro). In Docker, make sure to run with '--privileged'!"
    fi
fi
export BAR0_PATH="$DETECTED_BAR0"
export XSCHED_SM120_BAR0_RESOURCE="$DETECTED_BAR0"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:${LD_LIBRARY_PATH:-}"

echo ""
echo "========================================================================"
echo " [1/5] Building all SM120 Preemption Tools & Benchmarks..."
echo "========================================================================"
make all

echo ""
echo "========================================================================"
echo " [2/5] Running Full-Chip Broadcast Latency Probe (trap_latency_test 6)"
echo "       Testing BAR0 0x00424684 full-chip broadcast pulse across all SMs..."
echo "========================================================================"
./trap_latency_test 6

echo ""
echo "========================================================================"
echo " [3/5] Running Standalone 3-Granularity Fine-Grained Preemption Suite"
echo "       (stream_preempt_test: Mode 2 Pause/Resume, Mode 3 Per-CTA, Mode 1 Early-Exit)"
echo "========================================================================"
./stream_preempt_test

echo ""
echo "========================================================================"
echo " [4/5] Running XSched libhalcuda.so C++ API Integration Suite"
echo "       (hal_fine_grained_test: TrapSm120Context Fine-Grained Methods)"
echo "========================================================================"
./hal_fine_grained_test 2>/dev/null

echo ""
echo "========================================================================"
echo " [5/5] Running Full-Chip 30-SM Rigorous Contention & Overhead Benchmark"
echo "       (preempt_rigorous_bench: BENCH 1..4, 100% Slot Saturation & 50-Cycle Stress)"
echo "========================================================================"
./preempt_rigorous_bench

echo ""
echo "========================================================================"
echo " [Bonus] Running Detailed Commented Preemption Suite"
echo "========================================================================"
./stream_preempt_test_commented

echo ""
echo "========================================================================"
echo " >>> ALL FINE-GRAINED SM120 PREEMPTION TESTS PASSED SUCCESSFULLY! <<<"
echo "========================================================================"
