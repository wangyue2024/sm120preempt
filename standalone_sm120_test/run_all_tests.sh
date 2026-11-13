#!/usr/bin/env bash
# ==============================================================================
# Blackwell SM120 (RTX 5060 / RTX 5090 & Other GPUs)
# Level 3 Hardware Preemption & Universal Trap-Frame All-in-One Integration Suite
# ==============================================================================
set -euo pipefail

# Ensure running with root privileges (required for MMIO sysfs resource0 / /dev/mem mapping)
if [ "$(id -u)" -ne 0 ]; then
    echo "[*] Elevating to root privileges (required for BAR0 MMIO 0x00424684 trap broadcast)..."
    exec sudo -E "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-detect CUDA bin directory if nvcc is not in PATH
if ! command -v nvcc &>/dev/null; then
    for cand in /home/wy/cuda-12.9/bin /usr/local/cuda/bin /usr/local/cuda-12.9/bin /usr/local/cuda-12.8/bin /usr/local/cuda-13.0/bin; do
        if [ -x "$cand/nvcc" ]; then
            export PATH="$cand:$PATH"
            break
        fi
    done
fi

echo "========================================================================"
echo " [Setup] Auto-Detecting NVIDIA GPU PCI Bus ID & BAR0 Resource..."
echo "========================================================================"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,pci.bus_id,compute_cap,driver_version --format=csv
fi

DETECTED_BAR0=""
if [ -n "${BAR0_PATH:-}" ] && [ -e "$BAR0_PATH" ]; then
    DETECTED_BAR0="$BAR0_PATH"
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
echo " [0/7] Building All Preemption Tools, Probes & Benchmarks..."
echo "========================================================================"
make all

echo ""
echo "========================================================================"
echo " [1/7] Running GPU Architecture & 4-Stage Code Cave Compatibility Probe"
echo "       (cave_probe: Verifies .text.traphandler & Caves 0..3 via libcuxtra)"
echo "========================================================================"
./cave_probe

echo ""
echo "========================================================================"
echo " [2/7] Running Full-Chip Broadcast Latency Probe (trap_latency_test 6)"
echo "       Testing BAR0 0x00424684 full-chip broadcast pulse across all SMs..."
echo "========================================================================"
./trap_latency_test 6

echo ""
echo "========================================================================"
echo " [3/7] Running Standalone 3-Granularity Fine-Grained Preemption Suite"
echo "       (stream_preempt_test: Mode 2 Pause/Resume, Mode 3 Per-CTA, Mode 1 Early-Exit)"
echo "========================================================================"
./stream_preempt_test

echo ""
echo "========================================================================"
echo " [4/7] Running XSched libhalcuda.so C++ API Integration Suite"
echo "       (hal_fine_grained_test: TrapSm120Context Modes 1-3 API)"
echo "========================================================================"
./hal_fine_grained_test

echo ""
echo "========================================================================"
echo " [5/7] Running Full-Chip Multi-SM Contention & 100% Slot Saturation Suite"
echo "       (preempt_rigorous_bench: BENCH 1..4, 100% Occupancy & 50-Cycle Stress)"
echo "========================================================================"
./preempt_rigorous_bench

echo ""
echo "========================================================================"
echo " [6/7] Running Mode 4 Spatial Evict & Universal Mode 5 Zero-Source-Mod"
echo "       Hardware Trap-Frame Suite (checkpoint_bench all 5)"
echo "       - Mode 4: registers (144 regs), shared (64 KB), warps (6 CTA/SM), sgemm"
echo "       - Mode 5: trapframe, trapframe-univ-1w, trapframe-univ-2w (Multi-Wave"
echo "         3-State Restore), trapframe-univ-hireg (255 regs/thread via 3 Caves)"
echo "========================================================================"
./checkpoint_bench all 5

echo ""
echo "========================================================================"
echo " [7/7] Running Cross-Context Hardware CILP Preemption & Resume Benchmark"
echo "       (sm120_l3_resumable_bench: RM 0x20801210 CILP + 0x83de0317)"
echo "========================================================================"
./sm120_l3_resumable_bench

echo ""
echo "========================================================================"
echo " >>> ALL 7 STAGES OF THE SM120 PREEMPTION SUITE PASSED SUCCESSFULLY! <<<"
echo "========================================================================"
