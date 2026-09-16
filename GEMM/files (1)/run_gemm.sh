#!/usr/bin/env bash
# Build + run gemm_benchmark.cu (naive / SM-tiled / double-buffer / warp-optimized)
# Usage:
#   ./run_gemm.sh                  # default sizes: 512,1024,2048
#   ./run_gemm.sh 256,512,1024,2048
set -e

SRC="gemm_benchmark.cu"
BIN="gemm_benchmark"
mkdir -p results

ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
if [ -z "$ARCH" ]; then
    echo "Could not auto-detect GPU arch, defaulting to sm_75"
    ARCH="75"
fi
echo "Building for sm_${ARCH}..."
nvcc -O3 -arch=sm_${ARCH} "$SRC" -o "$BIN"

TS=$(date +%Y-%m-%d_%H%M%S)
LOG="results/run_gemm_${TS}.log"
echo "Running (sizes: ${1:-default})..."
./"$BIN" "$1" | tee "$LOG"

echo ""
echo "Log saved to: $LOG"
echo "CSV/MD saved alongside it in results/"
