#!/usr/bin/env bash
# Build + run gemm_wmma_benchmark.cu (WMMA tensor cores + cuBLAS reference)
# Requires sm_70+ (Volta or newer) for WMMA and -lcublas for cuBLAS.
# Usage:
#   ./run_wmma.sh                   # default sizes: 1024,2048,4096
#   ./run_wmma.sh 1024,2048,4096,8192
set -e

SRC="gemm_wmma_benchmark.cu"
BIN="gemm_wmma_benchmark"
mkdir -p results

ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
if [ -z "$ARCH" ]; then
    echo "Could not auto-detect GPU arch, defaulting to sm_75"
    ARCH="75"
fi
if [ "$ARCH" -lt 70 ]; then
    echo "WARNING: detected sm_${ARCH} — WMMA requires sm_70 (Volta) or newer. Build may fail."
fi
echo "Building for sm_${ARCH}..."
nvcc -O3 -arch=sm_${ARCH} "$SRC" -lcublas -o "$BIN"

TS=$(date +%Y-%m-%d_%H%M%S)
LOG="results/run_wmma_${TS}.log"
echo "Running (sizes: ${1:-default})..."
./"$BIN" "$1" | tee "$LOG"

echo ""
echo "Log saved to: $LOG"
echo "CSV/MD saved alongside it in results/"
