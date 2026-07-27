# CUDA GPU Optimization Portfolio

GPU performance engineering work covering parallel
primitives, matrix multiplication optimization, and
real-time image processing.

## Hardware
- GPU: NVIDIA RTX 3060 (SM86, Ampere, 12GB VRAM)
- Profiling: NVIDIA Nsight Compute (NCU)

## Projects

### 1. Parallel Primitives
Scan (inclusive/exclusive), histogram, reduction.
Optimized using shared memory, warp primitives,
bank conflict elimination.
→ [01-parallel-primitives/](./01-parallel-primitives/)

### 2. GEMM Optimization  
Tiled matrix multiply reaching 40-50% cuBLAS
on RTX 3060. NCU profiling confirmed cuBLAS
already optimal at batch=16 (44% SM throughput).
→ [02-gemm/](./02-gemm/)

### 3. Real-Time Image Filtering
CUDA kernels for Gaussian blur and custom filters
on live video frames. Demonstrates coalesced memory
access and shared memory tiling.
→ [03-image-filtering/](./03-image-filtering/)

## Key Findings
- Bank conflict elimination: +40% speedup on histogram
- Warp-level primitives vs shared memory reduction:
  eliminates __syncthreads() within warp boundary
- cuBLAS GEMM at batch=16 already at 44% SM util —
  custom kernel not beneficial for standard shapes
- NCU roofline confirms decode phase memory-bound
  at 71% HBM bandwidth utilization