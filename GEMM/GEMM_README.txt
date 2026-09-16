CUDA GEMM Optimization Experiments

This directory is a set of standalone CUDA implementations of square matrix multiplication, organized as an optimization study. It moves from a one-output-per-thread FP32 baseline to shared-memory tiling, double buffering, per-thread register tiles, and WMMA Tensor Core execution. The most complete measurement harness is `gemm_wmma_benchmark.cu`, which validates a hand-written WMMA kernel against a naive FP32 reference and cuBLAS SGEMM.

The code is intended for learning and profiling rather than as a replacement for cuBLAS.

## Kernel map

| File | Data path | Main idea | Status |
| --- | --- | --- | --- |
| `Naive_MM.cu` | FP32 | One thread computes one `C[row, col]` directly from global memory. | Baseline benchmark. |
| `SMoptmized_MM.cu` | Integer | 16 x 16 shared-memory tiles and a transpose kernel for B. | Early tiled experiment. |
| `SMtransposeoptmized.cu` | FP32 | 32 x 32 shared-memory tiles with two shared-memory buffers. | Double-buffered tiled experiment. |
| `warpoptmized_MM.cu` | FP32 | 32 x 32 shared-memory tile intended for warp-oriented execution. | Kernel-only source; no `main()`. |
| `WMMA_MM.cu` | FP32 staged to FP16, FP32 accumulation | WMMA 16 x 16 x 16 Tensor Core fragments. | Original WMMA experiment. |
| `gemm_wmma_benchmark.cu` | FP32 reference and cuBLAS; WMMA stages to FP16 | Timed, verified comparison of naive FP32, cuBLAS SGEMM, and WMMA. | Recommended benchmark entry point. |

The `.txt` files preserve intermediate versions, notes, and benchmark snapshots. `files (1).zip` is an accompanying archive.

## Optimization path

1. **Naive GEMM** uses a 16 x 16 block; each thread performs one length-`N` dot product. It is simple but repeatedly reads the same A and B elements from global memory.
2. **Shared-memory tiling** stages submatrices of A and B once per block, then reuses them across the tile computation. This increases arithmetic intensity but introduces barriers and shared-memory capacity/bank-layout concerns.
3. **Transpose-assisted tiling** transposes B so a row-major traversal can access both operands more favorably. The transpose itself is extra work and must be included when comparing end-to-end latency.
4. **Double buffering** allocates two A/B shared-memory tile sets and alternates them across K tiles. The current implementation prefetches the next tile before switching buffers, although it uses block-wide synchronization rather than asynchronous copy primitives.
5. **Register tiling / warp-oriented kernels** assign multiple C elements to a thread, keeping partial sums in registers. This reduces shared-memory reads per fused multiply-add but raises register pressure and may lower occupancy.
6. **WMMA** maps 16 x 16 x 16 matrix operations to Tensor Cores. The benchmark kernel converts FP32 inputs to FP16 in shared memory and accumulates into FP32, so it requires tolerance-based validation.

## Requirements

- NVIDIA GPU, driver, and CUDA Toolkit
- `nvcc` and a supported C++ host compiler
- Compute capability 7.0 or newer for WMMA (`sm_70+`)
- cuBLAS for `gemm_wmma_benchmark.cu`

Choose an architecture flag matching your GPU. `sm_80` is only an example; it is not portable to every NVIDIA GPU.

## Build and run

From this directory, compile each self-contained program independently:

```powershell
nvcc -O3 -arch=sm_80 Naive_MM.cu -o Naive_MM.exe
.\Naive_MM.exe

nvcc -O3 -arch=sm_80 SMoptmized_MM.cu -o SMoptmized_MM.exe
.\SMoptmized_MM.exe

nvcc -O3 -arch=sm_80 SMtransposeoptmized.cu -o SMtransposeoptmized.exe
.\SMtransposeoptmized.exe

nvcc -O3 -arch=sm_80 WMMA_MM.cu -o WMMA_MM.exe
.\WMMA_MM.exe

nvcc -O3 -arch=sm_80 gemm_wmma_benchmark.cu -lcublas -o gemm_wmma_benchmark.exe
.\gemm_wmma_benchmark.exe
```

`gemm_wmma_benchmark.exe` accepts a comma-separated list of square sizes:

```powershell
.\gemm_wmma_benchmark.exe 1024,2048,4096
```

It runs 20 timed iterations per implementation, reports average kernel time and GFLOPS, verifies results, and writes timestamped CSV and Markdown files in `results/`.

## Measurement conventions

Throughput is computed as:

```text
GFLOPS = (2 * N^3) / elapsed_seconds / 1e9
```

The standalone programs generally time a kernel launch with CUDA events after a warm-up. Host-device transfers are excluded. The transpose-based programs run the transpose separately, so their printed GEMM time is not an end-to-end time unless the transpose cost is added. The WMMA benchmark averages 20 launches and reports verification status.

## Historical FP32 snapshots

The text notes contain these older measurements. They are useful as a local historical reference only: GPU model, clocks, CUDA version, and whether preprocessing was timed are not recorded consistently.

| N | Naive (GFLOPS) | Shared-memory tiled (GFLOPS) | Warp-level note (GFLOPS) |
| ---: | ---: | ---: | ---: |
| 1024 | 64.30 | 69.43 | 54.60 |
| 2048 | 69.17 | 74.69 | 64.35 |
| 4096 | 70.57 | 78.55 | 65.80 |

## Correctness and comparison cautions

- The original experimental sources do not consistently check CUDA API returns, kernel-launch errors, or numerical output. Treat printed GFLOPS as provisional until verified.
- `WMMA_MM.cu` and the original WMMA notes should not be used as a correctness reference. Use `gemm_wmma_benchmark.cu`, which fixes warp-cooperative indexing and compares against an FP32 reference.
- WMMA stages input values as FP16, so results differ from full FP32 GEMM. The benchmark deliberately uses combined relative and size-scaled absolute tolerances.
- `SMoptmized_MM.cu` uses `int`, while most other experiments use `float`; do not compare its throughput directly with the FP32 figures.
- `warpoptmized_MM.cu` contains only device kernels and will not link as an executable without a driver program.
- The theoretical-peak value hard-coded in some early examples (`1000 GFLOPS`) is illustrative, not a hardware-derived efficiency metric.

## Suggested profiling workflow

1. Start with `gemm_wmma_benchmark.cu` and record the generated CSV together with GPU name, CUDA version, and clocks.
2. Use Nsight Compute to examine achieved occupancy, register count, shared-memory bank conflicts, global-load efficiency, and Tensor Core utilization.
3. Compare equivalent precision modes and include any transpose or conversion step when making an end-to-end claim.
4. Add `CUDA_CHECK`-style error handling and CPU/cuBLAS validation to an experiment before changing tile sizes or memory layouts.