# CUDA GPU Optimization Portfolio

This repository documents a hands-on CUDA optimization journey: starting with direct global-memory kernels, then improving data reuse with shared memory, reducing synchronization and memory traffic with warp/register techniques, and finally using Tensor Cores through WMMA. The work is organized as focused experiments rather than one monolithic application.

## What I built

| Area | What I implemented | Optimization focus |
| --- | --- | --- |
| Matrix multiplication (`GEMM/`) | Naive FP32 GEMM, tiled/shared-memory GEMM, transpose-assisted variants, double-buffered experiments, warp/register-tiled kernels, WMMA, and a cuBLAS comparison harness. | Coalescing, tile reuse, synchronization, register pressure, and Tensor Core execution. |
| Advanced GEMM (`GEMM2/`) | `float4` vectorized GEMM, 2D register tiling, double-buffered shared-memory kernels, and 32 x 32 through 128 x 128 WMMA tiles. | Wider memory transactions, arithmetic intensity, bank-conflict avoidance, and larger block-level tiles. |
| Histograms (`histogram github/`) | Naive global-atomic, per-block shared histogram, and replicated shared-subhistogram implementations. | Reducing contention on global atomics and balancing shared-memory use against contention. |
| Parallel scans (`parallel scan operation/`) | Inclusive and exclusive scans; naive, shared-memory, warp-oriented, and multi-block variants. | Prefix-sum parallelization, Blelloch up-sweep/down-sweep, block summaries, and inter-block propagation. |
| Optimization reference material | CUDA notes, benchmark reports, checklists, and interview-preparation documents. | Capturing measurement results and reusable GPU-performance concepts. |

## Technical progression

### 1. Baseline kernels

The starting point was the simplest correct GPU mapping: one output element per thread. These kernels establish a reference for execution time, memory behavior, and correctness, but repeatedly fetch the same data from global memory.

### 2. Shared-memory tiling

I introduced block-level tiles so values loaded from global memory could be reused by many threads. This raises arithmetic intensity, but performance becomes sensitive to tile shape, synchronization frequency, occupancy, and shared-memory bank access patterns.

### 3. Memory-layout optimization

For GEMM, I explored transposing B and using padded shared-memory arrays. The goal is to replace poorly coalesced or conflict-prone accesses with predictable row-major loads and conflict-resistant on-chip access.

### 4. Overlap and reuse

Double-buffered GEMM variants alternate between two shared-memory tile buffers. While one K tile is consumed, the next can be prepared. Register tiles further increase reuse by letting one thread accumulate several C elements before storing them.

### 5. Vectorization and warp-level work

The `GEMM2` kernels use `float4` loads/stores to move 128 bits per operation where alignment and dimensions allow. Other experiments map multiple output elements to one thread or use warp-oriented operations, trading simpler code and higher occupancy for more register reuse and lower instruction overhead.

### 6. Tensor Cores and WMMA

WMMA kernels use 16 x 16 x 16 fragments and FP16 staging with FP32 accumulation. The benchmark harness compares a custom WMMA implementation with a naive FP32 reference and cuBLAS SGEMM, adds CUDA/cuBLAS error checks, runs repeated timed launches, and uses tolerance-based verification for mixed-precision output.

## Measurement practice

The experiments use CUDA events to measure kernel execution and report throughput as:

```text
GFLOPS = (2 * N^3) / elapsed_seconds / 1e9
```

The more mature WMMA benchmark performs warm-up launches, averages multiple timed iterations, compares output with a reference, and writes results to CSV/Markdown. Earlier experiments primarily serve as exploratory benchmarks; their results should be re-run with recorded GPU model, CUDA version, clocks, precision, and inclusion/exclusion of preprocessing work.

## Key engineering lessons demonstrated

- GPU performance depends on data movement and execution layout, not only FLOP count.
- Shared memory improves reuse but can lose to synchronization, bank conflicts, or reduced occupancy.
- Vectorized access requires correct alignment and tail handling.
- Register tiling can improve arithmetic intensity but must be balanced against register pressure.
- A transpose can improve access locality but must be included in end-to-end cost comparisons.
- Tensor Core kernels need different correctness expectations because input conversion changes numerical error.
- Profiling and validation are required before interpreting a speedup as a real improvement.

## Recommended entry points

- Start with `GEMM/gemm_wmma_benchmark.cu` for the most complete GEMM benchmark and cuBLAS comparison.
- Read `GEMM2/README.md` for the vectorized/register-tiled FP32 and WMMA variants.
- Inspect `histogram github/histogram.cu` to compare atomic-contention reduction strategies.
- Inspect `parallel scan operation/` for scan implementations that progress from a single block to multi-block prefix sums.

## Next direction

The natural next step is to standardize every experiment around a shared benchmark harness: consistent input generation, warm-up and iteration counts, CUDA error checking, correctness validation, reproducible result files, and Nsight Compute metrics. That would turn the individual optimization studies into directly comparable performance experiments.
