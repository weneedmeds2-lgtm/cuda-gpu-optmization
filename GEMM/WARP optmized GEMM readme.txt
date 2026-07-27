#Warp-Level Optimized GEMM:-
Uses shared memory tiling along with warp-level execution to improve matrix multiplication performance.
Each warp computes multiple output elements using register-level operations and shuffle instructions instead of relying only on shared memory accesses.

Work: O(N³)
Algorithm:-
Divide matrices A and B into smaller tiles of size TILE_SIZE × TILE_SIZE.
Transpose matrix B:
Convert B into Bᵀ for better memory access pattern.
Allows more coalesced global memory access.
Each thread block loads tiles of matrix A and matrix B into shared memory.
Synchronize threads using __syncthreads() after loading tiles.
Perform warp-level computation:
Threads inside a warp exchange values using __shfl_sync().
Reduce dependency on shared memory during computation.
Each thread computes multiple output values using register storage.
Each thread maintains partial sums:
Accumulates results in registers.
Reuses loaded tile values multiple times.
Repeat for all tiles of matrices A and B.
Store computed values into output matrix C.

Bottlenecks (Warp-Level Optimized GEMM):-

1. Register usage:-
Each thread stores multiple partial sums:
sum[2][2]
Higher register usage can:
Reduce occupancy.
Limit number of active warps per SM.

2. Warp synchronization dependency:-
Warp shuffle operations require threads inside a warp to execute together.
Issues:
Divergent warps reduce efficiency.
Incorrect synchronization can produce invalid results.

3. Shared memory usage:-
Tiles of A and B are still stored in shared memory.
Problems:
Large tiles increase shared memory consumption.
High shared memory usage can reduce active blocks per SM.

4. Shared memory bank conflicts:-
Threads access shared memory tiles simultaneously.
Poor access pattern can cause:
Bank conflicts.
Serialized shared memory transactions.

5. Increased implementation complexity:-
Compared to tiled GEMM, warp-level optimization requires managing:
Warp IDs.
Lane IDs.
Shuffle operations.
Multiple output calculations per thread.
Debugging becomes more difficult.

6. Limited scalability:-
Warp-level optimization depends on:
Warp size (32 threads).
Tile size selection.
GPU architecture.
A configuration optimized for one GPU may not perform equally on another GPU.

7. Memory loading overhead:-
Tiles still need to be loaded from global memory.
For every tile iteration:
Load A tile.
Load B tile.
Synchronize.

Global memory bandwidth can still become a limiting factor.

8. Occupancy limitation:-
Higher register usage and shared memory usage reduce the number of active warps on an SM.
Lower occupancy can reduce the ability to hide memory latency.

Improvements over Shared Memory Tiled GEMM:-
#Warp-level execution reduces shared memory dependency during computation.
#Uses __shfl_sync() for fast register-level data exchange.
#Multiple output elements are computed per thread.
#Higher arithmetic intensity compared to basic tiled GEMM.
#Better utilization of CUDA cores through warp-level parallelism.
#Reduced repeated shared memory accesses during computation.
#Improved instruction efficiency by using register-level operations.
#Higher potential GFLOPS performance compared to standard tiled GEMM.
#Better mapping of computation to GPU warp execution model.


============================================
Benchmarking Matrix Dimension: 1024 x 1024
============================================
[-] Time        : 39.3329 ms
[-] Performance : 54.60 GFLOPS
[-] Efficiency  : 5.46% of theoretical peak

============================================
Benchmarking Matrix Dimension: 2048 x 2048
============================================
[-] Time        : 266.9937 ms
[-] Performance : 64.35 GFLOPS
[-] Efficiency  : 6.43% of theoretical peak

============================================
Benchmarking Matrix Dimension: 4096 x 4096
============================================
[-] Time        : 2088.6179 ms
[-] Performance : 65.80 GFLOPS
[-] Efficiency  : 6.58% of theoretical peak