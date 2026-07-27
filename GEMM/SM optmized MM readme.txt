Shared Memory Tiled GEMM:-
Uses shared memory tiling to reduce global memory accesses.
Each thread block loads a tile of matrix A and matrix B into shared memory, then performs multiplication using fast shared memory access.

Work: O(N³)

Algorithm:-
Divide matrices A and B into smaller tiles of size TILE_SIZE × TILE_SIZE.
Each thread block computes one tile of output matrix C.
For every tile iteration:
Load a tile of matrix A from global memory into shared memory.
Load a tile of matrix B from global memory into shared memory.
Synchronize threads using __syncthreads().
Each thread computes its output element:
Multiply corresponding values from shared memory tiles.
Accumulate the partial sum.
Repeat until all tiles are processed.
Store the final result into matrix C.
Bottlenecks (Shared Memory Tiled GEMM)

1. Shared memory capacity:-
Each block stores tiles of A and B in shared memory.
Larger tile sizes require more shared memory.
Excessive shared memory usage can reduce occupancy.

2. Synchronization overhead:-
Every tile loading step requires:
__syncthreads() after loading tiles.
__syncthreads() after computation.
Threads must wait before moving to the next tile.

3. Bank conflicts:-
Threads access shared memory simultaneously.
Poor access patterns can cause shared memory bank conflicts.
Bank conflicts serialize memory operations.

4. Limited tile size:-
Performance depends on choosing the correct tile size.
Small tiles:
Less data reuse.
More global memory transactions.
Large tiles:
More shared memory usage.
Lower occupancy.

5. Global memory loading overhead:-
Tiles still need to be loaded from global memory.
For every tile iteration, data movement is required.

6. Register pressure:-
Each thread keeps partial sums in registers.
Large tiles or additional optimizations increase register usage.
High register usage can reduce the number of active warps.

7. Transpose overhead:-
Additional transpose kernel is required for matrix B.
This adds extra kernel execution time and memory movement.
Improvements over Naive GEMM

#Data reuse introduced through shared memory
#Each value loaded from global memory is reused by multiple threads
#Global memory accesses are significantly reduced
#Better cache and memory bandwidth utilization
#Higher arithmetic intensity
#Threads spend more time performing computation instead of waiting for memory
#Better GPU utilization compared to naive global memory GEMM
#Higher GFLOPS performance


============================================
Benchmarking Matrix Dimension: 1024 x 1024
============================================
[-] Time        : 30.9289 ms
[-] Performance : 69.43 GFLOPS
[-] Efficiency  : 6.94% of theoretical peak

============================================
Benchmarking Matrix Dimension: 2048 x 2048
============================================
[-] Time        : 230.0119 ms
[-] Performance : 74.69 GFLOPS
[-] Efficiency  : 7.47% of theoretical peak

============================================
Benchmarking Matrix Dimension: 4096 x 4096
============================================
[-] Time        : 1749.6494 ms
[-] Performance : 78.55 GFLOPS
[-] Efficiency  : 7.86% of theoretical peak



























