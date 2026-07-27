Warp Optimized INCLUSIVE SCAN (Warp Shuffle)

Uses __shfl_up_sync() for register-level communication within a warp.
Work: O(N)

Bottlenecks (Warp Optimized Scan):-
1. Communication limited to one warp:-
Warp shuffle instructions can exchange data only between threads of the same warp.
Different warps cannot communicate directly.

2. Shared memory still required:-
Warp totals are stored in shared memory.
Shared memory is still needed to combine the results of multiple warps.

3. Block-wide synchronization:-
__syncthreads() is still required after writing and scanning warp sums.
Different warps must synchronize before proceeding.

4. Extra warp-total scan
After every warp computes its local scan, the warp totals must also be scanned.
This adds an additional step to the algorithm.

5. Single-block implementation:-
This implementation scans only one thread block.
Large arrays require additional kernels similar to the shared-memory version.

6. Warp-size dependency:-
Designed around a warp size of 32 threads.
Block sizes should be multiples of 32 for best performance.

Improvements over Shared Memory Scan
#Uses registers instead of shared memory for intra-warp scan
#Most shared memory reads and writes are eliminated
#Less shared memory is required (only warp totals are stored)
#Fewer __syncthreads() are required
#Lower communication latency within a warp
#Less shared memory traffic
#Fewer memory instructions are executed
#Higher throughput due to register-level communication
#Maintains O(N) work complexity while reducing synchronization and shared-memory overhead compared to the shared-memory implementation.