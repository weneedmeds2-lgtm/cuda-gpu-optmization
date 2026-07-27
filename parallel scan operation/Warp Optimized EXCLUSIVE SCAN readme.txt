Warp Optimized EXCLUSIVE SCAN (Warp Shuffle):-
Uses __shfl_up_sync() for register-level communication within each warp. The inclusive result is converted to an exclusive scan by subtracting each thread's original value.
Work: O(N)

Bottlenecks (Warp Optimized Exclusive Scan)

1. Communication limited to one warp:-
Warp shuffle instructions can exchange data only within a single warp.
Communication between different warps still requires shared memory.

2. Shared memory still required:-
Warp totals are stored in shared memory.
Shared memory is needed to combine the results of multiple warps.

3. Block-wide synchronization:-
__syncthreads() is still required after storing and scanning warp sums.
Different warps must synchronize before computing the final result.

4. Extra warp-total scan:-
Each warp computes a local scan.
Warp totals must also be scanned to obtain the offset for every warp.

5. Single-block implementation:-
This implementation scans only one thread block.
Larger arrays require a hierarchical multi-block scan.

6. Warp-size dependency:-
Designed around a warp size of 32 threads.
Best performance is achieved when block sizes are multiples of 32.

7. Extra subtraction step:-
The algorithm first computes an inclusive scan.
Each thread subtracts its original value to obtain the exclusive result.


Improvements over Shared Memory
#Uses registers instead of shared memory for intra-warp scan
#Most shared memory accesses are eliminated
#Less shared memory is required (only warp totals are stored)
#Fewer __syncthreads() are required
#Lower communication latency within a warp
#Less shared memory traffic
#Fewer memory instructions are executed
#Higher throughput due to register-level communication
#Maintains O(N) work complexity while reducing synchronization and shared-memory overhead