Shared Memory EXCLUSIVE SCAN (Blelloch Scan):-
Uses shared memory and the Blelloch upsweep/downsweep algorithm to compute an exclusive scan efficiently.
Work: O(N)

Bottlenecks (Shared Memory Blelloch Scan)

1. Multiple kernel launches:-
Still requires multiple kernel launches.
Block scan
Block sums scan
Offset addition
Although fewer than the naive implementation, kernel launch overhead still exists.

2. Shared memory usage:-
Each block stores its data in shared memory.
Larger block sizes consume more shared memory, which can reduce occupancy if shared memory becomes the limiting resource.

3. Block-wide synchronization:-
The upsweep and downsweep phases require several calls to:
__syncthreads();
All threads in the block must wait before proceeding to the next stage.

4. Extra block-sum scan:-
Each block computes only a local exclusive scan.
The block sums must also be scanned before the final offsets can be added.
Requires additional computation and another kernel launch.

5. Extra memory for block sums:-
Requires an additional global memory array (gblock) to store block totals.
Slightly increases memory usage.

6. Limited by shared memory capacity:-
A block can only scan as many elements as fit into shared memory.
Large arrays require multiple blocks along with the block-sum scan and offset-add phase.

7. More complex implementation:-
More difficult to implement than Hillis-Steele.
Involves:
Upsweep phase
Downsweep phase
Block sum computation
Block sum scan
Offset addition


Improvements over Naive
#Work complexity reduced from O(N log N) to O(N)
#Far fewer global memory accesses
#Only 3 kernel launches instead of log₂(N) + 1
#No repeated ping-pong of the entire input/output arrays every iteration
#Exclusive scan is produced directly (no separate shift kernel required)
#Better scalability for large arrays
#Much higher memory bandwidth utilization
#Reduced synchronization overhead compared to launching a kernel for every offset