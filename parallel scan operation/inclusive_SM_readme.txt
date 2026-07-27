

Shared Memory INCLUSIVE SCAN (Blelloch Scan)
shared memory optimization.
Work: O(N)

Bottlenecks (Shared Memory Blelloch Scan)

1) Multiple kernel launches
Still requires multiple kernel launches.
Block scan
Block sums scan
Offset addition
Although this is much fewer than the naive implementation, kernel launch overhead still exists.

2) Shared memory usage:-
Each block stores its data in shared memory.
Larger block sizes consume more shared memory, which can reduce occupancy if shared memory becomes the limiting resource.

3) Block-wide synchronization
The upsweep and downsweep phases require several calls.
__syncthreads();
All threads in the block must wait at every level of the reduction tree before continuing.

4) Extra block-sum scan:-
For multi-block arrays, each block computes only a local scan.
The block sums must be scanned separately before the final offsets can be added.
This introduces additional computation and another kernel launch.

5) Extra memory for block sums:-
Requires an additional global memory array (gblock) to store the sum of each block.
This slightly increases memory usage.

6) Limited by shared memory capacity:-
A block can only scan as many elements as fit into shared memory.
Very large arrays must be divided into multiple blocks, requiring the second-level scan and offset-add phase.

7) More complex implementation:-
Compared to Hillis-Steele, Blelloch scan is more difficult to implement correctly.
It involves:
1)Upsweep phase
2)Downsweep phase
3)Block sum computation
4)Block sum scan
5)Offset addition
making debugging and maintenance more challenging.



Improvements over Naive
 #Work complexity reduced from O(N log N) to O(N)
 #Far fewer global memory accesses
 #Only 3 kernel launches instead of log₂(N)
 #No repeated ping-pong of the entire input/output arrays every iteration
 #Better scalability
 #Much higher memory bandwidth utilization

