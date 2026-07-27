Naive EXCLUSIVE SCAN (Hillis-Steele):-
Uses repeated global memory updates to compute an inclusive scan, then shifts the result right by one element to obtain the exclusive scan.
Work: O(N log N)

Bottlenecks (Naive Exclusive Scan)

1. Work Inefficiency:-
Every iteration launches N threads, and there are log₂(N) iterations.
Total work becomes O(N log N) instead of O(N).

2. Multiple CUDA kernel launches:-
One kernel is launched for every offset.
After the inclusive scan completes, an additional kernel is launched to shift the array right by one.
Example for N = 1024
Scan kernels: offsets = 1,2,4,8,16,32,64,128,256,512
Shift kernel
Total = 11 kernel launches

3. Heavy global memory traffic:-
Every scan iteration:
Reads from global memory (d_in)
Writes to global memory (d_out)
The final shift kernel performs another full read and write of the array.
Makes the algorithm memory-bandwidth limited.

4. Requires multiple device buffers:-
Separate input and output buffers are needed for the scan.
An additional output buffer is required for the exclusive scan after shifting.


5. Frequent synchronization:-
cudaDeviceSynchronize() is called after every kernel launch.
Prevents overlapping execution and increases overhead.

6. Poor scalability for large arrays:-
Larger arrays require:
More scan iterations
More kernel launches
More global memory accesses
Performance does not scale well.

7. Extra shift operation:-
Exclusive scan cannot be produced directly in this implementation.
Requires an additional kernel to shift the inclusive result by one position.