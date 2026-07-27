
NAIVE INCLUSIVE SCAN (Hillis-Steele, global memory)
Single block, no shared memory optimization.
Work: O(N log N)
Hillis-Steele Inclusive Scan (Naive)

#Algorithm:-
1)Launch one thread for each element in the array.
2)Every thread starts with its own value.
3)In the first iteration (offset = 1), each thread adds the value immediately to its left.
4)In the next iteration (offset = 2), each thread adds the value two positions to its left.
5)Continue doubling the offset (1, 2, 4, 8, ...) until the offset is greater than or equal to the array size.
6)After each iteration, swap the input and output buffers so the next iteration uses the updated results.
7)After all iterations, every element contains the sum of all elements before it, including itself (inclusive prefix sum).


Bottlenecks:-
1)Work Inefficiency
Every iteration launches N threads, and there are log₂(N) iterations.
instead of O(N);

2)Multiple cuda kernel launches
A new CUDA kernel is launched for every offset.
eg for N = 1024
This requires 10 separate kernel launches, and each launch has overhead.
offsets:-1,2,4,8,16,32,64,128,256,512

3)Heavy global memory traffic
Every iteration:-

reads from global memory (d_in)
writes to global memory (d_out)
Since this happens every pass, the same data is read and written many times, making the algorithm memory-bandwidth limited.

4)Requires two device buffers
The algorithm cannot update the array in place because threads must read the previous iteration's values.
input and output buffer

5)Frequent synchronization
cudaDeviceSynchronize();
This prevents overlapping work and adds execution overhead.

6)Poor scalability for large arrays
As the array size increases:

more iterations are required
more global memory accesses occur
more kernel launches are needed
Performance does not scale as well as more optimized scan algorithms.
