
Naive Matrix Multiplication (Global Memory GEMM)
Each thread computes one element of the output matrix by reading an entire row from matrix A and an entire column from matrix B directly from global memory.
Work: O(N³)

Algorithm
Launch a 2D grid of thread blocks.
Each thread computes one output element C[row][col].
The thread loops over k = 0 → N-1.
For every iteration:
Read A[row][k]
Read B[k][col]
Multiply them
Accumulate the result.
Store the final value into C[row][col].
Bottlenecks (Naive GEMM)

1. Heavy global memory traffic:-
Every multiplication requires loading values directly from global memory.
The same elements are repeatedly fetched by different threads.

2. No data reuse:-
Once a value is loaded, it is used only by one thread.
Neighboring threads reload the same values again.

3. Memory bandwidth limited:-
Performance is dominated by global memory accesses rather than arithmetic operations.

4. Poor cache utilization:-
Matrix A has good locality.
Matrix B is accessed column-wise, resulting in poor cache reuse.

5. No shared memory usage:-
Fast on-chip shared memory is not utilized.
Every access goes to slower global memory.

6. Low arithmetic intensity:-
Very few arithmetic operations are performed for each global memory access.
GPU compute units spend time waiting for data.

7. Long execution time for large matrices:-
Every thread performs N iterations.
Total computation grows as O(N³).


Improvements over CPU
#Massive thread-level parallelism
#Thousands of output elements computed simultaneously
#Much higher throughput than a sequential CPU implementation
#Simple implementation and easy to understand

============================================

============================================
Benchmarking Matrix Dimension: 1024 x 1024
============================================
[-] Time        : 33.3967 ms
[-] Performance : 64.30 GFLOPS
[-] Efficiency  : 6.43% of theoretical peak

============================================
Benchmarking Matrix Dimension: 2048 x 2048
============================================
[-] Time        : 248.3692 ms
[-] Performance : 69.17 GFLOPS
[-] Efficiency  : 6.92% of theoretical peak

============================================
Benchmarking Matrix Dimension: 4096 x 4096
============================================
[-] Time        : 1947.4977 ms
[-] Performance : 70.57 GFLOPS
[-] Efficiency  : 7.06% of theoretical peak











