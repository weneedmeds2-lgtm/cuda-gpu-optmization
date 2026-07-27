Shared Memory parallel scan benchmarks


 ============================================
CUDA Reduction Benchmark
============================================
Data Size           : 1000000 elements
Input Size          : 3.81 MB
Threads Per Block   : 512
Blocks Per Grid     : 1954
Shared Memory/Block : 2048 Bytes (2.00 KB)
============================================
[Naive Kernel]      Time: 1.13 ms | Throughput: 3.54 GB/s
[Sequential Kernel] Time: 0.33 ms | Throughput: 11.98 GB/s
[Warp Shuffle]      Time: 0.35 ms | Throughput: 11.59 GB/s
[-] Verification: SUCCESS (All reduction sums match perfectly: 4501334)

============================================
CUDA Reduction Benchmark
============================================
Data Size           : 10000000 elements
Input Size          : 38.15 MB
Threads Per Block   : 512
Blocks Per Grid     : 19532
Shared Memory/Block : 2048 Bytes (2.00 KB)
============================================
[Naive Kernel]      Time: 3.42 ms | Throughput: 11.71 GB/s
[Sequential Kernel] Time: 3.16 ms | Throughput: 12.67 GB/s
[Warp Shuffle]      Time: 3.28 ms | Throughput: 12.18 GB/s
[-] Verification: SUCCESS (All reduction sums match perfectly: 44988199)

============================================
CUDA Reduction Benchmark
============================================
Data Size           : 50000000 elements
Input Size          : 190.73 MB
Threads Per Block   : 512
Blocks Per Grid     : 97657
Shared Memory/Block : 2048 Bytes (2.00 KB)
============================================
[Naive Kernel]      Time: 17.04 ms | Throughput: 11.74 GB/s
[Sequential Kernel] Time: 16.40 ms | Throughput: 12.19 GB/s
[Warp Shuffle]      Time: 13.61 ms | Throughput: 14.69 GB/s
[-] Verification: SUCCESS (All reduction sums match perfectly: 224978297)

Improvements from SM → warps
1)Thread underutilization: ❌ No improvement
Active threads still reduce as N → N/2 → N/4 →N/8

2)Warp divergence:   largely solved
Sequential addressing already reduced divergence significantly.

3)Shared memory bank conflicts:  Improved
The final warp no longer accesses shared memory repeatedly.

4)Shared memory accesses:  Reduced
The last 32-thread reduction happens entirely in registers.

5)__syncthreads() overhead:  Improved
Block-wide synchronization is removed once only one warp remains active.

6)Instruction efficiency:  Improved
Warp shuffle instructions are very efficient for intra-warp communication.

7)Register utilization:  Improved
Partial sums stay in registers instead of being written back to shared memory.

Warp shuffle reduction improves the final stage of the reduction by replacing shared-memory communication and block-wide synchronization with fast register-to-register communication inside a single warp.

Naive < Sequential < Warp Shuffle
especially for larger input sizes.

nsight compute screenshots
