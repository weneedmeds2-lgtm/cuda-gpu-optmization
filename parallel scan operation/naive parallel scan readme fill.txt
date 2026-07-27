Naive parallel scan benchmarks


 Data Size: 1000000 elements
============================================
[Naive Kernel]      Time: 109.726 ms | Throughput: 0.0364545 GB/s

 Data Size: 10000000 elements
============================================
[Naive Kernel]      Time: 2.93171 ms | Throughput: 13.6439 GB/s

 Data Size: 50000000 elements
============================================
[Naive Kernel]      Time: 14.9064 ms | Throughput: 13.4171 GB/s


Bottlenecks:-
1)Thread underutilization:-
For every N threads launched it is getting reduced by 2 factor (N/2,N/4,N/8)
Most threads are idle for much of the kernel

2)Warp divergence:-
Within a warp, some threads execute the addition while others do nothing.

3)Shared memory bank conflicts
Multiple threads access shared memory in patterns that map to the same memory bank,causing 
accesses to serialize

As stride changes theses access patterns can create bank conflicts on some GPU architectures

4)Too many __syncthreads()
for max threads 1024 there will be 10-block wide synchronizations
It is necessary but once only a single warp remains active, a full block synchronization is unnecessary because threads within a warp execute in lockstep.

5)Poor instruction efficiency
Most launched threads become idle

nsight compute screenshots
