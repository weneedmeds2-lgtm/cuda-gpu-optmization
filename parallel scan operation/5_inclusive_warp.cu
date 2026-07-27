// ===============================================
// WARP-OPTIMIZED INCLUSIVE SCAN
// Uses __shfl_up_sync for register-level intra-warp scan
// (no shared memory needed for the warp step itself).
// Warp totals are combined via a tiny shared-memory array.
// Single block (N <= blockDim.x), multiple warps per block.
// ===============================================
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

__device__ int warpInclusiveScan(int val) {
    int lane = threadIdx.x & 31;
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += n;
    }
    return val;
}

__global__ void blockInclusiveScanWarp(int* da, int N) {
    extern __shared__ int warpSums[]; // one slot per warp
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warpId = tid >> 5;

    int val = (tid < N) ? da[tid] : 0;
    int scanned = warpInclusiveScan(val); // inclusive scan within this warp

    // last lane of each warp holds that warp's total
    if (lane == 31) warpSums[warpId] = scanned;
    __syncthreads();

    // scan the (small) array of warp totals using warp 0
    if (warpId == 0) {
        int numWarps = blockDim.x / 32;
        int wv = (lane < numWarps) ? warpSums[lane] : 0;
        int wscanned = warpInclusiveScan(wv);
        if (lane < numWarps) warpSums[lane] = wscanned;
    }
    __syncthreads();

    // exclusive prefix of warp totals = offset to add to this warp's elements
    int warpOffset = (warpId == 0) ? 0 : warpSums[warpId - 1];

    if (tid < N) da[tid] = scanned + warpOffset;
}

int main() {
    int N = 128;
    int* h = new int[N];
    for (int i = 0; i < N; i++) h[i] = rand() % 100;

    cout << "Input array:\n";
    for (int i = 0; i < N; i++) cout << h[i] << " ";
    cout << endl;

    int* d;
    cudaMalloc(&d, N * sizeof(int));
    cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 128; // 4 warps, must be multiple of 32, N <= threads
    int numWarps = threads / 32;
    blockInclusiveScanWarp<<<1, threads, numWarps * sizeof(int)>>>(d, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Inclusive Scan Result (Warp-Optimized):\n";
    for (int i = 0; i < N; i++) cout << h[i] << " ";
    cout << endl;

    cudaFree(d);
    delete[] h;
    return 0;
}
