// ===============================================
// SHARED-MEMORY OPTIMIZED INCLUSIVE SCAN
// Work-efficient Blelloch upsweep/downsweep per block,
// then a second-level exclusive scan over block sums,
// then an offset-add pass. Multi-block capable.
//
// NOTE: assumes `blocks` (grid size of the per-block scan)
// is a power of two and fits in one block for the
// second-level scan (true for the demo sizes below).
// ===============================================
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

// Step 1: per-block work-efficient scan (Blelloch), writes INCLUSIVE result
__global__ void blockInclusiveScan(int* da, int* gblock, int N) {
    extern __shared__ int smscan[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int orig = (gid < N) ? da[gid] : 0;
    smscan[tid] = orig;
    __syncthreads();

    // upsweep
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int right = (tid + 1) * 2 * stride - 1;
        int left = right - stride;
        if (right < blockDim.x) {
            smscan[right] += smscan[left];
        }
        __syncthreads();
    }

    if (tid == 0) {
        gblock[blockIdx.x] = smscan[blockDim.x - 1]; // total sum of this block
        smscan[blockDim.x - 1] = 0;                  // seed for exclusive downsweep
    }
    __syncthreads();

    // downsweep
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        int right = (tid + 1) * 2 * stride - 1;
        int left = right - stride;
        if (right < blockDim.x) {
            int t = smscan[left];
            smscan[left] = smscan[right];
            smscan[right] += t;
        }
        __syncthreads();
    }

    // smscan[tid] now holds the EXCLUSIVE result -> add back original for INCLUSIVE
    if (gid < N) da[gid] = smscan[tid] + orig;
}

// Step 2: exclusive scan over the block sums (single block, size = number of blocks)
__global__ void blockSumsExclusiveScan(int* gblock, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    temp[tid] = (tid < n) ? gblock[tid] : 0;
    __syncthreads();

    for (int stride = 1; stride < n; stride *= 2) {
        int right = (tid + 1) * 2 * stride - 1;
        int left = right - stride;
        if (right < n) temp[right] += temp[left];
        __syncthreads();
    }
    if (tid == 0) temp[n - 1] = 0;
    __syncthreads();

    for (int stride = n / 2; stride >= 1; stride /= 2) {
        int right = (tid + 1) * 2 * stride - 1;
        int left = right - stride;
        if (right < n) {
            int t = temp[left];
            temp[left] = temp[right];
            temp[right] += t;
        }
        __syncthreads();
    }

    if (tid < n) gblock[tid] = temp[tid];
}

// Step 3: add each block's prefix offset to its elements (block 0 gets +0)
__global__ void addOffset(int* da, int* gblock, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N && blockIdx.x > 0) {
        da[gid] += gblock[blockIdx.x];
    }
}

int main() {
    int N = 128;
    int threadsPerBlock = 16;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock; // = 8, power of 2

    int* h = new int[N];
    for (int i = 0; i < N; i++) h[i] = rand() % 100;

    cout << "Input array:\n";
    for (int i = 0; i < N; i++) cout << h[i] << " ";
    cout << endl;

    int *da, *gblock;
    cudaMalloc(&da, N * sizeof(int));
    cudaMalloc(&gblock, blocks * sizeof(int));
    cudaMemcpy(da, h, N * sizeof(int), cudaMemcpyHostToDevice);

    blockInclusiveScan<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(da, gblock, N);
    blockSumsExclusiveScan<<<1, blocks, blocks * sizeof(int)>>>(gblock, blocks);
    addOffset<<<blocks, threadsPerBlock>>>(da, gblock, N);

    cudaMemcpy(h, da, N * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Inclusive Scan Result (SM-Optimized):\n";
    for (int i = 0; i < N; i++) cout << h[i] << " ";
    cout << endl;

    cudaFree(da);
    cudaFree(gblock);
    delete[] h;
    return 0;
}
