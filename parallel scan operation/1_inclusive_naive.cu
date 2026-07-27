// ===============================================
// NAIVE INCLUSIVE SCAN (Hillis-Steele, global memory)
// Single block, no shared memory optimization.
// Work: O(N log N), but simple to reason about.
// ===============================================
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
using namespace std;

__global__ void naiveInclusiveScanStep(int* in, int* out, int N, int offset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    if (tid >= offset)
        out[tid] = in[tid] + in[tid - offset];
    else
        out[tid] = in[tid];
}

int main() {
    int N = 128;
    int* h = new int[N];
    for (int i = 0; i < N; i++) h[i] = rand() % 100;

    cout << "Input array:\n";
    for (int i = 0; i < N; i++) cout << h[i] << " ";
    cout << endl;

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 128; // assumes N <= threads for this simple naive version
    int blocks = (N + threads - 1) / threads;

    for (int offset = 1; offset < N; offset *= 2) {
        naiveInclusiveScanStep<<<blocks, threads>>>(d_in, d_out, N, offset);
        cudaDeviceSynchronize();
        swap(d_in, d_out); // ping-pong the device buffers (host-side pointer swap)
    }

    cudaMemcpy(h, d_in, N * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Inclusive Scan Result (Naive):\n";
    for (int i = 0; i < N; i++) cout << h[i] << " ";
    cout << endl;

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h;
    return 0;
}
