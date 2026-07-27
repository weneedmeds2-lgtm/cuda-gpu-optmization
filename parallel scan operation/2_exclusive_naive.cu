// ===============================================
// NAIVE EXCLUSIVE SCAN (Hillis-Steele, global memory)
// Computes inclusive scan first, then shifts right by 1
// to get the exclusive result (element 0 = 0).
// ===============================================
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
using namespace std;

__global__ void naiveScanStep(int* in, int* out, int N, int offset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    if (tid >= offset)
        out[tid] = in[tid] + in[tid - offset];
    else
        out[tid] = in[tid];
}

__global__ void shiftRightByOne(int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    out[tid] = (tid == 0) ? 0 : in[tid - 1];
}

int main() {
    int N = 128;
    int* h = new int[N];
    for (int i = 0; i < N; i++) h[i] = rand() % 100;

    cout << "Input array:\n";
    for (int i = 0; i < N; i++) cout << h[i] << " ";
    cout << endl;

    int *d_in, *d_out, *d_excl;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMalloc(&d_excl, N * sizeof(int));
    cudaMemcpy(d_in, h, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (N + threads - 1) / threads;

    for (int offset = 1; offset < N; offset *= 2) {
        naiveScanStep<<<blocks, threads>>>(d_in, d_out, N, offset);
        cudaDeviceSynchronize();
        swap(d_in, d_out);
    }
    // d_in now holds the inclusive scan result

    shiftRightByOne<<<blocks, threads>>>(d_in, d_excl, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h, d_excl, N * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Exclusive Scan Result (Naive):\n";
    for (int i = 0; i < N; i++) cout << h[i] << " ";
    cout << endl;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_excl);
    delete[] h;
    return 0;
}
