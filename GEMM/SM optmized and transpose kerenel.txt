#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>

using namespace std;

#define TILE_SIZE 32

__global__ void transpose(float* matrix_b, int N, float* matrixbt) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    int globalrow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalcol = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (globalrow < N && globalcol < N) {
        tile[threadIdx.y][threadIdx.x] = matrix_b[globalrow * N + globalcol];
    }
    __syncthreads();

    int trow = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tcol = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (trow < N && tcol < N) {
        matrixbt[tcol * N + trow] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void matrix_multiply_tiled_kernel(float* matrix_a, float* matrix_b, float* matrix_c, int N) {
    __shared__ float tile_a[2][TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[2][TILE_SIZE][TILE_SIZE];
    int compute_idx = 0;
    int local_idx = 0;

    int globalrow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalcol = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    if (globalrow < N && threadIdx.x < TILE_SIZE) {
        tile_a[0][threadIdx.y][threadIdx.x] = matrix_a[globalrow * N + threadIdx.x];
    }
    if (globalcol < N && threadIdx.y < TILE_SIZE) {
        tile_b[0][threadIdx.y][threadIdx.x] = matrix_b[threadIdx.y * N + globalcol];
    }
    __syncthreads();

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
#pragma unroll
        for (int k = 0; k < 32; k++) {
            sum += tile_a[compute_idx][threadIdx.y][k] * tile_b[compute_idx][k][threadIdx.x];
        }

        if (t + 1 < (N + TILE_SIZE - 1) / TILE_SIZE) {
            local_idx = compute_idx ^ 1;
            int nextt = t + 1;
            if (globalrow < N && (nextt * TILE_SIZE + threadIdx.x) < N) {
                tile_a[local_idx][threadIdx.y][threadIdx.x] = matrix_a[globalrow * N + (nextt * TILE_SIZE + threadIdx.x)];
            }
            if (globalcol < N && (nextt * TILE_SIZE + threadIdx.y) < N) {
                tile_b[local_idx][threadIdx.y][threadIdx.x] = matrix_b[(nextt * TILE_SIZE + threadIdx.y) * N + globalcol];
            }
        }
        __syncthreads();
        compute_idx = compute_idx ^ 1;
    }

    if (globalrow < N && globalcol < N) {
        matrix_c[globalrow * N + globalcol] = sum;
    }
}

int main() {
    vector<int> test_sizes = { 512, 1024, 2048 };

    for (int N : test_sizes) {
        cout << "\n============================================\n";
        cout << "Benchmarking Matrix Dimension: " << N << " x " << N << "\n";
        cout << "============================================\n";

        size_t matrix_bytes = N * N * sizeof(float);
        vector<float> h_a(N * N, 1.0f), h_b(N * N, 1.0f), h_my_c(N * N);

        float* d_a, * d_b, * d_c, * d_bt;
        cudaMalloc(&d_a, matrix_bytes);
        cudaMalloc(&d_b, matrix_bytes);
        cudaMalloc(&d_c, matrix_bytes);
        cudaMalloc(&d_bt, matrix_bytes);

        cudaMemcpy(d_a, h_a.data(), matrix_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), matrix_bytes, cudaMemcpyHostToDevice);

        dim3 block_dim(TILE_SIZE, TILE_SIZE);
        dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        cudaEventRecord(start);
        transpose << <grid_dim, block_dim >> > (d_b, N, d_bt);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        matrix_multiply_tiled_kernel << <grid_dim, block_dim >> > (d_a, d_bt, d_c, N);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        double gflops = (2.0 * N * N * N) / (milliseconds / 1000.0) / 1e9;
        double peak_gflops = 1000.0;
        double efficiency = (gflops / peak_gflops) * 100.0;

        cout << "[-] Time: " << milliseconds << " ms" << endl;
        cout << "[-] Performance: " << gflops << " GFLOPS" << endl;
        cout << "[-] Efficiency: " << efficiency << "% of theoretical peak" << endl;

        cudaMemcpy(h_my_c.data(), d_c, matrix_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c), cudaFree(d_bt);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    return 0;
}