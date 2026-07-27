#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mma.h> 
#include <iostream>
#include <vector>
#include<cublas_v2.h>

using namespace std;

// Strict hardware dimensions for FP16 Tensor Cores
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

#define TILE_DIM 16

__global__ void matrix_multiply_tensor_cores_kernel(float* matrix_a, float* matrix_b, float* matrix_c, int N) {
    // Explicitly stage tiles as __half types
    __shared__ __half sh_a[TILE_DIM][TILE_DIM];
    __shared__ __half sh_b[TILE_DIM][TILE_DIM];

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    int row = warpM * WMMA_M;
    int col = warpN * WMMA_N;

    if (row >= N || col >= N) {
        return;
    }

    // Explicitly call full namespace variants to bypass compiler/Intellisense confusion
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < N; k += TILE_DIM) {

        int t_id = threadIdx.y * blockDim.x + threadIdx.x;

#pragma unroll
        for (int item = 0; item < 4; item++) {
            int element_idx = t_id * 4 + item;
            int tile_r = element_idx / TILE_DIM;
            int tile_c = element_idx % TILE_DIM;

            if (tile_r < TILE_DIM) {
                int gl_row_a = row + tile_r;
                int gl_col_a = k + tile_c;
                if (gl_row_a < N && gl_col_a < N) {
                    sh_a[tile_r][tile_c] = (__half)matrix_a[gl_row_a * N + gl_col_a];
                }
                else {
                    sh_a[tile_r][tile_c] = (__half)0.0f;
                }

                int gl_row_b = k + tile_r;
                int gl_col_b = col + tile_c;
                if (gl_row_b < N && gl_col_b < N) {
                    sh_b[tile_r][tile_c] = (__half)matrix_b[gl_row_b * N + gl_col_b];
                }
                else {
                    sh_b[tile_r][tile_c] = (__half)0.0f;
                }
            }
        }

        __syncthreads();

        // Load explicitly from our staging arrays using the fully specified namespaces
        nvcuda::wmma::load_matrix_sync(a_frag, (const __half*)sh_a, TILE_DIM);
        nvcuda::wmma::load_matrix_sync(b_frag, (const __half*)sh_b, TILE_DIM);

        // Run the hardware tensor engine
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(matrix_c + row * N + col, c_frag, N, nvcuda::wmma::mem_row_major);
}

int main() {
    vector<int> test_sizes = { 1024, 2048, 4096 };

    for (int N : test_sizes) {
        cout << "\n============================================\n";
        cout << "Benchmarking Tensor Cores Matrix Dimension: " << N << " x " << N << "\n";
        cout << "============================================\n";

        size_t matrix_bytes = N * N * sizeof(float);
        float* d_a, * d_b, * d_c;
        cudaMalloc(&d_a, matrix_bytes);
        cudaMalloc(&d_b, matrix_bytes);
        cudaMalloc(&d_c, matrix_bytes);

        vector<float> h_a(N * N, 1.0f), h_b(N * N, 1.0f);
        cudaMemcpy(d_a, h_a.data(), matrix_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), matrix_bytes, cudaMemcpyHostToDevice);

        dim3 block_dim(32, 2);
        dim3 grid_dim((N + (WMMA_N * 2) - 1) / (WMMA_N * 2), (N + WMMA_M - 1) / WMMA_M);

        matrix_multiply_tensor_cores_kernel << <grid_dim, block_dim >> > (d_a, d_b, d_c, N);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int i = 0; i < 100; i++) {
            matrix_multiply_tensor_cores_kernel << <grid_dim, block_dim >> > (d_a, d_b, d_c, N);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float total_ms = 0;
        cudaEventElapsedTime(&total_ms, start, stop);
        float avg_ms = total_ms / 100.0f;

        double gflops = (2.0 * (double)N * N * N) / (avg_ms / 1000.0) / 1e9;
        cout << "[-] Time: " << avg_ms << " ms" << endl;
        cout << "[-] Performance: " << gflops << " GFLOPS" << endl;

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return 0;
}