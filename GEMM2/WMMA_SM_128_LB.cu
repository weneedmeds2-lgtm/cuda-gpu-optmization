#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;
using namespace nvcuda;

// WMMA Hardware Tile Dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block Tile Dimensions (128x128)
#define BLOCK_DIM_M 128
#define BLOCK_DIM_N 128
#define CHUNK_K     16

// Helper function to load 128x16 A and 16x128 B into shared memory
__device__ void load_tile_to_smem_128x128(
    const half* __restrict__ matrix_a,
    const half* __restrict__ matrix_b,
    half ash[BLOCK_DIM_M][CHUNK_K],
    half bsh[CHUNK_K][BLOCK_DIM_N],
    int block_row, int block_col, int k, int N, int tid
) {
    // 512 threads load 2048 halfs for A and 2048 halfs for B
    // Exactly 4 elements per thread for A, and 4 for B
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int elem_idx = tid * 4 + i;

        // Tile A: 128 rows x 16 cols
        int a_r = elem_idx / CHUNK_K;
        int a_c = elem_idx % CHUNK_K;
        int g_a_r = block_row + a_r;
        int g_a_c = k + a_c;
        ash[a_r][a_c] = (g_a_r < N && g_a_c < N) ? matrix_a[g_a_r * N + g_a_c] : __float2half(0.0f);

        // Tile B: 16 rows x 128 cols
        int b_r = elem_idx / BLOCK_DIM_N;
        int b_c = elem_idx % BLOCK_DIM_N;
        int g_b_r = k + b_r;
        int g_b_c = block_col + b_c;
        bsh[b_r][b_c] = (g_b_r < N && g_b_c < N) ? matrix_b[g_b_r * N + g_b_c] : __float2half(0.0f);
    }
}

// =========================================================================
// 128x128 DOUBLE-BUFFERED SMEM + WMMA GEMM KERNEL 🏦⚡
// Launch bounds restrict per-thread registers to preserve occupancy
// =========================================================================
__launch_bounds__(512, 1)
__global__ void matrix_multiply_128x128_wmma_kernel(
    const half* __restrict__ matrix_a,
    const half* __restrict__ matrix_b,
    float* __restrict__ matrix_c,
    int N
) {
    // 1. Double Buffers in SMEM (16 KB Total)
    __shared__ half ash[2][BLOCK_DIM_M][CHUNK_K];
    __shared__ half bsh[2][CHUNK_K][BLOCK_DIM_N];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int warp_row = warp_id / 4; // 16 warps arranged in a 4x4 grid
    int warp_col = warp_id % 4;

    int block_row = blockIdx.y * BLOCK_DIM_M;
    int block_col = blockIdx.x * BLOCK_DIM_N;

    // Each warp computes a 32x32 sub-tile (2x2 WMMA fragments)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];

#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int read_buf = 0;
    int write_buf = 1;

    // Preamble: Load first tile into Buffer 0
    load_tile_to_smem_128x128(matrix_a, matrix_b, ash[read_buf], bsh[read_buf], block_row, block_col, 0, N, tid);
    __syncthreads();

    // Main Pipelined Loop
    for (int k = 0; k < N; k += CHUNK_K) {
        int next_k = k + CHUNK_K;

        // Async pre-fetch into Write Buffer
        if (next_k < N) {
            load_tile_to_smem_128x128(matrix_a, matrix_b, ash[write_buf], bsh[write_buf], block_row, block_col, next_k, N, tid);
        }

        // Load WMMA fragments for warp's 2x2 sub-tile
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            wmma::load_matrix_sync(a_frag[i], &ash[read_buf][warp_row * 32 + i * WMMA_M][0], CHUNK_K);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::load_matrix_sync(b_frag[j], &bsh[read_buf][0][warp_col * 32 + j * WMMA_N], BLOCK_DIM_N);
        }

        // Multiply-accumulate
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }

        __syncthreads();
        read_buf ^= 1;
        write_buf ^= 1;
    }

    // Store fragments to Global Memory C
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            int c_row = block_row + warp_row * 32 + i * WMMA_M;
            int c_col = block_col + warp_col * 32 + j * WMMA_N;

            if (c_row < N && c_col < N) {
                wmma::store_matrix_sync(matrix_c + c_row * N + c_col, c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

int main() {
    vector<int> test_sizes = { 1024, 2048, 4096, 8192 };

    for (int N : test_sizes) {
        cout << "\n============================================\n";
        cout << "Benchmarking 128x128 Double-Buffered SMEM + WMMA GEMM: " << N << " x " << N << "\n";
        cout << "============================================\n";

        size_t total_elements = (size_t)N * N;
        size_t half_bytes = total_elements * sizeof(half);
        size_t float_bytes = total_elements * sizeof(float);

        vector<half> h_a(total_elements, __float2half(1.0f));
        vector<half> h_b(total_elements, __float2half(2.0f));

        half* d_a, * d_b;
        float* d_c;
        cudaMalloc(&d_a, half_bytes);
        cudaMalloc(&d_b, half_bytes);
        cudaMalloc(&d_c, float_bytes);

        cudaMemcpy(d_a, h_a.data(), half_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), half_bytes, cudaMemcpyHostToDevice);

        // 512 threads per block (16 warps in 32x16 block configuration)
        dim3 block_dim(32, 16);
        dim3 grid_dim((N + BLOCK_DIM_N - 1) / BLOCK_DIM_N, (N + BLOCK_DIM_M - 1) / BLOCK_DIM_M);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        matrix_multiply_128x128_wmma_kernel << <grid_dim, block_dim >> > (d_a, d_b, d_c, N);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        matrix_multiply_128x128_wmma_kernel << <grid_dim, block_dim >> > (d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

        double total_flops = 2.0 * (double)N * (double)N * (double)N;
        double tflops = (total_flops / 1e12) / (milliseconds / 1000.0);

        cout << "[-] Time        : " << fixed << setprecision(4) << milliseconds << " ms" << endl;
        cout << "[-] Performance : " << setprecision(2) << tflops << " TFLOPS" << endl;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    return 0;
}