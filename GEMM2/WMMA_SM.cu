#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h> // Header for FP16 half type
#include <mma.h>       // Header for Tensor Core WMMA API
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;
using namespace nvcuda;

// Tensor Core Tile Dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block Level Dimensions (32x32 output tile per block handled by 4 Warps)
#define BLOCK_DIM_M 32
#define BLOCK_DIM_N 32
#define CHUNK_K     16

// =========================================================================
// SHARED MEMORY + TENSOR CORE (WMMA) GEMM KERNEL 🏦⚡
// =========================================================================
__global__ void matrix_multiply_smem_wmma_kernel(
    const half* __restrict__ matrix_a,
    const half* __restrict__ matrix_b,
    float* __restrict__ matrix_c,
    int N
) {
    // 1. Declare Shared Memory tiles 🏦
    __shared__ half ash[BLOCK_DIM_M][CHUNK_K]; // 32 x 16 tile for A
    __shared__ half bsh[CHUNK_K][BLOCK_DIM_N]; // 16 x 32 tile for B

    // Linear thread ID within the block (128 threads total)
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Identify warp position within the block's 2x2 warp layout
    int warp_id = tid / 32;
    int warp_row = warp_id / 2; // 0 or 1
    int warp_col = warp_id % 2; // 0 or 1

    // Global block top-left starting corner
    int block_row = blockIdx.y * BLOCK_DIM_M;
    int block_col = blockIdx.x * BLOCK_DIM_N;

    // Declare WMMA Fragments 🧩
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Main K-loop stepping by CHUNK_K (16)
    for (int k = 0; k < N; k += CHUNK_K) {

        // 2. Cooperative Load from Global Memory into Shared Memory 📥
        // 128 threads load 512 elements (4 elements per thread)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int elem_idx = tid * 4 + i;

            // Load Tile A into Shared Memory
            int a_r = elem_idx / CHUNK_K;
            int a_c = elem_idx % CHUNK_K;
            int g_a_r = block_row + a_r;
            int g_a_c = k + a_c;

            if (g_a_r < N && g_a_c < N) {
                ash[a_r][a_c] = matrix_a[g_a_r * N + g_a_c];
            }
            else {
                ash[a_r][a_c] = __float2half(0.0f);
            }

            // Load Tile B into Shared Memory
            int b_r = elem_idx / BLOCK_DIM_N;
            int b_c = elem_idx % BLOCK_DIM_N;
            int g_b_r = k + b_r;
            int g_b_c = block_col + b_c;

            if (g_b_r < N && g_b_c < N) {
                bsh[b_r][b_c] = matrix_b[g_b_r * N + g_b_c];
            }
            else {
                bsh[b_r][b_c] = __float2half(0.0f);
            }
        }

        // Wait for all threads to finish loading into Shared Memory
        __syncthreads();

        // 3. Load Matrix Fragments from Shared Memory into Tensor Cores 🚀
        wmma::load_matrix_sync(a_frag, &ash[warp_row * WMMA_M][0], CHUNK_K);
        wmma::load_matrix_sync(b_frag, &bsh[0][warp_col * WMMA_N], BLOCK_DIM_N);

        // Execute Tensor Core Multiply & Accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Wait for computation to finish before overwriting Shared Memory in next loop step
        __syncthreads();
    }

    // 4. Store Output Block back to Global Memory
    int c_row = block_row + warp_row * WMMA_M;
    int c_col = block_col + warp_col * WMMA_N;

    if (c_row < N && c_col < N) {
        wmma::store_matrix_sync(matrix_c + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
    }
}

// =========================================================================
// HOST BENCHMARK CODE
// =========================================================================
int main() {
    vector<int> test_sizes = { 512, 1024, 2048, 4096, 8192 };

    for (int N : test_sizes) {
        cout << "\n============================================\n";
        cout << "Benchmarking SMEM + WMMA Tensor Core GEMM: " << N << " x " << N << "\n";
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

        // 128 threads per block (4 Warps arranged in a 2x2 block)
        dim3 block_dim(32, 4);
        dim3 grid_dim((N + BLOCK_DIM_N - 1) / BLOCK_DIM_N, (N + BLOCK_DIM_M - 1) / BLOCK_DIM_M);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warmup launch
        matrix_multiply_smem_wmma_kernel << <grid_dim, block_dim >> > (d_a, d_b, d_c, N);
        cudaDeviceSynchronize();

        // Benchmark launch
        cudaEventRecord(start);
        matrix_multiply_smem_wmma_kernel << <grid_dim, block_dim >> > (d_a, d_b, d_c, N);
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