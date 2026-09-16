#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

// Tile dimensions for Shared Memory: 64x64 floats
#define TILE_DIM 64
#define RX 4 // Register tile width per thread
#define RY 4 // Register tile height per thread

// -----------------------------------------------------------------------------
// OPTIMIZED GEMM KERNEL (float4 Vectorized + 2D Register Tiling)
// - Block layout: (16, 16) = 256 threads
// - Output computed per block: 64x64 floats
// - Output computed per thread: 4x4 register tile (16 floats)
// -----------------------------------------------------------------------------
__global__ void matrix_multiply_vectorized_tiled_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
    float* __restrict__ matrix_c,
    int N
) {
    // Shared Memory tiles with +1 padding to avoid bank conflicts
    __shared__ float sa[TILE_DIM][TILE_DIM + 1];
    __shared__ float sb[TILE_DIM][TILE_DIM + 1];

    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    // 2D output tile indices computed by this thread (4x4 floats per thread)
    int row_start = blockIdx.y * TILE_DIM + ty * RY;
    int col_start = blockIdx.x * TILE_DIM + tx * RX;

    // Accumulators in registers for 4x4 output elements
    float sum[RY][RX] = { 0.0f };

    // Linear thread ID for memory loading (256 threads total)
    int tid = ty * 16 + tx;

    // Loop over k-dimension tiles
    for (int ph = 0; ph < N; ph += TILE_DIM) {

        // ---------------------------------------------------------------------
        // 1. Vectorized float4 Loads into Shared Memory (No Transpose Needed)
        // 256 threads load a 64x64 tile (4096 floats = 1024 float4s) -> 4 float4s/thread
        // ---------------------------------------------------------------------
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int vec_id = tid + i * 256;         // 0..1023 float4 vectors
            int load_row = vec_id / 16;          // Row inside tile (0..63)
            int load_col = (vec_id % 16) * 4;   // Column inside tile (0..60)

            // Global row/col for Matrix A
            int a_global_row = blockIdx.y * TILE_DIM + load_row;
            int a_global_col = ph + load_col;

            if (a_global_row < N && a_global_col < N) {
                float4 a_val = reinterpret_cast<const float4*>(&matrix_a[a_global_row * N + a_global_col])[0];
                sa[load_row][load_col + 0] = a_val.x;
                sa[load_row][load_col + 1] = a_val.y;
                sa[load_row][load_col + 2] = a_val.z;
                sa[load_row][load_col + 3] = a_val.w;
            }
            else {
                sa[load_row][load_col + 0] = 0.0f;
                sa[load_row][load_col + 1] = 0.0f;
                sa[load_row][load_col + 2] = 0.0f;
                sa[load_row][load_col + 3] = 0.0f;
            }

            // Global row/col for Matrix B (Direct Row-Major Load!)
            int b_global_row = ph + load_row;
            int b_global_col = blockIdx.x * TILE_DIM + load_col;

            if (b_global_row < N && b_global_col < N) {
                float4 b_val = reinterpret_cast<const float4*>(&matrix_b[b_global_row * N + b_global_col])[0];
                sb[load_row][load_col + 0] = b_val.x;
                sb[load_row][load_col + 1] = b_val.y;
                sb[load_row][load_col + 2] = b_val.z;
                sb[load_row][load_col + 3] = b_val.w;
            }
            else {
                sb[load_row][load_col + 0] = 0.0f;
                sb[load_row][load_col + 1] = 0.0f;
                sb[load_row][load_col + 2] = 0.0f;
                sb[load_row][load_col + 3] = 0.0f;
            }
        }

        __syncthreads();

        // ---------------------------------------------------------------------
        // 2. Compute Phase using Register Tiling
        // ---------------------------------------------------------------------
#pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            float reg_a[RY];
            float reg_b[RX];

            // Load 4 elements from sa and 4 elements from sb into registers
#pragma unroll
            for (int i = 0; i < RY; ++i) {
                reg_a[i] = sa[ty * RY + i][k];
            }

#pragma unroll
            for (int j = 0; j < RX; ++j) {
                reg_b[j] = sb[k][tx * RX + j];
            }

            // Perform 4x4 outer product in registers
#pragma unroll
            for (int i = 0; i < RY; ++i) {
#pragma unroll
                for (int j = 0; j < RX; ++j) {
                    sum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // 3. Write Back Results using float4 Vector Stores
    // -------------------------------------------------------------------------
#pragma unroll
    for (int i = 0; i < RY; ++i) {
        int r = row_start + i;
        if (r < N && col_start < N) {
            float4 out_val = make_float4(sum[i][0], sum[i][1], sum[i][2], sum[i][3]);
            reinterpret_cast<float4*>(&matrix_c[r * N + col_start])[0] = out_val;
        }
    }
}

// -----------------------------------------------------------------------------
// MAIN BENCHMARK DRIVER
// -----------------------------------------------------------------------------
int main() {
    vector<int> test_sizes = { 512, 1024, 2048, 4096 };

    for (int N : test_sizes) {
        cout << "\n============================================\n";
        cout << "Benchmarking Tiled + Vectorized (float4) GEMM: " << N << " x " << N << "\n";
        cout << "============================================\n";

        size_t total_elements = (size_t)N * N;
        size_t matrix_bytes = total_elements * sizeof(float);

        vector<float> h_a(total_elements, 1.0f);
        vector<float> h_b(total_elements, 2.0f);
        vector<float> h_c(total_elements, 0.0f);

        float* d_a, * d_b, * d_c;
        cudaMalloc(&d_a, matrix_bytes);
        cudaMalloc(&d_b, matrix_bytes);
        cudaMalloc(&d_c, matrix_bytes);

        cudaMemcpy(d_a, h_a.data(), matrix_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), matrix_bytes, cudaMemcpyHostToDevice);

        // 256 threads per block (16 x 16)
        dim3 block_dim(16, 16);
        // Grid dimension based on 64x64 tile size
        dim3 grid_dim((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warmup
        matrix_multiply_vectorized_tiled_kernel << <grid_dim, block_dim >> > (d_a, d_b, d_c, N);
        cudaDeviceSynchronize();

        // Timed GEMM execution
        cudaEventRecord(start);
        matrix_multiply_vectorized_tiled_kernel << <grid_dim, block_dim >> > (d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

        double total_flops = 2.0 * (double)N * (double)N * (double)N;
        double gflops = (total_flops / 1e9) / (milliseconds / 1000.0);

        cout << "[-] Time        : " << fixed << setprecision(4) << milliseconds << " ms" << endl;
        cout << "[-] Performance : " << setprecision(2) << gflops << " GFLOPS" << endl;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    return 0;
}