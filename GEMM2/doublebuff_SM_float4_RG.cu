#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

// Tile dimensions for Shared Memory: 32x32 floats
#define TILE_DIM 32
#define RX 4 // Register tile width per thread
#define RY 4 // Register tile height per thread

__global__ void matrix_multiply_double_buffered_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
    float* __restrict__ matrix_c,
    int N
) {
    // Shared Memory size per block: 2 * 32 * 33 * 4 * 2 bytes = 16,896 bytes (~16.5 KB)
    __shared__ float sa[2][TILE_DIM][TILE_DIM + 1];
    __shared__ float sb[2][TILE_DIM][TILE_DIM + 1];

    int tx = threadIdx.x; // 0..7
    int ty = threadIdx.y; // 0..7

    int row_start = blockIdx.y * TILE_DIM + ty * RY;
    int col_start = blockIdx.x * TILE_DIM + tx * RX;

    float sum[RY][RX] = { 0.0f };
    int tid = ty * 8 + tx; // 64 threads total (0..63)

    // =========================================================================
    // PROLOGUE: Load Tile 0 into Buffer 0
    // =========================================================================
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int vec_id = tid + i * 64;          // 0..255 float4 vectors
        int load_row = vec_id / 8;          // Row inside tile (0..31)
        int load_col = (vec_id % 8) * 4;    // Column inside tile (0..28)

        int a_global_row = blockIdx.y * TILE_DIM + load_row;
        int a_global_col = 0 + load_col;

        if (a_global_row < N && a_global_col < N) {
            float4 a_val = reinterpret_cast<const float4*>(&matrix_a[a_global_row * N + a_global_col])[0];
            sa[0][load_row][load_col + 0] = a_val.x;
            sa[0][load_row][load_col + 1] = a_val.y;
            sa[0][load_row][load_col + 2] = a_val.z;
            sa[0][load_row][load_col + 3] = a_val.w;
        }
        else {
            sa[0][load_row][load_col + 0] = 0.0f;
            sa[0][load_row][load_col + 1] = 0.0f;
            sa[0][load_row][load_col + 2] = 0.0f;
            sa[0][load_row][load_col + 3] = 0.0f;
        }

        int b_global_row = 0 + load_row;
        int b_global_col = blockIdx.x * TILE_DIM + load_col;

        if (b_global_row < N && b_global_col < N) {
            float4 b_val = reinterpret_cast<const float4*>(&matrix_b[b_global_row * N + b_global_col])[0];
            sb[0][load_row][load_col + 0] = b_val.x;
            sb[0][load_row][load_col + 1] = b_val.y;
            sb[0][load_row][load_col + 2] = b_val.z;
            sb[0][load_row][load_col + 3] = b_val.w;
        }
        else {
            sb[0][load_row][load_col + 0] = 0.0f;
            sb[0][load_row][load_col + 1] = 0.0f;
            sb[0][load_row][load_col + 2] = 0.0f;
            sb[0][load_row][load_col + 3] = 0.0f;
        }
    }

    __syncthreads();

    // =========================================================================
    // PIPELINED MAIN LOOP
    // =========================================================================
    for (int ph = 0; ph < N; ph += TILE_DIM) {

        int read_idx = (ph / TILE_DIM) % 2;
        int write_idx = 1 - read_idx;
        int next_ph = ph + TILE_DIM;

        // Prefetch Tile k+1
        if (next_ph < N) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int vec_id = tid + i * 64;
                int load_row = vec_id / 8;
                int load_col = (vec_id % 8) * 4;

                int a_global_row = blockIdx.y * TILE_DIM + load_row;
                int a_global_col = next_ph + load_col;

                if (a_global_row < N && a_global_col < N) {
                    float4 a_val = reinterpret_cast<const float4*>(&matrix_a[a_global_row * N + a_global_col])[0];
                    sa[write_idx][load_row][load_col + 0] = a_val.x;
                    sa[write_idx][load_row][load_col + 1] = a_val.y;
                    sa[write_idx][load_row][load_col + 2] = a_val.z;
                    sa[write_idx][load_row][load_col + 3] = a_val.w;
                }
                else {
                    sa[write_idx][load_row][load_col + 0] = 0.0f;
                    sa[write_idx][load_row][load_col + 1] = 0.0f;
                    sa[write_idx][load_row][load_col + 2] = 0.0f;
                    sa[write_idx][load_row][load_col + 3] = 0.0f;
                }

                int b_global_row = next_ph + load_row;
                int b_global_col = blockIdx.x * TILE_DIM + load_col;

                if (b_global_row < N && b_global_col < N) {
                    float4 b_val = reinterpret_cast<const float4*>(&matrix_b[b_global_row * N + b_global_col])[0];
                    sb[write_idx][load_row][load_col + 0] = b_val.x;
                    sb[write_idx][load_row][load_col + 1] = b_val.y;
                    sb[write_idx][load_row][load_col + 2] = b_val.z;
                    sb[write_idx][load_row][load_col + 3] = b_val.w;
                }
                else {
                    sb[write_idx][load_row][load_col + 0] = 0.0f;
                    sb[write_idx][load_row][load_col + 1] = 0.0f;
                    sb[write_idx][load_row][load_col + 2] = 0.0f;
                    sb[write_idx][load_row][load_col + 3] = 0.0f;
                }
            }
        }

        // Compute Tile k
#pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            float reg_a[RY];
            float reg_b[RX];

#pragma unroll
            for (int i = 0; i < RY; ++i) {
                reg_a[i] = sa[read_idx][ty * RY + i][k];
            }

#pragma unroll
            for (int j = 0; j < RX; ++j) {
                reg_b[j] = sb[read_idx][k][tx * RX + j];
            }

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

    // Store Output
#pragma unroll
    for (int i = 0; i < RY; ++i) {
        int r = row_start + i;
        if (r < N && col_start < N) {
            float4 out_val = make_float4(sum[i][0], sum[i][1], sum[i][2], sum[i][3]);
            reinterpret_cast<float4*>(&matrix_c[r * N + col_start])[0] = out_val;
        }
    }
}

int main() {
    vector<int> test_sizes = { 512, 1024, 2048, 4096 };

    for (int N : test_sizes) {
        cout << "\n============================================\n";
        cout << "Benchmarking Double-Buffered GEMM (32x32 Tile): " << N << " x " << N << "\n";
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

        // 64 threads per block (8 x 8) for TILE_DIM = 32
        dim3 block_dim(8, 8);
        dim3 grid_dim((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        matrix_multiply_double_buffered_kernel << <grid_dim, block_dim >> > (d_a, d_b, d_c, N);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        matrix_multiply_double_buffered_kernel << <grid_dim, block_dim >> > (d_a, d_b, d_c, N);
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