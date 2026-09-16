#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

// Tile size for output matrix region: 32x32 floats
#define TILE_DIM 32 

// -----------------------------------------------------------------------------
// TRANSPOSE KERNEL (Using float4 vectorized loads/writes)
// Block dim: (8, 32) threads -> covers (32, 32) elements
// -----------------------------------------------------------------------------
__global__ void transpose_vectorized(const float* __restrict__ matrix_b, int N, float* __restrict__ matrix_bt) {
    // Add +1 padding to avoid shared memory bank conflicts when reading column-wise
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int tx = threadIdx.x; // 0..7
    int ty = threadIdx.y; // 0..31

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + (tx * 4);

    // Vector load 4 floats from Matrix B
    if (row < N && col < N) {
        float4 val = reinterpret_cast<const float4*>(&matrix_b[row * N + col])[0];
        tile[ty][tx * 4 + 0] = val.x;
        tile[ty][tx * 4 + 1] = val.y;
        tile[ty][tx * 4 + 2] = val.z;
        tile[ty][tx * 4 + 3] = val.w;
    }

    __syncthreads();

    // Transposed write back to global memory using float4
    int t_row = blockIdx.x * TILE_DIM + ty;
    int t_col = blockIdx.y * TILE_DIM + (tx * 4);

    if (t_row < N && t_col < N) {
        float4 t_val;
        t_val.x = tile[tx * 4 + 0][ty];
        t_val.y = tile[tx * 4 + 1][ty];
        t_val.z = tile[tx * 4 + 2][ty];
        t_val.w = tile[tx * 4 + 3][ty];

        reinterpret_cast<float4*>(&matrix_bt[t_row * N + t_col])[0] = t_val;
    }
}

// -----------------------------------------------------------------------------
// MATRIX MULTIPLY KERNEL (float4 Vectorized)
// Each thread computes 4 contiguous output elements in matrix_c
// -----------------------------------------------------------------------------
__global__ void matrix_multiply_vectorized_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_bt,
    float* __restrict__ matrix_c,
    int N
) {
    // Add +1 padding to eliminate shared memory bank conflicts
    __shared__ float sa[TILE_DIM][TILE_DIM + 1];
    __shared__ float sb[TILE_DIM][TILE_DIM + 1];

    int tx = threadIdx.x; // 0..7
    int ty = threadIdx.y; // 0..31

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + (tx * 4);

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int ph = 0; ph < N; ph += TILE_DIM) {
        // Coalesced float4 load of Matrix A
        if (row < N && (ph + tx * 4) < N) {
            float4 a_val = reinterpret_cast<const float4*>(&matrix_a[row * N + ph + tx * 4])[0];
            sa[ty][tx * 4 + 0] = a_val.x;
            sa[ty][tx * 4 + 1] = a_val.y;
            sa[ty][tx * 4 + 2] = a_val.z;
            sa[ty][tx * 4 + 3] = a_val.w;
        }
        else {
            sa[ty][tx * 4 + 0] = 0.0f;
            sa[ty][tx * 4 + 1] = 0.0f;
            sa[ty][tx * 4 + 2] = 0.0f;
            sa[ty][tx * 4 + 3] = 0.0f;
        }

        // Coalesced float4 load of Transposed Matrix B (Fixed Row/Col indexing)
        int b_row = ph + ty;
        if (b_row < N && col < N) {
            float4 b_val = reinterpret_cast<const float4*>(&matrix_bt[b_row * N + col])[0];
            sb[ty][tx * 4 + 0] = b_val.x;
            sb[ty][tx * 4 + 1] = b_val.y;
            sb[ty][tx * 4 + 2] = b_val.z;
            sb[ty][tx * 4 + 3] = b_val.w;
        }
        else {
            sb[ty][tx * 4 + 0] = 0.0f;
            sb[ty][tx * 4 + 1] = 0.0f;
            sb[ty][tx * 4 + 2] = 0.0f;
            sb[ty][tx * 4 + 3] = 0.0f;
        }

        __syncthreads();

        // Compute phase
#pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_elem = sa[ty][k];
            sum.x += a_elem * sb[k][tx * 4 + 0];
            sum.y += a_elem * sb[k][tx * 4 + 1];
            sum.z += a_elem * sb[k][tx * 4 + 2];
            sum.w += a_elem * sb[k][tx * 4 + 3];
        }

        __syncthreads();
    }

    // Write out results
    if (row < N && col < N) {
        reinterpret_cast<float4*>(&matrix_c[row * N + col])[0] = sum;
    }
}

// -----------------------------------------------------------------------------
// MAIN BENCHMARK DRIVER
// -----------------------------------------------------------------------------
int main() {
    vector<int> test_sizes = { 512, 1024, 2048, 4096 };

    for (int N : test_sizes) {
        cout << "\n============================================\n";
        cout << "Benchmarking Vectorized (float4) GEMM: " << N << " x " << N << "\n";
        cout << "============================================\n";

        size_t total_elements = (size_t)N * N;
        size_t matrix_bytes = total_elements * sizeof(float);

        vector<float> h_a(total_elements, 1.0f);
        vector<float> h_b(total_elements, 2.0f);
        vector<float> h_c(total_elements, 0.0f);

        float* d_a, * d_b, * d_c, * d_bt;
        cudaMalloc(&d_a, matrix_bytes);
        cudaMalloc(&d_b, matrix_bytes);
        cudaMalloc(&d_c, matrix_bytes);
        cudaMalloc(&d_bt, matrix_bytes);

        cudaMemcpy(d_a, h_a.data(), matrix_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), matrix_bytes, cudaMemcpyHostToDevice);

        // Grid setup: Block dimensions are (8, 32) because each thread handles 4 floats horizontally
        dim3 block_dim(TILE_DIM / 4, TILE_DIM);
        dim3 grid_dim((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warmup
        transpose_vectorized << <grid_dim, block_dim >> > (d_b, N, d_bt);
        matrix_multiply_vectorized_kernel << <grid_dim, block_dim >> > (d_a, d_bt, d_c, N);
        cudaDeviceSynchronize();

        // Timed GEMM execution
        cudaEventRecord(start);
        matrix_multiply_vectorized_kernel << <grid_dim, block_dim >> > (d_a, d_bt, d_c, N);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

        double total_flops = 2.0 * (double)N * (double)N * (double)N;
        double gflops = (total_flops / 1e9) / (milliseconds / 1000.0);

        cout << "[-] Time        : " << fixed << setprecision(4) << milliseconds << " ms" << endl;
        cout << "[-] Performance : " << setprecision(2) << gflops << " GFLOPS" << endl;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_bt);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    return 0;
}