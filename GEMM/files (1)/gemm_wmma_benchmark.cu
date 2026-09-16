// ===============================================
// GEMM BENCHMARK — Tensor Cores (WMMA) + cuBLAS reference
// WMMA   : hand-written FP16 tensor-core kernel (from uploaded WMMA_MM.cu)
// cuBLAS : vendor-tuned SGEMM, included as the "how close are we to the
//          ceiling" reference point (cublas_v2.h was already included in
//          the uploaded WMMA file but unused — wired it up here).
//
// Separate from gemm_benchmark.cu because WMMA requires half-precision
// staging and has architecture requirements (sm_70+) the other kernels
// don't, and because comparing against cuBLAS needs an extra link flag.
//
// Compile: nvcc -O3 -arch=sm_XX gemm_wmma_benchmark.cu -lcublas -o gemm_wmma_benchmark
// (sm_70 minimum for WMMA; run.sh below checks this)
// ===============================================
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mma.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <ctime>
#include <functional>
using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        cerr << "CUDA error " << cudaGetErrorString(err) \
             << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = (call); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        cerr << "cuBLAS error " << stat << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(1); \
    } \
} while (0)

const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
#define TILE_DIM 16

// ===============================================
// Naive fp32 kernel — ground truth for verification
// ===============================================
__global__ void matrix_multiply_naive_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ===============================================
// WMMA tensor-core kernel (unchanged from uploaded WMMA_MM.cu)
// ===============================================
__global__ void matrix_multiply_tensor_cores_kernel(float* matrix_a, float* matrix_b, float* matrix_c, int N) {
    __shared__ __half sh_a[TILE_DIM][TILE_DIM];
    __shared__ __half sh_b[TILE_DIM][TILE_DIM];

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);
    int row = warpM * WMMA_M;
    int col = warpN * WMMA_N;
    if (row >= N || col >= N) return;

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
                int gl_row_a = row + tile_r, gl_col_a = k + tile_c;
                sh_a[tile_r][tile_c] = (gl_row_a < N && gl_col_a < N)
                    ? (__half)matrix_a[gl_row_a * N + gl_col_a] : (__half)0.0f;
                int gl_row_b = k + tile_r, gl_col_b = col + tile_c;
                sh_b[tile_r][tile_c] = (gl_row_b < N && gl_col_b < N)
                    ? (__half)matrix_b[gl_row_b * N + gl_col_b] : (__half)0.0f;
            }
        }
        __syncthreads();
        nvcuda::wmma::load_matrix_sync(a_frag, (const __half*)sh_a, TILE_DIM);
        nvcuda::wmma::load_matrix_sync(b_frag, (const __half*)sh_b, TILE_DIM);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    nvcuda::wmma::store_matrix_sync(matrix_c + row * N + col, c_frag, N, nvcuda::wmma::mem_row_major);
}

// ===============================================
// Verification (loose tolerance for WMMA: fp16 staging loses precision)
// ===============================================
bool verifyClose(const vector<float>& a, const vector<float>& b, float relTol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = fabsf(a[i] - b[i]);
        float scale = max(1.0f, fabsf(b[i]));
        if (diff / scale > relTol) return false;
    }
    return true;
}

struct Result {
    string algo, impl;
    int N;
    float ms;
    double gflops;
    bool verified;
};

string getGPUName() {
    cudaDeviceProp prop; int dev = 0;
    cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
    return string(prop.name);
}
string timestamp() {
    time_t t = time(nullptr); tm* lt = localtime(&t);
    char buf[32]; strftime(buf, sizeof(buf), "%Y-%m-%d_%H%M%S", lt);
    return string(buf);
}
double computeGflops(int N, float ms) { return (2.0 * (double)N * N * N) / (ms / 1000.0) / 1e9; }

const int N_ITERS = 20;
float timeKernel(function<void()> launch) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    launch();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < N_ITERS; i++) launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total = 0;
    cudaEventElapsedTime(&total, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return total / N_ITERS;
}

void writeCSV(const vector<Result>& results, const string& gpuName, const string& path) {
    ofstream f(path);
    f << "gpu,algorithm,implementation,N,time_ms,gflops,verified\n";
    for (auto& r : results)
        f << gpuName << "," << r.algo << "," << r.impl << "," << r.N << ","
          << fixed << setprecision(6) << r.ms << "," << setprecision(2) << r.gflops
          << "," << (r.verified ? "PASS" : "FAIL") << "\n";
}
void writeMarkdown(const vector<Result>& results, const string& gpuName, const string& path) {
    ofstream f(path);
    f << "# Tensor-Core / cuBLAS GEMM Benchmark\n\nGPU: **" << gpuName << "**\n\n";
    f << "| Algorithm | Implementation | N | Time (ms) | GFLOPS | Verified |\n|---|---|---|---|---|---|\n";
    for (auto& r : results)
        f << "| " << r.algo << " | " << r.impl << " | " << r.N << " | "
          << fixed << setprecision(4) << r.ms << " | " << setprecision(2) << r.gflops
          << " | " << (r.verified ? "PASS" : "FAIL") << " |\n";
}
void printTable(const vector<Result>& results) {
    cout << "\n" << left << setw(14) << "Algorithm" << setw(20) << "Implementation"
         << setw(10) << "N" << setw(14) << "Time (ms)" << setw(14) << "GFLOPS"
         << setw(10) << "Verified" << "\n" << string(82, '-') << "\n";
    for (auto& r : results)
        cout << left << setw(14) << r.algo << setw(20) << r.impl
             << setw(10) << r.N << setw(14) << fixed << setprecision(4) << r.ms
             << setw(14) << setprecision(2) << r.gflops << setw(10) << (r.verified ? "PASS" : "FAIL") << "\n";
    cout << string(82, '-') << "\n";
}

int main(int argc, char** argv) {
    vector<int> sizes = {1024, 2048, 4096};
    if (argc > 1) {
        sizes.clear();
        stringstream ss(argv[1]); string tok;
        while (getline(ss, tok, ',')) if (!tok.empty()) sizes.push_back(atoi(tok.c_str()));
    }

    string gpuName = getGPUName();
    cout << "GPU: " << gpuName << "\n";

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    vector<Result> results;
    srand(42);

    for (int N : sizes) {
        // WMMA tile dims must divide evenly into 16 for this simple kernel
        cout << "\n=== N=" << N << " ===\n";
        size_t bytes = (size_t)N * N * sizeof(float);
        vector<float> h_a(N * N), h_b(N * N);
        for (int i = 0; i < N * N; i++) { h_a[i] = (rand() % 100) / 50.0f - 1.0f; h_b[i] = (rand() % 100) / 50.0f - 1.0f; }

        float *d_a, *d_b, *d_c, *d_c_ref;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));
        CUDA_CHECK(cudaMalloc(&d_c_ref, bytes));
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

        // Ground truth
        dim3 block16(16, 16), grid16((N + 15) / 16, (N + 15) / 16);
        float naiveMs = timeKernel([&]() {
            matrix_multiply_naive_kernel<<<grid16, block16>>>(d_a, d_b, d_c_ref, N);
        });
        vector<float> h_ref(N * N);
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_c_ref, bytes, cudaMemcpyDeviceToHost));
        results.push_back({"GEMM", "Naive fp32 (ref)", N, naiveMs, computeGflops(N, naiveMs), true});

        // cuBLAS SGEMM (vendor-tuned reference ceiling)
        // Note: cuBLAS is column-major; for row-major C=A*B we compute C^T = B^T*A^T,
        // which is equivalent to calling sgemm with A and B swapped, no explicit transpose needed.
        const float alpha = 1.0f, beta = 0.0f;
        float cublasMs = timeKernel([&]() {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                        &alpha, d_b, N, d_a, N, &beta, d_c, N);
        });
        vector<float> h_out(N * N);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        results.push_back({"GEMM", "cuBLAS SGEMM", N, cublasMs, computeGflops(N, cublasMs), verifyClose(h_out, h_ref, 1e-2f)});

        // WMMA tensor cores (needs N a multiple of 16 for this simple kernel; pad check)
        if (N % 16 == 0) {
            dim3 blockWmma(32, 2);
            dim3 gridWmma((N + (WMMA_N * 2) - 1) / (WMMA_N * 2), (N + WMMA_M - 1) / WMMA_M);
            float wmmaMs = timeKernel([&]() {
                matrix_multiply_tensor_cores_kernel<<<gridWmma, blockWmma>>>(d_a, d_b, d_c, N);
            });
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes, cudaMemcpyDeviceToHost));
            // Looser tolerance: fp16 staging of inputs loses ~3 decimal digits of precision
            results.push_back({"GEMM", "WMMA (tensor cores, fp16)", N, wmmaMs, computeGflops(N, wmmaMs), verifyClose(h_out, h_ref, 5e-2f)});
        } else {
            cout << "Skipping WMMA for N=" << N << " (not a multiple of 16)\n";
        }

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_c_ref);
    }

    cublasDestroy(handle);
    printTable(results);

    system("mkdir -p results");
    string ts = timestamp();
    writeCSV(results, gpuName, "results/run_wmma_" + ts + ".csv");
    writeMarkdown(results, gpuName, "results/run_wmma_" + ts + ".md");
    cout << "\nSaved: results/run_wmma_" << ts << ".csv\nSaved: results/run_wmma_" << ts << ".md\n";
    return 0;
}
