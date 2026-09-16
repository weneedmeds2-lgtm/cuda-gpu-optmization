// ===============================================
// GEMM BENCHMARK — consolidated, de-duplicated
// Naive            : one thread per output element, global memory only
// SM-Tiled         : shared-memory tiling, single-buffered (B pre-transposed)
// SM-Double-Buffer : shared-memory tiling with prefetch/compute overlap
// Warp-Optimized   : register-blocked 2x2-per-thread SM tiling
//
// Redundant near-duplicate kernels from the uploaded set were collapsed:
//   SMtransposeoptmized.cu / "SM optmized and transpose kerenel.txt" /
//   "MM optmizmed double buffering .txt" -> one Double-Buffer kernel.
//   warpoptmized_MM.cu (no actual shuffle use) -> dropped in favor of the
//   register-blocked 2x2 kernel ("2 per thread element...") as the true
//   warp/ILP-optimized variant.
//
// Verification: each optimized kernel's output is compared against the
// Naive kernel's output on the SAME random input (fp32 tolerance), since
// full CPU GEMM at N>=1024 is impractically slow.
// ===============================================
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
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

#define TILE_SIZE 32

// ===============================================
// Shared transpose kernel (used by all tiled variants)
// ===============================================
__global__ void transpose(float* matrix_b, int N, float* matrixbt) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    int globalrow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalcol = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (globalrow < N && globalcol < N)
        tile[threadIdx.y][threadIdx.x] = matrix_b[globalrow * N + globalcol];
    __syncthreads();

    int trow = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tcol = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (trow < N && tcol < N)
        matrixbt[tcol * N + trow] = tile[threadIdx.x][threadIdx.y];
}

// ===============================================
// NAIVE (global memory only) - also used as ground truth
// ===============================================
__global__ void matrix_multiply_naive_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ===============================================
// SM-TILED, single buffer (B pre-transposed for coalesced loads)
// ===============================================
__global__ void matrix_multiply_sm_tiled_kernel(float* matrix_a, float* matrix_bt, float* matrix_c, int N) {
    __shared__ float sa[TILE_SIZE][TILE_SIZE];
    __shared__ float sb[TILE_SIZE][TILE_SIZE];

    int globalrow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalcol = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < N; k += TILE_SIZE) {
        sa[threadIdx.y][threadIdx.x] = (globalrow < N && (k + threadIdx.x) < N)
            ? matrix_a[globalrow * N + (k + threadIdx.x)] : 0.0f;
        // matrix_bt is B transposed, so a row-read here is a column of B
        sb[threadIdx.y][threadIdx.x] = (globalcol < N && (k + threadIdx.y) < N)
            ? matrix_bt[globalcol * N + (k + threadIdx.y)] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int offset = 0; offset < TILE_SIZE; offset++)
            sum += sa[threadIdx.y][offset] * sb[offset][threadIdx.x];
        __syncthreads();
    }

    if (globalrow < N && globalcol < N)
        matrix_c[globalrow * N + globalcol] = sum;
}

// ===============================================
// SM-TILED, double buffer (prefetch next tile while computing current)
// ===============================================
__global__ void matrix_multiply_double_buffer_kernel(float* matrix_a, float* matrix_bt, float* matrix_c, int N) {
    __shared__ float tile_a[2][TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[2][TILE_SIZE][TILE_SIZE];
    int compute_idx = 0;

    int globalrow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalcol = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Prime buffer 0
    tile_a[0][threadIdx.y][threadIdx.x] = (globalrow < N && threadIdx.x < N)
        ? matrix_a[globalrow * N + threadIdx.x] : 0.0f;
    tile_b[0][threadIdx.y][threadIdx.x] = (globalcol < N && threadIdx.y < N)
        ? matrix_bt[globalcol * N + threadIdx.y] : 0.0f;
    __syncthreads();

    for (int t = 0; t < numTiles; t++) {
#pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tile_a[compute_idx][threadIdx.y][k] * tile_b[compute_idx][k][threadIdx.x];

        if (t + 1 < numTiles) {
            int local_idx = compute_idx ^ 1;
            int nextt = t + 1;
            tile_a[local_idx][threadIdx.y][threadIdx.x] =
                (globalrow < N && (nextt * TILE_SIZE + threadIdx.x) < N)
                ? matrix_a[globalrow * N + (nextt * TILE_SIZE + threadIdx.x)] : 0.0f;
            tile_b[local_idx][threadIdx.y][threadIdx.x] =
                (globalcol < N && (nextt * TILE_SIZE + threadIdx.y) < N)
                ? matrix_bt[globalcol * N + (nextt * TILE_SIZE + threadIdx.y)] : 0.0f;
        }
        __syncthreads();
        compute_idx ^= 1;
    }

    if (globalrow < N && globalcol < N)
        matrix_c[globalrow * N + globalcol] = sum;
}

// ===============================================
// WARP-OPTIMIZED: register-blocked 2x2 output per thread
// (Each thread does 4x the work per shared-mem load -> higher arithmetic intensity)
// ===============================================
__global__ void matrix_multiply_warp_kernel(float* matrix_a, float* matrix_bt, float* matrix_c, int N) {
    __shared__ float sa[TILE_SIZE][TILE_SIZE];
    __shared__ float sb[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y * 2;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x * 2;

    if (threadIdx.y >= TILE_SIZE / 2 || threadIdx.x >= TILE_SIZE / 2) return;

    float sum[2][2] = { {0.0f, 0.0f}, {0.0f, 0.0f} };

    for (int k = 0; k < N; k += TILE_SIZE) {
        sa[threadIdx.y * 2][threadIdx.x * 2] =
            (row < N && (k + threadIdx.x * 2) < N) ? matrix_a[row * N + (k + threadIdx.x * 2)] : 0.0f;
        sa[threadIdx.y * 2 + 1][threadIdx.x * 2] =
            ((row + 1) < N && (k + threadIdx.x * 2) < N) ? matrix_a[(row + 1) * N + (k + threadIdx.x * 2)] : 0.0f;
        sa[threadIdx.y * 2][threadIdx.x * 2 + 1] =
            (row < N && (k + threadIdx.x * 2 + 1) < N) ? matrix_a[row * N + (k + threadIdx.x * 2 + 1)] : 0.0f;
        sa[threadIdx.y * 2 + 1][threadIdx.x * 2 + 1] =
            ((row + 1) < N && (k + threadIdx.x * 2 + 1) < N) ? matrix_a[(row + 1) * N + (k + threadIdx.x * 2 + 1)] : 0.0f;

        // matrix_bt is transposed: row-major read of bt[col][k] == B[k][col]
        sb[threadIdx.y * 2][threadIdx.x * 2] =
            (col < N && (k + threadIdx.y * 2) < N) ? matrix_bt[col * N + (k + threadIdx.y * 2)] : 0.0f;
        sb[threadIdx.y * 2 + 1][threadIdx.x * 2] =
            (col < N && (k + threadIdx.y * 2 + 1) < N) ? matrix_bt[col * N + (k + threadIdx.y * 2 + 1)] : 0.0f;
        sb[threadIdx.y * 2][threadIdx.x * 2 + 1] =
            ((col + 1) < N && (k + threadIdx.y * 2) < N) ? matrix_bt[(col + 1) * N + (k + threadIdx.y * 2)] : 0.0f;
        sb[threadIdx.y * 2 + 1][threadIdx.x * 2 + 1] =
            ((col + 1) < N && (k + threadIdx.y * 2 + 1) < N) ? matrix_bt[(col + 1) * N + (k + threadIdx.y * 2 + 1)] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int offset = 0; offset < TILE_SIZE; offset++) {
            float valA0 = sa[threadIdx.y * 2][offset];
            float valA1 = sa[threadIdx.y * 2 + 1][offset];
            sum[0][0] += valA0 * sb[offset][threadIdx.x * 2];
            sum[0][1] += valA0 * sb[offset][threadIdx.x * 2 + 1];
            sum[1][0] += valA1 * sb[offset][threadIdx.x * 2];
            sum[1][1] += valA1 * sb[offset][threadIdx.x * 2 + 1];
        }
        __syncthreads();
    }

    if (row < N) {
        if (col < N)       matrix_c[row * N + col] = sum[0][0];
        if (col + 1 < N)   matrix_c[row * N + col + 1] = sum[0][1];
    }
    if (row + 1 < N) {
        if (col < N)       matrix_c[(row + 1) * N + col] = sum[1][0];
        if (col + 1 < N)   matrix_c[(row + 1) * N + col + 1] = sum[1][1];
    }
}
// NOTE: matrix_bt indexing above uses bt[col*N+k] (transpose of B), consistent
// with the sm-tiled/double-buffer kernels' bt[col][k] convention. This differs
// from the original uploaded file, which read matrix_b non-transposed with the
// same layout as A — that version only produced correct results because every
// test matrix was filled with 1.0f (any layout bug is invisible when all values
// are identical). Fixed here so it verifies against real random data.

// ===============================================
// Verification + metrics
// ===============================================
bool verifyClose(const vector<float>& a, const vector<float>& b, float relTol = 1e-2f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = fabsf(a[i] - b[i]);
        float scale = max(1.0f, fabsf(b[i]));
        if (diff / scale > relTol) return false;
    }
    return true;
}

struct Result {
    string algo;
    string impl;
    int N;
    float ms;
    double gflops;
    bool verified;
};

string getGPUName() {
    cudaDeviceProp prop;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    return string(prop.name);
}

string timestamp() {
    time_t t = time(nullptr);
    tm* lt = localtime(&t);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d_%H%M%S", lt);
    return string(buf);
}

void writeCSV(const vector<Result>& results, const string& gpuName, const string& path) {
    ofstream f(path);
    f << "gpu,algorithm,implementation,N,time_ms,gflops,verified\n";
    for (auto& r : results) {
        f << gpuName << "," << r.algo << "," << r.impl << "," << r.N << ","
          << fixed << setprecision(6) << r.ms << "," << setprecision(2) << r.gflops
          << "," << (r.verified ? "PASS" : "FAIL") << "\n";
    }
}

void writeMarkdown(const vector<Result>& results, const string& gpuName, const string& path) {
    ofstream f(path);
    f << "# GEMM Benchmark\n\nGPU: **" << gpuName << "**\n\n";
    f << "| Algorithm | Implementation | N | Time (ms) | GFLOPS | Verified |\n";
    f << "|---|---|---|---|---|---|\n";
    for (auto& r : results) {
        f << "| " << r.algo << " | " << r.impl << " | " << r.N << " | "
          << fixed << setprecision(4) << r.ms << " | " << setprecision(2) << r.gflops
          << " | " << (r.verified ? "PASS" : "FAIL") << " |\n";
    }
    f << "\n## Speedup vs Naive\n\n| N | SM-Tiled | Double-Buffer | Warp-Optimized |\n|---|---|---|---|\n";
    for (size_t i = 0; i + 3 < results.size(); i += 4) {
        float naiveMs = results[i].ms;
        f << "| " << results[i].N << " | "
          << fixed << setprecision(2) << (naiveMs / results[i+1].ms) << "x | "
          << (naiveMs / results[i+2].ms) << "x | "
          << (naiveMs / results[i+3].ms) << "x |\n";
    }
}

void printTable(const vector<Result>& results) {
    cout << "\n" << left << setw(16) << "Algorithm" << setw(24) << "Implementation"
         << setw(10) << "N" << setw(14) << "Time (ms)" << setw(14) << "GFLOPS"
         << setw(12) << "Verified" << "\n" << string(90, '-') << "\n";
    for (auto& r : results) {
        cout << left << setw(16) << r.algo << setw(24) << r.impl
             << setw(10) << r.N << setw(14) << fixed << setprecision(4) << r.ms
             << setw(14) << setprecision(2) << r.gflops
             << setw(12) << (r.verified ? "PASS" : "FAIL") << "\n";
    }
    cout << string(90, '-') << "\n\nSpeedup vs Naive:\n";
    for (size_t i = 0; i + 3 < results.size(); i += 4) {
        float naiveMs = results[i].ms;
        cout << "  N=" << results[i].N
             << "  SM: " << fixed << setprecision(2) << (naiveMs / results[i+1].ms) << "x"
             << "  DoubleBuf: " << (naiveMs / results[i+2].ms) << "x"
             << "  Warp: " << (naiveMs / results[i+3].ms) << "x\n";
    }
}

// ===============================================
// Benchmark driver: runs one kernel N_ITERS times, averages
// ===============================================
const int N_ITERS = 20;

float timeKernel(function<void()> launch) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    launch(); // warm-up
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

double computeGflops(int N, float ms) {
    return (2.0 * (double)N * N * N) / (ms / 1000.0) / 1e9;
}

int main(int argc, char** argv) {
    vector<int> sizes = {512, 1024, 2048};
    if (argc > 1) {
        sizes.clear();
        stringstream ss(argv[1]);
        string tok;
        while (getline(ss, tok, ',')) if (!tok.empty()) sizes.push_back(atoi(tok.c_str()));
    }

    string gpuName = getGPUName();
    cout << "GPU: " << gpuName << "\n";
    vector<Result> results;
    srand(42);

    for (int N : sizes) {
        cout << "\n=== N=" << N << " ===\n";
        size_t bytes = (size_t)N * N * sizeof(float);
        vector<float> h_a(N * N), h_b(N * N);
        for (int i = 0; i < N * N; i++) { h_a[i] = (rand() % 1000) / 100.0f; h_b[i] = (rand() % 1000) / 100.0f; }

        float *d_a, *d_b, *d_bt, *d_c, *d_c_ref;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_bt, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));
        CUDA_CHECK(cudaMalloc(&d_c_ref, bytes));
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

        dim3 block16(16, 16);
        dim3 grid16((N + 15) / 16, (N + 15) / 16);
        dim3 blockTile(TILE_SIZE, TILE_SIZE);
        dim3 gridTile((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

        // Ground truth: naive kernel
        float naiveMs = timeKernel([&]() {
            matrix_multiply_naive_kernel<<<grid16, block16>>>(d_a, d_b, d_c_ref, N);
        });
        vector<float> h_ref(N * N);
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_c_ref, bytes, cudaMemcpyDeviceToHost));
        results.push_back({"GEMM", "Naive (global mem)", N, naiveMs, computeGflops(N, naiveMs), true});

        // Pre-transpose B once (shared by all tiled kernels)
        transpose<<<gridTile, blockTile>>>(d_b, N, d_bt);
        cudaDeviceSynchronize();

        // SM-Tiled
        float smMs = timeKernel([&]() {
            matrix_multiply_sm_tiled_kernel<<<gridTile, blockTile>>>(d_a, d_bt, d_c, N);
        });
        vector<float> h_out(N * N);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        results.push_back({"GEMM", "SM-Tiled (single buf)", N, smMs, computeGflops(N, smMs), verifyClose(h_out, h_ref)});

        // Double Buffer
        float dbMs = timeKernel([&]() {
            matrix_multiply_double_buffer_kernel<<<gridTile, blockTile>>>(d_a, d_bt, d_c, N);
        });
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        results.push_back({"GEMM", "SM-Double-Buffer", N, dbMs, computeGflops(N, dbMs), verifyClose(h_out, h_ref)});

        // Warp-Optimized (register-blocked 2x2)
        dim3 blockWarp(TILE_SIZE / 2, TILE_SIZE / 2);
        float warpMs = timeKernel([&]() {
            matrix_multiply_warp_kernel<<<gridTile, blockWarp>>>(d_a, d_bt, d_c, N);
        });
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        results.push_back({"GEMM", "Warp-Optimized (2x2/thread)", N, warpMs, computeGflops(N, warpMs), verifyClose(h_out, h_ref)});

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_bt); cudaFree(d_c); cudaFree(d_c_ref);
    }

    printTable(results);

    system("mkdir -p results");
    string ts = timestamp();
    writeCSV(results, gpuName, "results/run_gemm_" + ts + ".csv");
    writeMarkdown(results, gpuName, "results/run_gemm_" + ts + ".md");
    cout << "\nSaved: results/run_gemm_" << ts << ".csv\nSaved: results/run_gemm_" << ts << ".md\n";
    return 0;
}
