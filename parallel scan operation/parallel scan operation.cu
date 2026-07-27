#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// ============================================================================
// 1. NAIVE REDUCTION KERNEL (Causes Heavy Branch Divergence)
// ============================================================================
__global__ void naiveReductionKernel(const int* inputArray, int* outputArray, size_t dataSize) {
    extern __shared__ int sharedBuffer[];

    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalId < dataSize) {
        sharedBuffer[threadId] = inputArray[globalId];
    }
    else {
        sharedBuffer[threadId] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadId % (2 * stride) == 0) {
            sharedBuffer[threadId] += sharedBuffer[threadId + stride];
        }
        __syncthreads();
    }

    if (threadId == 0) {
        outputArray[blockIdx.x] = sharedBuffer[0];
    }
}

// ============================================================================
// 2. OPTIMIZED REDUCTION KERNEL (Sequential Addressing - Resolves Divergence)
// ============================================================================
__global__ void sequentialReductionKernel(const int* inputArray, int* outputArray, size_t dataSize) {
    extern __shared__ int sharedBuffer[];

    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalId < dataSize) {
        sharedBuffer[threadId] = inputArray[globalId];
    }
    else {
        sharedBuffer[threadId] = 0;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadId < stride) {
            sharedBuffer[threadId] += sharedBuffer[threadId + stride];
        }
        __syncthreads();
    }

    if (threadId == 0) {
        outputArray[blockIdx.x] = sharedBuffer[0];
    }
}

// ============================================================================
// 3. ADVANCED WARP REDUCTION KERNEL (Bypasses Shared Memory inside Final Warp)
// ============================================================================
__global__ void warpShuffleReductionKernel(const int* inputArray, int* outputArray, size_t dataSize) {
    extern __shared__ int sharedBuffer[];

    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalId < dataSize) {
        sharedBuffer[threadId] = inputArray[globalId];
    }
    else {
        sharedBuffer[threadId] = 0;
    }
    __syncthreads();

    // Changed condition from > 32 to >= 32 so stride 32 executes
    for (int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
        if (threadId < stride) {
            sharedBuffer[threadId] += sharedBuffer[threadId + stride];
        }
        __syncthreads();
    }

    if (threadId < 32) {
        int currentSum = sharedBuffer[threadId];
        currentSum += __shfl_down_sync(0xFFFFFFFF, currentSum, 16);
        currentSum += __shfl_down_sync(0xFFFFFFFF, currentSum, 8);
        currentSum += __shfl_down_sync(0xFFFFFFFF, currentSum, 4);
        currentSum += __shfl_down_sync(0xFFFFFFFF, currentSum, 2);
        currentSum += __shfl_down_sync(0xFFFFFFFF, currentSum, 1);

        if (threadId == 0) {
            outputArray[blockIdx.x] = currentSum;
        }
    }
}
// ============================================================================
// MAIN BENCHMARK HARNESS
// ============================================================================
int main() {
    vector<size_t> testSizes;
    testSizes.push_back(1000000);
    testSizes.push_back(10000000);
    testSizes.push_back(50000000);

    for (size_t t = 0; t < testSizes.size(); ++t) {
        size_t dataSize = testSizes[t];
        size_t inputBytes = dataSize * sizeof(int);

        int threadsPerBlock = 256;
        int blocksPerGrid = (static_cast<int>(dataSize) + threadsPerBlock - 1) / threadsPerBlock;
        size_t outputBytes = blocksPerGrid * sizeof(int);

        cout << "\n============================================\n";
        cout << "Benchmarking Reduction with Data Size: " << dataSize << " elements\n";
        cout << "============================================\n";

        vector<int> hostInput(dataSize);
        vector<int> hostOutputNaive(blocksPerGrid, 0);
        vector<int> hostOutputSequential(blocksPerGrid, 0);
        vector<int> hostOutputWarp(blocksPerGrid, 0);

        long long verificationTargetSum = 0;
        for (size_t i = 0; i < dataSize; ++i) {
            hostInput[i] = rand() % 10;
            verificationTargetSum += hostInput[i];
        }

        int* deviceInput;
        int* deviceOutput;
        cudaMalloc(&deviceInput, inputBytes);
        cudaMalloc(&deviceOutput, outputBytes);

        cudaMemcpy(deviceInput, hostInput.data(), inputBytes, cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0.0f;
        double throughput = 0.0;
        size_t sharedMemSize = threadsPerBlock * sizeof(int);

        // -------------------------------------------------------------
        // 1. Benchmarking Naive Kernel
        // -------------------------------------------------------------
        cudaMemset(deviceOutput, 0, outputBytes);
        cudaEventRecord(start);
        naiveReductionKernel << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (deviceInput, deviceOutput, dataSize);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&milliseconds, start, stop);
        throughput = (inputBytes / (milliseconds / 1000.0)) / 1e9;
        cout << "[Naive Kernel]      Time: " << milliseconds << " ms | Throughput: " << throughput << " GB/s\n";
        cudaMemcpy(hostOutputNaive.data(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost);

        // -------------------------------------------------------------
        // 2. Benchmarking Sequential Addressing Kernel
        // -------------------------------------------------------------
        cudaMemset(deviceOutput, 0, outputBytes);
        cudaEventRecord(start);
        sequentialReductionKernel << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (deviceInput, deviceOutput, dataSize);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&milliseconds, start, stop);
        throughput = (inputBytes / (milliseconds / 1000.0)) / 1e9;
        cout << "[Sequential Kernel] Time: " << milliseconds << " ms | Throughput: " << throughput << " GB/s\n";
        cudaMemcpy(hostOutputSequential.data(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost);

        // -------------------------------------------------------------
        // 3. Benchmarking Warp Shuffle Kernel
        // -------------------------------------------------------------
        cudaMemset(deviceOutput, 0, outputBytes);
        cudaEventRecord(start);
        warpShuffleReductionKernel << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (deviceInput, deviceOutput, dataSize);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&milliseconds, start, stop);
        throughput = (inputBytes / (milliseconds / 1000.0)) / 1e9;
        cout << "[Warp Shuffle]      Time: " << milliseconds << " ms | Throughput: " << throughput << " GB/s\n";
        cudaMemcpy(hostOutputWarp.data(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost);

        // -------------------------------------------------------------
        // Verification Step (Accumulating block sums on Host side)
        // -------------------------------------------------------------
        long long finalSumNaive = 0;
        long long finalSumSequential = 0;
        long long finalSumWarp = 0;

        for (int i = 0; i < blocksPerGrid; ++i) {
            finalSumNaive += hostOutputNaive[i];
            finalSumSequential += hostOutputSequential[i];
            finalSumWarp += hostOutputWarp[i];
        }

        if (finalSumNaive == verificationTargetSum &&
            finalSumSequential == verificationTargetSum &&
            finalSumWarp == verificationTargetSum) {
            cout << "[-] Verification: SUCCESS (All reduction sums match perfectly: " << finalSumWarp << ")\n";
        }
        else {
            cout << "[!] Verification: FAILURE (Mismatch detected! Host CPU: " << verificationTargetSum
                << " | Naive GPU: " << finalSumNaive << " | Sequential GPU: " << finalSumSequential
                << " | Warp GPU: " << finalSumWarp << ")\n";
        }

        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}