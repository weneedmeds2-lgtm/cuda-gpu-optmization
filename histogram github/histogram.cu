#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using std::cout;
using std::endl;
using std::vector;

__global__ void naiveHistogramKernel(const int* inputArray, int* outputArray, int dataSize) {
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    while (threadId < dataSize) {
        int binValue = inputArray[threadId];
        atomicAdd(&outputArray[binValue], 1);
        threadId += gridStride;
    }
}
__global__ void sharedHistogramKernel(const int* inputArray, int* outputArray, int dataSize, int totalBins) {
    extern __shared__ int localHistogramBuffer[];

    for (int i = threadIdx.x; i < totalBins; i += blockDim.x) {
        localHistogramBuffer[i] = 0;
    }
    __syncthreads();

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    while (threadId < dataSize) {
        int binValue = inputArray[threadId];
        atomicAdd(&localHistogramBuffer[binValue], 1);
        threadId += gridStride;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < totalBins; i += blockDim.x) {
        atomicAdd(&outputArray[i], localHistogramBuffer[i]);
    }
}

__global__ void multiBlockHistogramKernel(const int* inputArray, int* outputArray, int dataSize, int totalBins) {
    extern __shared__ int localSubHistograms[];
    int totalCopies = 4;

    int totalSharedElements = totalBins * totalCopies;
    for (int i = threadIdx.x; i < totalSharedElements; i += blockDim.x) {
        localSubHistograms[i] = 0;
    }
    __syncthreads();

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;
    int assignedCopyId = threadIdx.x % totalCopies;

    while (threadId < dataSize) {
        int binValue = inputArray[threadId];
        int flatSharedIndex = (assignedCopyId * totalBins) + binValue;

        atomicAdd(&localSubHistograms[flatSharedIndex], 1);
        threadId += gridStride;
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < totalBins; bin += blockDim.x) {
        int combinedBinSum = 0;
        for (int copyNum = 0; copyNum < totalCopies; ++copyNum) {
            combinedBinSum += localSubHistograms[(copyNum * totalBins) + bin];
        }
        atomicAdd(&outputArray[bin], combinedBinSum);
    }
}

int main() {
    int totalBins = 256;
    vector<int> testSizes;
    testSizes.push_back(1000000);
    testSizes.push_back(10000000);
    testSizes.push_back(50000000);

    for (size_t t = 0; t < testSizes.size(); ++t) {
        size_t dataSize = testSizes[t];
        size_t inputBytes = dataSize * sizeof(int);
        size_t outputBytes = totalBins * sizeof(int);

        cout << "\n============================================\n";
        cout << "Benchmarking Histogram with Data Size: " << dataSize << " elements\n";
        cout << "============================================\n";

        vector<int> hostInput(dataSize);
        vector<int> hostOutputNaive(totalBins, 0);
        vector<int> hostOutputShared(totalBins, 0);
        vector<int> hostOutputMulti(totalBins, 0);

        for (int i = 0; i < dataSize; ++i) {
            hostInput[i] = rand() % totalBins;
        }

        int* deviceInput;
        int* deviceOutput;
        cudaMalloc(&deviceInput, inputBytes);
        cudaMalloc(&deviceOutput, outputBytes);

        cudaMemcpy(deviceInput, hostInput.data(), inputBytes, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (dataSize + threadsPerBlock - 1) / threadsPerBlock;
        if (blocksPerGrid > 240) {
            blocksPerGrid = 240;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0.0f;
        double throughput = 0.0;

        cudaMemset(deviceOutput, 0, outputBytes);
        cudaEventRecord(start);
        naiveHistogramKernel << <blocksPerGrid, threadsPerBlock >> > (deviceInput, deviceOutput, dataSize);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&milliseconds, start, stop);
        throughput = (inputBytes / (milliseconds / 1000.0)) / 1e9;
        cout << "[Naive Kernel] Time: " << milliseconds << " ms | Throughput: " << throughput << " GB/s\n";
        cudaMemcpy(hostOutputNaive.data(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost);

        cudaMemset(deviceOutput, 0, outputBytes);
        size_t sharedMemSize = totalBins * sizeof(int);

        cudaEventRecord(start);
        sharedHistogramKernel << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (deviceInput, deviceOutput, dataSize, totalBins);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&milliseconds, start, stop);
        throughput = (inputBytes / (milliseconds / 1000.0)) / 1e9;
        cout << "[Shared Kernel] Time: " << milliseconds << " ms | Throughput: " << throughput << " GB/s\n";
        cudaMemcpy(hostOutputShared.data(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost);

        cudaMemset(deviceOutput, 0, outputBytes);
        size_t multiSharedMemSize = totalBins * 4 * sizeof(int);

        cudaEventRecord(start);
        multiBlockHistogramKernel << <blocksPerGrid, threadsPerBlock, multiSharedMemSize >> > (deviceInput, deviceOutput, dataSize, totalBins);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&milliseconds, start, stop);
        throughput = (inputBytes / (milliseconds / 1000.0)) / 1e9;
        cout << "[Multi-Block Kernel] Time: " << milliseconds << " ms | Throughput: " << throughput << " GB/s\n";
        cudaMemcpy(hostOutputMulti.data(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost);

        bool match = true;
        for (int i = 0; i < totalBins; ++i) {
            if (hostOutputNaive[i] != hostOutputShared[i] || hostOutputNaive[i] != hostOutputMulti[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            cout << "[-] Verification: SUCCESS (All array outputs match exactly!)\n";
        }
        else {
            cout << "[!] Verification: FAILURE (Outputs mismatch! Check implementation details)\n";
        }

        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}