#include <device_launch_parameters.h>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;


__global__ void multiWarpScan(int* da, int* gblock, int N) {
	extern __shared__ int blocksum[];
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + tid;
	int warpid = tid / 32;
	int laneid = tid % 32;

	int val = 0;
	if (gid < N) val = da[gid];

	unsigned mask = 0xffffffff;

	// Warp inclusive scan using shuffle
	for (int offset = 1; offset < 32; offset <<= 1) {
		int n = __shfl_up_sync(mask, val, offset);
		if (laneid >= offset) val += n;
	}

	if (laneid == 31) blocksum[warpid] = val;

	__syncthreads();

	if (warpid == 0) {
		int warpVal = 0;
		int warpCount = (blockDim.x + 31) / 32;  // Added missing variable for clarity

		if (laneid < warpCount)
			warpVal = blocksum[laneid];

		for (int offset = 1; offset < 32; offset <<= 1) {
			int n = __shfl_up_sync(mask, warpVal, offset);
			if (laneid >= offset) warpVal += n;
		}

		if (laneid < warpCount)
			blocksum[laneid] = warpVal;
	}

	__syncthreads();

	if (warpid > 0) val += blocksum[warpid - 1];

	int exclusive;
	if (laneid == 0) {
		if (warpid == 0) {
			exclusive = 0;
		}
		else {
			exclusive = blocksum[warpid - 1];
		}
	}
	else {
		exclusive = __shfl_up_sync(mask, val, 1);
	}
	if (gid < N)
		da[gid] = exclusive;

	if (tid == blockDim.x - 1)
		gblock[blockIdx.x] = val;
}


__global__ void bellochscan(int* gblock, int N) {
	extern __shared__ int smscan[];
	int tid = threadIdx.x;
	int right, left;
	if (tid < N) {
		smscan[tid] = gblock[tid];
	}
	__syncthreads();
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		right = (tid + 1) * (2 * stride) - 1;
		left = right - stride;
		if (right < N) {
			smscan[right] = smscan[right] + smscan[left];
		}
		__syncthreads();
	}
	if (tid == 0) {
		smscan[N - 1] = 0;
	}
	__syncthreads();

	for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
		right = (tid + 1) * (2 * stride) - 1;
		left = right - stride;
		if (right < N) {
			int t = smscan[left];
			smscan[left] = smscan[right];
			smscan[right] = smscan[right] + t;
		}
		__syncthreads();
	}
	if (tid < N) {
		gblock[tid] = smscan[tid];
	}
	__syncthreads();

}


__global__ void addoffset(int* da, int* gblock, int N) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < N && blockIdx.x > 0) {
		da[gid] += gblock[blockIdx.x];
	}
}


int main() {

	int N = 1 << 20;
	int* da;
	int paddedN = 1;
	while (paddedN < N) {
		paddedN *= 2;
	}
	int* inputarr = new int[paddedN];
	for (int i = 0; i < N; i++) {
		inputarr[i] = rand() % 100;
	}
	for (int i = N; i < paddedN; i++) {
		inputarr[i] = 0;
	}
	int* gblock, * blocksum;
	cudaMalloc(&da, paddedN * sizeof(int));
	cudaMalloc(&blocksum, paddedN * sizeof(int));
	int threadsPerBlock = 256; // Must be multiple of 32
	int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
	cudaMalloc(&gblock, blocks * sizeof(int));
	cudaMemcpy(da, inputarr, paddedN * sizeof(int), cudaMemcpyHostToDevice);

	multiWarpScan << <blocks, threadsPerBlock, sizeof(int)* ((threadsPerBlock + 31) / 32) >> > (da, gblock, N);

	cudaMemcpy(blocksum, gblock, blocks * sizeof(int), cudaMemcpyDeviceToHost);

	bellochscan << <1, blocks, sizeof(int)* blocks >> > (gblock, blocks);

	addoffset << <blocks, threadsPerBlock >> > (da, gblock, N);
	cudaDeviceSynchronize();
	cudaMemcpy(inputarr, da, N * sizeof(int), cudaMemcpyDeviceToHost);

	cout << "Exclusive Scan Result:\n";
	for (int i = 0; i < 100; i++)
		cout << inputarr[i] << " ";
	cout << endl;

	cudaFree(da);
	cudaFree(gblock);
	delete[] inputarr;
	return 0;
}
