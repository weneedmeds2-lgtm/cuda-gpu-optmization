# GEMM2 CUDA Matrix Multiplication Experiments

This directory contains standalone CUDA benchmarks for square matrix multiplication (GEMM). The implementations explore vectorized memory access, shared-memory tiling, register tiling, double buffering, and NVIDIA WMMA Tensor Core programming.

Each `.cu` file has its own `main()` function and is compiled and run independently.

## Implementations

| Source file | Data type | Technique | Output tile |
| --- | --- | --- | --- |
| `float4_MM.cu` | FP32 | `float4` vectorized loads/stores, padded shared memory, and a separate transpose of B | 32 x 32 |
| `float4_2Dregistertiling_MM.cu` | FP32 | `float4` vectorized loads/stores plus 4 x 4 per-thread register tiling | 64 x 64 |
| `doublebuff_SM_float4_RG.cu` | FP32 | Two shared-memory buffers, `float4` transfers, and 4 x 4 per-thread register tiling | 32 x 32 |
| `WMMA_SM.cu` | FP16 inputs, FP32 output | Shared-memory staging with 16 x 16 x 16 WMMA fragments | 32 x 32 |
| `WMMA_SM_128_LB.cu` | FP16 inputs, FP32 output | 128 x 128 block tile, double-buffered shared memory, and WMMA fragments | 128 x 128 |

## Requirements

- NVIDIA GPU and driver
- CUDA Toolkit with `nvcc`
- A C++17-capable host compiler
- For the WMMA programs: an NVIDIA GPU with Tensor Core support (compile for `sm_70` or an appropriate newer architecture)

The FP32 programs use `float4` reinterpret casts. The included benchmark sizes are multiples of four, which keeps those vector accesses aligned. If you add arbitrary sizes, update the tail handling before relying on the results.

## Build and run

Run these commands from this directory. Replace `sm_80` with your GPU architecture when needed.

```powershell
nvcc -O3 -arch=sm_80 float4_MM.cu -o float4_MM.exe
.\float4_MM.exe

nvcc -O3 -arch=sm_80 float4_2Dregistertiling_MM.cu -o float4_2Dregistertiling_MM.exe
.\float4_2Dregistertiling_MM.exe

nvcc -O3 -arch=sm_80 doublebuff_SM_float4_RG.cu -o doublebuff_SM_float4_RG.exe
.\doublebuff_SM_float4_RG.exe

nvcc -O3 -arch=sm_80 WMMA_SM.cu -o WMMA_SM.exe
.\WMMA_SM.exe

nvcc -O3 -arch=sm_80 WMMA_SM_128_LB.cu -o WMMA_SM_128_LB.exe
.\WMMA_SM_128_LB.exe
```

## Benchmark methodology

The benchmark drivers allocate square matrices, initialize A to `1.0` and B to `2.0`, launch one warm-up iteration, then time one kernel launch with CUDA events. Reported throughput is:

```text
GFLOPS = 2 * N^3 / elapsed_seconds / 1e9
```

The timed section excludes host-device transfers. In `float4_MM.cu`, the B transpose is performed during warm-up and is also excluded from the timed GEMM launch.

## Recorded FP32 results

The following values were recorded in `float4Matrix benchmarks.docx`. They are historical results from the test environment, not portable performance guarantees.

| N | float4 only (GFLOPS) | float4 + 4 x 4 register tiling (GFLOPS) | double-buffered 32 x 32 (GFLOPS) |
| ---: | ---: | ---: | ---: |
| 512 | 1628.22 | 2016.98 | 3120.76 |
| 1024 | 1823.61 | 3093.14 | 3261.51 |
| 2048 | 1687.52 | 4888.47 | 3546.23 |
| 4096 | 1917.81 | 5009.62 | 3206.29 |

On that system, double buffering was strongest at smaller sizes, while the 64 x 64 register-tiled kernel reached the highest throughput for the larger matrices.

## Notes and limitations

- These are experiments and benchmarks, not a drop-in replacement for cuBLAS.
- The programs print throughput but do not currently perform numerical-result validation or CUDA API error checks. Add both before using them as a correctness or regression suite.
- The FP32 kernels benchmark `N = 512, 1024, 2048, 4096`; the WMMA variants include larger sizes as well.
- The WMMA variants use FP16 inputs and FP32 accumulators. Compare them with an appropriate mixed-precision cuBLAS configuration.
- `files_updated.zip` is an accompanying archive; the source files in this directory are the readable implementations described above.

## Suggested next steps

1. Add CUDA error checking after every runtime API call and kernel launch.
2. Validate each output against a CPU reference or cuBLAS using an explicit error tolerance.
3. Run Nsight Compute to compare occupancy, memory throughput, shared-memory bank conflicts, and tensor-core utilization across variants.
