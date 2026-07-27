# [Topic Name]

## What This Is
[One sentence. Example: "Parallel scan implementation 
optimized for GPU execution using shared memory 
and warp-level primitives."]

## Implementations
- naive_scan.cu — baseline, O(n) sequential
- inclusive_scan.cu — parallel Hillis-Steele
- exclusive_scan.cu — parallel with identity element
- multiblock_scan.cu — handles arrays > block size

## Key Optimizations Applied
1. Shared memory to reduce global memory accesses
2. Bank conflict elimination via padding (+1 offset)
3. Warp-level primitives (__shfl_down_sync) for
   final reduction — eliminates shared memory in warp
4. Double buffering for multiblock coordination

## Performance Results
| Implementation | Time (ms) | Speedup vs Naive |
|---|---|---|
| Naive | X ms | 1x |
| Shared memory | Y ms | Zx |
| Warp optimized | Y ms | Zx |

## What I Learned
Bank conflicts caused 40% slowdown in naive shared
memory version. Padding shared memory array by +1
per 32 elements eliminated conflicts completely.
NCU confirmed: bank conflicts dropped from 18% to 0.