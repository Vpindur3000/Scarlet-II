# LD2 raw in-process performance audit

This branch removes the LC0 search-wrapper path from Scarlet's active search and runs LD2 as an in-process value+policy oracle.

## Important result

LD2 itself is not the problem. The previous raw path was slow because the 3x3 convolution kernel was scalar and checked board borders inside the innermost loops.

Measured in this Linux container after engine startup, 20 different opening positions:

| version | threads | uncached LD2 probe | approximate probes/s |
|---|---:|---:|---:|
| old scalar raw conv | 1 | ~157.5 ms | ~6.3/s |
| AVX2 row-vector conv | 1 | ~27.6 ms | ~36/s |
| AVX2 + OpenMP | 4 | ~8.6 ms | ~116/s |

Cached repeat probes are ~0.18 ms because the result is stored by `Position::key`.

## What was optimized

- 3x3 conv now pads each 8x8 channel to 10x10 once, removing per-tap border branches.
- 3x3 and 1x1 conv use AVX2/FMA over the 8 files of one rank.
- Optional OpenMP parallelizes output channels.
- Default `LeelaThreads` is clamped to 4 to avoid the catastrophic oversubscription seen with default OpenMP thread counts.
- Root LD2 candidates now also have a persistent cross-position worker pool:
  `LD2BatchSize` forms groups and `LD2BatchWorkers` selects its size. Each
  pooled task runs its inner convolution with one OpenMP thread.

## Why LC0 can still be faster

LC0 backends use heavily optimized inference engines and/or GPU/DML/ONNX batching. Scarlet's raw backend is now architecturally correct and usable for B* proof probes, but it is not yet LC0-inference-parity.

The next speed tier is one of:

1. ONNX Runtime in-process session, but called as raw network inference, not `go nodes` search.
2. NCHW blocked layout such as C8/C16 to reduce memory traffic.
3. tensor-N / blocked-layout batch kernels, rather than independent workers.
4. ONNX/GPU inference with native backend batching.
