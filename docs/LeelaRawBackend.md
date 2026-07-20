# LD2 raw in-process backend — correctness recovery

Scarlet loads `policy_value.pb` or `policy_value.pb.gz` and runs the CPU backend in-process:

```text
Position + exact 8-frame history -> [B,112,64]
  -> batched residual tower -> WDL/value + 73/80-plane policy
  -> legal-move softmax -> Modern B*
```

## Loading and validation

- Default files are resolved relative to the executable directory (and its
  packaged `../assets` directory). An explicit absolute path is unchanged; an
  explicit relative path is resolved against the current working directory.
- The UCI layer always prints the resolved absolute path. A load error emits
  `info string error network ...` and `info string warning degraded backend ...`.
- The protobuf reader checks the LC0 magic, complete length-delimited fields,
  LINEAR16 buffer parity and finite ranges, exact convolution/SE/FC tensor
  sizes, 73 or 80 policy planes, and a three-output WDL head before inference.
- Failed reloads never produce a partially valid network.

## Policy and cache correctness

- The convolutional policy follows LC0 exactly: knight promotion uses the
  ordinary ray plane; planes 64–72 are direction-major rook/bishop/queen.
- The mapping has a 24-case parity test against the vendored LC0
  `kConvPolicyMap` (four promotion pieces, three directions, both colours).
- Network results use an O(1) LRU. Its key includes the board, halfmove clock,
  repetition bits and exact Leela history; equal boards with different input
  histories cannot alias.
- Hits and overwrites move an entry to MRU. Eviction removes only the current
  list entry, so a stale FIFO record cannot erase a newer overwrite.

## Batched execution

`LD2BatchSize` is the maximum tensor batch. `LD2BatchWorkers` keeps its UCI
name for compatibility but now means the size of the persistent coarse-grained
OpenMP kernel team. It is not a count of independent inference jobs.

Every convolution and FC layer traverses one `[B,C,64]` tensor. Scalar probing
uses the identical route with `B=1`. Input, activation, padding, SE and head
workspace vectors retain their capacity after warmup. Result ownership and
legal-policy materialization may allocate, but the large layer workspaces do
not allocate again for an already warmed batch shape.

CTest compares batches `B=1/2/4/8/32` with the independent legacy scalar
reference for WDL, policy planes and legal priors at absolute error `1e-5`.
Cancellation is checked between residual blocks and cancelled results are not
inserted into the cache.
