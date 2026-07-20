# Modern B* correctness/performance audit — v0.7.1

## Correctness defects fixed

1. Berserk's file-order L1 weights were consumed as sparse-kernel order.
   v0.6 applies Berserk's `WeightIdxScrambled` permutation before inference.
2. The raw NNUE output lacked Berserk phase scaling and UCI normalization.
3. A timed-out negamax result could be negated, merged into the root and
   stored in TT. Completion now propagates explicitly and incomplete work is
   discarded.
4. The old Modern-B* driver repeatedly ran full-window root negamax and only
   chose a proof target afterwards. v0.6 owns an explicit proof tree and
   expands one selected frontier.
5. The LD2 value conversion was bounded relative to the Berserk fallback,
   compromising evaluator independence and cache correctness.
6. LD2 history used fabricated current-board copies. It now stores real
   eight-frame history with make/undo support.
7. Immediate `go infinite`/`stop` could race against the worker's stop reset.
8. Twofold history at the game root was incorrectly treated as a claimable
   threefold draw.

## Current performance shape

- Berserk NNUE uses path-local incremental accumulators; parity is checked for
  ordinary moves, castling, en passant and promotion.
- The sparse L1 implementation is scalar but correct.
- LD2 convolution is AVX2/FMA + OpenMP and is cached. Root siblings use a
  persistent parallel worker pool; `ld2_batch_calls`/`ld2_batch_positions`
  expose actual use, while `ld2_avg_us` feeds the time budget.
- HCE danger scans remain relatively expensive.
- Tactical sanitation is deliberately bounded to noisy positions, but still
  needs SEE/delta pruning and history ordering.
- Proof nodes hold an eight-frame board history, so the default tree cap is
  65,536 nodes. Cross-search context TT hints order moves but never cut off a
  Modern B* corridor.

## Verification gates

`scripts/regression_v071.sh` checks:

- UCI startup and LD2 policy output;
- plausible start and missing-queen NNUE signs/magnitudes;
- exact startpos `perft(5) = 4865609`;
- timed Modern B* and ClassicAB completion;
- explicit proof-tree diagnostics;
- root threefold handling.
- transactional `position`, `go searchmoves`, context-TT ordering and LD2 FIFO
  cache eviction diagnostics;
- real LD2 root batches and the configured persistent worker-pool size.

Strength claims require match testing and are intentionally not inferred from
these correctness gates.
