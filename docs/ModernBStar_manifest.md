# Modern B* implementation manifest — correctness recovery

The active search is an epoch-based best-first DAG, not a root loop around
full-window alpha-beta.

## Search lifecycle

1. Every `go` runs on a worker thread. The UCI thread continues reading.
2. `SearchControl` owns one stop token, atomic node counter, deadline and ponder
   state shared by Modern B*, classic AB/qsearch and LD2.
3. `stop` joins and prints exactly one `bestmove`; `quit` joins without one.
   `ponderhit` arms a deadline without rebuilding the tree.
4. `position`, `ucinewgame`, a new `go`, and search-affecting `setoption`
   stop and join the previous search first.

## Frontier and epochs

The fixed operation order is:

```text
select -> evaluate -> tactical verify -> revise local interval
       -> expand/defer/refute -> progressive widening -> DAG backup
```

Verification is required before first expansion. A node is reverified only
when its local revision changes or the epoch requests a deeper verification.
Sanitizer cutoffs use only isolated HCE/classic TT entries of sufficient depth;
Modern corridors and network values remain ordering/static evidence.

Nodes carry all legal move descriptors and materialize top-K children. The
initial `K=4` and active node budget `1024` double across epochs up to
`ProofNodeLimit`; deferred candidates retain provisional influence. Node-limit
expansion is partial, never all-or-nothing. Verification depth rises from
`max(1,TacticalDepth-1)` to 8 in finite search and 64 in infinite analysis.

The root always has a valid scored move: failed materialization runs bounded
fallback AB, then one-ply NNUE/HCE if interrupted. Forced moves are evaluated,
and an exact root mate stops immediately. Each epoch can publish a valid PV,
score and interval.

## DAG, state and backup

- `ProofNode` stores a `PositionId`, never a `Position` object.
- Immutable positions live in a search-local arena. Expanded/verified nodes
  use a bounded 2048-entry NNUE accumulator LRU. A miss starts at the nearest
  cached canonical ancestor rather than replaying every leaf from the root.
- DAG identity requires compatible draw/repetition context and equal root ply,
  preserving acyclicity and mate distance. The exact network-input cache is
  stricter and includes all 112-plane history inputs.
- Reverse backup is a decreasing-ply priority worklist. `backup_node()` reports
  change and changed nodes requeue all parents; there is no permanent visited
  set that can leave a diamond root stale.

## Result semantics

`Result` carries `score_source`, `degraded`, `lower`, `upper` and `confidence`.
UCI keeps an ordinary `score cp|mate` for GUI compatibility and additionally
prints:

```text
info string bounds [L, U] confidence N source NAME degraded 0|1
```

Final score priority is exact mate/tablebase, current tactical verification,
fused selected-move evaluation, then primary NNUE/HCE. Network failure is
visible and degraded search continues; silent fallback is prohibited.

## Automated gates

CTest covers promotion parity, multi-parent DAG backup, O(1) LRU semantics,
TT resize fault injection, corrupt/truncated network rejection, LD2 batch
parity, history-key separation, async UCI stop/quit/ponderhit, exact startpos
and Kiwipete perft, and deterministic mate-in-one positions.
