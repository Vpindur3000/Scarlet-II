#!/usr/bin/env bash
set -euo pipefail

ENGINE="${1:-./build/scarlet}"
OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

printf '%s\n' \
  'uci' \
  'isready' \
  'setoption name LD2CacheEntries value 2' \
  'position startpos' \
  'eval' \
  'leelaprobe' \
  'position fen rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' \
  'eval' \
  'leelaprobe' \
  'position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1' \
  'eval' \
  'leelaprobe' \
  'backend' \
  'position startpos' \
  'nnueverify 2' \
  'position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1' \
  'nnueverify 1' \
  'position fen 4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1' \
  'nnueverify 1' \
  'position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1' \
  'nnueverify 1' \
  'position startpos' \
  'perft 5' \
  'position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1' \
  'perft 4' \
  'position fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1' \
  'perft 5' \
  'quit' | "$ENGINE" > "$OUT"

rg -q '^uciok$' "$OUT"
rg -q '^readyok$' "$OUT"
rg -q '^option name LD2BatchSize type spin default 4 min 1 max 32$' "$OUT"
rg -q '^option name LD2BatchWorkers type spin default 0 min 0 max 16$' "$OUT"
rg -q '^option name UseSyzygy type check default true$' "$OUT"
rg -q '^option name SyzygyPath type string default <empty>$' "$OUT"
rg -q '^option name SyzygyProbeLimit type spin default 6 min 0 max 6$' "$OUT"
rg -q '^info string leela_raw cp .*p0=' "$OUT"
rg -q 'LeelaRaw=.*cache=2/2 evict=[1-9]' "$OUT"
if (( $(rg -c '^info string nnue incremental verify ok' "$OUT") != 4 )); then
  echo "FAIL: incremental NNUE verification did not cover all required positions" >&2
  exit 1
fi
rg -q '^perft\(5\) = 4865609$' "$OUT"
rg -q '^perft\(4\) = 4085603$' "$OUT"
rg -q '^perft\(5\) = 674624$' "$OUT"

mapfile -t evals < <(awk '/^eval primary / { print $3 }' "$OUT")
if (( ${#evals[@]} != 3 )); then
  echo "FAIL: expected three primary evaluations" >&2
  exit 1
fi
if (( evals[0] < -200 || evals[0] > 200 )); then
  echo "FAIL: implausible start evaluation: ${evals[0]}" >&2
  exit 1
fi
if (( evals[1] < 900 )); then
  echo "FAIL: missing black queen not recognized: ${evals[1]}" >&2
  exit 1
fi
if (( evals[2] > -800 )); then
  echo "FAIL: missing white queen not recognized: ${evals[2]}" >&2
  exit 1
fi

printf '%s\n' \
  'setoption name GuiProgress value false' \
  'setoption name LD2BatchSize value 4' \
  'setoption name LD2BatchWorkers value 2' \
  'setoption name ProofNodeLimit value malformed' \
  'go movetime malformed' \
  'position startpos' \
  'go movetime 200' \
  'backend' \
  'setoption name SearchCore value ClassicAB' \
  'position startpos' \
  'go movetime 100' \
  'setoption name SearchCore value ModernBStar' \
  'position fen 4k3/8/8/8/8/8/P3K3/8 w - - 0 1' \
  'go movetime 100' \
  'setoption name UseLD2 value false' \
  'position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8' \
  'go movetime 50' \
  'setoption name Clear Hash' \
  'position startpos' \
  'go movetime 45' \
  'position startpos' \
  'go movetime 45' \
  'position startpos moves e2e4 illegal' \
  'd' \
  'position startpos' \
  'go movetime 45 searchmoves e2e4' \
  'setoption name SearchCore value ClassicAB' \
  'position startpos' \
  'go movetime 45 searchmoves d2d4' \
  'quit' | "$ENGINE" > "$OUT"

if (( $(rg -c '^bestmove [a-h][1-8][a-h][1-8][qrbn]?$' "$OUT") < 8 )); then
  echo "FAIL: one of the searches did not return a legal-looking move" >&2
  exit 1
fi
rg -q '^info string ignored go command with invalid argument$' "$OUT"
rg -q 'string ModernBStar .*tree_nodes .*frontier_expansions' "$OUT"
rg -q 'ld2_batch_calls [1-9].*ld2_batch_positions [2-9].*ld2_batch_workers 2' "$OUT"
rg -q 'LeelaRaw=.*batches=[1-9].*/[2-9].*batchWorkers=2' "$OUT"
rg -q 'string ClassicAB iterative tactical baseline' "$OUT"
rg -q 'score cp 0 .*draw by rule at root' "$OUT"
rg -q 'tt_order_hits [1-9]' "$OUT"
rg -q '^info string Illegal move in position command: illegal$' "$OUT"
rg -q '^side white key ' "$OUT"
rg -q '^bestmove e2e4$' "$OUT"
rg -q '^bestmove d2d4$' "$OUT"
rg -q 'material_phase 64 leela_value_weight 45 leela_policy_weight 55' "$OUT"
rg -q 'material_phase 0 leela_value_weight 15 leela_policy_weight 20' "$OUT"
rg -q 'material_phase 0 .*ld2_probes [1-9]' "$OUT"
rg -q 'heuristic_early_stop 0' "$OUT"

# `go infinite` is asynchronous. Keep its input stream open long enough to
# distinguish an analysis search from a completed one: readyok must arrive
# before the sole bestmove, which is emitted only after the explicit stop.
TMPDIR_INFINITE="$(mktemp -d)"
trap 'rm -f "$OUT"; rm -rf "$TMPDIR_INFINITE"' EXIT
mkfifo "$TMPDIR_INFINITE/in"
"$ENGINE" < "$TMPDIR_INFINITE/in" > "$TMPDIR_INFINITE/out" &
ENGINE_PID=$!
exec 3>"$TMPDIR_INFINITE/in"
printf '%s\n' \
  'uci' \
  'isready' \
  'setoption name UseLD2 value false' \
  'setoption name ProofNodeLimit value 1024' \
  'position startpos' \
  'go infinite' >&3
sleep 0.35
printf '%s\n' 'isready' 'stop' 'quit' >&3
exec 3>&-
wait "$ENGINE_PID"

if (( $(rg -c '^bestmove ' "$TMPDIR_INFINITE/out") != 1 )); then
  echo "FAIL: go infinite did not produce exactly one bestmove after stop" >&2
  exit 1
fi
READY_LINE="$(rg -n '^readyok$' "$TMPDIR_INFINITE/out" | tail -n 1 | cut -d: -f1)"
BEST_LINE="$(rg -n '^bestmove ' "$TMPDIR_INFINITE/out" | cut -d: -f1)"
if [[ -z "$READY_LINE" || -z "$BEST_LINE" || "$BEST_LINE" -le "$READY_LINE" ]]; then
  echo "FAIL: go infinite returned bestmove before stop" >&2
  exit 1
fi

echo "Scarlet II v1.0.0 regression: PASS"
