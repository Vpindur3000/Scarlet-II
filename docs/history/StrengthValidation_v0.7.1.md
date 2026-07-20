# Empirical strength gate — Scarlet II v0.7.1

Technical correctness and an engine-versus-engine result are different gates.
`scripts/regression_v071.sh` must pass first, but its result says nothing about
Elo, NNUE/TT parameter quality, or the value of the LD2 batch path.

## Reproducible paired match

Use a fixed baseline binary, a versioned balanced opening suite, two reversed
colours per round, and a fixed time control. The helper only launches Cute
Chess and preserves the PGN; it deliberately does not call a score an Elo
result.

```bash
python3 scripts/run_strength_gate.py \
  --candidate ./build/scarlet \
  --baseline /path/to/frozen-v0.7.0/scarlet \
  --openings /path/to/openings-v1.pgn \
  --rounds 400 --tc 40/60+0.6 \
  --pgn artifacts/v071-vs-v070.pgn
```

The same command with `--dry-run` validates paths and prints the exact Cute
Chess command without starting a match. On Windows use `python` and absolute
`.exe` paths. Do not compare against a moving target: record the baseline
commit/hash, compiler flags, network files, opening-suite checksum, hardware,
and exact UCI options alongside the PGN.

## Acceptance rule

For tuning a single search/TT/NNUE parameter, use a sequential paired SPRT in
the external match harness. A conservative starting convention is:

```text
H0 = 0 Elo, H1 = +5 Elo, alpha = beta = 0.05
```

Only accept a new default after the harness accepts H1; reject/revert after it
accepts H0. The 400-round command above is a reproducible smoke sample, not a
substitute for a completed SPRT. Any inconclusive result remains inconclusive.

## Status of this release

The repository contains no frozen prior candidate binary, opening suite, or
external match runner in scope, so an Elo outcome cannot honestly be generated
inside this source-tree task. The launcher and protocol make that last empirical
step reproducible once those inputs are supplied.
