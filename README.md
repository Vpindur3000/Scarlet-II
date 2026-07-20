# Scarlet II v1.0.0 — Modern B*

Scarlet II is an experimental GPLv3 UCI chess engine built around a custom
best-first, interval-based search inspired by the B* family of algorithms.
Unlike a conventional engine that searches every root move primarily through a
single alpha-beta framework, Scarlet II maintains an explicit search tree/DAG,
expands selected frontiers, and allocates work using a B-PUCT-like policy.

The engine combines:

```text
external Berserk-compatible NNUE evaluation
              +
optional external Lc0-compatible policy/value evaluation
              +
classical alpha-beta and qsearch tactical checking
              -> heuristic score intervals
              -> Modern B* frontier allocation
              -> search decision
```

## Important terminology

Scarlet II uses terms such as `proof`, `proof leader`, `heuristic_strict`, and
`heuristic_practical` internally. Except for terminal positions and valid
Syzygy tablebase results, the intervals are heuristic evaluation corridors.
They are **not** mathematically rigorous minimax bounds and do not prove perfect
play or the objectively best move.

## Source release policy

This source release contains no neural-network weights and no prebuilt engine
binary. Compatible networks must be obtained separately and configured by the
user. See [NETWORKS.md](NETWORKS.md) and
[assets/networks/README.md](assets/networks/README.md).

Third-party source and attribution are documented in
[THIRD_PARTY.md](THIRD_PARTY.md). Scarlet II is not affiliated with or endorsed
by Berserk or Leela Chess Zero.

## Building

A C++20 compiler and CMake 3.20 or newer are required. The default optimized
configuration targets AVX2/BMI/BMI2/FMA/POPCNT and is tuned for AMD Zen 3.

Linux / WSL / MSYS2:

```bash
./scripts/build_zen3_avx2_linux.sh
```

Or directly with CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSCARLET_ZEN3_AVX2=ON
cmake --build build -j
```

Windows with MinGW:

```text
scripts\build_zen3_avx2_windows_mingw.bat
```

## External networks

Configure network paths through UCI:

```text
setoption name BerserkNNUEFile value /path/to/compatible-network.nn
setoption name LeelaWeightsFile value /path/to/compatible-network.pb
```

No particular weights file is licensed, endorsed, or redistributed by this
project. Use `backend` after `isready` to inspect the evaluators that actually
loaded. Missing networks must be reported as a fallback/degraded state rather
than silently represented as active.

## Syzygy

Syzygy WDL/DTZ files are not included. Set `SyzygyPath` to a local tablebase
directory. `SyzygyProbeLimit=6` restricts probing to positions with no more than
six pieces in the current implementation.

## Quick checks

```bash
./scripts/smoke_test_uci.sh ./build/scarlet
./scripts/regression_v071.sh ./build/scarlet
```

Expected `perft(5)` from the initial position: `4865609`.

Useful commands:

```text
uci
isready
backend
eval
leelaprobe
setoption name DebugModernBStar value true
position startpos
go movetime 1000
```

## Selected UCI options

```text
SearchCore              ModernBStar | ClassicAB
UseBerserkNNUE          true
UseLD2                  true
BerserkNNUEFile         <external path>
LeelaWeightsFile        <external path>
LeelaThreads            4
LD2CacheEntries         16384
LD2BatchSize            4
LD2BatchWorkers         0
UseSyzygy               true
SyzygyPath              <empty>
SyzygyProbeLimit         6
LeelaValueWeight        45
LeelaPolicyWeight       55
HeuristicEarlyStop      false
ProofNodeLimit          65536
TacticalDepth           3
PracticalMargin         24
MinProofExpansions      24
DebugModernBStar        false
```

## Acknowledgements

Scarlet II was developed with the help of the open-source computer-chess community and builds upon ideas, formats, tools, and research created by many other developers.

Special thanks to:

* [Berserk](https://github.com/jhonnold/berserk) by Jay Honnold — for the NNUE architecture and compatible evaluation networks used by Scarlet II.
* [Lc0](https://github.com/LeelaChessZero/lc0) and the Leela Chess Zero community — for policy/value-network formats, tooling, and research into neural-network-guided search.
* [Stockfish](https://github.com/official-stockfish/Stockfish) — for its major contributions to open-source chess-engine development, NNUE research, testing methodology, and UCI conventions.
* [python-chess](https://github.com/niklasf/python-chess) — for chess tooling, testing, and analysis utilities used during development.
* [Syzygy tablebases](https://github.com/syzygy1/tb) and their contributors — for endgame tablebase formats and research.
* The [Chess Programming Wiki](https://www.chessprogramming.org/) and its contributors — for documenting decades of chess-programming algorithms, terminology, and engine-design ideas.
* The authors and researchers behind B*, best-first search, proof-number search, PUCT, alpha-beta search, and related algorithms that inspired the experimental search architecture of Scarlet II.

Scarlet II is an independent experimental engine. References to other projects indicate compatibility, inspiration, or use of externally supplied resources and do not imply endorsement, official affiliation, or direct authorship by those projects.

Neural-network weights are not included in this repository or release. Users are responsible for obtaining compatible networks separately and complying with their respective licenses and terms.

## Strength claims

This repository does not claim an official universal Elo rating. Any published
strength number should include hardware, time control, opponents, opening set,
number of games, adjudication rules, and uncertainty, and should be described as
a local result unless reproduced by an independent rating list.

The reproducible match/SPSA helper is documented in
[docs/history/StrengthValidation_v0.7.1.md](docs/history/StrengthValidation_v0.7.1.md). That
file retains its historical name because it describes the pre-release test
procedure.

## License

Scarlet II is free software distributed under the GNU General Public License
version 3. See [LICENSE](LICENSE).

Third-party components remain under their respective compatible GPL terms and
retain their original notices. See [THIRD_PARTY.md](THIRD_PARTY.md).
