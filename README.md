# Scarlet II v1.0.0 — Modern B*

Scarlet II is an experimental GPLv3 UCI chess engine built around a custom best-first, interval-based search inspired by the B* family of algorithms.

Version 1.0.0 is the first public release of Scarlet II and, at the same time, a first practical step toward the more ambitious DCPS architecture. Modern B* provided a way to test several key ideas in a working engine: an explicit search graph, evaluation intervals, selective frontier expansion, and non-uniform allocation of computational effort among competing variations.

This is not intended to be a finished replacement for conventional alpha-beta engines. It is a self-contained experimental stage from which a more proof-oriented search system may eventually grow.

## Why Scarlet II?

Before Scarlet II, there was an earlier experimental engine called Scarlet. It combined a conventional PVS search with a PUCT-like search allocator. That version was never publicly released.

Scarlet II continues the same general line of experiments with non-standard search allocation, but it uses a fundamentally different foundation: explicit best-first search, interval evaluations, and Modern B* instead of PVS as the primary search core.

The `II` therefore marks the second generation of the project rather than an ordinary version increment.

## Core idea

A conventional chess engine usually attempts to assign each move a single, increasingly accurate score:

```text
move A = +0.42
move B = +0.31
```

Scarlet II instead works with ranges of plausible evaluations:

```text
move A ∈ [+0.20; +0.70]
move B ∈ [-0.10; +0.95]
```

In this example, move A currently appears better by its central evaluation, but move B remains a dangerous challenger because its upper bound is still high.

Scarlet II can direct additional search toward that challenger in order either to confirm its potential or to narrow its interval and remove it from contention.

The goal is not only to find the move with the highest current `score`, but to find a move that can be trusted with reasonable confidence under the available time limit.

## Search components

Scarlet II combines several sources of information:

```text
Berserk-compatible NNUE evaluation
                 +
optional Lc0-compatible policy/value evaluation
                 +
short alpha-beta and qsearch tactical verification
                 ↓
heuristic evaluation intervals
                 ↓
Modern B* and B-PUCT-like search allocation
                 ↓
move selection
```

### NNUE

A Berserk-compatible NNUE network is used as the main fast evaluator. It processes most positions and provides the engine with a strong modern baseline evaluation.

### Leela policy/value

A small Lc0-compatible network can be used as an additional source of evaluation and policy priors.

Its role is not only to provide a second opinion about the position, but also to suggest moves that may receive insufficient attention from conventional move ordering.

Scarlet II does not have to trust the network’s policy, but it may allocate additional search effort to candidates with sufficiently strong priors.

### Tactical verification

A short alpha-beta search and qsearch are used as tactical leaf verification rather than as the primary Modern B* search core.

This helps prevent the engine from trusting an overly narrow or optimistic interval in positions containing immediate threats, captures, checks, promotions, or other tactical complications.

## Modern B*

The classical B* idea is based on comparing optimistic and pessimistic bounds for competing moves.

In the ideal case, the best move becomes separated from every challenger:

```text
lower(best) > upper(challengers)
```

Scarlet II preserves this basic principle while adapting it to a practical chess engine:

* intervals are constructed from several heuristic sources of evidence;
* search operates on an explicit structure of nodes and edges;
* transpositions can be merged into a DAG;
* policy priors help identify promising candidates;
* search effort is directed toward nodes that have the greatest influence on the root decision;
* when complete interval separation cannot be reached, the engine makes a practical decision from the accumulated evidence.

Modern B* in Scarlet II is a working search architecture rather than only a conceptual design. It nevertheless remains an experimental stage of the project.

## Toward DCPS

The next major Scarlet II architecture is intended to develop the ideas of Modern B* into **DCPS — Dynamic Corridor Proof Search**.

DCPS is envisioned as a unified persistent best-first / proof-first search in which the primary object is neither depth nor a single point evaluation, but the current state of uncertainty around a position:

```text
evaluation corridor [lower, upper]
                    +
confidence
                    +
proof state
                    +
tactical risk
                    +
root decision pressure
```

The static evaluation foundation is expected to use **two independent NNUE networks**. They provide two fast but potentially different views of a position. Agreement between them increases confidence in the corridor center, while disagreement increases uncertainty and indicates that the branch requires further investigation.

DCPS is not merely a comparison between two evaluators. Its corridors are intended to be updated through several different forms of evidence:

```text
two NNUE evaluators
        +
DCS / Modern B* interval backup
        +
HCE routing for risk and tactical goals
        +
PNS / DFPN proof events
        +
optional policy priors
        +
exact Syzygy results
        ↓
Dynamic Corridor Proof Search
```

DCS remains the primary mode. It owns the persistent search tree, maintains the competition between the current best move and dangerous challengers, and directs work toward the branches that have the greatest influence on the root decision.

When a position is flat and DCS has no useful gradient, a short PUCT-like bootstrap may create an initial distinction between the candidate moves. It does not replace the main search or play the game on behalf of DCS. Its purpose is only to provide enough structure for further interval-based investigation.

When HCE identifies a concrete tactical goal—such as forcing mate, winning material, refuting a move, or promoting a pawn—the PNS/DFPN proof layer can be activated. Instead of returning an ordinary `score`, it produces a proof or disproof event, after which the relevant corridor is updated.

HCE is not intended to act as a third full evaluator. It serves as a router that determines:

* how dangerous it is to trust the static evaluation;
* whether the position is flat;
* whether a concrete tactical goal exists;
* whether proof search is justified;
* whether a branch may safely be reduced or temporarily frozen.

Syzygy remains a source of exact information and takes priority over every heuristic evaluation.

The main objective of DCPS is not simply to select the move with the best current evaluation center. It is to reduce uncertainty at the root, strengthen the lower bound of the best candidate, reduce the upper bounds of its challengers, and prove concrete tactical claims whenever possible.

Modern B* in version 1.0.0 serves as a practical foundation for this transition. It already tests several of the required ideas:

* an explicit persistent search structure;
* transpositions and nodes with multiple parents;
* frontier-based expansion;
* interval backup;
* competition between a proof leader and its challengers;
* non-uniform allocation of search effort;
* cooperation between multiple sources of evaluation;
* practical decision-making when the corridors remain partially overlapping.

DCPS is not implemented in version 1.0.0 and remains the next major development stage of Scarlet II.

## Implemented in version 1.0.0

The current version includes:

* the primary `ModernBStar` search mode;
* an alternative `ClassicAB` mode;
* an explicit search tree with DAG transpositions;
* `lower` and `upper` evaluation intervals;
* partial and full node expansion;
* interval backup through multiple parents;
* B-PUCT-like frontier selection;
* Berserk-compatible NNUE integration;
* optional Lc0-compatible value and policy integration;
* batched processing of selected Leela requests;
* a separate budget for expensive network calls;
* a short alpha-beta and qsearch tactical verifier;
* a transposition table;
* Syzygy WDL/DTZ probing;
* a UCI interface;
* diagnostic commands for inspecting loaded evaluators and the search process.

Scarlet II remains an experimental engine. The main value of version 1.0.0 is not only its playing strength, but the fact that Modern B* has been developed into a complete and usable UCI engine.

## Playing strength

In my local testing, the current version showed an approximate playing strength of around 2450 Elo.

The tests were performed on an AMD Ryzen 7 5700X with the following time control:

```text
20 seconds per game + 30 seconds per move
```

This is a rough local estimate rather than a rating from an independent rating list. The result may vary significantly depending on hardware, selected networks, opening set, opponents, and engine configuration.

The match and SPSA testing procedure is documented in:

```text
docs/history/StrengthValidation_v0.7.1.md
```

The file retains its historical name because it describes the testing procedure used before the public release.

## Networks

The source release does not include neural-network weights. Compatible networks must be obtained separately and configured through UCI options.

A supported Berserk-compatible network is required for the main NNUE evaluator.

A small Lc0-compatible network, such as a network from the LD2 family, may be used for the optional policy/value evaluation.

Configure the paths through UCI:

```text
setoption name BerserkNNUEFile value /path/to/compatible-network.nn
setoption name LeelaWeightsFile value /path/to/compatible-network.pb
```

After `isready`, use:

```text
backend
```

to inspect which evaluators were successfully loaded.

If a network is missing or fails to load, the engine reports a fallback or degraded state.

See:

* [NETWORKS.md](NETWORKS.md)
* [assets/networks/README.md](assets/networks/README.md)

## Syzygy

Syzygy WDL/DTZ files are not included in the release.

Configure the local tablebase path through:

```text
setoption name SyzygyPath value /path/to/tablebases
```

In the current implementation:

```text
SyzygyProbeLimit = 6
```

restricts probing to positions containing no more than six pieces.

## Building

The following are required:

* a compiler with C++20 support;
* CMake 3.20 or newer.

The default optimized configuration enables AVX2, BMI, BMI2, FMA, and POPCNT and is tuned for AMD Zen 3 processors.

### Linux / WSL / MSYS2

```bash
./scripts/build_zen3_avx2_linux.sh
```

Or build directly with CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSCARLET_ZEN3_AVX2=ON
cmake --build build -j
```

### Windows with MinGW

```text
scripts\build_zen3_avx2_windows_mingw.bat
```

## Quick checks

```bash
./scripts/smoke_test_uci.sh ./build/scarlet
./scripts/regression_v071.sh ./build/scarlet
```

Expected `perft(5)` result from the initial position:

```text
4865609
```

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

## About the term “proof”

The Scarlet II source code and diagnostic output use terms such as `proof`, `proof leader`, `heuristic_strict`, and `heuristic_practical`.

Within Scarlet II, a proof means separation of the engine’s current evaluation intervals.

For terminal positions and valid Syzygy results, the bounds may be exact. In ordinary positions, the intervals remain heuristic and do not constitute a mathematical proof of perfect play or of the objectively best move.

The terminology describes the purpose of the search process: to strengthen the selected candidate and eliminate dangerous challengers within the information available to the engine.

## Acknowledgements

Scarlet II was made possible by the open-source computer-chess community and by the work of many developers and researchers.

Special thanks to:

* [Berserk](https://github.com/jhonnold/berserk) and Jay Honnold for the open-source engine implementation, NNUE architecture, and compatible evaluation networks;
* [Leela Chess Zero](https://github.com/LeelaChessZero/lc0) and its community for policy/value methods, network formats, tools, and research into neural-network-guided search;
* [Stockfish](https://github.com/official-stockfish/Stockfish) for its enormous contribution to open-source chess-engine development, NNUE research, testing methodology, and the UCI ecosystem;
* [python-chess](https://github.com/niklasf/python-chess) for testing, analysis, and automation tools;
* the authors and contributors of Syzygy tablebases;
* the contributors to the [Chess Programming Wiki](https://www.chessprogramming.org/);
* the authors of research on B*, best-first search, proof-number search, PUCT, alpha-beta, and related algorithms;
* all developers of open-source chess engines and tools whose work makes projects like Scarlet II possible.

Scarlet II is an independent project. References to other projects indicate the use of compatible formats, ideas, tools, or open-source components and do not imply official affiliation or endorsement.

Detailed information about third-party source code, modifications, authorship, and licenses is provided in:

```text
THIRD_PARTY.md
```

## Source release policy

The Scarlet II v1.0.0 source release does not include:

* neural-network weights;
* a prebuilt engine binary;
* Syzygy tablebases.

Users must obtain external resources separately and are responsible for complying with their respective licenses and distribution terms.

## License

Scarlet II is free software distributed under the GNU General Public License version 3.

See:

```text
LICENSE
```

Third-party components remain subject to their respective licenses and retain their original copyright and attribution notices.

See:

```text
THIRD_PARTY.md
```
