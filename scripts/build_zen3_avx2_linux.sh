#!/usr/bin/env bash
set -euo pipefail
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSCARLET_ZEN3_AVX2=ON
cmake --build build -j"$(nproc 2>/dev/null || echo 4)"
printf '\nBuilt: build/scarlet and build/scarlet-perft\n'
