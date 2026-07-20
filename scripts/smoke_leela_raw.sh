#!/usr/bin/env bash
set -euo pipefail
ENGINE="${1:-./build/scarlet}"
"$ENGINE" < scripts/smoke_leela_raw.uci
