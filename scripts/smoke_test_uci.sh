#!/usr/bin/env bash
set -euo pipefail
BIN="${1:-./build/scarlet}"
printf 'uci\nisready\nbackend\neval\ngo depth 2\nperft 5\nquit\n' | "$BIN"
