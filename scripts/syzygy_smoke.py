#!/usr/bin/env python3
"""Minimal black-box UCI smoke test for Scarlet's 5--6 man Syzygy support.

Usage:
    python scripts/syzygy_smoke.py path/to/scarlet path/to/syzygy

The selected KPPPPvK position requires a six-piece WDL and DTZ pair.  A full
six-man set naturally satisfies that requirement; a deliberately partial set
must contain KPPPPvK.rtbw and KPPPPvK.rtbz plus its dependencies.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


FEN_6_MAN = "7k/8/8/8/8/8/PPPP4/K7 w - - 0 1"


def fail(message: str, output: str) -> None:
    print(f"Syzygy smoke FAILED: {message}", file=sys.stderr)
    print(output, file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} ENGINE SYZYGY_DIRECTORY")

    engine = Path(sys.argv[1])
    tables = Path(sys.argv[2])
    if not engine.is_file():
        raise SystemExit(f"Engine not found: {engine}")
    if not tables.is_dir():
        raise SystemExit(f"Syzygy directory not found: {tables}")

    commands = "\n".join((
        "uci",
        "setoption name UseSyzygy value true",
        f"setoption name SyzygyPath value {tables}",
        "setoption name SyzygyProbeLimit value 6",
        "isready",
        f"position fen {FEN_6_MAN}",
        "syzygy",
        "go depth 2",
        "quit",
        "",
    ))
    completed = subprocess.run(
        [str(engine)], input=commands, text=True, capture_output=True, timeout=30,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode:
        fail(f"engine exited with code {completed.returncode}", output)
    if "option name SyzygyPath type string" not in output:
        fail("Syzygy UCI option was not advertised", output)
    largest = re.search(r"SyzygyPath loaded largest (\d+)", output)
    if not largest or int(largest.group(1)) < 6:
        fail("the supplied directory did not load six-man tablebases", output)
    if not re.search(r"syzygy root wdl [-2-2] dtz \d+ best [a-h][1-8][a-h][1-8]", output):
        fail("root DTZ probe returned no legal six-man move", output)
    if "Syzygy exact root wdl" not in output:
        fail("finite root search did not use the exact Syzygy result", output)
    print("Syzygy smoke OK: 6-man WDL + DTZ root probe")


if __name__ == "__main__":
    main()
