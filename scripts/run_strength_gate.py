#!/usr/bin/env python3
"""Run a reproducible paired Cute Chess validation match for Scarlet.

This is intentionally a match launcher, not an Elo/SPRT implementation. The
recorded PGN and command line are the evidence that a later OpenBench/Fishtest
or SPRT analysis must consume; a technical regression must never manufacture a
strength claim on its own.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path


def existing_file(parser: argparse.ArgumentParser, value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        parser.error(f"{label} does not exist or is not a file: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch a colour-paired Cute Chess match and preserve a PGN evidence file.")
    parser.add_argument("--candidate", required=True, help="new Scarlet executable")
    parser.add_argument("--baseline", required=True, help="fixed reference executable")
    parser.add_argument("--openings", required=True, help="balanced opening-suite PGN/EPD")
    parser.add_argument("--cutechess", default="cutechess-cli", help="cutechess-cli executable or path")
    parser.add_argument("--rounds", type=int, default=400, help="paired rounds; each round plays two colours")
    parser.add_argument("--tc", default="40/60+0.6", help="Cute Chess time control, default 40/60+0.6")
    parser.add_argument("--pgn", default="strength-v1.0.0.pgn", help="destination PGN")
    parser.add_argument("--dry-run", action="store_true", help="print the command without launching it")
    args = parser.parse_args()

    if args.rounds < 1:
        parser.error("--rounds must be positive")
    candidate = existing_file(parser, args.candidate, "candidate")
    baseline = existing_file(parser, args.baseline, "baseline")
    openings = existing_file(parser, args.openings, "openings")
    pgn = Path(args.pgn).expanduser().resolve()

    cutechess = Path(args.cutechess).expanduser()
    if cutechess.is_file():
        launcher = str(cutechess.resolve())
    else:
        found = shutil.which(args.cutechess)
        if not found and not args.dry_run:
            parser.error("cutechess-cli was not found; pass --cutechess with its full path")
        launcher = found or args.cutechess

    command = [
        launcher,
        "-engine", "name=Scarlet-v1.0.0-candidate", f"cmd={candidate}", "proto=uci",
        "-engine", "name=Scarlet-baseline", f"cmd={baseline}", "proto=uci",
        "-each", f"tc={args.tc}",
        "-openings", f"file={openings}", "order=random",
        "-games", "2", "-repeat", "-rounds", str(args.rounds),
        "-pgnout", str(pgn),
    ]
    stamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    print(f"# Scarlet strength gate, generated {stamp}")
    print("# " + " ".join(command))
    print("# Acceptance requires external paired-match/SPRT analysis of the PGN; "
          "this launcher makes no Elo claim.")
    if args.dry_run:
        return 0

    pgn.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        print(f"Cute Chess failed with exit code {completed.returncode}; PGN: {pgn}", file=sys.stderr)
        return completed.returncode
    print(f"Cute Chess completed; analyse paired results in {pgn} before accepting a strength change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
