#!/usr/bin/env python3
import subprocess
import time
import statistics
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = ROOT / 'build' / 'scarlet'
if not EXE.exists():
    alt = ROOT / 'build_final' / 'scarlet'
    if alt.exists(): EXE = alt

POSITIONS = [
    'position startpos',
    'position startpos moves e2e4',
    'position startpos moves d2d4',
    'position startpos moves g1f3',
    'position startpos moves c2c4',
    'position startpos moves e2e4 e7e5',
    'position startpos moves e2e4 c7c5',
    'position startpos moves d2d4 d7d5',
    'position startpos moves g1f3 d7d5',
    'position startpos moves c2c4 e7e5',
    'position startpos moves e2e4 e7e5 g1f3',
    'position startpos moves e2e4 c7c5 g1f3',
    'position startpos moves d2d4 g8f6 c2c4',
    'position startpos moves g1f3 d7d5 d2d4',
    'position startpos moves c2c4 e7e5 g2g3',
    'position startpos moves e2e4 e7e5 g1f3 b8c6',
    'position startpos moves d2d4 d7d5 c2c4 e7e6',
    'position startpos moves e2e4 c7c5 g1f3 d7d6',
    'position startpos moves g1f3 g8f6 c2c4 g7g6',
    'position startpos moves d2d4 g8f6 c2c4 g7g6',
]

def run_cmd(p, cmd):
    p.stdin.write(cmd + '\n')
    p.stdin.flush()

def read_until(p, token):
    while True:
        line = p.stdout.readline()
        if not line:
            raise RuntimeError('engine exited')
        if token in line:
            return line.strip()

def bench(threads):
    p = subprocess.Popen([str(EXE)], cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)
    run_cmd(p, 'uci'); read_until(p, 'uciok')
    run_cmd(p, f'setoption name LeelaThreads value {threads}')
    run_cmd(p, 'isready'); read_until(p, 'readyok')
    uncached = []
    for pos in POSITIONS:
        run_cmd(p, pos)
        t = time.perf_counter()
        run_cmd(p, 'leelaprobe')
        read_until(p, 'info string leela_raw')
        uncached.append((time.perf_counter() - t) * 1000.0)
    cached = []
    for pos in POSITIONS:
        run_cmd(p, pos)
        t = time.perf_counter()
        run_cmd(p, 'leelaprobe')
        read_until(p, 'info string leela_raw')
        cached.append((time.perf_counter() - t) * 1000.0)
    run_cmd(p, 'quit')
    try: p.wait(timeout=2)
    except subprocess.TimeoutExpired: p.kill()
    return uncached, cached

for threads in (1, 2, 4, 8):
    u, c = bench(threads)
    print(f'LeelaThreads={threads}: uncached avg={statistics.mean(u):.3f} ms median={statistics.median(u):.3f} ms '
          f'min={min(u):.3f} max={max(u):.3f} probes/s={1000.0/statistics.mean(u):.1f}; '
          f'cached avg={statistics.mean(c):.3f} ms')
