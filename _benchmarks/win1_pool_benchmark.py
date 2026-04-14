#!/usr/bin/env python3
"""
WIN 1 Pool-based benchmark: A/B test using the production multiprocessing path.

Uses sweep._run_single_cutoff (12-way multiprocessing) so the timing is
directly comparable to historical results in results/iteration_*/results_*.json.

Usage:
    python win1_pool_benchmark.py [c] [N] [T] [dps]

Defaults: c=13 N=80 T=400 dps=80 (~30-90s, comparable to results_L_c13.json
which logged 602.8s at c=13 N=100 T=400 dps=80; smaller N/dps for budget).
"""
from __future__ import annotations
import os, platform, sys, time, json

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import mpmath as mp
from connes_cvs.sweep import _run_single_cutoff
from connes_cvs.operator import HAS_FLINT


def main() -> None:
    c   = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    N   = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    T   = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    dps = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    workers = 12

    print("=" * 72)
    print("Connes-van Suijlekom POOL benchmark (production-style, multiprocessing)")
    print("=" * 72)
    print(f"Python   : {platform.python_version()}")
    print(f"mpmath   : {mp.__version__}")
    print(f"HAS_FLINT: {HAS_FLINT}")
    print(f"Params   : c={c} N={N} T={T} dps={dps}  workers={workers}  DIM={2*N+1}")
    print("-" * 72)
    print(f"START at {time.strftime('%H:%M:%S')}")
    sys.stdout.flush()

    t_start = time.perf_counter()
    result = _run_single_cutoff(c=c, N=N, T=T, dps=dps, n_workers=workers)
    t_end = time.perf_counter()

    print(f"END   at {time.strftime('%H:%M:%S')}")
    print("-" * 72)

    timing = result.get("timing", {})
    print("Phase timings (seconds, from sweep._run_single_cutoff):")
    for k in ("cache_sec", "matrix_sec", "diag_sec", "zeros_sec", "total_sec"):
        v = timing.get(k)
        if v is not None:
            print(f"  {k:<14}: {v:9.3f}")
    print(f"  WALL TOTAL    : {(t_end - t_start):9.3f}")
    print("-" * 72)
    lam = result.get("lambda_min") or result.get("lambda_min_str")
    print(f"lambda_min : {lam}")
    print("=" * 72)


if __name__ == "__main__":
    main()
