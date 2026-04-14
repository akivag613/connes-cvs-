#!/usr/bin/env python3
"""
Baseline benchmark for the Connes-van Suijlekom package.

Times each phase of the pipeline separately using time.perf_counter():

  1. Prime-data + kernel cache setup (prime_powers_up_to + psi cache).
  2. Matrix assembly (build_galerkin_matrix without the psi cache cost,
     broken out so we can see assembly vs. psi-cache cost independently).
  3. Eigensolver (compute_ground_state / eigsy on even sector).
  4. Zero extraction via findroot (extract_zeros, first zero only for speed).

Baseline for measuring WIN 1 / WIN 2 optimizations against.

This script DOES NOT modify connes_cvs/ source files. It re-implements
the matrix-assembly loop here using the exact same logic as
build_galerkin_matrix so we can time the psi cache and the assembly loop
separately. The result must be bit-identical to the public API.

Usage
-----
    python baseline_benchmark.py [c] [N] [dps]

Defaults: c=13, N=50, dps=50.
"""

from __future__ import annotations

import math
import platform
import sys
import time

import mpmath as mp

from connes_cvs import (
    build_galerkin_matrix,
    compute_ground_state,
    extract_zeros,
)
from connes_cvs.operator import (
    HAS_FLINT,
    _compute_psi_pair,
    prime_powers_up_to,
)

if HAS_FLINT:
    from flint import ctx as flint_ctx


def bench(c: int, N: int, T: int, dps: int) -> dict:
    """
    Run a timed single-process pass of the full pipeline and return timings.

    Uses the internal helpers (prime_powers_up_to + _compute_psi_pair) to
    time the psi cache and matrix assembly separately.
    """
    mp.mp.dps = dps
    if HAS_FLINT:
        flint_ctx.prec = int(dps * 3.5)

    # ---- Phase 1: prime data + psi cache (this is the kernel cache) ----
    t0 = time.perf_counter()
    L = mp.log(c)
    prime_data, _ = prime_powers_up_to(int(c))
    n_indices = list(range(-N, N + 1))
    psi_vals = {}
    psi_deriv_vals = {}
    for n_idx in n_indices:
        psi, psi_d = _compute_psi_pair(n_idx, L, T, dps, prime_data)
        psi_vals[n_idx] = psi
        psi_deriv_vals[n_idx] = psi_d
    t_cache = time.perf_counter() - t0

    # ---- Phase 2: matrix assembly (same logic as build_galerkin_matrix) ----
    t0 = time.perf_counter()
    DIM = 2 * N + 1
    Q = mp.matrix(DIM, DIM)
    for i in range(DIM):
        m_idx = i - N
        for j in range(DIM):
            n_idx = j - N
            if m_idx == n_idx:
                Q[i, j] = psi_deriv_vals[n_idx]
            else:
                Q[i, j] = (psi_vals[m_idx] - psi_vals[n_idx]) / (m_idx - n_idx)
    for i in range(DIM):
        for j in range(i + 1, DIM):
            avg = (Q[i, j] + Q[j, i]) / 2
            Q[i, j] = avg
            Q[j, i] = avg
    t_assemble = time.perf_counter() - t0

    # ---- Phase 3: eigensolver (even-sector projection + eigsy) ----
    t0 = time.perf_counter()
    lam_min, eigvec = compute_ground_state(Q)
    t_eig = time.perf_counter() - t0

    # ---- Phase 4: findroot zero extraction (first zero only for speed) ----
    t0 = time.perf_counter()
    zeros = extract_zeros(eigvec, L=float(L), n_zeros=1, dps=dps)
    t_findroot = time.perf_counter() - t0

    return {
        "c": c, "N": N, "T": T, "dps": dps,
        "DIM": DIM,
        "t_cache": t_cache,
        "t_assemble": t_assemble,
        "t_eig": t_eig,
        "t_findroot": t_findroot,
        "t_total": t_cache + t_assemble + t_eig + t_findroot,
        "lambda_min": lam_min,
        "zeros": zeros,
    }


def main() -> None:
    # Defaults chosen to exercise full pipeline under ~60s on a modern mac.
    c = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    dps = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    # T scales with dps for adequate archimedean decay; keep it cheap for N=50.
    T = 200 if N <= 60 else 400

    print("=" * 72)
    print("Connes-van Suijlekom baseline benchmark")
    print("=" * 72)
    print(f"Python   : {platform.python_version()} ({sys.executable})")
    print(f"mpmath   : {mp.__version__}")
    print(f"HAS_FLINT: {HAS_FLINT}")
    if HAS_FLINT:
        import flint
        fv = getattr(flint, "__version__", "unknown")
        print(f"flint    : {fv}")
    print(f"Params   : c={c}, N={N}, T={T}, dps={dps}  (matrix DIM={2*N+1})")
    print("-" * 72)

    result = bench(c=c, N=N, T=T, dps=dps)

    pct = lambda x: 100.0 * x / result["t_total"]
    print("Timings (seconds, wall, time.perf_counter()):")
    print(f"  1. psi cache (prime+pole+arch)  : {result['t_cache']:9.3f}  ({pct(result['t_cache']):5.1f}%)")
    print(f"  2. matrix assembly + symmetrize : {result['t_assemble']:9.3f}  ({pct(result['t_assemble']):5.1f}%)")
    print(f"  3. eigensolver (even sector)    : {result['t_eig']:9.3f}  ({pct(result['t_eig']):5.1f}%)")
    print(f"  4. findroot (first zero)        : {result['t_findroot']:9.3f}  ({pct(result['t_findroot']):5.1f}%)")
    print(f"  TOTAL                           : {result['t_total']:9.3f}  (100.0%)")
    print("-" * 72)

    # Correctness fingerprint: capture lambda_min to many digits.
    lam = result["lambda_min"]
    print("Correctness fingerprint:")
    print(f"  lambda_min = {mp.nstr(lam, min(dps, 60))}")
    print(f"  lambda_min (full repr):")
    print(f"    {mp.nstr(lam, dps)}")

    z0 = result["zeros"][0]
    gd = z0["gamma_detected"]
    gt = z0["gamma_true"]
    err = z0["error"]
    print(f"  gamma_1 true     = {mp.nstr(gt, 25)}")
    if gd is not None:
        print(f"  gamma_1 detected = {mp.nstr(gd, 25)}")
        print(f"  |gamma_1 error|  = {mp.nstr(err, 6)}")
    else:
        print(f"  gamma_1 detected = None (findroot failed)")
    print("=" * 72)


if __name__ == "__main__":
    main()
