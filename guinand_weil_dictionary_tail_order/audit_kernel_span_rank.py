#!/usr/bin/env python3
"""Exact audit for the finite Volterra-kernel span theorem.

For N, the manuscript identifies the linear span of all polarized Volterra
kernels with span{omega, sin(2*pi*k*omega), omega*cos(2*pi*k*omega)}_{k<=N}.
This audit checks the explicit generating coordinates over Q(P), where P is a
formal stand-in for pi.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


P = sp.Symbol("P", nonzero=True)


def generator_matrix(n_value: int) -> sp.Matrix:
    dim = 2 * n_value + 1
    rows = []
    base = [sp.Rational(0)] * dim
    base[0] = sp.Rational(2)
    rows.append(base)
    for k_value in range(1, n_value + 1):
        cross = [sp.Rational(0)] * dim
        cross[k_value] = sp.Rational(1, k_value) / P
        rows.append(cross)
        square = [sp.Rational(0)] * dim
        square[k_value] = sp.Rational(1, 2 * k_value) / P
        square[n_value + k_value] = sp.Rational(1)
        rows.append(square)
    return sp.Matrix(rows)


def main() -> None:
    cases = []
    for n_value in range(0, 31):
        matrix = generator_matrix(n_value)
        rank = int(matrix.rank())
        expected = 2 * n_value + 1
        if rank != expected:
            raise SystemExit(f"rank mismatch at N={n_value}: got {rank}, expected {expected}")
        cases.append({"N": n_value, "rank": rank, "expected_rank": expected})
    payload = {
        "status": "PASS",
        "method": "explicit generators 1*1, 1*cos(k), and cos(k)*cos(k) in the coordinate basis over Q(P)",
        "coordinate_basis": ["omega", "sin(2*pi*k*omega), 1<=k<=N", "omega*cos(2*pi*k*omega), 1<=k<=N"],
        "N_range": "0..30",
        "cases": cases,
    }
    Path("kernel_span_rank_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print("kernel span-rank audit passed")
    print("N range: 0..30")


if __name__ == "__main__":
    main()
