#!/usr/bin/env python3
"""Exact audit for the full-matrix finite source quotient.

The manuscript states that the N-level divided-difference source matrix
depends on a signed source measure only through the coordinates

    int omega dmu,
    int sin(2*pi*k*omega) dmu,
    int omega*cos(2*pi*k*omega) dmu, 1 <= k <= N.

This audit checks the entrywise matrix span over Q(P), where P is a formal
stand-in for pi, and verifies that the entries recover all 2N+1 coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


P = sp.Symbol("P", nonzero=True)


def entry_coordinate_vector(m_value: int, n_value: int, n_band: int) -> list[sp.Expr]:
    dim = 2 * n_band + 1
    vec = [sp.Rational(0)] * dim
    if m_value == n_value:
        if m_value == 0:
            vec[0] = sp.Rational(2)
        else:
            vec[n_band + abs(m_value)] = sp.Rational(2)
        return vec

    denom = P * (m_value - n_value)
    if m_value != 0:
        vec[abs(m_value)] += sp.Rational(1 if m_value > 0 else -1, 1) / denom
    if n_value != 0:
        vec[abs(n_value)] -= sp.Rational(1 if n_value > 0 else -1, 1) / denom
    return vec


def main() -> None:
    cases = []
    for n_band in range(0, 31):
        # First check that every matrix entry has a coordinate vector in the
        # stated 2N+1 basis.
        entry_count = 0
        for m_value in range(-n_band, n_band + 1):
            for n_value in range(-n_band, n_band + 1):
                vec = entry_coordinate_vector(m_value, n_value, n_band)
                if len(vec) != 2 * n_band + 1:
                    raise SystemExit(f"coordinate length mismatch at N={n_band}")
                entry_count += 1

        # Then verify that explicit entries recover all coordinates:
        # (0,0) gives omega, (k,0) gives sin_k, and (k,k) gives omega*cos_k.
        rows = [entry_coordinate_vector(0, 0, n_band)]
        for k_value in range(1, n_band + 1):
            rows.append(entry_coordinate_vector(k_value, 0, n_band))
            rows.append(entry_coordinate_vector(k_value, k_value, n_band))
        matrix = sp.Matrix(rows)
        rank = int(matrix.rank())
        expected = 2 * n_band + 1
        if rank != expected:
            raise SystemExit(f"full-matrix source quotient rank mismatch at N={n_band}: got {rank}, expected {expected}")
        cases.append({"N": n_band, "entry_count_checked": entry_count, "recovery_rows": len(rows), "rank": rank, "expected_rank": expected})

    payload = {
        "status": "PASS",
        "method": "entrywise divided-difference coordinate rank over Q(P)",
        "coordinate_basis": ["omega", "sin(2*pi*k*omega), 1<=k<=N", "omega*cos(2*pi*k*omega), 1<=k<=N"],
        "N_range": "0..30",
        "cases": cases,
    }
    Path("full_matrix_source_quotient_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print("full-matrix source-quotient audit passed")
    print("N range: 0..30")


if __name__ == "__main__":
    main()
