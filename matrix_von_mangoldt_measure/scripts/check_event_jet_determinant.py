#!/usr/bin/env python3
"""Determinant audit for the Paper 3 event-jet map.

The manuscript proof gives the determinant formula.  This dependency-free
audit checks the formula over several large prime fields for large finite
levels, and also checks the triangular determinant of the tau-to-omega
transport.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


MODULI = [2_147_483_647, 1_000_000_007, 1_000_000_009, 998_244_353]
SELECTED_N = list(range(0, 16)) + [20, 30, 40, 60, 80, 100, 120, 160, 200]
TRANSPORT_SCALES = [2, 3, 5, 17, 257]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def det_mod(matrix: list[list[int]], mod: int) -> int:
    if not matrix:
        return 1
    rows = [row[:] for row in matrix]
    n_rows = len(rows)
    n_cols = len(rows[0])
    if n_rows != n_cols:
        raise ValueError("determinant matrix must be square")
    det = 1
    sign = 1
    for col in range(n_cols):
        pivot = None
        for row in range(col, n_rows):
            if rows[row][col] % mod:
                pivot = row
                break
        if pivot is None:
            return 0
        if pivot != col:
            rows[col], rows[pivot] = rows[pivot], rows[col]
            sign = -sign
        pivot_value = rows[col][col] % mod
        det = (det * pivot_value) % mod
        inv = pow(pivot_value, mod - 2, mod)
        for row in range(col + 1, n_rows):
            factor = rows[row][col] * inv % mod
            if factor:
                pivot_row = rows[col]
                rows[row] = [
                    (rows[row][j] - factor * pivot_row[j]) % mod
                    for j in range(n_cols)
                ]
    if sign < 0:
        det = (-det) % mod
    return det % mod


def binom_mod(n_value: int, k_value: int, mod: int) -> int:
    if k_value < 0 or k_value > n_value:
        return 0
    k_value = min(k_value, n_value - k_value)
    num = 1
    den = 1
    for j_value in range(1, k_value + 1):
        num = (num * (n_value - k_value + j_value)) % mod
        den = (den * j_value) % mod
    return num * pow(den, mod - 2, mod) % mod


def odd_jet_matrix(n_value: int, mod: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for ell in range(0, 2 * n_value + 1):
        row = [1 if ell == 0 else 0]
        for k_value in range(1, n_value + 1):
            x_value = k_value * k_value % mod
            row.append(pow(x_value, ell, mod))
        for k_value in range(1, n_value + 1):
            x_value = k_value * k_value % mod
            row.append((2 * ell + 1) * pow(x_value, ell, mod) % mod)
        rows.append(row)
    return rows


def expected_odd_det(n_value: int, mod: int) -> int:
    sign = -1 if (n_value * (n_value - 1) // 2) % 2 else 1
    out = pow(2, n_value, mod)
    for k_value in range(1, n_value + 1):
        out = out * pow(k_value, 6, mod) % mod
    for i_value in range(1, n_value + 1):
        i_square = i_value * i_value % mod
        for j_value in range(i_value + 1, n_value + 1):
            diff = (j_value * j_value - i_square) % mod
            out = out * pow(diff, 4, mod) % mod
    if sign < 0:
        out = (-out) % mod
    return out % mod


def tau_transport_matrix(n_value: int, scale: int, mod: int) -> list[list[int]]:
    max_index = 2 * n_value
    matrix: list[list[int]] = []
    for row_index in range(max_index + 1):
        r_value = 2 * row_index + 1
        scale_inv_power = pow(pow(scale, r_value, mod), mod - 2, mod)
        row: list[int] = []
        for col_index in range(max_index + 1):
            j_value = 2 * col_index + 1
            if r_value < j_value:
                row.append(0)
                continue
            row.append(
                binom_mod(r_value - 1, j_value - 1, mod) * scale_inv_power % mod
            )
        matrix.append(row)
    return matrix


def expected_tau_det(n_value: int, scale: int, mod: int) -> int:
    dimension = 2 * n_value + 1
    return pow(pow(scale, dimension * dimension, mod), mod - 2, mod)


def audit_case(n_value: int, mod: int) -> dict[str, object]:
    log(f"[mod={mod} N={n_value}] edge determinant")
    odd_det = det_mod(odd_jet_matrix(n_value, mod), mod)
    expected = expected_odd_det(n_value, mod)
    if odd_det != expected:
        raise SystemExit(
            f"N={n_value}, mod={mod}: odd determinant {odd_det} != {expected}"
        )

    transport = []
    for scale in TRANSPORT_SCALES:
        log(f"[mod={mod} N={n_value}] tau determinant, scale={scale}")
        tau_det = det_mod(tau_transport_matrix(n_value, scale, mod), mod)
        expected_tau = expected_tau_det(n_value, scale, mod)
        if tau_det != expected_tau:
            raise SystemExit(
                f"N={n_value}, mod={mod}, scale={scale}: tau determinant "
                f"{tau_det} != {expected_tau}"
            )
        transport.append(
            {
                "scale": scale,
                "det_mod": tau_det,
                "expected_det_mod": expected_tau,
            }
        )

    return {
        "N": n_value,
        "dimension": 2 * n_value + 1,
        "odd_edge_det_mod": odd_det,
        "expected_odd_edge_det_mod": expected,
        "tau_transport": transport,
    }


def main() -> None:
    start = time.time()
    cases_by_mod = {}
    for mod in MODULI:
        cases_by_mod[str(mod)] = [audit_case(n_value, mod) for n_value in SELECTED_N]
    payload = {
        "status": "PASS",
        "moduli": MODULI,
        "selected_N": SELECTED_N,
        "transport_scales": TRANSPORT_SCALES,
        "edge_determinant_formula": "(-1)^(N(N-1)/2) 2^N prod_k k^6 prod_{i<j}(j^2-i^2)^4",
        "tau_transport_determinant_formula": "scale^(-(2N+1)^2)",
        "runtime_seconds": round(time.time() - start, 6),
        "cases_by_modulus": cases_by_mod,
    }
    out = Path(__file__).resolve().parents[1] / "artifacts" / "event_jet_determinant_audit.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2))
    log(f"Paper 3 determinant audit passed in {payload['runtime_seconds']} seconds")


if __name__ == "__main__":
    main()
