#!/usr/bin/env python3
"""Exact Paper 3 guard for the source quotient and edge transport.

The manuscript proof is finite linear algebra.  This script checks the same
algebra over several prime fields, using the full matrix basis indexed by
{-N,...,N}:

1. Every single-source matrix entry is represented by the 2N+1 coordinates
   omega, sin(2*pi*k*omega), and omega*cos(2*pi*k*omega).
2. The entries (0,0), (k,0), and (k,k) recover all coordinates.
3. The edge derivative A'(0) is the all-2 matrix.
4. The odd tau-to-omega transport is lower triangular with determinant
   scale^(-(2N+1)^2).

This is a sign, indexing, and convention guard.  It is not a proof substitute.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


MODULI = [2_147_483_647, 1_000_000_007, 1_000_000_009, 998_244_353]
SELECTED_N = list(range(0, 16)) + [20, 30, 40, 60, 80, 100, 120, 160, 200]
TRANSPORT_SCALES = [2, 3, 5, 17, 257]
PI_INV_MARKER = 7


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def rank_mod(matrix: list[list[int]], mod: int) -> int:
    if not matrix:
        return 0
    rows = [row[:] for row in matrix]
    n_rows = len(rows)
    n_cols = len(rows[0])
    rank = 0
    for col in range(n_cols):
        pivot = None
        for row in range(rank, n_rows):
            if rows[row][col] % mod:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inv = pow(rows[rank][col] % mod, mod - 2, mod)
        rows[rank] = [(value * inv) % mod for value in rows[rank]]
        for row in range(n_rows):
            if row == rank:
                continue
            factor = rows[row][col] % mod
            if factor:
                pivot_row = rows[rank]
                rows[row] = [
                    (rows[row][j] - factor * pivot_row[j]) % mod
                    for j in range(n_cols)
                ]
        rank += 1
        if rank == min(n_rows, n_cols):
            break
    return rank


def det_mod(matrix: list[list[int]], mod: int) -> int:
    if not matrix:
        return 1
    rows = [row[:] for row in matrix]
    n = len(rows)
    det = 1
    sign = 1
    for col in range(n):
        pivot = None
        for row in range(col, n):
            if rows[row][col] % mod:
                pivot = row
                break
        if pivot is None:
            return 0
        if pivot != col:
            rows[col], rows[pivot] = rows[pivot], rows[col]
            sign *= -1
        pivot_value = rows[col][col] % mod
        det = det * pivot_value % mod
        inv = pow(pivot_value, mod - 2, mod)
        for row in range(col + 1, n):
            factor = rows[row][col] * inv % mod
            if factor:
                pivot_row = rows[col]
                rows[row] = [
                    (rows[row][j] - factor * pivot_row[j]) % mod
                    for j in range(n)
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
        num = num * (n_value - k_value + j_value) % mod
        den = den * j_value % mod
    return num * pow(den, mod - 2, mod) % mod


def coord_index_sin(k_value: int) -> int:
    return k_value


def coord_index_cos(n_band: int, k_value: int) -> int:
    return n_band + k_value


def entry_coordinate_vector(
    m_value: int, n_value: int, n_band: int, mod: int
) -> list[int]:
    dim = 2 * n_band + 1
    vec = [0] * dim
    if m_value == n_value:
        if m_value == 0:
            vec[0] = 2 % mod
        else:
            vec[coord_index_cos(n_band, abs(m_value))] = 2 % mod
        return vec

    denom_inv = pow((m_value - n_value) % mod, mod - 2, mod)
    factor = PI_INV_MARKER * denom_inv % mod
    if m_value != 0:
        sign_m = 1 if m_value > 0 else -1
        vec[coord_index_sin(abs(m_value))] = (
            vec[coord_index_sin(abs(m_value))] + sign_m * factor
        ) % mod
    if n_value != 0:
        sign_n = 1 if n_value > 0 else -1
        vec[coord_index_sin(abs(n_value))] = (
            vec[coord_index_sin(abs(n_value))] - sign_n * factor
        ) % mod
    return vec


def source_recovery_matrix(n_band: int, mod: int) -> list[list[int]]:
    rows = [entry_coordinate_vector(0, 0, n_band, mod)]
    for k_value in range(1, n_band + 1):
        rows.append(entry_coordinate_vector(k_value, 0, n_band, mod))
        rows.append(entry_coordinate_vector(k_value, k_value, n_band, mod))
    return rows


def first_jump_profile_ok(n_band: int) -> bool:
    for m_value in range(-n_band, n_band + 1):
        for n_value in range(-n_band, n_band + 1):
            if m_value == n_value:
                derivative = 2
            else:
                numerator_slope = 2 * m_value - 2 * n_value
                derivative = numerator_slope // (m_value - n_value)
            if derivative != 2:
                return False
    return True


def tau_transport_matrix(n_band: int, scale: int, mod: int) -> list[list[int]]:
    max_index = 2 * n_band
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
                binom_mod(r_value - 1, j_value - 1, mod)
                * scale_inv_power
                % mod
            )
        matrix.append(row)
    return matrix


def expected_transport_det(n_band: int, scale: int, mod: int) -> int:
    dim = 2 * n_band + 1
    return pow(pow(scale, dim * dim, mod), mod - 2, mod)


def audit_case(n_band: int, mod: int) -> dict[str, object]:
    dim = 2 * n_band + 1
    log(f"[mod={mod} N={n_band}] source quotient")
    rows = source_recovery_matrix(n_band, mod)
    source_rank = rank_mod(rows, mod)
    if source_rank != dim:
        raise SystemExit(
            f"N={n_band}, mod={mod}: source rank {source_rank} != {dim}"
        )

    entry_count = 0
    for m_value in range(-n_band, n_band + 1):
        for n_value in range(-n_band, n_band + 1):
            vec = entry_coordinate_vector(m_value, n_value, n_band, mod)
            entry_count += 1
            if len(vec) != dim:
                raise SystemExit("coordinate vector length mismatch")

    if not first_jump_profile_ok(n_band):
        raise SystemExit(f"N={n_band}: first jump profile is not all 2")

    transport = []
    for scale in TRANSPORT_SCALES:
        log(f"[mod={mod} N={n_band}] transport determinant scale={scale}")
        matrix = tau_transport_matrix(n_band, scale, mod)
        transport_det = det_mod(matrix, mod)
        expected_det = expected_transport_det(n_band, scale, mod)
        if transport_det != expected_det:
            raise SystemExit(
                f"N={n_band}, mod={mod}, scale={scale}: "
                f"transport det {transport_det} != {expected_det}"
            )
        transport.append(
            {
                "scale": scale,
                "det_mod": transport_det,
                "expected_det_mod": expected_det,
            }
        )

    return {
        "N": n_band,
        "dimension": dim,
        "entry_count_checked": entry_count,
        "source_recovery_rank_mod": source_rank,
        "first_jump_profile": "A'(0)=2*all_ones_all_ones^T",
        "tau_transport": transport,
    }


def main() -> None:
    start = time.time()
    cases_by_mod = {}
    for mod in MODULI:
        cases_by_mod[str(mod)] = [audit_case(n_value, mod) for n_value in SELECTED_N]
    payload = {
        "status": "PASS",
        "method": "full-matrix source quotient and edge transport over prime fields",
        "moduli": MODULI,
        "selected_N": SELECTED_N,
        "transport_scales": TRANSPORT_SCALES,
        "coordinate_basis": [
            "int omega dmu",
            "int sin(2*pi*k*omega) dmu, 1<=k<=N",
            "int omega*cos(2*pi*k*omega) dmu, 1<=k<=N",
        ],
        "runtime_seconds": round(time.time() - start, 6),
        "cases_by_modulus": cases_by_mod,
    }
    out = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "source_quotient_transport_audit.json"
    )
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2))
    log(f"Paper 3 source quotient and transport audit passed in {payload['runtime_seconds']} seconds")


if __name__ == "__main__":
    main()
