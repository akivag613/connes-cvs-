#!/usr/bin/env python3
"""Large exact audit for the Paper 3 event-jet theorem.

This script is intentionally dependency-free.  It checks the coefficient
algebra behind the finite prime-power event theorem over a large prime field:

1. The odd edge-jet map on
   span{w, sin(2*pi*k*w), w*cos(2*pi*k*w) : 1 <= k <= N}
   has full rank.
2. The centered finite-difference vector is the unique top-order blind even
   normal.
3. The odd tau-jets after omega=tau/(L+tau) recover the odd omega-jets.

The proof in the paper is analytic.  This audit is a sign, rank, indexing, and
endpoint-convention guard for large finite levels.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


MOD = 2_147_483_647
SELECTED_N = list(range(0, 16)) + [20, 30, 40, 60, 80, 100, 120, 160, 200]
TRANSPORT_SCALES = [2, 3, 5, 17]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def rank_mod(matrix: list[list[int]], mod: int = MOD) -> int:
    """Dense Gaussian rank over F_mod."""

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


def odd_jet_matrix(n_value: int, mod: int = MOD) -> list[list[int]]:
    rows: list[list[int]] = []
    for ell in range(0, 2 * n_value + 1):
        row = [1 if ell == 0 else 0]
        for k_value in range(1, n_value + 1):
            row.append(pow((k_value * k_value) % mod, ell, mod))
        for k_value in range(1, n_value + 1):
            row.append(((2 * ell + 1) * pow((k_value * k_value) % mod, ell, mod)) % mod)
        rows.append(row)
    return rows


def binom_mod(n_value: int, k_value: int, mod: int = MOD) -> int:
    if k_value < 0 or k_value > n_value:
        return 0
    k_value = min(k_value, n_value - k_value)
    num = 1
    den = 1
    for j_value in range(1, k_value + 1):
        num = (num * (n_value - k_value + j_value)) % mod
        den = (den * j_value) % mod
    return (num * pow(den, mod - 2, mod)) % mod


def tau_transport_matrix(n_value: int, scale: int, mod: int = MOD) -> list[list[int]]:
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
            sign = 1 if (r_value - j_value) % 2 == 0 else mod - 1
            row.append((sign * binom_mod(r_value - 1, j_value - 1, mod) * scale_inv_power) % mod)
        matrix.append(row)
    return matrix


def centered_stencil(radius: int, mod: int = MOD) -> list[int]:
    coeffs: list[int] = []
    c_value = 1
    for j_value in range(0, 2 * radius + 1):
        if j_value == 0:
            c_value = 1
        else:
            c_value = c_value * (2 * radius - j_value + 1) // j_value
        coeffs.append(((-1) ** j_value * c_value) % mod)
    return coeffs


def stencil_moment_mod(radius: int, order: int, mod: int = MOD) -> int:
    total = 0
    for idx, coeff in enumerate(centered_stencil(radius, mod)):
        m_value = idx - radius
        total = (total + coeff * pow(m_value % mod, order, mod)) % mod
    return total % mod


def extremizer_moment_matrix(radius: int, mod: int = MOD) -> list[list[int]]:
    rows: list[list[int]] = []
    for ell in range(radius):
        row = []
        for k_value in range(radius + 1):
            row.append(pow((k_value * k_value) % mod, ell, mod))
        rows.append(row)
    return rows


def factorial_mod(n_value: int, mod: int = MOD) -> int:
    out = 1
    for j_value in range(1, n_value + 1):
        out = (out * j_value) % mod
    return out


def audit_n(n_value: int) -> dict[str, object]:
    dimension = 2 * n_value + 1
    log(f"[N={n_value}] odd edge-jet rank")
    odd_rank = rank_mod(odd_jet_matrix(n_value))
    if odd_rank != dimension:
        raise SystemExit(f"N={n_value}: odd jet rank {odd_rank} != {dimension}")

    transport_results = []
    for scale in TRANSPORT_SCALES:
        log(f"[N={n_value}] tau transport rank, scale={scale}")
        transport_rank = rank_mod(tau_transport_matrix(n_value, scale))
        if transport_rank != dimension:
            raise SystemExit(
                f"N={n_value}, scale={scale}: tau rank {transport_rank} != {dimension}"
            )
        transport_results.append({"scale": scale, "rank_mod": transport_rank})

    extremizer_payload: dict[str, object] | None = None
    if n_value > 0:
        log(f"[N={n_value}] extremizer moment rank and top moment")
        moment_rank = rank_mod(extremizer_moment_matrix(n_value))
        if moment_rank != n_value:
            raise SystemExit(f"N={n_value}: moment rank {moment_rank} != {n_value}")
        for order in range(0, 2 * n_value):
            value = stencil_moment_mod(n_value, order)
            if value != 0:
                raise SystemExit(f"N={n_value}, order={order}: moment {value} != 0")
        top_moment = stencil_moment_mod(n_value, 2 * n_value)
        expected_top = factorial_mod(2 * n_value)
        if top_moment != expected_top:
            raise SystemExit(f"N={n_value}: top moment {top_moment} != {expected_top}")
        extremizer_payload = {
            "moment_rank_mod": moment_rank,
            "top_moment_mod": top_moment,
            "expected_top_moment_mod": expected_top,
        }

    return {
        "N": n_value,
        "dimension": dimension,
        "odd_jet_rank_mod": odd_rank,
        "first_jump_profile": "closed formula gives derivative 2 in every entry",
        "tau_transport": transport_results,
        "extremizer": extremizer_payload,
    }


def main() -> None:
    start = time.time()
    log("Paper 3 large exact event-jet audit started")
    cases = [audit_n(n_value) for n_value in SELECTED_N]
    payload = {
        "status": "PASS",
        "modulus": MOD,
        "selected_N": SELECTED_N,
        "transport_scales": TRANSPORT_SCALES,
        "runtime_seconds": round(time.time() - start, 6),
        "cases": cases,
    }
    out = Path(__file__).resolve().parents[1] / "artifacts" / "event_jet_largeN_audit.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2))
    log(f"Paper 3 large exact event-jet audit passed in {payload['runtime_seconds']} seconds")


if __name__ == "__main__":
    main()
