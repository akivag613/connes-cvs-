#!/usr/bin/env python3
"""Exact audit for the Paper 3 spectral-barrier event corollary.

At a prime-power event the finite CvS path has derivative jump

    Delta Q' = -a vv^T,      a > 0.

If the common matrix value Q0 is positive definite, then every elementary
spectral barrier e_r(Q) has a strictly negative derivative jump.  The proof is
standard matrix calculus:

    d e_r(Q)[-vv^T] = - v^T T_{r-1}(Q) v,

where T_{r-1} is the Newton transformation.  For positive definite Q, the
quadratic form is positive.

This dependency-free audit checks the identity exactly on deterministic
positive definite integer matrices.  It compares the Newton-transform formula
with the principal-minor rank-one update formula.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path


DIMENSIONS = list(range(1, 11))
SEEDS = [1, 2, 5, 11, 17]


def det_bareiss(matrix: list[list[int]]) -> int:
    """Exact integer determinant by the fraction-free Bareiss algorithm."""

    n_value = len(matrix)
    if n_value == 0:
        return 1
    rows = [row[:] for row in matrix]
    sign = 1
    denom = 1
    for k_value in range(n_value - 1):
        pivot = rows[k_value][k_value]
        if pivot == 0:
            swap = None
            for row in range(k_value + 1, n_value):
                if rows[row][k_value] != 0:
                    swap = row
                    break
            if swap is None:
                return 0
            rows[k_value], rows[swap] = rows[swap], rows[k_value]
            sign *= -1
            pivot = rows[k_value][k_value]
        for i_value in range(k_value + 1, n_value):
            for j_value in range(k_value + 1, n_value):
                rows[i_value][j_value] = (
                    rows[i_value][j_value] * pivot
                    - rows[i_value][k_value] * rows[k_value][j_value]
                ) // denom
        denom = pivot
        for i_value in range(k_value + 1, n_value):
            rows[i_value][k_value] = 0
        for j_value in range(k_value + 1, n_value):
            rows[k_value][j_value] = 0
    return sign * rows[n_value - 1][n_value - 1]


def transpose(matrix: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*matrix)]


def matmul(a_matrix: list[list[int]], b_matrix: list[list[int]]) -> list[list[int]]:
    rows = len(a_matrix)
    cols = len(b_matrix[0])
    mid = len(b_matrix)
    return [
        [
            sum(a_matrix[i_value][k_value] * b_matrix[k_value][j_value] for k_value in range(mid))
            for j_value in range(cols)
        ]
        for i_value in range(rows)
    ]


def matadd(
    a_matrix: list[list[int]], b_matrix: list[list[int]], scale_b: int = 1
) -> list[list[int]]:
    return [
        [
            a_matrix[i_value][j_value] + scale_b * b_matrix[i_value][j_value]
            for j_value in range(len(a_matrix))
        ]
        for i_value in range(len(a_matrix))
    ]


def identity(n_value: int) -> list[list[int]]:
    return [[1 if i_value == j_value else 0 for j_value in range(n_value)] for i_value in range(n_value)]


def principal_submatrix(matrix: list[list[int]], indices: tuple[int, ...]) -> list[list[int]]:
    return [[matrix[i_value][j_value] for j_value in indices] for i_value in indices]


def elementary_symmetric_matrix(matrix: list[list[int]], r_value: int) -> int:
    n_value = len(matrix)
    if r_value == 0:
        return 1
    total = 0
    for indices in itertools.combinations(range(n_value), r_value):
        total += det_bareiss(principal_submatrix(matrix, indices))
    return total


def all_elementary(matrix: list[list[int]]) -> list[int]:
    n_value = len(matrix)
    return [elementary_symmetric_matrix(matrix, r_value) for r_value in range(n_value + 1)]


def matrix_power_sequence(matrix: list[list[int]]) -> list[list[list[int]]]:
    n_value = len(matrix)
    powers = [identity(n_value)]
    for _ in range(1, n_value + 1):
        powers.append(matmul(powers[-1], matrix))
    return powers


def newton_transform(matrix: list[list[int]], r_minus_one: int) -> list[list[int]]:
    n_value = len(matrix)
    e_values = all_elementary(matrix)
    powers = matrix_power_sequence(matrix)
    out = [[0 for _ in range(n_value)] for _ in range(n_value)]
    for j_value in range(r_minus_one + 1):
        coeff = (-1) ** j_value * e_values[r_minus_one - j_value]
        out = matadd(out, powers[j_value], coeff)
    return out


def quadratic(matrix: list[list[int]], vector: list[int]) -> int:
    return sum(
        vector[i_value] * matrix[i_value][j_value] * vector[j_value]
        for i_value in range(len(vector))
        for j_value in range(len(vector))
    )


def deterministic_spd(n_value: int, seed: int) -> list[list[int]]:
    base = []
    for i_value in range(n_value):
        row = []
        for j_value in range(n_value):
            row.append(((i_value + 2) * (j_value + 3) + seed * (i_value + 1)) % 9 - 4)
        base.append(row)
    gram = matmul(transpose(base), base)
    for i_value in range(n_value):
        gram[i_value][i_value] += 3 + seed + i_value
    return gram


def rank_one_subtract(matrix: list[list[int]], vector: list[int]) -> list[list[int]]:
    n_value = len(matrix)
    return [
        [
            matrix[i_value][j_value] - vector[i_value] * vector[j_value]
            for j_value in range(n_value)
        ]
        for i_value in range(n_value)
    ]


def audit_case(n_value: int, seed: int) -> dict[str, object]:
    matrix = deterministic_spd(n_value, seed)
    vector = [1 for _ in range(n_value)]
    updated = rank_one_subtract(matrix, vector)
    e_matrix = all_elementary(matrix)
    e_updated = all_elementary(updated)
    min_barrier_quadratic: int | None = None
    barriers = []
    for r_value in range(1, n_value + 1):
        principal_drop = e_matrix[r_value] - e_updated[r_value]
        transform = newton_transform(matrix, r_value - 1)
        newton_quadratic = quadratic(transform, vector)
        if principal_drop != newton_quadratic:
            raise SystemExit(
                f"d={n_value}, seed={seed}, r={r_value}: "
                f"principal drop {principal_drop} != Newton {newton_quadratic}"
            )
        if principal_drop <= 0:
            raise SystemExit(
                f"d={n_value}, seed={seed}, r={r_value}: nonpositive drop {principal_drop}"
            )
        min_barrier_quadratic = (
            principal_drop
            if min_barrier_quadratic is None
            else min(min_barrier_quadratic, principal_drop)
        )
        barriers.append(
            {
                "r": r_value,
                "e_r_Q": e_matrix[r_value],
                "derivative_jump_for_a_1": -principal_drop,
                "newton_quadratic": newton_quadratic,
            }
        )
    return {
        "dimension": n_value,
        "seed": seed,
        "det_Q": e_matrix[n_value],
        "min_positive_newton_quadratic": min_barrier_quadratic,
        "barriers": barriers,
    }


def main() -> None:
    start = time.time()
    cases = [audit_case(n_value, seed) for n_value in DIMENSIONS for seed in SEEDS]
    payload = {
        "status": "PASS",
        "claim": "For Q positive definite and event jump -a vv^T, each elementary spectral barrier has strictly negative derivative jump.",
        "dimensions": DIMENSIONS,
        "seeds": SEEDS,
        "cases": cases,
        "runtime_seconds": round(time.time() - start, 6),
    }
    out = Path(__file__).resolve().parents[1] / "artifacts" / "spectral_barrier_jump_audit.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
