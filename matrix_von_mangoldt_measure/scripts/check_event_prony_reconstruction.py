#!/usr/bin/env python3
"""Exact audit for finite event-Prony reconstruction.

Paper 3's event-jet determinant says the first 2N+1 odd event jets form an
invertible finite coordinate system.  This audit checks the constructive
version over prime fields:

1. deterministic event-source coordinates are recovered exactly from the
   first 2N+1 odd jets;
2. the recovered coordinates predict later odd jets and satisfy the universal
   recurrence;
3. the first 2N odd jets leave a nonzero blind line whose next odd jet is
   nonzero.

The script is dependency-free and writes a JSON artifact for the paper
package.  The proof remains algebraic; this is a sign, indexing, and
constructivity guard.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


MODULI = [2_147_483_647, 1_000_000_007, 1_000_000_009, 998_244_353]
BASE_N = list(range(0, 11)) + [15, 20, 30, 40, 60, 80, 100, 120]
EXTRA_N_FIRST_MODULUS = [160, 200]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def selected_n_for_modulus(mod_index: int) -> list[int]:
    if mod_index == 0:
        return BASE_N + EXTRA_N_FIRST_MODULUS
    return BASE_N


def edge_row(n_value: int, ell_value: int, mod: int) -> list[int]:
    row = [1 if ell_value == 0 else 0]
    powers = [pow(k_value * k_value, ell_value, mod) for k_value in range(1, n_value + 1)]
    row.extend(powers)
    scale = (2 * ell_value + 1) % mod
    row.extend((scale * value) % mod for value in powers)
    return row


def edge_matrix(n_value: int, row_count: int, mod: int) -> list[list[int]]:
    return [edge_row(n_value, ell_value, mod) for ell_value in range(row_count)]


def deterministic_coefficients(n_value: int, mod: int) -> list[int]:
    coeffs = [(97 * n_value + 13) % mod]
    for k_value in range(1, n_value + 1):
        coeffs.append((17 * k_value * k_value + 5 * k_value + 3) % mod)
    for k_value in range(1, n_value + 1):
        coeffs.append((31 * k_value * k_value * k_value + 7 * k_value + 11) % mod)
    return coeffs


def mat_vec(matrix: list[list[int]], vector: list[int], mod: int) -> list[int]:
    return [sum((a_value * b_value) % mod for a_value, b_value in zip(row, vector)) % mod for row in matrix]


def solve_square(matrix: list[list[int]], rhs: list[int], mod: int) -> list[int]:
    size = len(matrix)
    work = [row[:] + [rhs[row_index] % mod] for row_index, row in enumerate(matrix)]

    for col in range(size):
        pivot = None
        for row_index in range(col, size):
            if work[row_index][col] % mod:
                pivot = row_index
                break
        if pivot is None:
            raise SystemExit(f"singular reconstruction matrix at column {col}")
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]

        inv_pivot = pow(work[col][col] % mod, -1, mod)
        pivot_tail = [(value * inv_pivot) % mod for value in work[col][col:]]
        work[col][col:] = pivot_tail

        for row_index in range(size):
            if row_index == col:
                continue
            factor = work[row_index][col] % mod
            if not factor:
                continue
            row = work[row_index]
            for offset, pivot_value in enumerate(pivot_tail):
                target_col = col + offset
                row[target_col] = (row[target_col] - factor * pivot_value) % mod

    return [work[row_index][size] % mod for row_index in range(size)]


def null_vector_rect(matrix: list[list[int]], mod: int) -> tuple[list[int], list[int]]:
    row_count = len(matrix)
    col_count = len(matrix[0]) if row_count else 1
    work = [row[:] for row in matrix]
    pivot_cols: list[int] = []
    pivot_row = 0

    for col in range(col_count):
        pivot = None
        for row_index in range(pivot_row, row_count):
            if work[row_index][col] % mod:
                pivot = row_index
                break
        if pivot is None:
            continue
        if pivot != pivot_row:
            work[pivot_row], work[pivot] = work[pivot], work[pivot_row]

        inv_pivot = pow(work[pivot_row][col] % mod, -1, mod)
        work[pivot_row][col:] = [(value * inv_pivot) % mod for value in work[pivot_row][col:]]

        for row_index in range(row_count):
            if row_index == pivot_row:
                continue
            factor = work[row_index][col] % mod
            if not factor:
                continue
            for target_col in range(col, col_count):
                work[row_index][target_col] = (
                    work[row_index][target_col] - factor * work[pivot_row][target_col]
                ) % mod

        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == row_count:
            break

    if len(pivot_cols) != row_count:
        raise SystemExit("truncated edge matrix did not have full row rank")

    free_cols = [col for col in range(col_count) if col not in set(pivot_cols)]
    if not free_cols:
        raise SystemExit("truncated edge matrix has no free column")
    free_col = free_cols[-1]

    vector = [0] * col_count
    vector[free_col] = 1
    for row_index, col in enumerate(pivot_cols):
        vector[col] = (-work[row_index][free_col]) % mod
    return vector, pivot_cols


def recurrence_polynomial(n_value: int, mod: int) -> list[int]:
    coeffs = [0, 1]
    for k_value in range(1, n_value + 1):
        x_value = (k_value * k_value) % mod
        factor = [x_value * x_value % mod, (-2 * x_value) % mod, 1]
        out = [0] * (len(coeffs) + 2)
        for i_value, a_value in enumerate(coeffs):
            if not a_value:
                continue
            for j_value, b_value in enumerate(factor):
                out[i_value + j_value] = (out[i_value + j_value] + a_value * b_value) % mod
        coeffs = out
    return coeffs


def moment_from_coeffs(coeffs: list[int], n_value: int, ell_value: int, mod: int) -> int:
    value = coeffs[0] if ell_value == 0 else 0
    scale = (2 * ell_value + 1) % mod
    for k_value in range(1, n_value + 1):
        x_power = pow(k_value * k_value, ell_value, mod)
        value += coeffs[k_value] * x_power
        value += coeffs[n_value + k_value] * scale * x_power
    return value % mod


def audit_case(n_value: int, mod: int) -> dict[str, object]:
    dimension = 2 * n_value + 1
    log(f"[mod={mod} N={n_value}] reconstructing {dimension} event coordinates")

    matrix = edge_matrix(n_value, dimension, mod)
    coeffs = deterministic_coefficients(n_value, mod)
    moments = mat_vec(matrix, coeffs, mod)
    recovered = solve_square(matrix, moments, mod)
    if recovered != coeffs:
        raise SystemExit(f"N={n_value}, mod={mod}: reconstruction mismatch")

    future_indices = list(range(dimension, dimension + min(12, dimension + 2)))
    future_ok = True
    for ell_value in future_indices:
        row_value = mat_vec([edge_row(n_value, ell_value, mod)], recovered, mod)[0]
        formula_value = moment_from_coeffs(coeffs, n_value, ell_value, mod)
        if row_value != formula_value:
            future_ok = False
            raise SystemExit(f"N={n_value}, mod={mod}, ell={ell_value}: future jet mismatch")

    poly = recurrence_polynomial(n_value, mod)
    recurrence_residue_max = 0
    for ell_value in range(0, min(8, dimension + 1)):
        residue = 0
        for j_value, coeff in enumerate(poly):
            residue = (
                residue + coeff * moment_from_coeffs(recovered, n_value, ell_value + j_value, mod)
            ) % mod
        recurrence_residue_max = max(recurrence_residue_max, residue)
        if residue:
            raise SystemExit(f"N={n_value}, mod={mod}, ell={ell_value}: recurrence residue {residue}")

    if dimension == 1:
        blind_next = 1
        blind_rank = 0
        blind_weight = [1]
    else:
        truncated = matrix[:-1]
        blind_weight, pivot_cols = null_vector_rect(truncated, mod)
        first_values = mat_vec(truncated, blind_weight, mod)
        if any(first_values):
            raise SystemExit(f"N={n_value}, mod={mod}: blind vector is visible too early")
        blind_next = mat_vec([matrix[-1]], blind_weight, mod)[0]
        if blind_next == 0:
            raise SystemExit(f"N={n_value}, mod={mod}: blind vector remained invisible at sharp jet")
        blind_rank = len(pivot_cols)

    return {
        "N": n_value,
        "dimension": dimension,
        "reconstruction": "PASS",
        "future_prediction": "PASS" if future_ok else "FAIL",
        "recurrence_residue_max": recurrence_residue_max,
        "twoN_jet_blind_rank": blind_rank,
        "twoN_jet_blind_next_value": blind_next,
        "twoN_jet_blind_support_nonzero": sum(1 for value in blind_weight if value % mod),
    }


def main() -> None:
    start = time.time()
    cases_by_mod = {}
    selected_by_mod = {}
    for mod_index, mod in enumerate(MODULI):
        selected = selected_n_for_modulus(mod_index)
        selected_by_mod[str(mod)] = selected
        cases_by_mod[str(mod)] = [audit_case(n_value, mod) for n_value in selected]

    payload = {
        "status": "PASS",
        "moduli": MODULI,
        "base_selected_N": BASE_N,
        "extra_N_first_modulus": EXTRA_N_FIRST_MODULUS,
        "selected_N_by_modulus": selected_by_mod,
        "object": "finite event-Prony reconstruction from the first 2N+1 odd event jets",
        "sharpness": "the first 2N odd jets leave a nonzero blind line whose next odd jet is nonzero",
        "runtime_seconds": round(time.time() - start, 6),
        "cases_by_modulus": cases_by_mod,
    }
    out = Path(__file__).resolve().parents[1] / "artifacts" / "event_prony_reconstruction_audit.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2))
    log(f"Paper 3 finite event-Prony reconstruction audit passed in {payload['runtime_seconds']} seconds")


if __name__ == "__main__":
    main()
