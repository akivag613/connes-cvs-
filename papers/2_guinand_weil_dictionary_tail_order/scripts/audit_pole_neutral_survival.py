#!/usr/bin/env python3
"""Exact symbolic audit for the pole-neutral survival corollary.

The proof in the manuscript is algebraic.  This guard checks two finite
identities over Q(b), with b = beta^2:

1. after collapsing +/- frequency pairs, the pole matrix is b r r^T;
2. the pole row is independent of the finite moment rows.

The script is not a proof substitute.  It is a sign and indexing guard for the
paper's displayed formulas.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def signs(index: int) -> list[int]:
    if index == 0:
        return [0]
    return [-index, index]


def weight(index: int) -> sp.Rational:
    if index == 0:
        return sp.Rational(1, 1)
    return sp.Rational(1, 2)


def full_pole_entry(m_value: int, n_value: int, b_symbol: sp.Symbol) -> sp.Expr:
    denominator = (sp.Integer(m_value * m_value) + b_symbol) * (
        sp.Integer(n_value * n_value) + b_symbol
    )
    return (b_symbol - sp.Integer(m_value * n_value)) / denominator


def collapsed_entry(p_value: int, q_value: int, b_symbol: sp.Symbol) -> sp.Expr:
    total = sp.Integer(0)
    for m_value in signs(p_value):
        for n_value in signs(q_value):
            total += (
                weight(p_value)
                * weight(q_value)
                * full_pole_entry(m_value, n_value, b_symbol)
            )
    return sp.factor(total)


def row_entry(index: int, b_symbol: sp.Symbol) -> sp.Expr:
    return sp.Integer(1) / (sp.Integer(index * index) + b_symbol)


def verify_pole_square(n_value: int, b_symbol: sp.Symbol) -> dict[str, int]:
    checked = 0
    for p_value in range(n_value + 1):
        for q_value in range(n_value + 1):
            left = collapsed_entry(p_value, q_value, b_symbol)
            right = sp.factor(
                b_symbol * row_entry(p_value, b_symbol) * row_entry(q_value, b_symbol)
            )
            if sp.simplify(left - right) != 0:
                raise SystemExit(
                    "pole-square mismatch: "
                    f"N={n_value}, p={p_value}, q={q_value}, "
                    f"left={left}, right={right}"
                )
            checked += 1
    return {"N": n_value, "entries_checked": checked}


def vandermonde_for_squared_nodes(nodes: list[int]) -> sp.Integer:
    product = sp.Integer(1)
    squared = [sp.Integer(k * k) for k in nodes]
    for index, left in enumerate(squared):
        for right in squared[index + 1 :]:
            product *= right - left
    return sp.expand(product)


def pole_moment_matrix(nodes: list[int], s_value: int, b_symbol: sp.Symbol) -> sp.Matrix:
    rows: list[list[sp.Expr]] = []
    for moment in range(s_value + 1):
        rows.append([sp.Integer(k * k) ** moment for k in nodes])
    rows.append([sp.Integer(1) / (sp.Integer(k * k) + b_symbol) for k in nodes])
    return sp.Matrix(rows)


def expected_minor(nodes: list[int], s_value: int, b_symbol: sp.Symbol) -> sp.Expr:
    denominator = sp.prod(sp.Integer(k * k) + b_symbol for k in nodes)
    sign = sp.Integer(-1) ** (s_value + 1)
    return sp.factor(sign * vandermonde_for_squared_nodes(nodes) / denominator)


def verify_moment_independence(
    nodes: list[int], s_value: int, b_symbol: sp.Symbol
) -> dict[str, object]:
    determinant = sp.factor(pole_moment_matrix(nodes, s_value, b_symbol).det())
    expected = expected_minor(nodes, s_value, b_symbol)
    if sp.simplify(determinant - expected) != 0:
        raise SystemExit(
            "pole-moment determinant mismatch: "
            f"s={s_value}, nodes={nodes}, det={determinant}, expected={expected}"
        )
    if determinant == 0:
        raise SystemExit(f"pole-moment determinant vanished: s={s_value}")
    return {
        "s": s_value,
        "nodes": nodes,
        "determinant": str(determinant),
    }


def main() -> None:
    b_symbol = sp.symbols("b")
    pole_square_cases = [verify_pole_square(n_value, b_symbol) for n_value in range(21)]

    independence_cases = []
    for s_value in range(9):
        independence_cases.append(
            verify_moment_independence(list(range(s_value + 2)), s_value, b_symbol)
        )
        independence_cases.append(
            verify_moment_independence(list(range(1, s_value + 3)), s_value, b_symbol)
        )

    dimension_cases = []
    for s_value in range(9):
        for n_value in range(s_value + 2, s_value + 7):
            # exact rank computation over Q(b): stack the moment rows
            # M_0, M_2, ..., M_{2s} and the pole row (in v-coordinates; the
            # sqrt(2) column rescaling does not change any rank)
            rows = [[sp.Integer(1)] + [sp.Integer(1)] * n_value]           # M_0
            for j_value in range(1, s_value + 1):
                rows.append([sp.Integer(0)] + [sp.Integer(k ** (2 * j_value))
                                               for k in range(1, n_value + 1)])
            rows.append([1 / b_symbol] + [1 / (sp.Integer(k * k) + b_symbol)
                                          for k in range(1, n_value + 1)])  # pole row
            rank = sp.Matrix(rows).rank()
            kernel_dim = (n_value + 1) - rank
            if rank != s_value + 2 or kernel_dim != n_value - s_value - 1:
                raise SystemExit(
                    f"dimension formula failed at N={n_value}, s={s_value}: "
                    f"rank={rank}, kernel={kernel_dim}"
                )
            dimension_cases.append(
                {
                    "N": n_value,
                    "s": s_value,
                    "ambient_dimension": n_value + 1,
                    "independent_rows_computed_rank": rank,
                    "kernel_dimension_computed": kernel_dim,
                }
            )

    results = {
        "status": "passed",
        "field": "Q(b), b=beta^2",
        "pole_square_N_range": [0, 20],
        "moment_s_range": [0, 8],
        "pole_square_entries_checked": sum(
            item["entries_checked"] for item in pole_square_cases
        ),
        "pole_square_cases": pole_square_cases,
        "moment_independence_cases": independence_cases,
        "dimension_cases": dimension_cases,
    }
    Path("pole_neutral_survival_audit.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("pole-neutral survival audit passed")
    print(f"pole-square entries checked: {results['pole_square_entries_checked']}")
    print(f"moment independence cases: {len(independence_cases)}")
    print(f"dimension cases: {len(dimension_cases)}")


if __name__ == "__main__":
    main()
