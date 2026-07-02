#!/usr/bin/env python3
"""Reproducibility guard for the finite-T archimedean tail theorem.

This script is not a proof substitute.  It checks the algebraic formulas used
in the tail-order section by three deterministic tests:

1. Arb interval evaluation of h_+(7), plus a certified bracket for the scalar
   positivity threshold.
2. Closed-form divided-difference factorization of the single-frequency
   density into two Cauchy channels.
3. A small numerical strict-total-positivity smoke test for an integrated
   tail increment.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import mpmath as mp
from flint import acb, arb, ctx


def h_plus_arb(tau: arb) -> arb:
    z = acb(arb("0.25"), tau / 2)
    return z.digamma().real - arb.pi().log()


def h_plus_mp(tau: mp.mpf) -> mp.mpf:
    return mp.re(mp.digamma(mp.mpf("0.25") + 0.5j * tau)) - mp.log(mp.pi)


def threshold_bracket() -> dict:
    lo = arb("6.289835988836902")
    hi = arb("6.289835988836903")
    hlo = h_plus_arb(lo)
    hhi = h_plus_arb(hi)
    return {
        "lower": str(lo),
        "upper": str(hi),
        "h_lower": str(hlo).replace("\n", " "),
        "h_upper": str(hhi).replace("\n", " "),
        "certified_bracket": bool(hlo.upper() < 0 and hhi.lower() > 0),
    }


def s_formula(c: int, tau: mp.mpf, x: mp.mpf) -> mp.mpf:
    L = mp.log(c)
    rho = 2 * mp.pi / L
    if x == 0:
        return mp.mpf("0")
    return 2 * rho * x * mp.sin(L * tau / 2) ** 2 / (tau**2 - (rho * x) ** 2)


def s_integral(c: int, tau: mp.mpf, x: mp.mpf) -> mp.mpf:
    L = mp.log(c)

    def integrand(y: mp.mpf) -> mp.mpf:
        return mp.sin(2 * mp.pi * x * (1 - y / L)) * mp.cos(tau * y)

    pieces = [L * k / 80 for k in range(81)]
    return mp.quad(integrand, pieces)


def ds_integral(c: int, tau: mp.mpf, x: mp.mpf) -> mp.mpf:
    L = mp.log(c)

    def integrand(y: mp.mpf) -> mp.mpf:
        return (
            2
            * mp.pi
            * (1 - y / L)
            * mp.cos(2 * mp.pi * x * (1 - y / L))
            * mp.cos(tau * y)
        )

    pieces = [L * k / 80 for k in range(81)]
    return mp.quad(integrand, pieces)


def density_checks() -> dict:
    mp.mp.dps = 70
    cells = [
        {"c": 13, "N": 3, "T": mp.mpf("20"), "quadrature": True},
        {"c": 100, "N": 5, "T": mp.mpf("30"), "quadrature": False},
        {"c": 100, "N": 8, "T": mp.mpf("800"), "quadrature": False},
    ]
    rows = []
    max_formula_quad = mp.mpf("0")
    max_diag_quad = mp.mpf("0")
    max_factorization = mp.mpf("0")

    for cell in cells:
        c = cell["c"]
        nmax = cell["N"]
        tau = cell["T"]
        L = mp.log(c)
        rho = 2 * mp.pi / L
        a = tau / rho
        nodes = list(range(-nmax, nmax + 1))
        s2 = mp.sin(L * tau / 2) ** 2
        vals = {}
        direct = {}

        for n in nodes:
            x = mp.mpf(n)
            vals[n] = s_formula(c, tau, x)
            if cell["quadrature"]:
                max_formula_quad = max(max_formula_quad, abs(vals[n] - s_integral(c, tau, x)))
                diag_formula = (
                    2
                    * rho
                    * s2
                    * (tau**2 + (rho * x) ** 2)
                    / (tau**2 - (rho * x) ** 2) ** 2
                )
                max_diag_quad = max(max_diag_quad, abs(diag_formula - ds_integral(c, tau, x)))

        for m in nodes:
            for n in nodes:
                if m == n:
                    direct[m, n] = (
                        2
                        * s2
                        / rho
                        * (a**2 + m**2)
                        / (a**2 - m**2) ** 2
                    )
                else:
                    direct[m, n] = (vals[m] - vals[n]) / (m - n)

                gram = s2 / rho * (
                    1 / ((a - m) * (a - n)) + 1 / ((a + m) * (a + n))
                )
                max_factorization = max(max_factorization, abs(direct[m, n] - gram))

        rows.append(
            {
                "c": c,
                "N": nmax,
                "T": mp.nstr(tau, 20),
                "T_over_rho": mp.nstr(a, 30),
                "threshold_pass": bool(tau > rho * nmax and tau > 7),
            }
        )

    return {
        "cells": rows,
        "max_S_formula_vs_quadrature_abs_error": mp.nstr(max_formula_quad, 20),
        "max_diagonal_derivative_vs_quadrature_abs_error": mp.nstr(max_diag_quad, 20),
        "max_divided_difference_factorization_abs_error": mp.nstr(max_factorization, 20),
    }


def strict_total_positivity_case(nmax: int, support_count: int) -> dict:
    mp.mp.dps = 80
    c = 13
    L = mp.log(c)
    rho = 2 * mp.pi / L
    T1 = mp.mpf("20")
    T2 = mp.mpf("22")
    A = T1 / rho
    B = T2 / rho
    nodes = list(range(-nmax, nmax + 1))
    support_positive = [
        A + (B - A) * mp.mpf(k) / (support_count + 1)
        for k in range(1, support_count + 1)
    ]
    support = [-a for a in reversed(support_positive)] + support_positive
    weights = {}
    for s in support:
        a = abs(s)
        T = rho * a
        weights[s] = h_plus_mp(T) * mp.sin(L * T / 2) ** 2 / (mp.pi**2 * rho)
        if weights[s] <= 0:
            raise AssertionError("chosen finite support weight is not positive")

    dim = len(nodes)
    mat = mp.matrix(dim)
    for i, m in enumerate(nodes):
        for j, n in enumerate(nodes):
            mat[i, j] = mp.fsum(
                weights[s] / ((s - m) * (s - n))
                for s in support
            )

    min_minor = None
    bad = []
    count = 0
    for k in range(1, dim + 1):
        for rows in itertools.combinations(range(dim), k):
            for cols in itertools.combinations(range(dim), k):
                sub = mp.matrix([[mat[i, j] for j in cols] for i in rows])
                det = mp.det(sub)
                count += 1
                if min_minor is None or det < min_minor:
                    min_minor = det
                if det <= 0:
                    bad.append({"k": k, "rows": rows, "cols": cols, "det": mp.nstr(det, 20)})

    eigvals = mp.eigsy(mat, eigvals_only=True)
    return {
        "c": c,
        "N": nmax,
        "T1": mp.nstr(T1, 20),
        "T2": mp.nstr(T2, 20),
        "positive_support_points": [mp.nstr(a, 20) for a in support_positive],
        "support_count_per_side": support_count,
        "minor_count": count,
        "bad_minor_count": len(bad),
        "minimum_minor": mp.nstr(min_minor, 20),
        "smallest_eigenvalue": mp.nstr(min(eigvals), 20),
        "bad_minors_sample": bad[:5],
    }


def strict_total_positivity_smoke() -> dict:
    cases = [
        strict_total_positivity_case(nmax=2, support_count=6),
        strict_total_positivity_case(nmax=3, support_count=8),
    ]
    return {
        "cases": cases,
        "bad_minor_count": sum(case["bad_minor_count"] for case in cases),
        "minimum_minor_over_cases": mp.nstr(
            min(mp.mpf(case["minimum_minor"]) for case in cases), 20
        ),
    }


def main() -> None:
    ctx.prec = 300
    h7 = h_plus_arb(arb(7))
    bracket = threshold_bracket()
    result = {
        "status": "PASS",
        "h_plus_7_interval": str(h7).replace("\n", " "),
        "h_plus_7_positive": bool(h7.lower() > 0),
        "h_plus_zero_bracket": bracket,
        "density_checks": density_checks(),
        "strict_total_positivity_smoke": strict_total_positivity_smoke(),
    }
    if not result["h_plus_7_positive"]:
        result["status"] = "FAIL"
    if not bracket["certified_bracket"]:
        result["status"] = "FAIL"
    if result["strict_total_positivity_smoke"]["bad_minor_count"]:
        result["status"] = "FAIL"
    text = json.dumps(result, indent=2, sort_keys=True)
    Path("arch_tail_order_check.json").write_text(text + "\n")
    print("archimedean tail-order checks passed")
    print(f"h_+(7): {result['h_plus_7_interval']}")
    print(
        "h_+ zero bracket:",
        f"[{bracket['lower']}, {bracket['upper']}]",
    )
    print(
        "max factorization abs error:",
        result["density_checks"]["max_divided_difference_factorization_abs_error"],
    )
    print(
        "strict TP smoke bad minors:",
        result["strict_total_positivity_smoke"]["bad_minor_count"],
    )


if __name__ == "__main__":
    main()
