#!/usr/bin/env python3
"""Audit the finite-T derivative bridge for the archimedean tail theorem.

The manuscript proves the archimedean tail order by differentiating the
finite-T archimedean source term in the finite dictionary.  This script checks
that bridge by three independent routes:

1. symbolic divided-difference algebra for the off-diagonal and confluent
   diagonal identities;
2. direct high-precision comparison between the differentiated source entry
   and the rank-two Cauchy density;
3. a crude central finite-difference check through the finite-T archimedean
   quadrature convention.

These are reproducibility guards for constants and signs.  They are not proof
substitutes.
"""

from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import sympy as sp


def h_plus_mp(tau: mp.mpf) -> mp.mpf:
    return mp.re(mp.digamma(mp.mpf("0.25") + 0.5j * tau)) - mp.log(mp.pi)


def stable_A(beta: mp.mpf, L: mp.mpf) -> mp.mpc:
    beta = mp.mpf(beta)
    if beta == 0:
        return mp.mpc(L, 0)
    bL = beta * L
    sin_half = mp.sin(bL / 2)
    sin_full = mp.sin(bL)
    return mp.mpc(sin_full / beta, 2 * sin_half * sin_half / beta)


def stable_B(beta: mp.mpf, L: mp.mpf) -> mp.mpc:
    beta = mp.mpf(beta)
    if beta == 0:
        return mp.mpc(L / 2, 0)
    bL = beta * L
    sin_half = mp.sin(bL / 2)
    sin_full = mp.sin(bL)
    real_part = 2 * sin_half * sin_half / (L * beta * beta)
    if abs(bL) < mp.mpf("1e-5"):
        bL2 = bL * bL
        correction = 1 - bL2 / 20 * (1 - bL2 / 42 *
                     (1 - bL2 / 72 * (1 - bL2 / 110)))
        imag_part = beta * L * L / 6 * correction
    else:
        imag_part = (bL - sin_full) / (L * beta * beta)
    return mp.mpc(real_part, imag_part)


def source_kernel_pair(tau: mp.mpf, x: mp.mpf, L: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """Return Re(S_hat(tau,x,L)) and Re(dS_hat/dx)."""
    x = mp.mpf(x)
    tau = mp.mpf(tau)
    alpha = 2 * mp.pi * x / L
    s2pi = mp.sin(2 * mp.pi * x)
    c2pi = mp.cos(2 * mp.pi * x)

    A_plus = stable_A(alpha - tau, L)
    A_minus = stable_A(-(alpha + tau), L)
    I_c = (A_plus + A_minus) / 2
    I_s = (A_plus - A_minus) / (2j)
    re_S = mp.re(s2pi * I_c - c2pi * I_s)

    B_plus = stable_B(alpha - tau, L)
    B_minus = stable_B(-(alpha + tau), L)
    C = c2pi * (B_plus + B_minus) / 2 + s2pi * (B_plus - B_minus) / (2j)
    re_dS = mp.re(2 * mp.pi * C)
    return re_S, re_dS


def s_integer_formula(c: int, T: mp.mpf, x: mp.mpf) -> mp.mpf:
    L = mp.log(c)
    rho = 2 * mp.pi / L
    x = mp.mpf(x)
    if x == 0:
        return mp.mpf("0")
    return 2 * rho * x * mp.sin(L * T / 2) ** 2 / (T**2 - (rho * x) ** 2)


def ds_integer_formula(c: int, T: mp.mpf, x: mp.mpf) -> mp.mpf:
    L = mp.log(c)
    rho = 2 * mp.pi / L
    x = mp.mpf(x)
    s2 = mp.sin(L * T / 2) ** 2
    return 2 * rho * s2 * (T**2 + (rho * x) ** 2) / (T**2 - (rho * x) ** 2) ** 2


def source_dt_entry(c: int, m: int, n: int, T: mp.mpf) -> mp.mpf:
    h = h_plus_mp(T)
    if m == n:
        return h * ds_integer_formula(c, T, m) / (mp.pi**2)
    return h * (s_integer_formula(c, T, m) - s_integer_formula(c, T, n)) / (
        (m - n) * mp.pi**2
    )


def cauchy_tail_density(c: int, m: int, n: int, T: mp.mpf) -> mp.mpf:
    L = mp.log(c)
    rho = 2 * mp.pi / L
    a = T / rho
    h = h_plus_mp(T)
    s2 = mp.sin(L * T / 2) ** 2
    return h * s2 / (mp.pi**2 * rho) * (
        1 / ((a - m) * (a - n)) + 1 / ((a + m) * (a + n))
    )


def symbolic_checks() -> dict:
    a, u, v = sp.symbols("a u v")
    f_u = u / (a**2 - u**2)
    f_v = v / (a**2 - v**2)
    cauchy_uv = sp.Rational(1, 2) * (
        1 / ((a - u) * (a - v)) + 1 / ((a + u) * (a + v))
    )
    cauchy_diag = sp.Rational(1, 2) * (
        1 / (a - u) ** 2 + 1 / (a + u) ** 2
    )
    off_residual = sp.factor(sp.simplify((f_u - f_v) / (u - v) - cauchy_uv))
    diag_residual = sp.factor(sp.simplify(sp.diff(f_u, u) - cauchy_diag))
    return {
        "off_diagonal_residual": str(off_residual),
        "diagonal_residual": str(diag_residual),
        "passed": bool(off_residual == 0 and diag_residual == 0),
    }


def direct_kernel_checks() -> dict:
    mp.mp.dps = 120
    cells = [
        (13, -3, -1, mp.mpf("20")),
        (13, -2, 2, mp.mpf("20")),
        (13, 0, 0, mp.mpf("20")),
        (100, -5, 4, mp.mpf("30")),
        (100, -8, 7, mp.mpf("800")),
    ]
    rows = []
    max_abs = mp.mpf("0")
    max_rel = mp.mpf("0")
    for c, m, n, T in cells:
        source_val = source_dt_entry(c, m, n, T)
        cauchy_val = cauchy_tail_density(c, m, n, T)
        abs_error = abs(source_val - cauchy_val)
        denom = max(abs(source_val), abs(cauchy_val), mp.mpf("1e-200"))
        rel_error = abs_error / denom
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
        rows.append(
            {
                "c": c,
                "m": m,
                "n": n,
                "T": mp.nstr(T, 20),
                "source_dt_entry": mp.nstr(source_val, 40),
                "cauchy_density": mp.nstr(cauchy_val, 40),
                "abs_error": mp.nstr(abs_error, 20),
                "rel_error": mp.nstr(rel_error, 20),
            }
        )
    return {
        "precision_dps": mp.mp.dps,
        "cells": rows,
        "max_abs_error": mp.nstr(max_abs, 20),
        "max_rel_error": mp.nstr(max_rel, 20),
        "passed": bool(max_abs < mp.mpf("1e-90") and max_rel < mp.mpf("1e-85")),
    }


def integration_breaks(L: mp.mpf, x: mp.mpf, T: mp.mpf) -> list[mp.mpf]:
    alpha = abs(2 * mp.pi * mp.mpf(x) / L)
    pts = [mp.mpf("0")]
    if 0 < alpha < T:
        pts.append(alpha)
    pts.append(T)
    pts = sorted(set(pts))
    return pts


def psi_arch_T(c: int, x: int, T: mp.mpf, dps: int) -> mp.mpf:
    mp.mp.dps = dps
    L = mp.log(c)
    x_mp = mp.mpf(x)
    if x_mp == 0:
        return mp.mpf("0")
    pts = integration_breaks(L, x_mp, T)

    def integrand(tau: mp.mpf) -> mp.mpf:
        re_S, _ = source_kernel_pair(tau, x_mp, L)
        return h_plus_mp(tau) * re_S

    return mp.fsum(mp.quad(integrand, [pts[i], pts[i + 1]]) for i in range(len(pts) - 1)) / (
        mp.pi**2
    )


def psi_arch_deriv_T(c: int, x: int, T: mp.mpf, dps: int) -> mp.mpf:
    mp.mp.dps = dps
    L = mp.log(c)
    x_mp = mp.mpf(x)
    pts = integration_breaks(L, x_mp, T)

    def integrand(tau: mp.mpf) -> mp.mpf:
        _, re_dS = source_kernel_pair(tau, x_mp, L)
        return h_plus_mp(tau) * re_dS

    return mp.fsum(mp.quad(integrand, [pts[i], pts[i + 1]]) for i in range(len(pts) - 1)) / (
        mp.pi**2
    )


def finite_T_entry(c: int, m: int, n: int, T: mp.mpf, dps: int) -> mp.mpf:
    if m == n:
        return psi_arch_deriv_T(c, m, T, dps)
    return (psi_arch_T(c, m, T, dps) - psi_arch_T(c, n, T, dps)) / (m - n)


def finite_difference_checks() -> dict:
    mp.mp.dps = 50
    dps = 50
    h = mp.mpf("1e-4")
    cells = [
        (13, -2, 1, mp.mpf("20")),
        (13, 0, 0, mp.mpf("20")),
        (100, -2, 3, mp.mpf("30")),
        (100, 2, 2, mp.mpf("30")),
    ]
    rows = []
    max_rel = mp.mpf("0")
    max_abs = mp.mpf("0")
    for c, m, n, T in cells:
        fd = (finite_T_entry(c, m, n, T + h, dps) - finite_T_entry(c, m, n, T - h, dps)) / (
            2 * h
        )
        density = cauchy_tail_density(c, m, n, T)
        abs_error = abs(fd - density)
        denom = max(abs(fd), abs(density), mp.mpf("1e-100"))
        rel_error = abs_error / denom
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
        rows.append(
            {
                "c": c,
                "m": m,
                "n": n,
                "T": mp.nstr(T, 20),
                "step": mp.nstr(h, 20),
                "finite_difference": mp.nstr(fd, 30),
                "cauchy_density": mp.nstr(density, 30),
                "abs_error": mp.nstr(abs_error, 20),
                "rel_error": mp.nstr(rel_error, 20),
            }
        )
    return {
        "precision_dps": dps,
        "step": mp.nstr(h, 20),
        "cells": rows,
        "max_abs_error": mp.nstr(max_abs, 20),
        "max_rel_error": mp.nstr(max_rel, 20),
        "passed": bool(max_rel < mp.mpf("1e-4")),
    }


def main() -> None:
    symbolic = symbolic_checks()
    direct = direct_kernel_checks()
    finite_diff = finite_difference_checks()
    result = {
        "status": "PASS" if symbolic["passed"] and direct["passed"] and finite_diff["passed"] else "FAIL",
        "symbolic_checks": symbolic,
        "direct_kernel_checks": direct,
        "finite_difference_checks": finite_diff,
    }
    Path("arch_tail_dt_bridge_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("archimedean dT bridge audit:", result["status"])
    print("symbolic off-diagonal residual:", symbolic["off_diagonal_residual"])
    print("symbolic diagonal residual:", symbolic["diagonal_residual"])
    print("direct max abs error:", direct["max_abs_error"])
    print("direct max rel error:", direct["max_rel_error"])
    print("finite-difference max rel error:", finite_diff["max_rel_error"])
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
