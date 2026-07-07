#!/usr/bin/env python3
"""Lightweight verification checks for the finite CvS zero-source dictionary.

These checks do not prove the theorem.  They guard the constants and signs in
the single-source identity and in the pole normalization used in the paper.
"""

from __future__ import annotations

import cmath
import math
import random

import mpmath as mp


def symmetric_coefficients(n: int, rng: random.Random) -> dict[int, float]:
    coeffs = {0: rng.uniform(-1.0, 1.0)}
    for k in range(1, n + 1):
        value = rng.uniform(-1.0, 1.0)
        coeffs[k] = value
        coeffs[-k] = value
    return coeffs


def k_from_integral_coefficients(coeffs: dict[int, float], omega: float) -> complex:
    """Exact double-sum form of 2 int_0^omega T(t)T(omega-t) dt."""
    out = 0j
    keys = list(coeffs)
    for m in keys:
        for n in keys:
            if m == n:
                integral = omega
            else:
                integral = (
                    cmath.exp(2j * math.pi * (m - n) * omega) - 1
                ) / (2j * math.pi * (m - n))
            out += (
                2
                * coeffs[m]
                * coeffs[n]
                * cmath.exp(2j * math.pi * n * omega)
                * integral
            )
    return out


def k_from_divided_difference(coeffs: dict[int, float], omega: float) -> float:
    """Contraction of the sine-source divided-difference kernel."""
    out = 0.0
    keys = list(coeffs)
    for m in keys:
        for n in keys:
            if m == n:
                value = 2 * omega * math.cos(2 * math.pi * omega * m)
            else:
                value = (
                    math.sin(2 * math.pi * omega * m)
                    - math.sin(2 * math.pi * omega * n)
                ) / (math.pi * (m - n))
            out += coeffs[m] * coeffs[n] * value
    return out


def k_mpmath(coeffs: dict[int, float], omega: mp.mpf) -> mp.mpf:
    """High-precision divided-difference form of K_v(omega)."""
    out = mp.mpf("0")
    keys = list(coeffs)
    for m in keys:
        for n in keys:
            umun = mp.mpf(coeffs[m]) * mp.mpf(coeffs[n])
            if m == n:
                value = 2 * omega * mp.cos(2 * mp.pi * omega * m)
            else:
                value = (
                    mp.sin(2 * mp.pi * omega * m)
                    - mp.sin(2 * mp.pi * omega * n)
                ) / (mp.pi * (m - n))
            out += umun * value
    return out


def pole_side(coeffs: dict[int, float], l_value: mp.mpf) -> mp.mpf:
    return mp.quad(
        lambda y: 2 * mp.cosh(y / 2) * k_mpmath(coeffs, 1 - y / l_value),
        [0, l_value],
    )


def two_g_i_half(coeffs: dict[int, float], l_value: mp.mpf) -> mp.mpf:
    """2 g_v(i/2) from the definition g_v(i/2) = int_{-Delta}^{Delta}
    ghat_v(xi) e^{-pi xi} dxi with ghat_v(xi) = pi K_v(1-|xi|/Delta).
    The integrand e^{-pi xi} over the full two-sided support makes this a
    different integral from pole_side; they agree only through the evenness
    of ghat_v, which is what the check exercises."""
    delta = l_value / (2 * mp.pi)
    return 2 * mp.pi * mp.quad(
        lambda xi: mp.e ** (-mp.pi * xi) * k_mpmath(coeffs, 1 - abs(xi) / delta),
        [-delta, 0, delta],
    )


def pole_closed_form_square(coeffs: dict[int, float], c_value: mp.mpf,
                            l_value: mp.mpf) -> mp.mpf:
    """C_c beta^2 (sum_m u_m/(m^2+beta^2))^2: the closed-form pole square of
    Corollary 2.7, a third route independent of any y- or xi-integral."""
    beta2 = (l_value / (4 * mp.pi)) ** 2
    c_const = l_value * (mp.sqrt(c_value) + 1 / mp.sqrt(c_value) - 2) / (2 * mp.pi ** 2)
    row = mp.fsum(mp.mpf(coeffs[m]) / (m * m + beta2) for m in coeffs)
    return c_const * beta2 * row ** 2


def main() -> None:
    rng = random.Random(20260625)
    max_single_error = 0.0
    for n in (1, 2, 3, 5, 8):
        for omega in (0.07, 0.13, 0.37, 0.81, 1.0):
            for _ in range(10):
                coeffs = symmetric_coefficients(n, rng)
                k_int = k_from_integral_coefficients(coeffs, omega)
                k_dd = k_from_divided_difference(coeffs, omega)
                max_single_error = max(max_single_error, abs(k_int - k_dd))
    if max_single_error > 1e-10:
        raise SystemExit(f"single-source identity failed: {max_single_error:.3e}")

    mp.mp.dps = 60
    max_pole_error = mp.mpf("0")
    max_square_error = mp.mpf("0")
    for n in (1, 2, 4):
        coeffs = symmetric_coefficients(n, rng)
        for c_value in (mp.mpf("5"), mp.mpf("13")):
            l_value = mp.log(c_value)
            left = pole_side(coeffs, l_value)
            right = two_g_i_half(coeffs, l_value)
            square = pole_closed_form_square(coeffs, c_value, l_value)
            scale = max(mp.mpf("1"), abs(left), abs(right))
            max_pole_error = max(max_pole_error, abs(left - right) / scale)
            max_square_error = max(max_square_error, abs(left - square) / scale)
    if max_pole_error > mp.mpf("1e-40"):
        raise SystemExit(f"pole normalization failed: {mp.nstr(max_pole_error, 8)}")
    if max_square_error > mp.mpf("1e-40"):
        raise SystemExit(f"pole closed-form square failed: {mp.nstr(max_square_error, 8)}")

    print("finite dictionary checks passed")
    print(f"max single-source error: {max_single_error:.3e}")
    print(f"max pole relative error: {mp.nstr(max_pole_error, 8)}")
    print(f"max pole closed-form-square relative error: {mp.nstr(max_square_error, 8)}")


if __name__ == "__main__":
    main()
