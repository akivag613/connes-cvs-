#!/usr/bin/env python3
"""Exact series audit for the finite zero-source dictionary.

This script is a reproducibility guard, not a proof substitute.  It attacks the
single-frequency identity by expanding the Volterra and divided-difference
expressions at omega = 0 and comparing the formal coefficients with exact
integer arithmetic.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def symmetric_integer_coefficients(n_value: int, rng: random.Random) -> dict[int, int]:
    coeffs = {0: rng.randint(-5, 5)}
    while coeffs[0] == 0:
        coeffs[0] = rng.randint(-5, 5)
    for k in range(1, n_value + 1):
        value = rng.randint(-5, 5)
        coeffs[k] = value
        coeffs[-k] = value
    return coeffs


def moment(coeffs: dict[int, int], order: int) -> int:
    return sum(value * (index**order) for index, value in coeffs.items())


def continuous_normalized_coefficient(coeffs: dict[int, int], k_value: int) -> int:
    """Coefficient core from 2 int T(t)T(omega-t) dt.

    Up to the common scalar 2(2*pi*i)^k/(k+1)!, the omega^(k+1)
    coefficient is sum_a M_a M_{k-a}.
    """
    moments = [moment(coeffs, j) for j in range(k_value + 1)]
    return sum(moments[a] * moments[k_value - a] for a in range(k_value + 1))


def divided_difference_normalized_coefficient(
    coeffs: dict[int, int], r_value: int
) -> int:
    """Coefficient core from the divided-difference side.

    Up to the common scalar 2(-1)^r(2*pi)^(2r)/(2r+1)!, the omega^(2r+1)
    coefficient is the exact double sum below.
    """
    power = 2 * r_value
    total = 0
    for m_value, u_m in coeffs.items():
        for n_value, u_n in coeffs.items():
            quotient_core = sum(
                (m_value**a) * (n_value ** (power - a))
                for a in range(power + 1)
            )
            total += u_m * u_n * quotient_core
    return total


def check_random_series_identities() -> dict:
    rng = random.Random(20260625)
    max_order = 30
    cases = 0
    coefficient_checks = 0
    even_power_zero_checks = 0
    for n_value in (1, 2, 4, 8, 12):
        for _ in range(25):
            coeffs = symmetric_integer_coefficients(n_value, rng)
            cases += 1
            for k_value in range(max_order + 1):
                continuous_core = continuous_normalized_coefficient(coeffs, k_value)
                if k_value % 2 == 1:
                    if continuous_core != 0:
                        raise SystemExit(
                            "odd derivative core failed to vanish: "
                            f"N={n_value}, k={k_value}, core={continuous_core}"
                        )
                    even_power_zero_checks += 1
                    continue

                r_value = k_value // 2
                divided_core = divided_difference_normalized_coefficient(
                    coeffs, r_value
                )
                if continuous_core != divided_core:
                    raise SystemExit(
                        "series coefficient mismatch: "
                        f"N={n_value}, k={k_value}, "
                        f"continuous={continuous_core}, divided={divided_core}"
                    )
                coefficient_checks += 1
    return {
        "random_seed": 20260625,
        "N_values": [1, 2, 4, 8, 12],
        "cases": cases,
        "max_order_k": max_order,
        "exact_coefficient_equalities": coefficient_checks,
        "exact_even_power_zero_checks": even_power_zero_checks,
    }


def main() -> None:
    results = {
        "status": "passed",
        "series_identity": check_random_series_identities(),
    }
    Path("exact_series_audit.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("exact series identity audit passed")
    print(
        "exact coefficient equalities: "
        f"{results['series_identity']['exact_coefficient_equalities']}"
    )
    print(
        "exact even-power zero checks: "
        f"{results['series_identity']['exact_even_power_zero_checks']}"
    )


if __name__ == "__main__":
    main()
