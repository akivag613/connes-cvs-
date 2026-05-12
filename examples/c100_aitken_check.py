"""Reproduce the headline c=100 Aitken-Δ² match against the Connes 2026 §6.4 prediction.

Loads the published N-sweep data from ``data/c100/`` and recomputes the Aitken-Δ²
limit of the sequence ``log_10 |lambda_N|`` for N in {100, 150, 200} at c=100.
Then evaluates the Connes 2026 §6.4 heuristic continuum asymptotic at c=100
and prints the agreement.

Runs in under a second on any machine. No mpmath required (the input JSONs
already carry full-precision decimal strings; this script only inspects their
exponents).

Usage:
    python examples/c100_aitken_check.py

Expected output:

    c=100 N-sweep: log_10 |lambda_N| at N = (100, 150, 200)
        N=100:  -190.92
        N=150:  -247.19
        N=200:  -294.31

    Aitken-Delta^2 extrapolation:
        log_10 |lambda_infinity (c=100)|  ≈  -536.97

    Connes 2026 §6.4 heuristic prediction at c=100:
        prefactor log_10[2^14 * sqrt(2*pi^5) / 3]  =   +5.13
        -4*pi*c / ln(10)                           =  -545.75
        +9 * log(c) / (2*ln(10))                   =   +9.00
        prediction log_10 |varepsilon(c=100)|      = ~ -531.62

    Agreement: |aitken - prediction|  ≈  5 OOM out of ~537 total
             (~ 1% of the exponent, out-of-sample).
"""
from __future__ import annotations
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "c100")


def log10_from_decimal_string(s: str) -> float:
    """Compute the base-10 log of a full-precision decimal string of an mpmath float.

    The Aitken arithmetic only needs the exponent; we use Python's ``float`` for
    the mantissa correction since 16-digit precision is more than sufficient.
    """
    s = s.strip().lower()
    if "e" in s:
        mantissa_str, exponent_str = s.split("e")
        return math.log10(float(mantissa_str)) + int(exponent_str)
    return math.log10(float(s))


def load_run(filename: str) -> dict:
    with open(os.path.join(DATA, filename)) as f:
        return json.load(f)


def aitken_delta2(x0: float, x1: float, x2: float) -> float:
    """Standard Aitken-Δ² acceleration starting from x0."""
    d1 = x1 - x0
    d2 = x2 - x1
    d2_d1 = d2 - d1
    if abs(d2_d1) < 1e-12:
        raise ValueError("Second difference is zero; Aitken-Δ² undefined.")
    return x0 - (d1 * d1) / d2_d1


def connes_2026_section_6_4_at(c: int) -> float:
    """Evaluate the Connes 2026 §6.4 heuristic prediction at integer cutoff c.

    Asymptotic (Connes 2026, eq just before §6.5):
        1 - chi^2(lambda) ~  (2^14 / 3) * sqrt(2*pi^5) * exp(-4*pi*e^L + 9*L/2)
    where L = log(c).
    """
    prefactor = math.log10((2**14) * math.sqrt(2 * math.pi**5) / 3.0)
    L = math.log(c)
    exponential_arg = -4.0 * math.pi * c / math.log(10.0)
    linear_arg = 9.0 * L / (2.0 * math.log(10.0))
    return prefactor + exponential_arg + linear_arg


def main() -> None:
    runs = {
        100: load_run("results_c100_N100_T800_dps500_v020.json"),
        150: load_run("results_c100_N150_T800_dps500_v020.json"),
        200: load_run("results_c100_N200_T800_dps500_v020.json"),
    }

    log10_lambda = {N: log10_from_decimal_string(run["lambda_even"])
                    for N, run in runs.items()}

    print("c=100 N-sweep: log_10 |lambda_N| at N = (100, 150, 200)")
    for N in (100, 150, 200):
        print(f"    N={N}:  {log10_lambda[N]:.2f}")
    print()

    aitken_limit = aitken_delta2(log10_lambda[100],
                                 log10_lambda[150],
                                 log10_lambda[200])
    print("Aitken-Delta^2 extrapolation:")
    print(f"    log_10 |lambda_infinity (c=100)|  ~  {aitken_limit:.2f}")
    print()

    pref = math.log10((2**14) * math.sqrt(2 * math.pi**5) / 3.0)
    exp_term = -4.0 * math.pi * 100 / math.log(10.0)
    lin_term = 9.0 * math.log(100) / (2.0 * math.log(10.0))
    prediction = pref + exp_term + lin_term

    print("Connes 2026 §6.4 heuristic prediction at c=100:")
    print(f"    prefactor log_10[2^14 * sqrt(2*pi^5) / 3]  =  {pref:+.2f}")
    print(f"    -4*pi*c / ln(10)                           =  {exp_term:+.2f}")
    print(f"    +9 * log(c) / (2*ln(10))                   =  {lin_term:+.2f}")
    print(f"    prediction log_10 |varepsilon(c=100)|      = ~ {prediction:.2f}")
    print()

    gap = abs(aitken_limit - prediction)
    fraction = gap / abs(aitken_limit) * 100.0
    print("Agreement:")
    print(f"    |aitken - prediction|  =  {gap:.2f} OOM "
          f"out of ~{abs(aitken_limit):.0f} total")
    print(f"    ({fraction:.2f}% of the exponent, out-of-sample).")


if __name__ == "__main__":
    main()
