"""Reproduce the c=100 Aitken-Δ² stress test of the Connes 2026 §6.4 asymptotic.

Loads the published N-sweep data from ``data/c100/`` and recomputes:
  - log_10 |lambda_N| at N in {100, 150, 200, 250}
  - successive first differences and consecutive ratios
    (the geometric-convergence consistency test from Paper §6.4)
  - Aitken-Δ² applied to two overlapping triples:
        (100, 150, 200)  — original three-point Aitken
        (150, 200, 250)  — deeper anchor with N=250 datum
  - Connes 2026 §6.4 heuristic continuum prediction at c=100
  - the gap of each Aitken anchor to the Connes prediction

Runs in under a second on any machine. No mpmath required (the input JSONs
already carry full-precision decimal strings; this script only inspects
their exponents and uses Python's float for the Aitken arithmetic).

Usage:
    python examples/c100_aitken_check.py

Expected output:

    c=100 N-sweep: log_10 |lambda_N| at N = (100, 150, 200, 250)
        N=100:  -190.92
        N=150:  -247.19
        N=200:  -294.31
        N=250:  -333.68

    First differences and consecutive ratios:
        |Delta_1| = 56.28   (N=100 -> 150)
        |Delta_2| = 47.12   (N=150 -> 200)    Delta_2 / Delta_1 = 0.8373
        |Delta_3| = 39.37   (N=200 -> 250)    Delta_3 / Delta_2 = 0.8355

    Aitken-Delta^2 extrapolations (two consecutive triples):
        from (100, 150, 200): log_10 |lambda_infinity (c=100)| ~ -536.76
        from (150, 200, 250): log_10 |lambda_infinity (c=100)| ~ -533.70

    Connes 2026 §6.4 heuristic prediction at c=100:
        prefactor log_10[2^14 * sqrt(2) * pi^5 / 3] =   +6.37
        -4*pi*c / ln(10)                            =  -545.75
        +9 * log(c) / (2*ln(10))                    =   +9.00
        prediction log_10 |varepsilon(c=100)|       = ~ -530.38

    Gaps from the Connes prediction:
        first triple anchor:   6.39 OOM
        second triple anchor:  3.32 OOM   (deeper Aitken anchor is closer)

The consistency of the two consecutive ratios (0.8373 and 0.8355) is
evidence for a local geometric model of the convergence sequence; this
does not rule out stretched-exponential or polynomial alternatives, only
shows that local geometric convergence fits the data without strain.
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

    Asymptotic (Connes 2026, displayed equation immediately before footnote 19;
    LaTeX in the arXiv HTML source reads ``\\sqrt{2}\\pi^{5}`` — the radical
    covers only the 2, then is multiplied by pi^5):
        1 - chi_2(lambda) ~  (2^14 / 3) * sqrt(2) * pi^5 * exp(-4*pi*e^L + 9*L/2)
    where L = 2*log(lambda) = log(c) at lambda = sqrt(c).
    """
    prefactor = math.log10((2**14) * math.sqrt(2) * (math.pi**5) / 3.0)
    L = math.log(c)
    exponential_arg = -4.0 * math.pi * c / math.log(10.0)
    linear_arg = 9.0 * L / (2.0 * math.log(10.0))
    return prefactor + exponential_arg + linear_arg


def main() -> None:
    runs = {
        100: load_run("results_c100_N100_T800_dps500_v020.json"),
        150: load_run("results_c100_N150_T800_dps500_v020.json"),
        200: load_run("results_c100_N200_T800_dps500_v020.json"),
        250: load_run("results_c100_N250_T800_dps500_v020.json"),
    }

    log10_lambda = {N: log10_from_decimal_string(run["lambda_even"])
                    for N, run in runs.items()}

    print("c=100 N-sweep: log_10 |lambda_N| at N = (100, 150, 200, 250)")
    for N in (100, 150, 200, 250):
        print(f"    N={N}:  {log10_lambda[N]:.2f}")
    print()

    # Consecutive first differences (per-step gap in log10 |lambda_N|)
    d1 = log10_lambda[150] - log10_lambda[100]
    d2 = log10_lambda[200] - log10_lambda[150]
    d3 = log10_lambda[250] - log10_lambda[200]

    print("First differences and consecutive ratios:")
    print(f"    |Delta_1| = {abs(d1):.2f}   (N=100 -> 150)")
    print(f"    |Delta_2| = {abs(d2):.2f}   (N=150 -> 200)    "
          f"Delta_2 / Delta_1 = {d2/d1:.4f}")
    print(f"    |Delta_3| = {abs(d3):.2f}   (N=200 -> 250)    "
          f"Delta_3 / Delta_2 = {d3/d2:.4f}")
    print()

    a1 = aitken_delta2(log10_lambda[100], log10_lambda[150], log10_lambda[200])
    a2 = aitken_delta2(log10_lambda[150], log10_lambda[200], log10_lambda[250])

    print("Aitken-Delta^2 extrapolations (two consecutive triples):")
    print(f"    from (100, 150, 200): log_10 |lambda_infinity (c=100)| ~ {a1:.2f}")
    print(f"    from (150, 200, 250): log_10 |lambda_infinity (c=100)| ~ {a2:.2f}")
    print()

    prediction = connes_2026_section_6_4_at(100)
    pref = math.log10((2**14) * math.sqrt(2) * (math.pi**5) / 3.0)
    exp_term = -4.0 * math.pi * 100 / math.log(10.0)
    lin_term = 9.0 * math.log(100) / (2.0 * math.log(10.0))

    print("Connes 2026 §6.4 heuristic prediction at c=100:")
    print(f"    prefactor log_10[2^14 * sqrt(2) * pi^5 / 3] =  {pref:+.2f}")
    print(f"    -4*pi*c / ln(10)                           =  {exp_term:+.2f}")
    print(f"    +9 * log(c) / (2*ln(10))                   =  {lin_term:+.2f}")
    print(f"    prediction log_10 |varepsilon(c=100)|      = ~ {prediction:.2f}")
    print()

    gap1 = abs(a1 - prediction)
    gap2 = abs(a2 - prediction)

    print("Gaps from the Connes prediction:")
    print(f"    first triple anchor (100-150-200):  {gap1:.2f} OOM")
    print(f"    second triple anchor (150-200-250): {gap2:.2f} OOM   "
          f"(deeper Aitken anchor is closer)")
    print()
    print("The consistency of the two consecutive ratios "
          f"({d2/d1:.4f} and {d3/d2:.4f}) is evidence for")
    print("a local geometric model of the convergence sequence; this does NOT rule")
    print("out stretched-exponential or polynomial alternatives, only shows that")
    print("local geometric convergence fits the data without strain.")


if __name__ == "__main__":
    main()
