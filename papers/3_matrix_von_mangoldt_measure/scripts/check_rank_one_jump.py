#!/usr/bin/env python3
"""Genuine reproducibility guard for the Paper 3 rank-one prime-power jump.

No third-party dependencies (Python standard library only, so a referee can run
it in isolation).  Unlike a formula-restating stub, this script EVALUATES the
actual divided-difference entries with math.sin/math.cos and checks the theorem
two ways:

  (1) Edge derivative (Lemma):  A_N(0)=0 and A_N'(0)=2 in every entry, for a
      range of N, verified by a central finite difference of the real trig
      formula (NOT by asserting the closed form).

  (2) First-jump theorem (Theorem):  for the actual finite prime-source path
      P_N(u) = - sum_{q=p^a <= e^u} (Lambda(q)/sqrt q) A_N(1 - log q / u),
      the right-minus-left first-derivative jump at u = log q equals
      -2 Lambda(q) / (sqrt q * log q) * 11^T, verified by finite differences of
      the assembled sum, for several prime powers q (primes and true powers).

Both are checked to a numerical tolerance; a wrong sign, factor, or index would
fail.  This is a reproducibility guard, not a proof substitute.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PI = math.pi


def A_entry(m: int, n: int, omega: float) -> float:
    """(A_N(omega))_{mn} from the actual divided-difference formula."""
    if m == n:
        return 2.0 * omega * math.cos(2.0 * PI * m * omega)
    return (math.sin(2.0 * PI * m * omega) - math.sin(2.0 * PI * n * omega)) / (
        PI * (m - n)
    )


def von_mangoldt(k: int) -> float:
    if k < 2:
        return 0.0
    r, p, factors = k, 2, {}
    while p * p <= r:
        while r % p == 0:
            factors[p] = factors.get(p, 0) + 1
            r //= p
        p += 1
    if r > 1:
        factors[r] = factors.get(r, 0) + 1
    return math.log(next(iter(factors))) if len(factors) == 1 else 0.0


def prime_part(N: int, u: float) -> list[list[float]]:
    """P_N(u): only the finite prime-source part (analytic background cancels
    in the jump). Sum over prime powers q with log q <= u (exact threshold)."""
    d = 2 * N + 1
    idx = list(range(-N, N + 1))
    M = [[0.0] * d for _ in range(d)]
    k = 2
    while math.log(k) <= u + 1e-15:
        Lam = von_mangoldt(k)
        if Lam != 0.0:
            omega = 1.0 - math.log(k) / u
            w = -Lam / math.sqrt(k)
            for i, m in enumerate(idx):
                for j, n in enumerate(idx):
                    M[i][j] += w * A_entry(m, n, omega)
        k += 1
    return M


def main() -> None:
    failures: list[dict] = []
    results: dict = {}

    # ---- (1) edge derivative A_N(0)=0, A_N'(0)=2, real central difference ----
    h = 1e-6
    edge = {}
    for N in [1, 2, 3, 4, 5, 6]:
        max_val0 = 0.0
        max_der_err = 0.0
        for m in range(-N, N + 1):
            for n in range(-N, N + 1):
                v0 = A_entry(m, n, 0.0)
                der = (A_entry(m, n, h) - A_entry(m, n, -h)) / (2 * h)
                max_val0 = max(max_val0, abs(v0))
                max_der_err = max(max_der_err, abs(der - 2.0))
        edge[str(N)] = {"max|A_N(0)|": max_val0, "max|A_N'(0)-2|": max_der_err}
        if max_val0 > 1e-9 or max_der_err > 1e-4:
            failures.append({"check": "edge_derivative", "N": N})
    results["edge_derivative"] = edge

    # ---- (2) first-jump theorem end-to-end, several prime powers ----
    N = 2
    d = 2 * N + 1
    hh = 1e-6
    jump = {}
    for q in [3, 4, 5, 7, 8, 9, 25]:
        u0 = math.log(q)
        Lp = prime_part(N, u0 - hh)
        Lp2 = prime_part(N, u0 - 2 * hh)
        Rp = prime_part(N, u0 + hh)
        Rp2 = prime_part(N, u0 + 2 * hh)
        # one-sided slopes just left and just right of the edge
        left = [[(Lp[i][j] - Lp2[i][j]) / hh for j in range(d)] for i in range(d)]
        right = [[(Rp2[i][j] - Rp[i][j]) / hh for j in range(d)] for i in range(d)]
        expected = -2.0 * von_mangoldt(q) / (math.sqrt(q) * math.log(q))
        num = 0.0
        den = 0.0
        for i in range(d):
            for j in range(d):
                jij = right[i][j] - left[i][j]
                num = max(num, abs(jij - expected))  # expected in every entry (11^T)
                den = max(den, abs(expected))
        rel = num / den if den else num
        jump[str(q)] = {"Lambda": von_mangoldt(q), "rel_err_vs_-2Lam/(sqrt q log q)*11^T": rel}
        if rel > 1e-3:
            failures.append({"check": "first_jump", "q": q, "rel_err": rel})
    results["first_jump"] = jump

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "description": "Genuine numeric verification of A_N(0)=0, A_N'(0)=2 (N<=6) "
        "and the -2 Lambda(q)/(sqrt q log q) 11^T first-derivative jump "
        "(q in {3,4,5,7,8,9,25}) from the assembled prime-source path.",
        "results": results,
        "failures": failures,
    }
    out = Path(__file__).resolve().parents[1] / "artifacts" / "rank_one_jump_audit.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
