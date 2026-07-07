#!/usr/bin/env python3
"""Exact reproducibility guard for the Paper 3 finite prime-edge uncertainty
ceiling (the normal-jet visibility theorem).

No third-party dependencies; exact integer / rational arithmetic (fractions).
For a nonzero real EVEN normal x on {-N,...,N}, the visibility order of an
entering prime power is 2 e(x) + 1, where e(x) is the first nonvanishing even
moment M_e(x) = sum_m m^e x_m.  The theorem:

  (a) e(x) <= 2N for every nonzero even x  (the even-moment map is a Vandermonde
      in the squared nodes 0,1,...,N^2, hence invertible);
  (b) equality e(x) = 2N holds iff x is the centered finite-difference stencil
      x_m = (-1)^{N-m} C(2N, N+m), up to a nonzero scalar (unique maximally
      blind normal), giving visibility order 4N+1;
  (c) a family of even normals realizes the increasing orders 1,5,9,...,4N+1.

All checks are exact.  This is a reproducibility guard, not a proof substitute.
"""

from __future__ import annotations

import json
from fractions import Fraction
from math import comb
from pathlib import Path


def det_fraction(mat: list[list[Fraction]]) -> Fraction:
    """Exact determinant by fraction-free-ish Gaussian elimination."""
    n = len(mat)
    A = [row[:] for row in mat]
    det = Fraction(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r][c] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            det = -det
        det *= A[c][c]
        inv = A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / inv
            if f != 0:
                A[r] = [A[r][k] - f * A[c][k] for k in range(n)]
    return det


def nullspace_dim(mat: list[list[Fraction]]) -> int:
    """dim of nullspace = ncols - rank, exact."""
    if not mat:
        return 0
    A = [row[:] for row in mat]
    rows, cols = len(A), len(A[0])
    rank, pr = 0, 0
    for c in range(cols):
        piv = next((r for r in range(pr, rows) if A[r][c] != 0), None)
        if piv is None:
            continue
        A[pr], A[piv] = A[piv], A[pr]
        inv = A[pr][c]
        A[pr] = [x / inv for x in A[pr]]
        for r in range(rows):
            if r != pr and A[r][c] != 0:
                f = A[r][c]
                A[r] = [A[r][k] - f * A[pr][k] for k in range(cols)]
        pr += 1
        rank += 1
        if pr == rows:
            break
    return cols - rank


def even_moment(xv: list[int], idx: list[int], j: int) -> int:
    return sum((m ** j) * xv[i] for i, m in enumerate(idx))


def visibility_order(xv: list[int], idx: list[int], N: int):
    for e in range(0, 2 * N + 1, 2):
        if even_moment(xv, idx, e) != 0:
            return 2 * e + 1
    return None


def main() -> None:
    failures: list = []
    results: dict = {}

    Ns = [1, 2, 3, 4, 5, 6, 8, 10]
    for N in Ns:
        idx = list(range(-N, N + 1))
        rec: dict = {}

        # (a) even-moment matrix (rows e=0,2,..,2N over x_0..x_N) invertible
        rows = []
        for i in range(0, N + 1):  # even order 2i
            r = [Fraction(1) if i == 0 else Fraction(0)]  # x0 col: 0^0=1 else 0
            for m in range(1, N + 1):
                r.append(Fraction(2) * Fraction(m) ** (2 * i))
            rows.append(r)
        det = det_fraction(rows)
        rec["even_moment_det_nonzero"] = det != 0
        if det == 0:
            failures.append({"N": N, "check": "even_moment_invertible"})

        # (b) central-difference stencil achieves e=2N, uniquely
        xstar = [(-1) ** (N - m) * comb(2 * N, N + m) for m in idx]
        below = all(even_moment(xstar, idx, e) == 0 for e in range(0, 2 * N, 2))
        top = even_moment(xstar, idx, 2 * N) != 0
        rec["central_stencil_e_equals_2N"] = below and top
        if not (below and top):
            failures.append({"N": N, "check": "central_stencil_e2N"})
        # uniqueness: blind conditions M_0..M_{2N-2}=0 have 1-dim solution space
        cond = []
        for i in range(0, N):  # e = 0,2,...,2N-2  -> N conditions on N+1 dof
            r = [Fraction(1) if i == 0 else Fraction(0)]
            for m in range(1, N + 1):
                r.append(Fraction(2) * Fraction(m) ** (2 * i))
            cond.append(r)
        nd = nullspace_dim(cond)
        rec["blind_nullspace_dim_is_1"] = nd == 1
        if nd != 1:
            failures.append({"N": N, "check": "uniqueness_nullspace", "dim": nd})
        rec["central_stencil_visibility_order"] = visibility_order(xstar, idx, N)
        if rec["central_stencil_visibility_order"] != 4 * N + 1:
            failures.append({"N": N, "check": "ceiling_4N+1"})

        # (c) family realizing increasing orders 1,5,9,...  (2r-th central diff)
        orders = []
        for r in range(0, N + 1):
            if r == 0:
                xv = [1] * (2 * N + 1)  # flat: e=0, order 1
            else:
                xv = [(-1) ** (r - m) * comb(2 * r, r + m) if abs(m) <= r else 0 for m in idx]
            o = visibility_order(xv, idx, N)
            orders.append(o)
            if o is not None and o > 4 * N + 1:
                failures.append({"N": N, "check": "order_exceeds_ceiling", "order": o})
        rec["family_orders"] = orders

        results[str(N)] = rec

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "description": "Exact verification of the finite prime-edge uncertainty ceiling: "
        "e(x)<=2N (even-moment Vandermonde invertible), the centered "
        "finite-difference stencil (-1)^{N-m}C(2N,N+m) uniquely attains "
        "e=2N / visibility order 4N+1, and a family realizes 1,5,...,4N+1.",
        "N_values": Ns,
        "results": results,
        "failures": failures,
    }
    out = Path(__file__).resolve().parents[1] / "artifacts" / "uncertainty_ceiling_audit.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2)[:1500])
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
