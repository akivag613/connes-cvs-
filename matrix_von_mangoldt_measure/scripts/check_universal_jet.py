#!/usr/bin/env python3
"""
Guard for Proposition (Closed form of the universal jet), Section: Arithmetic rigidity.

Verifies EXACTLY (sympy, symbolic) that the closed form
    B_{r,N}(u0) = -(r!/u0^r) * sum_{j odd, 1<=j<=r} (-1)^(r-j) C(r-1,j-1) C_j,
    C_j = [omega^j] A_N(omega),
equals the direct definition B_{r,N} = -d^r/dtau^r|_0 A_N(tau/(u0+tau)),
entrywise, over a spread of matrix entries (m,n) and orders r. Also checks the
stated closed-form entries of C_j and the r=1,2 specializations
(B_1 = -2/u0 * 11^T, B_2 = +4/u0^2 * 11^T).

No dependencies beyond sympy. Exact; not a floating-point check.
"""
import json
import sympy as sp

w, tau, u0 = sp.symbols('w tau u0', positive=True)


def A_entry(m, n):
    if m == n:
        return 2*w*sp.cos(2*sp.pi*m*w)
    return (sp.sin(2*sp.pi*m*w) - sp.sin(2*sp.pi*n*w))/(sp.pi*(m-n))


def Cj_entry_series(m, n, j):
    return sp.series(A_entry(m, n), w, 0, j+1).removeO().coeff(w, j)


def Cj_entry_closed(m, n, j):
    if j % 2 == 0:
        return sp.Integer(0)
    ell = (j-1)//2
    if m != n:
        return (sp.Integer(-1)**ell*(2*sp.pi)**j/sp.factorial(j))*(sp.Integer(m)**j-sp.Integer(n)**j)/(sp.pi*(m-n))
    return sp.Integer(2)*sp.Integer(-1)**ell*(2*sp.pi*m)**(j-1)/sp.factorial(j-1)


def B_direct(m, n, r):
    f = A_entry(m, n).subs(w, tau/(u0+tau))
    return sp.simplify(-sp.diff(f, tau, r).subs(tau, 0))


def B_closed(m, n, r):
    s = 0
    for j in range(1, r+1):
        if j % 2 == 1:
            s += (-1)**(r-j)*sp.binomial(r-1, j-1)*Cj_entry_closed(m, n, j)
    return sp.simplify(-sp.factorial(r)/u0**r*s)


def main():
    entries = [(2, -1), (3, 3), (1, 0), (4, 2), (5, -3)]
    orders = [1, 2, 3, 4, 5]
    failures = []
    # Cj closed form vs series
    for (m, n) in entries:
        for j in range(1, 8):
            if sp.simplify(Cj_entry_series(m, n, j) - Cj_entry_closed(m, n, j)) != 0:
                failures.append(f"Cj mismatch m={m},n={n},j={j}")
    # B_closed vs B_direct
    for (m, n) in entries:
        for r in orders:
            if sp.simplify(B_direct(m, n, r) - B_closed(m, n, r)) != 0:
                failures.append(f"B mismatch m={m},n={n},r={r}")
    # r=1,2 specializations (entry value; 11^T has every entry 1)
    spec_ok = (B_closed(2, -1, 1) == sp.simplify(-2/u0)
               and B_closed(3, 3, 1) == sp.simplify(-2/u0)
               and B_closed(2, -1, 2) == sp.simplify(4/u0**2)
               and B_closed(1, 0, 2) == sp.simplify(4/u0**2))
    if not spec_ok:
        failures.append("r=1,2 specialization (-2/u0, +4/u0^2) failed")
    out = {
        "status": "PASS" if not failures else "FAIL",
        "description": "Exact symbolic verification of the closed form B_{r,N}(u0) "
                       "for the universal singular jet, entrywise over entries "
                       f"{entries} and orders {orders}, plus C_j closed-form entries "
                       "and the r=1,2 (-2/u0, +4/u0^2) specializations.",
        "entries": entries,
        "orders": orders,
        "failures": failures,
    }
    print(json.dumps(out, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
