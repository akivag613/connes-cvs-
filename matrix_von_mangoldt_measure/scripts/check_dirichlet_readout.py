#!/usr/bin/env python3
"""
Guard for Corollary (Residue-class readout across the Dirichlet family), Section: scope.

Verifies the identity
   -(sqrt q log q /2) * (1/phi(m)) sum_{chi mod m} conj(chi(a)) j^{(chi)}
        = Lambda(q) * 1[q == a mod m],
with j^{(chi)} = -2 Lambda(q) chi(q) (log q)^-1 q^-1/2, i.e. the exact character
orthogonality (1/phi(m)) sum_chi conj(chi(a)) chi(q) = 1[q==a mod m] for (q,m)=1.

Builds all Dirichlet characters mod m from a primitive root (cyclic m), evaluates
the reconstruction over every admissible residue a and several prime-power edges q,
and reports the worst deviation from the indicator. Exact up to floating round-off
in the roots of unity.

Uses numpy + sympy (primitive root, totient). No heavy dependencies.
"""
import json
import cmath
import numpy as np
import sympy as sp


def characters_mod(m):
    """All Dirichlet characters mod m for cyclic (Z/m)^* (m with a primitive root)."""
    g = int(sp.primitive_root(m))
    units = [a for a in range(1, m) if sp.gcd(a, m) == 1]
    phi = len(units)
    # discrete log base g
    dlog = {}
    x = 1
    for k in range(phi):
        dlog[x] = k
        x = (x*g) % m
    chars = []
    for j in range(phi):  # character indexed by j
        tbl = {}
        for a in units:
            tbl[a] = cmath.exp(2j*cmath.pi*j*dlog[a]/phi)
        chars.append(tbl)
    return chars, units, phi


def main():
    moduli = [5, 7, 9, 13, 11]
    qs = [3, 4, 8, 9, 25, 27]  # prime powers; Lambda via sympy
    worst = 0.0
    failures = []
    for m in moduli:
        chars, units, phi = characters_mod(m)
        for q in qs:
            if sp.gcd(q, m) != 1:
                continue
            Lam = float(sp.log(sp.factorint(q).popitem()[0]))  # Lambda(p^a)=log p
            j_chi = [(-2*Lam*ch[q % m]/(np.log(q)*np.sqrt(q))) for ch in chars]
            for a in units:
                recon = -(np.sqrt(q)*np.log(q)/2)*(1.0/phi)*sum(
                    np.conj(ch[a])*jc for ch, jc in zip(chars, j_chi))
                target = Lam if (q % m) == (a % m) else 0.0
                dev = abs(recon - target)
                worst = max(worst, dev)
                if dev > 1e-9:
                    failures.append(f"m={m},q={q},a={a}: dev={dev}")
    out = {
        "status": "PASS" if not failures else "FAIL",
        "description": "Residue-class readout across the Dirichlet family: exact "
                       "character-orthogonality reconstruction of Lambda(q)*1[q==a mod m] "
                       "from the per-character first jumps, over cyclic moduli and prime-power edges.",
        "moduli": moduli, "prime_powers": qs,
        "worst_deviation_from_indicator": worst,
        "failures": failures,
    }
    print(json.dumps(out, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
