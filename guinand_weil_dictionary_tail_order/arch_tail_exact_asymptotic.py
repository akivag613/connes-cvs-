#!/usr/bin/env python3
"""Exact spectral tail B_T versus the closed asymptotic of Corollary 3.3(iii).

Computes, at the configurations quoted in the manuscript,
  B_exact = (1/pi^2) int_T^inf h_+(r) sin^2(Lr/2)/rho (||p_r||^2 + ||q_r||^2) dr
(with sin^2 split as 1/2 - cos(Lr)/2: smooth half by direct quadrature, cosine
half by two integrations by parts with an explicit remainder bound) against
  B_asym  = (2N+1) rho (log(T/(2 pi)) + 1) / (pi^2 T).

Also verifies the derivative envelope h_+'(t) <= 1/t + 13/(10 t^2) of
Lemma 3.1 on a t-ladder, and solves B_asym(T) = 1e-59 at (c,N) = (100,200)
for the "T of order 10^58" statement.

Writes arch_tail_exact_vs_asymptotic.json.  Requires mpmath only.
"""
import json
import os

import mpmath as mp

mp.mp.dps = 25
HERE = os.path.dirname(os.path.abspath(__file__))


def hplus(r):
    return mp.re(mp.digamma(mp.mpf(1)/4 + mp.mpc(0, r)/2)) - mp.log(mp.pi)


def envelope_tail(env, L, T0):
    """(1/pi^2) int_{T0}^inf h_+(r) sin^2(Lr/2)/rho env(r) dr, IBP scheme."""
    rho = 2*mp.pi/L
    G = lambda r: hplus(r)*env(r)/(rho*mp.pi**2)
    smooth = mp.quad(G, [T0, 2*T0, 8*T0, 64*T0, 1024*T0, mp.inf])/2
    dG = lambda r: mp.diff(G, r)
    d2G = lambda r: mp.diff(G, r, 2)
    b1 = -G(mp.mpf(T0))*mp.sin(L*T0)/L
    b2 = -dG(mp.mpf(T0))*mp.cos(L*T0)/(L**2)
    per = 2*mp.pi/L
    K = 60
    pts = [mp.mpf(T0) + k*per for k in range(K + 1)]
    rem = -(1/L**2)*mp.quad(lambda r: d2G(r)*mp.cos(L*r), pts)
    R = pts[-1]
    rem_bound = (1/L**2)*mp.quad(lambda r: abs(d2G(r)), [R, 4*R, 64*R, mp.inf])
    return smooth - (b1 + b2 + rem)/2, rem_bound/2


def B_exact(c, N, T):
    L = mp.log(c)
    rho = 2*mp.pi/L

    def norms(r):
        a = r/rho
        return mp.fsum(1/(a - n)**2 + 1/(a + n)**2 for n in range(-N, N + 1))
    val, rem = envelope_tail(norms, L, mp.mpf(T))
    return val, rem


def B_asym(c, N, T):
    L = mp.log(c)
    rho = 2*mp.pi/L
    return (2*N + 1)*rho*(mp.log(T/(2*mp.pi)) + 1)/(mp.pi**2*T)


def main():
    rows = []
    for (c, N, T) in [(13, 4, 40), (13, 4, 160), (13, 40, 160),
                      (100, 100, 800), (100, 200, 800), (100, 200, 1600),
                      (100, 200, 6400)]:
        be, rem = B_exact(c, N, T)
        ba = B_asym(c, N, T)
        rows.append(dict(c=c, N=N, T=T,
                         B_exact=mp.nstr(be, 12), remainder_bound=mp.nstr(rem, 4),
                         B_asym=mp.nstr(ba, 12), ratio=mp.nstr(be/ba, 8)))
        print("c=%3d N=%3d T=%5d  B_exact=%-14s B_asym=%-14s ratio=%s" %
              (c, N, T, mp.nstr(be, 8), mp.nstr(ba, 8), mp.nstr(be/ba, 6)), flush=True)

    # derivative envelope margin ladder (Lemma 3.1)
    def hplus_d(t):
        u = mp.mpf(t)/2
        return (mp.mpf(t)/2)*mp.nsum(
            lambda k: (k + mp.mpf(1)/4)/(((k + mp.mpf(1)/4)**2 + u**2)**2), [0, mp.inf])
    env_rows = []
    env_ok = True
    for t in (1, 2, 3, 5, 7, 10, 20, 50, 100, 1000):
        bound = 1/mp.mpf(t) + mp.mpf(13)/(10*mp.mpf(t)**2)
        margin = bound - hplus_d(t)
        env_ok = env_ok and margin > 0
        env_rows.append(dict(t=t, h_plus_prime=mp.nstr(hplus_d(t), 10),
                             bound=mp.nstr(bound, 10), margin=mp.nstr(margin, 6)))
    print("derivative envelope h_+' <= 1/t + 13/(10 t^2):",
          "PASS" if env_ok else "FAIL", flush=True)

    # solve B_asym(T) = 1e-59 at (c,N)=(100,200), bisection in log10 T
    c, N = 100, 200
    target = mp.mpf(10)**-59
    lo, hi = mp.mpf(55), mp.mpf(70)
    for _ in range(200):
        mid = (lo + hi)/2
        if B_asym(c, N, mp.mpf(10)**mid) > target:
            lo = mid
        else:
            hi = mid
    T_star = mp.mpf(10)**((lo + hi)/2)
    print("B_asym = 1e-59 at (c,N)=(100,200) requires T =", mp.nstr(T_star, 6), flush=True)

    out = dict(dps=mp.mp.dps, exact_vs_asymptotic=rows,
               derivative_envelope=dict(passed=bool(env_ok), rows=env_rows),
               certification_floor_solve=dict(c=c, N=N, target="1e-59",
                                              T_required=mp.nstr(T_star, 10)))
    path = os.path.join(HERE, "arch_tail_exact_vs_asymptotic.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=2)
    os.replace(path + ".tmp", path)
    print("wrote arch_tail_exact_vs_asymptotic.json", flush=True)


if __name__ == "__main__":
    main()
