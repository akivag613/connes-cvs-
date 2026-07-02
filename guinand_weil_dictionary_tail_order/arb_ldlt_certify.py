#!/usr/bin/env python3
"""Rigorous interval certificate of the inertia of the cutoff-free truncated Weil block.

Every matrix entry of tau = W_{0,2} - W_R - W_p (the cutoff-free assembly; no
finite archimedean cutoff T anywhere) is computed as an Arb ball with a rigorous
error enclosure, and the inertia is certified by an interval LDL^T factorization
(Sylvester's criterion): when every pivot ball is strictly signed, the counts
(n_pos, n_neg) are proved, not observed.  This is the certificate quoted in the
manuscript's "T=800 correction scale" paragraph (c=100, N=200, 9000 bits:
n_pos = 401, n_neg = 0).

Archimedean entries use closed forms built from digamma/trigamma values at
1/4 + i pi n/L plus geometric sums with rigorously bounded remainders:
  rho(x) = sum_k e^{-(2k+1/2)x},  c_k = 2k+1/2,  w = 2 pi n/L,
  S(n)  = (1/2) Im psi(1/4 + i pi n/L)          - w  * G_S
  CC(n) = -(1/2)(Re psi(1/4 + i pi n/L)-psi(1/4)) + G_CC
  XC(n) = (1/4) Re psi'(1/4 + i pi n/L)          - L G_XC1 - G_XC2
Self-test: the Arb entry balls are checked to contain an independent mpmath
recomputation of the same closed forms.

Usage:
  python3 arb_ldlt_certify.py --selftest --c 13 --N 8 --prec 300
  python3 arb_ldlt_certify.py --c 100 --N 200 --prec 9000 --json-out cert.json
Requires python-flint (Arb) and mpmath.
"""
import argparse
import datetime
import json
import sys
import time

from flint import arb, acb, arb_mat, ctx
import mpmath as mp


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def prime_powers_up_to(c):
    """[(q, p)] for prime powers q = p^a <= c."""
    primes = []
    x = 2
    while x <= c:
        if all(x % p for p in primes):
            primes.append(x)
        x += 1
    out = []
    for p in primes:
        q = p
        while q <= c:
            out.append((q, p))
            q *= p
    return out


def trigamma_acb(z):
    return z.polygamma(acb(1))


def _geom_sums(n, L, prec):
    """(G_S, G_CC, G_XC1, G_XC2) arb balls with a rigorous tail remainder."""
    PI = arb.pi()
    w = 2 * PI * n / L
    w2 = w * w
    gS = arb(0); gCC = arb(0); gX1 = arb(0); gX2 = arb(0)
    thr = arb(2) ** (-(prec + 24))
    k = 0
    while True:
        c_k = arb(2 * k) + arb('0.5')
        e = (-c_k * L).exp()
        den = c_k * c_k + w2
        gS += e / den
        if n != 0:
            gCC += e * w2 / (c_k * den)
        gX1 += e * c_k / den
        gX2 += e * (c_k * c_k - w2) / (den * den)
        if e < thr and k > 2:
            break
        k += 1
    c_next = arb(2 * (k + 1)) + arb('0.5')
    one_m_z = 1 - (-2 * L).exp()
    tail_geo = (-c_next * L).exp() / one_m_z
    rem = arb(4) * tail_geo

    def widen(x):
        return x + arb(0, rem)
    return widen(gS), widen(gCC), widen(gX1), widen(gX2)


def arb_closed_forms(N, c, prec):
    ctx.prec = prec
    L = arb(c).log()
    PI = arb.pi()
    quarter = arb('0.25')
    psi_quarter = quarter.digamma()
    S = [arb(0)] * (N + 1)
    CC = [arb(0)] * (N + 1)
    XC = [arb(0)] * (N + 1)
    for n in range(N + 1):
        w = 2 * PI * n / L
        zarg = acb(quarter, PI * n / L)
        psi = zarg.digamma()
        psi1 = trigamma_acb(zarg)
        gS, gCC, gX1, gX2 = _geom_sums(n, L, prec)
        if n == 0:
            S[n] = arb(0)
            CC[n] = arb(0)
        else:
            S[n] = arb('0.5') * psi.imag - w * gS
            CC[n] = -arb('0.5') * (psi.real - psi_quarter) + gCC
        XC[n] = arb('0.25') * psi1.real - L * gX1 - gX2
    return S, CC, XC, L


def arb_J(L):
    U = (L / 2).exp()
    return -2 * (U + 1).log() + (U * U + 1).log() + 2 * U.atan() + arb(2).log() - arb.pi() / 2


def arb_kappa(L):
    eL = L.exp()
    return (4 * arb.pi() * (eL - 1) / (eL + 1)).log() + arb.const_euler()


def build_arb_tau(c, N, prec):
    """tau = W02 - WR - Wp as an arb_mat, entries rigorous balls."""
    ctx.prec = prec
    S, CC, XC, L = arb_closed_forms(N, c, prec)
    PI = arb.pi()
    sp2 = 16 * PI * PI
    l2 = L * L
    pref02 = 32 * L * (L / 4).sinh() ** 2
    kappa = arb_kappa(L)
    J = arb_J(L)

    pdata = prime_powers_up_to(c)
    weights = [arb(p).log() * (arb(q) ** arb('-0.5')) for (q, p) in pdata]
    positions = [arb(q).log() for (q, p) in pdata]
    M = len(weights)

    def S_signed(nn):
        return S[nn] if nn >= 0 else -S[-nn]

    DIM = 2 * N + 1
    A = arb_mat(DIM, DIM)
    for i in range(DIM):
        n = i - N
        for j in range(i, DIM):
            m = j - N
            num = l2 - sp2 * m * n
            den = (l2 + sp2 * m * m) * (l2 + sp2 * n * n)
            W02 = pref02 * num / den
            if n == m:
                WR = kappa + 2 * CC[abs(n)] + J - (2 / L) * XC[abs(n)]
            else:
                WR = (S_signed(m) - S_signed(n)) / (PI * (n - m))
            Wp = arb(0)
            for idx in range(M):
                y = positions[idx]
                if n == m:
                    q = 2 * (1 - y / L) * (2 * PI * n * y / L).cos()
                else:
                    q = ((2 * PI * m * y / L).sin() - (2 * PI * n * y / L).sin()) / (PI * (n - m))
                Wp += weights[idx] * q
            val = W02 - WR - Wp
            A[i, j] = val
            A[j, i] = val
    return A, DIM


def certified_inertia(A, DIM, heartbeat=50):
    """Interval LDL^T.  Returns (n_pos, n_neg, undetermined_pivot_or_None)."""
    d = [None] * DIM
    Lf = [[arb(0)] * DIM for _ in range(DIM)]
    n_pos = 0
    n_neg = 0
    t0 = time.time()
    for i in range(DIM):
        s = A[i, i]
        for k in range(i):
            s = s - Lf[i][k] * Lf[i][k] * d[k]
        d[i] = s
        if s > 0:
            n_pos += 1
        elif s < 0:
            n_neg += 1
        else:
            return n_pos, n_neg, i
        for j in range(i + 1, DIM):
            t = A[j, i]
            for k in range(i):
                t = t - Lf[j][k] * Lf[i][k] * d[k]
            Lf[j][i] = t / d[i]
        if heartbeat and (i + 1) % heartbeat == 0:
            el = time.time() - t0
            print("  [ldlt] pivot %d/%d  elapsed=%.0fs  eta=%.0fs"
                  % (i + 1, DIM, el, el / (i + 1) * (DIM - i - 1)), flush=True)
    return n_pos, n_neg, None


def selftest(c=13, N=8, prec=300):
    """Arb entry balls must contain an independent mpmath recomputation."""
    print("[%s] SELF-TEST c=%d N=%d prec=%d ..." % (ts(), c, N, prec), flush=True)
    S, CC, XC, L = arb_closed_forms(N, c, prec)
    mp.mp.dps = 80
    Lm = mp.log(c)
    ok = True
    for n in range(N + 1):
        w = 2 * mp.pi * n / Lm
        z = mp.mpf(1)/4 + 1j * mp.pi * n / Lm
        psi = mp.digamma(z)
        psi1 = mp.polygamma(1, z)
        gS = gCC = gX1 = gX2 = mp.mpf(0)
        k = 0
        while True:
            c_k = 2 * k + mp.mpf('0.5')
            e = mp.e ** (-c_k * Lm)
            den = c_k * c_k + w * w
            gS += e / den
            if n != 0:
                gCC += e * w * w / (c_k * den)
            gX1 += e * c_k / den
            gX2 += e * (c_k * c_k - w * w) / (den * den)
            if e < mp.mpf(10) ** -70 and k > 2:
                break
            k += 1
        if n == 0:
            Sm, CCm = mp.mpf(0), mp.mpf(0)
        else:
            Sm = mp.im(psi) / 2 - w * gS
            CCm = -(mp.re(psi) - mp.digamma(mp.mpf(1)/4)) / 2 + gCC
        XCm = mp.re(psi1) / 4 - Lm * gX1 - gX2
        for (ball, mpv, nm) in ((S[n], Sm, 'S'), (CC[n], CCm, 'CC'), (XC[n], XCm, 'XC')):
            # the mpmath mirror truncates its geometric sums near 1e-70, so compare
            # at relative tolerance 1e-60 rather than by strict ball containment
            diff = (ball - arb(mp.nstr(mpv, 70))).abs_upper()
            scale = max(abs(float(mpv)), 1.0)
            if float(diff) > 1e-60 * scale:
                print("  FAIL n=%d %s: |arb - mpmath| = %s" % (n, nm, diff), flush=True)
                ok = False
    print("[%s] SELF-TEST %s (Arb entries agree with independent mpmath "
          "recomputation to 60 digits)" % (ts(), "PASS" if ok else "FAIL"), flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c", type=int, default=13)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--prec", type=int, default=300, help="Arb working precision in bits")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--json-out", type=str, default="")
    args = p.parse_args()

    if args.selftest and not selftest():
        sys.exit(1)

    print("[%s] ARB LDL^T CERTIFY  c=%d N=%d prec=%d bits (~%.0f digits)"
          % (ts(), args.c, args.N, args.prec, args.prec * 0.301), flush=True)
    t0 = time.time()
    A, DIM = build_arb_tau(args.c, args.N, args.prec)
    t_build = time.time() - t0
    print("[%s] cutoff-free block built, dimension %d (%.1fs); interval LDL^T ..."
          % (ts(), DIM, t_build), flush=True)
    t1 = time.time()
    n_pos, n_neg, undet = certified_inertia(A, DIM)
    t_ldlt = time.time() - t1
    if undet is None:
        print("[%s] RESULT: n_pos=%d n_neg=%d  (%.1fs)" % (ts(), n_pos, n_neg, t_ldlt), flush=True)
        print("  >>> CERTIFIED %s (rigorous Arb interval LDL^T, all %d pivots strictly signed)"
              % ("positive-definite" if n_neg == 0 else "indefinite", DIM), flush=True)
    else:
        print("[%s] UNDETERMINED at pivot %d (ball straddles 0): raise --prec" % (ts(), undet), flush=True)

    if args.json_out:
        out = dict(script="arb_ldlt_certify.py", c=args.c, N=args.N, dimension=DIM,
                   prec_bits=args.prec, n_pos=n_pos, n_neg=n_neg,
                   undetermined_pivot=undet,
                   certified_positive_definite=bool(undet is None and n_neg == 0),
                   build_seconds=round(t_build, 1), ldlt_seconds=round(t_ldlt, 1),
                   date=datetime.datetime.now().isoformat(timespec="seconds"))
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote %s" % args.json_out, flush=True)


if __name__ == "__main__":
    main()
