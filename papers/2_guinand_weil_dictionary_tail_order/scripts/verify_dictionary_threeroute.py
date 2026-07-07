#!/usr/bin/env python3
"""Three-route verification of the finite Guinand-Weil dictionary (Theorem 2.4).

Route 1: <v, Q_inf v> from the closed-form CCM assembly (2F1 / digamma / Lerch
         closed forms for the archimedean entries; prime and pole sources exact).
Route 2: the source side of the dictionary: prime sum + pole term 2 g_v(i/2) +
         archimedean integral, with g_v evaluated by an exact trigonometric-
         polynomial closed form and the archimedean tail integrated through the
         rank-two representation of Theorem 3.2 (no oscillatory-quadrature
         heuristics; the tail remainder carries an explicit bound).
Route 3: the zero side 2 * sum_{n<=M} g_v(gamma_n) over the first M nontrivial
         zeros of zeta, with a smooth-density tail correction, M up to 512.

Configs: (c=29, N=6, generic vector), (c=29, N=6, pole-neutral vector),
         (c=13, N=4, the regression vector of the worked example).
Routes 1 and 2 must agree to the tail remainder bound; route 3 must converge to
the same value; the pole-neutral config must show a vanishing pole term.

This script also generates, as route 1 of the c=13 configuration, the reference
constant used by verify_zero_side.py (previously recorded there as a literal).

Usage:  python3 verify_dictionary_threeroute.py [M_MAX]     (default M_MAX=512)
Output: threeroute_<tag>.json + zeta_zeros_<M>_dps30.json in this directory.
Requires mpmath only.
"""
import json
import os
import sys
import time

import mpmath as mp

mp.mp.dps = 40
HERE = os.path.dirname(os.path.abspath(__file__))
M_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 512
CHECKPOINTS = [m for m in (32, 64, 128, 256, 512) if m <= M_MAX] or [M_MAX]


def prime_powers(c):
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
            out.append((q, mp.log(p)))
            q *= p
    return out


def hplus(r):
    return mp.re(mp.digamma(mp.mpf(1)/4 + mp.mpc(0, r)/2)) - mp.log(mp.pi)


# ---------------- Route 1: closed-form CCM even block ----------------
def build_Me(c, N):
    """Even-sector cutoff-free matrix from the closed-form entries
    (hypergeometric/digamma/Lerch archimedean entries; exact prime and pole
    sources), assembled per CCM eqs. (3.10)-(3.11), (3.16) and Lemma 4.1."""
    L = mp.log(c)
    z = mp.e**(-2*L)
    PI = mp.pi
    gamma = mp.euler
    PP = prime_powers(c)

    def a_n(n):
        return mp.mpf(1)/4 + PI*1j*n/L

    def F(n):
        return mp.hyp2f1(1, a_n(n), a_n(n) + 1, z)

    def alpha_L(n):
        an = a_n(n)
        return (mp.e**(-L/2)*mp.im((2*L/(L + 4*PI*1j*n))*F(n))
                + mp.mpf(1)/2*mp.im(mp.digamma(an)))/PI

    def beta_L(n):
        an = a_n(n)
        t1 = -L*mp.e**(-L/2)*mp.im((2*L/(4*PI*n - 1j*L))*F(n))
        t2 = -(mp.e**(-L/2)/4)*mp.re(mp.lerchphi(z, 2, an))
        t3 = mp.mpf(1)/4*mp.re(mp.polygamma(1, an))
        return (t1 + t2 + t3)/L

    def c_w():
        return (mp.mpf(1)/2*mp.log((mp.e**(L/2) - 1)/(mp.e**(L/2) + 1))
                + mp.atan(mp.e**(L/2)) - PI/4 + gamma/2
                + mp.mpf(1)/2*mp.log(8*PI))

    def gamma_L(n):
        an = a_n(n)
        return (-mp.e**(-L/2)*mp.re((2*L/(L + 4*PI*1j*n))*F(n))
                + 2*mp.e**(-L/2)*mp.hyp2f1(mp.mpf(1)/4, 1, mp.mpf(5)/4, z)
                - mp.mpf(1)/2*(mp.re(mp.digamma(an)) - mp.digamma(mp.mpf(1)/4))
                + c_w())

    def psipr(m):
        return -(1/PI)*mp.fsum(lp/mp.sqrt(q)*mp.sin(2*PI*m*(1 - mp.log(q)/L))
                               for q, lp in PP)

    def psiprd(m):
        return -2*mp.fsum(lp/mp.sqrt(q)*(1 - mp.log(q)/L)
                          * mp.cos(2*PI*m*(1 - mp.log(q)/L)) for q, lp in PP)

    idx = range(-N, N + 1)
    P0 = {m: alpha_L(m) + psipr(m) for m in idx}
    P0d = {m: -2*(gamma_L(m) - beta_L(m)) + psiprd(m) for m in idx}

    def Cm(m):
        return mp.sinh(L/4)/mp.sqrt(L)*1/(mp.mpf(1)/4 + (2*PI*m/L)**2)

    def Sm(m):
        return (4*PI*mp.sinh(L/4)/(L*mp.sqrt(L))*m/(mp.mpf(1)/4 + (2*PI*m/L)**2))

    Cc_ = {m: Cm(m) for m in idx}
    Sc_ = {m: Sm(m) for m in idx}

    def Q(m, n):
        pole = 2*(Cc_[m]*Cc_[n] - Sc_[m]*Sc_[n])
        return (P0d[n] + pole) if m == n else (P0[m] - P0[n])/(m - n) + pole

    ne = N + 1
    Me = mp.matrix(ne, ne)
    for i in range(ne):
        for j in range(ne):
            if i == 0 and j == 0:
                Me[i, j] = Q(0, 0)
            elif i == 0:
                Me[i, j] = (Q(0, j) + Q(0, -j))/mp.sqrt(2)
            elif j == 0:
                Me[i, j] = (Q(i, 0) + Q(-i, 0))/mp.sqrt(2)
            else:
                Me[i, j] = Q(i, j) + Q(i, -j)
    return Me


def contract(Me, v):
    N = len(v) - 1
    return mp.fsum(v[i]*Me[i, j]*v[j] for i in range(N + 1) for j in range(N + 1))


# ---------------- test-function machinery (exact closed forms) ----------------
class TestFn:
    """The dictionary chain v -> T_v -> K_v -> ghat_v -> g_v in closed form.

    K_v(w) = sum_k (alpha_k + beta_k w) e^{2 pi i k w} with frequencies |k| <= N;
    g_v(z) is then an exact finite combination of elementary integrals."""

    def __init__(self, c, N, v):
        self.c, self.N = c, N
        self.L = mp.log(c)
        self.Delta = self.L/(2*mp.pi)
        self.v = [mp.mpf(x) for x in v]
        u = {0: self.v[0]}
        for k in range(1, N + 1):
            u[k] = self.v[k]/mp.sqrt(2)
            u[-k] = self.v[k]/mp.sqrt(2)
        self.u = u
        self.alpha = {}
        self.beta = {}
        for k in range(-N, N + 1):
            s = mp.fsum(u[n]/(k - n) for n in u if n != k)
            self.alpha[k] = 2*u[k]*s/(mp.pi*1j)
            self.beta[k] = 2*u[k]**2

    def K(self, w):
        return mp.re(mp.fsum((self.alpha[k] + self.beta[k]*w)*mp.exp(2j*mp.pi*k*w)
                             for k in self.u))

    def K_quad(self, w):
        def Tv(t):
            return mp.fsum(self.u[m]*mp.exp(2j*mp.pi*m*t) for m in self.u)
        return mp.re(2*mp.quad(lambda t: Tv(t)*Tv(w - t), [0, w]))

    def ghat(self, xi):
        ax = abs(xi)
        if ax > self.Delta:
            return mp.mpf(0)
        return mp.pi*self.K(1 - ax/self.Delta)

    @staticmethod
    def _int_poly_exp(al, be, a):
        """int_0^1 (al + be*w) e^{i a w} dw, complex a allowed."""
        if abs(a) < mp.mpf(10)**-8:
            tot_a = mp.mpc(0)
            tot_b = mp.mpc(0)
            for j in range(0, 25):
                cj = (1j*a)**j
                tot_a += cj/mp.factorial(j + 1)
                tot_b += cj/mp.factorial(j)*(mp.mpf(1)/(j + 2))
            return al*tot_a + be*tot_b
        ia = 1j*a
        e = mp.exp(ia)
        return al*(e - 1)/ia + be*((e*(ia - 1) + 1)/(ia**2))

    def g(self, zz):
        """g_v(z) = 2 pi Delta int_0^1 K_v(w) cos(zL(1-w)) dw, exact per term."""
        th = zz*self.L
        tot = mp.mpc(0)
        for k in self.u:
            al, be = self.alpha[k], self.beta[k]
            tot += mp.exp(1j*th)/2*self._int_poly_exp(al, be, 2*mp.pi*k - th)
            tot += mp.exp(-1j*th)/2*self._int_poly_exp(al, be, 2*mp.pi*k + th)
        val = 2*mp.pi*self.Delta*tot
        return mp.re(val) if abs(mp.im(zz)) < mp.mpf(10)**-30 else val

    def g_quad(self, r):
        return mp.quad(lambda y: self.K(1 - y/self.L)*mp.cos(r*y), [0, self.L])


# ---------------- archimedean tail via the rank-two representation ----------------
def ranktwo_tail(u, L, T0):
    """(1/pi^2) int_{T0}^inf h_+(r) sin^2(Lr/2)/rho [(p_r.u)^2 + (q_r.u)^2] dr.

    sin^2 = 1/2 - cos(Lr)/2; the smooth half converges by direct quadrature and
    the cosine half by two integrations by parts, with an explicit remainder
    bound.  Returns (value, remainder_bound)."""
    rho = 2*mp.pi/L

    def F(r):
        a = r/rho
        pu = mp.fsum(u[n]/(a - n) for n in u)
        qu = mp.fsum(u[n]/(a + n) for n in u)
        return pu**2 + qu**2

    G = lambda r: hplus(r)*F(r)/(rho*mp.pi**2)
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
    cospart = -(b1 + b2 + rem)/2
    return smooth + cospart, rem_bound/2


# ---------------- Route 2 and Route 3 ----------------
def source_side(tf):
    c, L = tf.c, tf.L
    PP = prime_powers(c)
    prime = -(1/mp.pi)*mp.fsum(lp/mp.sqrt(q)*tf.ghat(mp.log(q)/(2*mp.pi)) for q, lp in PP)
    gi2 = tf.g(mp.mpc(0, 0.5))
    pole = 2*mp.re(gi2)
    T0 = mp.mpf(60)
    arch_head = (1/mp.pi)*mp.quad(lambda r: hplus(r)*tf.g(r), [0, 5, 10, 20, 40, T0])
    arch_tail, tail_rem = ranktwo_tail(tf.u, L, T0)
    return dict(prime=prime, pole=pole, pole_imag_check=mp.im(gi2),
                arch=arch_head + arch_tail, tail_rem_bound=tail_rem,
                total=prime + pole + arch_head + arch_tail)


def zero_side(tf, zeros, checkpoints, source_total):
    rows = []
    running = mp.mpf(0)
    t0 = time.time()
    for i, gam in enumerate(zeros, 1):
        running += tf.g(gam)
        if i % 64 == 0 or i in checkpoints:
            print("  [zeros] %d/%d  elapsed=%.1fs" % (i, len(zeros), time.time() - t0), flush=True)
        if i in checkpoints:
            zs = 2*running
            tail, _ = ranktwo_tail(tf.u, tf.L, gam)
            rows.append(dict(M=i, gamma_M=mp.nstr(gam, 20),
                             zero_side=mp.nstr(zs, 30),
                             raw_diff=mp.nstr(zs - source_total, 8),
                             tail_corrected_diff=mp.nstr(zs + tail - source_total, 8)))
    return rows


def run_config(tag, c, N, v, zeros, checkpoints):
    print("== config %s: c=%d N=%d ==" % (tag, c, N), flush=True)
    tf = TestFn(c, N, v)
    gk = max(abs(tf.K(mp.mpf(w)) - tf.K_quad(mp.mpf(w))) for w in ('0.3', '0.77'))
    gg = max(abs(tf.g(mp.mpf(r)) - tf.g_quad(mp.mpf(r))) for r in ('0.9', '14.2'))
    print("  guard K closed-vs-quad: %s ; g closed-vs-quad: %s" % (mp.nstr(gk, 4), mp.nstr(gg, 4)), flush=True)
    Me = build_Me(c, N)
    r1 = contract(Me, tf.v)
    print("  route1 closed-form contraction = %s" % mp.nstr(r1, 30), flush=True)
    src = source_side(tf)
    print("  route2 source total            = %s" % mp.nstr(src['total'], 30), flush=True)
    print("  |route1 - route2| = %s  (tail remainder bound %s)" %
          (mp.nstr(abs(r1 - src['total']), 6), mp.nstr(src['tail_rem_bound'], 4)), flush=True)
    rows = zero_side(tf, zeros, checkpoints, src['total'])
    for row in rows:
        print("  route3 M=%4s raw_diff=%s tail_corrected=%s" %
              (row['M'], row['raw_diff'], row['tail_corrected_diff']), flush=True)
    out = dict(tag=tag, c=c, N=N, dps=mp.mp.dps, v=[mp.nstr(x, 20) for x in tf.v],
               guard_K=mp.nstr(gk, 6), guard_g=mp.nstr(gg, 6),
               route1_closed_form=mp.nstr(r1, 35),
               route2_source_total=mp.nstr(src['total'], 35),
               route2_prime=mp.nstr(src['prime'], 30),
               route2_pole=mp.nstr(src['pole'], 30),
               route2_arch=mp.nstr(src['arch'], 30),
               route2_tail_rem_bound=mp.nstr(src['tail_rem_bound'], 6),
               pole_imag_check=mp.nstr(src['pole_imag_check'], 6),
               r1_minus_r2=mp.nstr(r1 - src['total'], 8),
               zero_side_rows=rows)
    path = os.path.join(HERE, "threeroute_%s.json" % tag)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)
    print("  wrote %s" % os.path.basename(path), flush=True)
    return out


def main():
    zpath = os.path.join(HERE, "zeta_zeros_%d_dps30.json" % M_MAX)
    if os.path.exists(zpath):
        with open(zpath) as f:
            zeros = [mp.mpf(s) for s in json.load(f)]
        print("loaded %d cached zeros" % len(zeros), flush=True)
    else:
        print("computing %d zeta zeros (dps 30)..." % M_MAX, flush=True)
        old = mp.mp.dps
        mp.mp.dps = 30
        zeros = []
        t0 = time.time()
        for i in range(1, M_MAX + 1):
            zeros.append(mp.im(mp.zetazero(i)))
            if i % 64 == 0:
                el = time.time() - t0
                print("  [zetazero] %d/%d  elapsed=%.1fs" % (i, M_MAX, el), flush=True)
        mp.mp.dps = old
        tmp = zpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump([mp.nstr(zz, 25) for zz in zeros], f)
        os.replace(tmp, zpath)

    # config 1: generic (not pole- or moment-neutral) vector at c=29, N=6
    run_config("c29N6_generic", 29, 6,
               ['1', '0.7', '-0.4', '0.2', '-0.1', '0.3', '-0.2'], zeros, CHECKPOINTS)

    # config 2: pole- and moment-neutral vector at c=29, N=6
    c2 = 29
    b2 = (mp.log(c2)/(4*mp.pi))**2
    x2, x3, x4, x5, x6 = mp.mpf(1), mp.mpf('0.5'), mp.mpf(-3), mp.mpf('2'), mp.mpf('-1')
    A = mp.matrix([[1, mp.sqrt(2)], [1/b2, mp.sqrt(2)/(1 + b2)]])
    rhs = mp.matrix([-(mp.sqrt(2)*(x2 + x3 + x4 + x5 + x6)),
                     -(mp.sqrt(2)*(x2/(4 + b2) + x3/(9 + b2) + x4/(16 + b2)
                                   + x5/(25 + b2) + x6/(36 + b2)))])
    sol = mp.lu_solve(A, rhs)
    vpn = [sol[0], sol[1], x2, x3, x4, x5, x6]
    run_config("c29N6_poleneutral", 29, 6, vpn, zeros, CHECKPOINTS)

    # config 3: the worked-example vector at c=13, N=4 (pole- and moment-neutral)
    c3 = 13
    b3 = (mp.log(c3)/(4*mp.pi))**2
    y2, y3, y4 = mp.mpf(1), mp.mpf(0), mp.mpf(-3)
    r1_ = -(y2 + y3 + y4)
    r2_ = -(y2/(4 + b3) + y3/(9 + b3) + y4/(16 + b3))
    sol3 = mp.lu_solve(mp.matrix([[1, 1], [1/b3, 1/(1 + b3)]]), mp.matrix([r1_, r2_]))
    x0, x1 = sol3[0], sol3[1]
    v3 = [x0, x1/mp.sqrt(2), y2/mp.sqrt(2), y3/mp.sqrt(2), y4/mp.sqrt(2)]
    run_config("c13N4_package", 13, 4, v3, zeros, CHECKPOINTS)

    print("three-route dictionary check complete", flush=True)


if __name__ == "__main__":
    main()
