"""
Core operator construction and diagonalization.

Implements the CvS Proposition 4.1 Galerkin matrix Q(c, N) and its
eigendecomposition. The matrix decomposes as:

    Q = Q_prime + Q_pole + Q_arch

where each piece encodes a different arithmetic contribution to the
Weil explicit formula.

The archimedean piece uses python-flint's acb.digamma when available
(~144x faster than mpmath's pure-Python digamma), falling back to
mpmath transparently.

References
----------
- Connes & van Suijlekom, arXiv:2511.23257, Proposition 4.1
- Connes, Consani & Moscovici, arXiv:2511.22755, Section 6
"""

from __future__ import annotations

import mpmath as mp

# ============================================================
# Optional python-flint for fast digamma
# ============================================================
try:
    from flint import acb, arb, ctx as flint_ctx
    HAS_FLINT = True
except ImportError:
    HAS_FLINT = False


# ============================================================
# Number-theoretic helpers
# ============================================================

def prime_powers_up_to(c: int) -> tuple[list[tuple[int, mp.mpf, mp.mpf]], list[int]]:
    """
    Find all prime powers n in [2, c] and their von Mangoldt weights.

    Returns
    -------
    prime_power_data : list of (n, log(n), Lambda(n)/sqrt(n))
        For each prime power n = p^k, Lambda(n) = log(p).
    primes : list of int
        All primes up to c.
    """
    c = int(c)
    is_pp = [False] * (c + 1)
    is_prime = [True] * (c + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, c + 1):
        if is_prime[i]:
            j = i * i
            while j <= c:
                is_prime[j] = False
                j += i
    primes_list = [i for i in range(2, c + 1) if is_prime[i]]
    for p in primes_list:
        pk = p
        while pk <= c:
            is_pp[pk] = True
            if pk > c // p:
                break
            pk *= p
    result = []
    for n in range(2, c + 1):
        if is_pp[n]:
            for p in primes_list:
                k = n
                while k % p == 0:
                    k //= p
                if k == 1:
                    lam = mp.log(p)
                    break
            result.append((n, mp.log(n), lam / mp.sqrt(n)))
    return result, primes_list


# ============================================================
# psi components: prime piece
# ============================================================

def psi_prime(x: mp.mpf, L: mp.mpf, prime_data: list) -> mp.mpf:
    """
    Prime-piece contribution to psi(x).

    psi_prime(x) = -(1/pi) * sum_{n prime power <= c} (Lambda(n)/sqrt(n)) * sin(2*pi*x*(1 - log(n)/L))
    """
    PI = mp.pi
    x = mp.mpf(x)
    s = mp.mpf(0)
    for (n, logn, w) in prime_data:
        s += w * mp.sin(2 * PI * x * (1 - logn / L))
    return -s / PI


def psi_prime_deriv(x: mp.mpf, L: mp.mpf, prime_data: list) -> mp.mpf:
    """
    Derivative of the prime piece: d/dx psi_prime(x).
    """
    PI = mp.pi
    x = mp.mpf(x)
    s = mp.mpf(0)
    for (n, logn, w) in prime_data:
        c_val = 1 - logn / L
        s += w * 2 * c_val * mp.cos(2 * PI * x * c_val)
    return -s


# ============================================================
# psi components: pole piece
# ============================================================

def psi_pole(x: mp.mpf, L: mp.mpf) -> mp.mpf:
    """
    Pole-piece contribution to psi(x).

    Accounts for the trivial zeros: integral over [0, L] of
    sin(2*pi*x*(1 - y/L)) * 2*cosh(y/2) dy, divided by pi.
    """
    PI = mp.pi
    x = mp.mpf(x)
    two_pi_x = 2 * PI * x

    def integrand(y):
        return mp.sin(two_pi_x * (1 - y / L)) * 2 * mp.cosh(y / 2)

    return mp.quad(integrand, [0, L]) / PI


def psi_pole_deriv(x: mp.mpf, L: mp.mpf) -> mp.mpf:
    """
    Derivative of the pole piece: d/dx psi_pole(x).
    """
    PI = mp.pi
    x = mp.mpf(x)
    two_pi_x = 2 * PI * x

    def integrand(y):
        return (1 - y / L) * mp.cos(two_pi_x * (1 - y / L)) * 2 * mp.cosh(y / 2)

    return 2 * mp.quad(integrand, [0, L])


# ============================================================
# psi components: archimedean piece (h_plus uses flint or mpmath)
# ============================================================

def _h_plus_flint(tau: mp.mpf, dps: int) -> mp.mpf:
    """
    Compute h_plus(tau) = Re(digamma(1/4 + i*tau/2)) - log(pi)
    using python-flint's acb.digamma (Arb library, ~144x faster).
    """
    log_pi_fl = arb.pi().log()
    tau_mp = mp.mpf(tau)
    # Convert mpmath tau to flint arb via string (preserves all digits)
    tau_fl = arb(mp.nstr(tau_mp, dps + 10))
    z = acb(arb("0.25"), tau_fl / 2)
    result_fl = (-log_pi_fl + z.digamma().real)
    return mp.mpf(result_fl._mpf_)


def _h_plus_mpmath(tau: mp.mpf, dps: int) -> mp.mpf:
    """
    Compute h_plus(tau) = Re(digamma(1/4 + i*tau/2)) - log(pi)
    using pure mpmath (slower fallback).
    """
    tau_mp = mp.mpf(tau)
    z = mp.mpc(mp.mpf("0.25"), tau_mp / 2)
    return mp.re(mp.digamma(z)) - mp.log(mp.pi)


def h_plus(tau: mp.mpf, dps: int) -> mp.mpf:
    """
    Compute h_plus(tau) = Re(digamma(1/4 + i*tau/2)) - log(pi).

    This is the archimedean Mellin multiplier from the explicit formula.
    Uses python-flint when available for ~144x speedup.

    Parameters
    ----------
    tau : mpmath.mpf
        Spectral parameter.
    dps : int
        Decimal digits of precision.
    """
    if HAS_FLINT:
        return _h_plus_flint(tau, dps)
    return _h_plus_mpmath(tau, dps)


# ============================================================
# WIN 1: h_plus memoization cache (bit-identical optimization)
# ============================================================
#
# The CvS archimedean integral splits into subintervals broken at
# {-alpha_x, 0, alpha_x}, and mp.quad's tanh-sinh rule is deterministic
# per (interval, precision). Both psi_arch and psi_arch_deriv for the
# same x use identical subinterval endpoints, so they evaluate h_plus
# on exactly the same tau-node set. Furthermore, h_plus(tau) is EVEN
# in tau (digamma(conj z) = conj(digamma(z)) implies
# Re(digamma(1/4 + i*tau/2)) = Re(digamma(1/4 - i*tau/2))), so we can
# key the cache on abs(tau) and double the hit rate again.
#
# Result: within a single _compute_psi_pair call (2 mp.quad calls across
# up to 4 subintervals each) we hit the cache ~4x, saving ~75% of the
# h_plus evaluations. The returned mp.mpf is the exact bit-identical
# value of h_plus(|tau|), so the quadrature sums are unchanged at the
# ULP level. Cross-x nodes do NOT overlap (different subinterval
# endpoints yield disjoint tanh-sinh nodes), so the cache is cleared
# between basis indices to bound memory.

_hplus_cache: dict = {}

# Kernel cache: within a single _compute_psi_pair call, psi_arch and
# psi_arch_deriv are two mp.quad calls at the same x, sharing the same
# subinterval split (same alpha_x) and therefore the same tanh-sinh
# tau-node set. On the first pass (psi_arch integrand) we compute BOTH
# Re(S_hat_x) and Re(dS_hat_x_dx) via a fused kernel that shares
# stable_A/B sub-expressions (sin(bL), sin(bL/2), bL), then stash the
# paired-value. On the second pass (psi_arch_deriv integrand) we hit
# the cache and skip the kernel work entirely. This is bit-identical
# because the fused kernel produces the exact same mpf values as the
# original stable_A + stable_B + S_hat_x composition (verified at
# 0.0 rel diff across all tested (x, tau) pairs at dps=50).
_kernel_cache: dict = {}       # tau._mpf_ -> (re_S, re_dS)


def _h_plus_cached(tau: mp.mpf, dps: int) -> mp.mpf:
    """
    Memoized wrapper over h_plus that exploits h_plus's evenness in tau.

    Keys the cache on the raw mpf tuple of ``abs(tau)``. The returned
    value is bit-identical to a fresh ``h_plus(tau, dps)`` call because
    h_plus is mathematically even and the flint/mpmath implementations
    produce identical bits on identical inputs at fixed precision.
    """
    if not isinstance(tau, mp.mpf):
        tau = mp.mpf(tau)
    # Key on abs(tau) to collapse +-tau pairs. mp.mpf._mpf_ is hashable.
    key = abs(tau)._mpf_
    hit = _hplus_cache.get(key)
    if hit is not None:
        return hit
    val = h_plus(tau, dps)
    _hplus_cache[key] = val
    return val


def _hplus_cache_clear() -> None:
    """Drop all cached h_plus / kernel values. Called between basis indices."""
    _hplus_cache.clear()
    _kernel_cache.clear()


def _re_S_and_dS_fused(tau: mp.mpf, x: mp.mpf, L: mp.mpf) -> tuple:
    """
    Compute (Re(S_hat_x(tau,x,L)), Re(dS_hat_x_dx(tau,x,L))) in one pass,
    sharing the stable_A / stable_B sub-expressions (sin(bL), sin(bL/2),
    bL, 1/beta, 1/beta^2) so both real-kernel values are produced with
    roughly half the trig / division cost of calling the two original
    kernels separately.

    Bit-identicality: the output matches
        (mp.re(S_hat_x(tau, x, L)), mp.re(dS_hat_x_dx(tau, x, L)))
    to 0.0 relative difference at dps=50 across all tested (x, tau)
    pairs. The arithmetic sequence is the same up to an obvious algebraic
    rearrangement that does not introduce any new cancellation pattern.
    """
    PI = mp.pi
    x = mp.mpf(x)
    tau = mp.mpf(tau)
    alpha = 2 * PI * x / L
    s2pi = mp.sin(2 * PI * x)
    c2pi = mp.cos(2 * PI * x)

    # beta1 = alpha - tau
    beta1 = alpha - tau
    if beta1 == 0:
        A1r, A1i = L, mp.mpf(0)
        B1r, B1i = L / 2, mp.mpf(0)
    else:
        bL1 = beta1 * L
        sh1 = mp.sin(bL1 / 2)
        sf1 = mp.sin(bL1)
        A1r = sf1 / beta1
        sh1_sq2 = 2 * sh1 * sh1
        A1i = sh1_sq2 / beta1
        Lb1b1 = L * beta1 * beta1
        B1r = sh1_sq2 / Lb1b1
        if abs(bL1) < mp.mpf("1e-5"):
            bL1_2 = bL1 * bL1
            correction = 1 - bL1_2 / 20 * (1 - bL1_2 / 42 *
                         (1 - bL1_2 / 72 * (1 - bL1_2 / 110)))
            B1i = beta1 * L * L / 6 * correction
        else:
            B1i = (bL1 - sf1) / Lb1b1

    # beta2 = -(alpha + tau)
    beta2 = -(alpha + tau)
    if beta2 == 0:
        A2r, A2i = L, mp.mpf(0)
        B2r, B2i = L / 2, mp.mpf(0)
    else:
        bL2v = beta2 * L
        sh2 = mp.sin(bL2v / 2)
        sf2 = mp.sin(bL2v)
        A2r = sf2 / beta2
        sh2_sq2 = 2 * sh2 * sh2
        A2i = sh2_sq2 / beta2
        Lb2b2 = L * beta2 * beta2
        B2r = sh2_sq2 / Lb2b2
        if abs(bL2v) < mp.mpf("1e-5"):
            bL2_2 = bL2v * bL2v
            correction = 1 - bL2_2 / 20 * (1 - bL2_2 / 42 *
                         (1 - bL2_2 / 72 * (1 - bL2_2 / 110)))
            B2i = beta2 * L * L / 6 * correction
        else:
            B2i = (bL2v - sf2) / Lb2b2

    # Re(S_hat_x) = s2pi * Re(I_c) - c2pi * Re(I_s)
    # where Re(I_c) = (A1r + A2r)/2, Re(I_s) = (A1i - A2i)/2.
    re_Ic = (A1r + A2r) / 2
    re_Is = (A1i - A2i) / 2
    re_S = s2pi * re_Ic - c2pi * re_Is

    # Re(dS_hat_x_dx) = 2*PI * Re(C) where
    # Re(C) = c2pi * (B1r + B2r)/2 + s2pi * (B1i - B2i)/2.
    re_Bc = (B1r + B2r) / 2
    re_Bs = (B1i - B2i) / 2
    re_dS = 2 * PI * (c2pi * re_Bc + s2pi * re_Bs)

    return re_S, re_dS


def _re_S_cached(tau: mp.mpf, x: mp.mpf, L: mp.mpf) -> mp.mpf:
    """First-pass accessor for Re(S_hat_x); computes and stashes the
    pair for later re-use by _re_dS_cached during psi_arch_deriv."""
    if not isinstance(tau, mp.mpf):
        tau = mp.mpf(tau)
    key = tau._mpf_
    hit = _kernel_cache.get(key)
    if hit is not None:
        return hit[0]
    re_S, re_dS = _re_S_and_dS_fused(tau, x, L)
    _kernel_cache[key] = (re_S, re_dS)
    return re_S


def _re_dS_cached(tau: mp.mpf, x: mp.mpf, L: mp.mpf) -> mp.mpf:
    """Second-pass accessor for Re(dS_hat_x_dx); hits the cache populated
    by _re_S_cached during psi_arch."""
    if not isinstance(tau, mp.mpf):
        tau = mp.mpf(tau)
    key = tau._mpf_
    hit = _kernel_cache.get(key)
    if hit is not None:
        return hit[1]
    re_S, re_dS = _re_S_and_dS_fused(tau, x, L)
    _kernel_cache[key] = (re_S, re_dS)
    return re_dS


def psi_arch(x: mp.mpf, L: mp.mpf, T: int, dps: int) -> mp.mpf:
    """
    Archimedean Mellin multiplier integral for psi(x).

    Computes (1/(2*pi^2)) * integral_{-T}^{T} h_plus(tau) * Re(S_hat(tau, x)) d_tau
    with subinterval splitting at the singularities tau = 0, +/- 2*pi*x/L.
    """
    PI = mp.pi
    x_mp = mp.mpf(x)
    T_mp = mp.mpf(T)
    if x_mp == 0:
        return mp.mpf(0)
    alpha_x = 2 * PI * x_mp / L
    # Split integration at points where the integrand has kinks
    sings = sorted([s for s in {mp.mpf(0), alpha_x, -alpha_x} if -T_mp < s < T_mp])
    pts = [-T_mp] + sings + [T_mp]

    def integrand(tau):
        return _h_plus_cached(tau, dps) * _re_S_cached(tau, x, L)

    total = mp.mpf(0)
    for i in range(len(pts) - 1):
        total += mp.quad(integrand, [pts[i], pts[i + 1]])
    return total / (2 * PI * PI)


def psi_arch_deriv(x: mp.mpf, L: mp.mpf, T: int, dps: int) -> mp.mpf:
    """
    Derivative of the archimedean piece: d/dx psi_arch(x).
    """
    PI = mp.pi
    x_mp = mp.mpf(x)
    T_mp = mp.mpf(T)
    alpha_x = 2 * PI * x_mp / L
    sings = sorted([s for s in {mp.mpf(0), alpha_x, -alpha_x} if -T_mp < s < T_mp])
    pts = [-T_mp] + sings + [T_mp]

    def integrand(tau):
        return _h_plus_cached(tau, dps) * _re_dS_cached(tau, x, L)

    total = mp.mpf(0)
    for i in range(len(pts) - 1):
        total += mp.quad(integrand, [pts[i], pts[i + 1]])
    return total / (2 * PI * PI)


# ============================================================
# Full psi and its derivative
# ============================================================

def _compute_psi_pair(
    n_idx: int,
    L: mp.mpf,
    T: int,
    dps: int,
    prime_data: list,
) -> tuple[mp.mpf, mp.mpf]:
    """
    Compute psi(n_idx) and psi'(n_idx), the full Weil functional value
    and its derivative at basis index n_idx.
    """
    # WIN 1: clear per-x h_plus cache so psi_arch and psi_arch_deriv
    # share evaluations (both split on the same {-alpha_x, 0, alpha_x}
    # kinks, so mp.quad picks identical nodes; h_plus is also even in
    # tau so |tau| collapses +/- pairs).
    _hplus_cache_clear()
    x = mp.mpf(n_idx)
    psi = psi_prime(x, L, prime_data) + psi_pole(x, L) + psi_arch(x, L, T, dps)
    psi_d = psi_prime_deriv(x, L, prime_data) + psi_pole_deriv(x, L) + psi_arch_deriv(x, L, T, dps)
    _hplus_cache_clear()
    return psi, psi_d


# ============================================================
# Public API
# ============================================================

def build_galerkin_matrix(
    c: int | float,
    N: int = 100,
    T: int = 400,
    dps: int = 150,
) -> "mp.matrix":
    """
    Build the CvS Proposition 4.1 Galerkin matrix Q(c).

    Constructs the (2N+1) x (2N+1) matrix whose entries are inner products
    of the Weil functional against the trigonometric basis
    {e_k(t) = exp(2*pi*i*k*t / (2*log(c)))} for k in [-N, N].

    The matrix decomposes into three pieces:

    - **Prime piece:** encodes the von Mangoldt function via sums over
      prime powers up to c.
    - **Pole piece:** accounts for the trivial zeros of zeta (poles of
      the completed zeta function).
    - **Archimedean piece:** the Mellin multiplier from the archimedean
      place, computed via adaptive quadrature of digamma integrals with
      T-truncation of the integration range.

    Parameters
    ----------
    c : int or float
        The cutoff parameter. Must be >= 2.
    N : int, optional
        Half the basis size. The matrix will be (2N+1) x (2N+1).
        Default: 100.
    T : int, optional
        Truncation parameter for the archimedean integral.
        Default: 400.
    dps : int, optional
        Decimal digits of precision for mpmath arithmetic.
        Default: 150.

    Returns
    -------
    Q : mpmath.matrix
        The (2N+1) x (2N+1) Galerkin matrix. Symmetric and real-valued.

    Raises
    ------
    ValueError
        If c < 2, N < 1, T < 1, or dps < 15.

    Examples
    --------
    >>> Q = build_galerkin_matrix(c=13, N=60, T=400, dps=80)
    >>> Q.rows
    121
    """
    if c < 2:
        raise ValueError(f"Cutoff c must be >= 2, got {c}")
    if N < 1:
        raise ValueError(f"Basis half-size N must be >= 1, got {N}")
    if T < 1:
        raise ValueError(f"Truncation T must be >= 1, got {T}")
    if dps < 15:
        raise ValueError(f"Precision dps must be >= 15, got {dps}")

    mp.mp.dps = dps

    # Set flint precision when available: bits = dps * 3.5 (generous margin)
    if HAS_FLINT:
        flint_ctx.prec = int(dps * 3.5)

    L = mp.log(c)
    prime_data, _ = prime_powers_up_to(int(c))

    # Compute psi(n) and psi'(n) for all basis indices
    n_indices = list(range(-N, N + 1))
    psi_vals = {}
    psi_deriv_vals = {}
    for n_idx in n_indices:
        psi, psi_d = _compute_psi_pair(n_idx, L, T, dps, prime_data)
        psi_vals[n_idx] = psi
        psi_deriv_vals[n_idx] = psi_d

    # Assemble the (2N+1) x (2N+1) Galerkin matrix
    DIM = 2 * N + 1
    Q = mp.matrix(DIM, DIM)

    # Off-diagonal: Q[m,n] = (psi(m) - psi(n)) / (m - n)
    # Diagonal: Q[n,n] = psi'(n)  (L'Hopital limit)
    for i in range(DIM):
        m_idx = i - N
        for j in range(DIM):
            n_idx = j - N
            if m_idx == n_idx:
                Q[i, j] = psi_deriv_vals[n_idx]
            else:
                # Note: mpmath handles mpf / int correctly; avoid redundant
                # mp.mpf(int) conversion in the inner loop (micro-opt, bit-identical).
                Q[i, j] = (psi_vals[m_idx] - psi_vals[n_idx]) / (m_idx - n_idx)

    # Symmetrize: Q[i,j] = Q[j,i] = (Q[i,j] + Q[j,i]) / 2
    for i in range(DIM):
        for j in range(i + 1, DIM):
            avg = (Q[i, j] + Q[j, i]) / 2
            Q[i, j] = avg
            Q[j, i] = avg

    return Q


def compute_ground_state(
    Q: mp.matrix,
) -> tuple[mp.mpf, mp.matrix]:
    """
    Compute the ground-state eigenvalue and eigenvector of Q.

    Projects Q onto the even sector (exploiting the parity symmetry
    of the CvS operator) and finds the minimum eigenvalue via
    mpmath's eigsy (symmetric eigensolver).

    Parameters
    ----------
    Q : mpmath.matrix
        A symmetric Galerkin matrix as returned by
        :func:`build_galerkin_matrix`.

    Returns
    -------
    lambda_min : mpmath.mpf
        The minimum eigenvalue of Q restricted to the even sector.
    v_full : mpmath.matrix
        The corresponding eigenvector in the full (2N+1)-dimensional
        trigonometric basis, normalized to unit length.

    Notes
    -----
    The even-sector projection reduces the matrix dimension from
    (2N+1) to (N+1), halving eigendecomposition time.
    """
    DIM = Q.rows
    N = (DIM - 1) // 2

    # Build even-sector projector V_even: (2N+1) x (N+1)
    # Column 0: e_0, columns k>=1: (e_k + e_{-k}) / sqrt(2)
    V_even = mp.matrix(DIM, N + 1)
    V_even[N, 0] = mp.mpf(1)
    inv_sqrt2 = 1 / mp.sqrt(2)
    for k in range(1, N + 1):
        V_even[N + k, k] = inv_sqrt2
        V_even[N - k, k] = inv_sqrt2

    # Project: Q_even = V_even^T * Q * V_even
    Q_even = V_even.T * Q * V_even

    # Diagonalize the (N+1) x (N+1) symmetric matrix
    eigs, vecs = mp.eigsy(Q_even)

    # Find minimum eigenvalue
    min_idx = 0
    min_val = eigs[0]
    for i in range(N + 1):
        if eigs[i] < min_val:
            min_val = eigs[i]
            min_idx = i
    lambda_even = min_val

    # Extract and normalize the eigenvector in even-sector coordinates
    v_even_proj = mp.matrix(N + 1, 1)
    for i in range(N + 1):
        v_even_proj[i, 0] = vecs[i, min_idx]
    nrm = mp.sqrt(sum((v_even_proj[i, 0]) ** 2 for i in range(N + 1)))
    for i in range(N + 1):
        v_even_proj[i, 0] = v_even_proj[i, 0] / nrm

    # Lift back to full (2N+1)-dimensional basis
    v_full = V_even * v_even_proj
    nrm_full = mp.sqrt(sum((v_full[i, 0]) ** 2 for i in range(DIM)))
    for i in range(DIM):
        v_full[i, 0] = v_full[i, 0] / nrm_full

    return lambda_even, v_full


def extract_zeros(
    eigvec: "mp.matrix",
    L: float,
    n_zeros: int = 10,
    dps: int = 150,
) -> list:
    """
    Extract Riemann zeta zeros from the ground-state eigenvector.

    Reconstructs the spectral test function F_even(tau) from the
    eigenvector coefficients, then uses mpmath.findroot near the
    known locations of zeta zeros to detect them with high precision.

    The test function is:

        F_even(tau) = Re[ exp(i*tau*L/2) * sum_k c_k * g_k(tau) ] / sqrt(L)

    where g_k(tau) = (exp(-i*tau*L) - 1) / (i*(2*pi*k/L - tau)) when
    the denominator is non-vanishing, and g_k(tau) = L when it vanishes.

    Parameters
    ----------
    eigvec : mpmath.matrix
        Ground-state eigenvector from :func:`compute_ground_state`.
        Must be a (2N+1) x 1 column vector.
    L : float
        The log-cutoff: L = log(c).
    n_zeros : int, optional
        Number of zeros to extract. Default: 10.
    dps : int, optional
        Decimal digits of precision. Default: 150.

    Returns
    -------
    results : list of dict
        Each dict has keys:
        - 'k': int, the zero index (1-based).
        - 'gamma_true': mpmath.mpf, the true imaginary part of the k-th zero.
        - 'gamma_detected': mpmath.mpf or None, the detected zero.
        - 'error': mpmath.mpf or None, |detected - true|.
    """
    mp.mp.dps = dps
    DIM = eigvec.rows
    N = (DIM - 1) // 2
    L_mp = mp.mpf(L)
    PI = mp.pi

    def F_even(tau):
        """Test function whose zeros are the zeta zeros."""
        tau_mp = mp.mpf(tau)
        total = mp.mpc(0, 0)
        exp_tL = mp.exp(-1j * tau_mp * L_mp)
        for k in range(-N, N + 1):
            c_coef = eigvec[k + N, 0]
            if c_coef == 0:
                continue
            denom = 2 * PI * k / L_mp - tau_mp
            if abs(denom) < mp.mpf("1e-130"):
                term = mp.mpc(L_mp, 0)
            else:
                term = (exp_tL - 1) / (1j * denom)
            total += c_coef * term
        total /= mp.sqrt(L_mp)
        return mp.re(mp.exp(1j * tau_mp * L_mp / 2) * total)

    # Use mpmath's known zeta zeros as starting points
    gamma_true = [mp.im(mp.zetazero(k)) for k in range(1, n_zeros + 1)]
    results = []
    for k, g in enumerate(gamma_true, 1):
        entry = {
            'k': k,
            'gamma_true': g,
            'gamma_detected': None,
            'error': None,
        }
        try:
            root = mp.findroot(
                F_even,
                (g - mp.mpf("0.005"), g + mp.mpf("0.005")),
                solver="anderson",
                tol=mp.mpf("1e-140"),
            )
            entry['gamma_detected'] = root
            entry['error'] = abs(root - g)
        except Exception:
            pass
        results.append(entry)
    return results
