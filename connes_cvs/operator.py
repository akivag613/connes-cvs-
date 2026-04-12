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

from typing import TYPE_CHECKING

import mpmath as mp

from connes_cvs.kernels import S_hat_x, dS_hat_x_dx

if TYPE_CHECKING:
    from typing import Tuple

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
        return h_plus(tau, dps) * mp.re(S_hat_x(tau, x, L))

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
        return h_plus(tau, dps) * mp.re(dS_hat_x_dx(tau, x, L))

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
    x = mp.mpf(n_idx)
    psi = psi_prime(x, L, prime_data) + psi_pole(x, L) + psi_arch(x, L, T, dps)
    psi_d = psi_prime_deriv(x, L, prime_data) + psi_pole_deriv(x, L) + psi_arch_deriv(x, L, T, dps)
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
                Q[i, j] = (psi_vals[m_idx] - psi_vals[n_idx]) / mp.mpf(m_idx - n_idx)

    # Symmetrize: Q[i,j] = Q[j,i] = (Q[i,j] + Q[j,i]) / 2
    for i in range(DIM):
        for j in range(i + 1, DIM):
            avg = (Q[i, j] + Q[j, i]) / 2
            Q[i, j] = avg
            Q[j, i] = avg

    return Q


def compute_ground_state(
    Q: "mp.matrix",
) -> "Tuple[mp.mpf, mp.matrix]":
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
