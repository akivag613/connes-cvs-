"""
Core operator construction and diagonalization.

Implements the CvS Proposition 4.1 Galerkin matrix Q(c, N) and its
eigendecomposition. The matrix decomposes as:

    Q = Q_prime + Q_pole + Q_arch

where each piece encodes a different arithmetic contribution to the
Weil explicit formula.

The archimedean piece uses python-flint's compiled arbitrary-precision
``acb.digamma`` when available and falls back to mpmath transparently.
Performance depends on precision, workload, versions, and hardware; no
backend speedup multiplier is asserted here.

References
----------
- Connes & van Suijlekom, arXiv:2511.23257, Proposition 4.1
- Connes, Consani & Moscovici, arXiv:2511.22755, Section 6
"""

from __future__ import annotations

from functools import wraps
from numbers import Integral
from typing import Callable, Optional, ParamSpec, TypeVar, Union
import warnings

import mpmath as mp

from connes_cvs.kernels import _stable_b_imag

# ============================================================
# Optional python-flint for fast digamma
# ============================================================
_FLINT_VERSION: Optional[str]
try:
    import flint as _flint_module
    from flint import acb, arb, ctx as flint_ctx
    HAS_FLINT = True
    _FLINT_VERSION = str(getattr(_flint_module, "__version__", "unknown"))
except ImportError:
    HAS_FLINT = False
    _FLINT_VERSION = None


_PrecisionArgs = ParamSpec("_PrecisionArgs")
_PrecisionResult = TypeVar("_PrecisionResult")


def _preserve_context_on_failure(
    operation: Callable[_PrecisionArgs, _PrecisionResult],
) -> Callable[_PrecisionArgs, _PrecisionResult]:
    """Restore numerical context if an API call raises; keep success semantics.

    mpmath's binary precision is the exact state. Saving decimal ``dps``
    would round a caller's manually chosen bit count on restoration.
    This does not roll back caches or make the global contexts thread-safe.
    """
    @wraps(operation)
    def protected(*args: _PrecisionArgs.args, **kwargs: _PrecisionArgs.kwargs) -> _PrecisionResult:
        prior_bits = mp.mp.prec
        prior_flint = (flint_ctx.prec, flint_ctx.threads) if HAS_FLINT else None
        try:
            return operation(*args, **kwargs)
        except BaseException:
            mp.mp.prec = prior_bits
            if prior_flint is not None:
                flint_ctx.prec, flint_ctx.threads = prior_flint
            raise

    return protected


def _validated_int(name: str, value: int, minimum: int) -> int:
    """Return ``value`` as an int after rejecting lossy or boolean inputs."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value_int = int(value)
    if value_int < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value_int}")
    return value_int


def _validated_positive_mpf(name: str, value) -> mp.mpf:
    """Return a finite positive mpf, with a precise error for invalid input."""
    try:
        value_mp = mp.mpf(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric value, got {value!r}") from exc
    if not mp.isfinite(value_mp):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value_mp <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value_mp


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
    c = _validated_int("c", c, 2)
    is_prime = [True] * (c + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, c + 1):
        if is_prime[i]:
            j = i * i
            while j <= c:
                is_prime[j] = False
                j += i
    primes_list = [i for i in range(2, c + 1) if is_prime[i]]
    prime_base_by_power = {}
    for p in primes_list:
        pk = p
        while pk <= c:
            prime_base_by_power[pk] = p
            if pk > c // p:
                break
            pk *= p
    result = [
        (n, mp.log(n), mp.log(prime_base_by_power[n]) / mp.sqrt(n))
        for n in sorted(prime_base_by_power)
    ]
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

    Accounts for the zeta-pole term: integral over [0, L] of
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

_flint_log_pi_cache: dict[int, object] = {}


def _flint_log_pi():
    """Return ``log(pi)`` in Arb, cached separately at each Flint precision."""
    prec = int(flint_ctx.prec)
    cached = _flint_log_pi_cache.get(prec)
    if cached is None:
        cached = arb.pi().log()
        _flint_log_pi_cache[prec] = cached
    return cached


def _h_plus_flint(tau: mp.mpf, dps: int) -> mp.mpf:
    """
    Compute h_plus(tau) = Re(digamma(1/4 + i*tau/2)) - log(pi)
    using python-flint's acb.digamma (Arb library).
    """
    log_pi_fl = _flint_log_pi()
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
    Uses python-flint when available for a substantially faster compiled
    arbitrary-precision digamma backend.

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
# WIN 1: h_plus memoization cache (arithmetic-preserving optimization)
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
# h_plus evaluations. A cache hit returns the already computed mpf value
# for the identical absolute input. Cross-x nodes do NOT overlap (different subinterval
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
# the cache and skip the kernel work entirely. The fused kernel was checked
# against the original stable_A + stable_B + S_hat_x composition at its
# historical development test points; the release regression tests cover
# the resulting operator values independently.
_kernel_cache: dict = {}       # tau._mpf_ -> (re_S, re_dS)


def _arithmetic_context_key(dps: Optional[int] = None) -> tuple:
    """Identity of arithmetic state that can affect a cached value."""
    return (
        int(mp.mp.prec),
        int(mp.mp.dps),
        int(dps) if dps is not None else None,
        str(getattr(mp.libmp, "BACKEND", "unknown")),
        str(getattr(mp, "__version__", "unknown")),
        int(flint_ctx.prec) if HAS_FLINT else None,
        _FLINT_VERSION,
    )


def _h_plus_cached(tau: mp.mpf, dps: int) -> mp.mpf:
    """
    Memoized wrapper over h_plus that exploits h_plus's evenness in tau.

    Keys the cache on the raw mpf tuple of ``abs(tau)``. The returned
    value is the cached result for the identical absolute input. This uses
    the exact mathematical evenness of ``h_plus`` without recomputing it.
    """
    if not isinstance(tau, mp.mpf):
        tau = mp.mpf(tau)
    # Key on abs(tau) to collapse +-tau pairs. mp.mpf._mpf_ is hashable.
    key = (abs(tau)._mpf_, _arithmetic_context_key(dps))
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

    Historical development probes found 0.0 relative difference at dps=50
    from separately evaluating ``S_hat_x`` and ``dS_hat_x_dx`` on their
    sampled inputs. Release-level correctness is established by the operator
    regression tests, not by extrapolating that probe to other workloads.
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
        B1i = _stable_b_imag(beta1, L, bL1, sf1)

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
        B2i = _stable_b_imag(beta2, L, bL2v, sf2)

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
    x_mp = mp.mpf(x)
    L_mp = mp.mpf(L)
    key = (tau._mpf_, x_mp._mpf_, L_mp._mpf_, _arithmetic_context_key())
    hit = _kernel_cache.get(key)
    if hit is not None:
        return hit[0]
    re_S, re_dS = _re_S_and_dS_fused(tau, x_mp, L_mp)
    _kernel_cache[key] = (re_S, re_dS)
    return re_S


def _re_dS_cached(tau: mp.mpf, x: mp.mpf, L: mp.mpf) -> mp.mpf:
    """Second-pass accessor for Re(dS_hat_x_dx); hits the cache populated
    by _re_S_cached during psi_arch."""
    if not isinstance(tau, mp.mpf):
        tau = mp.mpf(tau)
    x_mp = mp.mpf(x)
    L_mp = mp.mpf(L)
    key = (tau._mpf_, x_mp._mpf_, L_mp._mpf_, _arithmetic_context_key())
    hit = _kernel_cache.get(key)
    if hit is not None:
        return hit[1]
    re_S, re_dS = _re_S_and_dS_fused(tau, x_mp, L_mp)
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

@_preserve_context_on_failure
def build_galerkin_matrix(
    c: Union[int, str, mp.mpf],
    N: int = 100,
    T: int = 400,
    dps: int = 150,
    flint_bits: Optional[int] = None,
) -> "mp.matrix":
    """
    Build the CvS Proposition 4.1 Galerkin matrix Q(c).

    Constructs the (2N+1) x (2N+1) matrix whose entries are inner products
    of the Weil functional against the trigonometric basis
    {e_k(t) = exp(2*pi*i*k*t / L)}, L = log(c), for k in [-N, N].

    The matrix decomposes into three pieces:

    - **Prime piece:** encodes the von Mangoldt function via sums over
      prime powers up to c.
    - **Pole piece:** the contribution associated with the pole of zeta.
    - **Archimedean piece:** the Mellin multiplier from the archimedean
      place, computed via adaptive quadrature of digamma integrals with
      T-truncation of the integration range.

    Parameters
    ----------
    c : int, str, or mpmath.mpf
        The cutoff parameter. Must be >= 2. Python ``float`` is rejected
        because its binary64 rounding silently caps input accuracy; pass a
        decimal string or ``mp.mpf`` for a nonintegral cutoff.
    N : int, optional
        Half the basis size. The matrix will be (2N+1) x (2N+1).
        Default: 100.
    T : int, optional
        Truncation parameter for the archimedean integral.
        Default: 400.
    dps : int, optional
        Decimal digits of precision for mpmath arithmetic.
        Default: 150.
    flint_bits : int, optional
        Explicit Arb working precision. The default preserves the historical
        package convention ``int(3.5*dps)``. Published c=100 production
        artifacts used ``4*dps`` and should be reproduced with that value.

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
    N = _validated_int("Basis half-size N", N, 1)
    T = _validated_int("Truncation T", T, 1)
    dps = _validated_int("Precision dps", dps, 15)
    if flint_bits is None:
        flint_bits = int(dps * 3.5)
    else:
        minimum_bits = (3322 * dps + 999) // 1000
        flint_bits = _validated_int("flint_bits", flint_bits, minimum_bits)
    mp.mp.dps = dps
    if isinstance(c, float):
        raise TypeError(
            "Cutoff c must not be a Python float; pass an integer, decimal "
            "string, or mp.mpf so the requested precision is preserved"
        )
    try:
        c_mp = mp.mpf(c)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Cutoff c must be a real numeric value, got {c!r}") from exc
    if not mp.isfinite(c_mp):
        raise ValueError(f"Cutoff c must be finite, got {c!r}")
    if c_mp < 2:
        raise ValueError(f"Cutoff c must be >= 2, got {c!r}")

    # Set flint precision when available: bits = dps * 3.5 (generous margin)
    if HAS_FLINT:
        flint_ctx.prec = flint_bits
        flint_ctx.threads = 1

    L = mp.log(c_mp)
    prime_data, _ = prime_powers_up_to(int(mp.floor(c_mp)))

    # The exact parity identities psi(-n) = -psi(n) and psi'(-n) = psi'(n)
    # let us retain each existing [-T,T] quadrature unchanged while evaluating
    # only N+1 basis points instead of 2N+1. Mirroring completed mpf values is
    # raw-identical to evaluating the negative index directly on the configured
    # backend; the hardening suite checks direct parity and legacy full assembly.
    psi_vals = {}
    psi_deriv_vals = {}
    for n_idx in range(0, N + 1):
        psi, psi_d = _compute_psi_pair(n_idx, L, T, dps, prime_data)
        psi_vals[n_idx] = psi
        psi_deriv_vals[n_idx] = psi_d
        if n_idx:
            psi_vals[-n_idx] = -psi
            psi_deriv_vals[-n_idx] = psi_d

    # Assemble the (2N+1) x (2N+1) Galerkin matrix
    DIM = 2 * N + 1
    Q = mp.matrix(DIM, DIM)

    # Off-diagonal: Q[m,n] = (psi(m) - psi(n)) / (m - n)
    # Diagonal: Q[n,n] = psi'(n)  (L'Hopital limit)
    for i in range(DIM):
        m_idx = i - N
        for j in range(i, DIM):
            n_idx = j - N
            if m_idx == n_idx:
                value = psi_deriv_vals[n_idx]
            else:
                # Note: mpmath handles mpf / int correctly; avoid redundant
                # mp.mpf(int) conversion in the inner loop (arithmetic-preserving).
                value = (psi_vals[m_idx] - psi_vals[n_idx]) / (m_idx - n_idx)
            Q[i, j] = value
            if i != j:
                Q[j, i] = value

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
    if not hasattr(Q, "rows") or not hasattr(Q, "cols"):
        raise TypeError("Q must be an mpmath matrix")
    if Q.rows != Q.cols:
        raise ValueError(f"Q must be square, got shape {Q.rows}x{Q.cols}")
    if Q.rows < 3 or Q.rows % 2 == 0:
        raise ValueError(
            "Q must have odd dimension 2N+1 with N >= 1, "
            f"got {Q.rows}x{Q.cols}"
        )
    DIM = Q.rows
    for i in range(DIM):
        if mp.im(Q[i, i]) != 0:
            raise ValueError(f"Q must be real; complex entry at ({i}, {i})")
        if not mp.isfinite(Q[i, i]):
            raise ValueError(f"Q contains a non-finite diagonal entry at ({i}, {i})")
        for j in range(i + 1, DIM):
            if mp.im(Q[i, j]) != 0 or mp.im(Q[j, i]) != 0:
                raise ValueError(f"Q must be real; complex entry at ({i}, {j})")
            if not mp.isfinite(Q[i, j]) or not mp.isfinite(Q[j, i]):
                raise ValueError(f"Q contains a non-finite entry at ({i}, {j})")
            if Q[i, j] != Q[j, i]:
                raise ValueError(f"Q must be exactly symmetric; mismatch at ({i}, {j})")
    # The projection below returns an eigenvector of the full matrix only when
    # Q preserves the reversal-even subspace.  Symmetry alone is insufficient:
    # for example, diag(1, 2, 3) is symmetric but its projected eigenvectors
    # need not be eigenvectors of Q.  CvS matrices commute exactly with the
    # reversal operator, so require that contract before using the fast sector
    # eigensolve rather than returning a mathematically false eigenpair.
    for i in range(DIM):
        for j in range(DIM):
            reverse_i = DIM - 1 - i
            reverse_j = DIM - 1 - j
            if Q[i, j] != Q[reverse_i, reverse_j]:
                raise ValueError(
                    "Q must be exactly centrosymmetric (reversal-invariant); "
                    f"mismatch at ({i}, {j}) versus ({reverse_i}, {reverse_j})"
                )
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


@_preserve_context_on_failure
def extract_zeros(
    eigvec: "mp.matrix",
    L: Optional[mp.mpf] = None,
    n_zeros: int = 10,
    dps: int = 150,
    c: Optional[Union[int, str, mp.mpf]] = None,
    tol: Optional[Union[mp.mpf, str]] = None,
    strict: bool = False,
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
    L : mpmath.mpf, optional
        The log-cutoff: L = log(c), as a full-precision ``mp.mpf``
        (e.g. ``mp.log(13)``). Accepted for backward compatibility.
        Passing a Python ``float`` (e.g. ``math.log(13)``) silently
        carries only ~16 significant digits and caps the extraction
        accuracy near 1e-16; a ``UserWarning`` is emitted in that case.
        Prefer the ``c`` parameter instead.
    n_zeros : int, optional
        Number of zeros to extract. Default: 10.
    dps : int, optional
        Decimal digits of precision. Default: 150.
    c : int, str, or mpmath.mpf, optional
        The exact cutoff parameter. Python ``float`` is rejected; pass a
        decimal string or ``mp.mpf`` for a nonintegral cutoff. If given,
        ``L = mp.log(mp.mpf(c))`` is computed internally at the active
        precision, which avoids the float64 pitfall entirely. Preferred over
        passing ``L`` directly (in v0.3.0). Exactly one of ``L`` and ``c``
        must be given.
    tol : mpmath.mpf or str, optional
        Positive root-finding tolerance. The default is mpmath's own
        precision-adaptive value, ``mp.eps * 2**10``, so high-dps calls are
        not capped by a fixed decimal threshold.
    strict : bool, optional
        If True, propagate root-finding nonconvergence. If False (default),
        emit a RuntimeWarning and return a diagnostic in that zero's result.

    Returns
    -------
    results : list of dict
        Each dict has keys ``k``, ``gamma_true``, ``gamma_detected``,
        ``error``, ``residual``, ``converged``, ``failure``, and
        ``tolerance``. The detected/error/residual fields are ``None`` on
        nonconvergence unless ``strict=True`` propagates the failure.
    """
    dps = _validated_int("Precision dps", dps, 15)
    n_zeros = _validated_int("n_zeros", n_zeros, 1)
    if not isinstance(strict, bool):
        raise TypeError(f"strict must be a bool, got {type(strict).__name__}")
    if not hasattr(eigvec, "rows") or not hasattr(eigvec, "cols"):
        raise TypeError("eigvec must be an mpmath matrix")
    if eigvec.cols != 1 or eigvec.rows < 3 or eigvec.rows % 2 == 0:
        raise ValueError(
            "eigvec must be a (2N+1)x1 column vector with N >= 1, "
            f"got {eigvec.rows}x{eigvec.cols}"
        )

    mp.mp.dps = dps
    DIM = eigvec.rows
    N = (DIM - 1) // 2
    coefficients = []
    for i in range(DIM):
        if mp.im(eigvec[i, 0]) != 0:
            raise ValueError(f"eigvec must be real; complex entry at row {i}")
        if not mp.isfinite(eigvec[i, 0]):
            raise ValueError(f"eigvec contains a non-finite entry at row {i}")
        coefficients.append(mp.mpf(eigvec[i, 0]))
    for i in range(N):
        reverse_i = DIM - 1 - i
        if coefficients[i] != coefficients[reverse_i]:
            raise ValueError(
                "eigvec must be exactly reversal-even; "
                f"mismatch at rows {i} and {reverse_i}"
            )
    norm = mp.sqrt(sum(value * value for value in coefficients))
    if norm == 0:
        raise ValueError("eigvec must be nonzero")
    coefficients = [value / norm for value in coefficients]
    if c is not None and L is not None:
        raise ValueError(
            "extract_zeros: pass either c or L, not both "
            "(c computes L = mp.log(c) internally at the active precision)."
        )
    if c is None and L is None:
        raise ValueError(
            "extract_zeros: one of c or L is required "
            "(prefer c; L is kept for backward compatibility)."
        )
    if c is not None:
        if isinstance(c, float):
            raise TypeError(
                "c must not be a Python float; pass an integer, decimal "
                "string, or mp.mpf so the requested precision is preserved"
            )
        try:
            c_mp = mp.mpf(c)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"c must be a real numeric value, got {c!r}") from exc
        if not mp.isfinite(c_mp) or c_mp <= 1:
            raise ValueError(f"c must be finite and > 1, got {c!r}")
        L_mp = mp.log(c_mp)
    else:
        if isinstance(L, float):
            warnings.warn(
                "extract_zeros received L as a Python float (float64), which "
                "carries only ~16 significant digits and caps the achievable "
                "zero-extraction accuracy near 1e-16 regardless of dps. Pass "
                "L = mp.log(c) as an mpmath mpf, or use the c= parameter "
                "(extract_zeros(eigvec, c=13, ...)) to compute L internally "
                "at full precision.",
                UserWarning,
                stacklevel=2,
            )
        try:
            L_mp = mp.mpf(L)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"L must be a real numeric value, got {L!r}") from exc
        if not mp.isfinite(L_mp) or L_mp <= 0:
            raise ValueError(f"L must be finite and positive, got {L!r}")
    if tol is None:
        tol_mp = mp.eps * (2 ** 10)
    else:
        tol_mp = _validated_positive_mpf("tol", tol)
    PI = mp.pi

    def F_even(tau):
        """Finite test function whose roots are sought near zeta ordinates."""
        tau_mp = mp.mpf(tau)
        total = mp.mpc(0, 0)
        exp_tL = mp.exp(-1j * tau_mp * L_mp)
        for k in range(-N, N + 1):
            c_coef = coefficients[k + N]
            if c_coef == 0:
                continue
            denom = 2 * PI * k / L_mp - tau_mp
            if denom == 0:
                term = mp.mpc(L_mp, 0)
            elif abs(denom * L_mp) < mp.sqrt(mp.eps):
                # Since tau*L = 2*pi*k - denom*L, the numerator is
                # expm1(i*denom*L). This avoids cancellation near the
                # removable singularity without changing the ordinary path.
                term = mp.expm1(1j * denom * L_mp) / (1j * denom)
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
            'residual': None,
            'converged': False,
            'failure': None,
            'tolerance': tol_mp,
        }
        try:
            lower = g - mp.mpf("0.005")
            upper = g + mp.mpf("0.005")
            root = mp.findroot(
                F_even,
                (lower, upper),
                solver="anderson",
                tol=tol_mp,
            )
            try:
                root_mp = mp.mpf(root)
            except (TypeError, ValueError) as exc:
                raise ValueError("root candidate must be a finite real number") from exc
            if not mp.isfinite(root_mp):
                raise ValueError("root candidate must be finite")
            if not lower <= root_mp <= upper:
                raise ValueError(
                    "root candidate left the local search window: "
                    f"{mp.nstr(root_mp, 12)} not in "
                    f"[{mp.nstr(lower, 12)}, {mp.nstr(upper, 12)}]"
                )
            residual = abs(F_even(root_mp))
            if not mp.isfinite(residual):
                raise ValueError("root residual must be finite")
            if residual > tol_mp:
                raise ValueError(
                    "root residual exceeds the requested tolerance: "
                    f"{mp.nstr(residual, 8)} > {mp.nstr(tol_mp, 8)}"
                )
            entry['gamma_detected'] = root_mp
            entry['error'] = abs(root_mp - g)
            entry['residual'] = residual
            entry['converged'] = True
        except ValueError as exc:
            entry['failure'] = str(exc)
            if strict:
                raise
            warnings.warn(
                f"extract_zeros did not converge for zeta zero {k}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        results.append(entry)
    return results
