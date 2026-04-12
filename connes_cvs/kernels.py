"""
Stable kernel functions for the CvS Galerkin matrix.

These implement the Fourier-space kernels A(beta) and B(beta) used to
evaluate the archimedean piece of the Weil explicit formula without
catastrophic cancellation near beta = 0.

The key identity is:

    A(beta) = integral_0^L exp(i*beta*y) dy  (split into real/imag via trig)
    B(beta) = (1/L) * integral_0^L y * exp(i*beta*y) dy  (derivative kernel)

Both are written in terms of sin(beta*L/2) to avoid the near-cancellation
in (exp(i*beta*L) - 1) / (i*beta) when beta*L is small.

References
----------
- Connes & van Suijlekom, arXiv:2511.23257, Proposition 4.1
- BUGFIX_eps_threshold.md in the research repository
"""

from __future__ import annotations

import mpmath as mp


def stable_A(beta: mp.mpf, L: mp.mpf) -> mp.mpc:
    """
    Stable evaluation of A(beta) = integral_0^L exp(i*beta*y) dy.

    Returns A(beta) = sin(beta*L)/beta + 2i*sin^2(beta*L/2)/beta,
    which avoids cancellation in the naive (exp(i*beta*L) - 1)/(i*beta).

    Parameters
    ----------
    beta : mpmath.mpf
        Frequency parameter.
    L : mpmath.mpf
        Log-cutoff L = log(c).

    Returns
    -------
    mp.mpc
        Complex value of A(beta).
    """
    beta = mp.mpf(beta)
    if beta == 0:
        return mp.mpc(L, 0)
    bL = beta * L
    sin_half = mp.sin(bL / 2)
    sin_full = mp.sin(bL)
    return mp.mpc(sin_full / beta, 2 * sin_half * sin_half / beta)


def stable_B(beta: mp.mpf, L: mp.mpf) -> mp.mpc:
    """
    Stable evaluation of B(beta) = (1/L) * integral_0^L y * exp(i*beta*y) dy.

    The imaginary part uses a Taylor fallback for |beta*L| < 1e-5 to avoid
    loss of significance in (beta*L - sin(beta*L)) / (L * beta^2).

    The Taylor expansion of (x - sin(x))/x^2 near x=0 is:
        x/6 * (1 - x^2/20 * (1 - x^2/42 * (1 - x^2/72 * (1 - x^2/110))))

    Parameters
    ----------
    beta : mpmath.mpf
        Frequency parameter.
    L : mpmath.mpf
        Log-cutoff L = log(c).

    Returns
    -------
    mp.mpc
        Complex value of B(beta).
    """
    beta = mp.mpf(beta)
    if beta == 0:
        return mp.mpc(L / 2, 0)
    bL = beta * L
    sin_half = mp.sin(bL / 2)
    real_part = 2 * sin_half * sin_half / (L * beta * beta)
    thresh = mp.mpf("1e-5")
    if abs(bL) < thresh:
        # Taylor fallback for (bL - sin(bL)) / (L * beta^2)
        bL2 = bL * bL
        correction = 1 - bL2 / 20 * (1 - bL2 / 42 *
                     (1 - bL2 / 72 * (1 - bL2 / 110)))
        imag_part = beta * L * L / 6 * correction
    else:
        imag_part = (bL - mp.sin(bL)) / (L * beta * beta)
    return mp.mpc(real_part, imag_part)


def S_hat_x(tau: mp.mpf, x: mp.mpf, L: mp.mpf) -> mp.mpc:
    """
    Compute S_hat(tau, x), the Fourier kernel for the archimedean integral.

    S_hat_x(tau, x) = sin(2*pi*x) * I_c(tau, x) - cos(2*pi*x) * I_s(tau, x)

    where I_c and I_s are symmetric/antisymmetric combinations of A(alpha +/- tau).

    Parameters
    ----------
    tau : mpmath.mpf
        Spectral parameter.
    x : mpmath.mpf
        Basis index (integer in practice).
    L : mpmath.mpf
        Log-cutoff L = log(c).

    Returns
    -------
    mp.mpc
        Complex value of S_hat(tau, x).
    """
    PI = mp.pi
    x = mp.mpf(x)
    tau = mp.mpf(tau)
    alpha = 2 * PI * x / L
    s2pi = mp.sin(2 * PI * x)
    c2pi = mp.cos(2 * PI * x)
    A_plus = stable_A(alpha - tau, L)
    A_minus = stable_A(-(alpha + tau), L)
    I_c = (A_plus + A_minus) / 2
    I_s = (A_plus - A_minus) / (2j)
    return s2pi * I_c - c2pi * I_s


def dS_hat_x_dx(tau: mp.mpf, x: mp.mpf, L: mp.mpf) -> mp.mpc:
    """
    Compute d/dx S_hat(tau, x), the derivative kernel for diagonal entries.

    Uses stable_B instead of stable_A for the derivative.

    Parameters
    ----------
    tau : mpmath.mpf
        Spectral parameter.
    x : mpmath.mpf
        Basis index.
    L : mpmath.mpf
        Log-cutoff L = log(c).

    Returns
    -------
    mp.mpc
        Complex value of dS_hat/dx(tau, x).
    """
    PI = mp.pi
    x = mp.mpf(x)
    tau = mp.mpf(tau)
    alpha = 2 * PI * x / L
    s2pi = mp.sin(2 * PI * x)
    c2pi = mp.cos(2 * PI * x)
    B_plus = stable_B(alpha - tau, L)
    B_minus = stable_B(-(alpha + tau), L)
    C = c2pi * (B_plus + B_minus) / 2 + s2pi * (B_plus - B_minus) / (2j)
    return 2 * PI * C
