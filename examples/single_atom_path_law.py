#!/usr/bin/env python3
"""Demonstrate the Single-Atom Path Law on this package's prime blocks.

The prime piece of the CvS Galerkin matrix is a sum of Loewner
divided-difference blocks, one per prime power q = p^k <= c. The
Single-Atom Path Law (White 2026) gives the infinite-band operator norm
of each block in closed form:

    ||Delta O_q|| = 2 cos(pi / (m + 1)) * Lambda(q) / sqrt(q),
    m = ceil(L / log q),  L = log c.

Mechanism: the discrete Fourier transform carries the single-atom block
to a masked rotation by angle 2*pi*log(q)/L on the circle; by the
Steinhaus Three-Distance theorem its orbits partition into path graphs
P_m and P_{m-1}, whose extreme adjacency eigenvalue is the Chebyshev
value 2 cos(pi/(m+1)). The finite-N norms Cauchy-interlace up to this
value from below.

This script builds each single-atom block with this package's own
``psi_prime`` / ``psi_prime_deriv`` (so the conventions are identical to
``build_galerkin_matrix``) and checks the closed form at float64
precision, then prints two consequences:

1. A dimension-independent Weyl stability budget for the prime piece:
   any prime atom moves every eigenvalue of tau(c, N) by at most
   ||Delta O_q||, for every N.
2. What actually changes across the non-prime step c = 13 -> 14 of
   Groskin (arXiv:2605.20224) Section 8.8: the capacity integers m and
   the surviving atom norms are unchanged, while the boundary atom
   q = 13 switches on discontinuously (norm 0 -> 0.7114) and every
   fractional measure frac(L / log q) increases, moving spectral mass
   from the short chains P_{m-1} into the long chains P_m.

Runs in about a second. Requires numpy (installed by the ``science`` or
``all`` extras) in addition to the package dependencies.

Usage:
    python examples/single_atom_path_law.py

Reference: J. White, "The Single-Atom Path Law: Masked Rotations on the
Circle, Three-Distance Dynamics, and the Prime Spectra of Truncated Weil
Matrices" (2026), with reference implementation at
https://github.com/j-white/pathlaw.
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np

from connes_cvs.operator import prime_powers_up_to, psi_prime, psi_prime_deriv


def path_law_capacity(q: int, c: int) -> int:
    """Return the capacity integer m = ceil(log(c) / log(q)).

    The small subtraction guards the exact-integer-ratio case (e.g.
    q = 5, c = 25, where L / log q = 2) against float rounding up.
    """
    return max(2, math.ceil(math.log(c) / math.log(q) - 1e-12))


def path_law_norm(q: int, c: int, w_q: mp.mpf) -> float:
    """Closed-form infinite-band norm of the single-atom block for q at cutoff c.

    Parameters
    ----------
    q : int
        Prime power, 2 <= q <= c.
    c : int
        Cutoff parameter.
    w_q : mpmath.mpf
        Arithmetic weight Lambda(q)/sqrt(q), as returned by
        :func:`connes_cvs.operator.prime_powers_up_to`.

    Returns
    -------
    float
        2 cos(pi/(m+1)) * w_q, or 0.0 for the boundary atom q = c
        (its lag equals the full window, so the mask has measure zero).
    """
    if q >= c:
        return 0.0
    m = path_law_capacity(q, c)
    return 2.0 * math.cos(math.pi / (m + 1)) * float(w_q)


def single_atom_block(q: int, c: int, N: int) -> np.ndarray:
    """Assemble the (2N+1) x (2N+1) prime block containing only the atom q.

    Uses the package's ``psi_prime`` / ``psi_prime_deriv`` restricted to a
    single prime power, with the same divided-difference assembly and
    parity mirroring as ``build_galerkin_matrix``, evaluated at float64.
    """
    L = mp.log(c)
    all_data, _ = prime_powers_up_to(c)
    data = [entry for entry in all_data if entry[0] == q]
    if len(data) != 1:
        raise ValueError(f"q={q} is not a prime power <= {c}")
    psi = {n: psi_prime(mp.mpf(n), L, data) for n in range(0, N + 1)}
    dpsi = {n: psi_prime_deriv(mp.mpf(n), L, data) for n in range(0, N + 1)}
    for n in range(1, N + 1):
        psi[-n] = -psi[n]
        dpsi[-n] = dpsi[n]
    dim = 2 * N + 1
    block = np.zeros((dim, dim))
    for i in range(dim):
        m_idx = i - N
        for j in range(i, dim):
            n_idx = j - N
            if m_idx == n_idx:
                value = dpsi[n_idx]
            else:
                value = (psi[m_idx] - psi[n_idx]) / (m_idx - n_idx)
            block[i, j] = block[j, i] = float(value)
    return block


def spectral_norm(matrix: np.ndarray) -> float:
    """Largest absolute eigenvalue of a real symmetric matrix."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    return float(max(abs(eigenvalues[0]), abs(eigenvalues[-1])))


def frac_measure(q: int, c: int) -> float:
    """Fractional part of L / log q: the Three-Distance measure of the long chains P_m."""
    ratio = math.log(c) / math.log(q)
    frac = ratio - math.floor(ratio + 1e-12)
    return 1.0 if abs(frac) < 1e-12 else frac


def main() -> None:
    mp.mp.dps = 30
    c, N = 25, 48

    print(f"Single-Atom Path Law at c = {c}, checked against psi_prime blocks at N = {N}")
    print(f"{'q':>4} {'m':>3} {'factor':>10} {'w_q':>10} {'closed form':>12} "
          f"{'empirical':>12} {'rel err':>9}")
    prime_data, _ = prime_powers_up_to(c)
    for q, _logq, w_q in prime_data:
        theory = path_law_norm(q, c, w_q)
        empirical = spectral_norm(single_atom_block(q, c, N))
        m = path_law_capacity(q, c)
        factor = 2.0 * math.cos(math.pi / (m + 1)) if q < c else 0.0
        rel = abs(empirical - theory) / theory if theory else 0.0
        print(f"{q:4d} {m:3d} {factor:10.6f} {float(w_q):10.6f} {theory:12.8f} "
              f"{empirical:12.8f} {rel:9.2e}")
    print("Chebyshev factors are exact algebraic numbers: 1 (m=2), sqrt(2) (m=3), "
          "golden ratio (m=4), sqrt(3) (m=5).")
    print("The q = 2 atom exceeds the rank-one heuristic Lambda(2)/sqrt(2) by "
          "sqrt(3), a 73% underestimate;")
    print(f"the boundary atom q = {c} has lag equal to the full window and vanishes.")
    print("Finite-N norms interlace up to the closed form from below; near-boundary atoms")
    print("(here q = 23, whose long-chain measure frac(L/log q) is small) converge slower")
    print("in N: its residual falls from 4.0e-3 at N = 48 to 5.4e-8 at N = 120.\n")

    c = 13
    prime_data, _ = prime_powers_up_to(c)
    budget = sum(path_law_norm(q, c, w_q) for q, _logq, w_q in prime_data)
    rank_one = sum(float(w_q) for _q, _logq, w_q in prime_data)
    full_block = sum(single_atom_block(q, c, N) for q, _logq, _w in prime_data)
    print(f"Weyl stability budget at c = {c} (dimension-independent, here shown at N = {N}):")
    print(f"  ||W_p|| = {spectral_norm(full_block):.6f}  <=  "
          f"sum ||Delta O_q|| = {budget:.6f}   (rank-one heuristic sum: {rank_one:.6f})")
    print("  Adding or perturbing one atom moves every eigenvalue of tau(c, N) by at most")
    print("  ||Delta O_q||, for every N; removing the boundary atom moves nothing.\n")

    print("The non-prime step c = 13 -> 14 (Groskin arXiv:2605.20224, Section 8.8):")
    print(f"{'q':>4} {'m':>3} {'norm(13)':>10} {'norm(14)':>10} {'frac(13)':>9} {'frac(14)':>9}")
    prime_data_14, _ = prime_powers_up_to(14)
    for q, _logq, w_q in prime_data_14:
        norm_13 = path_law_norm(q, 13, w_q) if q <= 13 else float("nan")
        norm_14 = path_law_norm(q, 14, w_q)
        frac_13 = frac_measure(q, 13) if q <= 13 else float("nan")
        print(f"{q:4d} {path_law_capacity(q, 14):3d} {norm_13:10.6f} {norm_14:10.6f} "
              f"{frac_13:9.4f} {frac_measure(q, 14):9.4f}")
    print("  No capacity integer m changes and no surviving norm moves. The step instead")
    print("  (a) switches on the boundary atom q = 13 discontinuously (norm 0 -> 0.7114), and")
    print("  (b) raises every frac(L/log q), shifting Three-Distance spectral mass from the")
    print("  short chains P_{m-1} into the long chains P_m across all prior atoms.")


if __name__ == "__main__":
    main()
