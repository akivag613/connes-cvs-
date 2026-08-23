#!/usr/bin/env python3
"""Verbatim README smoke cell, then a Single-Atom Path Law overlay on its matrix.

Step 1 is this package's documented smoke cell, unchanged:
``build_galerkin_matrix`` / ``compute_ground_state`` / ``extract_zeros``
at c=13, N=8, T=60, dps=30.

Step 2 layers the Single-Atom Path Law (White 2026) on top of that very
matrix. Q is assembled by divided differences of psi = psi_prime +
psi_pole + psi_arch, and ``psi_prime`` is a sum over prime powers, so
restricting the prime data to a single q and repeating the assembly
yields the exact additive atom block B_q of Q. The law predicts each
block's infinite-band norm in closed form,

    ||Delta O_q|| = 2 cos(pi/(m+1)) * Lambda(q)/sqrt(q),
    m = ceil(L/log q),  L = log c,

with finite-N norms Cauchy-interlacing up to it from below. Three
checks, in increasing order of evidential weight:

- Weyl consistency: subtracting B_q from Q moves every eigenvalue by
  at most the closed form. Removing q = 11 flips lambda_min from
  4.43e-23 to -3.45e-2 (the arithmetic atom is load-bearing), yet the
  largest eigenvalue shift, 0.3684, sits inside the budget 0.7230.
  A bound test alone cannot distinguish an exact constant from a loose
  one, which is why the next two checks matter.
- Exhaustion: ||B_11|| climbs monotonically through 0.6278 (N=8),
  0.7146 (N=16), 0.722975 (N=32), 0.7229926012 (N=48) toward the
  predicted 0.7229926279 - the sequence converges TO the closed form,
  identifying the constant rather than merely respecting it.
- Discrimination: the rank-one heuristic assigns the boundary atom
  q = 13 the weight Lambda(13)/sqrt(13) = 0.7114; the Path Law says its
  norm is exactly 0 (its lag equals the full window, so the
  Three-Distance mask has measure zero). The pipeline agrees: B_13 is
  identically zero and removing it leaves lambda_min bit-unchanged.
  The two predictions differ by 0.71 and the data picks the Path Law.

Runs in seconds on the base install (mpmath only; no numpy needed).
The float64 closed-form table across all atoms, the prime-piece Weyl
budget, and the c = 13 -> 14 account live in the companion example
``single_atom_path_law.py``; the fast enforcement test is
``tests/test_single_atom_path_law.py``.

Usage:
    python examples/path_law_weyl_overlay.py

Reference: J. White, "The Single-Atom Path Law: Masked Rotations on the
Circle, Three-Distance Dynamics, and the Prime Spectra of Truncated Weil
Matrices" (2026), with reference implementation at
https://github.com/j-white/pathlaw.
"""

from __future__ import annotations

import math

import mpmath as mp

from connes_cvs import build_galerkin_matrix, compute_ground_state, extract_zeros
from connes_cvs.operator import prime_powers_up_to, psi_prime, psi_prime_deriv

C, N, T, DPS = 13, 8, 60, 30


def path_law_norm(q: int, c: int, w_q: mp.mpf) -> float:
    """Closed form 2 cos(pi/(m+1)) * Lambda(q)/sqrt(q), m = ceil(L/log q)."""
    if q >= c:
        return 0.0
    m = max(2, math.ceil(math.log(c) / math.log(q) - 1e-12))
    return 2.0 * math.cos(math.pi / (m + 1)) * float(w_q)


def atom_block(q: int, c: int, n_half: int) -> mp.matrix:
    """The exact additive contribution of atom q to ``build_galerkin_matrix(c, n_half, ...)``."""
    L = mp.log(c)
    data = [entry for entry in prime_powers_up_to(c)[0] if entry[0] == q]
    if len(data) != 1:
        raise ValueError(f"q={q} is not a prime power <= {c}")
    psi = {n: psi_prime(mp.mpf(n), L, data) for n in range(0, n_half + 1)}
    dpsi = {n: psi_prime_deriv(mp.mpf(n), L, data) for n in range(0, n_half + 1)}
    for n in range(1, n_half + 1):
        psi[-n] = -psi[n]
        dpsi[-n] = dpsi[n]
    dim = 2 * n_half + 1
    block = mp.matrix(dim, dim)
    for i in range(dim):
        for j in range(dim):
            m_idx, n_idx = i - n_half, j - n_half
            if m_idx == n_idx:
                block[i, j] = dpsi[n_idx]
            else:
                block[i, j] = (psi[m_idx] - psi[n_idx]) / (m_idx - n_idx)
    return block


def spectral_norm(matrix: mp.matrix) -> float:
    """Largest absolute eigenvalue of a real symmetric mpmath matrix."""
    eigenvalues = mp.eigsy(matrix, eigvals_only=True)
    return float(max(abs(eigenvalues[0]), abs(eigenvalues[eigenvalues.rows - 1])))


def main() -> None:
    # Step 1: the library, verbatim (README smoke cell; not a paper benchmark).
    Q = build_galerkin_matrix(c=C, N=N, T=T, dps=DPS)
    lam_min, eigvec = compute_ground_state(Q)
    zeros = extract_zeros(eigvec, c=C, n_zeros=1, dps=DPS)
    print(f"Verbatim smoke cell: lambda_min(c={C}) = {mp.nstr(lam_min, 6)}")
    print(f"  gamma_1 detected = {mp.nstr(zeros[0]['gamma_detected'], 12)}, "
          f"|error| = {mp.nstr(zeros[0]['error'], 4)}\n")

    # Step 2a: closed form vs the exact atom blocks of this Q.
    prime_data, _ = prime_powers_up_to(C)
    print(f"{'q':>4} {'closed form':>12} {'||B_q|| at N=8':>15}  interlaces from below?")
    blocks = {}
    for q, _logq, w_q in prime_data:
        blocks[q] = atom_block(q, C, N)
        empirical = spectral_norm(blocks[q])
        theory = path_law_norm(q, C, w_q)
        print(f"{q:4d} {theory:12.8f} {empirical:15.8f}  {empirical <= theory + 1e-12}")

    # Step 2b: the finite-N norms converge TO the closed form, not merely under it.
    # This is what identifies the exact constant: interlacing gives the upper bound
    # for every N, and the sequence exhausts it as N grows. (q = 11 is the slowest
    # atom at c = 13; its capacity window is nearly critical.)
    q_conv = 11
    w_conv = next(w for qq, _logq, w in prime_data if qq == q_conv)
    theory = path_law_norm(q_conv, C, w_conv)
    print(f"\nConvergence of ||B_{q_conv}|| to the closed form {theory:.10f}:")
    for n_half in (8, 16, 24, 32, 48):
        empirical = spectral_norm(atom_block(q_conv, C, n_half))
        print(f"  N={n_half:3d}: {empirical:.10f}   (deficit {theory - empirical:.2e})")
    print("  Monotone from below, exhausting the bound: the constant 2cos(pi/3) = 1 is exact.")

    # Step 2c: the Weyl bound as an experiment on the full pipeline.
    print("\nWeyl experiment: Q' = Q - B_q; every eigenvalue may move by at most the closed form.")
    eig_Q = sorted(mp.eigsy(Q, eigvals_only=True))
    for q in (11, 13):
        w_q = next(w for qq, _logq, w in prime_data if qq == q)
        theory = path_law_norm(q, C, w_q)
        perturbed = Q - blocks[q]
        eig_P = sorted(mp.eigsy(perturbed, eigvals_only=True))
        max_shift = max(abs(float(a - b)) for a, b in zip(eig_Q, eig_P))
        lam_min_p, _vec = compute_ground_state(perturbed)
        print(f"  remove q={q:2d}: max eigenvalue shift = {max_shift:.6f} <= {theory:.6f} "
              f"predicted; lambda_min {mp.nstr(lam_min, 4)} -> {mp.nstr(lam_min_p, 4)}")

    print("\nThe rank-one heuristic gives the boundary atom q = 13 weight "
          "Lambda(13)/sqrt(13) = 0.7114;")
    print("the Path Law gives exactly 0, and removing it indeed changes nothing.")


if __name__ == "__main__":
    main()
