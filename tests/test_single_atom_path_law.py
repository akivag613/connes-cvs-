"""Fast checks of the Single-Atom Path Law against psi_prime-built blocks.

The law (White 2026; reference implementation at
https://github.com/j-white/pathlaw) states that the infinite-band operator
norm of the single-atom prime block for a prime power q at cutoff c is

    ||Delta O_q|| = 2 cos(pi / (m + 1)) * Lambda(q) / sqrt(q),
    m = ceil(log(c) / log(q)),

with the boundary atom q = c vanishing. These tests assemble each atom's
block from this package's own ``psi_prime`` / ``psi_prime_deriv`` with the
same divided-difference and parity conventions as ``build_galerkin_matrix``,
at float64, and check the closed form, Cauchy interlacing from below, and
the Weyl budget for the full prime piece. Everything runs at c = 13,
N = 48 in well under a second; no slow-marked cells are touched.
"""

from __future__ import annotations

import math

import mpmath as mp
import pytest

np = pytest.importorskip("numpy")

from connes_cvs.operator import prime_powers_up_to, psi_prime, psi_prime_deriv

C = 13
N = 48


def _capacity(q: int, c: int) -> int:
    return max(2, math.ceil(math.log(c) / math.log(q) - 1e-12))


def _path_law_norm(q: int, c: int, w_q: mp.mpf) -> float:
    if q >= c:
        return 0.0
    return 2.0 * math.cos(math.pi / (_capacity(q, c) + 1)) * float(w_q)


def _single_atom_block(q: int, c: int, n_half: int) -> "np.ndarray":
    L = mp.log(c)
    all_data, _ = prime_powers_up_to(c)
    data = [entry for entry in all_data if entry[0] == q]
    assert len(data) == 1
    psi = {n: psi_prime(mp.mpf(n), L, data) for n in range(0, n_half + 1)}
    dpsi = {n: psi_prime_deriv(mp.mpf(n), L, data) for n in range(0, n_half + 1)}
    for n in range(1, n_half + 1):
        psi[-n] = -psi[n]
        dpsi[-n] = dpsi[n]
    dim = 2 * n_half + 1
    block = np.zeros((dim, dim))
    for i in range(dim):
        m_idx = i - n_half
        for j in range(i, dim):
            n_idx = j - n_half
            if m_idx == n_idx:
                value = dpsi[n_idx]
            else:
                value = (psi[m_idx] - psi[n_idx]) / (m_idx - n_idx)
            block[i, j] = block[j, i] = float(value)
    return block


def _spectral_norm(matrix: "np.ndarray") -> float:
    eigenvalues = np.linalg.eigvalsh(matrix)
    return float(max(abs(eigenvalues[0]), abs(eigenvalues[-1])))


@pytest.fixture(scope="module", autouse=True)
def _dps():
    saved = mp.mp.dps
    mp.mp.dps = 30
    yield
    mp.mp.dps = saved


@pytest.fixture(scope="module")
def prime_data():
    data, _ = prime_powers_up_to(C)
    return data


def test_chebyshev_factors_are_exact_algebraic_constants(prime_data):
    """The capacity rungs hit 1, sqrt(2), golden ratio, sqrt(3) exactly."""
    weights = {q: w for q, _logq, w in prime_data}
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    assert _path_law_norm(2, C, weights[2]) == pytest.approx(golden * float(weights[2]), rel=1e-15)
    assert _path_law_norm(3, C, weights[3]) == pytest.approx(
        math.sqrt(2.0) * float(weights[3]), rel=1e-15
    )
    assert _path_law_norm(5, C, weights[5]) == pytest.approx(float(weights[5]), rel=1e-15)
    data_25, _ = prime_powers_up_to(25)
    w2 = next(w for q, _logq, w in data_25 if q == 2)
    assert _path_law_norm(2, 25, w2) == pytest.approx(math.sqrt(3.0) * float(w2), rel=1e-15)


def test_atom_norms_match_closed_form_and_interlace(prime_data):
    """Every interior atom at c=13 matches to <=1e-6 at N=48, from below."""
    for q, _logq, w_q in prime_data:
        theory = _path_law_norm(q, C, w_q)
        empirical = _spectral_norm(_single_atom_block(q, C, N))
        if theory == 0.0:
            continue
        assert empirical <= theory * (1.0 + 1e-9), f"interlacing violated for q={q}"
        assert empirical == pytest.approx(theory, rel=1e-6), f"norm mismatch for q={q}"


def test_boundary_atom_vanishes(prime_data):
    """The q = c atom has lag equal to the window; its block is identically zero."""
    assert _path_law_norm(C, C, prime_data[-1][2]) == 0.0
    assert _spectral_norm(_single_atom_block(C, C, 24)) <= 1e-12


def test_weyl_budget_bounds_full_prime_block(prime_data):
    """||W_p|| <= sum_q ||Delta O_q||, with the closed forms on the right."""
    full_block = sum(_single_atom_block(q, C, N) for q, _logq, _w in prime_data)
    budget = sum(_path_law_norm(q, C, w_q) for q, _logq, w_q in prime_data)
    assert _spectral_norm(full_block) <= budget
