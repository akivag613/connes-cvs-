"""
Multi-cutoff sweep runner with multiprocessing parallelism.

Runs :func:`connes_cvs.operator.build_galerkin_matrix` and
:func:`connes_cvs.operator.compute_ground_state` across a range of
cutoff values, distributing the expensive psi-cache computation across
CPU cores.

The parallelism strategy mirrors the production scripts: each worker
computes a single (psi, psi') pair for one basis index, and the matrix
assembly + eigensolver run in the main process after the cache is built.

References
----------
- Connes & van Suijlekom, arXiv:2511.23257
- Connes, Consani & Moscovici, arXiv:2511.22755
"""

from __future__ import annotations

import multiprocessing as mp_pool
import time
from typing import Any

import mpmath as mp

from connes_cvs.operator import (
    HAS_FLINT,
    prime_powers_up_to,
    psi_prime,
    psi_prime_deriv,
    psi_pole,
    psi_pole_deriv,
    psi_arch,
    psi_arch_deriv,
    compute_ground_state,
    extract_zeros,
)

if HAS_FLINT:
    from flint import ctx as flint_ctx


# ============================================================
# Worker initializer and task function for multiprocessing
# ============================================================

# Module-level state for worker processes (populated by init_worker)
_worker_state: dict[str, Any] = {}


def _init_worker(cutoff: int, dps: int, T: int) -> None:
    """
    Initializer for each worker process in the multiprocessing pool.

    Sets mpmath precision, flint precision, and precomputes shared data
    (L, prime_data) that every worker needs.
    """
    mp.mp.dps = dps
    if HAS_FLINT:
        flint_ctx.prec = int(dps * 3.5)

    L = mp.log(cutoff)
    prime_data, _ = prime_powers_up_to(cutoff)

    _worker_state["dps"] = dps
    _worker_state["T"] = T
    _worker_state["L"] = L
    _worker_state["prime_data"] = prime_data


def _compute_psi_pair_worker(n_idx: int) -> tuple[int, str, str]:
    """
    Worker function: compute psi(n_idx) and psi'(n_idx).

    Returns string representations to safely pass between processes
    (mpmath objects are not pickle-safe at high precision).
    """
    L = _worker_state["L"]
    T = _worker_state["T"]
    dps = _worker_state["dps"]
    prime_data = _worker_state["prime_data"]

    x = mp.mpf(n_idx)
    psi = psi_prime(x, L, prime_data) + psi_pole(x, L) + psi_arch(x, L, T, dps)
    psi_d = psi_prime_deriv(x, L, prime_data) + psi_pole_deriv(x, L) + psi_arch_deriv(x, L, T, dps)
    return (n_idx, mp.nstr(psi, dps + 5), mp.nstr(psi_d, dps + 5))


def _run_single_cutoff(
    c: int,
    N: int,
    T: int,
    dps: int,
    n_workers: int,
) -> dict[str, Any]:
    """
    Run the full pipeline for a single cutoff value using multiprocessing.

    1. Parallel psi-cache computation (workers compute individual psi pairs).
    2. Matrix assembly in main process.
    3. Even-sector diagonalization.
    4. Zero extraction.

    Returns a result dict with eigenvalue, errors, timing, etc.
    """
    mp.mp.dps = dps
    if HAS_FLINT:
        flint_ctx.prec = int(dps * 3.5)

    t_start = time.time()
    L = mp.log(c)

    # Step 1: parallel psi-cache
    n_indices = list(range(-N, N + 1))
    t0 = time.time()
    with mp_pool.Pool(n_workers, initializer=_init_worker, initargs=(c, dps, T)) as pool:
        cache_results = pool.map(_compute_psi_pair_worker, n_indices)
    t_cache = time.time() - t0

    psi_vals = {}
    psi_deriv_vals = {}
    for (n_idx, psi_str, psi_d_str) in cache_results:
        psi_vals[n_idx] = mp.mpf(psi_str)
        psi_deriv_vals[n_idx] = mp.mpf(psi_d_str)

    # Step 2: assemble the Galerkin matrix
    t_mat_start = time.time()
    DIM = 2 * N + 1
    Q = mp.matrix(DIM, DIM)
    for i in range(DIM):
        m_idx = i - N
        for j in range(DIM):
            n_idx = j - N
            if m_idx == n_idx:
                Q[i, j] = psi_deriv_vals[n_idx]
            else:
                Q[i, j] = (psi_vals[m_idx] - psi_vals[n_idx]) / mp.mpf(m_idx - n_idx)
    # Symmetrize
    for i in range(DIM):
        for j in range(i + 1, DIM):
            avg = (Q[i, j] + Q[j, i]) / 2
            Q[i, j] = avg
            Q[j, i] = avg
    t_mat = time.time() - t_mat_start

    # Step 3: even-sector diagonalization
    t_diag_start = time.time()
    lambda_min, v_full = compute_ground_state(Q)
    t_diag = time.time() - t_diag_start

    # Step 4: zero extraction
    t_zeros_start = time.time()
    zeros_results = extract_zeros(v_full, float(L), n_zeros=10, dps=dps)
    t_zeros = time.time() - t_zeros_start

    t_total = time.time() - t_start

    return {
        "cutoff": c,
        "L": L,
        "N": N,
        "T": T,
        "dps": dps,
        "lambda_min": lambda_min,
        "eigvec": v_full,
        "zeros": zeros_results,
        "gamma1_error": zeros_results[0]["error"] if zeros_results and zeros_results[0]["error"] is not None else None,
        "wall_time": t_total,
        "timing": {
            "cache_sec": t_cache,
            "matrix_sec": t_mat,
            "diag_sec": t_diag,
            "zeros_sec": t_zeros,
            "total_sec": t_total,
        },
    }


# ============================================================
# Public API
# ============================================================

def run_sweep(
    cutoffs: list[int],
    N: int = 100,
    T: int = 800,
    dps: int = 150,
    workers: int | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Run a multi-cutoff sweep of the CvS operator.

    For each cutoff c in ``cutoffs``, builds the Galerkin matrix Q(c),
    computes the ground-state eigenvalue, extracts the first 10 zeta
    zeros from the eigenvector, and records diagnostics.

    Each cutoff is processed sequentially (the parallelism is WITHIN
    each cutoff — the expensive psi-cache computation is distributed
    across ``workers`` CPU cores via multiprocessing.Pool).

    Parameters
    ----------
    cutoffs : list of int
        Cutoff values to sweep. Each must be >= 2.
    N : int, optional
        Basis half-size. Default: 100.
    T : int, optional
        Archimedean truncation parameter. Default: 800.
    dps : int, optional
        Decimal digits of precision. Default: 150.
    workers : int or None, optional
        Number of parallel processes for the psi-cache computation.
        If None, uses ``multiprocessing.cpu_count()``.

    Returns
    -------
    results : dict
        Mapping from cutoff c to a dict containing:

        - ``'lambda_min'`` (mpmath.mpf): Ground-state eigenvalue.
        - ``'gamma1_error'`` (mpmath.mpf or None): |gamma_1 - detected_zero|.
        - ``'eigvec'`` (mpmath.matrix): Ground-state eigenvector.
        - ``'zeros'`` (list of dict): Full zero extraction results.
        - ``'wall_time'`` (float): Computation time in seconds.
        - ``'timing'`` (dict): Breakdown of time per phase.

    Examples
    --------
    >>> results = run_sweep([7, 8, 9], N=60, T=400, dps=80, workers=4)
    >>> for c, r in sorted(results.items()):
    ...     print(f"c={c:2d}  lambda_min={r['lambda_min']:.4e}")
    """
    if workers is None:
        workers = mp_pool.cpu_count()

    mp.mp.dps = dps

    results = {}
    for c in cutoffs:
        result = _run_single_cutoff(c, N, T, dps, workers)
        results[c] = result

    return results
