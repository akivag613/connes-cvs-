"""
connes-cvs: First public implementation of the Connes-van Suijlekom Galerkin matrix.

This package constructs and diagonalizes the truncated Weil operator Q(c)
from Connes & van Suijlekom (arXiv:2511.23257), whose ground-state eigenvalue
measures proximity to the Riemann Hypothesis.

Basic usage::

    from connes_cvs import build_galerkin_matrix, compute_ground_state

    Q = build_galerkin_matrix(c=13, N=100, T=400, dps=80)
    lam_min, eigvec = compute_ground_state(Q)
"""

__version__ = "0.2.0"

from connes_cvs.operator import build_galerkin_matrix, compute_ground_state, extract_zeros

__all__ = [
    "__version__",
    "build_galerkin_matrix",
    "compute_ground_state",
    "extract_zeros",
]
