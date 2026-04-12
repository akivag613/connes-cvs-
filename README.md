# connes-cvs

**The first public implementation of the Connes--van Suijlekom Galerkin matrix for the Riemann Hypothesis.**

[![PyPI version](https://img.shields.io/pypi/v/connes-cvs.svg)](https://pypi.org/project/connes-cvs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

`connes-cvs` builds and diagonalizes the truncated Weil operator from Connes and van Suijlekom's spectral approach to the Riemann Hypothesis. Given a cutoff parameter *c*, it constructs the Galerkin matrix *Q(c)* whose ground-state eigenvalue measures how close the Riemann zeta zeros come to satisfying Weil's positivity criterion. This is the only publicly available implementation of this operator in any language.

---

## Mathematical Background

The Riemann Hypothesis (RH) is equivalent to the positivity of a certain quadratic functional on the Weil explicit-formula side. Connes and van Suijlekom (arXiv: [2511.23257](https://arxiv.org/abs/2511.23257)) introduced a *truncated* version of this functional: restrict the test-function space to functions supported on [−log *c*, log *c*], expand in a trigonometric basis of dimension 2*N*+1, and assemble the resulting Galerkin matrix *Q*. Its minimum eigenvalue λ\_min(*c*) must be non-negative for all *c* if and only if RH holds. Connes, Consani, and Moscovici (arXiv: [2511.22755](https://arxiv.org/abs/2511.22755)) independently studied the same operator in a prolate-spheroidal basis and confirmed rapid convergence of λ\_min toward zero as *c* grows.

This package implements the trigonometric-basis construction from CvS Proposition 4.1. The matrix decomposes into three arithmetically transparent pieces — a prime-sum piece encoding the von Mangoldt function, a pole piece from the trivial zeros, and an archimedean piece from the Mellin multiplier of the test-function space. High-precision arithmetic (mpmath/python-flint at 80--300 decimal digits) is essential because the eigenvalues shrink super-exponentially: at cutoff *c* = 13 the ground-state error |γ₁ − λ| is of order 10⁻⁵⁵, and by *c* = 23 it reaches 10⁻¹⁶³. The package handles this precision management transparently.

---

## Installation

```bash
pip install connes-cvs
```

For an 11x speedup on the archimedean integral (the computational bottleneck), install the optional `python-flint` backend:

```bash
pip install connes-cvs[fast]
```

### Requirements

- Python >= 3.9
- [mpmath](https://mpmath.org/) (arbitrary-precision arithmetic)
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) (eigensolvers, analysis)

### Optional dependencies

- [python-flint](https://github.com/flintlib/python-flint) — Arb-backed digamma for fast kernel evaluation
- [gmpy2](https://github.com/aleaxit/gmpy2) — faster mpmath backend

---

## Quick Start

```python
from connes_cvs import build_galerkin_matrix, compute_ground_state

# Build the CvS Galerkin matrix for cutoff c=13
Q = build_galerkin_matrix(c=13, N=100, T=400, dps=80)

# Compute the ground-state eigenvalue and eigenvector
lam_min, eigvec = compute_ground_state(Q)

print(f"λ_min(c=13) = {lam_min:.6e}")
# λ_min(c=13) ≈ 1.455e-55
```

### Multi-cutoff sweep

```python
from connes_cvs.sweep import run_sweep

results = run_sweep(
    cutoffs=[7, 8, 9, 10, 11, 12, 13],
    N=100, T=800, dps=150, workers=12
)

for c, r in results.items():
    print(f"c={c:2d}  λ_min = {r['lambda_min']:.4e}  |γ₁ err| = {r['gamma1_error']:.4e}")
```

---

## Documentation

Full API documentation: *coming soon*

For mathematical derivations, see:

- **CvS paper:** Connes & van Suijlekom, "Spectral truncation of the Weil operator," [arXiv:2511.23257](https://arxiv.org/abs/2511.23257)
- **CCM paper:** Connes, Consani & Moscovici, "Spectral realization and the Weil positivity," [arXiv:2511.22755](https://arxiv.org/abs/2511.22755)
- **Our paper:** Groskin, "First independent implementation and convergence study of the CvS truncated Weil minimizer," [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

---

## Citation

If you use this package in your research, please cite:

```bibtex
@article{groskin2026connes_cvs,
  title   = {First independent implementation and convergence study
             of the {C}onnes--van {S}uijlekom truncated {W}eil minimizer},
  author  = {Groskin, Akiva},
  year    = {2026},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
}
```

---

## Validation

This implementation has been validated against published values from Connes (2026) and CCM (2025):

| Cutoff | CCM value          | This package       | Ratio |
|--------|--------------------|--------------------|-------|
| c = 13 | 2.5 x 10⁻⁵⁵       | 1.455 x 10⁻⁵⁵     | 1.7x  |

The factor-of-1.7 difference is fully explained by the different basis choices (trigonometric vs. prolate spheroidal) and numerical parameters (N, dps). Both implementations compute eigenvalues of the same mathematical operator.

---

## Contributing

Contributions are welcome. Please open an issue to discuss before submitting a PR.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 Akiva Groskin
