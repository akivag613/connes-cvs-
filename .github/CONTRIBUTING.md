# Contributing to connes-cvs

Thanks for your interest. This project implements a specific mathematical construction (the Connes–van Suijlekom Galerkin matrix) and prioritises **correctness and reproducibility** over feature surface. The bar for any change that touches the numerical core is high.

## Development setup

```bash
git clone https://github.com/akivag613/connes-cvs-.git
cd connes-cvs-
python -m venv venv
source venv/bin/activate
pip install -e '.[all]'
```

The `all` extra installs `python-flint` (11× speedup on digamma), `gmpy2` (GMP-backed mpmath core), `numpy`, `scipy`, and the test tooling.

## Running tests

```bash
# Fast test suite (< 15 seconds)
pytest

# Full suite including slow regression tests (~1-2 minutes)
pytest -m slow --timeout=600

# With coverage
pytest --cov=connes_cvs --cov-report=term-missing
```

The slow regression tests in `tests/test_matrix_microopt_v0_2_0.py` validate against a pickled reference run at the published workload `c=13 N=100 T=800 dps=150`; they must pass before any merge.

## The bit-identicality contract

Any change to `connes_cvs/operator.py`, `connes_cvs/sweep.py`, or `connes_cvs/kernels.py` that could affect `λ_min` must:

1. Preserve `λ_min` to **≥ 18 leading decimal digits** against the pickled reference, enforced by `tests/test_matrix_microopt_v0_2_0.py::test_microopt_lambda_even_bit_identical`.
2. Include a before/after benchmark run via `_benchmarks/win1_pool_benchmark.py` (production-style 12-way multiprocessing, same code path as `run_sweep`).
3. Document the change in [CHANGELOG.md](CHANGELOG.md) under the next unreleased version, stating the speedup and the bit-identicality tolerance observed.

Performance optimizations that produce a mathematically different output (e.g. changing the quadrature rule, reordering summation steps that accumulate differently) must be behind a default-off flag, not replace the v0.1+ reference path.

## Style

- Python ≥ 3.9 syntax. `from __future__ import annotations` is used throughout; use PEP 585 built-in generics (`list[...]`, `dict[...]`, `tuple[...]`) rather than `typing.List`, etc.
- Line length: 100 characters (soft).
- Docstrings: NumPy style with `Parameters`, `Returns`, and optional `Notes` sections for public API.
- Private helpers are underscore-prefixed.

## Commit messages

- **Title:** imperative mood, ≤ 70 chars.
- **Body:** explain the **why**, reference measurements or test results, cite file paths where non-obvious.
- Keep commits focused — one logical change per commit.

## Opening a pull request

1. Open an issue first for anything non-trivial, to discuss the math and the approach.
2. Branch from `main`; keep PRs narrowly scoped.
3. The PR description should include: the motivation, the benchmark numbers (if perf-related), and a pointer to the test that enforces the claim.
4. All of `pytest` and `pytest -m slow` must pass.

## Reporting a numerical discrepancy

If you see `λ_min` or `|γ₁ err|` diverging from the [data/results_15pt_T800.json](data/results_15pt_T800.json) reference beyond the ~1.3× envelope (reflecting differences in `N`, precision, integration cutoff `T`, and normalization conventions — all three published computations use the same trigonometric basis), please open an issue with:

- exact `(c, N, T, dps)` used,
- platform (macOS/Linux, Python version, mpmath version, python-flint version/presence),
- output you got vs. expected.

Numerical reproducibility across platforms is something this package takes seriously; such reports are high priority.
