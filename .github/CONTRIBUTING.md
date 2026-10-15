# Contributing to connes-cvs

Thanks for your interest. This project implements a specific mathematical construction (the Connes-van Suijlekom Galerkin matrix) and prioritises **correctness and reproducibility** over feature surface. The bar for any change that touches the numerical core is high.

## Development setup

```bash
git clone https://github.com/akivag613/connes-cvs-.git
cd connes-cvs-
python -m venv venv
source venv/bin/activate
pip install -e '.[all,dev]'
```

The `all` extra installs the arithmetic, science, plotting, and test
dependencies. The smaller `dev` set contains pytest, pytest-timeout,
pytest-cov, python-flint, and gmpy2. python-flint supplies a compiled
arbitrary-precision digamma backend, but its speedup is workload-, version-,
precision-, and hardware-dependent.

## Running tests

```bash
# Fast test suite (well under a minute). pyproject.toml sets
# addopts = -m 'not slow', so this deselects the regression gate.
pytest

# Slow regression gate (run in addition to plain pytest, not instead of
# it: the CLI -m overrides the addopts selection, so this command runs
# the slow tests and nothing else). It computes real c=13 cells and
# compares against tests/reference_values.json, with per-leg thresholds:
# the A/B cell (c=13, N=80, T=400, dps=80) is a same-precision reference
# and must match in full, all 80 stored digits; the extended cell (c=13,
# N=100, T=400, dps=80) gates at >= 20 digits on lambda and >= 18 on the
# gamma_1 error.
pytest -m slow --timeout=1800

# With coverage
pytest --cov=connes_cvs --cov-report=term-missing
```

The slow regression tests in `tests/test_c13_regression.py` compute c=13 cells end-to-end and enforce the reference contract against the committed values; they must pass before any merge. (`tests/test_matrix_microopt_v0_2_0.py` keeps one unmarked source-form check that always runs, plus slow legs that validate against a pickled psi-cache reference when that pickle is present locally and skip cleanly otherwise.)

The numerical core mutates process-global mpmath/Flint precision contexts and caches. Do not run different-precision cells concurrently in Python threads; use the process-based runner/sweep. Resume only trusted local checkpoint files: their hashes detect corruption but are not authentication, and their locks only enforce a single writer.

## Numerical regression contract

Any change to `connes_cvs/operator.py`, `connes_cvs/sweep.py`, or `connes_cvs/kernels.py` that could affect `λ_min` must:

1. Reproduce the committed c=13 references in `tests/reference_values.json`: **all 80 stored printed digits** on the same-precision A/B cell, and **>=20** lambda plus **>=18** gamma-error digits on the cross-precision extended validation cell.
2. Preserve the exact small-cell equality checks in `tests/test_runner_identity.py` and the independent parity/full-assembly checks in `tests/test_operator_hardening.py`. Do not generalize those scopes into a blanket raw-value claim.
3. Include a matched before/after benchmark for any speed claim: same checkout except for the proposed change, same workload, precision, backend, worker count, output phases, hardware, and timing boundary. `benchmarks/win1_pool_benchmark.py` is a current-path timer, not by itself a historical v0.1 baseline.
4. Document the change in [CHANGELOG.md](../CHANGELOG.md) under an unreleased heading (or a dated documentation heading for documentation-only changes), with the exact measured scope and numerical tolerance.

Performance optimizations that produce a mathematically different output (e.g. changing the quadrature rule, reordering summation steps that accumulate differently) must be behind a default-off flag, not replace the v0.1+ reference path.

## Checking distribution archives

Work from a clean public source tree and use a new, empty output directory.
The source-distribution allowlist defines the package contents. Fast tests
check source coverage and configuration; they do not inspect a built archive.

After the fast suite, pinned reference suite and type checks pass, use the same
build tools and archive checks as the release workflow:

```bash
python -m pip install 'build==1.4.3' 'twine==7.0.0'
python -m build --sdist --wheel --outdir release-dist
python tools/verify_release.py --dist release-dist
python -m twine check --strict release-dist/*
```

The build backend is pinned in `pyproject.toml`. Test each archive in a fresh
virtual environment at the declared mpmath floor, from outside the source tree,
so a local import cannot hide an incomplete distribution. The workflow records
the complete clean-install procedure. Any failed gate stops publication.
These checks do not upload a release; a new release also needs its own version
and synchronized metadata.

## Style

- Python ≥ 3.10 syntax. `from __future__ import annotations` is used throughout; use PEP 585 built-in generics (`list[...]`, `dict[...]`, `tuple[...]`) rather than `typing.List`, etc.
- Line length: 100 characters (soft).
- Docstrings: NumPy style with `Parameters`, `Returns`, and optional `Notes` sections for public API.
- Private helpers are underscore-prefixed.

## Commit messages

- **Title:** imperative mood, ≤ 70 chars.
- **Body:** explain the **why**, reference measurements or test results, cite file paths where non-obvious.
- Keep commits focused: one logical change per commit.

## Opening a pull request

1. Open an issue first for anything non-trivial, to discuss the math and the approach.
2. Branch from `main`; keep PRs narrowly scoped.
3. The PR description should include: the motivation, the benchmark numbers (if perf-related), and a pointer to the test that enforces the claim.
4. All of `pytest` and `pytest -m slow` must pass.

## Reporting a numerical discrepancy

If a like-for-like cell differs from the [published reference data](../data/results_15pt_T800.json), please open an issue with:

- exact `(c, N, T, dps)` used,
- platform (Windows/macOS/Linux, Python version, mpmath version, python-flint version/presence),
- output you got vs. expected.

Numerical reproducibility across platforms is something this package takes seriously; such reports are high priority.
