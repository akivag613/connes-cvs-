# `tests/` — Regression test suite for the `connes-cvs` package

## What's here

Bit-identity and micro-optimization regression tests for the public `connes-cvs` PyPI package.

| File | Purpose |
|---|---|
| `test_c13_regression.py` | Canonical bit-identity regression at `c = 13`, `N = 100`, `T = 800`, `dps = 150`. Verifies that the package reproduces the paper's `lambda_even` reference value `2.86545361493028029516151514986747977533...` to ≥ 22 matching digits. `test_package_imports` uses `importlib.metadata.version("connes-cvs")` as the version-identity check (no manual per-release update needed). |
| `test_matrix_microopt_v0_2_0.py` | Verifies that v0.2.0's `_h_plus_cached` memoization + fused real kernel produces bit-identical output to v0.1.0 at `c = 13`, `N = 80`, `T = 400`, `dps = 80` across a 500-digit mantissa. |

## Running

```bash
pip install -e .
pytest tests/ -v
```

Expected: both tests pass. Wall-clock < 5 min total on a 12-core modern machine.

## Cross-references (public)

- Package source: [`../connes_cvs/`](../connes_cvs/)
- Paper reference data the c=13 test compares against: [`../data/results_15pt_T800.json`](../data/results_15pt_T800.json)
- CI configuration: [`../.github/workflows/tests.yml`](../.github/workflows/tests.yml)

## Discipline

- Every PyPI release MUST pass these tests before upload.
- A failure at `c = 13` is a **blocking release issue** — do not publish.
- Bit-identity is the gate: if mpmath internals change (e.g., `mpmath 1.4.x → 1.5.x`) and the `c = 13` `lambda_even` mantissa shifts, investigate; do not silently accept drift.
