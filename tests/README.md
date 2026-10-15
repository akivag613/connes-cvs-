# `tests/` - Regression test suite for the `connes-cvs` package

## What's here

| File | Purpose |
|---|---|
| `reference_values.json` | The committed c = 13 reference values the gate compares against: the published T = 800 reference cell (39-digit lambda mantissa from the exact-workload run recorded in `benchmarks/AB_VERIFIED_2026-04-14.md`, whose first 25 digits also appear in `data/results_15pt_T800.json`), the T = 400 extended validation cell of `examples/basic_compute.py --extended` (25-digit lambda, 20-digit gamma_1 error), and the 80-digit A/B benchmark cell. Sources are documented per cell inside the file. |
| `test_c13_regression.py` | Public-API and reference gate. Fast tests cover input validation, the float64-`L` warning and `c=` path, the finite-matrix Arb helper, the deliberate absence of unsupported precision-recommender APIs, canonical digit comparison, and karl-keysingularity's credited Windows artifact. Slow tests recompute the A/B cell (all 80 stored printed digits required), extended validation cell (>=20 lambda digits, >=18 gamma-error digits), and published T=800 cell (fatal >=22-digit lambda gate plus gamma-error comparison). |
| `test_operator_hardening.py` | Direct checks of the fast prime-power enumerator, stable kernel, positive/negative index parity, and upper-triangle assembly against the legacy full assembly. These are independent of the runner-vs-classic comparison that shares the optimized operator. |
| `test_runner_identity.py` | Exact small-cell scope: c=13, N=8, T=60, dps=30 through the runner and classic path must agree in every raw mpmath matrix entry, selected eigenvalue and eigenvector component; a second cell compares pooled and serial runner output. This does not assert identity at untested workloads. |
| `test_runner_checkpoints.py` | Checkpoint, artifact and input-contract tests for `connes_cvs.runner` and `connes_cvs.sweep`: identity between a clean run and one resumed from a checkpoint; rejection of checkpoints whose checksum, config, backend or JSON canonicality does not match; artifact provenance fields and the path-collision policy; `CellConfig` immutability and input validation. |
| `test_validation.py` | Adversarial tests for the outward Arb finite-matrix residual bound, exact dyadic serialization, shape/symmetry checks, canonical hashes and extreme exponents. The module requires python-flint and is skipped when it is unavailable; install the `dev` extra to run it. |
| `test_matrix_microopt_v0_2_0.py` | Historical v0.2.0 regression. Its unmarked source-form check runs on a clean checkout. Pickle-dependent slow legs use a local research reference that is not distributed and skip when absent; they are not the release's portable numerical gate. |
| `test_release_artifacts.py` | Fast source checks: public files in the checked directories must appear in the distribution manifest, and on Python 3.11+ the package configuration must agree with that manifest. These tests do not build an archive. `tools/verify_release.py` separately verifies the actual sdist and wheel after a build. |

The source inventory checks immediate files in the package, tests, examples,
benchmarks, tools and data directories, including the named public data
subdirectories. It is not a recursive audit of every repository path. In a Git
checkout it includes tracked files and nonignored new files. Without Git
metadata, run it from a clean source export. The release manifest remains the
explicit list of distribution contents.

## Running

```bash
pip install -e '.[dev]'

# Fast suite (well under a minute). The `addopts = -m 'not slow'` in
# pyproject.toml deselects the slow gate here.
pytest -v

# Slow exact-reference gate ONLY. Reproduce the recorded arithmetic stack;
# newer dependency versions are compatibility targets, not bit-identity
# substitutes for this provenance-labelled reference.
pip install 'mpmath==1.4.1' 'python-flint==0.8.0' 'gmpy2==2.3.0' pytest-timeout
pytest -v -s -m slow --timeout=1800  # -s keeps the per-point heartbeat visible
```

## Discipline

- Every PyPI release MUST pass `pytest` and `pytest -m slow` before upload.
- A failure of the c = 13 gate is a **blocking release issue** - do not publish.
- Reproduction of the committed decimal contract is the gate: if dependency internals change and the c=13 digits shift, investigate rather than silently accepting drift. The same-precision A/B test requires all 80 stored printed digits; raw-tuple identity is asserted only in the explicitly documented small-cell tests.
- The committed references in `reference_values.json` are frozen history. Extending them (more digits, new cells) is fine; changing recorded digits requires a documented erratum.
