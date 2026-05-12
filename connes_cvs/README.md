# `connes_cvs/` — Python package source (published to PyPI)

**Package:** `connes-cvs` on PyPI — the first public implementation of the Connes–van Suijlekom Galerkin matrix for the Riemann Hypothesis.

**Current PyPI release:** `0.2.2` (2026-04-19, aligned with Paper Zenodo Version 2).

## Source files

| File | Content |
|---|---|
| `__init__.py` | Public API: `build_galerkin_matrix`, `compute_ground_state`, `extract_zeros`, and `__version__`. |
| `operator.py` | Core math: `prime_powers_up_to`, `psi_prime`, `psi_pole`, `psi_arch`, `_h_plus_cached` (v0.2.0 WIN 1 memoization), `_re_S_and_dS_fused` (v0.2.0 fused real kernel), `_compute_psi_pair`, matrix assembly, `mp.eigsy` wrapper, `F_even` test function, `extract_zeros`. |
| `kernels.py` | Low-level kernel helpers. |
| `sweep.py` | Multi-cutoff sweep runner with an `engine=` kwarg for feature flags. |
| `py.typed` | PEP 561 marker — this package is typed. |

## Version history

| Version | Date | Status | Headline |
|---|---|---|---|
| 0.1.0 | 2026-04-14 | Live on PyPI | Initial release. Paper Zenodo Version 1 lineage. |
| 0.2.0 | 2026-04-14 | Live on PyPI | `_h_plus_cached` memoization + fused real kernel; 2.06× psi-cache speedup; bit-identical to v0.1.0. |
| 0.2.1 | 2026-04-19 | **Yanked** | Internal `__version__` drift (stuck at `"0.2.0"` inside the 0.2.1 wheel). Superseded same day. |
| 0.2.2 | 2026-04-19 | Live on PyPI, current default | `__version__` fix; tests now use `importlib.metadata.version("connes-cvs")`; README aligned with Zenodo V2 DOI. |

## Install

```bash
pip install connes-cvs                # latest (currently 0.2.2)
pip install connes-cvs==0.2.0         # Paper Zenodo Version 1 lineage
pip install connes-cvs==0.2.2         # Paper Zenodo Version 2 lineage
```

## Build + publish

```bash
python -m build                       # creates dist/*.{whl,tar.gz}
twine upload dist/*                   # → PyPI
git tag v0.2.2 && git push --tags
```

## Cross-references (public)

- Top-level README: [`../README.md`](../README.md)
- Tests (bit-identity + micro-opt regressions): [`../tests/`](../tests/)
- Minimal runnable example: [`../examples/basic_compute.py`](../examples/basic_compute.py)
- Benchmarks: [`../benchmarks/`](../benchmarks/)
- Paper reproducibility data: [`../data/results_15pt_T800.json`](../data/results_15pt_T800.json)
- Changelog: [`../CHANGELOG.md`](../CHANGELOG.md)

## Discipline

- **Bit-identity contract.** Every change MUST preserve `c=13 N=100 T=800 dps=150` output to ≥ 22 matching digits on `lambda_even` (Paper reference value `2.86545361493028029516151514986747977533...`). This is enforced by `tests/test_c13_regression.py`.
- **Version-number contract.** Never bump `__version__` in `__init__.py` without simultaneously updating `pyproject.toml`, `CHANGELOG.md`, and creating a matching git tag (v0.2.1 was yanked for this).
- **Backward-compatibility contract.** Do not delete v0.1.0 or v0.2.0 from PyPI — paper reproducibility depends on them remaining installable.
