# `examples/` - Minimal examples for the `connes-cvs` package

## What's here

| File | Purpose |
|---|---|
| `basic_compute.py` | Defaults to the seconds-level `c=13, N=8, T=60, dps=30` API smoke cell and one seeded finite-test root. `--extended` runs the `N=100, T=400, dps=80` validation cell through the process-based runner, with progress and the default eight-worker cap, then locates three roots. Heavy work is protected by `main()`, so importing the module is safe. |
| `c100_aitken_check.py` | Loads the published `c = 100` N-sweep data from [`../data/c100/`](../data/c100/) and reproduces the Aitken extrapolation check in under a second. |
| `path_law_weyl_overlay.py` | Runs the README smoke cell verbatim (`build_galerkin_matrix` / `compute_ground_state` / `extract_zeros` at `c=13, N=8, T=60, dps=30`), then layers the Single-Atom Path Law on that very matrix: extracts each exact additive atom block `B_q` of `Q`, checks the closed-form norm interlaces from above, and runs the Weyl experiment `Q' = Q - B_q` — removing `q=11` flips `lambda_min` to `-3.45e-2` with every eigenvalue shift inside the predicted budget, while removing the boundary atom `q=13` (rank-one heuristic weight `0.7114`, Path Law norm exactly `0`) changes nothing. Runs in seconds on the base install (mpmath only). |
| `single_atom_path_law.py` | Demonstrates the Single-Atom Path Law (White 2026): the exact closed-form operator norm `2 cos(pi/(m+1)) * Lambda(q)/sqrt(q)`, `m = ceil(L/log q)`, of each per-prime-power Loewner block of the prime piece, checked against blocks assembled from this package's own `psi_prime`/`psi_prime_deriv` to about `1e-15` relative error. Also prints the dimension-independent Weyl stability budget for the prime piece and an exact account of the non-prime step `c = 13 -> 14` (Groskin arXiv:2605.20224 §8.8). Runs in about a second; requires numpy (the `science` or `all` extras). Enforced by [`../tests/test_single_atom_path_law.py`](../tests/test_single_atom_path_law.py). |
| `make_fig9_c100_aitken.py`, `make_fig10_c100_gamma_digits.py` | Generators for the two `c = 100` figures of the accompanying paper's reproducibility package (Aitken extrapolation; canonical `floor(-log10(error))` counts for `gamma_1..gamma_10`). They require Matplotlib in addition to the package dependencies. Papers 2 and 3 keep their own figure generators inside their [paper folders on GitHub](https://github.com/akivag613/connes-cvs-/tree/main/papers). |

## Running

```bash
pip install connes-cvs
python examples/basic_compute.py

# Multi-minute validation cell, with progress and process isolation:
python examples/basic_compute.py --extended

# Figure generators additionally require matplotlib:
python -m pip install matplotlib
```

Expected output for the default `c = 13`, `N = 8`, `T = 60`, `dps = 30` smoke cell:

- `lambda_min` approximately `4.43043e-23`
- `gamma_1` detected as `14.13472514173469...` with `|gamma_1 error|` approximately `2.52738e-17`

Measured release-environment runtime was about 1.8 seconds with python-flint 0.8.0 and 4.8 seconds through the mpmath fallback; runtime varies by versions and hardware. This is only an API smoke test. The `--extended` mode is the `T=400` validation cell whose committed slow gate requires at least 20 lambda digits and 18 gamma-error digits. The paper's c=13 reference datum (`|gamma_1 error|` approximately `2.005e-55`) is the separate `N=100`, `T=800`, `dps=150` cell.

## Cross-references (public)

- Top-level README + headline result: [`../README.md`](../README.md)
- Package source: [`../connes_cvs/`](../connes_cvs/)
- Canonical paper data: [`../data/results_15pt_T800.json`](../data/results_15pt_T800.json)
- Tests: [`../tests/`](../tests/)
- Benchmarks: [`../benchmarks/`](../benchmarks/)
