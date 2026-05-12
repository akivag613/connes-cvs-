# `examples/` — Minimal examples for the `connes-cvs` package

## What's here

| File | Purpose |
|---|---|
| `basic_compute.py` | Smallest-possible working example. Builds the CvS Galerkin matrix at `c = 13`, `N = 60`, `T = 400`, `dps = 80`; computes its ground-state eigenvalue and the first Riemann zero via `extract_zeros`. Runs in under 2 minutes on a modern laptop. |

## Running

```bash
pip install connes-cvs
python examples/basic_compute.py
```

Expected output at `c = 13`, modest precision:

- `lambda_min` at magnitude ~10⁻³⁶
- `gamma_1` extraction matching `14.1347...` to roughly 30 digits

For the full published precision (matching the paper's 113-OOM convergence), use `c = 13`, `N = 100`, `T = 800`, `dps = 150` and verify against [`../data/results_15pt_T800.json`](../data/results_15pt_T800.json) — the result matches the stored `lambda_even` to 22+ digits.

## Cross-references (public)

- Top-level README + headline result: [`../README.md`](../README.md)
- Package source: [`../connes_cvs/`](../connes_cvs/)
- Canonical paper data: [`../data/results_15pt_T800.json`](../data/results_15pt_T800.json)
- Tests: [`../tests/`](../tests/)
- Benchmarks: [`../benchmarks/`](../benchmarks/)
