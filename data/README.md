# `data/` — Canonical numerical data (public)

**Contents.** `results_15pt_T800.json` — the 15-cutoff Paper summary JSON containing `lambda_even` and `gamma_1_abs_error` for each of `c ∈ {13, 14, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67}` at `N = 100`, `T = 800`, `dps = 150` or `200`.

**Purpose.** This is the public-facing numerical summary of the paper's 15-cutoff sweep. Linked from [`../README.md`](../README.md) and used by [`../examples/basic_compute.py`](../examples/basic_compute.py) for cross-validation.

## Schema

```json
{
  "cutoffs": [13, 14, 17, ..., 67],
  "lambda_even": {"13": "...", "14": "...", ...},
  "gamma_1_abs_error": {"13": "...", ...},
  "timings": {...},
  "parameters": {"N": 100, "T": 800, "dps_min": 150, "dps_max": 200}
}
```

Values are stored as full-precision decimal strings (not floats); they span 113 orders of magnitude.

## Reproducing this dataset

```python
from connes_cvs.sweep import run_sweep

CUTOFFS = [13, 14, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]

results = run_sweep(cutoffs=CUTOFFS, N=100, T=800, dps=150, workers=12)
```

`dps = 200` is used for `c ≥ 41` to stay clear of the precision floor; see [`../README.md`](../README.md) for the full reproducibility recipe.

## Discipline

- This file is **public** (git-tracked, part of the PyPI distribution). Never include in-progress or revision-pending findings here.
- The file is frozen with the Zenodo Version 2 deposit (concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514)). If a correction is needed, deposit a new Zenodo version and update this file in lockstep.
