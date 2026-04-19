# Changelog

All notable changes to `connes-cvs` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [paper — Zenodo Version 2] — 2026-04-19

Paper revision; no code changes. Published on Zenodo as Version 2 with version-specific DOI [10.5281/zenodo.19655106](https://doi.org/10.5281/zenodo.19655106) (concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) now resolves to Version 2). An erratum accompanies the revised PDF as a supplementary file on the Zenodo record.

### Corrected

- **Basis attribution throughout paper body** (§1, §2.2, §4.2 Table 1, §4.3, §5.1.1, §8.5, §9): the CCM 2025 and Connes 2026 Galerkin computations use the same **trigonometric basis** as this work, not prolate-spheroidal — as is evident from CCM Lemma 5.1 (matrix entries defined via the kernel $\sin(2\pi n y/L)$) and Connes 2026 §6 (referring to the "trigonometric orthonormal basis"). The prior prolate-basis attribution was an error of our reading; the correct attribution has always been in the published sources. Prolate wave functions appear in a distinct role in the program (approximation construction for the limit $k_\lambda$ per Connes 2026 §6.3–§6.4).
- **§5.1.1 arithmetic typo**: "factor of approximately 30" → "factor of approximately 3" at the $c=14$ CCM cross-validation paragraph. Actual ratio: $1.07 \times 10^{-60} / 3.541 \times 10^{-61} = 3.02$. Identified during 2026-04-19c self-audit.
- **§2.3 internal consistency**: "in different bases" → "via unitarily equivalent matrix representations."
- **§1 introduction**: "a single numerical datum" → "numerical data for the first fifty zeros at $c=13$."
- **README**: Validation section (lines 48, 189–193, 260) updated on 2026-04-19 to remove basis-misattribution wording.

### Unchanged

- All numerical data (15-point sweep, Table 3 verified byte-identical against ancillary `results_15pt_T800.json`).
- All structural observations (Sobolev scaling, multi-zero universality, eigenvector near-invariance, bulk Poisson statistics, spectral-gap $\lambda_2/\lambda_1 \sim 10^{7-8}$ verified against raw pickle).
- All theorems and derivations (§2.3 unitary-equivalence derivation mathematically correct).
- The `connes_cvs` v0.2.0 PyPI package is unaffected.

### Added

- Comprehensive erratum document (`papers/submission/erratum_2026-04-19b.md`) covering all three classes of correction (basis-attribution, arithmetic, internal-consistency), to be uploaded as supplementary on the new Zenodo version.
- Acknowledgment of A. Connes in §12 of the revised paper.
- Version-history entry in paper front matter listing Version 1 (2026-04-13) and Version 2 (2026-04-19).

### Public locations

- **Zenodo** (primary public venue): concept DOI `10.5281/zenodo.19546514` (always resolves to the latest version); Version 1 DOI `10.5281/zenodo.19546515` (2026-04-13); a new version-specific DOI is minted by Zenodo on Version 2 upload.
- **GitHub** (`github.com/akivag613/connes-cvs-`): `paper-v2` git tag on Version 2 upload. HAL submission has been dropped; Zenodo + GitHub are the canonical public venues.

## [0.2.0] — 2026-04-14

Performance release: archimedean integral phase is **2× faster** at production scale, with output bit-identical to v0.1.0.

### Performance

- **WIN 1 — h_plus memoization + fused real kernel.** `h_plus(τ) = Re ψ(¼ + iτ/2) − log π` is mathematically even in τ. The new code memoizes `h_plus` keyed on `|τ|` and reuses it across `psi_arch` and `psi_arch_deriv` (which share quadrature nodes). A fused real-arithmetic kernel `_re_S_and_dS_fused` computes both `Re S_hat_x` and `Re dS_hat_x_dx` in one pass, sharing all sub-expressions; a pair-cache hands the result from the first quadrature pass to the second. Net effect at production scale (c=13, N=80, T=400, dps=80, 12-way Pool):

  | Phase | v0.1.0 | v0.2.0 | Speedup |
  |---|---|---|---|
  | psi cache | 57.55 s | 27.94 s | **2.06×** |
  | total wall | 64.94 s | 35.40 s | **1.83×** |

  Saves ~40 minutes per full 15-cutoff production sweep at dps=150.

- **WIN 3 — drop redundant `mp.mpf(int)` conversion in Galerkin Q-matrix assembly** (commit 82f0953). Bit-identical micro-opt; visible only at very large N.

### Correctness

- λ_min reproduces v0.1.0 to all 80 decimal digits printed at the A/B test workload (c=13, N=80, T=400, dps=80).
- At the **published reference workload** (c=13, N=100, T=800, dps=150), the v0.2.0 code computes `λ_min = 2.86545361493028029516…e-59`, exactly matching the paper Table 18 published value of `2.865 × 10⁻⁵⁹` to all reported precision. End-to-end wall time at this workload: 127.3 s (vs. historical baseline 214.8 s = **1.69× faster** on the paper-canonical run).
- The slow regression test `test_microopt_lambda_even_bit_identical` reproduces the published reference pickle to ≥18 leading digits.
- Full pytest suite green: 6 passed, 2 skipped (the skips are `@pytest.mark.slow` regression tests; they pass when run with `-m slow`).

### Files added

- `_benchmarks/baseline_benchmark.py` — small-workload baseline driver (c=13 N=50 dps=50).
- `_benchmarks/win1_benchmark.py` — small-workload WIN 1 driver (same params, direct comparison).
- `_benchmarks/win1_pool_benchmark.py` — production-style A/B harness using `sweep._run_single_cutoff` (12-way multiprocessing).
- `_benchmarks/AB_VERIFIED_2026-04-14.md` — A/B verification record (apples-to-apples baseline-vs-WIN-1, bit-identical λ_min to 80 digits).

### No API changes

All public functions (`build_galerkin_matrix`, `compute_ground_state`, `extract_zeros`, `run_sweep`) keep their v0.1.0 signatures. v0.2.0 is a drop-in replacement.

## [0.1.0] — 2026-04-13

Initial public release. First independent implementation of the Connes–van Suijlekom Galerkin matrix from Proposition 4.1 of [arXiv:2511.23257](https://arxiv.org/abs/2511.23257). 15-cutoff production sweep (c = 13–67) at dps = 80–200, spanning 113 orders of magnitude in |γ₁ error|.
