# Changelog

All notable changes to `connes-cvs` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [paper — Zenodo Version 3.2] — 2026-05-13

Title-only revision; paper body and all numerical data bit-identical to Version 3.1.  Title shortened to *"High-Precision Approximation of Riemann Zeros via the Truncated Weil Form."*  Published on Zenodo as **Version 3.2** with version-specific DOI [10.5281/zenodo.20156914](https://doi.org/10.5281/zenodo.20156914) (concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) now resolves to Version 3.2).

## [paper — Zenodo Version 3.1] — 2026-05-13

Acknowledgments-only revision; paper body and all numerical data bit-identical to Version 3.  Published on Zenodo as **Version 3.1** with version-specific DOI [10.5281/zenodo.20153365](https://doi.org/10.5281/zenodo.20153365).

## [paper — Zenodo Version 3] — 2026-05-13

Paper revision; no changes to the `connes-cvs` PyPI package.  Published on Zenodo as **Version 3** with version-specific DOI [10.5281/zenodo.20150435](https://doi.org/10.5281/zenodo.20150435).  Title changed to *"High-Precision Galerkin Experiments on the Connes–van Suijlekom Truncated Weil Form, with an Out-of-Sample Empirical Test at $c = 100$"*.

### Added

- **§6 — Out-of-sample empirical test at $c = 100$.** A new section reporting an $N$-sweep at $c = 100$, $N \in \{100, 150, 200, 250\}$, $T = 800$, $\mathrm{dps} = 500$, plus a precision retest at $N = 150$, $\mathrm{dps} = 1000$. Two consecutive Aitken-$\Delta^2$ accelerations on the overlapping triples give $\log_{10}|\lambda_\infty^{\mathrm{even}}(c{=}100)| \approx -536.76$ and $\approx -533.70$, approaching the Connes 2026 §6.4 heuristic continuum prediction ($\approx -530.38$) monotonically with $N$; consecutive first-difference ratios $0.8373$ and $0.8355$ match to two decimal places, evidence for a local geometric model. Gaps to the prediction are 6.39 OOM and 3.32 OOM respectively, out of $|x_\infty| \sim 530$ — agreement at the under-1%-of-exponent level on the deeper anchor, out-of-sample (the in-sample fit window was $c \leq 67$). To our knowledge this is the first independent out-of-sample empirical test of the §6.4 heuristic continuum prediction at $c > 14$.
- **§6.5 — $\gamma_k$ extraction at 307–329 matching digits.** The first ten Riemann zeros are extracted from the smallest-positive even-sector eigenvector at $c = 100$, $N = 250$, $\mathrm{dps} = 500$ to **307–329 matching digits**; the $N = 150$, $\mathrm{dps} = 1000$ precision retest reaches 219–242 digits at the same $\gamma_k$. For reference, CCM 2025 §6 reports $\gamma_1$ at ~55 digits ($c = 13$, $N = 120$). The depth record holds in the public CvS/CCM literature.
- **§2.4 — Spectral-triple recognition.** Under the unitary equivalence with CCM 2025 Lemma 5.1, the $F_{\mathrm{even}}$ test function used throughout this work coincides with $\widehat{\xi}_N$ in CCM 2025 Theorem 1.1(iii). Modulo a hypothesis-status caveat at $c = 100$ (the raw matrix carries a small block of negative-sign eigenvalues; we report the smallest-positive branch as an empirically distinguished object, not a theorem-derived ground state), every $\gamma_k$ extraction is equivalently an eigenvalue of the rank-one perturbed scaling operator $D_{\log}^{(\lambda,N)}$ at $\lambda = \sqrt{c}$.
- **§6.6 — Disclosure: dps-stable negative-eigenvalue block.** At $c = 100$, $N = 150$, five negative-sign eigenvalues reproduce identically across $\mathrm{dps} \in \{500, 1000\}$ (and the count $\{3, 5, 8, 11\}$ at $N \in \{100, 150, 200, 250\}$ scales linearly in $N$). Consistent with a condition-driven finite-$N$ artifact at marginal basis resolution rather than precision noise; certification as either a fixable conditioning artifact, a finite-$N$ structural feature, or a persistent feature is left to future work. Continuum positivity of $QW_\lambda$ is RH-equivalent and is not assumed at $\lambda = \sqrt{100}$.
- **§6.7 — Reframing of the empirical fit.** The Paper 1 fit $|\log_{10}\lambda_{\min}(c)| \approx 13.24 \, c^{0.634}$ on $c \leq 67$ at $N = 100$ is shown to be a **finite-$N$ rate**, not the continuum asymptote. The $c = 100$, $N = 200$ datum falsifies the pure-power-law extrapolation by 49 orders of magnitude. Corroborated by a $c = 67$, $N = 150$, $\mathrm{dps} = 500$ rerun: $\log_{10}|\lambda_{\min}| = -218.27$, a 46-OOM drop below the same-cutoff $N = 100$ value of $-172.10$ reported in Paper 1. The $N = 100$ data of the $c \leq 67$ sweep are Galerkin upper bounds rather than near-continuum values.
- **Statement on use of AI tools** added to the manuscript per arXiv submission policy.
- **Acknowledgment of Alain Connes** (see paper §Acknowledgments for details).
- **Bibliography.** Five new entries: Connes–Consani 2023 (arXiv:2106.01715), Davies–Plum 2004 (IMA JNA 24), Levitin–Shargorodsky 2004 (IMA JNA 24), Parlett 1998 (SIAM), Aitken 1926 (Proc. Roy. Soc. Edin.).

### Changed

- **Title.** Rewritten from V2's *"Structural Properties of the Connes–van Suijlekom Truncated Weil Minimizer: Sobolev Scaling, Multi-Zero Universality, and L-Function Extension"* to *"High-Precision Galerkin Experiments on the Connes–van Suijlekom Truncated Weil Form, with an Out-of-Sample Empirical Test at $c = 100$"*.  The new title foregrounds §6's c=100 empirical content over V2's structural-properties framing.
- **Zenodo bundle layout** is now flat (28 files at top level, individually previewable on the Zenodo Files panel) matching V1/V2 aesthetic. LaTeX source is distributed by the arXiv submission rather than the Zenodo deposit; figures are embedded in the PDF and need not be regenerated.

### Unchanged

- All numerical data on the original 15-cutoff sweep ($c = 13, 14, 17, \ldots, 67$ at $N = 100$, $T = 800$, $\mathrm{dps} = 150$–$200$) is bit-identical to Zenodo Version 2.
- The `connes-cvs` v0.2.2 PyPI package is unaffected. The new $c = 100$ data was produced by a local-only v0.2.3 port (in preparation for an eventual v0.3.0 umbrella release per the no-intermediate-releases discipline).
- All theorems, derivations, and structural observations from earlier versions are preserved.

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

- Comprehensive erratum document covering all three classes of correction (basis-attribution, arithmetic, internal-consistency), deposited as a supplementary file (`erratum_2026-04-19.pdf`) on the Version 2 Zenodo record.
- Acknowledgment of A. Connes in §12 of the revised paper.
- Version-history entry in paper front matter listing Version 1 (2026-04-13) and Version 2 (2026-04-19).

### Public locations

- **Zenodo** (primary public venue): concept DOI `10.5281/zenodo.19546514` (always resolves to the latest version); Version 1 DOI `10.5281/zenodo.19546515` (2026-04-13); Version 2 DOI `10.5281/zenodo.19655106` (2026-04-19).
- **GitHub** (`github.com/akivag613/connes-cvs-`): `paper-v2` git tag marks Version 2. HAL submission has been dropped; Zenodo + GitHub are the canonical public venues.

## [0.2.2] — 2026-04-19

Patch release superseding [0.2.1]. Fixes an internal version-string drift that slipped into the 0.2.1 wheel (`connes_cvs.__version__` was stuck at `"0.2.0"` while the installer-reported version was `0.2.1`). 0.2.1 has been yanked from PyPI in favor of this release. Runtime numerical behavior remains bit-identical to 0.2.0.

### Fixed

- `connes_cvs/__init__.py`: `__version__` now reflects the package version (was `"0.2.0"` in the 0.2.1 wheel).
- `tests/test_c13_regression.py::test_package_imports`: assertion replaced with a structural check that `connes_cvs.__version__ == importlib.metadata.version("connes-cvs")`, so the test no longer requires manual updates on each release and will catch any future drift.

### Unchanged from 0.2.1

- `connes_cvs/py.typed` marker file (PEP 561), so downstream type-checkers pick up the package's in-tree type annotations.
- `README.md`: validation-section wording aligned with the Zenodo Version 2 paper (`10.5281/zenodo.19655106`) — trigonometric-basis attribution throughout, corrected cross-validation factors (1.3 at $c=13$, 3 at $c=14$), paper DOI switched to the concept DOI (`10.5281/zenodo.19546514`) which always resolves to the latest version.
- `CITATION.cff`, `pyproject.toml` paper URL: concept DOI.
- `.github/CONTRIBUTING.md`, `tests/test_c13_regression.py` docstring: factor-1.7 discrepancy between this work and CCM §6 at $c=13$ reattributed to $N$ / precision / normalization differences (same trigonometric basis), per the Version 2 erratum.

### Unchanged from 0.2.0

- Public API surface (`build_galerkin_matrix`, `compute_ground_state`, `extract_zeros`; `connes_cvs.sweep.run_sweep`) — signatures and semantics identical.
- All numerical output — bit-identical to 0.2.0 at every workload tested, including the paper-canonical $c = 13$, $N = 100$, $T = 800$, dps $= 150$ run that reproduces the published Table 18 $\lambda_\min = 2.865 \times 10^{-59}$.
- All 15 rows of the production sweep.

## [0.2.1] — 2026-04-19 — **YANKED (superseded by 0.2.2)**

Documentation release. Yanked from PyPI due to an internal `__version__` string drift (`connes_cvs.__version__ == "0.2.0"` inside the 0.2.1 wheel). Functionally equivalent to 0.2.0; users should install 0.2.2 or later.

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
