# Changelog

All notable changes to `connes-cvs` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

- Restore the caller's exact mpmath bit precision and Flint precision/thread
  settings when matrix construction or zero extraction raises, including
  interruption. Successful calls retain the existing precision behavior;
  the numerical formulas and evaluation order are unchanged.
- Add independent tests for rejection, runtime failure, interruption and
  successful-call context behavior, following the issue reported by Dan Alec
  Yamaguchi in [PR #6](https://github.com/akivag613/connes-cvs-/pull/6).
- Strengthen release-source coverage and manifest agreement checks and verify
  that Windows tests activate the intended Flint backend.
- Credit the issue report and add a standalone notice for Paper 1's existing
  CC BY 4.0 manuscript license.
- Correct three historical benchmark path references below and clarify what
  the fast release tests check. These edits do not create a PyPI release.
- Release tooling: list `.gitignore` in the pyproject sdist allowlist so the
  two release manifests agree entry for entry (only `PKG-INFO` remains
  treated as build-generated), and bind the release version across
  `pyproject.toml`, `.github/release-files.json` and `connes_cvs.__version__`
  so a partial version bump fails the fast suite instead of reaching the
  release-archives job. The release inventory test now walks each
  allowlisted tree recursively, so a new nested public file must be
  allowlisted, and it no longer exempts paths for a test file that was
  removed (`tests/test_paper2_public_artifacts.py`).
- `examples/make_fig10_c100_gamma_digits.py` and
  `examples/make_fig9_c100_aitken.py` resolve their data bundle inside
  `main()` and report a missing or partial bundle on stderr with exit code 2
  instead of raising at import; a bundle must contain every file the figure
  reads.

## [Repository documentation audit] - 2026-09-08

- Completed sibling navigation across all four paper folders and clarified where
  each paper keeps its data, programs and figure sources. The sine Loewner paper
  contains two separate TikZ inputs and a third figure directly in its TeX section.
- Scoped the historical package pin to the Riemann-zeros reproduction and kept the
  sine Loewner verification independent of the Python package.
- Clarified that the displayed Fuchs asymptotic in Connes §6.4 is for a prolate
  quantity; comparison with the Weil spectral floor is numerical and heuristic.
  Corrected “above” to “below” for two negative logarithmic estimates, without
  changing any value, and clarified the finite-cutoff and earlier digit-count text.
- Updated the manuscript-version description for the von Mangoldt measure paper,
  removed a stale errata count, and corrected an obsolete historical test command.
- Refreshed checksums for the edited paper-folder documentation. Repaired five stale
  Guinand-Weil checksum entries for artifact and script revisions already committed
  on 2026-08-16, and included the previously unlisted full pivot transcript.
  Clarified the distinction between the historical certificate log and the later
  provenance JSON and pivot transcript. The underlying artifacts and numerical
  values are unchanged. Manuscripts, released
  source archives, numerical artifacts and software behavior are unchanged.
  This is a repository documentation update; no PyPI release is made.

## [Sine Loewner operator - first publication] - 2026-09-08

Added Paper 4, *Bulk trace laws and algebraic boundary interaction for the sine Loewner operator*,
with its complete 88-page manuscript, three figures and all 64 released source files.
Zenodo version DOI [10.5281/zenodo.22662222](https://doi.org/10.5281/zenodo.22662222);
concept DOI [10.5281/zenodo.22662221](https://doi.org/10.5281/zenodo.22662221).

- The paper proves uniform trace laws, a dimension-independent finite-matrix comparison,
  half-line boundary flow and algebraic splitting, with three interval certificates.
- Its own verification programs and exact witnesses are included under
  `papers/4_sine_loewner_bulk_boundary/source/anc/`; they do not depend on `connes-cvs`.
- Updated the paper index, citation guidance and license scope, and made the release
  manifest tests explicitly exclude every `papers/` member from both package archives.
- This is a paper and repository update, not a software release. The package remains
  version 0.3.1; its implementation, distribution allowlists and root software citation
  are unchanged. No new PyPI distribution accompanies this update.

## [0.3.1] - 2026-08-14

Documentation and metadata only. **No code path, numerical value, or public API changes**, and the c=13 bit-identity regression gate is unaffected.

- `connes_cvs/operator.py`: the `build_galerkin_matrix` docstring gave the trigonometric basis denominator as `2*log(c)`; the code uses `L = log(c)`. Reported by M. Osman. Docstring only; the assembled matrix was always correct.
- The PyPI long description is this repository's `README.md`, and PyPI freezes it at upload, so any paper *version* DOI in it goes stale by construction. Every paper reference now uses its **concept DOI**, which always resolves to the current version. The same change removes the hard-coded errata count and the version-pinned statements about what arXiv is currently serving.
- The `ERRATA.md` shipped inside the source distribution is refreshed to the current one.
- Third-party version references are stated in the past tense beside the collaborator's own concept DOI, so they remain accurate as those authors update their deposits.

## [Riemann zeros - Zenodo Version 3.7] - 2026-08-14

Deposited as **Version 3.7**, version DOI [10.5281/zenodo.21934908](https://doi.org/10.5281/zenodo.21934908); the concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) resolves to it. The manuscript folds in the correction recorded in `ERRATA.md` for 2026-08-14: the statement that raising the archimedean cutoff to `T = 1200` removes the small block of negative-sign eigenvalues at `c = 100` was wrong. That block rearranges with the cutoff rather than clearing at any cutoff tested, and the conclusion rests instead on the cutoff-free interval `LDL^T` certificate, which is unaffected. The revision also records that the odd sector, which this paper's own computations do not probe, has since been probed independently by M. Osman and reproduced here, and it cites two Connes references in their published journal versions. Every numerical-data file in the deposit is byte-identical to Version 3.6; the five replaced files are the manuscript PDF and its four supporting documents. The record additionally gains the author ORCID, an affiliation, and a supplemented-by relation to this repository. **No measured value changes.**

## [Guinand-Weil dictionary - Zenodo v1.3] - 2026-08-14

Deposited as **v1.3**, version DOI [10.5281/zenodo.21935332](https://doi.org/10.5281/zenodo.21935332); the concept DOI [10.5281/zenodo.21124802](https://doi.org/10.5281/zenodo.21124802) resolves to it. Two references that have since appeared in print are updated: Connes and van Suijlekom to *Communications in Mathematical Physics* **406** (2025), article 312, and Connes, Consani and Moscovici to a chapter in *Applications of Noncommutative Geometry to Gauge Theories, Field Theories, and Quantum Space-Time*, EMS Series of Lectures in Mathematics, EMS Press (2026), pages 39-76. In the reproducibility archive, `arch_tail_exact_vs_asymptotic.json` tabulated `h_+'(t)` at `t = 50`, `100` and `1000` from a series acceleration that had not converged at the working precision; the `t = 1000` entry was low by a factor of 4.194. The corrected values agree with the asymptotic `1/t + 1/(12 t^3)` to better than one part in `10^8`, every tabulated row still satisfies the Lemma 3.1 envelope `1/t + 13/(10 t^2)`, and the certification floor `T_required = 7.98e+62` is unchanged; the margin at `t = 1000` falls from 7.6e-4 to 1.3e-6. The archive also replaces its separate release manifest and verification report with a single verification document. **No theorem, proof, or reported bound changes.**

## [von Mangoldt measure - Zenodo v2.2] - 2026-08-14

Deposited as **v2.2**, version DOI [10.5281/zenodo.21935511](https://doi.org/10.5281/zenodo.21935511); the concept DOI [10.5281/zenodo.21242028](https://doi.org/10.5281/zenodo.21242028) resolves to it. The same two Connes references are updated to their published journal versions. The manuscript is otherwise the corrected v2 of 2026-07-27, unchanged. **No theorem, proof, or numerical result changes.**

## [Riemann zeros - Zenodo Version 3.6] - 2026-08-13

Corrected supporting documentation for the revised manuscript, deposited as **Version 3.6**, version DOI [10.5281/zenodo.21918387](https://doi.org/10.5281/zenodo.21918387); the concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) resolves to it. The manuscript PDF and every numerical-data file are byte-identical to Version 3.5, and no statement about the mathematics changes. Four supporting documents are corrected. `ERRATA.md` had said that all three post-June corrections were recorded at Version 3.4, when only the two dated 2026-08-12 were, and stated its entries in the present tense although the PDF deposited beside them already carried the fixes. `MANIFEST.sha256` did not cover the manuscript PDF; it now does, so the deposit is self-verifying in full. `README.md` described the manifest's scope inconsistently in two places, cited the RMT-spacing figure by its file number rather than its printed number (Figure 9), and carried a wall-clock table whose `c = 100` rows were estimates that disagreed with the deposit's own data; those rows now report the `t_total_s` values recorded inside the deposited result files, and the rows that remain estimates are labelled as such. `CHANGELOG.md` carried the same figure-number error.

## [Guinand-Weil dictionary - Zenodo v1.2] - 2026-08-13

Deposited as **v1.2**, version DOI [10.5281/zenodo.21918605](https://doi.org/10.5281/zenodo.21918605); the concept DOI [10.5281/zenodo.21124802](https://doi.org/10.5281/zenodo.21124802) resolves to it. The manuscript adds a citation identifying the earlier finite-cutoff computation that motivates the tail-order theorem, previously described only as unpublished, and corrects the companion-package repository path to `papers/2_guinand_weil_dictionary_tail_order/`. **No mathematical statement changes.** In the reproducibility archive, `requirements.txt` was missing `matplotlib`, which `make_figures.py` needs; the dependency paragraph said three Arb scripts and then listed four, and never mentioned sympy; and `arch_tail_exact_asymptotic.py` shipped with no command in the verification report to run it. The same revision has been submitted to arXiv.

## [von Mangoldt measure - Zenodo v2.1] - 2026-08-13

Deposited as **v2.1**, version DOI [10.5281/zenodo.21911012](https://doi.org/10.5281/zenodo.21911012); the concept DOI [10.5281/zenodo.21242028](https://doi.org/10.5281/zenodo.21242028) resolves to it. The manuscript is the corrected v2 of 2026-07-27, unchanged; v2.1 revises documentation in the reproducibility archive only.

## [Riemann zeros - Zenodo Version 3.5] - 2026-08-13

The revised manuscript, deposited on Zenodo as **Version 3.5**, version DOI [10.5281/zenodo.21916945](https://doi.org/10.5281/zenodo.21916945); the concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) resolved to it on publication, and now resolves to Version 3.6, which corrects four of its supporting documents. Where Version 3.4 recorded the 2026-08-12 corrections in [ERRATA.md](ERRATA.md) beside an unchanged PDF, this version folds all of them into the paper text, together with the 2026-08-13 Section 6.5 correction found in a pre-submission audit. The Section 8.2 Paley-Wiener mechanism is withdrawn and the section additionally records that the original argument placed the Galerkin interval at `[0, T]` when the form acts on `[0, L]` with `L = log c`; the `c = 67` count reads 167; the Section 6.5 summary ranges read 93-117 and 181-203; and the section formerly reporting the absence of independent reproductions now records those reported by five parties. **No measured value changes**, and every numerical-data file is bit-identical to Versions 3.3 and 3.4. The same revision has been submitted to arXiv.

## [Riemann zeros - Zenodo Version 3.4] - 2026-08-13

Errata deposit for [arXiv:2605.20224](https://arxiv.org/abs/2605.20224), published on Zenodo as **Version 3.4**, version DOI [10.5281/zenodo.21910402](https://doi.org/10.5281/zenodo.21910402); the concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) resolved to it on publication. The deposit carries the two 2026-08-12 entries added to [ERRATA.md](ERRATA.md): the Section 8.2 Paley-Wiener mechanism is withdrawn (its own Section 11 test shows the Sobolev exponent is insensitive to `T`: 46.140, 46.031, 45.934 at `T` = 400, 800, 1600 for `c=23` on `N` in {40, 60, 80} at `dps=150`), so `A = 0.432` has no meaning as derived and Table 14 is measured at `T = 400`, not `T = 800`; and the `c=67` matching-digit count is 167, not 168, under the paper's own definition. The three-cutoff test was first carried out by M. Osman and reported in [issue #2](https://github.com/akivag613/connes-cvs-/issues/2); it has been reproduced independently here. **Neither correction changes a quantitative result, and the Version 3.4 manuscript PDF is byte-identical to Version 3.3** - the corrections live in the errata file until the next manuscript revision. No changes to the `connes-cvs` package or to any numerical data.

## [0.3.0] - 2026-08-12

Release combining the `extract_zeros` correctness fix with a validated, resumable production runner. Its numerical contract is deliberately scoped:

- The same-precision A/B cell (`c=13`, `N=80`, `T=400`, `dps=80`) reproduces all 80 committed printed digits of `lambda_even`.
- The extended validation cell (`c=13`, `N=100`, `T=400`, `dps=80`) matches the committed cross-precision reference by at least 20 lambda digits and 18 gamma-error digits.
- On the small identity cell (`c=13`, `N=8`, `T=60`, `dps=30`), the classic and runner paths agree exactly in every raw mpmath matrix entry, eigenvalue and eigenvector component; the pooled and serial runner paths also agree exactly.
- The exact-reference leg also recomputes the published `c=13`, `N=100`, `T=800`, `dps=150` cell, applies its fatal 22-digit lambda gate, and checks the first-zero error. These gates do not recompute the complete 15-cutoff or `c=100` production sweeps and do not assert blanket raw-value equality across versions or platforms.

The published paper values are unchanged. The exact package version cited by the papers remains installable as `connes-cvs==0.2.2`.

### Fixed (correctness)

- **`extract_zeros` float64 `L` pitfall.** A Python-float `L` (e.g. `math.log(13)`) carries only ~16 significant digits and silently capped the zero-extraction accuracy near 1e-16 regardless of `dps`. `extract_zeros` now accepts a `c=` parameter (preferred; computes `L = mp.log(c)` internally at the active precision) and emits a `UserWarning` when `L` arrives as a Python float. Reported independently by two users: [issue #2](https://github.com/akivag613/connes-cvs-/issues/2) (M. Osman) and [issue #3](https://github.com/akivag613/connes-cvs-/issues/3) (karl-keysingularity). Both reports diagnosed the cause precisely, and this fix follows their diagnosis; our thanks to both reporters.
- **`sweep.run_sweep` gamma accuracy.** The sweep passed a float64 `L` into `extract_zeros`, capping the reported `gamma1_error` near 1e-16; it now passes the integer cutoff through `c=`, so `L` is evaluated at full working precision. The change is confined to the extraction step: matrix assembly and the eigensolve never saw the narrowed `L`, so `lambda_min` is unaffected.

### Added

- **`connes_cvs.runner`** - production cell runner (`CellConfig`, `GalerkinCell`, `run_cell`) with explicit `dps` and optional `flint_bits`, spawn-safe parallel psi-cache computation, progress callbacks, lossless mpmath transport, integrity-checked atomic checkpoints, environment fingerprints and JSON artifacts. It does not auto-select or certify sufficient working precision. Checkpoint hashes are corruption checks rather than authentication; resume only trusted local files, and treat locks as single-writer guards.
- **Concurrency boundary** - mpmath/Flint precision contexts and arithmetic caches are process-global. Do not run different-precision cells concurrently in Python threads; use the process-based runner or sweep.
- **`connes_cvs.validation.arb_eigenpair_residual_bound`** - an outward Arb residual bound for an exact supplied finite real-symmetric mpmath matrix. Its result explicitly excludes truncation-limit and infinite-operator claims.
- **`tests/reference_values.json`** - three provenance-labelled c=13 references: the published `T=800` cell, the extended `T=400` validation cell and the 80-digit A/B cell. The slow exact-reference suite recomputes all three; the `T=800` runner cell also performs a fatal 22-digit in-run regression check.
- **Third-party validation artifact** - karl-keysingularity's native-Windows/Python-3.11.9 `connes-cvs` 0.2.2 reproduction at `c=13`, `N=100`, `T=400`, `dps=80`, credited and checked against the extended validation reference.

### Changed

- **`sweep.run_sweep`** delegates each cutoff to the validated runner while retaining the v0.2 result fields. The default process cap is `min(cpu_count(), 8)`; this is a resource cap, not a CPU-affinity guarantee.
- **Operator hardening** validates inputs, enumerates prime powers without the old nested search, evaluates only nonnegative indices and mirrors the exact parity identities, and fills one matrix triangle before reflection. Direct parity and legacy full-assembly comparisons gate those changes.
- **Test gate is now real.** `tests/test_c13_regression.py` previously skipped its slow leg unconditionally; it now computes the A/B, extended validation and published `T=800` cells. Fast tests cover the `extract_zeros` warning/`c=` API, finite-matrix Arb validation, hardening identities, runner identity and the credited Windows artifact.
- **CI** (`.github/workflows/tests.yml`): the fast suite spans supported Python versions and operating systems at the mpmath floor; separate legs cover oldest/latest Flint, Windows spawn, static typing, the pinned exact-reference stack (`mpmath==1.4.1`, `python-flint==0.8.0`, `gmpy2==2.3.0`), and fail-closed archive verification plus clean installs.
- **Packaging**: `requires-python` is now `>=3.10`. Python 3.9 reached end of life in October 2025, and the pinned build backend (`hatchling==1.32.0`, which keeps the release archives byte-reproducible) requires 3.10 or newer, so a source build on 3.9 could not succeed. Installations that need Python 3.9 should pin `connes-cvs==0.2.2`, which is unaffected. Version metadata is aligned at 0.3.0.
- **Docs refresh**: examples use full-precision `c=` extraction and guarded multiprocessing; the published mixed-precision sweep is split correctly; `c=100` recipes state `flint_bits=4*dps` and the smallest-positive branch; matching digits use `floor(-log10(error))`, so the `c=67` error `1.478e-168` is 167 matching digits under the repository convention (the paper's historical “168” is an error-exponent label).

### Papers

- **The von Mangoldt measure paper: v2 is the version of record**: the corrected v2 of *A matrix-valued von Mangoldt measure in the finite Connes-van Suijlekom path* was published on Zenodo on 2026-07-27 (version DOI [10.5281/zenodo.21612746](https://doi.org/10.5281/zenodo.21612746)); the concept DOI [10.5281/zenodo.21242028](https://doi.org/10.5281/zenodo.21242028) resolved to it on publication, and now resolves to v2.1, which carries the same manuscript. V2 adds the reciprocal-zero-set restriction, corrects pole and interpretation scope, and replaces the unqualified variance claim with an explicitly pairwise-uncorrelated version. It makes no claim regarding RH.

## [von Mangoldt measure - Zenodo 10.5281/zenodo.21242028] - 2026-07-07

A third companion paper is published on **Zenodo** (concept DOI [10.5281/zenodo.21242028](https://doi.org/10.5281/zenodo.21242028), resolves to the latest version), *A matrix-valued von Mangoldt measure in the finite Connes-van Suijlekom path* (math.NT, math.SP; Zenodo is the venue of record). It realizes the prime side of the Weil-Guinand explicit formula as an exact, cutoff-free matrix-valued von Mangoldt measure on the finite path, and proves arithmetic rigidity, a finite source-to-jet dictionary, and a sharp finite vanishing-moment ceiling at the prime edge (with an uncertainty-principle interpretation in the band-limited sense); it makes no claims regarding the Riemann Hypothesis. Manuscript and full reproducibility package (organized into `source/`, `figures/`, `scripts/`, `artifacts/`) are in [`papers/3_matrix_von_mangoldt_measure/`](https://github.com/akivag613/connes-cvs-/tree/main/papers/3_matrix_von_mangoldt_measure). No changes to the `connes-cvs` PyPI package or to any numerical data.

## [Guinand-Weil dictionary - arXiv:2607.02828] - 2026-07-07

A second companion paper is now publicly available on **arXiv**: [arXiv:2607.02828](https://arxiv.org/abs/2607.02828) (math.NT primary, math.SP; submitted 2 July 2026), *A finite Guinand-Weil dictionary and archimedean tail order for the truncated Weil quadratic form*. It gives an exact finite Guinand-Weil zero-source dictionary for the truncated Weil form and a finite-cutoff archimedean tail-order theorem with a two-sided certification rule; it makes no claims regarding the Riemann Hypothesis. Manuscript and full reproducibility package are in [`papers/2_guinand_weil_dictionary_tail_order/`](https://github.com/akivag613/connes-cvs-/tree/main/papers/2_guinand_weil_dictionary_tail_order); archived on Zenodo, concept DOI [10.5281/zenodo.21124802](https://doi.org/10.5281/zenodo.21124802) (always resolves to the latest version). No changes to the `connes-cvs` PyPI package or to any numerical data.

## [Riemann zeros - correction] - 2026-06-26

Correction to [arXiv:2605.20224](https://arxiv.org/abs/2605.20224). The negative-sign even-sector eigenvalue blocks reported at $c=100$ (abstract, §2.4, §6.6, the $N$-sweep table) and for $L(s,\chi_3)$ at $c=23,29$ (§8.10, and the Future Directions section) are artifacts of the finite archimedean integration cutoff $T$, not features of the operator: they are stable in working precision but vanish once $T$ is increased, so cutoff-free the relevant even sectors are non-negative and the smallest-positive branch is the genuine smallest eigenvalue. **No quantitative result changes**: the $\gamma_k$ recovery (307–329 digits at $c=100$, $N=250$), the Aitken extrapolation, and all convergence data are unaffected. The "structural character dependence" reading of the $\chi_3$ case is withdrawn; both blocks are the same archimedean-truncation artifact. See [ERRATA.md](ERRATA.md). Professor A. Connes prompted the investigation; the cutoff sensitivity was then independently identified by B. W. A. Silva (Zenodo 20650146), consistent with the naturally even, positive ground state of R. Andrews (Zenodo 20427500). The corrected version is now live as **arXiv:2605.20224v2** (announced 2026-06-29) and **Zenodo Version 3.3** (version DOI [10.5281/zenodo.20931069](https://doi.org/10.5281/zenodo.20931069); the concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) continues to resolve to the latest version). No changes to the `connes-cvs` package or to any numerical values; one descriptive `notes` string in a $c=100$ data file was corrected to the artifact framing.

## [Riemann zeros - arXiv:2605.20224] - 2026-05-13

The paper is now publicly available on **arXiv**: [arXiv:2605.20224](https://arxiv.org/abs/2605.20224) (math.NT, primary; submitted 13 May 2026), *High-Precision Approximation of Riemann Zeros via the Truncated Weil Form*. The arXiv version corresponds to Zenodo **Version 3.2** (concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) always resolves to the latest version). No changes to the `connes-cvs` PyPI package or to any numerical data.

## [Riemann zeros - Zenodo Version 3.2] - 2026-05-13

Title-only revision; paper body and numerical-data files are byte-for-byte unchanged from Version 3.1. Title shortened to *"High-Precision Approximation of Riemann Zeros via the Truncated Weil Form."* Published on Zenodo as **Version 3.2** with version-specific DOI [10.5281/zenodo.20156914](https://doi.org/10.5281/zenodo.20156914) (concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) now resolves to Version 3.2).

## [Riemann zeros - Zenodo Version 3.1] - 2026-05-13

Acknowledgments-only revision; paper body and numerical-data files are byte-for-byte unchanged from Version 3. Published on Zenodo as **Version 3.1** with version-specific DOI [10.5281/zenodo.20153365](https://doi.org/10.5281/zenodo.20153365).

## [Riemann zeros - Zenodo Version 3] - 2026-05-13

Paper revision; no changes to the `connes-cvs` PyPI package.  Published on Zenodo as **Version 3** with version-specific DOI [10.5281/zenodo.20150435](https://doi.org/10.5281/zenodo.20150435).  Title changed to *"High-Precision Galerkin Experiments on the Connes–van Suijlekom Truncated Weil Form, with an Out-of-Sample Empirical Test at $c = 100$"*.

### Added

- **§6 - Out-of-sample empirical test at $c = 100$.** A new section reporting an $N$-sweep at $c = 100$, $N \in \{100, 150, 200, 250\}$, $T = 800$, $\mathrm{dps} = 500$, plus a precision retest at $N = 150$, $\mathrm{dps} = 1000$. Two consecutive Aitken-$\Delta^2$ accelerations on the overlapping triples give $\log_{10}|\lambda_\infty^{\mathrm{even}}(c{=}100)| \approx -536.76$ and $\approx -533.70$, approaching the Connes 2026 §6.4 heuristic continuum prediction ($\approx -530.38$) monotonically with $N$; consecutive first-difference ratios $0.8373$ and $0.8355$ match to two decimal places, evidence for a local geometric model. Gaps to the prediction are 6.39 OOM and 3.32 OOM respectively, out of $|x_\infty| \sim 530$. Four points do not exclude alternative convergence laws.
- **§6.5 - $\gamma_k$ extraction at 307–329 matching digits.** The first ten Riemann zeros are extracted from the smallest-positive even-sector eigenvector at $c = 100$, $N = 250$, $\mathrm{dps} = 500$ to **307–329 matching digits**; the $N = 150$, $\mathrm{dps} = 1000$ precision retest reaches 219–242 digits at the same $\gamma_k$. Matching digits here use `floor(-log10(error))`. For reference, CCM 2025 §6 reports $\gamma_1$ at approximately 55 digits ($c = 13$, $N = 120$).
- **§2.4 - Spectral-triple recognition.** Under the unitary equivalence with CCM 2025 Lemma 5.1, the $F_{\mathrm{even}}$ test function used throughout this work coincides with $\widehat{\xi}_N$ in CCM 2025 Theorem 1.1(iii). Modulo a hypothesis-status caveat at $c = 100$ (the raw matrix carries a small block of negative-sign eigenvalues; we report the smallest-positive branch as an empirically distinguished object, not a theorem-derived ground state), every $\gamma_k$ extraction is equivalently an eigenvalue of the rank-one perturbed scaling operator $D_{\log}^{(\lambda,N)}$ at $\lambda = \sqrt{c}$.
- **§6.6 - Disclosure: dps-stable negative-eigenvalue block.** At $c = 100$, $N = 150$, five negative-sign eigenvalues reproduce identically across $\mathrm{dps} \in \{500, 1000\}$ (and the count $\{3, 5, 8, 11\}$ at $N \in \{100, 150, 200, 250\}$ scales linearly in $N$). Consistent with a condition-driven finite-$N$ artifact at marginal basis resolution rather than precision noise; certification as either a fixable conditioning artifact, a finite-$N$ structural feature, or a persistent feature is left to future work. Continuum positivity of $QW_\lambda$ is RH-equivalent and is not assumed at $\lambda = \sqrt{100}$.
- **§6.7 - Reframing of the empirical fit.** The paper's fit $|\log_{10}\lambda_{\min}(c)| \approx 13.24 \, c^{0.634}$ on $c \leq 67$ at $N = 100$ is shown to be a **finite-$N$ rate**, not the continuum asymptote. The $c = 100$, $N = 200$ datum falsifies the pure-power-law extrapolation by 49 orders of magnitude. Corroborated by a $c = 67$, $N = 150$, $\mathrm{dps} = 500$ rerun: $\log_{10}|\lambda_{\min}| = -218.27$, a 46-OOM drop below the same-cutoff $N = 100$ value of $-172.10$ reported in the paper. The $N = 100$ data of the $c \leq 67$ sweep are Galerkin upper bounds rather than near-continuum values.
- **Statement on use of AI tools** added to the manuscript per arXiv submission policy.
- **Acknowledgment of Alain Connes** (see paper §Acknowledgments for details).
- **Bibliography.** Five new entries: Connes–Consani 2023 (arXiv:2106.01715), Davies–Plum 2004 (IMA JNA 24), Levitin–Shargorodsky 2004 (IMA JNA 24), Parlett 1998 (SIAM), Aitken 1926 (Proc. Roy. Soc. Edin.).

### Changed

- **Title.** Rewritten from V2's *"Structural Properties of the Connes–van Suijlekom Truncated Weil Minimizer: Sobolev Scaling, Multi-Zero Universality, and L-Function Extension"* to *"High-Precision Galerkin Experiments on the Connes–van Suijlekom Truncated Weil Form, with an Out-of-Sample Empirical Test at $c = 100$"*.  The new title foregrounds §6's c=100 empirical content over V2's structural-properties framing.
- **Zenodo bundle layout** is now flat (28 files at top level, individually previewable on the Zenodo Files panel) matching V1/V2 aesthetic. LaTeX source is distributed by the arXiv submission rather than the Zenodo deposit; figures are embedded in the PDF and need not be regenerated.

### Unchanged

- The original 15-cutoff data file ($c = 13, 14, 17, \ldots, 67$ at $N = 100$, $T = 800$, $\mathrm{dps} = 150$–$200$) is byte-for-byte unchanged from Zenodo Version 2.
- The `connes-cvs` v0.2.2 PyPI package is unaffected. The new $c = 100$ data was produced by the unreleased v0.2.3-local runner.
- All theorems, derivations, and structural observations from earlier versions are preserved.

## [Riemann zeros - Zenodo Version 2] - 2026-04-19

Paper revision; no code changes. Published on Zenodo as Version 2 with version-specific DOI [10.5281/zenodo.19655106](https://doi.org/10.5281/zenodo.19655106) (concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) now resolves to Version 2). An erratum accompanies the revised PDF as a supplementary file on the Zenodo record.

### Corrected

- **Basis attribution throughout paper body** (§1, §2.2, §4.2 Table 1, §4.3, §5.1.1, §8.5, §9): the CCM 2025 and Connes 2026 Galerkin computations use the same **trigonometric basis** as this work, not prolate-spheroidal - as is evident from CCM Lemma 5.1 (matrix entries defined via the kernel $\sin(2\pi n y/L)$) and Connes 2026 §6 (referring to the "trigonometric orthonormal basis"). The prior prolate-basis attribution was an error of our reading; the correct attribution has always been in the published sources. Prolate wave functions appear in a distinct role in the program (approximation construction for the limit $k_\lambda$ per Connes 2026 §6.3–§6.4).
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

## [0.2.2] - 2026-04-19

Patch release superseding [0.2.1]. Fixes an internal version-string drift that slipped into the 0.2.1 wheel (`connes_cvs.__version__` was stuck at `"0.2.0"` while the installer-reported version was `0.2.1`). 0.2.1 has been yanked from PyPI in favor of this release. The documented reference outputs are unchanged from 0.2.0.

### Fixed

- `connes_cvs/__init__.py`: `__version__` now reflects the package version (was `"0.2.0"` in the 0.2.1 wheel).
- `tests/test_c13_regression.py::test_package_imports`: assertion replaced with a structural check that `connes_cvs.__version__ == importlib.metadata.version("connes-cvs")`, so the test no longer requires manual updates on each release and will catch any future drift.

### Unchanged from 0.2.1

- `connes_cvs/py.typed` marker file (PEP 561), so downstream type-checkers pick up the package's in-tree type annotations.
- `README.md`: validation-section wording aligned with the Zenodo Version 2 paper (`10.5281/zenodo.19655106`) - trigonometric-basis attribution throughout, corrected cross-validation factors (1.3 at $c=13$, 3 at $c=14$), paper DOI switched to the concept DOI (`10.5281/zenodo.19546514`) which always resolves to the latest version.
- `CITATION.cff`, `pyproject.toml` paper URL: concept DOI.
- `.github/CONTRIBUTING.md`, `tests/test_c13_regression.py` docstring: factor-1.7 discrepancy between this work and CCM §6 at $c=13$ reattributed to $N$ / precision / normalization differences (same trigonometric basis), per the Version 2 erratum.

### Unchanged from 0.2.0

- Public API surface (`build_galerkin_matrix`, `compute_ground_state`, `extract_zeros`; `connes_cvs.sweep.run_sweep`) - signatures and semantics identical.
- The tested reference workloads reproduce the same reported values as 0.2.0, including the paper-canonical $c = 13$, $N = 100$, $T = 800$, dps $= 150$ run and the 15 published sweep rows.
- All 15 rows of the production sweep.

## [0.2.1] - 2026-04-19 - **YANKED (superseded by 0.2.2)**

Documentation release. Yanked from PyPI due to an internal `__version__` string drift (`connes_cvs.__version__ == "0.2.0"` inside the 0.2.1 wheel). Functionally equivalent to 0.2.0; users should install 0.2.2 or later.

## [0.2.0] - 2026-04-14

Performance release: the recorded same-environment A/B cell shows an approximately **2× faster** psi-cache phase, with all 80 printed lambda digits agreeing with the v0.1.0 run.

### Performance

- **WIN 1 - h_plus memoization + fused real kernel.** `h_plus(τ) = Re ψ(¼ + iτ/2) − log π` is mathematically even in τ. The new code memoizes `h_plus` keyed on `|τ|` and reuses it across `psi_arch` and `psi_arch_deriv` (which share quadrature nodes). A fused real-arithmetic kernel `_re_S_and_dS_fused` computes both `Re S_hat_x` and `Re dS_hat_x_dx` in one pass, sharing all sub-expressions; a pair-cache hands the result from the first quadrature pass to the second. Net effect at production scale (c=13, N=80, T=400, dps=80, 12-way Pool):

  | Phase | v0.1.0 | v0.2.0 | Speedup |
  |---|---|---|---|
  | psi cache | 57.55 s | 27.94 s | **2.06×** |
  | total wall | 64.94 s | 35.40 s | **1.83×** |

  Saves ~40 minutes per full 15-cutoff production sweep at dps=150.

- **WIN 3 - drop redundant `mp.mpf(int)` conversion in Galerkin Q-matrix assembly** (commit 82f0953). Arithmetic-preserving micro-optimization; visible only at very large N.

### Correctness

- λ_min reproduces v0.1.0 to all 80 decimal digits printed at the A/B test workload (c=13, N=80, T=400, dps=80).
- At the **published reference workload** (c=13, N=100, T=800, dps=150), the v0.2.0 code computes `λ_min = 2.86545361493028029516…e-59`, exactly matching the paper Table 18 published value of `2.865 × 10⁻⁵⁹` to all reported precision. End-to-end wall time at this workload: 127.3 s (vs. historical baseline 214.8 s = **1.69× faster** on the paper-canonical run).
- The historical pickle-dependent slow regression leg required at least 18 leading digits when the local, undistributed research pickle was present; it skipped otherwise.
- At the time of that release the suite reported 6 passed and 2 slow tests deselected or skipped, depending on invocation. The v0.3.0 gate above supersedes this historical test accounting.

### Files added

- `benchmarks/baseline_benchmark.py` - historical small-workload baseline driver (c=13 N=50 dps=50).
- `benchmarks/win1_benchmark.py` - historical small-workload WIN 1 driver (same params, direct comparison).
- `benchmarks/win1_pool_benchmark.py` - historical production-style A/B harness using the then-current internal sweep path (12-way multiprocessing).
- `benchmarks/AB_VERIFIED_2026-04-14.md` - A/B verification record (same-environment baseline-vs-WIN-1; all 80 printed lambda digits agree).

### No API changes

All public functions (`build_galerkin_matrix`, `compute_ground_state`, `extract_zeros`, `run_sweep`) keep their v0.1.0 signatures. v0.2.0 is a drop-in replacement.

## [0.1.0] - 2026-04-13

Initial public repository release implementing the Connes–van Suijlekom Galerkin matrix from Proposition 4.1 of [arXiv:2511.23257](https://arxiv.org/abs/2511.23257). It was tagged in the repository but was never uploaded to PyPI. The associated 15-cutoff production sweep (c = 13–67) used dps = 150–200 and spans 113 orders of magnitude in |γ₁ error|.
