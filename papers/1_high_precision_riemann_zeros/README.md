[**← `connes-cvs`**](../../README.md) · [**Papers**](../README.md) &nbsp;|&nbsp; **Paper 1** · [Paper 2](../2_guinand_weil_dictionary_tail_order/) · [Paper 3](../3_matrix_von_mangoldt_measure/)

<div align="center">

# High-Precision Approximation of Riemann Zeros<br>via the Truncated Weil Form

**Paper 1 — _the numerics_ · Akiva Groskin, 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2605.20224-b31b1b.svg)](https://arxiv.org/abs/2605.20224)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19546514.svg)](https://doi.org/10.5281/zenodo.19546514)
[![PyPI](https://img.shields.io/pypi/v/connes-cvs.svg?label=connes-cvs&color=4c1)](https://pypi.org/project/connes-cvs/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://doi.org/10.5281/zenodo.19546514)

</div>

> Builds and diagonalizes the finite Connes–van Suijlekom **Galerkin matrix** at high precision,
> extracting Riemann zeros to **hundreds of matching digits** and giving an independent
> out-of-sample test of the Connes 2026 §6.4 continuum asymptotic. Empirical measurements only;
> no claim regarding the Riemann Hypothesis.

Part of the [`connes-cvs` series](../../README.md#papers): **Paper 1 — the numerics** · [**Paper 2** — the structure](../2_guinand_weil_dictionary_tail_order/) · [**Paper 3** — the arithmetic](../3_matrix_von_mangoldt_measure/). Published on arXiv, [arXiv:2605.20224](https://arxiv.org/abs/2605.20224) (math.NT); archived on Zenodo, concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) (resolves to the latest version).

## About this folder

`high_precision_approximation_of_riemann_zeros.pdf` is the published manuscript (as on arXiv).
Unlike Papers 2 and 3 — self-contained theorem notes whose code lives beside them — **Paper 1's
reproducibility package is the `connes-cvs` package itself**, which is the whole repository. So
this folder holds the paper; the code and data live at the repository root:

| Paper 1 artifact | Location |
| :--- | :--- |
| The implementation (builds + diagonalizes the CvS Galerkin matrix, extracts zeros) | [`../../connes_cvs/`](../../connes_cvs/) — installable as `pip install connes-cvs` |
| The `c = 100` datasets (`gamma` extractions, `N`-sweep) | [`../../data/c100/`](../../data/c100/) |
| Runnable examples (incl. the Aitken-Δ² check) | [`../../examples/`](../../examples/) |
| Bit-identical regression tests | [`../../tests/`](../../tests/) |
| Performance A/B benchmarks | [`../../benchmarks/`](../../benchmarks/) |
| Erratum (finite-cutoff sign correction) | [`../../ERRATA.md`](../../ERRATA.md) |

The [repository root README](../../README.md) documents the headline result, installation,
quick start, the `c = 100` verification, and how it works.

## What the paper does

Builds and diagonalizes the finite Connes-van Suijlekom Galerkin matrix of the truncated Weil
quadratic form at high precision. It computes the smallest-positive eigenvalue across a
15-cutoff sweep and at `c = 100`, extracts the first ten Riemann zeros to hundreds of matching
digits, and gives an independent out-of-sample numerical test of the Connes 2026 §6.4 continuum
asymptotic. It reports empirical measurements only; it makes no claim regarding the Riemann
Hypothesis. See [`../../ERRATA.md`](../../ERRATA.md) for a finite-cutoff sign correction (no
quantitative result changed).

## License

Manuscript: CC BY 4.0. The `connes-cvs` package is [MIT](../../LICENSE).
