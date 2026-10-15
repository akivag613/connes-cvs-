[**← `connes-cvs`**](../../README.md) · [**Papers**](../README.md) &nbsp;|&nbsp; **Riemann zeros** · [Guinand-Weil dictionary](../2_guinand_weil_dictionary_tail_order/) · [von Mangoldt measure](../3_matrix_von_mangoldt_measure/) · [Sine Loewner operator](../4_sine_loewner_bulk_boundary/)

<div align="center">

# High-Precision Approximation of Riemann Zeros<br>via the Truncated Weil Form

**Riemann zeros at high precision - _the numerics_ · Akiva Groskin, 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2605.20224-b31b1b.svg)](https://arxiv.org/abs/2605.20224)
[![Zenodo DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.19546514-1682D4.svg?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.19546514)
[![PyPI](https://img.shields.io/pypi/v/connes-cvs.svg?label=connes-cvs&color=4c1)](https://pypi.org/project/connes-cvs/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://doi.org/10.5281/zenodo.19546514)

</div>

> Builds and diagonalizes the finite Connes–van Suijlekom **Galerkin matrix** at high precision,
> extracting Riemann zeros to **hundreds of matching digits** and giving an independent
> numerical comparison with the prolate asymptotic discussed in Connes 2026 §6.4. Empirical measurements only;
> no claim regarding the Riemann Hypothesis.

Part of the [`connes-cvs` series](../../README.md#papers): **Riemann zeros - the numerics** · [**Guinand-Weil dictionary** - the structure](../2_guinand_weil_dictionary_tail_order/) · [**von Mangoldt measure** - the arithmetic](../3_matrix_von_mangoldt_measure/) · [**Sine Loewner operator** - bulk and boundary spectra](../4_sine_loewner_bulk_boundary/). Published on arXiv, [arXiv:2605.20224](https://arxiv.org/abs/2605.20224) (math.NT); archived on Zenodo, concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) (resolves to the latest version).

## About this folder

`high_precision_approximation_of_riemann_zeros.pdf` is the current published manuscript, 46 pages,
carrying every recorded correction in its text. It is byte-identical to the manuscript in the
Zenodo deposit that the concept DOI
[10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) resolves to.
**This paper's reproducibility package is the `connes-cvs` package**, together with
the reference data, examples and tests at the repository root. The other papers
keep their reproducibility material in their own folders. This folder holds the
Riemann-zeros manuscript:

| Artifact | Location |
| :--- | :--- |
| The implementation (builds + diagonalizes the CvS Galerkin matrix, extracts zeros) | [`../../connes_cvs/`](../../connes_cvs/) - installable as `pip install connes-cvs` |
| The `c = 100` datasets (`gamma` extractions, `N`-sweep) | [`../../data/c100/`](../../data/c100/) |
| Runnable examples (incl. the Aitken-Δ² check) | [`../../examples/`](../../examples/) |
| Regression tests against the committed `c = 13` references | [`../../tests/`](../../tests/) |
| Performance A/B benchmarks | [`../../benchmarks/`](../../benchmarks/) |
| Recorded errata (see below) | [`../../ERRATA.md`](../../ERRATA.md) |

The [repository root README](../../README.md) documents the headline result, installation,
quick start, the `c = 100` verification, and how it works.

## What the paper does

Builds and diagonalizes the finite Connes-van Suijlekom Galerkin matrix of the truncated Weil
quadratic form at high precision. It computes the smallest-positive eigenvalue across a
15-cutoff sweep and at `c = 100`, extracts the first ten Riemann zeros to hundreds of matching
digits, and gives a numerical comparison with the prolate asymptotic discussed in
Connes 2026 §6.4. That source reports numerical similarity between the full Weil
floor and the prolate quantity; their asymptotic identity is not established by
that comparison. The `c = 100` study is outside this paper's earlier `c <= 67` fit
window. It reports empirical measurements only; it makes no claim regarding the Riemann
Hypothesis. See [`../../ERRATA.md`](../../ERRATA.md) for the recorded corrections, none of
which changes a measured value: the 2026-06-26 finite-cutoff sign correction; the withdrawal of
the Section 8.2 Paley-Wiener mechanism (Table 14 is measured at `T = 400`, not `T = 800`); the
`c = 67` matching-digit count, 167 rather than 168 under the paper's own definition; two
summary digit-increment ranges in Section 6.5, 93-117 and 181-203; and the correction that the
`c = 100` negative block does not vanish at any finite cutoff tested, though a cutoff-free
evaluation still leaves the even sector non-negative.

**All of them are already incorporated in the text of the PDF in this folder**, so it needs no
errata overlay. `ERRATA.md` is retained as the record of what changed and when, and because it
still applies to anyone reading an earlier version of the manuscript, on either Zenodo or
[arXiv](https://arxiv.org/abs/2605.20224).

## License

Manuscript: [CC BY 4.0](LICENSE-PAPER-CC-BY-4.0.txt). The `connes-cvs` package is [MIT](../../LICENSE).
