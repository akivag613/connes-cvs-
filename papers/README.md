[**← `connes-cvs`**](../README.md) &nbsp;|&nbsp; **Papers**

<div align="center">

# The `connes-cvs` paper series

**Four papers by Akiva Groskin on the truncated Weil form and associated finite and continuum spectral problems**

[![Riemann zeros · arXiv](https://img.shields.io/badge/Riemann_zeros-arXiv%3A2605.20224-b31b1b.svg)](https://arxiv.org/abs/2605.20224)
[![Guinand-Weil dictionary · arXiv](https://img.shields.io/badge/Guinand--Weil_dictionary-arXiv%3A2607.02828-b31b1b.svg)](https://arxiv.org/abs/2607.02828)
[![von Mangoldt measure · Zenodo](https://img.shields.io/badge/von_Mangoldt_measure-Zenodo%3A10.5281%2Fzenodo.21242028-1682D4.svg?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.21242028)
[![Sine Loewner operator · Zenodo](https://img.shields.io/badge/Sine_Loewner_operator-Zenodo%3A10.5281%2Fzenodo.22662221-1682D4.svg?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.22662221)

</div>

The [`connes-cvs`](../README.md) package implements the finite Galerkin calculations of Paper 1.
Papers 2–4 have their own reproducibility programs; Paper 4 does not depend on the package.
The series combines numerical evidence, finite-dimensional theorems and infinite-dimensional
spectral results. None of the papers claims a proof of the Riemann Hypothesis.

| | Paper | Home | Preprint / DOI |
| :--- | :--- | :--- | :--- |
| **1** | [**High-Precision Approximation of Riemann Zeros via the Truncated Weil Form**](https://arxiv.org/abs/2605.20224) - the numerics: builds and diagonalizes the CvS Galerkin matrix at high precision and extracts Riemann zeros to hundreds of matching digits. | [`1_high_precision_riemann_zeros/`](1_high_precision_riemann_zeros/) - the code is the [`connes-cvs`](../README.md) package | [arXiv:2605.20224](https://arxiv.org/abs/2605.20224)<br>[10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) |
| **2** | [**A finite Guinand-Weil dictionary and archimedean tail order for the truncated Weil quadratic form**](https://arxiv.org/abs/2607.02828) - the structure: an exact finite zero-source dictionary + a finite-cutoff archimedean tail-order theorem with a two-sided certification rule. | [`2_guinand_weil_dictionary_tail_order/`](2_guinand_weil_dictionary_tail_order/) | [arXiv:2607.02828](https://arxiv.org/abs/2607.02828)<br>[10.5281/zenodo.21124802](https://doi.org/10.5281/zenodo.21124802) |
| **3** | **A matrix-valued von Mangoldt measure in the finite Connes–van Suijlekom path** - the arithmetic: realizes the prime side of the explicit formula as an exact matrix-valued von Mangoldt measure, with arithmetic rigidity, a finite source-to-jet dictionary, and a sharp finite vanishing-moment ceiling at the prime edge (an uncertainty-principle interpretation in the band-limited sense). | [`3_matrix_von_mangoldt_measure/`](3_matrix_von_mangoldt_measure/) | Zenodo [10.5281/zenodo.21242028](https://doi.org/10.5281/zenodo.21242028)<br>(concept DOI; resolves to the<br>current version) |
| **4** | **Bulk trace laws and algebraic boundary interaction for the sine Loewner operator** - uniform trace laws, finite-matrix comparison, half-line boundary flow and algebraic splitting, with three interval certificates. | [`4_sine_loewner_bulk_boundary/`](4_sine_loewner_bulk_boundary/) | Zenodo [10.5281/zenodo.22662221](https://doi.org/10.5281/zenodo.22662221)<br>(concept DOI) |

## Which manuscript each folder holds

**Every folder holds the current Zenodo deposit's manuscript, byte-identical to it.** That is the
rule for all four papers, without exception. Each paper's Zenodo concept DOI is therefore the
reference for what that paper currently says; an arXiv version may briefly lag a new deposit while
arXiv processes a replacement. When a new version is deposited, refresh the PDF here in the same
pass, so this rule keeps holding.

## Layout

Papers 2 and 3 are self-contained theorem notes; each folder holds the compiled PDF plus a
uniform reproducibility package:

```text
<paper>/
├── README.md · VERIFICATION.md · requirements.txt · LICENSE-PAPER-CC-BY-4.0.txt · SHA256SUMS
├── <paper>.pdf        the compiled manuscript
├── source/            LaTeX source (main.tex, .bib, .bbl, .bst)
├── figures/           figure PDFs + their generator
├── scripts/           reproducibility guards
└── artifacts/         guard outputs (JSON / logs)
```

The Guinand-Weil dictionary folder additionally includes `LICENSE` for its software and scripts; the
paper manuscripts in both folders use the explicitly named CC BY 4.0 license.

The Riemann-zeros paper is the exception: its reproducibility package is the `connes-cvs` package itself, so
[`1_high_precision_riemann_zeros/`](1_high_precision_riemann_zeros/) holds the manuscript and a
README pointing to the package (`../../connes_cvs/`), the `c = 100` data (`../../data/`), the
examples, and the tests at the repository root.

Paper 4 preserves its complete released source layout under [`4_sine_loewner_bulk_boundary/source/`](4_sine_loewner_bulk_boundary/source/):

```text
4_sine_loewner_bulk_boundary/
├── README.md · CITATION.bib · SHA256SUMS
├── paper.pdf         the complete manuscript (88 pages, three figures)
└── source/           all 64 released source files, with their relative paths preserved
    ├── main.tex · references.tex · SOURCE_README.md · LICENSES.md
    ├── sections/     all mathematical statements and proofs
    ├── figures/      TikZ inputs
    └── anc/          23 verification programs, their helper, exact witnesses and requirements
```

Its [build instructions](4_sine_loewner_bulk_boundary/source/SOURCE_README.md) and
[verification guide](4_sine_loewner_bulk_boundary/source/anc/README.md) are self-contained.
The 23 checks comprise 18 floating-point diagnostics, two exact rational scalar checks
and three outward-rounded interval certificates. The proofs are in the manuscript;
there is no separate mathematical supplement. The repository's package CI does not run
these paper-specific checks. Paper files are excluded from the PyPI wheel and source
distribution. The GitHub folder keeps unpacked sources and one PDF; the source ZIP is
available from Zenodo.
