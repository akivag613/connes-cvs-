[**← `connes-cvs`**](../README.md) &nbsp;|&nbsp; **Papers**

<div align="center">

# The `connes-cvs` paper series

**Three papers by Akiva Groskin on the truncated Weil quadratic form of Connes–van Suijlekom**

[![Paper 1 · arXiv](https://img.shields.io/badge/Paper_1-arXiv%3A2605.20224-b31b1b.svg)](https://arxiv.org/abs/2605.20224)
[![Paper 2 · arXiv](https://img.shields.io/badge/Paper_2-arXiv%3A2607.02828-b31b1b.svg)](https://arxiv.org/abs/2607.02828)
[![Paper 3 · DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21242028.svg)](https://doi.org/10.5281/zenodo.21242028)

</div>

The three papers form one series; the [`connes-cvs`](../README.md) package (repository root) is the
finite Galerkin implementation the series studies. None of the papers claims a proof of the
Riemann Hypothesis.

| | Paper | Home | Preprint / DOI |
| :--- | :--- | :--- | :--- |
| **1** | **High-Precision Approximation of Riemann Zeros via the Truncated Weil Form** — the numerics: builds and diagonalizes the CvS Galerkin matrix at high precision and extracts Riemann zeros to hundreds of matching digits. | [`1_high_precision_riemann_zeros/`](1_high_precision_riemann_zeros/) — the code is the [`connes-cvs`](../README.md) package | [arXiv:2605.20224](https://arxiv.org/abs/2605.20224)<br>[10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) |
| **2** | **A finite Guinand-Weil dictionary and archimedean tail order for the truncated Weil quadratic form** — the structure: an exact finite zero-source dictionary + a finite-cutoff archimedean tail-order theorem with a two-sided certification rule. | [`2_guinand_weil_dictionary_tail_order/`](2_guinand_weil_dictionary_tail_order/) | [arXiv:2607.02828](https://arxiv.org/abs/2607.02828)<br>[10.5281/zenodo.21124802](https://doi.org/10.5281/zenodo.21124802) |
| **3** | **A matrix-valued von Mangoldt measure in the finite Connes–van Suijlekom path** — the arithmetic: realizes the prime side of the explicit formula as an exact matrix-valued von Mangoldt measure, with arithmetic rigidity, a finite source-to-jet dictionary, and a prime-edge uncertainty principle. | [`3_matrix_von_mangoldt_measure/`](3_matrix_von_mangoldt_measure/) | Zenodo [10.5281/zenodo.21242028](https://doi.org/10.5281/zenodo.21242028)<br>_arXiv pending_ |

## Layout

Papers 2 and 3 are self-contained theorem notes; each folder holds the compiled PDF plus a
uniform reproducibility package:

```text
<paper>/
├── README.md · VERIFICATION.md · requirements.txt · LICENSE · SHA256SUMS
├── <paper>.pdf        the compiled manuscript
├── source/            LaTeX source (main.tex, .bib, .bbl, .bst)
├── figures/           figure PDFs + their generator
├── scripts/           reproducibility guards
└── artifacts/         guard outputs (JSON / logs)
```

Paper 1 is the exception: its reproducibility package is the `connes-cvs` package itself, so
[`1_high_precision_riemann_zeros/`](1_high_precision_riemann_zeros/) holds the manuscript and a
README pointing to the package (`../../connes_cvs/`), the `c = 100` data (`../../data/`), the
examples, and the tests at the repository root.
