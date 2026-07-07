[**← `connes-cvs`**](../../README.md) · [**Papers**](../README.md) &nbsp;|&nbsp; [Paper 1](../1_high_precision_riemann_zeros/) · [Paper 2](../2_guinand_weil_dictionary_tail_order/) · **Paper 3**

<div align="center">

# A matrix-valued von Mangoldt measure<br>in the finite Connes–van Suijlekom path

**Paper 3 — _the arithmetic_ · Akiva Groskin, 2026**

[![Zenodo DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21242028-1682D4.svg?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.21242028)
![arXiv](https://img.shields.io/badge/arXiv-pending-lightgrey.svg)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](LICENSE-PAPER-CC-BY-4.0.txt)

</div>

> Realizes the prime side of the Weil–Guinand explicit formula as an exact, cutoff-free
> **matrix-valued von Mangoldt measure** on the finite Connes–van Suijlekom path, and proves
> arithmetic rigidity, a finite source-to-jet dictionary, and a finite prime-edge uncertainty
> principle. Every statement is finite and exact; it makes no claim regarding the Riemann Hypothesis.

Part of the [`connes-cvs` series](../../README.md#papers): [**Paper 1** — the numerics](../1_high_precision_riemann_zeros/) (the [`connes-cvs`](../../README.md) package) · [**Paper 2** — the structure](../2_guinand_weil_dictionary_tail_order/) · **Paper 3 — the arithmetic**. Archived on Zenodo, concept DOI [10.5281/zenodo.21242028](https://doi.org/10.5281/zenodo.21242028) (resolves to the latest version); arXiv id pending.

Fix a Galerkin level `N` in the finite Connes-van Suijlekom truncation of the Weil
quadratic form (no archimedean cutoff), and vary the prime cutoff `u = log c`.
Differentiating the finite matrix path `u -> Q_N(u)` across a prime-power threshold
`u = log q` returns the von Mangoldt weight exactly, in every matrix entry: the
first-derivative jump is `-2 Lambda(q)/(sqrt(q) log q)` times the all-ones rank-one
matrix. The paper proves the event is arithmetically rigid, develops the finite
source-to-jet dictionary (confluent Vandermonde, sharp `2N+1` window, universal
recurrence), a finite prime-edge uncertainty principle, a coincidence-resolvent
generating identity, and a Krein-string boundary-mass reading. It proves no positivity,
no Riemann Hypothesis, and no prime-counting, next-prime, or factoring statement.

## Layout

```text
matrix_valued_von_Mangoldt_measure_finite_CvS_path.pdf   the paper (15 pp)
README.md                                                this file
VERIFICATION.md                                          what each guard checks
requirements.txt                                         optional dependencies
LICENSE-PAPER-CC-BY-4.0.txt                              license (CC BY 4.0)
SHA256SUMS                                               checksums for every file
source/      main.tex, main.bib, main.bbl, plainurl.bst  (LaTeX source)
figures/     fig_*.pdf + make_figures.py                 (the three figures + generator)
scripts/     check_*.py                                  (13 reproducibility guards)
artifacts/   *_audit.json                                (13 guard outputs, status PASS)
```

## Reproduce

The thirteen guards are independent checks (exact symbolic, exact modular over prime
fields to `N=1000`, and a floating-point check at the canonical scale `N=200`). Ten use
only the Python 3 standard library; three use `sympy`/`numpy` (see `requirements.txt`).
Each prints a JSON status and exits non-zero on failure. See `VERIFICATION.md` for the
guard-to-theorem map.

```bash
for s in scripts/check_*.py; do python3 "$s"; done
```

## Build the paper

```bash
cd source && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
(The three figure PDFs are in `figures/`; copy them next to `main.tex`, or regenerate
with `python3 figures/make_figures.py`.)

## License

Manuscript and figures: CC BY 4.0. The verification scripts may be used freely for
reproduction.
