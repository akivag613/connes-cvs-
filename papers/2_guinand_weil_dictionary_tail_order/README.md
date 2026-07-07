[**← `connes-cvs`**](../../README.md) · [**Papers**](../README.md) &nbsp;|&nbsp; [Paper 1](../1_high_precision_riemann_zeros/) · **Paper 2** · [Paper 3](../3_matrix_von_mangoldt_measure/)

<div align="center">

# A finite Guinand-Weil dictionary and archimedean tail order<br>for the truncated Weil quadratic form

**Paper 2 — _the structure_ · Akiva Groskin, 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2607.02828-b31b1b.svg)](https://arxiv.org/abs/2607.02828)
[![Zenodo DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21124802-1682D4.svg?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.21124802)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](LICENSE-PAPER-CC-BY-4.0.txt)

</div>

> An exact finite **Guinand-Weil zero-source dictionary** for the truncated Weil form: every value
> of the quadratic form is an exact sum over the nontrivial zeros of ζ. Plus a finite-cutoff
> **archimedean tail-order theorem** with a two-sided certification rule. No claim regarding the
> Riemann Hypothesis.

Part of the [`connes-cvs` series](../../README.md#papers): [**Paper 1** — the numerics](../1_high_precision_riemann_zeros/) (the [`connes-cvs`](../../README.md) package) · **Paper 2 — the structure** · [**Paper 3** — the arithmetic](../3_matrix_von_mangoldt_measure/). Published on arXiv, [arXiv:2607.02828](https://arxiv.org/abs/2607.02828) (math.NT, math.SP), and archived on Zenodo, concept DOI [10.5281/zenodo.21124802](https://doi.org/10.5281/zenodo.21124802) (resolves to the latest version).

## What this paper proves

For a real even finite Connes-van Suijlekom / Connes-Consani-Moscovici Galerkin coefficient
vector at cutoff `c` and band `N`:

1. an **exact finite Guinand-Weil zero-source dictionary** (Theorem 2.5): every value of the
   truncated Weil quadratic form is an exact sum over the nontrivial zeros of the Riemann zeta
   function, via a band-limited Guinand-Weil test function, with an entry-identification lemma
   pinning the source assembly to the CCM closed forms at equation level (Lemma 2.1);
2. an exact finite **source quotient** of dimension `2N+1`, stated as an iff (Corollary 2.4),
   and a positive-dimensional non-collapsing **pole-neutral** subfamily (Corollary 2.7);
3. a finite-`T` **archimedean tail-order theorem** (Theorem 3.2): past the Galerkin band the
   omitted archimedean tail is a strictly positive definite, strictly totally positive
   Cauchy-Stieltjes increment;
4. a **two-sided certification rule** with an explicit closed-form budget and asymptotic
   `B_T = (2N+1) rho (log(T/2pi)+1)/(pi^2 T)(1+o(1))` (Corollary 3.3), plus a self-contained
   `h_+` envelope lemma (Lemma 3.1).

The manuscript includes a worked example verified against the first 512 nontrivial zeros of
zeta and an eigenvalue-flow demonstration of the certification rule. It makes no claim
regarding the Riemann Hypothesis, Weil positivity, or a prime-location bound.

## Layout

```text
finite_guinand_weil_dictionary_tail_order.pdf   the paper (15 pp)
README.md                                       this file
VERIFICATION.md                                 what each guard checks
requirements.txt                                dependencies
LICENSE-PAPER-CC-BY-4.0.txt · LICENSE           licenses (paper CC-BY-4.0; scripts MIT)
SHA256SUMS                                      checksums for every file
source/      main.tex, main.bib, main.bbl, plainurl.bst   (LaTeX source)
figures/     fig_dictionary.pdf, fig_tailorder.pdf + make_figures.py
scripts/     verification guards (exact symbolic/integer + Arb interval)
artifacts/   guard outputs (JSON) and the 9000-bit certificate log
audit/       CLAIM_TRACE_AUDIT.md, NOVELTY_BOUNDARY_AUDIT.md, MANIFEST_RELEASE.txt
```

The `scripts/` guards, grouped by result:

- **Dictionary / source quotient** — `audit_exact_series_identity.py` (single-frequency
  identity), `audit_kernel_span_rank.py` (finite Volterra-kernel span), `audit_full_matrix_source_quotient.py`
  (factorization through the `2N+1` quotient), `audit_pole_neutral_survival.py` (pole-square
  factorization + dimension formula), `verify_finite_dictionary.py` (single-frequency source
  identity + pole normalization by three routes), `verify_dictionary_threeroute.py` (three-route
  dictionary confirmation over the first 512 zeros), `verify_zero_side.py` (original `c=13,N=4`
  zero-side check, retained for continuity).
- **Archimedean tail order** — `verify_arch_tail_order.py` (tail-order algebra + strict total
  positivity), `audit_arch_tail_dt_bridge.py` (rank-two Cauchy density = finite-`T` derivative),
  `arch_tail_budget.py` (Arb interval budget at `c=100, N=200, T=800`), `arch_tail_stress_ladder.py`
  (interval stress ladder across `T, N`, precision), `arch_tail_exact_asymptotic.py` (exact `B_T`
  vs the closed asymptotic; the `B_T = 1e-59` solve).
- **Cutoff-free inertia certificate** — `arb_ldlt_certify.py` (rigorous Arb interval `LDL^T`
  inertia certificate; generator of the 9000-bit `c=100, N=200` certificate).

## Reproduce

`mpmath` suffices for the exact/symbolic guards; the four Arb interval scripts
(`verify_arch_tail_order.py`, `arch_tail_budget.py`, `arch_tail_stress_ladder.py`,
`arb_ldlt_certify.py`) need `python-flint`, and `figures/make_figures.py` needs `matplotlib`
(see `requirements.txt`).

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/audit_exact_series_identity.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/audit_kernel_span_rank.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/audit_full_matrix_source_quotient.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/audit_pole_neutral_survival.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_finite_dictionary.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/audit_arch_tail_dt_bridge.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_zero_side.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_dictionary_threeroute.py 512
PYTHONDONTWRITEBYTECODE=1 python3 scripts/arch_tail_exact_asymptotic.py
# with python-flint:
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_arch_tail_order.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/arch_tail_budget.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/arch_tail_stress_ladder.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/arb_ldlt_certify.py --selftest --c 13 --N 8 --prec 300
```

The headline 9000-bit certificate is regenerated (about 15 minutes on an Apple M2 Max core) by

```bash
python3 scripts/arb_ldlt_certify.py --selftest --c 100 --N 200 --prec 9000 \
        --json-out artifacts/c100_N200_arb_ldlt_prec9000_provenance.json
```

See `VERIFICATION.md` for the guard-to-theorem map, and `audit/CLAIM_TRACE_AUDIT.md` for the
claim-to-artifact trace.

## Build the paper

The compiled PDF is provided at the top level. To rebuild, place the figure PDFs from
`figures/` next to `source/main.tex` (or regenerate them with `python3 figures/make_figures.py`),
then:

```bash
cd source && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## License

Manuscript and figures: [CC BY 4.0](LICENSE-PAPER-CC-BY-4.0.txt). Verification scripts:
[MIT](LICENSE). Checksums for every file are in `SHA256SUMS`.
