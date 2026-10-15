<div align="center">

# connes-cvs

### Open-source arbitrary-precision construction and validation of the Connes–van Suijlekom Galerkin matrix.

[![PyPI version](https://img.shields.io/pypi/v/connes-cvs.svg?color=4c1&cacheSeconds=300)](https://pypi.org/project/connes-cvs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/akivag613/connes-cvs-/blob/main/LICENSE)
[![Tests](https://github.com/akivag613/connes-cvs-/actions/workflows/tests.yml/badge.svg)](https://github.com/akivag613/connes-cvs-/actions/workflows/tests.yml)
[![Riemann zeros · arXiv](https://img.shields.io/badge/Riemann_zeros-arXiv%3A2605.20224-b31b1b.svg)](https://arxiv.org/abs/2605.20224)
[![Guinand-Weil dictionary · arXiv](https://img.shields.io/badge/Guinand--Weil_dictionary-arXiv%3A2607.02828-b31b1b.svg)](https://arxiv.org/abs/2607.02828)
[![von Mangoldt measure · Zenodo](https://img.shields.io/badge/von_Mangoldt_measure-Zenodo%3A10.5281%2Fzenodo.21242028-1682D4.svg?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.21242028)
[![Sine Loewner operator · Zenodo](https://img.shields.io/badge/Sine_Loewner_operator-Zenodo%3A10.5281%2Fzenodo.22662221-1682D4.svg?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.22662221)

</div>

> Connes & van Suijlekom (2025) proposed a spectral route to the Riemann Hypothesis through a truncated Weil quadratic form. This package builds and diagonalizes its finite Galerkin matrices at arbitrary precision. The published study follows the smallest-positive even-sector branch across **275 orders of magnitude**, from $\sim 10^{-59}$ at $c = 13$, $N = 100$, $T = 800$, $\mathrm{dps} = 150$ to $\sim 10^{-334}$ at $c = 100$, $N = 250$, $T = 800$, $\mathrm{dps} = 500$, and reports **329 matching digits** for $\gamma_1$ at the latter cell. These are finite-cutoff numerical results, not a proof of the Riemann Hypothesis.

---

## Papers

This repository hosts the `connes-cvs` package together with four papers by **Akiva Groskin** on the truncated Weil form of Connes–van Suijlekom and associated finite and continuum spectral problems. The package implements the finite Galerkin calculations of Paper 1; the other papers have their own reproducibility programs. Each paper's manuscript and reproducibility package lives in [`papers/`](https://github.com/akivag613/connes-cvs-/tree/main/papers) (see the [papers index](https://github.com/akivag613/connes-cvs-/blob/main/papers/README.md)).

| Paper | What it does | Links |
| :--- | :--- | :--- |
| [**Riemann zeros at high precision**](https://arxiv.org/abs/2605.20224)<br>_Paper 1 in the series · the numerics_ | **High-Precision Approximation of Riemann Zeros via the Truncated Weil Form.** Builds and diagonalizes the CvS Galerkin matrix at high precision: extracts Riemann zeros to hundreds of matching digits and compares the _c_ = 100 results with the prolate asymptotic discussed in Connes 2026 §6.4. **This is the paper the `connes-cvs` package implements** - its reproducibility package is the package itself (`connes_cvs/`, `data/`, `examples/`, `tests/`); see [ERRATA.md](https://github.com/akivag613/connes-cvs-/blob/main/ERRATA.md) for the recorded corrections, none of which changes a measured value. | [arXiv:2605.20224](https://arxiv.org/abs/2605.20224) (math.NT)<br>[`papers/1_.../`](https://github.com/akivag613/connes-cvs-/tree/main/papers/1_high_precision_riemann_zeros)<br>Zenodo [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) |
| [**The finite Guinand-Weil dictionary**](https://arxiv.org/abs/2607.02828)<br>_Paper 2 in the series · the structure_ | **A finite Guinand-Weil dictionary and archimedean tail order for the truncated Weil quadratic form.** An exact finite Guinand-Weil zero-source dictionary for the truncated Weil form, plus a finite-cutoff archimedean tail-order theorem with a two-sided certification rule. | [arXiv:2607.02828](https://arxiv.org/abs/2607.02828) (math.NT, math.SP)<br>[`papers/2_.../`](https://github.com/akivag613/connes-cvs-/tree/main/papers/2_guinand_weil_dictionary_tail_order)<br>Zenodo [10.5281/zenodo.21124802](https://doi.org/10.5281/zenodo.21124802) |
| [**The matrix-valued von Mangoldt measure**](https://github.com/akivag613/connes-cvs-/tree/main/papers/3_matrix_von_mangoldt_measure)<br>_Paper 3 in the series · the arithmetic_ | **A matrix-valued von Mangoldt measure in the finite Connes–van Suijlekom path.** The corrected v2 is the version of record. It realizes the prime-power side of the Weil–Guinand explicit formula as an exact, cutoff-free matrix-valued von Mangoldt measure on the finite path and proves finite arithmetic-rigidity and source-to-jet results; its statements are finite-dimensional and make no claim of proving RH. | [`papers/3_.../`](https://github.com/akivag613/connes-cvs-/tree/main/papers/3_matrix_von_mangoldt_measure)<br>Zenodo [10.5281/zenodo.21242028](https://doi.org/10.5281/zenodo.21242028) |
| [**The sine Loewner operator**](https://github.com/akivag613/connes-cvs-/tree/main/papers/4_sine_loewner_bulk_boundary)<br>_Paper 4 in the series · bulk and boundary spectra_ | **Bulk trace laws and algebraic boundary interaction for the sine Loewner operator.** Proves uniform trace laws for a continuum operator and its finite matrices, determines half-line boundary flow and algebraic interaction, and certifies a mode with nonzero splitting. The complete paper and its independent verification programs are included. | [`papers/4_.../`](https://github.com/akivag613/connes-cvs-/tree/main/papers/4_sine_loewner_bulk_boundary)<br>Zenodo [concept DOI 10.5281/zenodo.22662221](https://doi.org/10.5281/zenodo.22662221) |

The series combines high-precision numerical work, finite-dimensional theorems, and infinite-dimensional spectral results. None of the papers claims a proof of the Riemann Hypothesis.

<div align="center">

| Cutoff range | lambda_min span | gamma_1 accuracy | Cross-check |
| :---: | :---: | :---: | :---: |
| `c = 13 … 67` | `10⁻⁵⁹ → 10⁻¹⁷³` | up to **167 matching digits** (`c=67, N=100, dps=200`; error `1.478e-168`) | matches CCM 2025 at `c=14` to factor 3 |
| `c = 100` | `10⁻³³⁴` (`N=250, dps=500`) | **329 matching digits** (`N=250, dps=500`) | two consecutive Aitken-Δ² estimates approach the prolate comparison scale in Connes 2026 §6.4 (≈ −530.4); deeper triple within 3.32 OOM, ratios 0.8373 / 0.8355 |

</div>

---

## Table of contents

- [Papers](#papers)
- [Headline result](#headline-result)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The c = 100 verification](#the-c--100-verification)
- [Reproduce the published sweep](#reproduce-the-published-sweep)
- [Validation against published data](#validation-against-published-data)
- [Performance](#performance)
- [How it works](#how-it-works)
- [Further reading](#further-reading)
- [Citation](#citation)
- [References to this work](#references-to-this-work)
- [Contributing](#contributing)
- [License](#license)

---

## Headline result

**A numerical comparison at $c = 100$ with the prolate asymptotic in Connes 2026 §6.4.**

Connes 2026 (arXiv:2602.04022) §6.4 displays the Fuchs asymptotic for a prolate quantity:
$$1 - \chi_2(\lambda) \;\sim\; \frac{2^{14}}{3}\,\sqrt{2}\,\pi^{5}\; e^{-4\pi e^{L} + 9L/2}, \qquad L = 2\log\lambda,$$
Here $\chi_2$ is a prolate angular eigenvalue. Connes reports numerical similarity between $1-\chi_2$ and the full Weil spectral floor; the displayed asymptotic is for the prolate quantity, and does not establish their asymptotic identity. Its use as a comparison scale for the Weil floor is therefore heuristic. CCM 2025 §6 reports comparisons through $\lambda \leq 14$ with $N = 120$. This study evaluates a separate finite-$N$ sequence at $c=100$; “out-of-sample” refers to this paper's earlier $c\leq67$ fit window, not to the range of all published comparisons.

Using this package at $c = 100$ with $N \in \{100, 150, 200, 250\}$ at $\mathrm{dps} = 500$, two consecutive Aitken-Δ² extrapolations on the overlapping triples give
$$\log_{10}\bigl|\lambda_\infty^{\mathrm{even}}(c{=}100)\bigr| \;\approx\; -536.76 \;\;\text{and}\;\; -533.70,$$
approaching the prolate comparison scale of $\approx -530.38$ monotonically with $N$; the consecutive first-difference ratios `0.8373` and `0.8355` match to two decimal places, evidence for a local geometric model. The deeper-anchored triple sits **3.32 OOM** below that comparison scale, out of $|x_\infty| \sim 530$ - agreement at the under-1%-of-exponent level on the deeper anchor, out-of-sample (the in-sample fit window was $c \leq 67$ at $N = 100$). Four points do not distinguish this local geometric model from all alternative convergence laws.

**Companion observations** (full details in the paper):

- $\gamma_1$ through $\gamma_{10}$ extracted to **307–329 matching digits** at $c = 100$, $N = 250$, $\mathrm{dps} = 500$ (and **219–242** at $N = 150$, $\mathrm{dps} = 1000$).
- Under the unitary equivalence with CCM 2025 Lemma 5.1, every $\gamma_k$ extraction here is, modulo a hypothesis-status caveat at $c = 100$ documented in the paper, an eigenvalue of the rank-one perturbed scaling operator $D_{\log}^{(\lambda,N)}$ of CCM Theorem 1.1(iii) at $\lambda = \sqrt{c}$.
- The empirical fit $|\log_{10}\lambda_{\min}(c)| \approx 13.24 \, c^{0.634}$ valid on $c \leq 67$ at $N = 100$ is a fixed-$N$ empirical fit: the $c = 100$, $N = 200$ datum falsifies the pure-power-law extrapolation by 49 orders of magnitude.

The accompanying paper is on **arXiv** - [arXiv:2605.20224](https://arxiv.org/abs/2605.20224) (math.NT) - and archived on **Zenodo**, where the concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) always resolves to the latest version. The corrections dated through 2026-08-14 in [ERRATA.md](https://github.com/akivag613/connes-cvs-/blob/main/ERRATA.md) change no measured value: the $c=100$ and $L(s,\chi_3)$ negative-sign eigenvalue claims are finite-cutoff artifacts (2026-06-26, in the published manuscript text); the Section 8.2 Paley-Wiener mechanism is withdrawn and Table 14 is measured at $T=400$ (2026-08-12); the $c=67$ matching-digit count is 167 rather than 168 (2026-08-12); two summary digit-increment ranges in Section 6.5 are 93-117 and 181-203 (2026-08-13); and the $c=100$ negative block does not vanish at any finite cutoff tested, though a cutoff-free evaluation still leaves the even sector non-negative (2026-08-14). Those corrections are incorporated in the published manuscript. The later [Martini attribution update](ERRATA.md#2026-10-15---martini-replication-attribution-in-section-98) applies to Section 9.8 and remains an overlay pending a manuscript revision.

---

## Installation

```bash
pip install connes-cvs
```

For the optional compiled Arb digamma backend:

```bash
pip install 'connes-cvs[fast]'
```

To install from source (recommended for development):

```bash
git clone https://github.com/akivag613/connes-cvs-.git
cd connes-cvs-
pip install -e '.[all]'
```

### Requirements

- Python ≥ 3.10
- [mpmath](https://mpmath.org/) ≥ 1.3 (arbitrary-precision arithmetic)

### Optional dependencies

- [python-flint](https://github.com/flintlib/python-flint) - Arb-backed arbitrary-precision digamma; install the supported version selected by the `fast` extra
- [gmpy2](https://github.com/aleaxit/gmpy) ≥ 2.1 - GMP-backed mpmath core
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) - for downstream analysis

---

## Quick start

```python
from connes_cvs import build_galerkin_matrix, compute_ground_state, extract_zeros
import mpmath as mp

# Seconds-level API smoke cell (not a paper benchmark)
Q = build_galerkin_matrix(c=13, N=8, T=60, dps=30)

# Diagonalize
lam_min, eigvec = compute_ground_state(Q)
print(f"λ_min(c=13) = {mp.nstr(lam_min, 6)}")
# λ_min(c=13) ≈ 4.43043e-23

# Extract the first detected Riemann zero.
# Preferred form (v0.3.0): pass the cutoff c and the package computes
# L = log(c) internally at full working precision.
zeros = extract_zeros(eigvec, c=13, n_zeros=1, dps=30)
# Equivalent, still supported: extract_zeros(eigvec, L=mp.log(13), ...).
# Passing L as a Python float carries only ~16 digits and caps the
# extraction accuracy near 1e-16; v0.3.0 emits a UserWarning.
print(f"γ₁ detected = {mp.nstr(zeros[0]['gamma_detected'], 12)}")
print(f"|γ₁ error|  = {mp.nstr(zeros[0]['error'], 4)}")
# γ₁ detected ≈ 14.1347251417
# |γ₁ error|  ≈ 2.52738e-17
```

Values are printed with `mp.nstr` rather than an f-string format spec: `mpmath.mpf`
does not implement `__format__` on mpmath 1.3.0, the declared minimum, so
`f"{lam_min:.6e}"` raises `TypeError` there.

This smoke cell was measured at about 1.8 seconds with python-flint 0.8.0 and 4.8 seconds through the mpmath fallback on the release machine. Runtime varies with versions and hardware. It checks the API and root-extraction path; it is not the high-precision paper cell. Run `python examples/basic_compute.py --extended` for the longer `N=100`, `T=400`, `dps=80` validation example through the progress-reporting process runner.

<details>
<summary><b>Multi-cutoff sweep (click to expand)</b></summary>

`run_sweep` starts a `multiprocessing` pool, so in a script it belongs inside an
`if __name__ == "__main__":` guard: under the `spawn` start method (macOS, Windows)
each worker re-imports the module, and without the guard that re-executes the sweep
in every worker.

```python
from connes_cvs.sweep import run_sweep
import mpmath as mp

if __name__ == "__main__":
    results = run_sweep(
        cutoffs=[13, 17, 19, 23, 29],
        N=100, T=400, dps=80,
    )

    for c, r in results.items():
        print(f"c={c:2d}  λ_min = {mp.nstr(r['lambda_min'], 4)}"
              f"  |γ₁ err| = {mp.nstr(r['gamma1_error'], 4)}")
```

</details>

A runnable smoke/extended example is also available at [`examples/basic_compute.py`](https://github.com/akivag613/connes-cvs-/blob/main/examples/basic_compute.py).

---

## The c = 100 verification

The headline analysis is reproducible from the committed data. A minimal verification script that loads the published $N$-sweep and recomputes both Aitken-Δ² anchors and the prolate comparison scale in Connes 2026 §6.4 in under a second is at [`examples/c100_aitken_check.py`](https://github.com/akivag613/connes-cvs-/blob/main/examples/c100_aitken_check.py); the underlying data is in [`data/c100/`](https://github.com/akivag613/connes-cvs-/tree/main/data/c100).

> **Data provenance and reproducibility.** The $c = 100$ dataset in [`data/c100/`](https://github.com/akivag613/connes-cvs-/tree/main/data/c100) was generated by a local production runner built on the v0.2.2 mathematical kernels. The historical v0.1.0/v0.2.0 A/B cell at $c = 13$, $N = 80$ agrees in all 80 printed decimal digits; that check does not by itself establish raw arithmetic identity at $c = 100$. Version 0.3.0 separately tests exact entrywise agreement between its classic and runner paths on a small $c = 13$ cell and regression agreement on larger $c = 13$ cells. Claims below are therefore scoped to the recorded artifacts and explicit test cells.
>
> To reproduce a tabulated production cell, use `CellConfig(c=100, N=..., T=800, dps=..., flint_bits=4*dps)` and `GalerkinCell(..., ground_state="smallest_positive")`. The explicit `flint_bits` matters: the recorded artifacts used `4*dps`, whereas the package default preserves the historical `int(3.5*dps)` convention. At $c = 100$, $T = 800$, the raw finite-$T$ even-sector matrix contains negative-sign eigenvalues whose values and counts rearrange with $T$. No finite cutoff tested removed the whole block; cutoff-free interval certificates instead establish non-negativity for the tested even-sector blocks. [ERRATA.md](https://github.com/akivag613/connes-cvs-/blob/main/ERRATA.md) gives the exact scope and identifies the negatives as archimedean-cutoff artifacts. The table follows the empirically distinguished smallest-positive branch. `compute_ground_state` instead returns the raw minimum, so it is not the reproduction selector for these finite-$T$ rows. For the published $c \leq 67$ cells the minimum and smallest-positive selections coincide.

### N-sweep at c = 100, T = 800

| N | dps | lambda_min^even | log10 abs(lambda_min) | wall-clock |
| :---: | :---: | :---: | :---: | :---: |
| 100 | 500  | `1.22e-191` | `-190.92` | 13.9 min |
| 150 | 500  | `6.42e-248` | `-247.19` | ~21 min |
| 200 | 500  | `4.87e-295` | `-294.31` | 28.4 min |
| 250 | 500  | `2.08e-334` | `-333.68` | ~38 min |
| 150 | 1000 | `6.42e-248` | `-247.19` | ~111 min |

Recorded wall-clock from the JSON artifacts on a 12-worker Apple M-series machine. The $N = 150$ values at $\mathrm{dps}=500$ and $1000$ agree for 25 leading significant digits and then diverge; this is a cross-precision check of the $N=150$ cell only. It does not certify the separate $N=250$ cell.

### Aitken-Δ² extrapolation

The four-point sequence $x_N = \log_{10}|\lambda_N|$ at $c = 100$ admits Aitken-$\Delta^2$ acceleration on two overlapping triples:

```
x_inf(100,150,200) ~= -536.76        x_inf(150,200,250) ~= -533.70
```

The consecutive first-difference ratios $|\Delta_2/\Delta_1| = 0.8373$ and $|\Delta_3/\Delta_2| = 0.8355$ match to two decimal places, evidence for a local geometric model of the convergence sequence (not a 3-point forced fit).

The prolate asymptotic in Connes 2026 §6.4 gives the comparison scale at $c = 100$:
```
log10( 2^14 * sqrt(2) * pi^5 / 3 ) - (4*pi*100)/ln(10) + (9*log(100))/(2*ln(10))
    ~= 6.37 - 545.75 + 9.00 ~= -530.38
```

The two Aitken anchors sit 6.39 OOM and 3.32 OOM below this comparison scale respectively, out of a magnitude range $|x_\infty| \sim 530$, with the trend monotone in $N$ - agreement at the **under-1%-of-exponent level on the deeper anchor**, out-of-sample (the in-sample fit window was $c \leq 67$ at $N = 100$). Four points do not rule out alternative convergence-model fits; see the paper for the model-sensitivity discussion.

### γ_k extraction at c = 100

| k | N=150, dps=1000 | N=250, dps=500 | k | N=150, dps=1000 | N=250, dps=500 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 242 | **329** | 6 | 228 | 316 |
| 2 | 239 | 325 | 7 | 226 | 313 |
| 3 | 236 | 323 | 8 | 224 | 312 |
| 4 | 233 | 320 | 9 | 221 | 309 |
| 5 | 231 | 318 | 10 | 219 | 307 |

The canonical matching-digit count is $\left\lfloor-\log_{10}|\gamma_k^{\text{detected}}-\gamma_k^{\text{exact}}|\right\rfloor$. Thus an error of $1.478\times10^{-168}$ counts as **167** matching digits. Earlier manuscript versions labelled that same $c=67$ result “168 digits,” using the exponent/error-decade convention; the current manuscript gives 167. [ERRATA.md](https://github.com/akivag613/connes-cvs-/blob/main/ERRATA.md) records the correction to 167, and the numerical error itself is unchanged. The reference is `mpmath.zetazero(k).imag` at `dps=400`. For comparison, CCM 2025 §6 reports $\gamma_1$ matching to approximately 55 digits at $c = 13$, $N = 120$.

---

## Reproduce the published sweep

To replicate the 15-cutoff sweep at $c \in \{13, 14, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67\}$ - 113 orders of magnitude in $|\gamma_1\,\mathrm{err}|$:

```python
from connes_cvs.sweep import run_sweep
import json

LOW_DPS = [13, 14, 17, 19, 23, 29, 31, 37]
HIGH_DPS = [41, 43, 47, 53, 59, 61, 67]

# The multiprocessing pool inside run_sweep requires the __main__ guard
# whenever this is saved as a script (spawn start method: macOS, Windows).
if __name__ == "__main__":
    results = run_sweep(LOW_DPS, N=100, T=800, dps=150)
    results.update(
        run_sweep(HIGH_DPS, N=100, T=800, dps=200)
    )

    with open("my_sweep.json", "w") as f:
        json.dump(
            {str(c): {"lambda_min":   str(r["lambda_min"]),
                      "gamma1_error": str(r["gamma1_error"]),
                      "wall_time":    r["wall_time"]}
             for c, r in results.items()},
            f, indent=2,
        )
```

Wall-clock is platform- and backend-dependent. Compare the resulting decimal values against the published [`data/results_15pt_T800.json`](https://github.com/akivag613/connes-cvs-/blob/main/data/results_15pt_T800.json) at the precision carried by that file; this recipe does not claim raw arithmetic equality across package or dependency versions.

### Convergence at a glance

```text
c=13   ████████                                                              -55
c=14   █████████                                                             -60
c=17   ███████████                                                           -76
c=19   █████████████                                                         -86
c=23   ███████████████                                                      -102
c=29   ██████████████████                                                   -119
c=31   ██████████████████                                                   -124
c=37   ████████████████████                                                 -135
c=41   █████████████████████                                                -142
c=43   █████████████████████                                                -144
c=47   ██████████████████████                                               -149
c=53   ███████████████████████                                              -156
c=59   ████████████████████████                                             -161
c=61   ████████████████████████                                             -163
c=67   █████████████████████████                                            -168
c=100  ███████████████████████████████████████████████  (N=250, dps=500)  -330
                                                                          log₁₀|γ₁ err|
```

Rows $c \leq 67$ use $N = 100$; the $c = 100$ row uses $N = 250$, $\mathrm{dps} = 500$ (headline cell). These are finite-matrix zero-extraction errors. Because both $c$ and $N$ change in the last row, this chart does not isolate a continuum decay law; the prolate comparison above concerns a different quantity, the spectral floor.

---

## Validation against published data

Independent cross-checks of this package against published values. The $c = 13$ and $c = 14$ rows compare the **first-zero error** $\lvert\gamma_1 - t_1\rvert$ (which is orders of magnitude larger than $\lambda_{\min}$ itself); the $c = 100$ row compares the **smallest-eigenvalue decay** $\log_{10}\lvert\varepsilon\rvert$ against the prolate comparison scale in Connes 2026 §6.4. The two quantities are distinct - do not read the $\sim 10^{-55}$ values as $\lambda_{\min}$.

| Cutoff | Quantity | Published | This package | Agreement |
| :---: | :---: | :---: | :---: | :--- |
| $c = 13$ | $\lvert\gamma_1\text{ err}\rvert$ | `2.6e-55` (Connes 2026 §6) | $\mathbf{2.005 \times 10^{-55}}$ | factor 1.3 |
| $c = 13$ | $\lvert\gamma_1\text{ err}\rvert$ | `2.44e-55` (CCM 2025 §6, $N=120$, 200-digit) | $\mathbf{2.005 \times 10^{-55}}$ | factor 1.2 |
| $c = 14$ | $\lvert\gamma_1\text{ err}\rvert$ | `1.07e-60` (CCM 2025 §6) | $\mathbf{3.541 \times 10^{-61}}$ | factor 3 |
| $c = 100$ | $\log_{10}\lvert\varepsilon\rvert$ | $\approx -530.38$ (Connes 2026 §6.4, prolate comparison) | two Aitken-Δ² anchors at $\mathbf{-536.76}$ and $\mathbf{-533.70}$ | 3.32 OOM (deeper anchor); under 1% of exponent |

Cells for the "This package" column: the $c = 13$ and $c = 14$ rows are the published sweep cells at $N = 100$, $T = 800$, $\mathrm{dps} = 150$ ([`data/results_15pt_T800.json`](https://github.com/akivag613/connes-cvs-/blob/main/data/results_15pt_T800.json)); the $c = 100$ row is the Aitken-Δ² pair over $N \in \{100, 150, 200, 250\}$ at $T = 800$, $\mathrm{dps} = 500$.

Our matrix rows use the truncated Weil form $Q(c)$ in the **trigonometric basis** and report the different quantities identified above. The last row compares its extrapolated floor with a separate prolate quantity; this does not assert operator identity. The factor-of-1.3 spread at $c = 13$ is consistent with the differing $N$, $T$, precision and normalization conventions; it is not, by itself, a proof of cross-implementation identity.

### Independent verification by third parties

Independent reproductions, software comparisons and papers citing this work are collected in [Selected references to this work](REFERENCES_TO_THIS_WORK.md). The page identifies the source version and distinguishes independent implementations from reruns of this package.

Among the reported checks are Bradley Martin's separate implementation, Ronnie Andrews's reproduction study, Mohamed Osman's numerical comparisons, and karl-keysingularity's Windows validation artifact. Ariel Fernando Martini's revised replication reports a first-zero error of approximately $4.2 \times 10^{-55}$ at $c=13$, $N=60$, $T=400$, $\mathrm{dps}=80$, using `connes-cvs` v0.3.1 ([concept DOI](https://doi.org/10.5281/zenodo.21864191); reviewed [v4](https://zenodo.org/records/23393038), Table 2). The correction to our earlier attribution is recorded in [ERRATA.md](ERRATA.md#2026-10-15---martini-replication-attribution-in-section-98).

We also acknowledge Julius Stricker's independently executed reruns of the Paper 2 interval certificates, using the original certifier at revision `0675989`; the original report is no longer publicly available.

**Spectral-triple interpretation (CCM 2025 Lemma 5.1 + Theorem 1.1(iii)).** Under the unitary equivalence of $Q(c)$ with the CCM matrix $\tau_{i,j}$, the $F_{\mathrm{even}}$ test function used by this package's `extract_zeros` is, up to a positive scaling constant and the change of variable $u = e^x$, the same finite Fourier–Mellin transform $\widehat{\xi}_N(z)$ appearing in CCM 2025. Roots located near known zeta ordinates are accordingly eigenvalues of the associated finite rank-one perturbed scaling operator. At $c=100$, this interpretation is applied to the empirically selected smallest-positive finite-$T$ branch described above, not to the raw minimum of the truncated matrix.

**Coverage.** The public dataset contains the thirteen additional sweep cutoffs $c \in \{17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67\}$ and the separate $c=100$ study. The linked references identify the cutoffs tested by each reported reproduction; no blanket priority claim is made for the remaining rows.

Full dataset for the 15-cutoff sweep in [`data/results_15pt_T800.json`](https://github.com/akivag613/connes-cvs-/blob/main/data/results_15pt_T800.json).

---

## Performance

The historical April 2026 same-environment A/B record reports v0.2.0 as **2.06× faster** on the psi-cache phase and **1.83× faster** end-to-end than v0.1.0 at the stated cell. The two runs agree in all 80 printed decimal digits of $\lambda_{\min}$; the record does not establish raw-value equality beyond that rendering.

### A/B test ($c=13$, $N=80$, $T=400$, $\mathrm{dps}=80$, 12-way Pool)

| Phase | v0.1.0 | v0.2.0 | Speedup |
| :--- | ---: | ---: | :---: |
| Archimedean integral (cache) | 57.55 s | **27.94 s** | **2.06×** |
| Matrix assembly | 0.11 s | 0.12 s | unchanged |
| Symmetric eigensolver | 6.11 s | 6.19 s | unchanged |
| Root extraction | 1.16 s | 1.15 s | unchanged |
| **Total wall time** | **64.94 s** | **35.40 s** | **1.83×** |
| $\lambda_{\min}$ | `2.52826614019657560…e-59` | `2.52826614019657560…e-59` | **all 80 printed digits agree** |

### Published reference workload ($c=13$, $N=100$, $T=800$, $\mathrm{dps}=150$)

| | v0.1.0 | v0.2.0 |
| :--- | ---: | ---: |
| Wall time | 214.8 s | **127.3 s** (**1.69× faster**) |
| $\lambda_{\min}$ | `2.8654536149302802951…e-59` | `2.8654536149302802951…e-59` |

See [`benchmarks/AB_VERIFIED_2026-04-14.md`](https://github.com/akivag613/connes-cvs-/blob/main/benchmarks/AB_VERIFIED_2026-04-14.md) for the historical A/B protocol and recorded summary. The raw console logs are not distributed.

### Validated runner (v0.3.0)

Version 0.3.0 adds a resumable production runner with explicit precision, lossless mpmath transport, atomic checkpoints, progress reporting and environment fingerprints. It deliberately does not choose precision automatically: validate a chosen `dps` and `flint_bits` through independent reference or cross-precision runs. The arithmetic mutates process-global mpmath/Flint contexts and module caches, so do not run cells at different precisions concurrently in Python threads; use the process-based runner or sweep. Checkpoint hashes detect accidental corruption, not malicious modification: resume only trusted local checkpoint files. File locks provide a single-writer guard, not an authentication boundary.

Runner timing fields measure the compute pipeline through diagonalization; they
exclude artifact hashing, JSON serialization and optional disk-write overhead.
`run_sweep` adds zero-extraction time but retains that same boundary. Use an
external monotonic timer when measuring the complete API call.

The runner starts a `multiprocessing` pool, so a script must protect the call under the `spawn` start method used by macOS and Windows:

```python
from connes_cvs.runner import CellConfig, run_cell

if __name__ == "__main__":
    artifact = run_cell(CellConfig(c=13, N=100, T=400, dps=80))
    print(artifact["lambda_even"][:24],
          artifact["timings_seconds"]["total_s"])
```

`connes_cvs.validation.arb_eigenpair_residual_bound(Q, v, lam)` gives an outward Arb residual bound for an exact supplied finite symmetric mpmath matrix. Its scope is that finite matrix only; it is not a truncation-error or infinite-operator certificate. Exact classic/runner equality is tested on a small $c=13$ cell, while the slow gate compares two larger $c=13$ cells against committed decimal references. No v0.3 speedup multiplier is claimed without a matched before/after benchmark.

---

## How it works

The truncated Weil quadratic form decomposes into three arithmetically transparent pieces:

$$
Q(c) = D_\infty + D_{\text{pole}} + D_{\text{prime}}
$$

- $D_\infty$ - archimedean Mellin multiplier $h_+(\tau) = \mathrm{Re}\,\psi(\tfrac{1}{4} + i\tfrac{\tau}{2}) - \log\pi$
- $D_{\text{pole}}$ - rank-one correction from the pole of $\zeta(s)$ at $s=1$
- $D_{\text{prime}}$ - finite von-Mangoldt sum over prime powers $q=p^a \leq c$

The Galerkin matrix entries are
$$
q_{m,n} = \frac{\psi(m) - \psi(n)}{m - n}, \qquad q_{n,n} = \psi'(n),
$$
where $\psi(x) = \tfrac{1}{\pi} \int_0^L \sin\bigl(2\pi x(1-y/L)\bigr)\, D(y)\, dy$ and $L = \log c$.

The bottleneck is the archimedean integral: evaluating the digamma function at thousands of adaptive quadrature nodes per basis index. The historical v0.2.0 implementation evaluated all $2N{+}1$ indices and exploited two observations to reduce that cost:

1. **$h_+$ is even in $\tau$** and mpmath's tanh-sinh rule is deterministic per `(interval, precision)`, so `psi_arch` and `psi_arch_deriv` share quadrature nodes. A dict keyed on $|\tau|$ gives a 4× hit rate on digamma calls.
2. **A fused real-arithmetic kernel** computes $\mathrm{Re}\,\hat{S}_x(\tau)$ and $\mathrm{Re}\,\partial_x \hat{S}_x(\tau)$ in one pass, sharing $\sin(\beta L)$, $\sin(\beta L / 2)$, $1/\beta$, and related sub-expressions.

Version 0.3.0 additionally uses the exact index parities $\psi(-n)=-\psi(n)$ and $\psi'(-n)=\psi'(n)$ to evaluate $N+1$ nonnegative indices and mirror them; direct positive/negative and legacy/full-assembly comparisons are covered by the hardening tests.

Precision management is explicit. Eigenvalues shrink super-exponentially ($\lambda_{\min} \sim 10^{-173}$ at $c = 67$, $N = 100$, $T = 800$, $\mathrm{dps} = 200$; $\sim 10^{-248}$ at $c = 100$, $N = 150$, $T = 800$, $\mathrm{dps} = 500$). The published 15-cutoff sweep uses 150–200 mpmath decimal digits; the $c = 100$ dataset includes cells at 500 and 1000 digits and records `flint_prec=4*dps`.

---

## Further reading

- **The paper this package implements** ([folder](https://github.com/akivag613/connes-cvs-/tree/main/papers/1_high_precision_riemann_zeros)) - Groskin 2026, *High-Precision Approximation of Riemann Zeros via the Truncated Weil Form*, [arXiv:2605.20224](https://arxiv.org/abs/2605.20224) (math.NT). Archived on Zenodo; the concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) always resolves to the latest version, which carries the current manuscript and the current [ERRATA.md](https://github.com/akivag613/connes-cvs-/blob/main/ERRATA.md). The reproducibility package is this repository.
- **Companion note on the structure** ([folder](https://github.com/akivag613/connes-cvs-/tree/main/papers/2_guinand_weil_dictionary_tail_order)) - Groskin 2026, *A finite Guinand-Weil dictionary and archimedean tail order for the truncated Weil quadratic form*, [arXiv:2607.02828](https://arxiv.org/abs/2607.02828) (math.NT): an exact finite Guinand-Weil zero-source dictionary for the truncated Weil form, and a finite-cutoff archimedean tail-order theorem with a two-sided certification rule. Archived on Zenodo, concept DOI [10.5281/zenodo.21124802](https://doi.org/10.5281/zenodo.21124802), which always resolves to the latest version.
- **Companion note on the arithmetic** ([folder](https://github.com/akivag613/connes-cvs-/tree/main/papers/3_matrix_von_mangoldt_measure)) - Groskin 2026, *A matrix-valued von Mangoldt measure in the finite Connes–van Suijlekom path*. This paper is archived on Zenodo only; it is not on arXiv. The corrected v2 is the version of record: it adds the reciprocal-zero-set restriction, narrows pole and interpretation language, and replaces the unqualified variance claim with a corollary carrying an explicit pairwise-uncorrelatedness hypothesis while preserving the finite matrix-valued von Mangoldt construction. It makes no claim regarding RH. Archived on Zenodo, concept DOI [10.5281/zenodo.21242028](https://doi.org/10.5281/zenodo.21242028), which always resolves to the latest version.
- **Bulk and boundary spectra** ([folder](https://github.com/akivag613/connes-cvs-/tree/main/papers/4_sine_loewner_bulk_boundary)) - Groskin 2026, *Bulk trace laws and algebraic boundary interaction for the sine Loewner operator*. Uniform trace laws, finite-matrix comparison, half-line boundary flow and algebraic splitting, with three interval certificates. Archived on Zenodo, concept DOI [10.5281/zenodo.22662221](https://doi.org/10.5281/zenodo.22662221). Its verification programs are independent of the `connes-cvs` package.
- **CvS - mathematical foundation** - Connes & van Suijlekom, *Quadratic forms, real zeros and echoes of the spectral action*, [arXiv:2511.23257](https://arxiv.org/abs/2511.23257).
- **CCM - the rank-one spectral-triple construction whose spectrum this package measures** - Connes, Consani & Moscovici, *Zeta spectral triples*, [arXiv:2511.22755](https://arxiv.org/abs/2511.22755).
- **Connes 2026 - the §6.4 prolate asymptotic used for the $c = 100$ comparison** - *The Riemann Hypothesis: Past, Present and a Letter Through Time*, [arXiv:2602.04022](https://arxiv.org/abs/2602.04022).
- **Connes–Consani 2023 - qualitative motivation for the $k_\lambda$ approximation in Connes 2026 §6.6** - *Spectral triples and $\zeta$-cycles*, [arXiv:2106.01715](https://arxiv.org/abs/2106.01715), Enseign. Math. 69.

---

## Citation

If you use this package in academic work, please cite the software and the paper it implements, [*High-Precision Approximation of Riemann Zeros via the Truncated Weil Form*](https://arxiv.org/abs/2605.20224) ([paper folder](https://github.com/akivag613/connes-cvs-/tree/main/papers/1_high_precision_riemann_zeros)):

```bibtex
@software{connes_cvs_package,
  title   = {connes-cvs: An arbitrary-precision implementation of the
             {C}onnes--van {S}uijlekom {G}alerkin matrix},
  author  = {Groskin, Akiva},
  year    = {2026},
  version = {0.3.1},
  url     = {https://github.com/akivag613/connes-cvs-},
}

@article{groskin2026weil_form_approximation,
  title         = {High-Precision Approximation of {R}iemann Zeros
                   via the Truncated {W}eil Form},
  author        = {Groskin, Akiva},
  year          = {2026},
  eprint        = {2605.20224},
  archivePrefix = {arXiv},
  primaryClass  = {math.NT},
  doi           = {10.5281/zenodo.19546514},
  note          = {arXiv:2605.20224; archived on Zenodo (concept DOI
                   10.5281/zenodo.19546514, always resolves to the latest version).},
}
```

The companion notes, [*A finite Guinand-Weil dictionary and archimedean tail order*](https://arxiv.org/abs/2607.02828) ([Zenodo DOI](https://doi.org/10.5281/zenodo.21124802)) and [*A matrix-valued von Mangoldt measure*](https://github.com/akivag613/connes-cvs-/tree/main/papers/3_matrix_von_mangoldt_measure) ([DOI](https://doi.org/10.5281/zenodo.21242028)), are separate works with their own DOIs (see [Papers](#papers)); cite them directly if you use their results. [*Bulk trace laws and algebraic boundary interaction for the sine Loewner operator*](https://github.com/akivag613/connes-cvs-/tree/main/papers/4_sine_loewner_bulk_boundary) ([DOI](https://doi.org/10.5281/zenodo.22662221)) likewise has its own [citation](https://github.com/akivag613/connes-cvs-/blob/main/papers/4_sine_loewner_bulk_boundary/CITATION.bib). Its standalone verification programs do not require the `connes-cvs` package. Those paper DOIs are not software identifiers. The software citation metadata is in [CITATION.cff](https://github.com/akivag613/connes-cvs-/blob/main/CITATION.cff).

---

## References to this work

[Selected references to this work](https://github.com/akivag613/connes-cvs-/blob/main/REFERENCES_TO_THIS_WORK.md)
highlights independent reproductions and research that cites or uses the papers
and software, with links to the original sources.

---

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/akivag613/connes-cvs-/blob/main/.github/CONTRIBUTING.md) for developer-setup instructions and the numerical regression protocol. Version history is in [CHANGELOG.md](https://github.com/akivag613/connes-cvs-/blob/main/CHANGELOG.md).

Thanks to Dan Alec Yamaguchi ([danalec](https://github.com/danalec)) for reporting
the failure-path precision leak and identifying release-check and documentation
improvements in [PR #6](https://github.com/akivag613/connes-cvs-/pull/6).

---

## License

Software: [MIT License](https://github.com/akivag613/connes-cvs-/blob/main/LICENSE). Copyright (c) 2026 Akiva Groskin.

Paper manuscripts and figures use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as specified in their paper folders. The sine Loewner paper's exact mathematical witness data also use CC BY 4.0; its original Python programs use MIT. See its [license notice](https://github.com/akivag613/connes-cvs-/blob/main/papers/4_sine_loewner_bulk_boundary/source/LICENSES.md). Third-party libraries retain their own licenses.
