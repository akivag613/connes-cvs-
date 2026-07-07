<div align="center">

# connes-cvs

### To our knowledge, the first public implementation of the Connes–van Suijlekom Galerkin matrix for the Riemann Hypothesis.

[![PyPI version](https://img.shields.io/pypi/v/connes-cvs.svg?color=4c1&cacheSeconds=300)](https://pypi.org/project/connes-cvs/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper 1 · arXiv](https://img.shields.io/badge/Paper_1-arXiv%3A2605.20224-b31b1b.svg)](https://arxiv.org/abs/2605.20224)
[![Paper 2 · arXiv](https://img.shields.io/badge/Paper_2-arXiv%3A2607.02828-b31b1b.svg)](https://arxiv.org/abs/2607.02828)
[![Zenodo DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.19546514-1f74b7.svg)](https://doi.org/10.5281/zenodo.19546514)
[![Tests](https://img.shields.io/badge/tests-passing-4c1.svg)](tests/)

</div>

> Connes & van Suijlekom (2025) proposed a spectral route to the Riemann Hypothesis: a truncated Weil quadratic form whose ground-state eigenvalue encodes how close the Riemann zeros come to satisfying Weil's positivity criterion. **This package is, to our knowledge, the first publicly available code that builds and diagonalizes that operator.** It computes the smallest-positive eigenvalue across the full sweep — **275 orders of magnitude** spanning $\sim 10^{-59}$ at $c = 13$ down to $\sim 10^{-334}$ at $c = 100$, $N = 250$ — reaches **329 matching digits** on $\gamma_1$ at $c = 100$, and supplies, to our knowledge, the first independent out-of-sample numerical test of the Connes 2026 §6.4 continuum asymptotic.

---

## Papers

This repository hosts the `connes-cvs` package together with the two papers by **Akiva Groskin** that build on the truncated Weil quadratic form of Connes–van Suijlekom. The package implements the Galerkin operator both papers study.

<table>
<tr><td valign="top" width="50%">

**Paper 1 — the numerics**

*High-Precision Approximation of Riemann Zeros via the Truncated Weil Form.*

[arXiv:2605.20224](https://arxiv.org/abs/2605.20224) (math.NT) · Zenodo concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514)

Builds and diagonalizes the CvS Galerkin matrix at high precision: extracts Riemann zeros to hundreds of matching digits and tests the Connes 2026 §6.4 continuum asymptotic out-of-sample at $c = 100$. **This is the paper the `connes-cvs` package implements** (see [ERRATA.md](ERRATA.md) for a finite-cutoff sign correction).

</td><td valign="top" width="50%">

**Paper 2 — the structure**

*A finite Guinand-Weil dictionary and archimedean tail order for the truncated Weil quadratic form.*

[arXiv:2607.02828](https://arxiv.org/abs/2607.02828) (math.NT, math.SP) · Zenodo concept DOI [10.5281/zenodo.21124802](https://doi.org/10.5281/zenodo.21124802)

An exact finite Guinand-Weil zero-source dictionary for the truncated Weil form, plus a finite-cutoff archimedean tail-order theorem with a two-sided certification rule. Manuscript and full reproducibility package in [`guinand_weil_dictionary_tail_order/`](guinand_weil_dictionary_tail_order/).

</td></tr>
</table>

Both papers report empirical measurements and finite-cutoff structural results; neither claims a proof of the Riemann Hypothesis.

<div align="center">

| Cutoff range | $\lambda_{\min}$ span | $\gamma_1$ accuracy | Cross-check |
| :---: | :---: | :---: | :---: |
| `c = 13 … 67` | `10⁻⁵⁹ → 10⁻¹⁷³` | up to **168 matching digits** (`c=67, N=100, dps=200`) | matches CCM 2025 at `c=14` to factor 3 |
| `c = 100` | `10⁻³³⁴` (`N=250, dps=500`) | **329 matching digits** (`N=250, dps=500`) | two consecutive Aitken-Δ² approaching Connes 2026 §6.4 (≈ −530.4) monotonically; deeper triple within 3.32 OOM, ratios 0.8373 / 0.8355 |

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
- [Contributing](#contributing)
- [License](#license)

---

## Headline result

**The Connes 2026 §6.4 heuristic continuum asymptotic, tested out-of-sample at $c = 100$.**

Connes 2026 (arXiv:2602.04022) §6.4 gives a heuristic continuum decay rate
$$1 - \chi_2(\lambda) \;\sim\; \frac{2^{14}}{3}\,\sqrt{2}\,\pi^{5}\; e^{-4\pi e^{L} + 9L/2}, \qquad L = 2\log\lambda,$$
for the second angular function $\chi_2$, tracking the smallest eigenvalue of the truncated Weil quadratic form. Until now this prediction was supported only by agreement with the smallest eigenvalue $\varepsilon(\lambda)$ for $\lambda \leq 14$ (the cutoffs reported in CCM 2025 §6 with $N = 120$).

Using this package at $c = 100$ with $N \in \{100, 150, 200, 250\}$ at $\mathrm{dps} = 500$, two consecutive Aitken-Δ² extrapolations on the overlapping triples give
$$\log_{10}\bigl|\lambda_\infty^{\mathrm{even}}(c{=}100)\bigr| \;\approx\; -536.76 \;\;\text{and}\;\; -533.70,$$
approaching the Connes 2026 §6.4 prediction of $\approx -530.38$ monotonically with $N$; the consecutive first-difference ratios $0.8373$ and $0.8355$ match to two decimal places, evidence for a local geometric model. The deeper-anchored triple sits **3.32 OOM** above the prediction, out of $|x_\infty| \sim 530$ — agreement at the under-1%-of-exponent level on the deeper anchor, out-of-sample (the in-sample fit window was $c \leq 67$ at $N = 100$). This is, to our knowledge, the first independent out-of-sample empirical test of the §6.4 heuristic continuum prediction at $c > 14$.

**Companion observations** (full details in the paper):

- $\gamma_1$ through $\gamma_{10}$ extracted to **307–329 matching digits** at $c = 100$, $N = 250$, $\mathrm{dps} = 500$ (and **219–242** at $N = 150$, $\mathrm{dps} = 1000$).
- Under the unitary equivalence with CCM 2025 Lemma 5.1, every $\gamma_k$ extraction here is, modulo a hypothesis-status caveat at $c = 100$ documented in the paper, an eigenvalue of the rank-one perturbed scaling operator $D_{\log}^{(\lambda,N)}$ of CCM Theorem 1.1(iii) at $\lambda = \sqrt{c}$.
- The empirical fit $|\log_{10}\lambda_{\min}(c)| \approx 13.24 \, c^{0.634}$ valid on $c \leq 67$ at $N = 100$ is shown to be a finite-$N$ rate, not the continuum asymptote: the $c = 100$, $N = 200$ datum falsifies the pure-power-law extrapolation by 49 orders of magnitude.

The accompanying paper is on **arXiv** — [arXiv:2605.20224](https://arxiv.org/abs/2605.20224) (math.NT) — and archived on **Zenodo**, where the concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) always resolves to the latest version. A correction to the $c=100$ and $L(s,\chi_3)$ negative-sign eigenvalue claims (both finite-cutoff artifacts; no quantitative result changes) is recorded in [ERRATA.md](ERRATA.md).

---

## Installation

```bash
pip install connes-cvs
```

For the optional **11× speedup** on the archimedean integral via Arb's arbitrary-precision digamma:

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

- Python ≥ 3.9
- [mpmath](https://mpmath.org/) ≥ 1.3 (arbitrary-precision arithmetic)

### Optional dependencies

- [python-flint](https://github.com/flintlib/python-flint) ≥ 0.5 — Arb-backed digamma (≈ 11× speedup)
- [gmpy2](https://github.com/aleaxit/gmpy2) ≥ 2.1 — GMP-backed mpmath core
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) — for downstream analysis

---

## Quick start

```python
from connes_cvs import build_galerkin_matrix, compute_ground_state, extract_zeros
import mpmath as mp

# Build the CvS Galerkin matrix at cutoff c = 13
Q = build_galerkin_matrix(c=13, N=100, T=400, dps=80)

# Diagonalize
lam_min, eigvec = compute_ground_state(Q)
print(f"λ_min(c=13) = {lam_min:.6e}")
# λ_min(c=13) ≈ 2.077e-59

# Extract the first detected Riemann zero.
# Pass L = log(c) at full working precision (mp.log, NOT a float math.log):
# a float L carries only ~16 digits and silently caps the extraction at ~16 digits.
zeros = extract_zeros(eigvec, L=mp.log(13), n_zeros=1, dps=80)
print(f"γ₁ detected = {zeros[0]['gamma_detected']}")
print(f"|γ₁ error|  = {zeros[0]['error']:.4e}")
# γ₁ detected ≈ 14.1347251417...
# |γ₁ error|  ≈ 1.45e-55
```

End-to-end: ≈ 5 minutes on a 12-core machine with `python-flint` installed (the script is single-process; the 12-core advantage shows up in `examples/c100_aitken_check.py` and the full 15-cutoff sweep, both of which use `multiprocessing`). Without `python-flint` the run is slower (~30 minutes) since the archimedean digamma evaluation dominates wall-clock.

<details>
<summary><b>Multi-cutoff sweep (click to expand)</b></summary>

```python
from connes_cvs.sweep import run_sweep

results = run_sweep(
    cutoffs=[13, 17, 19, 23, 29],
    N=100, T=400, dps=80, workers=12,
)

for c, r in results.items():
    print(f"c={c:2d}  λ_min = {r['lambda_min']:.4e}  |γ₁ err| = {r['gamma1_error']:.4e}")
```

</details>

A minimal runnable example is also available at [`examples/basic_compute.py`](examples/basic_compute.py).

---

## The c = 100 verification

The headline measurement is computable with this package by combining the standard primitives with a higher-$N$ basis and elevated working precision. A minimal verification script that loads the published $N$-sweep data and reproduces the Aitken-Δ² match to the Connes 2026 §6.4 prediction in under a second is at [`examples/c100_aitken_check.py`](examples/c100_aitken_check.py); the underlying data is in [`data/c100/`](data/c100/).

> **Data provenance and reproducibility.** The $c = 100$ dataset in [`data/c100/`](data/c100/) was generated by a performance-optimized local runner (staged for a future `v0.3.0` umbrella release) whose numerical core is **bit-identical** to the released `connes-cvs` v0.2.2 — verified on the shared workloads (the $c = 13$, $N = 80$ A/B cell agrees to all 80 printed digits of $\lambda_{\min}$ in the v0.1.0/v0.2.0 benchmark). The released package computes the same mathematical object: calling `build_galerkin_matrix` → `compute_ground_state` → `extract_zeros` at $c = 100$ with the tabulated $(N, T, \mathrm{dps})$ reproduces these values — the local runner changes only wall-clock, not the arithmetic.

### N-sweep at c = 100, T = 800

| $N$ | $\mathrm{dps}$ | $\lambda_{\min}^{\mathrm{even}}$ | $\log_{10}\!\lvert\lambda_{\min}\rvert$ | wall-clock |
| :---: | :---: | :---: | :---: | :---: |
| 100 | 500  | $1.22 \times 10^{-191}$ | $-190.92$ | ~10 min |
| 150 | 500  | $6.42 \times 10^{-248}$ | $-247.19$ | ~21 min |
| 200 | 500  | $4.87 \times 10^{-295}$ | $-294.31$ | ~35 min |
| 250 | 500  | $2.08 \times 10^{-334}$ | $-333.68$ | ~38 min |
| 150 | 1000 | $6.42 \times 10^{-248}$ | $-247.19$ | ~111 min |

Wall-clock on an Apple M-series machine with 12 workers. The bit-identical match between the $\mathrm{dps} = 500$ and $\mathrm{dps} = 1000$ rows at $N = 150$ certifies the precision claim.

### Aitken-Δ² extrapolation

The four-point sequence $x_N = \log_{10}|\lambda_N|$ at $c = 100$ admits Aitken-$\Delta^2$ acceleration on two overlapping triples:

$$\hat{x}_\infty^{\{100,150,200\}} \;\approx\; -536.76, \qquad \hat{x}_\infty^{\{150,200,250\}} \;\approx\; -533.70.$$

The consecutive first-difference ratios $|\Delta_2/\Delta_1| = 0.8373$ and $|\Delta_3/\Delta_2| = 0.8355$ match to two decimal places, evidence for a local geometric model of the convergence sequence (not a 3-point forced fit).

The Connes 2026 §6.4 heuristic prediction at $c = 100$ is
$$\log_{10}\!\frac{2^{14}\sqrt{2}\,\pi^{5}}{3} \;-\; \frac{4\pi\cdot 100}{\ln 10} \;+\; \frac{9\log 100}{2\ln 10} \;\approx\; 6.37 - 545.75 + 9.00 \;\approx\; -530.38.$$

The two Aitken anchors sit 6.39 OOM and 3.32 OOM above the prediction respectively, out of a magnitude range $|x_\infty| \sim 530$, with the trend monotone in $N$ — agreement at the **under-1%-of-exponent level on the deeper anchor**, out-of-sample (the in-sample fit window was $c \leq 67$ at $N = 100$). Four points do not rule out alternative convergence-model fits; see the paper for the model-sensitivity discussion.

### γ_k extraction at c = 100

| $k$ | $N=150$, dps=1000 | $N=250$, dps=500 | $k$ | $N=150$, dps=1000 | $N=250$, dps=500 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 242 | **329** | 6 | 228 | 316 |
| 2 | 239 | 325 | 7 | 226 | 313 |
| 3 | 236 | 323 | 8 | 224 | 312 |
| 4 | 233 | 320 | 9 | 221 | 309 |
| 5 | 231 | 318 | 10 | 219 | 307 |

Matching-digit count is floor($-\log_{10}|\gamma_k^{\text{detected}} - \gamma_k^{\text{exact}}|$); the reference is `mpmath.zetazero(k).imag` at `dps=400`. For comparison, CCM 2025 §6 reports $\gamma_1$ matching to ~55 digits at $c = 13$, $N = 120$.

---

## Reproduce the published sweep

To replicate the 15-cutoff sweep at $c \in \{13, 14, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67\}$ — 113 orders of magnitude in $|\gamma_1\,\mathrm{err}|$:

```python
from connes_cvs.sweep import run_sweep
import json

CUTOFFS = [13, 14, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]

results = run_sweep(
    cutoffs=CUTOFFS,
    N=100,
    T=800,
    dps=150,            # use dps=200 for c >= 41
    workers=12,
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

Expected wall time on a 12-core modern machine (v0.2.0): **~48 minutes**. The result is bit-identical in $\lambda_{\min}$ to the reference dataset in [`data/results_15pt_T800.json`](data/results_15pt_T800.json).

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
c=100  ███████████████████████████████████████████████  (N=250, dps=500)  -329
                                                                          log₁₀|γ₁ err|
```

Rows $c \leq 67$ use $N = 100$; the $c = 100$ row uses $N = 250$, $\mathrm{dps} = 500$ (headline cell). The $c \leq 67$ rows report the finite-$N = 100$ rate; the continuum asymptote (Connes 2026 §6.4) decays significantly faster, as the $c = 100$ row makes visible.

---

## Validation against published data

Independent cross-checks of this package against published values. The $c = 13$ and $c = 14$ rows compare the **first-zero error** $\lvert\gamma_1 - t_1\rvert$ (which is orders of magnitude larger than $\lambda_{\min}$ itself); the $c = 100$ row compares the **smallest-eigenvalue decay** $\log_{10}\lvert\varepsilon\rvert$ against the Connes 2026 §6.4 heuristic. The two quantities are distinct — do not read the $\sim 10^{-55}$ values as $\lambda_{\min}$.

| Cutoff | Quantity | Published | This package | Agreement |
| :---: | :---: | :---: | :---: | :--- |
| $c = 13$ | $\lvert\gamma_1\text{ err}\rvert$ | $2.6 \times 10^{-55}$ — Connes 2026 §6 | $\mathbf{2.005 \times 10^{-55}}$ | factor 1.3 |
| $c = 13$ | $\lvert\gamma_1\text{ err}\rvert$ | $2.44 \times 10^{-55}$ — CCM 2025 §6, $N=120$, 200-digit | $\mathbf{2.005 \times 10^{-55}}$ | factor 1.2 |
| $c = 14$ | $\lvert\gamma_1\text{ err}\rvert$ | $1.07 \times 10^{-60}$ — CCM 2025 §6 | $\mathbf{3.541 \times 10^{-61}}$ | factor 3 |
| $c = 100$ | $\log_{10}\lvert\varepsilon\rvert$ | $\approx -530.38$ — Connes 2026 §6.4 (heuristic) | two Aitken-Δ² anchors at $\mathbf{-536.76}$ and $\mathbf{-533.70}$ | 3.32 OOM (deeper anchor); under 1% of exponent |

All rows probe the same operator — the truncated Weil minimizer $Q(c)$ in the **trigonometric basis** — but report different quantities, as noted above. The factor-of-1.3 spread at $c = 13$ reflects differences in $N$, $T$, $\mathrm{dps}$, and normalization conventions, not a correctness gap.

**Independent third-party reproduction.** B. Martin (Skyline Trail Computing) independently reimplemented the CvS/CCM Galerkin matrix from scratch — a separate multiprecision assembly with no shared code — and reports agreement with this package and the published CCM/Connes values at $c = 13$ to ~54 digits on $\gamma_1$ (first-zero error $1.77 \times 10^{-55}$). Martin's frozen `connes-cvs` oracle of the $c = 13$, $N = 80$, $T = 400$, $\mathrm{dps} = 80$ A/B cell matches this package's computed $\lambda_{\min}$ to all 79 printed digits. See the [reproduction notes](https://github.com/skylinetrailcomputing/zeta-spectral-gpu/blob/main/knowledge/ccm-reproduction-notes.md) and [issue #1](https://github.com/akivag613/connes-cvs-/issues/1).

**Spectral-triple interpretation (CCM 2025 Lemma 5.1 + Theorem 1.1(iii)).** Under the unitary equivalence of $Q(c)$ with the CCM matrix $\tau_{i,j}$, the $F_{\mathrm{even}}$ test function used by this package's `extract_zeros` is, up to a positive scaling constant and the change of variable $u = e^x$, the same as the Fourier–Mellin transform $\widehat{\xi}_N(z)$ appearing in CCM 2025. Every $\gamma_k$ extracted by this package is, therefore, equivalently a zero of $\widehat{\xi}_N$ and an eigenvalue of the rank-one perturbed scaling operator $D_{\log}^{(\lambda,N)}$ at $\lambda = \sqrt{c}$.

**New cutoffs.** The thirteen cutoffs $c \in \{17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67\}$ and the cutoff $c = 100$ have no prior published independent measurement in the literature we have located.

Full dataset for the 15-cutoff sweep in [`data/results_15pt_T800.json`](data/results_15pt_T800.json).

---

## Performance

Version 0.2.0 is **2.06× faster** on the dominant archimedean-integral phase and **1.83× faster** end-to-end than v0.1.0, with **bit-identical** output.

### A/B test ($c=13$, $N=80$, $T=400$, $\mathrm{dps}=80$, 12-way Pool)

| Phase | v0.1.0 | v0.2.0 | Speedup |
| :--- | ---: | ---: | :---: |
| Archimedean integral (cache) | 57.55 s | **27.94 s** | **2.06×** |
| Matrix assembly | 0.11 s | 0.12 s | unchanged |
| Symmetric eigensolver | 6.11 s | 6.19 s | unchanged |
| Root extraction | 1.16 s | 1.15 s | unchanged |
| **Total wall time** | **64.94 s** | **35.40 s** | **1.83×** |
| $\lambda_{\min}$ | `2.52826614019657560…e-59` | `2.52826614019657560…e-59` | **bit-identical (80 digits)** |

### Published reference workload ($c=13$, $N=100$, $T=800$, $\mathrm{dps}=150$)

| | v0.1.0 | v0.2.0 |
| :--- | ---: | ---: |
| Wall time | 214.8 s | **127.3 s** (**1.69× faster**) |
| $\lambda_{\min}$ | `2.8654536149302802951…e-59` | `2.8654536149302802951…e-59` |

See [`benchmarks/AB_VERIFIED_2026-04-14.md`](benchmarks/AB_VERIFIED_2026-04-14.md) for the full A/B protocol and raw timings.

---

## How it works

The truncated Weil quadratic form decomposes into three arithmetically transparent pieces:

$$
Q(c) = D_\infty + D_{\text{pole}} + D_{\text{prime}}
$$

- $D_\infty$ — archimedean Mellin multiplier $h_+(\tau) = \mathrm{Re}\,\psi(\tfrac{1}{4} + i\tfrac{\tau}{2}) - \log\pi$
- $D_{\text{pole}}$ — rank-one correction from the pole of $\zeta(s)$ at $s=1$
- $D_{\text{prime}}$ — finite von-Mangoldt sum over primes $p \leq c$

The Galerkin matrix entries are
$$
q_{m,n} = \frac{\psi(m) - \psi(n)}{m - n}, \qquad q_{n,n} = \psi'(n),
$$
where $\psi(x) = \tfrac{1}{\pi} \int_0^L \sin\bigl(2\pi x(1-y/L)\bigr)\, D(y)\, dy$ and $L = \log c$.

The bottleneck is the archimedean integral: evaluating the digamma function at ~11,000 adaptive quadrature nodes for each of $2N{+}1$ basis indices. Version 0.2.0 exploits two observations to halve the cost:

1. **$h_+$ is even in $\tau$** and mpmath's tanh-sinh rule is deterministic per `(interval, precision)`, so `psi_arch` and `psi_arch_deriv` share quadrature nodes. A dict keyed on $|\tau|$ gives a 4× hit rate on digamma calls.
2. **A fused real-arithmetic kernel** computes $\mathrm{Re}\,\hat{S}_x(\tau)$ and $\mathrm{Re}\,\partial_x \hat{S}_x(\tau)$ in one pass, sharing $\sin(\beta L)$, $\sin(\beta L / 2)$, $1/\beta$, and related sub-expressions.

Precision management is transparent. Eigenvalues shrink super-exponentially ($\lambda_{\min} \sim 10^{-168}$ at $c = 67$ with $N = 100$; $\sim 10^{-247}$ at $c = 100$ with $N = 150$). The published 15-cutoff sweep threads 80–200 decimal digits of mpmath precision end-to-end; the $c = 100$ datum uses up to 1000 digits.

---

## Further reading

- **Our paper** — Groskin 2026, *High-Precision Approximation of Riemann Zeros via the Truncated Weil Form*, [arXiv:2605.20224](https://arxiv.org/abs/2605.20224) (math.NT). Archived on Zenodo; the concept DOI [10.5281/zenodo.19546514](https://doi.org/10.5281/zenodo.19546514) always resolves to the latest version.
- **Companion note (Paper 2)** — Groskin 2026, *A finite Guinand-Weil dictionary and archimedean tail order for the truncated Weil quadratic form*, [arXiv:2607.02828](https://arxiv.org/abs/2607.02828) (math.NT): an exact finite Guinand-Weil zero-source dictionary for the truncated Weil form, and a finite-cutoff archimedean tail-order theorem with a two-sided certification rule. Manuscript and full reproducibility package in [`guinand_weil_dictionary_tail_order/`](guinand_weil_dictionary_tail_order/); archived on Zenodo, concept DOI [10.5281/zenodo.21124802](https://doi.org/10.5281/zenodo.21124802) (always resolves to the latest version).
- **CvS — mathematical foundation** — Connes & van Suijlekom, *Quadratic forms, real zeros and echoes of the spectral action*, [arXiv:2511.23257](https://arxiv.org/abs/2511.23257).
- **CCM — the rank-one spectral-triple construction whose spectrum this package measures** — Connes, Consani & Moscovici, *Zeta spectral triples*, [arXiv:2511.22755](https://arxiv.org/abs/2511.22755).
- **Connes 2026 — the §6.4 heuristic asymptotic this work tests at $c = 100$** — *The Riemann Hypothesis: Past, Present and a Letter Through Time*, [arXiv:2602.04022](https://arxiv.org/abs/2602.04022).
- **Connes–Consani 2023 — qualitative motivation for the $k_\lambda$ approximation in Connes 2026 §6.6** — *Spectral triples and $\zeta$-cycles*, [arXiv:2106.01715](https://arxiv.org/abs/2106.01715), Enseign. Math. 69.

---

## Citation

If you use this package in academic work, please cite both the software and the accompanying paper:

```bibtex
@software{connes_cvs_package,
  title   = {connes-cvs: Public implementation of the
             {C}onnes--van {S}uijlekom {G}alerkin matrix},
  author  = {Groskin, Akiva},
  year    = {2026},
  version = {0.2.2},
  url     = {https://github.com/akivag613/connes-cvs-},
  doi     = {10.5281/zenodo.19546514},
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

---

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for developer-setup instructions, the test-running protocol, and the bit-identicality contract any performance PR must respect. Version history is in [CHANGELOG.md](CHANGELOG.md).

---

## License

[MIT License](LICENSE). Copyright (c) 2026 Akiva Groskin.
