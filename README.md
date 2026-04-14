<div align="center">

# connes-cvs

### The first public implementation of the Connes–van Suijlekom Galerkin matrix for the Riemann Hypothesis.

[![PyPI version](https://img.shields.io/pypi/v/connes-cvs.svg?color=4c1)](https://pypi.org/project/connes-cvs/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zenodo DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19546515-1f74b7.svg)](https://zenodo.org/records/19546515)
[![arXiv](https://img.shields.io/badge/arXiv-2511.23257-b31b1b.svg)](https://arxiv.org/abs/2511.23257)
[![Tests](https://img.shields.io/badge/tests-6%2F6%20passing-4c1.svg)](tests/)

</div>

> The Riemann Hypothesis has resisted proof for over 140 years. Connes and van Suijlekom (2025) proposed a spectral route: construct a truncated Weil operator whose ground-state eigenvalue encodes how close the Riemann zeros come to satisfying Weil's positivity criterion. **This package is the only publicly available code that builds and diagonalizes that operator**, and reports convergence across **113 orders of magnitude** in the first-zero error.

<div align="center">

| Cutoff range | Eigenvalue span | Precision | Speedup (v0.2.0) | Published paper |
| :---: | :---: | :---: | :---: | :---: |
| `c = 13 … 67` | `10⁻⁵⁹ → 10⁻¹⁷³` | **80–200** decimal digits (tested) | **1.83×** total wall | [Zenodo 10.5281/zenodo.19546515](https://zenodo.org/records/19546515) |

</div>

---

## Table of contents

- [At a glance](#at-a-glance)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproduce the paper](#reproduce-the-paper)
- [Validation](#validation)
- [Performance](#performance)
- [How it works](#how-it-works)
- [Further reading](#further-reading)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## At a glance

**What is this?** An arbitrary-precision Python implementation of the Galerkin matrix $Q(c)$ from Proposition 4.1 of [Connes–van Suijlekom 2025](https://arxiv.org/abs/2511.23257). Its ground-state eigenvalue $\lambda_{\min}(c)$ tracks the spectral positivity condition that implies the Riemann Hypothesis.

**Why does it matter?** Until now, no independent public code existed for this construction. Connes (2026) reported a single number at $c = 13$; Connes–Consani–Moscovici (2025) reported a second number in a different basis. This package provides the third independent measurement, plus fourteen new cutoffs that together span **113 orders of magnitude** in the first-zero error.

**Convergence at a glance** (15 prime cutoffs, $T = 800$, $\mathrm{dps} = 150$–$200$):

```text
c=13  ████████                                                               -55
c=14  █████████                                                              -60
c=17  ███████████                                                            -76
c=19  █████████████                                                          -86
c=23  ███████████████                                                       -102
c=29  ██████████████████                                                    -119
c=31  ██████████████████                                                    -124
c=37  ████████████████████                                                  -135
c=41  █████████████████████                                                 -142
c=43  █████████████████████                                                 -145
c=47  ██████████████████████                                                -149
c=53  ███████████████████████                                               -156
c=59  ████████████████████████                                              -161
c=61  ████████████████████████                                              -163
c=67  █████████████████████████                                             -168
                                                                         log₁₀|γ₁ err|
```

Each step adds one prime. The convergence is monotone but not smooth (nine parametric models tested; all fail to capture the shape at residual ≤ 0.5 OOM).

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
import math

# Build the CvS Galerkin matrix at cutoff c = 13
Q = build_galerkin_matrix(c=13, N=100, T=400, dps=80)

# Diagonalize
lam_min, eigvec = compute_ground_state(Q)
print(f"λ_min(c=13) = {lam_min:.6e}")
# λ_min(c=13) ≈ 2.077e-59

# Extract the first detected Riemann zero
zeros = extract_zeros(eigvec, L=math.log(13), n_zeros=1)
print(f"γ₁ detected = {zeros[0]['gamma_detected']}")
print(f"|γ₁ error|  = {zeros[0]['error']:.4e}")
# γ₁ detected ≈ 14.1347251417...
# |γ₁ error|  ≈ 1.5e-55
```

End-to-end: ≈ 60 seconds on a 12-core machine with `python-flint` installed.

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

---

## Reproduce the paper

To replicate the full 15-cutoff sweep from [Groskin 2026](https://zenodo.org/records/19546515) (113 orders of magnitude in $|\gamma_1\,\mathrm{err}|$):

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

---

## Validation

This implementation reproduces the published benchmarks from Connes (2026) and Connes–Consani–Moscovici (2025), which are the only independent measurements of the CvS Galerkin spectrum in the literature prior to this work.

| Cutoff | Published | This package | Agreement |
| :---: | :---: | :---: | :--- |
| $c = 13$ | $2.6 \times 10^{-55}$ *(Connes 2026)* | $\mathbf{2.005 \times 10^{-55}}$ | factor 1.3 (basis-dependent constant) |
| $c = 13$ | $2.44 \times 10^{-55}$ *(CCM §6, $N=120$, 200-digit)* | $\mathbf{2.005 \times 10^{-55}}$ | factor 1.2 |
| $c = 14$ | $1.07 \times 10^{-60}$ *(CCM §6)* | $\mathbf{3.541 \times 10^{-61}}$ | order-of-magnitude (first independent $c=14$ measurement) |

All four numbers compute the same mathematical object — the truncated Weil minimizer $Q(c)$ — in different function bases (trigonometric vs. prolate-spheroidal) with different $N$, $T$, and precision settings. The factor-of-1.3 spread reflects Sobolev-norm differences between bases, not a correctness gap.

**New in this work:** 14 of the 15 cutoffs tested here ($c \geq 17$) have no prior published measurement in the literature. Full dataset in [`data/results_15pt_T800.json`](data/results_15pt_T800.json).

---

## Performance

Version 0.2.0 is **2.06× faster** on the dominant archimedean-integral phase and **1.83× faster** end-to-end, with **bit-identical** output to v0.1.0.

### Apples-to-apples A/B test ($c=13$, $N=80$, $T=400$, $\mathrm{dps}=80$, 12-way Pool)

| Phase | v0.1.0 | v0.2.0 | Speedup |
| :--- | ---: | ---: | :---: |
| Archimedean integral (cache) | 57.55 s | **27.94 s** | **2.06×** |
| Matrix assembly | 0.11 s | 0.12 s | unchanged |
| Symmetric eigensolver | 6.11 s | 6.19 s | unchanged |
| Root extraction | 1.16 s | 1.15 s | unchanged |
| **Total wall time** | **64.94 s** | **35.40 s** | **1.83×** |
| $\lambda_{\min}$ | `2.52826614019657560…e-59` | `2.52826614019657560…e-59` | **bit-identical (80 digits)** |

### At the published reference workload ($c=13$, $N=100$, $T=800$, $\mathrm{dps}=150$)

| | Historical (v0.1.0) | v0.2.0 |
| :--- | ---: | ---: |
| Wall time | 214.8 s | **127.3 s** (**1.69× faster**) |
| $\lambda_{\min}$ (paper Table 18: $2.865 \times 10^{-59}$) | `2.8654536149302802951…e-59` | `2.8654536149302802951…e-59` — **matches paper to all reported precision** |

See [`benchmarks/AB_VERIFIED_2026-04-14.md`](benchmarks/AB_VERIFIED_2026-04-14.md) for the full A/B protocol and raw timings. Reproduce on your hardware:

```bash
python benchmarks/win1_pool_benchmark.py 13 80 400 80
```

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

**The bottleneck** is the archimedean integral: evaluating the digamma function at ~11,000 adaptive quadrature nodes for each of $2N{+}1 = 201$ basis indices. Version 0.2.0 exploits two observations to halve the cost:

1. **$h_+$ is even in $\tau$** and mpmath's tanh-sinh rule is deterministic per `(interval, precision)`, so `psi_arch` and `psi_arch_deriv` share quadrature nodes. A dict keyed on $|\tau|$ gives a 4× hit rate on digamma calls.
2. **A fused real-arithmetic kernel** computes $\mathrm{Re}\,\hat{S}_x(\tau)$ and $\mathrm{Re}\,\partial_x \hat{S}_x(\tau)$ in one pass, sharing all sub-expressions ($\sin(\beta L)$, $\sin(\beta L/2)$, $1/\beta$, etc.).

Precision management is transparent. Eigenvalues shrink super-exponentially (at $c = 67$ the ground-state error reaches $10^{-168}$), so the published sweep threads 80–200 decimal digits of mpmath precision end-to-end (dps = 80 for the primary $c \leq 37$ cells; dps = 150 for the $c \in \\{13, 17, 19\\}$ retest; dps = 200 for $c \geq 41$). mpmath supports arbitrary precision beyond 200 digits; the paper notes that reaching $c \approx 97$ cleanly would require dps = 300, but such runs are not included in the published dataset.

---

## Further reading

- **Our paper** (computational study, 15 cutoffs, 113 OOM) — Groskin 2026, [Zenodo DOI 10.5281/zenodo.19546515](https://zenodo.org/records/19546515)
- **CvS** (mathematical foundation) — Connes & van Suijlekom, *Quadratic forms, real zeros and echoes of the spectral action*, [arXiv:2511.23257](https://arxiv.org/abs/2511.23257)
- **CCM** (prolate-basis companion) — Connes, Consani & Moscovici, *Zeta spectral triples*, [arXiv:2511.22755](https://arxiv.org/abs/2511.22755)
- **Connes 2026** (context) — *The Riemann Hypothesis: Past, Present and a Letter Through Time*, [arXiv:2602.04022](https://arxiv.org/abs/2602.04022)

---

## Citation

If you use this package in academic work, please cite both the software and the accompanying paper:

```bibtex
@software{connes_cvs_package,
  title   = {connes-cvs: First public implementation of the
             {C}onnes--van {S}uijlekom {G}alerkin matrix},
  author  = {Groskin, Akiva},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/akivag613/connes-cvs-},
  doi     = {10.5281/zenodo.19546515},
}

@article{groskin2026connes_cvs,
  title   = {Structural Properties of the {C}onnes--van {S}uijlekom
             Truncated {W}eil Minimizer: {S}obolev Scaling,
             Multi-Zero Universality, and {L}-Function Extension},
  author  = {Groskin, Akiva},
  year    = {2026},
  doi     = {10.5281/zenodo.19546515},
  url     = {https://zenodo.org/records/19546515},
}
```

---

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for developer-setup instructions, the test-running protocol, and the bit-identicality contract any performance PR must respect. Version history is in [CHANGELOG.md](CHANGELOG.md).

---

## License

[MIT License](LICENSE). Copyright (c) 2026 Akiva Groskin.
