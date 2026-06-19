# `data/c100/` — Verification data for the $c = 100$ out-of-sample test (public)

**Contents.** Eleven small JSON files that together let any reader reproduce the headline claim of the accompanying paper revision: that Aitken-$\Delta^2$ extrapolation of an $N$-sweep at $c = 100$ with $N \in \{100, 150, 200, 250\}$ approaches the Connes 2026 §6.4 heuristic continuum prediction $\approx -530.38$ monotonically with $N$ (6.39 OOM gap for the (100,150,200) anchor; 3.32 OOM for the deeper (150,200,250) anchor, out of $|x_\infty| \sim 530$), with consecutive first-difference ratios 0.8373 and 0.8355 consistent with a local geometric model.

A minimal verification script is at [`../../examples/c100_aitken_check.py`](../../examples/c100_aitken_check.py); it loads the JSONs in this directory and reproduces both Aitken triples plus the Connes prediction in under a second.

## Files

| File | Content |
|---|---|
| `results_c100_N100_T800_dps500_v020.json` | $c=100$, $N=100$, $T=800$, $\mathrm{dps}=500$. $\lambda_{\min}^{\mathrm{even}} \approx 1.22 \times 10^{-191}$. |
| `results_c100_N150_T800_dps500_v020.json` | $c=100$, $N=150$, $T=800$, $\mathrm{dps}=500$. $\lambda_{\min}^{\mathrm{even}} \approx 6.42 \times 10^{-248}$. |
| `results_c100_N200_T800_dps500_v020.json` | $c=100$, $N=200$, $T=800$, $\mathrm{dps}=500$. $\lambda_{\min}^{\mathrm{even}} \approx 4.87 \times 10^{-295}$. |
| `results_c100_N250_T800_dps500_v020.json` | $c=100$, $N=250$, $T=800$, $\mathrm{dps}=500$. $\lambda_{\min}^{\mathrm{even}} \approx 2.08 \times 10^{-334}$. |
| `results_c100_N150_T800_dps1000_v020.json` | $c=100$, $N=150$, $T=800$, $\mathrm{dps}=1000$. Precision retest of the $N=150$ row above. $\lambda_{\min}^{\mathrm{even}}$ agrees with the $\mathrm{dps}=500$ value to 25 leading significant digits. |
| `results_c67_N150_T800_dps500_v020.json` | $c=67$, $N=150$, $T=800$, $\mathrm{dps}=500$. Corroborative measurement at the deepest in-sample cutoff. $\lambda_{\min}^{\mathrm{even}} \approx 5.33 \times 10^{-219}$, a 46-OOM drop versus the same-$c$, $N=100$ value reported in [`../results_15pt_T800.json`](../results_15pt_T800.json). |
| `richardson_n_extrapolation.json` | Aitken-$\Delta^2$ extrapolation of the three-point sequence $\{\log_{10}\lvert\lambda_N\rvert\}$ at $c=100$, $N\in\{100,150,200\}$. Reports `aitken: -536.965`. Note: the four-point analysis (incorporating $N=250$) is computed in [`../../examples/c100_aitken_check.py`](../../examples/c100_aitken_check.py); the deeper-anchored triple gives Aitken $\approx -533.70$. |
| `c100_N150_dps1000_gamma_extraction.json` | $\gamma_1$ through $\gamma_{10}$ extraction from the $c=100$, $N=150$, $\mathrm{dps}=1000$ smallest-positive eigenvector. Matching-digit counts range 219–242. Per-$\gamma_k$: detected value, true `mp.zetazero(k).imag` reference, error magnitude, log10 error. Independent verification path documented in the embedded `verification_protocol` field. |
| `c100_N250_dps500_gamma_extraction.json` | $\gamma_1$ through $\gamma_{10}$ extraction from the $c=100$, $N=250$, $\mathrm{dps}=500$ smallest-positive eigenvector with tight findroot tolerance $10^{-380}$. Matching-digit counts range 307–329 (deeper than the $N=150$, $\mathrm{dps}=1000$ extraction). Both detected and reference values stored to 400 significant digits; the `verification_protocol` field describes how to independently confirm `matching_digits = floor(-log10(error))` using `mp.zetazero(k)` at $\mathrm{dps}=400$. |
| `c100_N120_dps560_gamma_extraction.json` | $\gamma_1$ through $\gamma_{10}$ from the $c=100$, $N=120$, $T=800$, $\mathrm{dps}=560$ smallest-positive eigenvector (tight findroot tolerance $10^{-520}$). $\gamma_1$ matches `mp.zetazero(1)` to 210 digits; per-$k$ counts range 186–210. Stores **detected** $\gamma_k$ and the true reference to 500 significant digits (verified dps-stable to $>515$ digits), enabling an eigenvalue-vs-eigenvalue diff. Off-sweep cross-validation cell at an independent grid point (not part of the four-point Aitken sweep); reproduction recipe in the embedded `verification_protocol`. |
| `c100_N160_dps560_gamma_extraction.json` | $\gamma_1$ through $\gamma_{10}$ from the $c=100$, $N=160$, $T=800$, $\mathrm{dps}=560$ smallest-positive eigenvector (tight findroot tolerance $10^{-520}$). $\gamma_1$ matches `mp.zetazero(1)` to 253 digits; per-$k$ counts range 230–253. Stores **detected** $\gamma_k$ and the true reference to 500 significant digits, enabling an eigenvalue-vs-eigenvalue diff. Off-sweep cross-validation cell at an independent grid point; reproduction recipe in the embedded `verification_protocol`. |

## Schema (representative)

The five `results_c100_*_v020.json` files share a common shape:

```json
{
  "tag":          "...",
  "c":            100,
  "N":            150,
  "T":            800,
  "dps":          500,
  "flint_prec":   2000,
  "engine":       "v020",
  "lambda_even":  "<full-precision decimal string>",
  "t_cache_s":    <float>,
  "t_mat_s":      <float>,
  "t_diag_s":     <float>,
  "t_total_s":    <float>,
  "n_workers":    12,
  "version":      "v0.2.3-local"
}
```

`lambda_even` is stored as a full-precision decimal string (not a Python float) — its magnitude reaches $10^{-334}$ at $N=250$, far below double-precision floating-point representability.

`richardson_n_extrapolation.json` carries the three-point $N$-sweep array (legacy; pre-$N{=}250$), fitted models (exponential, power, $1/N$, stretched-exponential), and the Aitken-$\Delta^2$ acceleration scalar for that three-point case.

`c100_N*_gamma_extraction.json` files list per-$k$ detected $\gamma_k$ (decimal-string), true `mp.zetazero(k).imag` reference, absolute error, $\log_{10}$ error, and floor matching-digit count. The $N=250$ extraction stores all $\gamma$ values to 400 significant digits, sufficient for independent verification past the 329-digit headline.

## Provenance

All files were produced on a 12-worker Apple M-series workstation using the v0.2.0 math core of the `connes-cvs` PyPI package (with local v0.2.3 cell-runner additions for live-stream progress and adaptive precision calibration). Wall-clock budgets per file:

- $c=100$, $N=100$, $\mathrm{dps}=500$ — ~10 min
- $c=100$, $N=150$, $\mathrm{dps}=500$ — ~21 min
- $c=100$, $N=200$, $\mathrm{dps}=500$ — ~35 min
- $c=100$, $N=250$, $\mathrm{dps}=500$ — ~38 min
- $c=100$, $N=150$, $\mathrm{dps}=1000$ — ~111 min
- $c=67$, $N=150$, $\mathrm{dps}=500$ — ~26 min

The agreement to 25 leading significant digits between the two $N=150$ rows at $\mathrm{dps} \in \{500, 1000\}$ certifies that the $\mathrm{dps}=500$ working precision is comfortably above the result's effective floor at $N=150$, supporting the use of $\mathrm{dps}=500$ at $N=250$ as well.

## Cross-references (public)

- Top-level headline + Aitken match: [`../../README.md`](../../README.md)
- Verification script: [`../../examples/c100_aitken_check.py`](../../examples/c100_aitken_check.py)
- 15-cutoff Paper 1 data: [`../results_15pt_T800.json`](../results_15pt_T800.json)
- Package source: [`../../connes_cvs/`](../../connes_cvs/)
