# `data/c100/` — Verification data for the $c = 100$ out-of-sample test (public)

**Contents.** Seven small JSON files (≈ 8 KB total) that together let any reader reproduce the headline claim of the accompanying paper revision: that an Aitken-$\Delta^2$ extrapolation of an $N$-sweep at $c = 100$ matches the Connes 2026 §6.4 heuristic continuum asymptotic to ~1% of the exponent.

A minimal verification script is at [`../../examples/c100_aitken_check.py`](../../examples/c100_aitken_check.py); it loads the JSONs in this directory and reproduces the headline computation in under a second.

## Files

| File | Content |
|---|---|
| `results_c100_N100_T800_dps500_v020.json` | $c=100$, $N=100$, $T=800$, $\mathrm{dps}=500$. $\lambda_{\min}^{\mathrm{even}} \approx 1.22 \times 10^{-191}$. |
| `results_c100_N150_T800_dps500_v020.json` | $c=100$, $N=150$, $T=800$, $\mathrm{dps}=500$. $\lambda_{\min}^{\mathrm{even}} \approx 6.42 \times 10^{-248}$. |
| `results_c100_N200_T800_dps500_v020.json` | $c=100$, $N=200$, $T=800$, $\mathrm{dps}=500$. $\lambda_{\min}^{\mathrm{even}} \approx 4.87 \times 10^{-295}$. |
| `results_c100_N150_T800_dps1000_v020.json` | $c=100$, $N=150$, $T=800$, $\mathrm{dps}=1000$. Precision retest of the $N=150$ row above. $\lambda_{\min}^{\mathrm{even}}$ is bit-identical between $\mathrm{dps}=500$ and $\mathrm{dps}=1000$ to 25 leading digits. |
| `results_c67_N150_T800_dps500_v020.json` | $c=67$, $N=150$, $T=800$, $\mathrm{dps}=500$. Corroborative measurement at the deepest in-sample cutoff. $\lambda_{\min}^{\mathrm{even}} \approx 5.33 \times 10^{-219}$, a 46-OOM drop versus the same-$c$, $N=100$ value reported in [`../results_15pt_T800.json`](../results_15pt_T800.json). |
| `richardson_n_extrapolation.json` | Aitken-$\Delta^2$ extrapolation of the three-point sequence $\{\log_{10}\lvert\lambda_N\rvert\}$ at $c=100$. Reports `aitken: -536.965`, against the Connes 2026 §6.4 prediction of $\approx -532$. |
| `c100_N150_dps1000_gamma_extraction.json` | $\gamma_1$ through $\gamma_{10}$ matching-digit counts extracted from the smallest-positive eigenvector at $c=100$, $N=150$, $\mathrm{dps}=1000$. Ranges 219–242 digits. |

## Schema (representative)

The four `results_c100_*_v020.json` files share a common shape:

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

`lambda_even` is stored as a full-precision decimal string (not a Python float) — its magnitude reaches $10^{-294}$ at $N=200$, far below double-precision floating-point representability.

`richardson_n_extrapolation.json` carries the $N$-sweep array, fitted models (exponential, power, $1/N$, stretched-exponential), and the Aitken-$\Delta^2$ acceleration scalar.

`c100_N150_dps1000_gamma_extraction.json` lists per-$k$ error magnitudes and matching-digit counts for $\gamma_1$ through $\gamma_{10}$.

## Provenance

All files were produced on a 12-worker Apple M-series workstation using the v0.2.0 math core of the `connes-cvs` PyPI package (with local v0.2.3 cell-runner additions for live-stream progress and adaptive precision calibration). Wall-clock budgets per file:

- $c=100$, $N=100$, $\mathrm{dps}=500$ — ~10 min
- $c=100$, $N=150$, $\mathrm{dps}=500$ — ~21 min
- $c=100$, $N=200$, $\mathrm{dps}=500$ — ~35 min
- $c=100$, $N=150$, $\mathrm{dps}=1000$ — ~111 min
- $c=67$, $N=150$, $\mathrm{dps}=500$ — ~26 min

The bit-identical match between the two $N=150$ rows at $\mathrm{dps} \in \{500, 1000\}$ certifies that the $\mathrm{dps}=500$ working precision is comfortably above the result's effective floor.

## Cross-references (public)

- Top-level headline + Aitken match: [`../../README.md`](../../README.md)
- Verification script: [`../../examples/c100_aitken_check.py`](../../examples/c100_aitken_check.py)
- 15-cutoff Paper 1 data: [`../results_15pt_T800.json`](../results_15pt_T800.json)
- Package source: [`../../connes_cvs/`](../../connes_cvs/)
