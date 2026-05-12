# `benchmarks/` — Per-optimization speed benchmarks

**Purpose.** Empirical measurements of the v0.2.0 math-core optimizations (`_h_plus_cached` memoization + `_re_S_and_dS_fused` real kernel) versus v0.1.0, plus the profile and probe scripts used during development.

## A/B benchmark summaries (v0.1.0 vs v0.2.0)

| File | Content |
|---|---|
| `AB_VERIFIED_2026-04-14.md` | Authoritative A/B summary. v0.2.0 delivers 2.06× on the psi-cache phase and 1.83×–1.69× end-to-end at production scale. Bit-identical to v0.1.0 at `c = 13`, `N = 80`, `T = 400`, `dps = 80` across a 500-digit mantissa. |
| `AB_baseline_no_win1.txt` | Raw wall-clock numbers for v0.1.0 (baseline). *(Local-only; gitignored.)* |
| `AB_with_win1.txt` | Raw wall-clock numbers for v0.2.0. *(Local-only; gitignored.)* |
| `AB_published_reference_check.txt` | Cross-check against the canonical paper data. *(Local-only; gitignored.)* |
| `HOTSPOTS.md` | Per-function CPU profile analysis that motivated the v0.2.0 optimization. *(Local-only; gitignored.)* |
| `WIN1_SUMMARY.md` | Narrative summary of the v0.2.0 optimization design. *(Local-only; gitignored.)* |

## Runnable benchmark scripts

| Script | Purpose |
|---|---|
| `baseline_benchmark.py` | Re-run the v0.1.0 baseline timing on your hardware. |
| `win1_benchmark.py` | Re-run the v0.2.0 timing on your hardware. |
| `win1_pool_benchmark.py` | v0.2.0 with multi-process Pool parallelism. Used in the README quoted timings. |

Example:

```bash
python benchmarks/win1_pool_benchmark.py 13 80 400 80
```

## Cross-references (public)

- Package source: [`../connes_cvs/`](../connes_cvs/)
- Tests (bit-identity gate): [`../tests/`](../tests/)
- Top-level performance section: [`../README.md`](../README.md)

## Discipline

- These benchmarks are historical (Apr 2026) and document the v0.1.0 → v0.2.0 transition. The A/B summary at `AB_VERIFIED_2026-04-14.md` is the authoritative numerical record.
- Probe scripts (`_probe_*.py`, `_compare_psi_full.py`, `run_profile.py`) are gitignored development artifacts; they remain in the local working tree for archaeological reference.
