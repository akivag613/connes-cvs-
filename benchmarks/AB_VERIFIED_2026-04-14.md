# WIN 1 A/B Verification — Production Scale

**Date:** 2026-04-14
**Branch:** `feature/v0.2.0-matrix-microopt` (WIN 1 still uncommitted)
**Method:** Stash WIN 1, run baseline; pop stash, run WIN 1; compare. Clean apples-to-apples.

## Test config

- `c=13, N=80, T=400, dps=80`
- 12-way multiprocessing Pool (production code path via `sweep._run_single_cutoff`)
- Python 3.12.11, mpmath 1.4.1, python-flint enabled (HAS_FLINT=True)
- Matrix DIM = 2N+1 = 161

## Results

| Phase | Baseline (no WIN 1) | WIN 1 | Speedup |
|---|---|---|---|
| cache_sec | 57.554 | 27.941 | **2.06×** |
| matrix_sec | 0.112 | 0.119 | ~unchanged |
| diag_sec | 6.113 | 6.185 | ~unchanged |
| zeros_sec | 1.156 | 1.149 | ~unchanged |
| **WALL TOTAL** | **64.939** | **35.398** | **1.83×** |
| **λ_min** | 2.5282661401965756026025862533001704392434144948201908268289070778008968019858182e-59 | 2.5282661401965756026025862533001704392434144948201908268289070778008968019858182e-59 | **bit-identical (80 digits match)** |

## Implication

- Cache phase 2.06× confirmed at production scale.
- Total wall 1.83× (matrix/diag/zeros are unchanged ~7.5s constant overhead).
- Bit-identicality holds in the multiprocessing path, not just the serial small benchmark.

## Extrapolation against historical production records

| Historical record | c | N | T | dps | Old | Projected w/ WIN 1 (cache × 0.486 + other) |
|---|---|---|---|---|---|---|
| results_A_c13.json | 13 | 100 | 400 | 150 | 199s | ~108s |
| results_U_T800_c13.json | 13 | 100 | 800 | 150 | 215s | ~117s |
| results_U_T800_c67.json | 67 | 100 | 800 | 200 | 543s | ~280s |
| Full 15-cutoff sweep (sum) | — | — | — | — | ~5400s | **~2900s (saves ~40 min)** |

## Files

- `_benchmarks/AB_baseline_no_win1.txt` — baseline run output
- `_benchmarks/AB_with_win1.txt` — WIN 1 run output
- `_benchmarks/win1_pool_benchmark.py` — A/B harness using production Pool path
- `_benchmarks/win1_benchmark.py` — small-workload benchmark (serial; the 2.06× verification)
- `_benchmarks/baseline_benchmark.py` — original baseline (serial, c=13 N=50 dps=50)

## Gold-standard verification at the published reference workload

The c=13 N=80 T=400 dps=80 A/B above proves the speedup at moderate scale. To prove the new code reproduces the *published headline number* in Paper 1 Table 18 (`λ_min(c=13) = 2.865 × 10⁻⁵⁹` at N=100 T=800 dps=150), v0.2.0 was run end-to-end at the exact paper-reference workload:

- **Computed:** `λ_min = 2.86545361493028029516151514986747977533798676783773219101029565377637421791530494377666704141009139776092287892559119370499915456183497252629004918875e-59`
- **Paper Table 18:** `2.865 × 10⁻⁵⁹` ✓ exact match to all reported precision
- **Wall time:** 127.3 s
- **Historical baseline (`results/iteration_7/results_U_T800_c13.json`):** 214.8 s
- **Speedup at exact paper workload:** **1.69×** total wall

Raw output: `_benchmarks/AB_published_reference_check.txt` (gitignored; full output preserved locally).

## Verdict

WIN 1 ships a real, reproducible, bit-identical speedup at production scale, verified against the canonical published reference. Safe to commit and publish.
