# Claim Trace Audit

> Paths below are relative to this package: verification scripts are in `scripts/`, their JSON/log outputs in `artifacts/`, LaTeX source in `source/`, figures in `figures/`.

Timestamp: 2026-07-02 (post-revision pass)

Scope: this audit traces every displayed numerical claim in the manuscript to a
raw artifact shipped in the release.  The checks are traceability guards, not
proof substitutes, and not interval certificates unless explicitly marked as
Arb interval output.

## Source Files

- `audit_exact_series_identity.py` -> `exact_series_audit.json`
- `audit_kernel_span_rank.py` -> `kernel_span_rank_audit.json`
- `audit_full_matrix_source_quotient.py` -> `full_matrix_source_quotient_audit.json`
- `audit_pole_neutral_survival.py` -> `pole_neutral_survival_audit.json`
  (dimension cases are exact rank computations over Q(b) as of 2026-07-02)
- `verify_finite_dictionary.py` -> stdout guards (three independent pole routes)
- `audit_arch_tail_dt_bridge.py` -> `arch_tail_dt_bridge_audit.json`
- `verify_arch_tail_order.py` -> `arch_tail_order_check.json`
- `arch_tail_budget.py` -> `arch_tail_budget_c100_N200_T800.json`
- `arch_tail_stress_ladder.py` -> `arch_tail_stress_ladder.json`
- `arch_tail_exact_asymptotic.py` -> `arch_tail_exact_vs_asymptotic.json`
- `verify_zero_side.py` -> `zero_side_values.json`
- `verify_dictionary_threeroute.py` -> `threeroute_*.json`, `zeta_zeros_512_dps30.json`
- `arb_ldlt_certify.py` -> `c100_N200_arb_ldlt_prec9000.log`,
  `c100_N200_arb_ldlt_prec9000_provenance.json`
- `make_figures.py` -> `fig_dictionary.pdf`, `fig_tailorder.pdf`
  (reads `eigenflow_c13N4.json`, `zeta_zeros_512_dps30.json`)

## Claim Trace

| Manuscript location | Displayed claim | Raw source |
| --- | ---: | --- |
| Lemma 2.1 | generic-vector agreement of closed-form assembly and source assembly: `2.0e-10` at (29,6), `2.1e-15` at (13,4) | `threeroute_c29N6_generic.json`, `threeroute_c13N4_package.json`: `r1_minus_r2` |
| Cor 2.4 / Sec 4 | `dim K_N = 2N+1`, exact source quotient through N=30 | `kernel_span_rank_audit.json`, `full_matrix_source_quotient_audit.json` |
| Cor 2.7 / Sec 4 | pole-square, moment independence, and `dim = N-s-1` (45 exact rank cases) | `pole_neutral_survival_audit.json` |
| Sec 2.3 worked example | vector `v = (-0.0859452, 1.4749860, 0.7071068, 0, -2.1213203)`; contraction `0.049968414571096979730...` | `threeroute_c13N4_package.json`: `v`, `route1_closed_form` |
| Table 1 | zero-side rows M=32..512, raw and tail-corrected residuals; `gamma_512 = 826.90...` | `threeroute_c13N4_package.json`: `zero_side_rows` |
| Sec 2.3 | pole-neutral pole term vanishes to `9e-41` | `threeroute_c29N6_poleneutral.json`: `route2_pole` |
| Lemma 3.1 | `h_+(7) = 0.10717967... > 0` (Arb interval) | `arch_tail_order_check.json`: `h_plus_7_interval` |
| Lemma 3.1 | `h_+'(t) <= 1/t + 13/(10 t^2)` margin ladder | `arch_tail_exact_vs_asymptotic.json`: `derivative_envelope` |
| Thm 3.2 proof | rank-two density = derivative of the source (symbolic residual 0; direct error `9.5e-125`) | `arch_tail_dt_bridge_audit.json` |
| Thm 3.2 / Sec 4 | strict-TP smoke: 0 bad minors, N=2 (251) and N=3 (3431) | `arch_tail_order_check.json`: `strict_total_positivity_smoke` |
| T=800 paragraph | `rho N = 272.875270768... < 800` | `arch_tail_budget_c100_N200_T800.json`: `rho_N` |
| T=800 paragraph | interval budget `B_800 < 0.897` | `arch_tail_budget_c100_N200_T800.json`: `trace_budget_upper` |
| T=800 paragraph | exact `B_800 = 0.4203`; asymptotic prediction `0.4051` | `arch_tail_exact_vs_asymptotic.json`: row (100,200,800) |
| T=800 paragraph | closed-form bound of Cor 3.3(iii) gives `1.58` | direct evaluation of the displayed closed form (also reproducible from `arch_tail_budget.py` inputs) |
| T=800 paragraph | entrywise tail bound `3.18e-3` | `arch_tail_budget_c100_N200_T800.json`: `entry_abs_upper` |
| T=800 paragraph | cutoff-free Arb LDL^T: `n_+ = 401`, `n_- = 0` at 9000 bits | `c100_N200_arb_ldlt_prec9000_provenance.json` (regenerated 2026-07-02 by `arb_ldlt_certify.py`; original log retained) |
| T=800 paragraph | `B_T < 1e-59` requires `T ~ 8e62` | `arch_tail_exact_vs_asymptotic.json`: `certification_floor_solve` |
| Figure 2 | eigenvalue flow, strict order + within-B_T + monotone on the full T ladder; `lambda_min` negatives `-1.9e-2, -5.3e-7, -3.9e-10` at T=11,14,18; limit `9.7e-15` | `eigenflow_c13N4.json` |
| (package-level; not displayed in the manuscript) | original zero-side confirmation at M<=64, signed error `-8.798e-8` | `zero_side_values.json` |

## Boundary

The traced values are finite reproducibility data.  They do not prove RH,
Weil positivity, an asymptotic prime-location bound, a next-prime theorem, or
a factoring result.
