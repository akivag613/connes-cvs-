# Verification package

Reproducibility guards for *A matrix-valued von Mangoldt measure in the finite
Connes-van Suijlekom path* (A. Groskin). The proofs in the paper are algebraic;
these scripts are independent reproducibility guards for signs, indexing,
determinant factors, and finite-field behavior. They are not proof substitutes.

There are thirteen guards. Ten use only the Python 3 standard library (no
third-party dependencies); three (`check_universal_jet`, `check_coincidence_readout`,
`check_dirichlet_readout`) use `sympy` / `numpy`. Each writes a JSON artifact with
`"status": "PASS"` and exits non-zero on any failure. All run in seconds to a few
minutes each.

## Run

```
python3 scripts/check_rank_one_jump.py
python3 scripts/check_uncertainty_ceiling.py
python3 scripts/check_canonical_scale.py
python3 scripts/check_elevations.py
python3 scripts/check_universal_jet.py
python3 scripts/check_coincidence_readout.py
python3 scripts/check_dirichlet_readout.py
python3 scripts/check_event_jet_largeN.py
python3 scripts/check_event_jet_determinant.py
python3 scripts/check_event_jet_recurrence.py
python3 scripts/check_event_prony_reconstruction.py
python3 scripts/check_source_quotient_and_transport.py
python3 scripts/check_spectral_barrier_jump.py
```

(In the arXiv ancillary layout the scripts and their JSON artifacts sit together
in `anc/`; adjust the paths accordingly.)

## What each guard verifies

Theorem numbers refer to the compiled `main.pdf`.

| Guard | Manuscript result |
|---|---|
| `check_rank_one_jump.py` | Lemma 2.3 (edge derivative) and Theorem 3.1 (the `-2 Lambda(q)/(sqrt q log q) 11^T` first-derivative jump), by evaluating `A_N` and finite-differencing the assembled prime path for `q in {3,4,5,7,8,9,25}`, `N<=6`. |
| `check_uncertainty_ceiling.py` | Theorem 6.1 (finite prime-edge uncertainty ceiling): even-moment Vandermonde invertible, the centered finite-difference stencil uniquely attains `e=2N` / visibility order `4N+1`, and a family realizes `1,5,...,4N+1`. Exact rational arithmetic, `N<=10`. |
| `check_canonical_scale.py` | Theorem 3.1 at the program's canonical scale: at `N=200` (dimension 401) every prime power `q<=100` has the rank-one first jump `-2 Lambda(q)/(sqrt q log q) 11^T`, floating point. |
| `check_elevations.py` | The divided-difference identity for `A_N`, Proposition 3.3 (second-order event law, `+4 Lambda/(sqrt q (log q)^2) 11^T`, PSD), and Theorem 7.5 (Krein boundary-mass increment `1/W_+ - 1/W_- = -a_q`, z-independent). |
| `check_universal_jet.py` | Proposition 4.2 (closed form of the universal jet `B_{r,N}(u0)`), exact symbolic (sympy), entrywise, orders `r<=5`, incl. the `r=1,2` specializations. |
| `check_coincidence_readout.py` | Corollary 3.4 (coincidence-averaged weight readout): exact clean recovery + rank-one certificate, and the `(2N+1)^2` variance reduction of the matched average (Monte Carlo, fixed seed). |
| `check_dirichlet_readout.py` | Corollary 8.1 (residue-class readout across the Dirichlet family): character-orthogonality reconstruction of `Lambda(q) 1[q==a mod m]` from the per-character first jumps, cyclic moduli `m<=13`. |
| `check_event_jet_largeN.py` | Theorem 5.2 edge-jet rank and Lemma 5.3 transport rank, exact modular, `N<=200`. |
| `check_event_jet_determinant.py` | Theorem 5.2 determinant `(-1)^{N(N-1)/2} 2^N prod k^6 prod (j^2-i^2)^4` and the Lemma 5.3 `tau`-transport determinant, exact modular over four prime fields, `N<=200`. |
| `check_event_jet_recurrence.py` | Theorem 5.5(iii) recurrence `S prod (S-k^2)^2`, exact modular over four prime fields, `N<=1000`. |
| `check_event_prony_reconstruction.py` | Theorem 5.5 window / blind line / recurrence residues, exact modular, `N<=120` over four fields (`N=160,200` over one). |
| `check_source_quotient_and_transport.py` | Lemma 5.1 source quotient + Lemma 5.3 transport, exact modular, `N<=200`. |
| `check_spectral_barrier_jump.py` | Theorem 7.4 elementary-symmetric barrier deceleration, exact integer, 50 positive-definite cases. |

## Figures

`make_figures.py` regenerates the three manuscript figures
(`fig_event_signal.pdf`, `fig_reconstruction.pdf`, `fig_uncertainty.pdf`); it
needs `matplotlib`, `mpmath`, `numpy` (see `requirements.txt`).

## License

Manuscript: CC-BY-4.0 (see `LICENSE-PAPER-CC-BY-4.0.txt`). The verification
scripts may be used freely for reproduction.
