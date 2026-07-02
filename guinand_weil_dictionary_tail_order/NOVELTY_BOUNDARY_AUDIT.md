# Novelty Boundary Audit

Timestamp: 2026-06-26 02:02:26 IDT

Scope: this file records the package-local novelty and attribution boundary for

```text
A finite CvS-to-Guinand-Weil dictionary and archimedean tail order
for the truncated Weil form
```

It is not a proof substitute.  It is a collision and attribution guard for the
exact object claimed in the paper.

## Checked Object

The lean Paper 2 object is:

- a finite forward dictionary
  `v -> T_v -> K_v -> hat(g_v) -> g_v` from a real even finite CvS Galerkin
  vector to a compactly supported Guinand-Weil test function;
- the exact source-by-source identity showing that the cutoff-free finite CvS
  quadratic form equals the classical nontrivial-zero side for this induced
  test object;
- the exact finite source quotient of dimension `2N+1`;
- the positive-dimensional non-collapsing pole-neutral finite subfamily, used
  as a support corollary rather than a standalone priority claim;
- the finite-cutoff archimedean tail-order theorem obtained by differentiating
  the finite-T archimedean source;
- strict observability and strict total positivity of that isolated post-band
  archimedean tail increment.

The claim is not the classical Guinand-Weil formula, not the CvS or CCM finite
matrix construction, not the prime-piece Loewner representation, not a broad
Loewner/operator-monotone framework, not the closed-form archimedean-entry
correction, and not Cauchy total positivity itself.

## Verdict

The novelty wording supported by the audit is:

```text
To our knowledge, within the cited CvS, CCM, Silva, Suzuki, and bandlimited
Guinand-Weil boundary,
this paper gives an exact finite forward dictionary from a real even CvS
Galerkin coefficient vector to a compactly supported Guinand-Weil test
function.  For this induced test object, the cutoff-free finite CvS quadratic
form equals the classical nontrivial-zero sum, and finite signed sources factor
through an exact `2N+1` source quotient and admits a non-collapsing
positive-dimensional pole-neutral subfamily.  The same finite source
convention yields the finite-cutoff archimedean tail-order theorem, including
strict observability and strict total positivity for the isolated post-band
archimedean increment.
```

The verdict is `DEFENSIBLE WITH QUALIFIER`.  The audit does not support
stronger priority wording.

## Prior-Art Boundary

The manuscript should continue to credit:

- Guinand and Weil for the classical explicit formula;
- Bombieri, Connes, Connes-Consani, CvS, and CCM for the surrounding
  Weil-positivity, trace-formula, and finite-matrix framework;
- Suzuki for the screw-function framework;
- Silva for the prime-piece Loewner and operator-monotone boundary, plus the
  exact archimedean-entry correction;
- Simon, Bertola-Gekhtman-Szmigielski, and the classical total-positivity
  literature for the Cauchy total-positivity mechanism.

The package-level attribution boundary also records Andrews's independent
reproduction and convergence work; that work is not a load-bearing citation for
the compact theorem note.

The correction-facing motivation paragraph credits Andrews and Silva for the
cutoff-free and cutoff-sensitivity boundary.  It does not alter the novelty
claim above.

## Removed From Paper 2 Claim

The active Paper 2 release is narrowed to the results listed above.  Earlier
working-draft material outside that boundary is
not part of the Paper 2 novelty claim.

## Exact-Object Search Record

Earlier exact-object refreshes in this package found no collision for:

```text
all:"finite-cutoff archimedean tail order"
all:"CvS-to-Guinand-Weil"
all:"finite CvS-to-Guinand-Weil dictionary"
all:"finite source quotient" AND all:"Guinand-Weil"
all:"finite-cutoff archimedean total positivity"
all:"archimedean tail total positivity"
all:"Cauchy-Stieltjes archimedean tail"
```

The broader Zenodo, web, and GitHub checks returned classical or adjacent
material, Silva/Andrews attribution-boundary records, or noisy non-collisions,
not the finite CvS coefficient-vector-to-Guinand-Weil dictionary plus
tail-order object.

The machine-readable refresh artifact is
`novelty_refresh_20260625_202712.json`.

A fresh web sweep at 2026-06-25 20:42 IDT again found no exact-object
collision for the finite CvS-to-Guinand-Weil dictionary, the `2N+1` source
quotient, or the finite-cutoff archimedean tail-order theorem.  Exact-object
queries returned Paper 1, classical Guinand-Weil background, unrelated
GitHub or Zenodo noise, and adjacent but non-colliding material.

## Exact-Object Web Refresh, 2026-06-26 04:06:04 IDT

A narrow exact-object web pass was rerun before the lean-completeness refresh.
The checked query families were:

- `"finite CvS-to-Guinand-Weil dictionary"`;
- `"archimedean tail order" "Weil"`;
- `"finite source quotient" "Guinand-Weil"`;
- `"Cauchy-Stieltjes" "archimedean tail" "Weil"`;
- the same dictionary and tail-order phrases restricted to `arxiv.org`,
  `zenodo.org`, and `github.com`.

Result: no exact-object collision was found.  The arXiv searches returned
classical or adjacent explicit-formula and CvS/CCM material, including
Connes 2602.04022 and CCM 2511.22755, but not the finite coefficient-vector
dictionary or the post-band archimedean tail-order theorem.  The Zenodo and
GitHub restricted searches returned unrelated or noisy hits, including generic
explicit-formula material and non-number-theory repository text, not the object
claimed here.  This refresh supports the same qualified boundary as above; it
does not support stronger priority language.

If external-facing priority wording is strengthened, run a fresh exact-object
novelty check before release.

## Full Novelty + Recency Sweep, 2026-07-02

A complete cite-search / recency / exact-object sweep was run as part of the
2026-07-02 pre-release audit (independent adversarial review round; all
findings independently verified).  Results:

- **Cite-search (who cites us):** ecosystem actors identified and checked:
  Silva (7 Zenodo deposits, June 9-17; latest 20737111; nothing newer on
  Zenodo at sweep time), Andrews (20427500 v2.0; 20938200 Quantitative
  Convergence Law, June 26; orthogonal to this paper), Bradley Martin
  (github `bradleypmartin/dirichlet-bridge`, Hurwitz-zeta/Euler-Maclaurin
  interpolation; orthogonal, no Weil form), plus two new near-field actors:
  Bergen (Zenodo 20684022, Toeplitz recasting, June 14) and Ye (Zenodo
  20998327, regularized-determinant conditional framework, June 28).  None
  collides with any of the four results; none states the dictionary, the
  source quotient, the pole-neutral family, or the tail-order theorem.
- **Recency sweep (May-July 2026):** arXiv date-sorted queries over Weil
  positivity / truncated Weil / Guinand-Weil / zeta spectral triple / zeta
  cycles: only Groskin Paper 1 v2, Suzuki 2606.09096, Connes 2602.04022, and
  Connes-Consani 2606.06604 (unrelated).  No collision.
- **Suzuki 2606.09096 deep check (previously queued, now DONE):** the
  screw-function framework treats the same Weil form in a continuous-variable
  setting (Friedrichs extensions, small-a positivity/simplicity, entire
  boundary-form functions).  It contains NO finite Galerkin dictionary, no
  coefficient-vector-to-test-function transport, no archimedean-cutoff tail
  theorem, no total positivity, and no source quotient.  Framework-adjacent;
  zero collision.  It is cited in the manuscript introduction.
- **Bandlimited literature:** Chirre-Molero 2602.06199 and the
  Carneiro-Chandee-Milinovich line parametrize Guinand-Weil test functions by
  finite coefficient families operationally, but never the exact CvS-matrix
  zero-sum identity, the quotient, or the tail order.  Qualifier-level prior
  art; cited.
- **Classical mechanism:** the autocorrelation g = f * f~ device is Weil's;
  finite-dimensional restrictions of the Weil form are Yoshida 1992 (now
  cited); the divided-difference matrix structure is CvS Prop. 4.1 (now cited
  at equation level).

Per-result verdicts (confirmed at the same qualified ceiling as above):
dictionary + exact zero sum = NOVEL-WITH-QUALIFIER; 2N+1 source quotient =
NOVEL; pole-neutral positive-dimensional family = NOVEL-WITH-QUALIFIER
(Silva 20682834 has the adjacent rank-two pole splitting; cited); tail-order
theorem with strict total positivity and the explicit B_T budget =
NOVEL-WITH-QUALIFIER (finite-T sensitivity first documented empirically by
Silva 20650146, cited; the theorem, the strict TP statement, and the budget
asymptotics have no located precedent).
