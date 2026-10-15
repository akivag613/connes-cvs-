# Errata

Corrections to the accompanying paper, [arXiv:2605.20224](https://arxiv.org/abs/2605.20224), *High-Precision Approximation of Riemann Zeros via the Truncated Weil Form*. This note records them for transparency. Entries are newest first. No entry changes a measured value from this project.

**Status in the manuscript.** The corrections dated through 2026-08-14 are incorporated in the published manuscript ([arXiv](https://arxiv.org/abs/2605.20224); [Zenodo concept DOI](https://doi.org/10.5281/zenodo.19546514)). The attribution update below is an overlay for Section 9.8 and is **not yet incorporated in that PDF**.

## 2026-10-15 - Martini replication attribution in Section 9.8

A. F. Martini's [revised replication](https://doi.org/10.5281/zenodo.21864191), reviewed at [v4, record 23393038](https://zenodo.org/records/23393038), corrects the account attributed to him in Section 9.8. The reported numerical replication uses `connes-cvs`; his separate from-scratch assembly has not been rerun with its corrected formulas and should not be described as a successful independent implementation.

At c=13, N=60, T=400, dps=80, his revised Table 2 reports a first-zero error of 4.215 × 10⁻⁵⁵ using v0.3.1, consistent with this paper's rounded 4.22 × 10⁻⁵⁵. The earlier roughly 10⁻¹⁷ errors arose from a float64 `L=log(c)` input, not from finite N or the stated working precision. His earlier “same form in different bases” explanation is withdrawn in v4 and is no longer attributed here. Accordingly, Section 9.8's introductory count should read **two independent implementations and two package reruns**, plus the separate table-row reproduction.

This updates third-party attribution, not this project's numerical data or its independently stated operator identities. The README and selected references use the revised account. The published manuscript remains unchanged pending a new revision.

## 2026-08-14 - the c = 100 negative block does not vanish at any finite cutoff we tested

The 2026-06-26 entry below stated that the c = 100 negative-sign block is "absent at T = 1200", and the manuscript stated that raising the cutoff to T = 1200 removes it. Both are wrong, and both are corrected.

Every c = 100, T = 1200 cell computed for this work still contains negative-sign eigenvalues: eight at N = 20 (dps 80), three at N = 100 (dps 150), two at N = 100 (dps 500), three at N = 150 (dps 500), and four at N = 200 (dps 500). At the N = 150, dps = 500 cell the paper actually reports, the five negatives at T = 800, at log₁₀|e| in {−40.66, −59.75, −96.75, −165.15, −223.69}, are replaced at T = 1200 by three at entirely different magnitudes, {−71.73, −101.66, −176.21}. What the underlying observation actually showed is therefore that the negative set *rearranges* with the cutoff rather than surviving it. In compressing that into an errata line, "those negatives vanish" became "the negatives are absent". Two further cautions apply to any such ladder. The count is not monotone in T at fixed working precision, since past some cutoff the arithmetic no longer resolves the deep spectrum at all; and the count at a fixed cell depends on the working precision, because eigenvalues at the precision floor are not meaningful. The counts above are raw counts at the stated precision.

**The conclusion is unchanged**, because it rests on the second clause, which is unaffected: a cutoff-free evaluation of the archimedean entries, certified by a rigorous Arb interval LDLᵀ factorization, leaves the even sector with no negative eigenvalues at N = 100, 150 and 200. The block is a finite-cutoff artifact. What was wrong is only the specific claim that T = 1200 is a large enough cutoff to remove it.

For completeness: because the omitted archimedean tail is a positive-definite increment (a result of the companion Guinand-Weil paper) and the cutoff-free block is certified positive definite, the block must be absent for all sufficiently large finite T. The paper's more general statements that the negatives are absent at larger T are therefore correct as written. What this entry corrects is the one place a specific finite cutoff was named. How large T must be was not determined here, and it is evidently far above the cutoffs tested.

No measured value changes. The L(s, χ₃) half of the 2026-06-26 entry is unaffected and was re-verified against its own cutoff sweep.

Raised by M. Osman in [issue #4](https://github.com/akivag613/connes-cvs-/issues/4).

## 2026-08-13 - two summary digit-increment ranges in Section 6.5

Section 6.5 summarises how the matching-digit counts at c = 100 respond to increasing the working precision and the Galerkin level. Both summary ranges were misstated. Recomputing the per-gamma_k counts from the deposited extraction files gives:

| change | correct | printed |
|---|---|---|
| dps 500 -> 1000 at N = 150 | 93-117 | 95-115 |
| N 150 -> 250 at dps = 500 | 181-203 | 179-201 |

These ranges are differences of the per-gamma_k counts, and those counts are unchanged and reported correctly throughout, including the 219-242 and 307-329 ranges quoted in the same paragraph, the latter also in the abstract. No measured value changes. Found in a pre-submission audit of the version 3 manuscript, where the ranges are corrected.

## 2026-08-12 - the Section 8.2 Paley-Wiener mechanism is withdrawn, and Table 14 is measured at T = 400

**Cutoff provenance.** The caption of Table 14 (`tab:sobolev-scaling`, the Sobolev regularity exponent s(c)) did not state the archimedean integration cutoff. Its rows are measured at **T = 400**, inherited from the per-cutoff N-convergence ladders, not at the T = 800 used for the 15-cutoff sweep. Section 8.2 had nevertheless derived A = 55 * 2*pi / 800 = 0.432 as though the slope were a T = 800 measurement. That substitution was incorrect; the caption now states T = 400.

**Mechanism withdrawn.** Section 8.2 proposed s = sigma_eff * T / (2*pi), which predicts s proportional to T at fixed c, and Section 11 preregistered exactly that test. The test has now been run at c = 23 on the N in {40, 60, 80} grid at dps = 150:

| T | s |
|---:|---|
| 400 | 46.140 |
| 800 | 46.031 |
| 1600 | 45.934 |

Across a fourfold increase in T the exponent moves by 0.206, where proportionality would require roughly a fourfold increase. The Paley-Wiener reading of Section 8.2 is therefore **withdrawn**, and the constant A = 0.432 has no meaning as derived.

The empirical scaling s(c) ~ 55 log c - 128 is a fit to the measured exponents, not a consequence of the withdrawn mechanism, and is unaffected; so are the measured values in Table 14, once correctly labelled T = 400. The three-cutoff test was first carried out by M. Osman ([github.com/Osman209/prime-number-studies](https://github.com/Osman209/prime-number-studies)), who reported it in [issue #2 of this repository](https://github.com/akivag613/connes-cvs-/issues/2), and it has since been reproduced independently here.

## 2026-08-12 - matching-digit count at c = 67

Section 6 defines the matching-digit count as the floor of -log10 of the absolute error. Applied to the c = 67, N = 100, dps = 200 datum, whose error is 1.478e-168, that definition gives **167** matching digits; the text printed 168, which labels the error by its decade instead. The measured error value is unchanged and is reported correctly throughout, and every other matching-digit count in the paper follows the stated definition exactly: 307 to 329 at c = 100, N = 250, dps = 500, and 219 to 242 at N = 150, dps = 1000. Only the c = 67 corroborative comparison figure is affected; no quantitative result changes.

## 2026-06-26 - negative-sign eigenvalue blocks are a finite-cutoff artifact

The paper reported small blocks of negative-sign even-sector eigenvalues in two places, and interpreted them as features of the finite-N truncation:

- at c=100 (abstract, §2.4, §6.6, the N-sweep table): a block of dps-stable negative-sign eigenvalues, taken to mean the matrix-level smallest eigenvalue is negative, attributed tentatively to condition-driven sign loss; and
- for L(s, χ₃) at c=23, 29 (§8.10, and the Future Directions section): negative even-sector eigenvalues, interpreted as a character-dependent positivity breakdown.

Both are artifacts of the finite archimedean integration cutoff T, not features of the operator. The negatives are stable under increasing working precision but not under increasing T:

- c=100 (T=800): the negative set rearranges with the cutoff rather than persisting, and a cutoff-free evaluation of the archimedean entries leaves the even sector with no negative eigenvalues. (The original wording of this line, "absent at T=1200", was wrong; see the 2026-08-14 entry above.)
- χ₃ (T=400): re-running the exact original computation with only T varied, the c=23 negative (−6.46×10⁻²³) becomes positive by T=800, and the c=29 negative (−5.82×10⁻¹⁷) by T=1200; both even sectors are then non-negative.

dps-stability was mistaken for correctness; the correct diagnostic of a deep-spectrum value is agreement between two values of T. The "structural character dependence" reading of the χ₃ case, and the suggestion that the c=100 and χ₃ blocks arise from different mechanisms, are withdrawn: both are the same archimedean-truncation artifact.

No quantitative result changes. The reported smallest-positive branch is the genuine smallest eigenvalue, and the recovery of γ₁ through γ₁₀ to 307–329 matching digits at c=100, N=250 stands, as do the Aitken extrapolation and all convergence data.

Professor A. Connes prompted this investigation through his questions about the c=100 spectrum. The cutoff sensitivity was then independently identified by B. W. A. Silva ([zenodo.org/records/20650146](https://zenodo.org/records/20650146)) and is consistent with the naturally even, positive ground state reported by R. Andrews ([zenodo.org/records/20427500](https://zenodo.org/records/20427500)).
