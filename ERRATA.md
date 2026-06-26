# Errata

Corrections to the accompanying paper, [arXiv:2605.20224](https://arxiv.org/abs/2605.20224), *High-Precision Approximation of Riemann Zeros via the Truncated Weil Form*. These corrections are incorporated into the revised version of the paper; this note records them for transparency.

## 2026-06-26 - negative-sign eigenvalue blocks are a finite-cutoff artifact

The paper reported small blocks of negative-sign even-sector eigenvalues in two places, and interpreted them as features of the finite-N truncation:

- at c=100 (abstract, §2.4, §6.6, the N-sweep table): a block of dps-stable negative-sign eigenvalues, taken to mean the matrix-level smallest eigenvalue is negative, attributed tentatively to condition-driven sign loss; and
- for L(s, χ₃) at c=23, 29 (§8.10, and the Future Directions section): negative even-sector eigenvalues, interpreted as a character-dependent positivity breakdown.

Both are artifacts of the finite archimedean integration cutoff T, not features of the operator. The negatives are stable under increasing working precision but not under increasing T:

- c=100 (T=800): absent at T=1200, and a cutoff-free evaluation of the archimedean entries leaves the even sector with no negative eigenvalues.
- χ₃ (T=400): re-running the exact original computation with only T varied, the c=23 negative (−6.46×10⁻²³) becomes positive by T=800, and the c=29 negative (−5.82×10⁻¹⁷) by T=1200; both even sectors are then non-negative.

dps-stability was mistaken for correctness; the correct diagnostic of a deep-spectrum value is agreement between two values of T. The "structural character dependence" reading of the χ₃ case, and the suggestion that the c=100 and χ₃ blocks arise from different mechanisms, are withdrawn: both are the same archimedean-truncation artifact.

No quantitative result changes. The reported smallest-positive branch is the genuine smallest eigenvalue, and the recovery of γ₁ through γ₁₀ to 307–329 matching digits at c=100, N=250 stands, as do the Aitken extrapolation and all convergence data.

Professor A. Connes prompted this investigation through his questions about the c=100 spectrum. The cutoff sensitivity was then independently identified by B. W. A. Silva ([zenodo.org/records/20650146](https://zenodo.org/records/20650146)) and is consistent with the naturally even, positive ground state reported by R. Andrews ([zenodo.org/records/20427500](https://zenodo.org/records/20427500)).
