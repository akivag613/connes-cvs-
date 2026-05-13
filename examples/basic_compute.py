#!/usr/bin/env python3
"""
Minimal example: compute the CvS ground-state eigenvalue at cutoff c=13.

This reproduces the central numerical result from:
  Connes & van Suijlekom, arXiv:2511.23257

Settings here are tuned for fast reproducibility (T=400, dps=80,
typically ~1-3 minutes wall-clock on a single core, depending on
whether python-flint is installed for accelerated digamma):
  λ_min(c=13) ≈ 2.077e-59
  |γ₁ error|  ≈ 1.5e-55

This is the T=400 leg of the c=13 study (Paper 1 §4.4 Table 3.2).
The paper's HEADLINE c=13 result, used in the factor-of-1.3
agreement with Connes 2026 and CCM 2025 at c=13, is the T=800
datum |γ₁ error| ≈ 2.005e-55 — to reproduce that value, change
T=400 to T=800 below; the run takes roughly 4x longer.
"""

import mpmath as mp

from connes_cvs import build_galerkin_matrix, compute_ground_state, extract_zeros

# Working precision: 80 decimal digits.
mp.mp.dps = 80

# Known value of the first Riemann zero (imaginary part), to 80 digits.
GAMMA_1 = mp.mpf("14.134725141734693790457251983562470270784257115699243175685567460149963")

# Build the Galerkin matrix
# - c=13: prime cutoff
# - N=100: basis half-size (201x201 full matrix, 101-dim even sector)
# - T=400: archimedean integration truncation
# - dps=80: 80 decimal digits of precision
print("Building CvS Galerkin matrix for c=13 ...")
Q = build_galerkin_matrix(c=13, N=100, T=400, dps=80)
print(f"  Matrix size: {Q.rows} x {Q.cols}")

# Diagonalize to find the ground state
print("Computing ground-state eigenvalue ...")
lam_min, eigvec = compute_ground_state(Q)
print(f"  λ_min(c=13) = {mp.nstr(lam_min, 6)}")

# Extract the first three detected Riemann zeros.
# We pass L as an mpmath mpf (not a Python float) so the root-finding
# proceeds at full working precision. Passing math.log(13) would silently
# cap the accuracy at ~1e-17 (the double-precision ceiling).
L = mp.log(mp.mpf(13))
zeros = extract_zeros(eigvec, L=L, n_zeros=3)
gamma1_detected = zeros[0]["gamma_detected"]
gamma1_error = zeros[0]["error"]

print("\nResults:")
if gamma1_detected is not None:
    # Compute the error against the 80-digit reference GAMMA_1.
    err_full = abs(gamma1_detected - GAMMA_1)
    print(f"  Detected γ₁   = {mp.nstr(gamma1_detected, 40)}")
    print(f"  Known    γ₁   = {mp.nstr(GAMMA_1, 40)}")
    print(f"  |γ₁ error|    = {mp.nstr(err_full, 4)}")
else:
    print("  Detected γ₁  = None (findroot failed)")

print("\n  First 3 detected zeros (vs. published truth):")
for z in zeros:
    gd = z["gamma_detected"]
    gt = z["gamma_true"]
    gd_str = "N/A" if gd is None else mp.nstr(gd, 12)
    print(f"    k={z['k']}: γ_true={mp.nstr(gt, 12)}, γ_detected={gd_str}")
