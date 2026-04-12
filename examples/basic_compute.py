#!/usr/bin/env python3
"""
Minimal example: compute the CvS ground-state eigenvalue at cutoff c=13.

This reproduces the central numerical result from:
  Connes & van Suijlekom, arXiv:2511.23257

Expected output (at T=400, dps=80):
  λ_min(c=13) ≈ 2.077e-59
  |γ₁ error|  ≈ 1.5e-55
"""

import math

from connes_cvs import build_galerkin_matrix, compute_ground_state, extract_zeros

# Known value of the first Riemann zero (imaginary part)
GAMMA_1 = 14.134725141734693790457251983562

# Build the Galerkin matrix
# - c=13: cutoff parameter
# - N=100: basis half-size (201x201 matrix)
# - T=400: archimedean integration truncation
# - dps=80: 80 decimal digits of precision
print("Building CvS Galerkin matrix for c=13 ...")
Q = build_galerkin_matrix(c=13, N=100, T=400, dps=80)
print(f"  Matrix size: {Q.rows} x {Q.cols}")

# Diagonalize to find the ground state
print("Computing ground-state eigenvalue ...")
lam_min, eigvec = compute_ground_state(Q)
print(f"  λ_min(c=13) = {lam_min:.6e}")

# Extract the first Riemann zero from the eigenvector
zeros = extract_zeros(eigvec, L=math.log(13), n_zeros=3)
gamma1_detected = zeros[0]['gamma_detected']
gamma1_error = zeros[0]['error']

print(f"\nResults:")
if gamma1_detected is not None:
    print(f"  Detected γ₁  = {gamma1_detected:.15f}")
else:
    print(f"  Detected γ₁  = None (findroot failed)")
print(f"  Known γ₁     = {GAMMA_1:.15f}")
if gamma1_error is not None:
    print(f"  |γ₁ error|   = {gamma1_error:.4e}")
else:
    print(f"  |γ₁ error|   = N/A")
print(f"\n  First 3 zeros:")
for z in zeros:
    gd = z['gamma_detected']
    print(f"    k={z['k']}: γ_true={z['gamma_true']:.10f}, γ_detected={'N/A' if gd is None else f'{gd:.10f}'}")
