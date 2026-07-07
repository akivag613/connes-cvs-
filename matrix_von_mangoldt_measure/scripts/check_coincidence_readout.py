#!/usr/bin/env python3
"""
Guard for Corollary (Coincidence-averaged weight readout), Section: the measure.

Checks two exact/statistical facts:
 (1) On a clean rank-one jump J0 = -a_q 11^T (a_q>0), the coincidence average
     -(1/(2N+1)^2) sum_{m,n} J0_{mn} recovers a_q exactly, and the second
     singular value of J0 is 0 (rank-one certificate).
 (2) Under iid additive noise E (variance sigma^2), the coincidence average has
     empirical variance ~ sigma^2/(2N+1)^2, i.e. a (2N+1)^2 reduction versus a
     single-entry readout (empirical variance ~ sigma^2). Verified by Monte
     Carlo with a fixed seed (deterministic).

Uses numpy only. Fixed RNG seed for reproducibility.
"""
import json
import numpy as np


def main():
    rng = np.random.default_rng(20260707)
    sigma = 1e-2
    trials = 40000
    a_q = 0.7314  # arbitrary positive amplitude
    rows = []
    failures = []
    for N in [1, 3, 5, 10]:
        d = 2*N + 1
        J0 = -a_q*np.ones((d, d))
        # (1) clean recovery + rank-one certificate
        rec = -J0.sum()/d**2
        sv = np.linalg.svd(J0, compute_uv=False)
        clean_rec_err = abs(rec - a_q)
        sv2_over_sv1 = sv[1]/sv[0] if len(sv) > 1 else 0.0
        # (2) noise variance reduction
        matched = np.empty(trials)
        single = np.empty(trials)
        for t in range(trials):
            E = rng.normal(0.0, sigma, size=(d, d))
            J = J0 + E
            matched[t] = -J.sum()/d**2
            single[t] = -J[0, 0]
        var_matched = matched.var()
        var_single = single.var()
        ratio = var_single/var_matched
        rows.append({
            "N": N, "d": d,
            "clean_recovery_err": clean_rec_err,
            "rank_one_sv2_over_sv1": float(sv2_over_sv1),
            "var_single": float(var_single),
            "var_matched": float(var_matched),
            "empirical_ratio": float(ratio),
            "predicted_ratio_d2": d**2,
        })
        if clean_rec_err > 1e-12:
            failures.append(f"N={N}: clean recovery error {clean_rec_err}")
        if sv2_over_sv1 > 1e-12:
            failures.append(f"N={N}: rank-one certificate sv2/sv1 {sv2_over_sv1}")
        # allow 8% Monte-Carlo tolerance on the variance-ratio estimate
        if abs(ratio - d**2)/d**2 > 0.08:
            failures.append(f"N={N}: variance ratio {ratio} vs predicted {d**2}")
    out = {
        "status": "PASS" if not failures else "FAIL",
        "description": "Coincidence-averaged weight readout: exact clean recovery + "
                       "rank-one (sv2/sv1) certificate, and Monte-Carlo (2N+1)^2 "
                       "variance reduction of the matched average vs a single entry.",
        "sigma": sigma, "trials": trials,
        "results": rows,
        "failures": failures,
    }
    print(json.dumps(out, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
