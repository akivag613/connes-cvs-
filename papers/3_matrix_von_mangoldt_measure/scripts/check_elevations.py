#!/usr/bin/env python3
"""Reproducibility guard for the Paper 3 depth results (Sec 2 remark, Prop 3.4,
Thm 7.5): the divided-difference identity, the second-order event law, and the
Krein-string boundary-mass identity. Requires numpy.

 (L3) A_N(omega) = DD_X[g_omega], g_omega(x)=sin(2 pi x omega)/pi, X={-N..N}.
 (L4) second-derivative jump = +4 Lambda(q)/(sqrt q (log q)^2) 11^T (rank-one PSD).
 (L1) 1/W_+(z) - 1/W_-(z) = -a_q, z-independent, with W_pm the coincidence-Weyl
      function <1,(V_pm - z)^{-1} 1> of the velocities V_+ = V_- - a_q 11^T;
      and Lambda(q) = (sqrt q log q / 2)(1/W_- - 1/W_+).
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

PI = math.pi
np.seterr(divide="ignore", invalid="ignore")


def vonm(k):
    r, p, f = k, 2, {}
    while p * p <= r:
        while r % p == 0:
            f[p] = f.get(p, 0) + 1
            r //= p
        p += 1
    if r > 1:
        f[r] = f.get(r, 0) + 1
    return math.log(next(iter(f))) if len(f) == 1 else 0.0


def make_A(N):
    idx = np.arange(-N, N + 1)
    MN = (idx[:, None] - idx[None, :]).astype(float)

    def A(om):
        s = np.sin(2 * PI * om * idx); c = np.cos(2 * PI * om * idx)
        M = (s[:, None] - s[None, :]) / (PI * MN); np.fill_diagonal(M, 2 * om * c)
        return M
    return A, idx


def prime_part(N, u, A):
    d = 2 * N + 1; M = np.zeros((d, d)); k = 2
    while math.log(k) <= u + 1e-15:
        L = vonm(k)
        if L != 0.0:
            M += (-L / math.sqrt(k)) * A(1 - math.log(k) / u)
        k += 1
    return M


def main():
    failures = []
    res = {}

    # ---- L3: DD identity ----
    l3 = 0.0
    for N in [2, 4, 6]:
        A, idx = make_A(N)
        om = 0.123
        AN = A(om)
        g = np.sin(2 * PI * idx * om) / PI
        gp = 2 * om * np.cos(2 * PI * idx * om)
        MN = (idx[:, None] - idx[None, :]).astype(float)
        DD = (g[:, None] - g[None, :]) / MN
        np.fill_diagonal(DD, gp)
        l3 = max(l3, float(np.max(np.abs(AN - DD))))
    res["L3_DD_identity_max_err"] = l3
    if l3 > 1e-10:
        failures.append("L3")

    # ---- L4: second-order event law ----
    N = 6; d = 2 * N + 1; h = 1e-4; ones = np.ones(d)
    A, _ = make_A(N)
    l4 = 0.0; l4sign_ok = True
    for q in [3, 5, 7, 9, 25]:
        u0 = math.log(q)
        P = lambda uu: prime_part(N, uu, A)
        P0 = P(u0)
        Rpp = (2 * P0 - 5 * P(u0 + h) + 4 * P(u0 + 2 * h) - P(u0 + 3 * h)) / h ** 2
        Lpp = (2 * P0 - 5 * P(u0 - h) + 4 * P(u0 - 2 * h) - P(u0 - 3 * h)) / h ** 2
        j2 = Rpp - Lpp
        coeff = 4 * vonm(q) / (math.sqrt(q) * (math.log(q)) ** 2)
        l4 = max(l4, float(np.linalg.norm(j2 - coeff * np.outer(ones, ones)) / np.linalg.norm(coeff * np.outer(ones, ones))))
        if j2[N, N] <= 0:
            l4sign_ok = False
    res["L4_second_order_max_rel_err"] = l4
    res["L4_positive_sign"] = l4sign_ok
    if l4 > 1e-3 or not l4sign_ok:
        failures.append("L4")

    # ---- L1: Krein boundary-mass identity (exact algebra; background-independent) ----
    rng = np.random.default_rng(0)
    l1 = 0.0; l1lam = 0.0
    for N in [3, 5]:
        d = 2 * N + 1; ones = np.ones(d)
        B = rng.standard_normal((d, d)); Vm = (B + B.T) / 2
        for q in [3, 5, 9, 25]:
            aq = 2 * vonm(q) / (math.sqrt(q) * math.log(q))
            Vp = Vm - aq * np.outer(ones, ones)
            for z in [0.1j, 1 + 1j, -2 + 0.3j, 5 - 4j, 0.5 + 0j]:
                Wm = ones @ np.linalg.inv(Vm - z * np.eye(d)) @ ones
                Wp = ones @ np.linalg.inv(Vp - z * np.eye(d)) @ ones
                l1 = max(l1, abs((1 / Wp - 1 / Wm) - (-aq)))
                lam = (math.sqrt(q) * math.log(q) / 2) * (1 / Wm - 1 / Wp)
                l1lam = max(l1lam, abs(lam.real - vonm(q)) + abs(lam.imag))
    res["L1_krein_increment_max_err"] = l1
    res["L1_lambda_recovery_max_err"] = l1lam
    if l1 > 1e-9 or l1lam > 1e-8:
        failures.append("L1")

    payload = {"status": "PASS" if not failures else "FAIL",
               "description": "Depth results: L3 divided-difference identity, L4 "
               "second-order event law (+4 Lambda/(sqrt q (log q)^2) 11^T, PSD), "
               "L1 Krein boundary-mass identity 1/W_+ - 1/W_- = -a_q (z-independent).",
               "results": res, "failures": failures}
    out = Path(__file__).resolve().parents[1] / "artifacts" / "elevations_audit.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
