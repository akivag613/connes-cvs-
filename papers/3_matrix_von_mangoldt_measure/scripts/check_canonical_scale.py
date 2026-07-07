#!/usr/bin/env python3
"""Canonical-scale numerical demonstration for Paper 3.

Unlike the eight exact/modular guards, this is a floating-point demonstration on
the ACTUAL assembled finite CvS prime path at the program's canonical scale
N=200 (dimension 401), c up to 100. It confirms that at every prime power q<=100
the first-derivative jump of the finite path is (a) rank one, (b) equal to
-2 Lambda(q)/(sqrt q log q) times the all-ones matrix, and recovers Lambda(q)
from a single entry - i.e. Theorem 3.1 and Corollary 3.2 hold on the real
operator at scale, not only for small N.

The archimedean and pole blocks are analytic (Lemma 2.3) and cancel in the jump,
so the prime part carries the entire singular event; the jump is extracted with
a second-order one-sided finite difference of the (continuous) path at the kink.

Requires: numpy (see requirements.txt).
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

PI = math.pi
np.seterr(divide="ignore", invalid="ignore")


def von_mangoldt(k: int) -> float:
    if k < 2:
        return 0.0
    r, p, f = k, 2, {}
    while p * p <= r:
        while r % p == 0:
            f[p] = f.get(p, 0) + 1
            r //= p
        p += 1
    if r > 1:
        f[r] = f.get(r, 0) + 1
    return math.log(next(iter(f))) if len(f) == 1 else 0.0


def make_A(N: int):
    idx = np.arange(-N, N + 1)
    MN = (idx[:, None] - idx[None, :]).astype(float)  # m-n, 0 on diagonal

    def A(om: float):
        s = np.sin(2 * PI * om * idx)
        c = np.cos(2 * PI * om * idx)
        M = (s[:, None] - s[None, :]) / (PI * MN)
        np.fill_diagonal(M, 2 * om * c)
        return M

    return A, idx


def prime_part(N: int, u: float, A) -> np.ndarray:
    d = 2 * N + 1
    M = np.zeros((d, d))
    k = 2
    while math.log(k) <= u + 1e-15:
        L = von_mangoldt(k)
        if L != 0.0:
            M += (-L / math.sqrt(k)) * A(1 - math.log(k) / u)
        k += 1
    return M


def main() -> None:
    N = 200
    d = 2 * N + 1
    h = 1e-6
    A, _ = make_A(N)
    ones = np.ones(d)
    J = np.outer(ones, ones)
    prime_powers = [q for q in range(2, 101) if von_mangoldt(q) > 0]

    events = []
    for q in prime_powers:
        u0 = math.log(q)
        aq = 2 * von_mangoldt(q) / (math.sqrt(q) * math.log(q))
        P0 = prime_part(N, u0, A)
        Ph = prime_part(N, u0 + h, A)
        P2h = prime_part(N, u0 + 2 * h, A)
        Pmh = prime_part(N, u0 - h, A)
        P2mh = prime_part(N, u0 - 2 * h, A)
        Rs = (-3 * P0 + 4 * Ph - P2h) / (2 * h)  # P'(u0+)
        Ls = (3 * P0 - 4 * Pmh + P2mh) / (2 * h)  # P'(u0-)
        jump = Rs - Ls
        sv = np.linalg.svd(jump, compute_uv=False)
        rank1 = float(sv[1] / sv[0]) if sv[0] > 0 else 0.0
        rel = float(np.linalg.norm(jump + aq * J) / np.linalg.norm(aq * J))
        lam_rec = -math.sqrt(q) * math.log(q) / 2 * float(jump[N, N + 1])
        events.append({"q": q, "Lambda": von_mangoldt(q), "sv2_over_sv1": rank1,
                       "rel_err_jump": rel, "Lambda_recovered": lam_rec})

    max_rel = max(e["rel_err_jump"] for e in events)
    max_sv = max(e["sv2_over_sv1"] for e in events)
    max_lam = max(abs(e["Lambda_recovered"] - e["Lambda"]) for e in events)
    ok = max_rel < 1e-4 and max_sv < 1e-4 and max_lam < 1e-4
    payload = {
        "status": "PASS" if ok else "FAIL",
        "description": "Canonical-scale demonstration: at N=200 (dim 401), every prime "
        "power q<=100 has a rank-one first-derivative jump equal to "
        "-2 Lambda(q)/(sqrt q log q) 11^T, recovering Lambda(q).",
        "N": N, "n_events": len(events),
        "max_rel_err_jump_vs_minus_aq_11T": max_rel,
        "max_sv2_over_sv1_rank_one_witness": max_sv,
        "max_abs_Lambda_recovery_error": max_lam,
        "events": events,
    }
    out = Path(__file__).resolve().parents[1] / "artifacts" / "canonical_scale_audit.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps({k: payload[k] for k in payload if k != "events"}, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
