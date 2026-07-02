#!/usr/bin/env python3
"""Interval tail-budget bounds for the finite-T archimedean tail theorem.

The theorem proves

    0 <= Q_infty - Q_T <= B_T I

after the cutoff T is past the Galerkin band and the scalar positivity
threshold.  This script evaluates the scalar trace and entry envelopes used in
the c=100, N=200, T=800 paragraph.  The dyadic finite pieces use Arb interval
evaluation of h_+.  The final tail uses the envelope h_+(tau) <= log(tau),
which the manuscript proves self-containedly for ALL tau >= 7 (Lemma 3.1:
the trigamma-series derivative bound h_+'(t) <= 1/t + 13/(10 t^2) integrated
from the Arb-certified value h_+(7) < 0.1072 gives h_+(t) <= log t - 8/5).
The historical threshold field records that the final dyadic endpoint also
exceeds 10^6, far above the proven validity edge t = 7.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flint import acb, arb, ctx


LOG_TAIL_THRESHOLD = arb("1000000")


def h_plus(tau: arb) -> arb:
    z = acb(arb("0.25"), tau / 2)
    return z.digamma().real - arb.pi().log()


def j_log_tail(a0: arb, n: int, rho: arb) -> arb:
    """Integral from a0 to infinity of log(rho*a)/(a-n)^2 da."""
    nn = arb(n)
    if n == 0:
        return ((rho * a0).log() + 1) / a0
    return (rho * a0).log() / (a0 - nn) + (a0 / (a0 - nn)).log() / nn


def trace_norm_integral(a0: arb, a1: arb, nmax: int) -> arb:
    total = arb(0)
    for n in range(-nmax, nmax + 1):
        nn = arb(n)
        total += 1 / (a0 - nn) - 1 / (a1 - nn)
        total += 1 / (a0 + nn) - 1 / (a1 + nn)
    return total


def trace_log_tail(a0: arb, nmax: int, rho: arb) -> arb:
    total = arb(0)
    for n in range(-nmax, nmax + 1):
        total += j_log_tail(a0, n, rho)
        total += j_log_tail(a0, -n, rho)
    return total


def entry_max_integral(a0: arb, a1: arb, nmax: int) -> arb:
    return 2 * (1 / (a0 - nmax) - 1 / (a1 - nmax))


def entry_log_tail(a0: arb, nmax: int, rho: arb) -> arb:
    return 2 * j_log_tail(a0, nmax, rho)


def up(x: arb) -> arb:
    return x.upper()


def arb_to_text(x: arb) -> str:
    return str(x).replace("\n", " ")


def compute_bounds(c: int, nmax: int, cutoff: int, prec: int, dyadic_count: int) -> dict:
    ctx.prec = prec
    pi = arb.pi()
    L = arb(c).log()
    rho = 2 * pi / L
    T = arb(cutoff)
    a_start = T / rho

    trace_sum = arb(0)
    entry_sum = arb(0)
    intervals = []

    A = T
    for k in range(dyadic_count):
        B = 2 * A
        a0 = A / rho
        a1 = B / rho
        hB = up(h_plus(B))
        trace_piece = hB / (pi * pi) * trace_norm_integral(a0, a1, nmax)
        entry_piece = hB / (pi * pi) * entry_max_integral(a0, a1, nmax)
        trace_sum += trace_piece
        entry_sum += entry_piece
        if k < 6 or k == dyadic_count - 1:
            intervals.append(
                {
                    "k": k,
                    "A": arb_to_text(A),
                    "B": arb_to_text(B),
                    "h_plus_B_upper": arb_to_text(hB),
                    "trace_piece_upper": arb_to_text(trace_piece),
                    "entry_piece_upper": arb_to_text(entry_piece),
                }
            )
        A = B

    R = A
    aR = R / rho
    tail_log_envelope_pass = bool(R > LOG_TAIL_THRESHOLD)
    trace_tail = trace_log_tail(aR, nmax, rho) / (pi * pi)
    entry_tail = entry_log_tail(aR, nmax, rho) / (pi * pi)
    trace_total = trace_sum + trace_tail
    entry_total = entry_sum + entry_tail

    global_trace_log = trace_log_tail(a_start, nmax, rho) / (pi * pi)
    global_entry_log = entry_log_tail(a_start, nmax, rho) / (pi * pi)
    h7 = h_plus(arb(7))

    return {
        "status": "INTERVAL-DYADIC-UPPER, final tail bounded by the envelope h_plus(tau)<=log(tau), proved in the manuscript (Lemma 3.1) for all tau>=7",
        "c": c,
        "N": nmax,
        "dimension": 2 * nmax + 1,
        "T": cutoff,
        "prec_bits": prec,
        "dyadic_count": dyadic_count,
        "L": arb_to_text(L),
        "rho": arb_to_text(rho),
        "rho_N": arb_to_text(rho * nmax),
        "two_rho_N": arb_to_text(2 * rho * nmax),
        "a_start": arb_to_text(a_start),
        "threshold_pass": bool(T > rho * nmax and T > arb(7)),
        "h_plus_7_interval": arb_to_text(h7),
        "h_plus_7_positive": bool(h7.lower() > 0),
        "tail_after_R": arb_to_text(R),
        "tail_log_envelope_threshold": arb_to_text(LOG_TAIL_THRESHOLD),
        "tail_log_envelope_pass": tail_log_envelope_pass,
        "trace_budget_upper": arb_to_text(trace_total),
        "entry_abs_upper": arb_to_text(entry_total),
        "global_log_trace_upper": arb_to_text(global_trace_log),
        "global_log_entry_upper": arb_to_text(global_entry_log),
        "finite_dyadic_trace_sum": arb_to_text(trace_sum),
        "finite_dyadic_entry_sum": arb_to_text(entry_sum),
        "log_tail_trace_remainder": arb_to_text(trace_tail),
        "log_tail_entry_remainder": arb_to_text(entry_tail),
        "final_tail_bound": "uses the envelope h_plus(tau) <= log(tau), proved in manuscript Lemma 3.1 for all tau >= 7",
        "sample_intervals": intervals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=int, default=100)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--T", type=int, default=800)
    parser.add_argument("--prec", type=int, default=300)
    parser.add_argument("--dyadic-count", type=int, default=80)
    parser.add_argument("--json-out", type=Path, default=Path("arch_tail_budget_c100_N200_T800.json"))
    args = parser.parse_args()

    result = compute_bounds(args.c, args.N, args.T, args.prec, args.dyadic_count)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
