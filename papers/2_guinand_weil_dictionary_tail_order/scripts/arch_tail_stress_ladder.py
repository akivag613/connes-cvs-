#!/usr/bin/env python3
"""Stress ladder for the finite-T archimedean tail budget.

The theorem is proved in the manuscript.  This script is a release-package
guard for the artifact axes that mattered in the withdrawn T=800 claim: cutoff
T, Galerkin size N, and interval precision.  It reruns the interval tail-budget
calculation on a small deterministic ladder and records the resulting envelopes.
"""

from __future__ import annotations

import json
from pathlib import Path

from arch_tail_budget import compute_bounds


CELLS = [
    {"label": "c13_N40_T160_prec300", "c": 13, "N": 40, "T": 160, "prec": 300, "dyadic_count": 60},
    {"label": "c100_N100_T800_prec300", "c": 100, "N": 100, "T": 800, "prec": 300, "dyadic_count": 60},
    {"label": "c100_N150_T800_prec300", "c": 100, "N": 150, "T": 800, "prec": 300, "dyadic_count": 60},
    {"label": "c100_N200_T800_prec300", "c": 100, "N": 200, "T": 800, "prec": 300, "dyadic_count": 60},
    {"label": "c100_N200_T1600_prec300", "c": 100, "N": 200, "T": 1600, "prec": 300, "dyadic_count": 60},
    {"label": "c100_N200_T800_prec500", "c": 100, "N": 200, "T": 800, "prec": 500, "dyadic_count": 70},
]


def compact_row(cell: dict) -> dict:
    result = compute_bounds(
        c=cell["c"],
        nmax=cell["N"],
        cutoff=cell["T"],
        prec=cell["prec"],
        dyadic_count=cell["dyadic_count"],
    )
    return {
        "label": cell["label"],
        "c": result["c"],
        "N": result["N"],
        "dimension": result["dimension"],
        "T": result["T"],
        "prec_bits": result["prec_bits"],
        "dyadic_count": result["dyadic_count"],
        "threshold_pass": result["threshold_pass"],
        "rho_N": result["rho_N"],
        "trace_budget_upper": result["trace_budget_upper"],
        "entry_abs_upper": result["entry_abs_upper"],
        "global_log_trace_upper": result["global_log_trace_upper"],
        "global_log_entry_upper": result["global_log_entry_upper"],
        "h_plus_7_positive": result["h_plus_7_positive"],
        "tail_after_R": result["tail_after_R"],
        "tail_log_envelope_threshold": result["tail_log_envelope_threshold"],
        "tail_log_envelope_pass": result["tail_log_envelope_pass"],
    }


def main() -> None:
    rows = [compact_row(cell) for cell in CELLS]
    failures = [
        row["label"]
        for row in rows
        if (
            not row["threshold_pass"]
            or not row["h_plus_7_positive"]
            or not row["tail_log_envelope_pass"]
        )
    ]
    output = {
        "status": "PASS" if not failures else "FAIL",
        "purpose": "deterministic interval stress ladder over T, N, and precision; guard only, not a proof substitute",
        "cells": rows,
        "failures": failures,
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    Path("arch_tail_stress_ladder.json").write_text(text + "\n")
    print("archimedean tail stress ladder:", output["status"])
    for row in rows:
        print(
            "{label}: threshold={threshold_pass} trace={trace_budget_upper} entry={entry_abs_upper}".format(
                **row
            )
        )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
