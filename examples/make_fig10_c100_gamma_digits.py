"""Generate fig10_c100_gamma_digits.pdf — matching-digit recovery of γ_1..γ_10 at c=100.

Visualizes the three precision cells reported in Table~\\ref{tab:gamma-c100}
of Paper 1 v3:

    Cell A: N=150, dps=500  (tight findroot tolerance retest)
    Cell B: N=150, dps=1000 (precision-doubling retest)
    Cell C: N=250, dps=500  (deepest cell; paper's headline matching-digit count)

For each γ_k (k=1..10), plots the floor of -log10(|detected - reference|) where
the reference is mpmath.zetazero(k).imag computed at dps=400. Each cell's curve
falls monotonically with k (as expected for a Galerkin-truncated spectrum); the
gap between Cell A and Cell B isolates the precision-floor effect at fixed N,
and the gap between Cell B and Cell C isolates the N-extension effect at fixed
working precision dps=500.

Usage:
    python examples/make_fig10_c100_gamma_digits.py
    # writes ./figures/fig10_c100_gamma_digits.pdf (relative to current directory)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _resolve_data_dir() -> str:
    for candidate in (
        os.path.join(REPO, "data", "c100"),                  # source-repo layout
        os.path.join(REPO, "data", "c100_verification"),     # Zenodo-bundle layout
    ):
        # Require at least one of the gamma-extraction files to be present.
        if os.path.isdir(candidate) and any(
            os.path.exists(os.path.join(candidate, f))
            for f in (
                "c100_N250_dps500_gamma_extraction.json",
                "c100_N150_dps1000_gamma_extraction.json",
            )
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate the gamma_extraction JSONs in any of the "
        "candidate data directories."
    )


DATA_DIR = _resolve_data_dir()
DEFAULT_OUT = os.path.join("figures", "fig10_c100_gamma_digits.pdf")

CELLS = [
    {
        "label": r"$N=150$, $\mathrm{dps}=500$ (retight)",
        "file": "c100_N150_gamma_extraction_retight.json",
        "color": "#94a3b8",
        "marker": "o",
        "linestyle": "-",
        "digit_field": None,  # derive from log10_error
    },
    {
        "label": r"$N=150$, $\mathrm{dps}=1000$",
        "file": "c100_N150_dps1000_gamma_extraction.json",
        "color": "#0ea5e9",
        "marker": "s",
        "linestyle": "-",
        "digit_field": "matching_digits",
    },
    {
        "label": r"$N=250$, $\mathrm{dps}=500$ (headline cell)",
        "file": "c100_N250_dps500_gamma_extraction.json",
        "color": "#1d4ed8",
        "marker": "D",
        "linestyle": "-",
        "digit_field": "matching_digits",
    },
]


def load_cell(meta: dict) -> tuple[list[int], list[int]]:
    path = os.path.join(DATA_DIR, meta["file"])
    d = json.load(open(path))
    ks, digits = [], []
    for entry in sorted(d["gamma"], key=lambda e: e["k"]):
        ks.append(entry["k"])
        if meta["digit_field"] and meta["digit_field"] in entry:
            digits.append(entry[meta["digit_field"]])
        else:
            digits.append(int(math.floor(-entry["log10_error"])))
    return ks, digits


def make_figure(out_path: str) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "STIX Two Text",
                       "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    for meta in CELLS:
        ks, digits = load_cell(meta)
        ax.plot(ks, digits,
                marker=meta["marker"], linestyle=meta["linestyle"],
                color=meta["color"], linewidth=1.6, markersize=7,
                markerfacecolor="white", markeredgewidth=1.4,
                label=meta["label"])
        # Annotate the deepest cell only, to keep the figure readable.
        if "headline" in meta["label"]:
            for k, d in zip(ks, digits):
                ax.annotate(f"{d}", (k, d), xytext=(0, 9),
                            textcoords="offset points", ha="center",
                            fontsize=8.2, color=meta["color"])

    ax.set_xticks(range(1, 11))
    ax.set_xlabel(r"Riemann-zero index $k$")
    ax.set_ylabel(r"matching digits $= \lfloor -\log_{10}|\gamma_k - \gamma_k^{\rm exact}|\rfloor$")
    ax.set_title(
        r"$\gamma_k$ recovery at $c=100$:  three precision cells"
    )
    # Legend below the plot, in a single horizontal row so it never
    # collides with the three data curves (which span ~115 to ~330 in y).
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, borderaxespad=0, framealpha=1.0, edgecolor="none",
              columnspacing=1.6, handletextpad=0.6, handlelength=1.8)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=DEFAULT_OUT, help="output PDF path")
    args = p.parse_args(argv)
    if not os.path.isdir(DATA_DIR):
        print(f"error: data directory not found at {DATA_DIR}", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    make_figure(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
