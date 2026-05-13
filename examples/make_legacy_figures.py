"""Regenerate fig1-fig8 of Paper 1 v3 from the deposited JSON / pickle data.

All seven legacy figures are produced with TrueType embedded fonts
(pdf.fonttype=42) so that the resulting PDFs satisfy arXiv's requirement
that figures not use Type 3 fonts. The output is byte-different from the
prior Type-3 PDFs but visually equivalent (axis labels, colour scheme,
annotations, fit lines).

Usage:
    python examples/make_legacy_figures.py
    # writes fig1..fig5, fig7, fig8 PDFs to
    # _research/manuscripts/paper_1_v3_in_prep/figures/

Or to write to a different directory:
    python examples/make_legacy_figures.py --out-dir /tmp/figs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

BUNDLE = os.path.join(
    REPO, "_research", "manuscripts", "paper_1_v3_in_prep",
    "zenodo_v3_bundle", "data",
)
PICKLES = os.path.join(REPO, "_research", "pickles", "paper_canonical")
MULTI_ZERO = os.path.join(PICKLES, "results_L_multi_zero.json")
DEFAULT_OUT_DIR = os.path.join(
    REPO, "_research", "manuscripts", "paper_1_v3_in_prep", "figures",
)


def rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "STIX Two Text",
                       "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.unicode_minus": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def fig1(out: str) -> None:
    """log10 |gamma_1 error| vs L = log c, with dps floors."""
    rc()
    d = json.load(open(os.path.join(BUNDLE, "sweep_15cutoff", "results_15pt_T800.json")))
    results = d["results"]
    L = np.array([r["L"] for r in results])
    err = np.array([r["log10_abs_error"] for r in results])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(L, err, "o-", color="black", linewidth=1.2, markersize=6, zorder=3)
    ax.axhline(-79.0, linestyle="--", color="#3b82f6", linewidth=1.0,
               label="dps=80 floor")
    ax.axhline(-149.0, linestyle="--", color="#ef4444", linewidth=1.0,
               label="dps=150 floor")
    for i, c in enumerate([13, 23, 67]):
        idx = next((j for j, r in enumerate(results) if r["cutoff"] == c), None)
        if idx is None:
            continue
        dx, dy = (-3, 4), 8
        ax.annotate(fr"$c={c}$", (L[idx], err[idx]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=9, color="black")
    ax.set_xlabel(r"$L = \log c$")
    ax.set_ylabel(r"$\log_{10}|\gamma_{1}\,\mathrm{error}|$")
    ax.set_title(r"Convergence of CvS eigenvalue approximation ($T=800$)")
    ax.legend(loc="upper right", borderaxespad=0.6)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fig2(out: str) -> None:
    """gamma_1 / lambda vs log c, with linear fit."""
    rc()
    d = json.load(open(os.path.join(BUNDLE, "sweep_15cutoff", "results_15pt_T800.json")))
    results = d["results"]
    L = np.array([r["L"] for r in results])
    ratio = np.array([float(r["gamma_1_over_lambda"]) for r in results])

    slope, intercept = np.polyfit(L, ratio, 1)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(L, ratio, s=44, color="black", zorder=3)
    xs = np.linspace(L.min() - 0.05, L.max() + 0.05, 100)
    ax.plot(xs, slope * xs + intercept, "--", color="#666", linewidth=1.0,
            label=fr"fit: slope = {slope:.0f}")
    ax.set_xlabel(r"$\log c$")
    ax.set_ylabel(r"$\gamma_{1}/\lambda$")
    ax.set_title(r"Ratio $\gamma_{1}/\lambda$ vs cutoff ($T=800$)")
    ax.legend(loc="upper left", borderaxespad=0.6)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fig3(out: str) -> None:
    """15x15 pairwise eigenvector overlap heatmap."""
    rc()
    d = json.load(open(os.path.join(BUNDLE, "structural", "eigenvector_overlaps.json")))
    cutoffs = d["cutoffs"]
    m = np.array(d["overlap_matrix"])

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    im = ax.imshow(m, cmap="RdYlGn", vmin=0.94, vmax=1.0, origin="upper")
    ax.set_xticks(range(len(cutoffs)))
    ax.set_yticks(range(len(cutoffs)))
    ax.set_xticklabels(cutoffs, rotation=45)
    ax.set_yticklabels(cutoffs)
    ax.set_xlabel(r"Cutoff $c$")
    ax.set_ylabel(r"Cutoff $c$")
    ax.set_title(r"Eigenvector overlap $|\langle\eta_{c_{1}}|\eta_{c_{2}}\rangle|$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Overlap")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fig4(out: str) -> None:
    """Sobolev N-scaling at c=23 with saturation region."""
    rc()
    d = json.load(open(os.path.join(BUNDLE, "sweep_15cutoff", "results_N_sweep_c23.json")))
    results = d["results"]
    Ns = np.array([int(r["N"]) for r in results])
    le = np.array([float(r["log10_gamma_1_error"]) for r in results])
    logN = np.log10(Ns)

    fit_mask = Ns <= 80
    slope, intercept = np.polyfit(logN[fit_mask], le[fit_mask], 1)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(logN[fit_mask], le[fit_mask], s=70, color="#1f4e79", zorder=3,
               label=r"fit region ($N \leq 80$)")
    ax.scatter(logN[~fit_mask], le[~fit_mask], s=70, color="#b91c1c", zorder=3,
               label=r"saturation ($N \geq 100$)")
    xs = np.linspace(logN.min() - 0.02, logN.max() + 0.02, 100)
    ax.plot(xs, slope * xs + intercept, "--", color="#888", linewidth=1.1,
            label=fr"$s \approx {-slope/2:.0f}$,  $R^{{2}} > 0.9999$")
    ax.axvspan(2.0, logN.max() + 0.05, color="#fee2e2", zorder=1, label="saturation region")
    if (~fit_mask).any():
        last_two = np.where(~fit_mask)[0][-2:]
        for idx in last_two:
            extrap = slope * logN[idx] + intercept
            ax.annotate("", xy=(logN[idx], extrap),
                        xytext=(logN[idx], le[idx]),
                        arrowprops=dict(arrowstyle="<->", color="#b91c1c", lw=1.1))
            ax.text(logN[idx] + 0.015, (extrap + le[idx]) / 2,
                    f"{abs(extrap - le[idx]):.1f}", color="#b91c1c", fontsize=9)
    ax.set_xlabel(r"$\log_{10} N$")
    ax.set_ylabel(r"$\log_{10}|\lambda_{\rm even}|$")
    ax.legend(loc="upper right", borderaxespad=0.6)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fig5(out: str) -> None:
    """Nearest-neighbour spacing distribution at c=23 vs Poisson/GOE/GUE."""
    rc()
    pkl = pickle.load(open(os.path.join(PICKLES, "results_A_c23.pickle"), "rb"))
    rmt = json.load(open(os.path.join(BUNDLE, "structural", "results_rmt_analysis.json")))["23"]

    all_eig = sorted(float(e) for e in pkl["all_eigenvalues"] if abs(float(e)) > 0)
    # Take the positive bulk only: drop the ~deepest minuscule eigenvalue
    # (the ground state) and any negatives — RMT analysis is bulk only.
    positives = [e for e in all_eig if e > 0]
    bulk = positives[1:]  # drop the smallest (ground state); rest is bulk
    spacings = np.diff(bulk)
    s_normalized = spacings / np.mean(spacings)

    s = np.linspace(0.01, 4.0, 400)
    poisson = np.exp(-s)
    goe = (np.pi / 2) * s * np.exp(-np.pi * s ** 2 / 4)
    gue = (32 / np.pi ** 2) * s ** 2 * np.exp(-4 * s ** 2 / np.pi)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(s_normalized, bins=np.arange(0, 4.0, 0.5), density=True,
            color="#93c5fd", edgecolor="#1e3a8a", linewidth=0.7,
            label=r"CvS bulk ($c=23$)", zorder=2)
    ax.plot(s, poisson, "k-", linewidth=1.4, label="Poisson")
    ax.plot(s, goe, "--", color="#b91c1c", linewidth=1.2, label="GOE (Wigner)")
    ax.plot(s, gue, ":", color="#1d4ed8", linewidth=1.2, label="GUE")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(r"Normalised spacing $s$")
    ax.set_ylabel(r"$P(s)$")
    ax.set_title("Nearest-neighbour spacing distribution")
    ax.text(0.97, 0.95,
            f"Brody $\\beta < 0.05$\nBest fit: {rmt['best_fit']}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9c3",
                      edgecolor="#ca8a04", linewidth=0.7),
            fontsize=9)
    ax.legend(loc="center right", borderaxespad=0.6)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fig7(out: str) -> None:
    """Eigenvector deviation 1-|<.,.>| vs c_min, grouped by prime-cutoff gap."""
    rc()
    d = json.load(open(os.path.join(BUNDLE, "structural", "eigenvector_overlaps.json")))
    cutoffs = d["cutoffs"]
    m = np.array(d["overlap_matrix"])

    # Group pairs by prime-cutoff gap (difference in c values, since the
    # cutoffs themselves are primes the gap is exactly c2-c1).
    groups: dict[int, list[tuple[int, float]]] = {}
    for i in range(len(cutoffs)):
        for j in range(i + 1, len(cutoffs)):
            gap = cutoffs[j] - cutoffs[i]
            cmin = cutoffs[i]
            dev = 1.0 - m[i][j]
            groups.setdefault(gap, []).append((cmin, dev))

    # Pick the gaps that have at least 3 points and reasonable spread.
    pick = [2, 4, 6, 10, 18, 30]
    colors = ["#1d4ed8", "#dc2626", "#16a34a", "#9333ea", "#d97706", "#65a30d"]

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for color, gap in zip(colors, pick):
        if gap not in groups or len(groups[gap]) < 3:
            continue
        pts = sorted(groups[gap])
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        if (ys <= 0).any():
            continue
        log_x, log_y = np.log10(xs), np.log10(ys)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        # R^2 for the linear-log fit
        pred = slope * log_x + intercept
        ss_res = np.sum((log_y - pred) ** 2)
        ss_tot = np.sum((log_y - log_y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        xs_fit = np.linspace(xs.min(), xs.max(), 50)
        ax.scatter(xs, ys, s=40, color=color, zorder=3)
        ax.plot(xs_fit, 10 ** (slope * np.log10(xs_fit) + intercept), "--",
                color=color, linewidth=1.0,
                label=fr"gap {gap}: $\alpha = {-slope:.1f}$, $R^{{2}} = {r2:.4f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$c_{\min} = \min(c_{1}, c_{2})$")
    ax.set_ylabel(r"$1 - |\langle\eta_{c_{1}}|\eta_{c_{2}}\rangle|$")
    ax.set_title("Eigenvector deviation by prime-cutoff gap")
    ax.legend(loc="lower left", borderaxespad=0.6)
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fig8(out: str) -> None:
    """Multi-zero convergence: log10|gamma_k error| vs L=log c for k=1..5."""
    rc()
    d = json.load(open(MULTI_ZERO))
    cutoffs = d["cutoffs"]
    Ls = [math.log(c) for c in cutoffs]
    err = d["log10_errors"]  # dict[str(c)] -> list of 10

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    colors = ["#1d4ed8", "#dc2626", "#16a34a", "#9333ea", "#d97706"]
    markers = ["o", "s", "^", "D", "v"]
    # plot k=1..5
    for k_idx in range(5):
        true_first_label = {1: 14.13, 2: 21.02, 3: 25.01, 4: 30.42, 5: 32.94}[k_idx + 1]
        ys = [err[str(c)][k_idx] for c in cutoffs]
        ax.plot(Ls, ys, marker=markers[k_idx], color=colors[k_idx],
                linewidth=1.0, markersize=6,
                label=fr"$\gamma_{{{k_idx + 1}}} = {true_first_label}\ldots$")

    ax.axhline(-149.0, linestyle="--", color="#aaa", linewidth=0.8)
    ax.text(Ls[0] + 0.02, -145, "dps=150 floor", fontsize=8.5, color="#666",
            ha="left", style="italic")
    ax.axhline(-199.0, linestyle="--", color="#aaa", linewidth=0.8)
    ax.text(Ls[0] + 0.02, -195, "dps=200 floor", fontsize=8.5, color="#666",
            ha="left", style="italic")

    ax2 = ax.twiny()
    show_cs = [13, 19, 29, 41, 53]
    ax2.set_xticks([math.log(c) for c in show_cs])
    ax2.set_xticklabels([str(c) for c in show_cs])
    ax2.set_xlabel(r"cutoff $c$")

    ax.set_xlabel(r"$L = \log c$")
    ax.set_ylabel(r"$\log_{10}|\gamma_{k}\,\mathrm{error}|$")
    ax.legend(loc="upper right", borderaxespad=0.6)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_ylim(-215, -30)
    # Force the twin-axis limits to match
    ax.set_xlim(Ls[0] - 0.05, Ls[-1] + 0.05)
    ax2.set_xlim(ax.get_xlim())

    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


GENERATORS = [
    ("fig1_convergence.pdf", fig1),
    ("fig2_ratio_growth.pdf", fig2),
    ("fig3_overlap_matrix.pdf", fig3),
    ("fig4_sobolev.pdf", fig4),
    ("fig5_rmt_spacing.pdf", fig5),
    ("fig7_eigenvector_universality.pdf", fig7),
    ("fig8_multi_zero.pdf", fig8),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="output directory for figure PDFs")
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    for name, fn in GENERATORS:
        out_path = os.path.join(args.out_dir, name)
        try:
            fn(out_path)
            print(f"  wrote {out_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {name}: {exc}", file=sys.stderr)
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
