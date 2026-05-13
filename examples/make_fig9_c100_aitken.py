"""Generate fig9_c100_aitken.pdf — the headline c=100 Aitken figure for Paper 1 v3.

Reads the public N-sweep JSONs at ``data/c100/`` and produces a two-panel
vector PDF that summarizes:

Left panel
  The four-point N-sweep at c=100, dps=500 (smallest-positive even-sector
  eigenvalue exponents), with horizontal reference lines at the two
  consecutive Aitken-Delta^2 anchors and the Connes 2026 §6.4 heuristic
  continuum prediction.

Right panel
  The consecutive first-difference ratios |Delta_{i+1}/Delta_i| with the
  geometric-convergence reference line at their average (0.836).

The Connes 2026 §6.4 prediction is computed from the displayed equation
immediately before footnote 19 of Connes (arXiv 2602.04022). The arXiv
HTML LaTeX source reads:

    1 - chi_2 ~ (2^14 / 3) * sqrt(2) * pi^5 * exp(-4*pi*e^L + 9*L/2)

i.e. the radical covers only the 2, then is multiplied by pi^5. At c=100
(L = log 100) this gives log_10(prediction) ≈ -530.38.

Usage:
    python examples/make_fig9_c100_aitken.py
    # writes _research/manuscripts/paper_1_v3_in_prep/figures/fig9_c100_aitken.pdf

Or with --out to control output path:
    python examples/make_fig9_c100_aitken.py --out /tmp/fig9.pdf
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
    """Try the source-repo path first, then the Zenodo-bundle path."""
    for candidate in (
        os.path.join(REPO, "data", "c100"),                  # source repo
        os.path.join(REPO, "data", "c100_verification"),     # Zenodo bundle
    ):
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not locate the c=100 result JSONs in either data/c100/ "
        "(source repo) or data/c100_verification/ (Zenodo bundle)."
    )


DATA = _resolve_data_dir()
DEFAULT_OUT = os.path.join(
    REPO, "_research", "manuscripts", "paper_1_v3_in_prep",
    "figures", "fig9_c100_aitken.pdf",
)

N_LIST = (100, 150, 200, 250)


def log10_from_decimal_string(s: str) -> float:
    s = s.strip().lower()
    if "e" in s:
        mantissa_str, exponent_str = s.split("e")
        return math.log10(float(mantissa_str)) + int(exponent_str)
    return math.log10(float(s))


def aitken(a: float, b: float, c: float) -> float:
    return a - (b - a) ** 2 / (c - 2.0 * b + a)


def connes_2026_section_6_4(c: int) -> float:
    """Connes 2026 §6.4 heuristic at integer cutoff c (see module docstring)."""
    prefactor = math.log10((2 ** 14) * math.sqrt(2) * (math.pi ** 5) / 3.0)
    L = math.log(c)
    return prefactor - 4.0 * math.pi * c / math.log(10.0) + 9.0 * L / (2.0 * math.log(10.0))


def load_x() -> dict[int, float]:
    runs = {
        N: json.load(open(os.path.join(DATA, f"results_c100_N{N}_T800_dps500_v020.json")))
        for N in N_LIST
    }
    return {N: log10_from_decimal_string(r["lambda_even"]) for N, r in runs.items()}


def make_figure(x: dict[int, float], out_path: str) -> None:
    a1 = aitken(x[100], x[150], x[200])
    a2 = aitken(x[150], x[200], x[250])
    connes = connes_2026_section_6_4(100)

    d = {N: x[N] for N in N_LIST}
    diffs = [d[150] - d[100], d[200] - d[150], d[250] - d[200]]
    ratios = [abs(diffs[1] / diffs[0]), abs(diffs[2] / diffs[1])]
    ratio_avg = sum(ratios) / len(ratios)

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

    # Broken-y-axis layout: two stacked panels sharing x.  The top panel
    # shows the data curve at full resolution (~140 OOM), the bottom panel
    # shows the three reference lines (Aitken anchors + Connes prediction)
    # at full resolution (~15 OOM band).  The break collapses the empty
    # middle region (~190 OOM) where there are no data points.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.0, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.08},
    )

    Ns = list(N_LIST)
    xs = [d[N] for N in Ns]

    # ---- Top panel: data curve only ----
    ax_top.plot(Ns, xs, "o-", color="#1f4e79", linewidth=1.8, markersize=7.5,
                markerfacecolor="white", markeredgewidth=1.5, zorder=3,
                label=r"$\log_{10}|\lambda_{\min}^{\rm even}(c{=}100,\,N)|$ "
                      r"at $\mathrm{dps}=500$")
    # Annotate data points on the upper-right side; N=100 below-right so the
    # label does not collide with the title.
    for i, (N, x_N) in enumerate(zip(Ns, xs)):
        dx, dy = (7, 7) if i > 0 else (7, -14)
        ax_top.annotate(f"${x_N:.2f}$", (N, x_N), xytext=(dx, dy),
                        textcoords="offset points", fontsize=8.8,
                        ha="left", color="#1f4e79")
    ax_top.set_ylim(min(xs) - 22, max(xs) + 28)
    ax_top.set_title(r"$c=100$ four-point $N$-sweep with Aitken-$\Delta^{2}$ anchors")
    ax_top.set_ylabel(r"$\log_{10}|\lambda^{\rm even}|$")
    # Legend in the upper-LEFT (the data curve starts upper-left and
    # descends to lower-right, so upper-left is below the curve and clean).
    ax_top.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02),
                  borderaxespad=0, framealpha=1.0, edgecolor="none",
                  handletextpad=0.6, handlelength=2.0, fontsize=8.5)

    # ---- Bottom panel: reference lines at proper scale ----
    span = max(connes, a1, a2) - min(connes, a1, a2)
    margin = max(2.5, span * 0.6)
    y_low = min(connes, a1, a2) - margin
    y_high = max(connes, a1, a2) + margin
    ax_bot.set_ylim(y_low, y_high)
    ax_bot.axhline(a1, linestyle="--", color="#d97706", linewidth=1.6, zorder=2)
    ax_bot.axhline(a2, linestyle="--", color="#15803d", linewidth=1.6, zorder=2)
    ax_bot.axhline(connes, linestyle=":",  color="#b91c1c", linewidth=2.0, zorder=2)
    # Direct labels on each line (placed at the right edge, with a white
    # background box so they read cleanly over the colored dashed lines).
    label_bbox = dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=1.0)
    x_lbl = 268
    ax_bot.text(x_lbl, a1,
                fr"$\{{100,150,200\}}$: ${a1:.2f}$",
                ha="left", va="center", fontsize=8.5, color="#d97706",
                bbox=label_bbox, zorder=4)
    ax_bot.text(x_lbl, a2,
                fr"$\{{150,200,250\}}$: ${a2:.2f}$",
                ha="left", va="center", fontsize=8.5, color="#15803d",
                bbox=label_bbox, zorder=4)
    ax_bot.text(x_lbl, connes,
                fr"Connes Sec. 6.4: ${connes:.2f}$",
                ha="left", va="center", fontsize=8.5, color="#b91c1c",
                bbox=label_bbox, zorder=4)
    ax_bot.set_xticks(Ns)
    ax_bot.set_xlim(85, 320)
    ax_top.set_xlim(85, 320)
    ax_bot.set_xlabel(r"Galerkin basis size $N$")
    ax_bot.set_ylabel(r"$\log_{10}|\lambda^{\rm even}|$ (extrap.)", fontsize=9.5)

    # Hide spines facing the break and draw the diagonal break marks.
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_bot.tick_params(top=False)
    d_kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=9,
                linestyle="none", color="gray", mec="gray", mew=1.0,
                clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **d_kw)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **d_kw)

    # ---- Small inset: consecutive Delta-ratio over-determination ----
    # Sits in the upper panel's upper-right corner, in the empty space ABOVE
    # the data curve (which descends from upper-left to lower-right; the
    # upper-right region above the curve is the cleanest empty area).
    axin = ax_top.inset_axes([0.55, 0.55, 0.36, 0.36])
    positions = [0.5, 1.5]
    axin.scatter(positions, ratios, s=55, color="#1f4e79", zorder=3,
                 edgecolor="white", linewidth=1.0)
    for pos, r in zip(positions, ratios):
        axin.annotate(f"${r:.4f}$", (pos, r), xytext=(0, 9),
                      textcoords="offset points", ha="center",
                      fontsize=8, color="#1f4e79")
    axin.axhline(ratio_avg, linestyle=":", color="#555", linewidth=0.9, zorder=1)
    axin.text(0.97, 0.05, fr"avg $= {ratio_avg:.3f}$",
              transform=axin.transAxes, ha="right", va="bottom",
              fontsize=7.8, color="#555")
    axin.set_xticks(positions)
    axin.set_xticklabels([r"$|\Delta_{2}/\Delta_{1}|$",
                          r"$|\Delta_{3}/\Delta_{2}|$"], fontsize=8)
    axin.set_xlim(0.0, 2.0)
    axin.set_ylim(ratio_avg - 0.012, ratio_avg + 0.012)
    axin.tick_params(axis="y", labelsize=7.5)
    axin.set_title("ratio over-determination", fontsize=8.5, pad=4)
    for spine in ("top", "right"):
        axin.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axin.spines[spine].set_linewidth(0.7)
    axin.patch.set_facecolor("white")

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=DEFAULT_OUT, help="output PDF path")
    args = p.parse_args(argv)

    if not os.path.isdir(DATA):
        print(f"error: data directory not found at {DATA}", file=sys.stderr)
        return 2
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    x = load_x()
    make_figure(x, args.out)

    print(f"wrote {args.out}")
    a1 = aitken(x[100], x[150], x[200])
    a2 = aitken(x[150], x[200], x[250])
    connes = connes_2026_section_6_4(100)
    print(f"  Aitken anchors: {a1:.2f}, {a2:.2f}")
    print(f"  Connes 2026 §6.4 prediction: {connes:.2f}")
    print(f"  gaps: {abs(a1 - connes):.2f} OOM, {abs(a2 - connes):.2f} OOM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
