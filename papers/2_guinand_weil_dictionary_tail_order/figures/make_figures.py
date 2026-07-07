#!/usr/bin/env python3
"""Generate the two manuscript figures.

Figure 1 (fig_dictionary.pdf): the dictionary chain v -> K_v -> ghat_v -> g_v
for the worked example (c=13, N=4, pole- and moment-neutral vector), with the
first zeta ordinates marked on g_v.  Laid out as a 2x2 panel (row major:
v, K_v, ghat_v, g_v) so each panel renders at readable size at text width.

Figure 2 (fig_tailorder.pdf): eigenvalue flow lambda_j(Q_T^tot) -> lambda_j(Q_inf)
at c=13, N=4 (left), and the certified minimal-eigenvalue gap against the tail
budget B_T and its asymptotic form (right).

Both figures are drawn at the manuscript text width (6.5 in) so that
\\includegraphics[width=\\textwidth] applies no rescaling and the type renders
at its intended size.

Inputs: eigenflow_c13N4.json (produced by the eigenflow audit script) placed
alongside this script, or regenerated from the package scripts.  All curve
data for Figure 1 is computed here from the closed forms in the paper.

Requires: mpmath, matplotlib.
"""
import json
import os

import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = 30

# Drawn at text width (6.5 in): no \includegraphics rescaling, so these point
# sizes are the sizes that appear in the PDF.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,   # TrueType, no Type 3 fonts (arXiv-safe)
})
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"
TEXTWIDTH_IN = 6.5

# ---------------- worked example: c=13, N=4, pole- and moment-neutral ----------------
c, N = 13, 4
L = mp.log(c)
Delta = L/(2*mp.pi)
b2 = (L/(4*mp.pi))**2
x2, x3, x4 = mp.mpf(1), mp.mpf(0), mp.mpf(-3)
r1 = -(x2 + x3 + x4)
r2 = -(x2/(4 + b2) + x3/(9 + b2) + x4/(16 + b2))
sol = mp.lu_solve(mp.matrix([[1, 1], [1/b2, 1/(1 + b2)]]), mp.matrix([r1, r2]))
x0, x1 = sol[0], sol[1]
v = [x0, x1/mp.sqrt(2), x2/mp.sqrt(2), x3/mp.sqrt(2), x4/mp.sqrt(2)]

u = {0: v[0]}
for k in range(1, N + 1):
    u[k] = v[k]/mp.sqrt(2)
    u[-k] = v[k]/mp.sqrt(2)

alpha = {}
beta = {}
for k in range(-N, N + 1):
    s = mp.fsum(u[n]/(k - n) for n in u if n != k)
    alpha[k] = 2*u[k]*s/(mp.pi*1j)
    beta[k] = 2*u[k]**2

def Kv(w):
    return mp.re(mp.fsum((alpha[k] + beta[k]*w)*mp.exp(2j*mp.pi*k*w) for k in u))

def _int_poly_exp(al, be, a):
    if abs(a) < mp.mpf(10)**-8:
        tot_a = mp.mpc(0)
        tot_b = mp.mpc(0)
        for j in range(0, 25):
            cj = (1j*a)**j
            tot_a += cj/mp.factorial(j + 1)
            tot_b += cj/mp.factorial(j)*(mp.mpf(1)/(j + 2))
        return al*tot_a + be*tot_b
    ia = 1j*a
    e = mp.exp(ia)
    return al*(e - 1)/ia + be*((e*(ia - 1) + 1)/(ia**2))

def gv(z):
    th = z*L
    tot = mp.mpc(0)
    for k in u:
        tot += mp.exp(1j*th)/2*_int_poly_exp(alpha[k], beta[k], 2*mp.pi*k - th)
        tot += mp.exp(-1j*th)/2*_int_poly_exp(alpha[k], beta[k], 2*mp.pi*k + th)
    return mp.re(2*mp.pi*Delta*tot)

def ghat(xi):
    if abs(xi) > Delta:
        return mp.mpf(0)
    return mp.pi*Kv(1 - abs(xi)/Delta)

def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)

# ---------------- Figure 1: 2x2 dictionary chain ----------------
fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 4.45))

# (0,0) coefficients v_k
ax = axes[0, 0]
ks = list(range(N + 1))
vals = [float(x) for x in v]
ax.axhline(0, color=GREY, lw=0.5)
ml, sl, bl = ax.stem(ks, vals)
plt.setp(ml, color=BLUE, markersize=5.0)
plt.setp(sl, color=BLUE, lw=1.3)
plt.setp(bl, visible=False)
ax.set_xlabel(r"$k$")
ax.set_title(r"coefficients $v_k$")
ax.set_xticks(ks)
ax.margins(x=0.08)

# (0,1) Volterra kernel K_v(omega)
ax = axes[0, 1]
ws = [i/300 for i in range(301)]
ax.axhline(0, color=GREY, lw=0.5)
ax.plot(ws, [float(Kv(mp.mpf(w))) for w in ws], color=BLUE)
ax.set_xlabel(r"$\omega$")
ax.set_title(r"Volterra kernel $K_v(\omega)$")

# (1,0) Fourier weight ghat_v(xi)
ax = axes[1, 0]
xs = [(-1 + 2*i/400)*1.15*float(Delta) for i in range(401)]
ax.axhline(0, color=GREY, lw=0.5)
ax.plot(xs, [float(ghat(mp.mpf(x))) for x in xs], color=BLUE)
ax.axvline(float(Delta), color=GREY, lw=0.6, ls=":")
ax.axvline(-float(Delta), color=GREY, lw=0.6, ls=":")
ax.set_xlabel(r"$\xi$")
ax.set_title(r"weight $\widehat g_v(\xi)$, $\mathrm{supp}\subset[-\Delta,\Delta]$")

# (1,1) test function g_v(r) with zeros of zeta + tail inset
ax = axes[1, 1]
rs = [60*i/600 for i in range(601)]
gvals = [float(gv(mp.mpf(r))) for r in rs]
ax.axhline(0, color=GREY, lw=0.5)
ax.plot(rs, gvals, color=BLUE, zorder=2)
zpath = os.path.join(HERE, "zeta_zeros_512_dps30.json")
if os.path.exists(zpath):
    zeros = [mp.mpf(s) for s in json.load(open(zpath))]
else:
    zeros = [mp.im(mp.zetazero(n)) for n in range(1, 25)]
gam = [float(g) for g in zeros if float(g) <= 60]
ax.plot(gam, [float(gv(g)) for g in gam], "o", color=ORANGE, markersize=4.0,
        zorder=3, label=r"$g_v(\gamma_n)$")
ax.set_xlabel(r"$r$")
ax.set_title(r"test function $g_v(r)$ with zeros of $\zeta$")
ax.set_xlim(-1.5, 61.5)
ax.set_ylim(-0.35, 7.4)   # headroom above the peak (~6.2) for the legend band
ax.legend(frameon=False, loc="upper left", handletextpad=0.4, borderaxespad=0.3)
# zoom inset: the small tail oscillation carrying the zero sum (r >= 14, where
# g_v has fallen below the first lobes; ylim comfortably contains the curve so
# nothing is clipped at the inset border).
axin = ax.inset_axes([0.42, 0.30, 0.55, 0.52])
rs2 = [14 + 46*i/700 for i in range(701)]
axin.axhline(0, color=GREY, lw=0.4)
axin.plot(rs2, [float(gv(mp.mpf(r))) for r in rs2], color=BLUE, lw=0.9)
gam2 = [g for g in gam if g >= 14]
axin.plot(gam2, [float(gv(g)) for g in gam2], "o", color=ORANGE, markersize=3.0)
axin.set_xlim(14, 60)
axin.set_ylim(-0.05, 0.05)
axin.set_yticks([-0.025, 0.0, 0.025])
axin.set_xticks([20, 40, 60])
axin.tick_params(labelsize=6.5, length=2)
axin.set_facecolor("white")
for sp in axin.spines.values():
    sp.set_linewidth(0.5)

for ax in axes.flat:
    despine(ax)
fig.tight_layout(w_pad=1.4, h_pad=1.4, pad=0.5)
fig.savefig(os.path.join(HERE, "fig_dictionary.pdf"))
plt.close(fig)
print("wrote fig_dictionary.pdf")

# ---------------- Figure 2: eigenvalue flow + tail-order law ----------------
ef_path = os.path.join(HERE, "eigenflow_c13N4.json")
ef = json.load(open(ef_path))
Ts = [row["T"] for row in ef["ladder"]]
BTs = [float(mp.mpf(row["B_T"])) for row in ef["ladder"]]
lams_inf = [float(mp.mpf(s)) for s in ef["lambdas_inf"]]
lam_flow = [[float(mp.mpf(s)) for s in row["lambdas_T"]] for row in ef["ladder"]]
gaps = [float(mp.mpf(ef["lambdas_inf"][0]) - mp.mpf(row["lambdas_T"][0])) for row in ef["ladder"]]
max_gaps = [float(mp.mpf(row["max_gap"])) for row in ef["ladder"]]

rho = float(2*mp.pi/L)
def B_asym(T):
    return (2*N + 1)*rho*(mp.log(T/(2*mp.pi)) + 1)/(mp.pi**2*T)

fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.55))

# left: monotone eigenvalue flow with the lambda_min sign-crossing inset
ax = axes[0]
nj = len(lams_inf)
for j in range(nj):
    ax.plot(Ts, [lam_flow[i][j] for i in range(len(Ts))], "-o", ms=3,
            color=BLUE, alpha=0.45 + 0.55*j/(nj - 1), lw=1.0,
            label=r"$\lambda_j(Q_T^{\rm tot})$" if j == 0 else None)
    ax.axhline(lams_inf[j], color=ORANGE, lw=0.8, ls="--",
               label=r"$\lambda_j(Q_\infty)$" if j == 0 else None)
ax.set_xscale("log")
ax.set_xlabel(r"archimedean cutoff $T$")
ax.set_ylabel("eigenvalue")
ax.set_title(r"monotone eigenvalue flow, $c=13$, $N=4$ (even sector)")
ax.legend(frameon=False, loc="upper left", handletextpad=0.5, borderaxespad=0.4)
# inset (lower right, clear of the rising branch): lambda_min crosses zero from
# below, i.e. the finite-T spurious negatives lift to the positive limit.  The
# panel label sits inside the inset so it never overlaps the main curve.
axin = ax.inset_axes([0.46, 0.13, 0.50, 0.46])
lam_min_flow = [lam_flow[i][0] for i in range(len(Ts))]
axin.axhline(0, color=GREY, lw=0.5)
axin.plot(Ts, lam_min_flow, "-o", ms=2.5, color=BLUE, lw=1.0)
axin.axhline(lams_inf[0], color=ORANGE, lw=0.8, ls="--")
axin.set_xscale("log")
axin.set_yscale("symlog", linthresh=1e-14)
axin.set_ylim(-0.05, 1e-13)
axin.set_yticks([-1e-2, -1e-6, -1e-10, 0.0])
axin.tick_params(labelsize=6.0, length=2)
axin.set_facecolor("white")
# short tag in the empty lower-right corner (large T is near zero, so this
# region is clear of the curve and of the y-axis tick labels)
axin.text(0.965, 0.06, r"$\lambda_{\min}$, symlog", transform=axin.transAxes,
          ha="right", va="bottom", fontsize=6.5)
for sp in axin.spines.values():
    sp.set_linewidth(0.5)

# right: certified gap vs. budget, the log T / T law
ax = axes[1]
ax.loglog(Ts, max_gaps, "o-", color=BLUE, ms=4,
          label=r"$\max_j\,[\lambda_j(Q_\infty)-\lambda_j(Q_T^{\rm tot})]$")
ax.loglog(Ts, BTs, "s--", color=ORANGE, ms=4, label=r"tail budget $B_T$")
Tgrid = [Ts[0]*(Ts[-1]/Ts[0])**(i/100) for i in range(101)]
ax.loglog(Tgrid, [float(B_asym(t)) for t in Tgrid], ":", color=GREEN,
          label=r"$(2N{+}1)\rho\,(\log\frac{T}{2\pi}+1)/(\pi^2 T)$")
ax.set_xlabel(r"archimedean cutoff $T$")
ax.set_title(r"certified gap vs. budget: the $\log T/T$ law")
ax.legend(frameon=False, loc="lower left", handletextpad=0.5, borderaxespad=0.4)

for ax in axes:
    despine(ax)
fig.tight_layout(w_pad=1.6, pad=0.5)
fig.savefig(os.path.join(HERE, "fig_tailorder.pdf"))
plt.close(fig)
print("wrote fig_tailorder.pdf")
