#!/usr/bin/env python3
"""Paper 3 figures. House style matches Paper 2 (Okabe-Ito, STIX serif, 6.5in).
Fig 1 (fig_event_signal.pdf): the finite path as an arithmetic event signal,
   and its second-derivative singular part = the matrix-valued von Mangoldt
   event measure (rank-one shocks at u=log q, height 2 Lambda(q)/(sqrt q log q)).
Fig 2 (fig_reconstruction.pdf): the confluent-Vandermonde node set {0,1^2,..,N^2}
   with double multiplicity, and the sharp 2N+1 event-Prony reconstruction window
   with its one-jet-short blind line.
Fig 3 (fig_uncertainty.pdf): the finite prime-edge uncertainty ceiling: visibility
   order 2e(x)+1 <= 4N+1 over even band-limited normals, uniquely attained by the
   central-difference stencil x_m = (-1)^{N-m} C(2N,N+m).
"""
import os
from fractions import Fraction as Fr
from math import comb, log, sqrt
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

mp.mp.dps = 30
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.0,
    "legend.fontsize": 7.8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "figure.dpi": 200, "savefig.dpi": 200, "pdf.fonttype": 42,
})
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"
INK = "#1a1a1a"
TW = 6.5
OUT = os.path.dirname(os.path.abspath(__file__))

def vonmangoldt(n):
    if n < 2: return 0.0
    m, p, f = n, 2, {}
    while p*p <= m:
        while m % p == 0: f[p] = f.get(p,0)+1; m//=p
        p += 1
    if m > 1: f[m] = f.get(m,0)+1
    return log(list(f)[0]) if len(f) == 1 else 0.0

def prime_powers_up_to(x):
    return [n for n in range(2, x+1) if vonmangoldt(n) > 0]

# ============================ FIGURE 1 ============================
def fig_event_signal():
    N = 5
    umax = log(30)
    us = [i*umax/1400 for i in range(15, 1401)]     # avoid u->0
    # scalar readout = (0,0) entry of the prime part: f(u) = -2 sum_q (Lam/sqrt q)(1-log q/u)
    def f(u):
        s = 0.0
        for q in prime_powers_up_to(int(mp.e**u + 1e-9)):
            if log(q) <= u:
                s += vonmangoldt(q)/sqrt(q)*(1 - log(q)/u)
        return -2*s
    ys = [f(u) for u in us]
    edges = [q for q in prime_powers_up_to(30) if log(q) <= umax]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(TW, 3.5), sharex=True,
                                   gridspec_kw={"height_ratios":[1.55,1.0], "hspace":0.14})
    # top: the signal
    for q in edges:
        ax1.axvline(log(q), color=GREY, lw=0.4, ls=(0,(1,2)), alpha=0.55, zorder=1)
    ax1.plot(us, ys, color=BLUE, lw=1.4, zorder=3, solid_capstyle="round")
    ax1.set_ylabel(r"$\left(Q_N(u)\right)_{00}$  (prime part)")
    ax1.tick_params(labelbottom=False)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.margins(x=0.01)
    ax1.annotate("continuous,\npiecewise real-analytic;\nslope kink at each $u=\\log q$",
                 xy=(log(7), f(log(7))), xytext=(log(3.05), min(ys)*0.62),
                 fontsize=7.2, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6,
                                 connectionstyle="arc3,rad=-0.2"))

    # bottom: the von Mangoldt event measure (rank-one shocks), height a_q
    for q in edges:
        aq = 2*vonmangoldt(q)/(sqrt(q)*log(q))
        ax2.plot([log(q), log(q)], [0, -aq], color=ORANGE, lw=1.5, zorder=3, solid_capstyle="round")
        ax2.plot([log(q)], [-aq], marker="v", ms=4.2, color=ORANGE, zorder=4)
    ax2.axhline(0, color=INK, lw=0.6)
    ax2.set_ylabel(r"jump in $\left(Q_N'\right)_{mn}$")
    ax2.set_xlabel(r"$u=\log c$")
    ax2.spines[["top","right"]].set_visible(False)
    # label q at each edge
    ymin = min(-2*vonmangoldt(q)/(sqrt(q)*log(q)) for q in edges)
    # q-labels above the zero line, horizontal, staggered into two rows so that
    # clustered prime powers (16/17, 23/25/27/29) never overlap
    last_u, row = -10.0, 0
    for q in edges:
        lu = log(q)
        row = (1 - row) if (lu - last_u) < 0.12 else 0
        ax2.annotate(f"${q}$", xy=(lu, 0),
                     xytext=(lu, (0.055 if row == 0 else 0.155) * abs(ymin)),
                     fontsize=6.6, color=GREY, ha="center", va="bottom")
        last_u = lu
    ax2.set_ylim(1.06 * ymin, 0.28 * abs(ymin))
    # mark prime powers 4,8,9 explicitly
    for q,txt in [(4,"$2^2$"),(8,"$2^3$"),(9,"$3^2$")]:
        aq = 2*vonmangoldt(q)/(sqrt(q)*log(q))
        ax2.annotate(txt, xy=(log(q), -aq), xytext=(log(q)+0.04, -aq-0.02*abs(ymin)),
                     fontsize=6.4, color=ORANGE, ha="left", va="top")
    ax2.text(0.985, 0.08, r"height $=\frac{2\Lambda(q)}{\sqrt{q}\,\log q}$",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.4, color=INK)
    ax2.margins(x=0.01)
    fig.savefig(os.path.join(OUT,"fig_event_signal.pdf"), bbox_inches="tight")
    plt.close(fig)

def _factor(n):
    m,p,f=n,2,[]
    while p*p<=m:
        while m%p==0: f.append(p); m//=p
        p+=1
    if m>1: f.append(m)
    return f
def _isprime(n):
    return len(_factor(n))==1

# ============================ FIGURE 2 ============================
def M_N(N):
    rows=[]
    for l in range(0,2*N+1):
        r=[1 if l==0 else 0]
        r+=[(k*k)**l for k in range(1,N+1)]
        r+=[(2*l+1)*(k*k)**l for k in range(1,N+1)]
        rows.append([Fr(x) for x in r])
    return rows
def rank_first_rows(M,k):
    # exact rank of first k rows via fraction gaussian elimination
    A=[row[:] for row in M[:k]]
    rank=0; ncol=len(M[0]); pr=0
    for c in range(ncol):
        piv=None
        for r in range(pr,len(A)):
            if A[r][c]!=0: piv=r; break
        if piv is None: continue
        A[pr],A[piv]=A[piv],A[pr]
        inv=A[pr][c]
        A[pr]=[x/inv for x in A[pr]]
        for r in range(len(A)):
            if r!=pr and A[r][c]!=0:
                fac=A[r][c]; A[r]=[a-fac*b for a,b in zip(A[r],A[pr])]
        pr+=1; rank+=1
        if pr==len(A): break
    return rank

def fig_reconstruction():
    N=4; d=2*N+1
    M=M_N(N)
    ks=list(range(0,2*N+2))
    rec=[rank_first_rows(M,k) for k in ks]     # dimension pinned down by first k odd jets

    fig,(axL,axR)=plt.subplots(1,2,figsize=(TW,2.5),gridspec_kw={"width_ratios":[1.05,1.25],"wspace":0.32})

    # LEFT: node set with multiplicity (annihilator P_N(S)=S prod (S-k^2)^2)
    nodes=[0]+[k*k for k in range(1,N+1)]
    mult=[1]+[2]*N
    axL.axhline(0,color=INK,lw=0.6)
    for x,mu in zip(nodes,mult):
        col = GREY if mu==1 else BLUE
        axL.plot([x,x],[0,mu],color=col,lw=2.0 if mu==2 else 1.4,solid_capstyle="round",zorder=3)
        axL.plot([x],[mu],marker="o",ms=4.5 if mu==2 else 3.5,color=col,zorder=4)
        axL.annotate((r"$0$" if x==0 else fr"${int(sqrt(x))}^2$"),xy=(x,0),xytext=(x,-0.28),
                     ha="center",va="top",fontsize=7.2,color=INK)
    axL.set_ylim(-0.9,2.7); axL.set_xlim(-1.4,N*N+1.6)
    axL.set_yticks([1,2]); axL.set_ylabel("node multiplicity")
    axL.set_xlabel(r"annihilator nodes of $P_N(S)=S\prod_{k=1}^N (S-k^2)^2$")
    axL.spines[["top","right"]].set_visible(False)
    axL.text(0.5,0.93,f"$N={N}$: one simple + $N$ double $=2N{{+}}1$ modes",
             transform=axL.transAxes,ha="center",va="top",fontsize=7.2,color=INK)

    # RIGHT: reconstruction staircase (recovered dim vs # odd jets used)
    axR.step(ks,rec,where="post",color=BLUE,lw=1.4,zorder=3)
    axR.plot(ks,rec,"o",ms=3.2,color=BLUE,zorder=4)
    axR.axhline(d,color=GREEN,lw=0.8,ls=(0,(4,2)))
    axR.axvline(d,color=GREY,lw=0.5,ls=(0,(1,2)))
    # highlight the sharp blind line at k=2N (one jet short)
    # both annotations sit in the empty lower-right triangle (below the staircase),
    # with short arrows up to their target points, so neither crosses the blue steps
    axR.annotate(f"one jet short ($2N={2*N}$):\n1-dim blind line",
                 xy=(2*N,rec[2*N]),xytext=(3.15,1.15),
                 fontsize=7.0,color=ORANGE,ha="left",va="center",
                 arrowprops=dict(arrowstyle="->",color=ORANGE,lw=0.7,connectionstyle="arc3,rad=-0.28"))
    axR.plot([2*N],[rec[2*N]],marker="o",ms=5,mfc="none",mec=ORANGE,mew=1.2,zorder=5)
    axR.annotate(f"$2N{{+}}1={d}$ jets:\nfull source recovered",xy=(d,d),xytext=(5.4,3.15),
                 fontsize=7.0,color=GREEN,ha="left",va="center",
                 arrowprops=dict(arrowstyle="->",color=GREEN,lw=0.7,connectionstyle="arc3,rad=-0.28"))
    axR.set_xlabel(r"number of odd event jets $m_0,\dots,m_{k-1}$")
    axR.set_ylabel("source coordinates pinned")
    axR.set_xticks(range(0,2*N+2,2)); axR.set_yticks(range(0,d+1,2))
    axR.set_xlim(-0.3,2*N+1.3); axR.set_ylim(-0.3,d+0.7)
    axR.spines[["top","right"]].set_visible(False)
    fig.savefig(os.path.join(OUT,"fig_reconstruction.pdf"),bbox_inches="tight")
    plt.close(fig)

# ============================ FIGURE 3 ============================
def even_moment(xv, idx, j):
    return sum((m**j)*xv[i] for i,m in enumerate(idx))
def visibility_order(xv, idx, N):
    for e in range(0,2*N+1,2):
        if even_moment(xv,idx,e)!=0:
            return 2*e+1
    return None

def fig_uncertainty():
    N=4; idx=list(range(-N,N+1))
    ceiling=4*N+1
    # extremal central-difference stencil
    xstar=[(-1)**(N-m)*comb(2*N,N+m) for m in idx]
    # a representative family of even normals (integer, symmetric): pick simple ones + extremizer
    fam=[]
    # single symmetric pair bumps x_{±k}=1 (and x0 chosen 0) plus flat, second difference, etc.
    fam.append(("flat $\\mathbf{1}$",[1]*(2*N+1)))
    fam.append(("$\\delta_0$",[1 if m==0 else 0 for m in idx]))
    fam.append(("2nd diff",[{-1:1,0:-2,1:1}.get(m,0) for m in idx]))
    fam.append(("4th diff",[ (-1)**(2-abs(m))*comb(4,2+m) if abs(m)<=2 else 0 for m in idx]))
    fam.append(("6th diff",[ (-1)**(3-abs(m))*comb(6,3+m) if abs(m)<=3 else 0 for m in idx]))
    fam.append(("central $2N$ stencil",xstar))
    labels=[n for n,_ in fam]
    orders=[visibility_order(v,idx,N) for _,v in fam]

    fig,(axL,axR)=plt.subplots(1,2,figsize=(TW,2.5),gridspec_kw={"width_ratios":[1.0,1.25],"wspace":0.40})

    # LEFT: the extremal stencil shape
    axL.axhline(0,color=INK,lw=0.6)
    for m,val in zip(idx,xstar):
        col = BLUE if val>=0 else ORANGE
        axL.plot([m,m],[0,val],color=col,lw=1.6,solid_capstyle="round",zorder=3)
        axL.plot([m],[val],marker="o",ms=3.6,color=col,zorder=4)
    axL.set_xlabel(r"index $m$"); axL.set_ylabel(r"$x_m$")
    axL.set_xticks(range(-N,N+1))
    axL.spines[["top","right"]].set_visible(False)
    axL.set_title(r"maximally blind normal $x_m=(-1)^{N-m}\binom{2N}{N+m}$",fontsize=7.8)

    # RIGHT: visibility order per normal, with ceiling
    ypos=list(range(len(fam)))[::-1]
    cols=[GREEN if o<ceiling else ORANGE for o in orders]
    for y,o,c in zip(ypos,orders,cols):
        axR.plot([1,o],[y,y],color=c,lw=2.4,solid_capstyle="round",zorder=3,alpha=0.9)
        axR.plot([o],[y],marker="o",ms=4.5,color=c,zorder=4)
        axR.annotate(str(o),xy=(o,y),xytext=(o+0.4,y),va="center",fontsize=7.2,color=INK)
    axR.axvline(ceiling,color=ORANGE,lw=0.8,ls=(0,(4,2)),zorder=2)
    axR.annotate(f"ceiling $4N{{+}}1={ceiling}$",xy=(ceiling,len(fam)-1),
                 xytext=(ceiling-0.3,len(fam)-0.55),ha="right",va="center",
                 fontsize=7.2,color=ORANGE)
    axR.set_yticks(ypos); axR.set_yticklabels(labels,fontsize=7.2)
    axR.set_xlabel(r"visibility order $\;2e(x)+1$")
    axR.set_xlim(0.3,ceiling+2.2); axR.set_ylim(-0.6,len(fam)-0.3)
    axR.spines[["top","right"]].set_visible(False)
    axR.xaxis.set_major_locator(MultipleLocator(4))
    fig.savefig(os.path.join(OUT,"fig_uncertainty.pdf"),bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    fig_event_signal(); print("wrote fig_event_signal.pdf")
    fig_reconstruction(); print("wrote fig_reconstruction.pdf")
    fig_uncertainty(); print("wrote fig_uncertainty.pdf")
