"""Independent confirmation of the zero-source identity against zeta zeros.

The manuscript proves <v, Q_inf v> = sum_{z in Z_zeta^*} g_v(z) exactly.  This
script performs a small numerical check on a fixed c=13, N=4 regression vector:
it sums the induced test function g_v over the first M nontrivial zeta zeros
and confirms convergence to the same finite source scalar.  This is a
confirmation, not part of the proof.

g_v is real-even, so  sum_{z in Z_zeta^*} g_v(z) = 2 sum_{n>=1} g_v(gamma_n),
where 1/2 + i*gamma_n are the nontrivial zeros (gamma_n real under RH, which
holds far beyond the range used here).

Note: g_v(r) = int_0^L K_v(1 - y/L) cos(r y) dy.  For the M <= 64 zeros used
here, direct tanh-sinh quadrature is accurate to the displayed digits.  For
substantially larger M the interval should be subdivided at the half-period
nodes pi/r.  Run time is dominated by mpmath's zetazero.

Provenance of SOURCE_TOTAL: this constant is <v, Q_inf v> for the fixed vector
below, and is generated in-package by verify_dictionary_threeroute.py (route 1,
config c13N4_package: the closed-form CCM assembly at 40 digits; route 2, the
independent source-side evaluation, agrees to ~2e-15).  The coefficients xs
below are full-coefficient values with u_{+-k} = x_k/2; in the manuscript's
even-sector normalization (u_{+-k} = v_k/sqrt(2)) the same vector is
v_k = x_k/sqrt(2), the worked-example vector of Section 2.3.  The three-route
script extends this check to M = 512 zeros.
"""
import json
import os
import mpmath as mp

mp.mp.dps = 30

HERE = os.path.dirname(os.path.abspath(__file__))

c = 13
N = 4
L = mp.log(c)
Delta = L / (2 * mp.pi)
beta = L / (4 * mp.pi)
b2 = beta ** 2
SOURCE_TOTAL = mp.mpf("0.04996841457109697973028989877059503442314262931484")

# Reconstruct the fixed c=13, N=4 regression vector.
x2, x3, x4 = mp.mpf(1), mp.mpf(0), mp.mpf(-3)
r1 = -(x2 + x3 + x4)
r2 = -(x2 / (4 + b2) + x3 / (9 + b2) + x4 / (16 + b2))
sol = mp.lu_solve(mp.matrix([[1, 1], [1 / b2, 1 / (1 + b2)]]), mp.matrix([r1, r2]))
x0, x1 = sol[0], sol[1]
xs = [x0, x1, x2, x3, x4]

# symmetric full coefficients u_m (m=-N..N): u_0=x0, u_{+-k}=x_k/2
u = {0: x0}
for k in range(1, N + 1):
    u[k] = xs[k] / 2
    u[-k] = xs[k] / 2


def Kv(omega):
    """Exact finite Volterra kernel via the divided-difference contraction."""
    s = mp.mpf(0)
    for m in range(-N, N + 1):
        for n in range(-N, N + 1):
            if m == n:
                E = 2 * omega * mp.cos(2 * mp.pi * omega * m)
            else:
                E = (mp.sin(2 * mp.pi * omega * m) - mp.sin(2 * mp.pi * omega * n)) / (mp.pi * (m - n))
            s += u[m] * u[n] * E
    return s


def gv(r):
    """Evaluate g_v(r) by direct quadrature in the M <= 64 range."""
    return mp.quad(lambda y: Kv(1 - y / L) * mp.cos(r * y), [0, L])


checkpoints = [16, 32, 48, 64]
rows = []
running = mp.mpf(0)
for n in range(1, checkpoints[-1] + 1):
    running += gv(mp.im(mp.zetazero(n)))
    if n in checkpoints:
        zero_side = 2 * running
        diff = zero_side - SOURCE_TOTAL
        rows.append({
            "M": n,
            "zero_side": mp.nstr(zero_side, 20),
            "signed_diff_to_source": mp.nstr(diff, 6),
            "abs_diff_to_source": mp.nstr(abs(diff), 6),
        })
        print("M=%4d  2*sum g_v(gamma_n) = %s   source = %s   diff = %s" % (
            n, mp.nstr(zero_side, 14), mp.nstr(SOURCE_TOTAL, 14), mp.nstr(diff, 4)))

out = {
    "c": c,
    "N": N,
    "dps": mp.mp.dps,
    "source_total_tau_contraction": mp.nstr(SOURCE_TOTAL, 22),
    "rows": rows,
    "note": "2*sum_{n=1..M} g_v(gamma_n) converges to <v,Q_inf v>; confirmation only, not part of the proof.",
}
with open(os.path.join(HERE, "zero_side_values.json"), "w") as f:
    json.dump(out, f, indent=2)
print("wrote zero_side_values.json")
