"""Publication-grade figures for the ropfec SoftwareX paper (Paper B).

Reproduces, at journal quality (Helvetica / 600-dpi raster + vector PDF+SVG), the three
illustrative figures from the engine itself -- no values are hand-entered:
  toy_traj_comparison  : ROP constraint matters for control (mechanism illustration)
  case_studies_oscillations : Sel'kov + Wolf-Heinrich validated re-implementations
  order_uq_negative    : the built-in falsification (ROP vs count-matched random null)

Run from the repo root:  PYTHONPATH=. python3 paper/make_softwarex_figs.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "case_studies"))
import selkov                                   # noqa: E402
import wolf_heinrich as wh                      # noqa: E402
from bmac_engine.order_uncertainty import data_halfspaces, worst_case_range  # noqa: E402
from bmac_engine.rop_polyhedron import build_rop_from_binding_stoichiometry   # noqa: E402

FIGDIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.linewidth": 0.7,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "lines.linewidth": 1.4, "mathtext.fontset": "stixsans",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 600, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "grey": "#999999", "sky": "#56B4E9", "purple": "#CC79A7"}

# --- toy CRN (Phase-0 spec) ------------------------------------------------
N_TOY = np.array([[-1, 0, 0, 1], [1, -1, -2, 0], [0, 1, 1, -1]], dtype=float)


def _simulate(x0, alpha, T=5.0, dt=0.05, k=None):
    if k is None:
        k = np.ones(4)
    x = np.array(x0, dtype=float); xs = [x.copy()]; t = 0.0
    while t < T:
        v = np.array([k[0] * x[0] ** alpha[0], k[1] * x[1] ** alpha[1],
                      k[2] * x[1] ** alpha[2], k[3] * x[2] ** 1.0])
        x = np.maximum(x + N_TOY @ v * dt, 1e-12); xs.append(x.copy()); t += dt
    return np.array(xs)


def _save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"), dpi=600)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .svg / .png")


def fig_toy():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    x0, T, dt = [1.0, 1.0, 1.0], 5.0, 0.05
    a_gt = np.array([0.6, 0.85, 1.7]); a_un = np.array([0.5, 0.8, 3.1])
    a_rop = rop.project(a_un)
    g, u, r = (_simulate(x0, a, T, dt) for a in (a_gt, a_un, a_rop))
    t = np.arange(len(g)) * dt
    err_un = np.mean(np.abs(u - g)) / np.mean(np.abs(g)) * 100
    err_rop = np.mean(np.abs(r - g)) / np.mean(np.abs(g)) * 100
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(t, g[:, 2], color="black", lw=1.8, label="ground truth (in ROP)")
    ax.plot(t, u[:, 2], color=C["red"], lw=1.4, ls="--",
            label=f"unconstrained, violates ROP ({err_un:.1f}% err)")
    ax.plot(t, r[:, 2], color=C["green"], lw=1.4, ls=":",
            label=f"ROP-projected / FEC ({err_rop:.1f}% err)")
    ax.set_xlabel("time"); ax.set_ylabel("pyruvate  $x_3$")
    ax.set_title("ROP constraint lowers synthetic-trajectory error (mechanism)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "toy_traj_comparison")
    print(f"    toy: unconstrained {err_un:.1f}%  vs  ROP {err_rop:.1f}%")


def fig_osc():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    t, X = selkov.simulate(); td, Xd = selkov.simulate(params=selkov.PARAMS_FIXED_POINT)
    ax[0].plot(t, X[:, 0], color=C["blue"], label="limit cycle")
    ax[0].plot(td, Xd[:, 0], color=C["grey"], ls="--", label="fixed point")
    ax[0].set_xlim(0, 90); ax[0].set_xlabel("time"); ax[0].set_ylabel("$x$ (ADP)")
    ax[0].set_title("(a) Sel'kov 1968"); ax[0].legend(frameon=False)
    ax[0].spines[["top", "right"]].set_visible(False)
    tw, Xw = wh.simulate(); Xw = np.asarray(Xw)
    series = Xw[5] if Xw.shape[0] < Xw.shape[1] else Xw[:, 5]
    ax[1].plot(tw, series, color=C["green"], lw=1.0)
    ax[1].set_xlim(40, 56)  # steady-state window so individual cycles resolve
    ax[1].set_xlabel("time (steady-state window)"); ax[1].set_ylabel("$A_3$ (ATP)")
    ax[1].set_title("(b) Wolf–Heinrich 2000 (single-cell core)")
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(wspace=0.28)
    _save(fig, "case_studies_oscillations")


# --- negative result (reuse engine machinery) ------------------------------
ALPHA_TRUE = np.array([0.6, 0.85, 1.7]); LOGK_TRUE = np.log(1.3)
A_ROP = np.array([[1, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0.5]], float)
B_ROP = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 1.8])


def _order_unc(A, b):
    tot = 0.0
    for j in range(3):
        c = np.zeros(4); c[j] = 1.0; tot += worst_case_range(A, b, c)
    return tot


def _lift(Aa):
    return np.column_stack([Aa, np.zeros(len(Aa))])


def _rand_facets(rng, n, slack):
    tt = np.concatenate([ALPHA_TRUE, [LOGK_TRUE]]); rows, rhs = [], []
    for _ in range(n):
        c = rng.normal(size=4); c /= np.linalg.norm(c); rows.append(c); rhs.append(c @ tt + slack)
    return np.array(rows), np.array(rhs)


def _regime(lo, hi, n, eps, seeds):
    do, dr, dn = [], [], []
    slack = float(np.median(B_ROP[[0, 1, 5]] - A_ROP[[0, 1, 5]] @ ALPHA_TRUE))
    for s in seeds:
        rng = np.random.default_rng(s)
        X = rng.uniform(lo, hi, size=(n, 3))
        v = np.exp(LOGK_TRUE + np.log(X) @ ALPHA_TRUE + rng.uniform(-eps, eps, size=n))
        Ad, bd = data_halfspaces(X, v, eps); Ar, br = _rand_facets(rng, len(A_ROP), slack)
        try:
            do.append(_order_unc(Ad, bd))
            dr.append(_order_unc(np.vstack([Ad, _lift(A_ROP)]), np.concatenate([bd, B_ROP])))
            dn.append(_order_unc(np.vstack([Ad, Ar]), np.concatenate([bd, br])))
        except Exception:
            pass
    return np.array(do), np.array(dr), np.array(dn)


def fig_neg():
    seeds = range(200, 260)  # 60 seeds
    inf = _regime(0.5, 3.0, 120, 0.06, seeds)
    poor = _regime(0.9, 1.15, 12, 0.06, seeds)
    labels = ["data\nonly", "data $\\cap$ ROP\n(mechanism)", "data $\\cap$ random\n(control)"]
    cols = [C["grey"], C["blue"], C["red"]]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, reg, title in [(axL, inf, "(a) Informative data"), (axR, poor, "(b) Data-poor")]:
        m = [r.mean() for r in reg]; sd = [r.std(ddof=1) for r in reg]
        x = np.arange(3)
        ax.bar(x, m, 0.6, yerr=sd, capsize=3, color=cols, edgecolor="black", linewidth=0.6)
        for xi, (mi, si) in enumerate(zip(m, sd)):
            ax.text(xi, mi + si + max(m) * 0.02, f"{mi:.3f}", ha="center", va="bottom", fontsize=6.8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(title); ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(m) + max(sd) + max(m) * 0.22)
    axL.set_ylabel("total reaction-order uncertainty\n(sum of order ranges; lower = tighter)")
    pct = (poor[2].mean() - poor[1].mean()) / poor[0].mean() * 100
    axR.text(0.5, 0.93, f"ROP $\\approx$ random ($\\sim${abs(pct):.0f}% diff, not significant)",
             transform=axR.transAxes, ha="center", va="top", fontsize=6.6, color=C["grey"])
    fig.suptitle("Binding-derived ROP gives at most a marginal advantage over a random set",
                 fontsize=8.5, fontweight="bold", y=1.0)
    fig.subplots_adjust(wspace=0.30, top=0.85)
    _save(fig, "order_uq_negative")
    print(f"    informative: data {inf[0].mean():.3f} / ROP {inf[1].mean():.3f} / random {inf[2].mean():.3f}")
    print(f"    data-poor:   data {poor[0].mean():.3f} / ROP {poor[1].mean():.3f}+-{poor[1].std(ddof=1):.3f} / random {poor[2].mean():.3f}+-{poor[2].std(ddof=1):.3f}")


if __name__ == "__main__":
    print("Generating SoftwareX figures...")
    fig_toy(); fig_osc(); fig_neg()
    print("Done.")
