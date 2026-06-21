"""Generate the additional figures for the honest ropfec preprint:
  (1) case-study oscillations (Selkov + Wolf-Heinrich) -> paper/figures/case_studies_oscillations.png
  (2) the honest negative result on ROP-vs-data-vs-random order sets
      -> paper/figures/order_uq_negative.png
Run: PYTHONPATH=. python3 experiments/make_paper_figs.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "case_studies"))
import selkov  # noqa: E402
import wolf_heinrich as wh  # noqa: E402
from bmac_engine.order_uncertainty import (  # noqa: E402
    data_halfspaces,
    worst_case_range,
)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

ALPHA_TRUE = np.array([0.6, 0.85, 1.7])
LOGK_TRUE = np.log(1.3)
A_ROP = np.array(
    [[1, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0.5]], dtype=float
)
B_ROP = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 1.8])


def fig_oscillations():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    # Selkov limit cycle vs damped fixed point
    t, X = selkov.simulate()
    td, Xd = selkov.simulate(params=selkov.PARAMS_FIXED_POINT)
    ax[0].plot(t, X[:, 0], lw=1.2, label="limit cycle (a=0.06, b=0.6)")
    ax[0].plot(td, Xd[:, 0], lw=1.2, ls="--", color="gray", label="fixed point (a=0.1, b=1.0)")
    ax[0].set_xlim(0, 200)
    ax[0].set_xlabel("time"); ax[0].set_ylabel("x (ADP)")
    ax[0].set_title("(a) Sel'kov 1968"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # Wolf-Heinrich oscillation
    tw, Xw = wh.simulate()
    Xw = np.asarray(Xw)
    if Xw.shape[0] < Xw.shape[1]:  # (n_species, n_t) -> rows are species
        series = Xw[5]  # A3 (ATP-like)
    else:
        series = Xw[:, 5]
    ax[1].plot(tw, series, lw=1.0, color="C2")
    ax[1].set_xlabel("time"); ax[1].set_ylabel("A3 (ATP)")
    ax[1].set_title("(b) Wolf & Heinrich 2000 (single-cell core)"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "case_studies_oscillations.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def order_uncertainty(A, b):
    tot = 0.0
    for j in range(3):
        c = np.zeros(4); c[j] = 1.0
        tot += worst_case_range(A, b, c)
    return tot


def lift(A_alpha):
    return np.column_stack([A_alpha, np.zeros(len(A_alpha))])


def random_facets(rng, n, slack):
    tt = np.concatenate([ALPHA_TRUE, [LOGK_TRUE]])
    rows, rhs = [], []
    for _ in range(n):
        c = rng.normal(size=4); c /= np.linalg.norm(c)
        rows.append(c); rhs.append(c @ tt + slack)
    return np.array(rows), np.array(rhs)


def regime_means(lo, hi, n, eps, seeds):
    d_only, d_rop, d_rand = [], [], []
    slack = float(np.median(B_ROP[[0, 1, 5]] - A_ROP[[0, 1, 5]] @ ALPHA_TRUE))
    for s in seeds:
        rng = np.random.default_rng(s)
        X = rng.uniform(lo, hi, size=(n, 3))
        v = np.exp(LOGK_TRUE + np.log(X) @ ALPHA_TRUE + rng.uniform(-eps, eps, size=n))
        A_data, b_data = data_halfspaces(X, v, eps)
        Ar, br = random_facets(rng, len(A_ROP), slack)
        try:
            d_only.append(order_uncertainty(A_data, b_data))
            d_rop.append(order_uncertainty(np.vstack([A_data, lift(A_ROP)]), np.concatenate([b_data, B_ROP])))
            d_rand.append(order_uncertainty(np.vstack([A_data, Ar]), np.concatenate([b_data, br])))
        except Exception:
            pass
    return np.mean(d_only), np.mean(d_rop), np.mean(d_rand)


def fig_negative():
    seeds = range(200, 215)
    inf = regime_means(0.5, 3.0, 120, 0.06, seeds)
    poor = regime_means(0.9, 1.15, 12, 0.06, seeds)
    labels = ["data only", "data ∩ ROP\n(mechanism)", "data ∩ random\n(control)"]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - w / 2, inf, w, label="informative data", color="C0")
    ax.bar(x + w / 2, poor, w, label="data-poor", color="C3")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("total reaction-order uncertainty\n(sum of order ranges; lower = tighter)")
    ax.set_title("Binding-derived ROP gives no advantage over a count-matched random set")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "order_uq_negative.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)
    print(f"  informative: data-only={inf[0]:.3f}  ROP={inf[1]:.3f}  random={inf[2]:.3f}")
    print(f"  data-poor:   data-only={poor[0]:.3f}  ROP={poor[1]:.3f}  random={poor[2]:.3f}")


if __name__ == "__main__":
    fig_oscillations()
    fig_negative()
