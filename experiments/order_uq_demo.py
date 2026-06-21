"""Gate experiment (Route 2+): does the binding-derived ROP tighten a
data-only reaction-order set, and does a matched random-facet control not?

Thesis under test: intersecting bounded-error reaction-order constraints (from
noisy rate data) with the binding-derived ROP (a generally NON-axis-aligned
polytope) yields a strictly tighter feasible set than the data-only axis-aligned
box, reducing the worst-case spread of the predicted log-flux -- hence a
uniformly less-conservative robust-control certificate. The advantage must come
from the MECHANISM geometry, not facet count: a count-matched RANDOM-facet set
that still covers the truth should NOT tighten beyond chance.

All set operations use the unit-tested helpers in bmac_engine.order_uncertainty.
Run:  PYTHONPATH=. python3 experiments/order_uq_demo.py
"""

import numpy as np

from bmac_engine.order_uncertainty import (
    bounding_box,
    data_halfspaces,
    polytope_contains,
    worst_case_range,
)

# Engine "glycolysis_upper" toy ROP (binding-derived): 3 orders [a0,a1,a2].
# The last row a0 + 0.5*a2 <= 1.8 is a NON-axis-aligned binding facet.
A_ROP = np.array(
    [[1, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0.5]],
    dtype=float,
)
B_ROP = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 1.8])
ALPHA_TRUE = np.array([0.6, 0.85, 1.7])  # binding-limited true orders (in ROP)
LOGK_TRUE = np.log(1.3)
X_STAR = np.array([1.6, 0.7, 1.4])  # operating point (log != 0)


def lift(A_alpha):
    """Lift an alpha-space facet matrix to theta=[alpha, logk] space (0 on logk)."""
    return np.column_stack([A_alpha, np.zeros(len(A_alpha))])


def make_data(rng, n=120, eps=0.06):
    X = rng.uniform(0.5, 3.0, size=(n, 3))
    logv = LOGK_TRUE + np.log(X) @ ALPHA_TRUE
    v = np.exp(logv + rng.uniform(-eps, eps, size=n))
    return data_halfspaces(X, v, epsilon=eps)


def random_facets(rng, n_facets, slack):
    """Count-matched random linear facets that still contain the true theta."""
    theta_true = np.concatenate([ALPHA_TRUE, [LOGK_TRUE]])
    rows, rhs = [], []
    for _ in range(n_facets):
        c = rng.normal(size=4)
        c /= np.linalg.norm(c)
        rows.append(c)
        rhs.append(c @ theta_true + slack)  # true point interior by `slack`
    return np.array(rows), np.array(rhs)


def main():
    rng = np.random.default_rng(7)
    c_flux = np.concatenate([np.log(X_STAR), [1.0]])  # predicted log-flux functional
    theta_true = np.concatenate([ALPHA_TRUE, [LOGK_TRUE]])

    spreads_box, spreads_rop, spreads_rand, cover_rop, cover_rand = [], [], [], [], []
    for trial in range(20):
        rng_t = np.random.default_rng(100 + trial)
        A_data, b_data = make_data(rng_t)

        # data-only axis-aligned box
        lo, hi = bounding_box(A_data, b_data)
        A_box = np.vstack([np.eye(4), -np.eye(4)])
        b_box = np.concatenate([hi, -lo])

        # data ∩ ROP (mechanism)
        A_rop = np.vstack([A_data, lift(A_ROP)])
        b_rop = np.concatenate([b_data, B_ROP])

        # negative control: data ∩ count-matched random facets (cover truth)
        slack = float(np.median(B_ROP[[0, 1, 5]] - A_ROP[[0, 1, 5]] @ ALPHA_TRUE))
        A_r, b_r = random_facets(rng_t, n_facets=len(A_ROP), slack=slack)
        A_rand = np.vstack([A_data, A_r])
        b_rand = np.concatenate([b_data, b_r])

        spreads_box.append(worst_case_range(A_box, b_box, c_flux))
        spreads_rop.append(worst_case_range(A_rop, b_rop, c_flux))
        spreads_rand.append(worst_case_range(A_rand, b_rand, c_flux))
        cover_rop.append(polytope_contains(A_rop, b_rop, theta_true))
        cover_rand.append(polytope_contains(A_rand, b_rand, theta_true))

    box = np.array(spreads_box); rop = np.array(spreads_rop); rand = np.array(spreads_rand)
    print("=== Gate experiment: ROP-tightening of data-only order set ===")
    print(f"trials: {len(box)}   operating point x* = {X_STAR.tolist()}")
    print(f"worst-case log-flux spread  (data-only BOX):     {box.mean():.4f} ± {box.std():.4f}")
    print(f"worst-case log-flux spread  (data ∩ ROP):        {rop.mean():.4f} ± {rop.std():.4f}")
    print(f"worst-case log-flux spread  (data ∩ RANDOM):     {rand.mean():.4f} ± {rand.std():.4f}")
    print(f"ROP    / box  ratio (lower = tighter):           {(rop/box).mean():.3f}")
    print(f"RANDOM / box  ratio (negative control):          {(rand/box).mean():.3f}")
    print(f"coverage (true orders in set):  ROP {np.mean(cover_rop)*100:.0f}%   RANDOM {np.mean(cover_rand)*100:.0f}%")
    print()
    red = (1 - (rop / box).mean()) * 100
    red_rand = (1 - (rand / box).mean()) * 100
    print(f"ROP reduces worst-case flux-prediction spread by {red:.1f}% vs data-only box.")
    print(f"Random-facet control reduces it by only {red_rand:.1f}% (should be ~0 if effect is mechanism-driven).")


if __name__ == "__main__":
    main()
