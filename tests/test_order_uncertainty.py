"""Tests for reaction-order uncertainty-set construction (Route 2+ method).

Builds, test-first, the machinery for:
  - log-linear estimation of reaction orders from (concentration, rate) data,
  - a bounded-error data-consistent order set (half-space inequalities),
  - mechanism facets (sign / cooperativity caps) from binding structure,
  - the combined feasible polytope vs its axis-aligned bounding box,
  - tightness comparison (the empirical question of Route 2+).
"""

import numpy as np

from bmac_engine.order_uncertainty import (
    bounding_box,
    data_halfspaces,
    estimate_orders_loglinear,
    polytope_contains,
    worst_case_range,
)


def test_loglinear_recovers_true_orders_clean_data():
    rng = np.random.default_rng(0)
    true_alpha = np.array([2.0, 1.0])  # power-law orders for 2 species
    logk = np.log(1.5)
    X = rng.uniform(0.5, 3.0, size=(200, 2))  # concentration samples
    v = np.exp(logk) * X[:, 0] ** true_alpha[0] * X[:, 1] ** true_alpha[1]

    alpha_hat, logk_hat = estimate_orders_loglinear(X, v)

    assert np.allclose(alpha_hat, true_alpha, atol=1e-6)
    assert np.isclose(logk_hat, logk, atol=1e-6)


def test_data_halfspaces_contain_true_orders_under_bounded_noise():
    rng = np.random.default_rng(1)
    true_alpha = np.array([2.0, 1.0])
    logk = np.log(1.5)
    X = rng.uniform(0.5, 3.0, size=(100, 2))
    logv_clean = logk + np.log(X) @ true_alpha
    eps = 0.05
    noise = rng.uniform(-eps, eps, size=len(logv_clean))  # |noise| <= eps
    v = np.exp(logv_clean + noise)

    A, b = data_halfspaces(X, v, epsilon=eps)

    # theta = [alpha..., logk]; the true parameters must be data-feasible
    theta = np.concatenate([true_alpha, [logk]])
    assert np.all(A @ theta <= b + 1e-9)
    # a clearly-wrong order vector must be excluded by at least one half-space
    bad = np.concatenate([true_alpha + np.array([1.0, 1.0]), [logk]])
    assert np.any(A @ bad > b + 1e-9)


def test_bounding_box_is_looser_than_coupled_polytope():
    # Polytope with a NON-axis-aligned coupling facet a0 + a1 <= 2,
    # on top of the box 0 <= a_i <= 2.
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]], dtype=float)
    b = np.array([2, 2, 0, 0, 2], dtype=float)

    lo, hi = bounding_box(A, b)
    assert np.allclose(lo, [0, 0], atol=1e-6)
    assert np.allclose(hi, [2, 2], atol=1e-6)  # the box cannot see the coupling facet

    # The corner (2, 2) lies in the box but NOT in the polytope.
    assert polytope_contains(A, b, np.array([1.0, 1.0]))
    assert not polytope_contains(A, b, np.array([2.0, 2.0]))


def test_worst_case_range_smaller_on_coupled_polytope_than_box():
    # Coupling facet a0 + a1 <= 2 constrains the functional c = [1, 1].
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]], dtype=float)
    b = np.array([2, 2, 0, 0, 2], dtype=float)
    c = np.array([1.0, 1.0])  # e.g. predicted log-flux direction log x*

    lo, hi = bounding_box(A, b)
    A_box = np.vstack([np.eye(2), -np.eye(2)])
    b_box = np.concatenate([hi, -lo])

    rng_poly = worst_case_range(A, b, c)
    rng_box = worst_case_range(A_box, b_box, c)

    assert np.isclose(rng_poly, 2.0, atol=1e-6)  # max-min of a0+a1 over polytope
    assert np.isclose(rng_box, 4.0, atol=1e-6)   # over the looser box
    assert rng_poly < rng_box  # mechanism coupling tightens the worst-case spread
