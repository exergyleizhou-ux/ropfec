"""Reaction-order uncertainty sets for structure-aware robust FEC (Route 2+).

This module layers on top of the existing ropfec engine (ROPPolyhedron, the
CasADi/IPOPT FEC solver). It builds, from data and from binding mechanism, a
feasible set over the power-law reaction orders, and lets us compare a
mechanism-structured polytope against a naive axis-aligned box.

Power-law rate model (per reaction):  v = k * prod_i x_i^{alpha_i},
so  log v = log k + sum_i alpha_i * log x_i  (linear in the orders alpha).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def polytope_contains(A, b, x, tol=1e-9):
    """True if point ``x`` satisfies ``A @ x <= b`` (within ``tol``)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)
    return bool(np.all(A @ x <= b + tol))


def bounding_box(A, b):
    """Axis-aligned bounding box of the polytope ``{x : A @ x <= b}``.

    Each coordinate's min and max are found by linear programming. This is the
    naive uncertainty set a method ignorant of non-axis-aligned (coupling /
    thermodynamic) facets would use.

    Returns
    -------
    lo, hi : numpy.ndarray, shape (n_dims,)
        Per-coordinate lower and upper bounds.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[1]
    lo = np.empty(n)
    hi = np.empty(n)
    for j in range(n):
        c = np.zeros(n)
        c[j] = 1.0
        res_lo = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
        res_hi = linprog(-c, A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
        if not (res_lo.success and res_hi.success):
            raise ValueError("bounding_box: polytope is empty or unbounded")
        lo[j] = res_lo.x[j]
        hi[j] = -res_hi.fun
    return lo, hi


def worst_case_range(A, b, c):
    """Width of a linear functional ``c . x`` over the polytope ``A @ x <= b``.

    Equals ``max c.x - min c.x``. With ``c = log x*`` this is the worst-case
    spread of the predicted log-flux at operating point ``x*`` implied by the
    order-uncertainty set; a tighter set gives a smaller spread and hence a
    uniformly less-conservative robust-control certificate.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    n = A.shape[1]
    res_min = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
    res_max = linprog(-c, A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
    if not (res_min.success and res_max.success):
        raise ValueError("worst_case_range: polytope is empty or unbounded")
    return float((-res_max.fun) - res_min.fun)


def estimate_orders_loglinear(X, v):
    """Least-squares estimate of power-law orders from (concentration, rate) data.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_species)
        Strictly positive concentration samples.
    v : array_like, shape (n_samples,)
        Strictly positive reaction-rate samples.

    Returns
    -------
    alpha_hat : numpy.ndarray, shape (n_species,)
        Estimated reaction orders.
    logk_hat : float
        Estimated log rate constant.
    """
    X = np.asarray(X, dtype=float)
    v = np.asarray(v, dtype=float)
    logX = np.log(X)
    logv = np.log(v)
    # design matrix [1, logX] for [logk, alpha]
    A = np.column_stack([np.ones(len(logv)), logX])
    coef, *_ = np.linalg.lstsq(A, logv, rcond=None)
    logk_hat = float(coef[0])
    alpha_hat = coef[1:]
    return alpha_hat, logk_hat


def data_halfspaces(X, v, epsilon):
    """Bounded-error data-consistent set over theta = [alpha..., logk].

    Under the bounded-error model |log v_i - (alpha . log x_i + log k)| <= epsilon,
    the data-consistent parameters form a polytope A @ theta <= b. Each sample
    contributes two half-spaces (upper and lower residual bound).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_species)  -- positive concentrations.
    v : array_like, shape (n_samples,)            -- positive rates.
    epsilon : float                               -- residual bound (log scale).

    Returns
    -------
    A : numpy.ndarray, shape (2 * n_samples, n_species + 1)
    b : numpy.ndarray, shape (2 * n_samples,)
        such that data-consistent theta satisfy ``A @ theta <= b``; the last
        component of theta is log k.
    """
    X = np.asarray(X, dtype=float)
    v = np.asarray(v, dtype=float)
    logX = np.log(X)
    logv = np.log(v)
    n = len(logv)
    row = np.column_stack([logX, np.ones(n)])  # [log x_i, 1] coefficients on theta
    A = np.vstack([row, -row])
    b = np.concatenate([logv + epsilon, -logv + epsilon])
    return A, b
