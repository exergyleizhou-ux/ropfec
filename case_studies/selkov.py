"""Sel'kov (1968) minimal glycolytic oscillator (2-variable, dimensionless).

This module implements the canonical two-variable dimensionless Sel'kov model of
glycolytic oscillations. In the dimensionless form used by Strogatz (*Nonlinear
Dynamics and Chaos*, Example/Exercise on the glycolytic oscillator) the system is::

    dx/dt = -x + a*y + x**2 * y
    dy/dt =  b - a*y - x**2 * y

where

    x  ~  dimensionless concentration of ADP (the product / activator)
    y  ~  dimensionless concentration of F6P (fructose-6-phosphate, the substrate)
    a  ~  rate constant for the (un-catalyzed) decay of F6P to ADP
    b  ~  constant supply rate of substrate F6P

The product x autocatalytically activates its own production via the x**2 * y term
(positive feedback of ADP on phosphofructokinase). For a suitable window of (a, b)
the unique interior fixed point loses stability through a supercritical Hopf
bifurcation and the system settles onto a unique, globally attracting stable limit
cycle -- i.e. sustained glycolytic oscillations.

Fixed point and stability (derived analytically):
    Adding the two equations gives d(x+y)/dt = b - x, so at steady state x* = b and
    y* = b / (a + b**2). The Jacobian at the fixed point is

        J = [[-1 + 2*x*y,   a + x**2],
             [   -2*x*y,   -(a + x**2)]]

    A stable limit cycle exists when trace(J) > 0 and det(J) > 0 (unstable spiral
    enclosed by a limit cycle). det(J) = a + x**2 > 0 always (for a, x not both 0),
    so the Hopf condition is trace(J) > 0.

Default parameters (a=0.06, b=0.6) lie INSIDE the limit-cycle window: there
trace(J) = +0.294 > 0 and det(J) = +0.42 > 0, giving sustained oscillations with
peak-to-peak amplitude in x of ~1.8. A contrasting damped (stable fixed point) set
is PARAMS_FIXED_POINT = {'a': 0.1, 'b': 1.0}, where trace(J) = -0.28 < 0 and
trajectories spiral into the fixed point.

Source / citation:
    E. E. Sel'kov, "Self-Oscillations in Glycolysis. 1. A Simple Kinetic Model",
    European Journal of Biochemistry 4 (1968) 79-86.
    Dimensionless form and (a, b) analysis as presented in
    S. H. Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Section 7.3
    (glycolytic oscillator). See also Brechmann & Rendall,
    "Dynamics of the Sel'kov oscillator" (Math. Biosci., 2018) for the Hopf
    analysis confirming the unique stable limit cycle.

Dependencies: numpy and scipy only.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "PARAMS",
    "PARAMS_FIXED_POINT",
    "X0",
    "dxdt",
    "fixed_point",
    "simulate",
]

# Default parameters: INSIDE the Hopf / limit-cycle window -> sustained oscillation.
PARAMS: dict[str, float] = {"a": 0.06, "b": 0.6}

# Contrast set: stable fixed point (damped to equilibrium, no sustained oscillation).
PARAMS_FIXED_POINT: dict[str, float] = {"a": 0.1, "b": 1.0}

# Default initial state (x0, y0) = (ADP, F6P). Chosen away from the fixed point so
# the transient onto the limit cycle is visible.
X0: np.ndarray = np.array([1.0, 1.0], dtype=float)


def dxdt(t: float, x: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Right-hand side of the dimensionless Sel'kov glycolytic oscillator.

    Parameters
    ----------
    t : float
        Time (the system is autonomous; ``t`` is unused but kept for the
        ``solve_ivp`` signature).
    x : array_like, shape (2,)
        State vector ``[x, y]`` = ``[ADP, F6P]`` (dimensionless).
    params : dict
        Must contain ``"a"`` and ``"b"`` (positive floats).

    Returns
    -------
    numpy.ndarray, shape (2,)
        Time derivative ``[dx/dt, dy/dt]``.
    """
    a = params["a"]
    b = params["b"]
    xv, yv = x[0], x[1]
    xy_auto = xv * xv * yv  # autocatalytic ADP*ADP*F6P term
    dx = -xv + a * yv + xy_auto
    dy = b - a * yv - xy_auto
    return np.array([dx, dy], dtype=float)


def fixed_point(params: dict[str, float]) -> np.ndarray:
    """Return the interior fixed point (x*, y*) = (b, b/(a + b**2))."""
    a = params["a"]
    b = params["b"]
    return np.array([b, b / (a + b * b)], dtype=float)


def simulate(
    params: dict[str, float] | None = None,
    x0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 400.0),
    n_points: int = 4000,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 0.05,
):
    """Integrate the Sel'kov model and return ``(t, X)``.

    Parameters
    ----------
    params : dict, optional
        Parameter dict with keys ``"a"``, ``"b"``. Defaults to :data:`PARAMS`
        (inside the limit-cycle window).
    x0 : array_like, optional
        Initial state ``[x, y]``. Defaults to :data:`X0`.
    t_span : (float, float)
        Integration interval ``(t0, t_end)``.
    n_points : int
        Number of evenly spaced output samples over ``t_span``.
    rtol, atol, max_step :
        Passed to :func:`scipy.integrate.solve_ivp` (stiff-safe defaults; the
        small ``max_step`` keeps the limit-cycle amplitude faithful).

    Returns
    -------
    t : numpy.ndarray, shape (n_points,)
        Time samples.
    X : numpy.ndarray, shape (n_points, 2)
        State trajectory; column 0 is ``x`` (ADP), column 1 is ``y`` (F6P).
    """
    if params is None:
        params = PARAMS
    if x0 is None:
        x0 = X0
    x0 = np.asarray(x0, dtype=float)

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        dxdt,
        t_span,
        x0,
        args=(params,),
        method="LSODA",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y.T


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    t, X = simulate()
    transient = t >= 0.5 * t[-1]
    x = X[transient, 0]
    print("default (a=0.06, b=0.6) limit cycle: x peak-to-peak =",
          round(float(x.max() - x.min()), 4))
    t2, X2 = simulate(params=PARAMS_FIXED_POINT)
    x2 = X2[t2 >= 0.5 * t2[-1], 0]
    print("contrast (a=0.1, b=1.0) fixed point: x peak-to-peak =",
          round(float(x2.max() - x2.min()), 6))
