"""Wolf & Heinrich (2000) yeast glycolytic oscillator -- single-cell core.

This module implements the SINGLE-CELL core of the model published in:

    Wolf J, Heinrich R. "Effect of cellular interaction on glycolytic
    oscillations in yeast: a theoretical investigation."
    Biochem. J. 2000 Jan; 345 Pt 2: 321-334. PubMed: 10702114.

The equations and parameter values are taken verbatim from the BioModels
encoding BIOMD0000000691 (SBML L2V4), which is the two-cell coupled model.
Here we implement ONE cell only: we keep Cell 1 together with its own external
acetaldehyde pool (S4_ex) and drop the entire second cell (and therefore the
second cell's contribution to S4_ex). No rate law or parameter value is
altered. With the published parameters this single-cell core exhibits a stable
limit cycle (sustained glycolytic oscillations).

Why S4_ex is retained
---------------------
In the SBML the only outflow of the pyruvate/acetaldehyde pool S4 is the
trans-membrane exchange flux J = kappa*(S4 - S4_ex); S4_ex in turn has its own
sink k*S4_ex. Removing this exchange entirely (kappa = 0) would leave S4 with no
degradation and the inflow J0 unbalanced, so the single cell would drift. The
faithful single-cell reduction therefore keeps Cell 1 + S4_ex with the original
exchange flux, simply dropping the second cell. Setting kappa = 0 recovers the
isolated reaction core (provided for completeness; that regime is non-oscillatory).

State variables (concentrations, mM):
    S1    glucose (input pool)
    S2    triose-phosphate / upper-glycolysis pool
    S3    1,3-bisphosphoglycerate pool (BPG)
    S4    pyruvate / acetaldehyde pool (intracellular)
    N2    NADH
    A3    ATP
    S4_ex external acetaldehyde pool

Conserved moieties (held fixed, as in the SBML assignment rules):
    N1 = N - N2     (NAD+),  total N = 1
    A2 = A - A3     (ADP),   total A = 4

Reactions and rate laws (exactly as in the SBML kinetic laws):
    v1 (HK/PFK, ATP-inhibited):  S1 + 2 A3 -> 2 S2
        v1 = k1 * S1 * A3 / (1 + (A3 / K_I) ** q)          (q = 4)
    v2 (GAPDH):                  S2 + N1 -> S3 + N2
        v2 = k2 * S2 * N1
    v3 (PGK/PK):                 S3 + A2 -> S4 + 2 A3
        v3 = k3 * S3 * A2
    v4 (ADH):                    S4 + N2 ->
        v4 = k4 * S4 * N2
    v5 (ATP consumption):        A3 ->
        v5 = k5 * A3
    v6 (glycerol branch):        S2 + N2 ->
        v6 = k6 * S2 * N2
    J0 (glucose inflow):         -> S1
        J0 = const
    J  (membrane exchange):      S4 <-> S4_ex
        J  = kappa * (S4 - S4_ex)
    vsink (S4_ex degradation):   S4_ex ->
        vsink = k * S4_ex

Net ODEs (single cell):
    dS1/dt    = J0 - v1
    dS2/dt    = 2*v1 - v2 - v6
    dS3/dt    = v2 - v3
    dS4/dt    = v3 - v4 - J
    dN2/dt    = v2 - v4 - v6
    dA3/dt    = -2*v1 + 2*v3 - v5
    dS4_ex/dt = phi * J - k * S4_ex
        (phi = ratio of intracellular to extracellular volume; with the second
         cell dropped only this one cell feeds S4_ex, hence a single phi*J term.)

Only numpy and scipy.integrate.solve_ivp are used.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# State variable order used throughout this module.
STATE_NAMES = ("S1", "S2", "S3", "S4", "N2", "A3", "S4_ex")


def default_params() -> dict:
    """Return the published default parameter set (BIOMD0000000691).

    Values are taken verbatim from the SBML <listOfParameters>:
        k1=100, K_I=0.52, q=4, k2=6, k3=16, k4=100, k5=1.28, k6=12,
        k=1.5 (S4_ex sink), J0=3, kappa=13, phi=0.1, N=1, A=4.

    With these values the single-cell core oscillates (stable limit cycle).
    """
    return {
        "J0": 3.0,      # glucose inflow rate
        "k1": 100.0,    # PFK-type rate constant
        "K_I": 0.52,    # ATP inhibition constant of v1
        "q": 4.0,       # Hill cooperativity of ATP inhibition
        "k2": 6.0,      # GAPDH rate constant
        "k3": 16.0,     # PGK/PK rate constant
        "k4": 100.0,    # ADH rate constant
        "k5": 1.28,     # ATP consumption rate constant
        "k6": 12.0,     # glycerol branch rate constant
        "k": 1.5,       # S4_ex degradation rate constant
        "kappa": 13.0,  # trans-membrane exchange rate of S4
        "phi": 0.1,     # intra/extracellular volume ratio
        "N": 1.0,       # total NAD(H) moiety  (N1 = N - N2)
        "A": 4.0,       # total adenine moiety (A2 = A - A3)
    }


def default_x0() -> np.ndarray:
    """Return the published single-cell initial concentrations (Cell 1).

    Taken from the SBML initialConcentration of the Cell_1 species and S4_ex:
    S1=5.8, S2=0.9, S3=0.2, S4=0.2, N2=0.1, A3=3.2, S4_ex=0.1.
    """
    return np.array([5.8, 0.9, 0.2, 0.2, 0.1, 3.2, 0.1], dtype=float)


def dxdt(t: float, x: np.ndarray, params: dict) -> np.ndarray:
    """Right-hand side of the single-cell Wolf-Heinrich glycolytic oscillator.

    Parameters
    ----------
    t : float
        Time (unused; the system is autonomous). Present for solve_ivp.
    x : array_like, shape (7,)
        State vector [S1, S2, S3, S4, N2, A3, S4_ex].
    params : dict
        Parameter dictionary (see :func:`default_params`).

    Returns
    -------
    numpy.ndarray, shape (7,)
        Time derivatives, in the order of :data:`STATE_NAMES`.
    """
    S1, S2, S3, S4, N2, A3, S4_ex = x

    J0 = params["J0"]
    k1 = params["k1"]
    K_I = params["K_I"]
    q = params["q"]
    k2 = params["k2"]
    k3 = params["k3"]
    k4 = params["k4"]
    k5 = params["k5"]
    k6 = params["k6"]
    k = params["k"]
    kappa = params["kappa"]
    phi = params["phi"]
    N = params["N"]
    A = params["A"]

    # Conserved-moiety partners (SBML assignment rules).
    N1 = N - N2     # NAD+
    A2 = A - A3     # ADP

    # Reaction rates (exact SBML kinetic laws).
    v1 = k1 * S1 * A3 / (1.0 + (A3 / K_I) ** q)
    v2 = k2 * S2 * N1
    v3 = k3 * S3 * A2
    v4 = k4 * S4 * N2
    v5 = k5 * A3
    v6 = k6 * S2 * N2
    J = kappa * (S4 - S4_ex)     # trans-membrane exchange of S4

    dS1 = J0 - v1
    dS2 = 2.0 * v1 - v2 - v6
    dS3 = v2 - v3
    dS4 = v3 - v4 - J
    dN2 = v2 - v4 - v6
    dA3 = -2.0 * v1 + 2.0 * v3 - v5
    dS4_ex = phi * J - k * S4_ex

    return np.array([dS1, dS2, dS3, dS4, dN2, dA3, dS4_ex], dtype=float)


def simulate(
    t_end: float = 100.0,
    n_points: int = 5000,
    x0: np.ndarray | None = None,
    params: dict | None = None,
    t_start: float = 0.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    method: str = "LSODA",
):
    """Integrate the single-cell model and return ``(t, X)``.

    Parameters
    ----------
    t_end : float
        Final integration time (arbitrary time units of the model).
    n_points : int
        Number of evenly spaced output samples on [t_start, t_end].
    x0 : array_like, optional
        Initial state; defaults to :func:`default_x0`.
    params : dict, optional
        Parameters; defaults to :func:`default_params`.
    t_start : float
        Start time.
    rtol, atol : float
        Solver tolerances.
    method : str
        scipy.integrate.solve_ivp method.

    Returns
    -------
    t : numpy.ndarray, shape (n_points,)
    X : numpy.ndarray, shape (7, n_points)
        Rows ordered as :data:`STATE_NAMES`.
    """
    if x0 is None:
        x0 = default_x0()
    if params is None:
        params = default_params()

    t_eval = np.linspace(t_start, t_end, n_points)
    sol = solve_ivp(
        dxdt,
        (t_start, t_end),
        np.asarray(x0, dtype=float),
        t_eval=t_eval,
        args=(params,),
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.t, sol.y


if __name__ == "__main__":
    t, X = simulate(t_end=60.0, n_points=3000)
    half = len(t) // 2
    for name in ("S2", "N2", "A3"):
        v = X[STATE_NAMES.index(name)][half:]
        print(f"{name} peak-to-peak (2nd half): {v.max() - v.min():.4f}")
