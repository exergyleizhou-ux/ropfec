"""
Policy benchmarks: ROP-constrained FEC vs FBA / MM baselines on the toy DT.

Used by tests and correspondence_verification — failures here block run_all.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import linprog

from .fec_solver import TOY_N, solve_ROP_constrained_OCP
from .robust_extension import solve_robust_scenario_alpha
from .rop_polyhedron import ROPPolyhedron


def dt_final_cost(
    dt: Any,
    alpha: Any,
    ideal: Optional[List[float]] = None,
    L: Optional[Any] = None,
    x0: Optional[List[float]] = None,
) -> float:
    ideal_arr = np.asarray(ideal or [0.6, 0.85, 1.7], dtype=float)
    x0_arr = x0 or [1.0, 1.0, 1.0]
    if hasattr(dt, "simulate_with_sensitivity"):
        _, traj, _ = dt.simulate_with_sensitivity(x0_arr, alpha, T=2.0, L=L)
    else:
        _, traj = dt.simulate_trajectory(x0_arr, alpha, T=2.0)
    return float(np.linalg.norm(traj[-1] - ideal_arr))


def fba_policy(hat_x: List[float], rop: ROPPolyhedron, n_rxn: int = 4) -> np.ndarray:
    c = -np.ones(n_rxn)
    bounds = [(0.01, 10.0)] * n_rxn
    A_eq = np.asarray(TOY_N, dtype=float)
    b_eq = np.zeros(3)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    v_opt = res.x if res.success else np.ones(n_rxn) * 0.5
    x = np.maximum(np.asarray(hat_x, dtype=float), 1e-3)
    v_use = np.asarray(v_opt, dtype=float)[:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_fba = np.log(np.maximum(v_use, 1e-8)) / np.log(x)
    alpha_fba = np.nan_to_num(alpha_fba, nan=0.6, posinf=2.0, neginf=0.1)
    alpha_fba = np.clip(alpha_fba[:3], 0.1, 3.0)
    return rop.project(alpha_fba[:3])


def mm_policy(hat_x: List[float], rop: ROPPolyhedron) -> np.ndarray:
    x = np.asarray(hat_x, dtype=float)
    alpha_mm = 0.6 + 0.8 / (1.0 + 1.0 / np.maximum(x, 0.1))
    alpha_mm = np.clip(alpha_mm[:3], 0.2, 2.5)
    return rop.project(alpha_mm[:3])


def run_fba_mm_benchmark(
    rop: ROPPolyhedron,
    dt: Any,
    n_runs: int = 50,
    cost_fn: Optional[Callable[[Any], float]] = None,
    L: Optional[Any] = None,
) -> Dict[str, float]:
    if cost_fn is None:
        cost_fn = lambda a: dt_final_cost(dt, a, L=L)

    fba_costs: List[float] = []
    mm_costs: List[float] = []
    rop_costs: List[float] = []
    scen_costs: List[float] = []

    hat_x = [1.0, 1.0, 1.0]
    for _ in range(n_runs):
        a_rop = solve_ROP_constrained_OCP(hat_x, [0.8, 0.8, 0.8], rop, 3)
        a_fba = fba_policy(hat_x, rop)
        a_mm = mm_policy(hat_x, rop)
        a_scen = solve_robust_scenario_alpha(rop, dt, n_samples=8).get("alpha", a_rop)
        for label, alpha in (
            ("rop", a_rop),
            ("fba", a_fba),
            ("mm", a_mm),
            ("scenario", a_scen),
        ):
            if not rop.is_feasible(alpha):
                raise AssertionError(f"{label} policy produced infeasible alpha")
        fba_costs.append(cost_fn(a_fba))
        mm_costs.append(cost_fn(a_mm))
        rop_costs.append(cost_fn(a_rop))
        scen_costs.append(cost_fn(a_scen))

    fba_m, fba_s = float(np.mean(fba_costs)), float(np.std(fba_costs))
    mm_m, mm_s = float(np.mean(mm_costs)), float(np.std(mm_costs))
    rop_m, rop_s = float(np.mean(rop_costs)), float(np.std(rop_costs))
    scen_m, scen_s = float(np.mean(scen_costs)), float(np.std(scen_costs))
    red_fba = fba_m / max(rop_m, 1e-9)
    red_mm = mm_m / max(rop_m, 1e-9)
    red_scen_pct = (fba_m - scen_m) / max(fba_m, 1e-9) * 100.0

    return {
        "fba_mean": fba_m,
        "fba_std": fba_s,
        "mm_mean": mm_m,
        "mm_std": mm_s,
        "rop_mean": rop_m,
        "rop_std": rop_s,
        "scenario_mean": scen_m,
        "scenario_std": scen_s,
        "red_fba_vs_rop": red_fba,
        "red_mm_vs_rop": red_mm,
        "scenario_better_than_fba_pct": red_scen_pct,
        "n_runs": float(n_runs),
    }
