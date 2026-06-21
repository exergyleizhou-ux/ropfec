"""
fec_solver.py

Per BOS-BMAC Phase 0 Spec v1.0 §2.2 and §5:
- ROP-constrained dynamic flux modeling (FEC).
- The key implementation detail (added in v1.0 per review): how to encode the ROP
  H-representation (A, b) coming from rop_polyhedron.py as constraints inside the
  optimizer.

This file currently contains:
1. A pure-Python "manual" solver for the glycolysis toy that explicitly
   applies the inequalities exactly as the CasADi comment in the spec says:
       for i in rows:   subject_to( A[i,:] @ alpha <= b[i] )
2. The public function signature expected by the pseudocode and by bos_integration.
3. A placeholder for the real CasADi + IPOPT path (when casadi is importable).

The toy version is sufficient to make the executable validation in
tests/test_fec_toy_glycolysis.py (and any higher-level integration test) pass
while demonstrating the exact encoding.
"""
from __future__ import annotations
from typing import Any, List, Tuple, Optional
import numpy as np
try:
    from scipy import optimize as sco
except Exception:
    sco = None
from .rop_polyhedron import project_onto_ROP, is_in_ROP, ROPPolyhedron

# Try to import casadi for the real path (optional for Phase 0 toy)
try:
    import casadi as ca  # type: ignore
    HAS_CASADI = True
except Exception:
    ca = None  # type: ignore
    HAS_CASADI = False

# Toy constants from Phase 0 Spec v1.0 §6 (for realistic dynamic simulation in FEC toy)
TOY_N = np.array([
    [-1,  0,  0,  1],
    [ 1, -1, -2,  0],
    [ 0,  1,  1, -1],
], dtype=float)

# Module-level cache for warmstart (Musk/Huang: reuse previous solutions for faster/more reliable solves)
_LAST_SUCCESSFUL_TRAJ = None

# Numerical guards for long multicell rollouts (prevents x**alpha overflow in scipy/CasADi toy paths)
_X_SIM_MAX = 50.0
_ALPHA_SIM_MAX = 5.0


def _clip_state(x: Array, xmax: float = _X_SIM_MAX) -> Array:
    x = np.asarray(x, dtype=float)
    return np.clip(x, 1e-12, xmax)


def _clip_alpha(alpha: Array, amax: float = _ALPHA_SIM_MAX) -> Array:
    alpha = np.asarray(alpha, dtype=float).ravel()
    return np.clip(alpha, 0.0, amax)


def _safe_pow(base: Array, exp: float) -> float:
    """base**exp with clipped base/exp to avoid overflow in long simulations."""
    b = float(np.clip(base, 1e-12, _X_SIM_MAX))
    e = float(np.clip(exp, 0.0, _ALPHA_SIM_MAX))
    if e * np.log(b) > 700.0:
        return float(np.exp(700.0))
    return float(b ** e)


def compute_v_toy(x: Array, alpha: Array, k: Optional[Array] = None) -> Array:
    """Power-law v for the glycolysis toy (extended for Phase3 'more real' net with 5th reaction)."""
    if k is None:
        k = np.ones(5)
    x = _clip_state(x)
    alpha = _clip_alpha(alpha)
    n_a = alpha.size
    if n_a == 3:
        alpha = np.concatenate([alpha, np.ones(2)])
    elif n_a == 4:
        alpha = np.concatenate([alpha, [1.0]])  # pad for extended
    # v for extended (use 4 for core dynamics compat, extra alpha affects ROP only for Phase3 demo)
    v = np.zeros(4)
    v[0] = k[0] * _safe_pow(x[0], alpha[0])          # G -> F
    v[1] = k[1] * _safe_pow(x[1], alpha[1])          # F -> P
    v[2] = k[2] * _safe_pow(x[1], alpha[2])          # 2F -> P (dimer)
    if n_a >= 4:
        v[3] = k[3] * _safe_pow(x[2], alpha[3])      # use 4th alpha for P recycle in extended
    else:
        v[3] = k[3] * _safe_pow(x[2], 1.0)
    return np.clip(v, 0.0, _X_SIM_MAX)

def _solve_toy_with_explicit_hrep_constraints(
    naive_alpha: List[float],
    P: ROPPolyhedron,
    f_star: Optional[List[float]] = None,
    horizon: int = 5,
    x0: Optional[List[float]] = None,
) -> List[float]:
    """
    Explicitly encodes the ROP H-rep as hard constraints, exactly like the
    comment we added to the v1.0 spec:

        // Encode ROP H-representation (A,b) from rop_polyhedron.py directly as CasADi inequalities:
        //   for i in rows: opti.subject_to( A[i,:] @ alpha <= b[i] )   (or mtimes(A, alpha) <= b)

    For the 3-variable glycolysis toy we do a real-ish bounded optimization
    with the H-rep constraints using scipy, and the cost now uses full
    multi-step forward simulation with TOY_N and compute_v_toy to reflect
    dynamic flux tracking (ell on f vs f*) over the horizon. This makes the
    toy solver much closer to a real dynamic OCP.
    """
    if hasattr(P, "A"):
        A = P.A
        b = P.b
    else:
        A, b = P
    alpha = np.asarray(naive_alpha, dtype=float).ravel()
    n = len(alpha)
    # Phase3: support extended dim (4+) for more real nets; no force trim
    target = np.array([0.6, 0.85, 1.7] if f_star is None else f_star[:n], dtype=float)
    if len(target) < n:
        target = np.concatenate([target, np.ones(n - len(target))])
    if x0 is None:
        x0 = [1.0] * 3 + [0.5] * (n - 3) if n > 3 else [1.0, 1.0, 1.0]
    x0 = np.asarray(x0, dtype=float)[:3]  # sim x always 3 for core dynamics

    # Try a real bounded optimization with the H-rep constraints (scipy)
    # Cost uses full multi-step simulation with TOY_N for realistic dynamics.
    try:
        def cost(x):
            # Simulate horizon steps with constant alpha (Phase 0 simplification for toy)
            x_sim = _clip_state(x0)
            alpha_vec = _clip_alpha(x)
            total_cost = 0.5 * np.sum((alpha_vec[:3] - target[:3])**2)  # alpha reg (core)
            k = np.ones(4)
            dt = 0.1
            for _ in range(max(1, horizon)):
                v = compute_v_toy(x_sim, alpha_vec, k)
                f = v  # flux
                # tracking cost (part of ell)
                total_cost += np.sum((f[:3] - target[:3])**2) * dt
                if not np.isfinite(total_cost):
                    return 1e30
                # euler: dx = N @ v * dt
                dx = TOY_N @ v * dt
                x_sim = _clip_state(x_sim + dx)
            return float(total_cost)

        bounds = [(0, None)] * n

        # Build inequality constraints exactly as "subject_to(A[i] @ alpha <= b[i])"
        constraints = []
        for i in range(len(b)):
            def make_con(i):
                def con(x):
                    return b[i] - A[i] @ x
                return con
            constraints.append({'type': 'ineq', 'fun': make_con(i)})

        res = sco.minimize(cost, alpha, bounds=bounds, constraints=constraints,
                           method='SLSQP', options={'ftol': 1e-10, 'maxiter': 100})
        if res.success:
            alpha = res.x
    except Exception:
        # Fallback to the explicit successive enforcement
        alpha = alpha.tolist()
        for _ in range(50):
            for i in range(len(b)):
                val = sum(A[i][j] * alpha[j] for j in range(n))
                if val > b[i]:
                    denom = sum(a * a for a in A[i])
                    if denom > 1e-14:
                        step = (val - b[i]) / denom
                        for j in range(n):
                            alpha[j] -= step * A[i][j]
            alpha = [max(0.0, v) for v in alpha]
        alpha = np.array(alpha)

    # Final guarantee
    alpha = project_onto_ROP(P, alpha) if hasattr(P, "project") else project_onto_ROP(alpha, P)
    return alpha.tolist() if isinstance(alpha, np.ndarray) else alpha

def solve_ROP_constrained_OCP(
    hat_x: List[float],
    f_star: List[float],
    P,
    horizon: int,
    bos_api: Any = None,
    return_traj: bool = False,
    robust_samples: Optional[List["ROPPolyhedron"]] = None,
) -> Any:
    """
    The function referenced in the FEC pseudocode in the v1.0 spec.

    Real implementation (future):
        - Build CasADi Opti
        - Add decision var alpha(t) for t in horizon
        - Discretize xdot = N @ v(x, alpha) with v_j = k_j * prod x_i**alpha_ji
    Phase2 high-quality: robust_samples (from DT MC) enables conservative robust MPC by adding H-rep constraints from all samples (alpha feasible in every realization).
        - For every collocation node and every row i of the H-rep:
              opti.subject_to( A[i,:] @ alpha_t <= b[i] )
        - Objective = integral( ell(x, f, alpha) ) + reg on alpha deviation
        - Solve with IPOPT (or do-mpc wrapper)

    Current (toy):
        - Uses the explicit row-by-row H-rep enforcement above so that the
          "how to encode ROP as CasADi inequalities" is not only a comment in
          the spec but also executable code you can read and extend.

    Deepened for correspondence:
    - Supports return_traj=True -> returns (alpha_mean_or_safe, alpha_traj)
      where alpha_traj is list of horizon alphas (time-varying sequence).
      This is directly consumable by bos_platform.DigitalTwin.simulate_trajectory(..., alpha_seq=alpha_traj)
      for exact optimizer-chosen trajectory simulation (CasADi or constant-repeated for scipy path).
    - When bos_api given, the traj is also published under "fec_alpha_traj" (or "fec_alpha_traj_casadi").
    """
    # Naive / unconstrained guess (in real code this would come from a previous
    # MPC step or a simple flux-balance solve).
    # We deliberately start from a point that violates the toy ROP so the
    # constraint encoding has something to do.
    n = getattr(P, 'dim', 3) if hasattr(P, 'dim') else (P.A.shape[1] if hasattr(P, 'A') else 3)
    naive = [0.5, 0.8, 2.8] + [1.0] * (n - 3) if n > 3 else [0.5, 0.8, 2.8]

    if HAS_CASADI:
        # Use the full CasADi skeleton (time-varying alpha, explicit subject_to for H-rep,
        # dynamics simulation, IPOPT) if available. This is the "real" version per spec.
        out = solve_ROP_constrained_OCP_casadi_skeleton(
            hat_x, f_star, P, horizon, bos_api=bos_api, return_traj=return_traj, robust_samples=robust_samples
        )
        if return_traj and isinstance(out, (list, tuple)) and len(out) == 2:
            alpha, traj = out
            # ensure published under a common key too
            if bos_api is not None:
                try:
                    bos_api.publish("fec_alpha_traj", traj, meta={"source": "casadi", "horizon": horizon})
                except Exception:
                    pass
            return out
        else:
            alpha = out
            if bos_api is not None:
                try:
                    bos_api.publish("fec_alpha_traj", [alpha for _ in range(horizon)], meta={"source": "casadi_fallback_const", "horizon": horizon})
                except Exception:
                    pass
            if return_traj:
                return alpha, [alpha for _ in range(horizon)]
            return alpha

    safe = _solve_toy_with_explicit_hrep_constraints(naive, P, f_star, horizon, hat_x)
    # P here can be ROPPolyhedron or (A,b) tuple
    if hasattr(P, "is_feasible"):
        assert P.is_feasible(safe), "FEC solver must always return a point inside the ROP"
    else:
        assert is_in_ROP(P, safe), "FEC solver must always return a point inside the ROP"

    # Phase2: if robust_samples, make conservative (project to intersection approx by successive project)
    if robust_samples:
        for sP in robust_samples:
            safe = project_onto_ROP(sP, safe) if hasattr(sP, "project") else safe
        # re-assert on nominal too
        if hasattr(P, "is_feasible"):
            assert P.is_feasible(safe), "Robust alpha must remain feasible in nominal ROP"

    if bos_api is not None:
        # Deep correspondence: publish intermediate + apply (so Signal history has the FEC decision)
        try:
            bos_api.publish("fec_alpha_opt", safe, meta={"source": "fec_solver", "horizon": horizon})
        except Exception:
            pass
        bos_api.apply_control(safe)

    if return_traj:
        const_traj = [list(safe) for _ in range(max(1, horizon))]
        if bos_api is not None:
            try:
                bos_api.publish("fec_alpha_traj", const_traj, meta={"source": "scipy_const", "horizon": horizon})
            except Exception:
                pass
        return safe, const_traj
    return safe


def solve_ROP_constrained_OCP_casadi_skeleton(
    hat_x: List[float],
    f_star: List[float],
    P: ROPPolyhedron,
    horizon: int,
    bos_api: Any = None,
    return_traj: bool = False,
    robust_samples: Optional[List[ROPPolyhedron]] = None,
) -> Any:
    """
    COMPLETE SKELETON for real CasADi + IPOPT implementation (Phase 0 spec §2.2 and §5).

    This demonstrates the exact way to encode ROP H-rep as CasADi constraints:
        for each row i in the H-rep (A, b) from rop_polyhedron:
            opti.subject_to( A[i, :] @ alpha[:, t] <= b[i] )

    Deepened: time-varying alpha(t) decision var, full Euler rollout in cost,
    explicit per-t subject_to (the comment in spec is now live code),
    initial guess seeded from scipy explicit H-rep solver (backend->frontend init correspondence),
    and optional bos_api publishes at key modeling points (for audit + real Signal history).
    Supports return_traj=True to return (mean, traj) for direct consumption by DigitalTwin.simulate_trajectory(alpha_seq=...).

    If casadi not installed, falls back to the scipy version (which already uses the explicit constraints).
    """
    if not HAS_CASADI:
        # Fall back to the scipy/explicit version which already encodes the H-rep
        return solve_ROP_constrained_OCP(hat_x, f_star, P, horizon, bos_api=bos_api, return_traj=return_traj)

    # --- Real CasADi implementation below (will run only if casadi present) ---
    global _LAST_SUCCESSFUL_TRAJ  # for warmstart cache (must be before any use in function)
    import casadi as ca
    A, b = P.A, P.b
    n_alpha = A.shape[1]
    dt = 0.1  # simple step

    # First get a good base from scipy explicit (deep correspondence: backend provides feasible point for frontend init)
    n = getattr(P, 'dim', 3) if hasattr(P, 'dim') else (P.A.shape[1] if hasattr(P, 'A') else 3)
    naive_local = [0.5, 0.8, 2.8] + [1.0] * (n - 3) if n > 3 else [0.5, 0.8, 2.8]
    base_alpha = _solve_toy_with_explicit_hrep_constraints(naive_local, P, f_star, horizon, hat_x)
    base_alpha = np.asarray(base_alpha, dtype=float)

    if bos_api is not None:
        try:
            bos_api.publish("fec_casadi_init", base_alpha.tolist(), meta={"from": "scipy_explicit_hrep"})
        except Exception:
            pass

    opti = ca.Opti()

    # Decision variables: time-varying alpha(t) for horizon (more complete than constant)
    # For toy, we optimize alpha over 'horizon' steps
    alpha = opti.variable(n_alpha, horizon)

    # Encode ROP H-rep constraints EXACTLY as in spec comment and review feedback
    # for every time step t
    # Note: use ca.mtimes/ca.dot to avoid numpy 2.0+ @ ufunc issues with casadi MX
    for t in range(horizon):
        for i in range(A.shape[0]):
            opti.subject_to( ca.dot( ca.DM(A[i, :]), alpha[:, t] ) <= b[i] )
        opti.subject_to( alpha[:, t] >= 0 )
        opti.subject_to( alpha[:, t] <= 5.0 )  # reasonable physical upper bound for reaction orders (stability)

    # Phase2 high-quality robust scenario MPC (Musk/Huang min-max / conservative robust):
    # If robust_samples provided (from DT.sample_robust or interval), add H-rep subject_to for EVERY sample.
    # This makes the solved alpha(t) feasible in all realizations (robust counterpart, structure-derived from first principles).
    # (For full min-max cost would duplicate rollout per sample + epigraph t; here conservative feasibility + nominal cost is high-quality practical.)
    if robust_samples:
        for si, sP in enumerate(robust_samples):
            sA, sb = getattr(sP, 'A', None), getattr(sP, 'b', None)
            if sA is None or sb is None:
                continue
            sA = np.asarray(sA); sb = np.asarray(sb).ravel()
            for t in range(horizon):
                for i in range(sA.shape[0]):
                    opti.subject_to( ca.dot( ca.DM(sA[i, :]), alpha[:, t] ) <= sb[i] )

    # Add smoothness on alpha changes for better numerical behavior (Phase1 style regularization)
    for t in range(1, horizon):
        opti.subject_to( ca.sum1( alpha[:, t] - alpha[:, t-1] )**2 <= 0.5 )  # limit big jumps

    # Better initial: project the scipy base (or safe nominal) onto ROP for a good, feasible warm start
    # Try to use last successful traj from bos_api if available for warm-start (deep correspondence)
    # Also use module cache for persistence across calls without bos_api (for reliability)
    init_guess = base_alpha
    last_traj = None
    if bos_api is not None:
        last_traj = bos_api.get("fec_alpha_traj_casadi") or bos_api.get("fec_alpha_traj")
    if not last_traj and _LAST_SUCCESSFUL_TRAJ is not None and len(_LAST_SUCCESSFUL_TRAJ) >= horizon:
        last_traj = _LAST_SUCCESSFUL_TRAJ
    if last_traj and len(last_traj) >= horizon:
        # average the time-varying traj for initial (or use first)
        traj_arr = np.array(last_traj[:horizon])
        init_guess = np.mean(traj_arr, axis=0)
    init_guess = np.asarray(init_guess, dtype=float)
    if len(init_guess) < n_alpha:
        init_guess = np.concatenate([init_guess, np.ones(n_alpha - len(init_guess))])
    try:
        init_guess = project_onto_ROP(P, init_guess) if hasattr(P, 'project') else init_guess
        init_guess = np.clip(init_guess, 0.0, 4.0)
        opti.set_initial(alpha, ca.repmat(ca.DM(init_guess), 1, horizon))
    except Exception:
        safe_init = np.array([0.55, 0.82, 1.65] + [1.0]*(n_alpha-3) if n_alpha > 3 else [0.55, 0.82, 1.65], dtype=float)
        opti.set_initial(alpha, ca.repmat(ca.DM(safe_init), 1, horizon))

    # Simple forward simulation of dynamics for cost (Euler)
    # x0 from hat_x (as list to array) - keep unscaled for correct dynamics (TOY_N is for actual concentrations)
    x = ca.DM(hat_x[:3])
    total_cost = 0
    k = ca.DM.ones(5)
    for t in range(horizon):
        # v = k .* x .^ alpha(:,t)  -- but for toy use the 3 alphas
        # Simplified v computation for toy (matches numerical_toy_validation)
        # Use fmax for x to avoid 0**alpha or negative base issues (numerical robustness)
        x_safe = ca.fmax(x, 1e-8)
        na = alpha.shape[0]
        # general for extended (Phase3 more real net); keep v=4 for N 3x4 compat, extra alpha only in ROP/constraints
        v_list = [k[i] * x_safe[min(i,2)] ** alpha[i, t] for i in range(min(na,4))]
        if len(v_list) < 4:
            v_list += [k[3] * x_safe[2] ** 1.0] * (4 - len(v_list))
        v = ca.vertcat(*v_list)
        f = v
        # tracking cost (ell) - scaled by 1/horizon for stability across different horizons (Musk-style normalization)
        total_cost += ca.sumsqr(f[:3] - ca.DM(f_star[:3])) * dt / horizon
        # alpha reg + smoothness reg (extra for robustness)
        target = ca.DM([0.6, 0.85, 1.7] + [1.0]*(na-3)) if na > 3 else ca.DM([0.6, 0.85, 1.7])
        total_cost += 0.05 * ca.sumsqr(alpha[:, t] - target[:na])
        if t > 0:
            total_cost += 0.01 * ca.sumsqr(alpha[:, t] - alpha[:, t-1])  # extra smoothness
        # euler step
        dx = ca.mtimes(ca.DM(TOY_N), v) * dt
        x = x + dx
        x = ca.fmax(x, 1e-8)  # keep positive

    opti.minimize(total_cost)

    # Solver settings (IPOPT as in spec) - relaxed a bit for toy robustness while keeping accuracy
    opts = {
        'print_time': False,
        'ipopt.print_level': 0,
        'ipopt.tol': 1e-6,
        'ipopt.acceptable_tol': 1e-4,
        'ipopt.max_iter': 200,
        'ipopt.hessian_approximation': 'limited-memory',
    }
    opti.solver('ipopt', opts)

    try:
        sol = opti.solve()
        # Return average alpha (mean over time/horizon) 
        # ca.sum2 for row means (sum over columns) / horizon ; no ca.mean in this casadi
        alpha_mean = sol.value( ca.sum2(alpha) / horizon ).tolist()
        alpha_traj = [sol.value(alpha[:, t]).tolist() for t in range(horizon)]
        if bos_api is not None:
            try:
                # publish the time-varying trajectory too (for DT simulate with alpha_seq)
                bos_api.publish("fec_alpha_traj_casadi", alpha_traj, meta={"horizon": horizon, "solver": "ipopt"})
            except Exception:
                pass
        if return_traj:
            return alpha_mean, alpha_traj
        print("[CasADi] IPOPT solve succeeded (full time-varying path used)")
        # Cache for warmstart (improves reliability on subsequent calls)
        _LAST_SUCCESSFUL_TRAJ = alpha_traj
        return alpha_mean
    except Exception as e:
        # If solve fails for any reason, fallback
        print(f"[CasADi warning] solve failed: {e}, falling back to scipy version")
        fb = base_alpha.tolist() if np.all(np.isfinite(base_alpha)) else [0.55, 0.82, 1.65]
        if return_traj:
            return fb, [fb for _ in range(max(1, horizon))]
        return fb  # use the backend feasible point


# ---------------------------------------------------------------------------
# Small standalone demo (useful while reading the spec)
# ---------------------------------------------------------------------------
def _demo_explicit_hrep_encoding():
    """Run this to see the exact H-rep encoding in action on the numbers
    from Phase 0 Spec v1.0 §6 (the 'alpha3=3.1' violation case)."""
    from .rop_polyhedron import build_rop_from_binding_stoichiometry, is_in_ROP

    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    P = rop.to_tuple()
    violating = [0.5, 0.8, 3.1]
    print("\n=== FEC toy demo: explicit H-rep constraint encoding (v1.0 spec) ===")
    print(f"Input (deliberately violating): {violating}")
    print(f"Inside ROP before? {rop.is_feasible(violating)}")

    result = solve_ROP_constrained_OCP(violating, [1.0, 1.0, 1.0], rop, horizon=3)
    print(f"Output after explicit row-by-row enforcement + projection: { [round(v, 3) for v in result] }")
    print(f"Inside ROP after? {rop.is_feasible(result)}")
    print("This is the manual version of what the CasADi loop `for i: subject_to(A[i]@alpha <= b[i])` will do.\n")
    return result

if __name__ == "__main__":
    _demo_explicit_hrep_encoding()
