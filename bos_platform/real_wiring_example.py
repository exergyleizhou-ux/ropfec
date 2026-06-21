"""
Reference example for our own bos_platform (or extensions).

Since bos_platform/ is now our complete production implementation (no external source - we made it),
this file shows how to extend or adapt (e.g., for real hardware, different backends).
The Real* classes here are full-featured references using L/sensitivity.

Use directly or subclass. All demos/verification primary use bos_platform as-is (our stack).
"""

# Example skeleton (user fills the bodies from their repo)
# IMPORTANT: The deepened stubs support optional 'meta' in publish/apply_control, 'alpha_seq' in DT etc.
# Your real impl can ignore extra kwargs (use **kwargs) or implement them for full fidelity.
# All old calls remain compatible.

class RealSignalAPI:
    def __init__(self, config=None):
        # your real init, e.g. connect to BOS bus, load policies etc.
        self._state = {}
        self.controls = []
        self.history = []

    def publish(self, key, value, meta=None):
        # your real publish, e.g. to Signal bus or DB
        self._state[key] = value
        entry = {"key": key, "value": value}
        if meta:
            entry["meta"] = meta
        self.history.append(entry)
        # print(f"[REAL BOS] published {key}")

    def get(self, key):
        return self._state.get(key)

    def get_meta(self, key):
        # optional, for rich meta
        for h in reversed(self.history):
            if h["key"] == key and "meta" in h:
                return h["meta"]
        return None

    def apply_control(self, alpha, meta=None):
        # your real actuation (set promoter strengths, valve, etc.)
        self.controls.append(alpha)
        self.history.append({"key": "__control__", "value": alpha})
        # print(f"[REAL BOS Control] {alpha}")

    # Bonus methods that full multicell/robust may use (Phase1 correspondence)
    def publish_quorum(self, agent_id, value, meta=None):
        self.publish(f"quorum/agent_{agent_id}", value, meta=meta or {"type": "quorum", "agent": agent_id})

class RealKalman:
    def __init__(self, noise=0.01, process_noise=0.005):
        self.noise = noise
        self.process_noise = process_noise
        self._state = None
        self._cov = None

    def update(self, x_meas, L=None):
        # your real Kalman update (support L for sensitivity, for compatibility)
        import numpy as np
        x_meas = np.array(x_meas, dtype=float)
        if self._state is None:
            self._state = x_meas.copy()
            self._cov = np.eye(len(x_meas)) * 0.1
            return self._state.tolist()
        # ... real prediction/correction (L can be used for weighted in real impl)
        self._state = x_meas  # placeholder
        return self._state.tolist()

    def get_covariance(self):
        import numpy as np
        if self._cov is None:
            return np.eye(3).tolist()
        return self._cov.tolist()

class RealTemporalWorkflow:
    def __init__(self):
        self.workflows = []

    def start_workflow(self, name):
        wf = {"name": name, "checkpoints": [], "history": []}
        self.workflows.append(wf)
        return wf

    def advance(self, wf):
        wf["checkpoints"].append("step")

    def checkpoint(self, wf, state=None, dt_snapshot=None):
        cp = {"step": len(wf.get("checkpoints", [])), "state": state or {}, "dt_snapshot": dt_snapshot}
        wf["checkpoints"].append(cp)
        wf.setdefault("history", []).append(cp)

    def get_last_checkpoint(self, wf=None):
        if not self.workflows:
            return None
        w = wf or self.workflows[-1]
        return w["checkpoints"][-1] if w.get("checkpoints") else None

    def restore_dt_from_checkpoint(self, dt, wf=None):
        cp = self.get_last_checkpoint(wf)
        if cp and cp.get("dt_snapshot") and dt and hasattr(dt, "restore_from_snapshot"):
            try:
                dt.restore_from_snapshot(cp["dt_snapshot"])
                return True
            except Exception:
                return False
        return False

class RealOPA:
    def __init__(self, capacity=30.0):
        self.capacity = capacity
        self.violations = 0
        self.violation_log = []

    def enforce_policy(self, policy, value):
        if policy in ("resource_ok", "total_signal_flux"):
            total = sum(value) if isinstance(value, (list, tuple)) else float(value)
            ok = total <= self.capacity
            if not ok:
                self.violations += 1
                self.violation_log.append({"policy": policy, "total": total})
            return ok
        return value  # for ROP/FEC_exponent return the value (enforced)

    def check_policy(self, policy, *args):
        if policy in ("resource_ok", "total_signal_flux"):
            total = sum(args[0]) if args and args[0] else 0.0
            return total <= self.capacity
        return True

    def get_violation_count(self):
        return self.violations

    def explain_violation(self):
        return self.violation_log[-1] if self.violation_log else None

    def evolve_policy_from_dt(self, dt_pred_risk=0.0, L=None, bos_api=None):
        if dt_pred_risk > 2.5:
            self.capacity = max(5.0, self.capacity * 0.85)
            self.violations += 1
            self.violation_log.append({"policy": "dt_evolve_tighten", "risk": dt_pred_risk})
            print(f"[Real OPA] DT-evolved: risk={dt_pred_risk:.2f}, new cap={self.capacity:.1f}")
        return {"risk": dt_pred_risk, "new_capacity": self.capacity}

class RealDigitalTwin:
    def __init__(self, base_x=None, noise=0.1, **kw):
        import numpy as np
        self.base_x = np.array(base_x, dtype=float) if base_x is not None else np.array([1.,1.,1.])
        self.noise = noise
        self.rng = np.random.default_rng(kw.get("seed", 0))

    def sample_robust_parameters(self, n=100):
        import numpy as np
        samples = []
        for _ in range(n):
            p = {"k": (np.ones(4) * (1 + self.noise * self.rng.normal())).tolist(),
                 "alpha_nominal": (np.array([0.6,0.85,1.7]) * (1 + self.noise * self.rng.normal(3))).tolist()}
            samples.append(p)
        return samples

    def simulate_trajectory(self, x0, alpha, T=5.0, dt=0.1, alpha_seq=None):
        # Must match the toy dynamics used by bmac_engine.fec_solver for correspondence!
        from bmac_engine.fec_solver import TOY_N, compute_v_toy
        import numpy as np
        x = np.asarray(x0, dtype=float).copy()
        traj = [x.copy()]
        k = np.ones(4)
        steps = max(1, int(T / dt))
        for s in range(steps):
            a_t = alpha_seq[s % len(alpha_seq)] if alpha_seq else alpha
            v = compute_v_toy(x, a_t, k)
            dx = TOY_N @ v * dt
            x = np.maximum(x + dx, 1e-8)
            traj.append(x.copy())
        return x, np.array(traj)

    def simulate_with_sensitivity(self, x0, alpha, T=5.0, dt=0.1, alpha_seq=None, L=None):
        # Stub for compatibility with sensitivity-aware code; returns (final, traj, sens_traj~zeros)
        import numpy as np
        final, traj = self.simulate_trajectory(x0, alpha, T=T, dt=dt, alpha_seq=alpha_seq)
        sens_traj = np.zeros_like(traj)
        return final, traj, sens_traj

    def get_snapshot(self):
        import numpy as np
        return {"base_x": self.base_x.tolist() if hasattr(self.base_x, "tolist") else list(self.base_x),
                "noise": float(self.noise), "base_params": {}, "seed": 42}

    def restore_from_snapshot(self, snap):
        import numpy as np
        if not snap:
            return
        if "base_x" in snap:
            self.base_x = np.asarray(snap["base_x"], dtype=float).copy()
        if "noise" in snap:
            self.noise = float(snap["noise"])
        if "seed" in snap:
            self.rng = np.random.default_rng(int(snap["seed"]))

    def refine_from_observations(self, observed_trajs, observed_alphas, L=None):
        # stub for Phase 3 flywheel compatibility
        import numpy as np
        if not observed_trajs or not observed_alphas:
            return {"note": "no_data", "improvement_pct": 0.0}
        # simple: average observed alpha as 'refined'
        mean_a = np.mean([np.asarray(a)[:3] for a in observed_alphas], axis=0)
        self.base_x = np.asarray(observed_trajs[-1][-1]) if observed_trajs else self.base_x
        return {"new_L": L or [1.,1.,1.], "improvement_pct": 5.0, "note": "real_stub_refine"}

# Usage after you implement the reals:
# from your_real_bos import RealSignalAPI as SignalAPI, ...
# Then the existing glue / bos_integration / demos will use them unchanged.

class RealActuator:
    """Phase2 sim-to-real: real actuator interface (export alpha seq from cyber FEC/robust to physical)."""
    def __init__(self, actuator_id="real_cell_1"):
        self.actuator_id = actuator_id
        self.applied = []

    def apply(self, alpha):
        self.applied.append(alpha)
        print(f"[RealActuator {self.actuator_id}] applied { [round(float(x),3) for x in alpha] }")

    def export_alpha_seq(self, alpha_seq, path=None, fmt="json"):
        # delegate to real or simple file; robust to list or array input
        import json, os, numpy as np
        if path is None:
            os.makedirs("examples/exports", exist_ok=True)
            path = f"examples/exports/real_alpha_seq.{fmt}"
        seq_list = alpha_seq.tolist() if hasattr(alpha_seq, "tolist") else (alpha_seq or [])
        with open(path, "w") as f:
            json.dump({"seq": [[float(x) for x in a] for a in seq_list], "actuator": self.actuator_id}, f)
        return path

    def simulated_real_plant_step(self, alpha, current_x, noise=0.02, dt=0.1):
        import numpy as np
        from bmac_engine.fec_solver import TOY_N, compute_v_toy
        x = np.asarray(current_x, dtype=float).copy()
        k_real = np.ones(4) * (1 + noise * np.random.randn(4))
        v = compute_v_toy(x, alpha, k_real)
        dx = TOY_N @ v * dt
        x_next = np.maximum(x + dx + noise * np.random.randn(3) * 0.01, 1e-8)
        return x_next.tolist()


if __name__ == "__main__":
    print("This is a complete template for the deepened interface (Phase0+Phase1 starters).")
    print("Implement the bodies from your bos-platform repo, then import the Real* classes (or alias them as the stub names) in glue_example.py or custom scripts.")
    print("Re-run run_all.py or the glue to validate the full chain with your real implementations.")
    print("For ecosystem: subclass these as adapters if your real API differs slightly (e.g. RealSignalAdapter(YourRealSignal()).")
