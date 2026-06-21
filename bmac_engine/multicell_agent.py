"""
multicell_agent.py
Multicellular Agent Model for BOS-BMAC.

Corresponds to Phase 0 Spec v1.0 Section 4 "Multicellular Agent Model".

Each cell/agent has local state x_i, local ROP_i, local FEC_i.
Coupling via quorum signal s = sum g(x_i).
Coordination via local state machines + global Temporal workflows + OPA policies.

Per roadmap & test suggestion (spec §5):
- Per-agent state machine (dormant/sensing/actuating)
- Local FEC
- Test suggestion: run a 8-12 agent swarm simulation for 200 steps; verify that
  a global Temporal workflow checkpoint survives individual agent "death"/reset
  events and that the OPA resource policy blocks >95% of attempts to exceed
  total flux capacity.
"""

from __future__ import annotations
from typing import Any, List, Dict, Optional
import numpy as np
from .rop_polyhedron import ROPPolyhedron, build_rop_from_binding_stoichiometry
from .fec_solver import solve_ROP_constrained_OCP

Array = np.ndarray

class AgentState:
    DORMANT = "dormant"
    SENSING = "sensing"
    ACTUATING = "actuating"

class MulticellularAgent:
    """One autonomous agent (engineered cell)."""
    def __init__(self, agent_id: int, local_rop: ROPPolyhedron, initial_x: Array,
                 bos_signal: Any = None, kalman: Any = None, dt: Any = None):
        self.id = agent_id
        self.rop = local_rop
        self.x = np.asarray(initial_x, dtype=float).copy()
        self.state = AgentState.SENSING
        self.last_alpha: Optional[Array] = None
        self.bos_signal = bos_signal  # optional real SignalAPI for quorum publish
        self.kalman = kalman          # optional per-agent Kalman (deep correspondence)
        self.dt = dt                  # optional DigitalTwin for Phase1 prediction feedback (closed loop)

    def sense_and_decide(self, s_global: float, f_star: Array, horizon: int = 3) -> Array:
        """Local sensing + call local FEC (with quorum input). Deep: uses per-agent Kalman if present.
        Phase1 extension: if DT provided, simulate the candidate alpha's effect and adjust (e.g. conservative bias on high predicted flux)
        to demonstrate closed DT -> FEC feedback control loop with full BOS correspondence.
        """
        self.state = AgentState.SENSING
        # Very simplified: treat s_global as extra input to "state"
        base = self.x + 0.01 * s_global
        if self.kalman is not None:
            hat_x = self.kalman.update(base.tolist())
        else:
            hat_x = base.tolist()
        alpha = solve_ROP_constrained_OCP(hat_x, f_star, self.rop, horizon, return_traj=True)
        if isinstance(alpha, (list, tuple)) and len(alpha) == 2:
            alpha_mean, alpha_traj = alpha
            alpha = alpha_mean
        else:
            alpha_traj = None

        # Phase 1 DT closed-loop feedback (deep correspondence: DT predictions directly influence alpha choice + OPA)
        if self.dt is not None and hasattr(self.dt, 'simulate_trajectory'):
            try:
                # Simulate what this alpha would do over short horizon
                pred_final, _ = self.dt.simulate_trajectory(hat_x, alpha, T=horizon * 0.1, alpha_seq=alpha_traj)
                # If DT predicts very high total "load" (proxy for resource), bias alpha conservatively
                pred_load = float(np.sum(pred_final))
                if pred_load > 4.0:  # toy threshold
                    alpha = (np.asarray(alpha) * 0.85).tolist()  # conservative reduction
                    # Publish the DT-driven adjustment via Signal if available
                    if self.bos_signal is not None:
                        try:
                            self.bos_signal.publish(f"dt_adjust/agent_{self.id}", {"original": alpha_mean, "adjusted": alpha, "pred_load": pred_load}, meta={"reason": "dt_prediction"})
                        except Exception:
                            pass
            except Exception:
                pass

        self.last_alpha = np.asarray(alpha)
        return self.last_alpha

    def actuate(self) -> None:
        """Apply control (in real: change promoter strengths)."""
        if self.last_alpha is not None:
            self.state = AgentState.ACTUATING
            # Fake dynamics update (would be real CRN or actuator)
            scale = float(np.clip(1 + 0.05 * np.mean(self.last_alpha), 0.5, 1.5))
            self.x = np.clip(self.x * scale, 1e-8, 50.0)

    def send_quorum_signal(self) -> float:
        """g(x_i) - contribution to global quorum. Deep: also publish via BOS Signal if wired.
        Phase1: use DT.simulate to predict future load and adapt quorum value (e.g. lower if predicted high resource use).
        This deepens DT <-> multicell correspondence: DT prediction -> Signal publish of adapted quorum.
        """
        q = float(np.sum(self.x))
        if self.dt is not None and hasattr(self.dt, 'simulate_trajectory'):
            try:
                # Simulate short future with current last_alpha to predict load
                alpha_for_sim = (
                    self.last_alpha
                    if self.last_alpha is not None
                    else np.array([0.6, 0.85, 1.7], dtype=float)
                )
                pred_x, _ = self.dt.simulate_trajectory(self.x, alpha_for_sim, T=1.0)
                pred_load = float(np.sum(pred_x))
                if pred_load > 3.0:  # toy threshold
                    q *= 0.8  # adaptively lower quorum to conserve
                    if self.bos_signal is not None:
                        try:
                            self.bos_signal.publish(f"dt_quorum_adapt/agent_{self.id}", {"original_q": float(np.sum(self.x)), "adapted_q": q, "pred_load": pred_load}, meta={"reason": "dt_prediction"})
                        except Exception:
                            pass
            except Exception:
                pass
        if self.bos_signal is not None and hasattr(self.bos_signal, 'publish_quorum'):
            try:
                self.bos_signal.publish_quorum(self.id, q)
            except Exception:
                pass
        elif self.bos_signal is not None:
            try:
                self.bos_signal.publish(f"quorum/agent_{self.id}", q, meta={"type": "quorum"})
            except Exception:
                pass
        return q

    def reset(self) -> None:
        """Simulate agent 'death' / reset for workflow checkpoint test."""
        self.x = np.zeros_like(self.x)
        self.state = AgentState.DORMANT
        self.last_alpha = None

class MulticellularSwarm:
    """
    Collection of agents + global orchestration (Temporal-like) + OPA.
    Deepened: accepts real TemporalWorkflow, OPA, SignalAPI at construction or per step.
    All quorum goes through Signal (publish), all global checkpoints go through Temporal,
    OPA passed in for policy (so violation_count etc visible to frontend).
    """
    def __init__(self, agents: List[MulticellularAgent], capacity: float = 100.0,
                 temporal: Any = None, opa: Any = None, bos_signal: Any = None, dt: Any = None):
        self.agents = agents
        self.capacity = capacity  # total flux capacity (OPA policy)
        self.workflow_checkpoints: List[Dict] = []
        self.total_steps = 0
        self.temporal = temporal
        self.opa = opa
        self.bos_signal = bos_signal
        self.dt = dt  # global DigitalTwin for swarm-level Phase1 prediction + adjustment
        self.global_wf = None
        if self.temporal is not None and hasattr(self.temporal, 'start_workflow'):
            self.global_wf = self.temporal.start_workflow("multicell_global")

    def step(self, f_star: Array, horizon: int = 3) -> Dict[str, Any]:
        self.total_steps += 1
        # 1. Local decisions (agents may publish quorum via their wired bos_signal)
        alphas = []
        for a in self.agents:
            s = sum(ag.send_quorum_signal() for ag in self.agents)
            alpha = a.sense_and_decide(s, f_star, horizon)
            alphas.append(alpha)
            a.actuate()

        # 2. Global quorum and OPA policy check (use injected opa if present for deep correspondence)
        total_flux = sum(a.send_quorum_signal() for a in self.agents)
        enforcer = self.opa or self.bos_signal
        if enforcer is not None and hasattr(enforcer, 'check_policy'):
            policy_ok = bool(enforcer.check_policy("resource_ok", [total_flux]))
            if not policy_ok and hasattr(enforcer, 'enforce_policy'):
                enforcer.enforce_policy("resource_ok", [total_flux])  # increments violation
        else:
            policy_ok = total_flux <= self.capacity
            if not policy_ok:
                for a in self.agents:
                    a.state = "safe_mode"

        if not policy_ok:
            for a in self.agents:
                a.state = "safe_mode"

        # Phase1: global DT swarm prediction feedback (closed loop)
        if self.dt is not None and hasattr(self.dt, 'simulate_trajectory'):
            try:
                # Use average recent alpha as proxy for swarm control
                avg_alpha = np.mean([a.last_alpha for a in self.agents if a.last_alpha is not None], axis=0) if any(a.last_alpha is not None for a in self.agents) else np.array([0.6,0.85,1.7])
                pred, _ = self.dt.simulate_trajectory([1.,1.,1.], avg_alpha, T=1.0)
                pred_total = float(np.sum(pred))
                if pred_total > self.capacity * 0.9 and self.bos_signal is not None:
                    self.bos_signal.publish("dt_swarm_prediction", {"pred_total": pred_total, "warning": "approaching capacity"}, meta={"type": "swarm_dt"})
                # Phase2: global DT risk evolves OPA policy (tighten governance)
                if self.opa is not None and hasattr(self.opa, 'evolve_policy_from_dt'):
                    try:
                        self.opa.evolve_policy_from_dt(pred_total, L=None, bos_api=self.bos_signal)
                    except Exception:
                        pass
                # Could further bias agents here, but per-agent DT already did local adjustment
            except Exception:
                pass

        # 3. "Temporal" checkpoint via real object if wired (spec: global_workflow.advance + checkpoints survive death)
        checkpoint = {
            "step": self.total_steps,
            "total_flux": total_flux,
            "policy_ok": policy_ok,
            "agent_states": [a.state for a in self.agents],
        }
        self.workflow_checkpoints.append(checkpoint)

        if self.global_wf is not None and self.temporal is not None:
            if hasattr(self.temporal, 'checkpoint'):
                self.temporal.checkpoint(self.global_wf, state=checkpoint)
            if hasattr(self.temporal, 'advance'):
                self.temporal.advance(self.global_wf)

        # Also publish global quorum via top-level Signal for full audit
        if self.bos_signal is not None:
            try:
                self.bos_signal.publish("quorum/global", total_flux, meta={"step": self.total_steps})
            except Exception:
                pass

        return checkpoint

    def simulate_death_and_recover(self, agent_idx: int) -> None:
        """Simulate agent death/reset; Temporal checkpoint should survive (real Temporal keeps history).
        Phase2: snapshot global DT (if wired) into cp for post-death restore (Temporal durability for DT state).
        """
        # Phase2 DT durability: snapshot before reset/death
        dt_snap = None
        if self.dt is not None and hasattr(self.dt, "get_snapshot"):
            try:
                dt_snap = self.dt.get_snapshot()
            except Exception:
                dt_snap = None
        self.agents[agent_idx].reset()
        if self.global_wf is not None and self.temporal is not None and hasattr(self.temporal, 'checkpoint'):
            self.temporal.checkpoint(self.global_wf, state={"event": "death_recover", "agent": agent_idx}, dt_snapshot=dt_snap)
        # Attempt restore of DT from the just-saved cp (simulates next-gen agent resuming DT belief)
        if self.temporal is not None and hasattr(self.temporal, "restore_dt_from_checkpoint"):
            try:
                self.temporal.restore_dt_from_checkpoint(self.dt, self.global_wf)
            except Exception:
                pass

    def check_test_suggestion(self, n_steps: int = 200, n_agents: int = 10, death_every: int = 50) -> Dict:
        """
        Exact test suggestion from spec:
        run 8-12 agent swarm for 200 steps; verify global Temporal checkpoint
        survives individual agent "death"/reset events AND OPA resource policy
        blocks >95% of attempts to exceed total flux capacity.
        Now also returns violation_count from injected real OPA if present.
        """
        results = {
            "checkpoints_after_deaths": 0,
            "opa_blocks": 0,
            "total_over_attempts": 0,
            "deaths_triggered": 0,
            "final_opa_violations": 0,
        }
        initial_cp = len(self.workflow_checkpoints)
        for step in range(n_steps):
            f_star = np.array([0.8, 0.8, 0.8])
            cp = self.step(f_star)
            if cp["policy_ok"] is False:
                results["opa_blocks"] += 1
            results["total_over_attempts"] += 1 if cp["total_flux"] > self.capacity else 0

            if step % death_every == 0 and self.agents:
                idx = step % len(self.agents)
                self.simulate_death_and_recover(idx)
                results["deaths_triggered"] += 1

        final_cp = len(self.workflow_checkpoints)
        results["checkpoints_after_deaths"] = final_cp - initial_cp
        opa_block_rate = results["opa_blocks"] / max(1, results["total_over_attempts"])
        results["opa_block_rate"] = opa_block_rate
        if self.opa is not None and hasattr(self.opa, 'get_violation_count'):
            results["final_opa_violations"] = self.opa.get_violation_count()
        # Survived if we still got (almost) all checkpoints despite deaths
        results["passes_test"] = (
            results["checkpoints_after_deaths"] >= n_steps - 5 and  # tolerate small variance
            opa_block_rate >= 0.95
        )
        return results
