"""
Integration wiring tests — invariants that must hold for real BOS correspondence.

These are stricter than toy demos: they fail if imports, checkpoints, or
robust feasibility regress.
"""

import numpy as np

from bmac_engine.rop_polyhedron import ROPPolyhedron, build_rop_from_binding_stoichiometry
from bmac_engine.robust_extension import robust_fec_alpha, DT
from bmac_engine.bos_integration import run_fec_step, simulate_fec_workflow, verify_correspondence
from bmac_engine.multicell_agent import MulticellularAgent, MulticellularSwarm
from bos_platform import SignalAPI, Kalman, TemporalWorkflow, OPA, DigitalTwin


def test_bos_platform_importable_from_engine():
    assert DT is not None, "robust_extension must import bos_platform.DigitalTwin"


def test_temporal_checkpoint_records_entry_state():
    tw = TemporalWorkflow()
    wf = tw.start_workflow("integration_fec")
    run_fec_step(
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.8, 0.8, 0.8],
        horizon=2,
        temporal_wf=tw,
        workflow=wf,
    )
    assert len(wf["checkpoints"]) == 1
    state = wf["checkpoints"][0]["state"]
    assert "x_meas" in state and "hat_x" in state and "f_star" in state


def test_full_bos_workflow_records_checkpoints_and_feasible_alphas():
    signal = SignalAPI()
    kalman = Kalman()
    temporal = TemporalWorkflow()
    opa = OPA(capacity=50.0)
    dt = DigitalTwin([1.0, 1.0, 1.0], noise=0.05)
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")

    hist = simulate_fec_workflow(
        [1.0, 1.0, 1.0],
        [0.8, 0.8, 0.8],
        n_steps=3,
        horizon=2,
        bos_api=signal,
        kalman=kalman,
        temporal=temporal,
        opa=opa,
        dt=dt,
    )
    assert len(hist) == 3
    assert all(rop.is_feasible(a) for a in hist)

    wf = temporal.workflows[-1]
    assert len(wf["checkpoints"]) >= 1

    inv = verify_correspondence(
        bos_api=signal,
        kalman=kalman,
        temporal=temporal,
        opa=opa,
        dt=dt,
        rop=rop,
        last_alpha=hist[-1],
    )
    assert inv.get("alpha_in_rop", False)
    assert inv.get("rop_published", False)


def test_robust_median_regression_constructed_case():
    samples = [
        ROPPolyhedron(np.array([[1.0]]), np.array([1.0])),
        ROPPolyhedron(np.array([[1.0]]), np.array([10.0])),
    ]
    nominal = ROPPolyhedron(np.array([[1.0]]), np.array([5.0]))
    rob = robust_fec_alpha(nominal, samples, np.array([5.0]))
    assert all(s.is_feasible(rob) for s in samples)
    assert nominal.is_feasible(rob)


def test_multicell_dt_quorum_adaptation_with_ndarray_alpha():
    class _ToyDT:
        def simulate_trajectory(self, x, alpha, T=1.0, **kwargs):
            load = float(np.sum(x)) + float(np.sum(alpha)) * T
            return np.asarray(x, dtype=float) * (1.0 + 0.1 * load), []

    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    agent = MulticellularAgent(0, rop, np.array([2.0, 2.0, 2.0]), dt=_ToyDT())
    agent.last_alpha = np.array([2.0, 2.0, 2.0], dtype=float)
    base_q = float(np.sum(agent.x))
    adapted_q = agent.send_quorum_signal()
    assert adapted_q < base_q


if __name__ == "__main__":
    test_bos_platform_importable_from_engine()
    test_temporal_checkpoint_records_entry_state()
    test_full_bos_workflow_records_checkpoints_and_feasible_alphas()
    test_robust_median_regression_constructed_case()
    test_multicell_dt_quorum_adaptation_with_ndarray_alpha()
    print("integration wiring tests passed")
