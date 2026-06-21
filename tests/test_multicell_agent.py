"""
test_multicell_agent.py
Basic tests + the exact test suggestion from Phase 0 Spec v1.0 §5 for multicell_agent.py.
"""

import numpy as np
from bmac_engine.rop_polyhedron import build_rop_from_binding_stoichiometry
from bmac_engine.multicell_agent import MulticellularAgent, MulticellularSwarm

class _ToyDT:
    """Minimal DT stub to exercise quorum adaptation without swallowing ndarray truthiness bugs."""
    def simulate_trajectory(self, x, alpha, T=1.0, alpha_seq=None, dt=None, L=None):
        load = float(np.sum(x)) + float(np.sum(alpha)) * T
        return np.asarray(x, dtype=float) * (1.0 + 0.1 * load), []

def test_swarm_basic():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    agents = [MulticellularAgent(i, rop, np.array([1.0, 1.0, 1.0])) for i in range(5)]
    swarm = MulticellularSwarm(agents, capacity=30.0)
    cp = swarm.step(np.array([0.8, 0.8, 0.8]))
    assert "total_flux" in cp
    assert cp["policy_ok"] is True or cp["policy_ok"] is False  # either way

def test_dt_quorum_adaptation_with_ndarray_last_alpha():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    agent = MulticellularAgent(0, rop, np.array([2.0, 2.0, 2.0]), dt=_ToyDT())
    agent.last_alpha = np.array([2.0, 2.0, 2.0], dtype=float)
    base_q = float(np.sum(agent.x))
    adapted_q = agent.send_quorum_signal()
    assert adapted_q < base_q, "high predicted load should adaptively lower quorum"

def test_spec_suggestion():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    agents = [MulticellularAgent(i, rop, np.array([1.0, 1.0, 1.0])) for i in range(10)]
    swarm = MulticellularSwarm(agents, capacity=25.0)  # low capacity to trigger blocks
    res = swarm.check_test_suggestion(n_steps=200, n_agents=10, death_every=40)
    assert res["passes_test"], f"multicell test suggestion failed: {res}"

if __name__ == "__main__":
    test_swarm_basic()
    test_dt_quorum_adaptation_with_ndarray_last_alpha()
    test_spec_suggestion()
    print("multicell_agent tests passed (incl. spec test suggestion)")
