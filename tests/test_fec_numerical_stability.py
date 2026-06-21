"""FEC toy dynamics must stay finite under long multicell-style rollouts."""

import warnings

import numpy as np

from bmac_engine.fec_solver import compute_v_toy, _clip_state
from bmac_engine.multicell_agent import MulticellularAgent, MulticellularSwarm
from bmac_engine.rop_polyhedron import build_rop_from_binding_stoichiometry


def test_compute_v_toy_large_state_no_overflow():
    x = np.array([1e6, 1e6, 1e6])
    alpha = np.array([4.0, 4.0, 4.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        v = compute_v_toy(x, alpha)
    overflow = [w for w in caught if "overflow" in str(w.message).lower()]
    assert not overflow, f"unexpected overflow warnings: {overflow}"
    assert np.all(np.isfinite(v))


def test_multicell_200_steps_no_fec_overflow_warnings():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    agents = [MulticellularAgent(i, rop, np.array([1.0, 1.0, 1.0])) for i in range(10)]
    swarm = MulticellularSwarm(agents, capacity=25.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        res = swarm.check_test_suggestion(n_steps=200, n_agents=10, death_every=40)
    overflow = [w for w in caught if "overflow" in str(w.message).lower()]
    assert res["passes_test"]
    assert not overflow, f"multicell run emitted overflow: {overflow}"


def test_clip_state_bounds():
    x = _clip_state(np.array([0.0, 1e9, -5.0]))
    assert x[0] >= 1e-12
    assert x[1] <= 50.0
    assert x[2] >= 1e-12
