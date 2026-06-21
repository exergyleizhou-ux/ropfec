"""
test_robust_extension.py
Basic tests + the exact test suggestion from Phase 0 Spec v1.0 §5 for robust_extension.py.
"""

import numpy as np
from bmac_engine.rop_polyhedron import build_rop_from_binding_stoichiometry
from bmac_engine.rop_polyhedron import ROPPolyhedron
from bmac_engine.robust_extension import (
    build_interval_rop,
    robust_fec_alpha,
    check_robust_test_suggestion,
)
try:
    from bos_platform import DigitalTwin
except Exception:
    DigitalTwin = None

def test_interval_and_robust():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    interval = build_interval_rop(rop, uncertainty=0.15)
    samples = interval.sample(20)
    assert len(samples) == 20

    nominal = np.array([0.5, 0.8, 1.5])
    rob = robust_fec_alpha(rop, samples, nominal)
    assert rop.is_feasible(rob)
    assert all(s.is_feasible(rob) for s in samples), "robust alpha must lie in every scenario ROP"

def test_robust_alpha_inside_all_samples():
    """Regression: median-of-projections could be outside some scenario ROPs."""
    samples = [
        ROPPolyhedron(np.array([[1.0]]), np.array([1.0])),
        ROPPolyhedron(np.array([[1.0]]), np.array([10.0])),
    ]
    nominal = ROPPolyhedron(np.array([[1.0]]), np.array([5.0]))
    rob = robust_fec_alpha(nominal, samples, np.array([5.0]))
    assert all(s.is_feasible(rob) for s in samples)
    assert nominal.is_feasible(rob)

def test_spec_suggestion():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    interval = build_interval_rop(rop, uncertainty=0.1)
    dt = DigitalTwin([1.,1.,1.]) if DigitalTwin else None
    passes = check_robust_test_suggestion(interval, n_samples=60, dt=dt)
    # For this toy data the simple robust choice should pass the 80% + all-inside bar
    assert passes, "robust test suggestion should pass for reasonable uncertainty"

if __name__ == "__main__":
    test_interval_and_robust()
    test_spec_suggestion()
    print("robust_extension tests passed (incl. spec test suggestion)")
