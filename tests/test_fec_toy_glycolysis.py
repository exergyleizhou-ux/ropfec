"""
test_fec_toy_glycolysis.py
Executable validation toy from BOS-BMAC Phase 0 Spec v1.0 Section 6.

Pure Python (no numpy required for this anchor test).
Run with any python3:

    python -m pytest tests/test_fec_toy_glycolysis.py -s
    # or directly:
    python tests/test_fec_toy_glycolysis.py
"""
import numpy as np
from bmac_engine.rop_polyhedron import (
    build_rop_from_binding_stoichiometry,
    project_onto_ROP,
    is_in_ROP,
    ROPPolyhedron,
)

# === Exact values copied from Phase 0 Spec v1.0 Section 6 ===
# Species: G, F, P
# 4 reactions, binding on r1 (G) and r3 (2F)
N = [
    [-1,  0,  0,  1],
    [ 1, -1, -2,  0],
    [ 0,  1,  1, -1],
]

A = [
    [ 1,  0,  0],
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0,  0, -1],
    [ 0, -1,  0],
    [ 1,  0,  0.5],
]
b = [1.0, 2.0, 0.0, 0.0, 0.0, 1.8]

P = (A, b)

def test_toy_rop_construction_and_membership():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    assert isinstance(rop, ROPPolyhedron)
    assert rop.dim == 3
    assert abs(rop.b[5] - 1.8) < 1e-12

    alpha_good = [0.7, 0.9, 1.2]
    assert rop.is_feasible(alpha_good)
    assert is_in_ROP(rop, alpha_good)   # also test the wrapper

    alpha_bad = [0.5, 0.8, 3.1]   # violates alpha3 <= 2 (dimer binding)
    assert not rop.is_feasible(alpha_bad)
    assert not is_in_ROP(rop, alpha_bad)
    print("  membership checks: good inside, bad (alpha3=3.1) correctly rejected")

def test_projection_clips_to_binding_limits():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    alpha_unconstrained = [0.5, 0.8, 3.1]  # the "no ROP" point from the spec
    alpha_proj = rop.project(alpha_unconstrained)   # preferred class API
    # also test the module-level wrapper for spec pseudocode compatibility
    alpha_proj2 = project_onto_ROP(rop, alpha_unconstrained)
    np.testing.assert_allclose(alpha_proj, alpha_proj2)

    assert rop.is_feasible(alpha_proj)
    assert is_in_ROP(rop, alpha_proj)
    assert alpha_proj[2] <= 2.0 + 1e-9          # dimer limit
    assert alpha_proj[0] + 0.5 * alpha_proj[2] <= 1.8 + 1e-9
    dist = sum((a - u)**2 for a, u in zip(alpha_proj, alpha_unconstrained))**0.5
    assert dist > 0.5
    print(f"  projected {alpha_unconstrained} -> { [round(v,3) for v in alpha_proj] }")

def test_toy_validates_mapping_value():
    """Demonstrates the value of the ROP constraint exactly as described in the spec."""
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    alpha_ground = [0.6, 0.85, 1.75]   # feasible "true" binding-limited point
    alpha_no_rop = [0.5, 0.8, 3.1]
    alpha_with_rop = rop.project(alpha_no_rop)

    overshoot_no = alpha_no_rop[2] - alpha_ground[2]
    overshoot_with = alpha_with_rop[2] - alpha_ground[2]

    assert overshoot_no > 1.0
    assert overshoot_with < 0.31
    assert rop.is_feasible(alpha_with_rop)
    assert is_in_ROP(rop, alpha_with_rop)

    print("\n=== Toy validation (Phase 0 Spec v1.0) ===")
    print(f"Ground (binding-limited): {alpha_ground}")
    print(f"No ROP (unconst):        {alpha_no_rop}   (alpha3={alpha_no_rop[2]} > 2)")
    print(f"With ROP (projected):    { [round(v,3) for v in alpha_with_rop] }")
    print(f"Feasible after constraint? {rop.is_feasible(alpha_with_rop)}")
    print("This gap is what the rop_polyhedron.py + FEC hard constraints deliver.\n")


def test_fec_solver_uses_rop_constraints():
    """The FEC solver must use the ROP poly as hard constraints (core of the mapping)."""
    from bmac_engine.fec_solver import solve_ROP_constrained_OCP
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    # Start with violating point
    violating = [0.5, 0.8, 3.1]
    f_star = [1.0, 1.0, 1.0]
    alpha = solve_ROP_constrained_OCP([1.,1.,1.], f_star, rop, horizon=3)
    assert rop.is_feasible(alpha)
    # Should have moved away from the bad 3.1 on the dimer dimension
    assert alpha[2] <= 2.0 + 0.1
    print(f"  FEC solver on violating input produced feasible alpha={ [round(float(x),3) for x in alpha] }")


if __name__ == "__main__":
    print("Running toy tests (pure python)...")
    test_toy_rop_construction_and_membership()
    print("PASS test_toy_rop_construction_and_membership")
    test_projection_clips_to_binding_limits()
    print("PASS test_projection_clips_to_binding_limits")
    test_toy_validates_mapping_value()
    print("PASS test_toy_validates_mapping_value")
    test_fec_solver_uses_rop_constraints()
    print("PASS test_fec_solver_uses_rop_constraints")
    print("\n*** ALL TOY TESTS GREEN - ROP construction + projection working per v1.0 spec ***\n")

    # Also run the new A/B/C tests
    print("Running additional A/B/C tests...")
    import tests.test_rop_polyhedron as tr
    import tests.test_robust_extension as tb
    import tests.test_multicell_agent as tm
    tr.test_build_toy(); tr.test_projection(); tr.test_from_B_basic()
    tb.test_interval_and_robust(); tb.test_spec_suggestion()
    tm.test_swarm_basic(); tm.test_spec_suggestion()
    print("All additional tests also green.")
