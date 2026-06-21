"""
test_rop_polyhedron.py
Unit tests for rop_polyhedron.py as requested in the implementation prompt.

Covers:
- Building from toy binding (exact A/b from Phase 0 Spec v1.0 §6)
- Feasibility and projection
- log-derivative
- augment_with_data (basic)
- BOS integration hooks (stubs)
- Basic B -> poly construction (simple case)
"""

import numpy as np
from bmac_engine.rop_polyhedron import (
    build_rop_from_binding_stoichiometry,
    ROPPolyhedron,
    project_onto_ROP,
    is_in_ROP,
    compute_log_derivative,
    augment_with_data,
    publish_rop,
    enforce_rop,
)

# Exact from spec §6
TOY_A = np.array([
    [ 1,  0,  0],
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0,  0, -1],
    [ 0, -1,  0],
    [ 1,  0,  0.5],
], dtype=float)
TOY_b = np.array([1., 2., 0., 0., 0., 1.8])

def test_build_toy():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    assert isinstance(rop, ROPPolyhedron)
    assert rop.dim == 3
    assert np.allclose(rop.b, TOY_b)
    # Check one facet
    assert rop.is_feasible([0.5, 0.5, 1.5])
    assert not rop.is_feasible([0.5, 0.5, 2.5])

def test_projection():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    bad = np.array([0.5, 0.8, 3.1])
    assert not rop.is_feasible(bad)
    proj = rop.project(bad)
    assert rop.is_feasible(proj)
    # Dimer binding
    assert proj[2] <= 2.0 + 1e-8
    # Wrapper
    proj2 = project_onto_ROP(rop, bad)
    assert np.allclose(proj, proj2)

def test_log_derivative():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    alpha = [0.7, 0.9, 1.2]
    L = rop.compute_log_derivative(alpha)
    assert np.allclose(L, alpha)
    # top level
    assert np.allclose(compute_log_derivative(alpha), alpha)

def test_augment():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    # Fake time series matching the 3 order dimensions (species-like for toy)
    x = np.array([[1.0, 1.2, 0.9], [1.5, 1.0, 1.1], [0.8, 1.2, 1.0]])
    v = np.array([0.5, 0.7, 0.4])  # fake "fluxes" for the regulated step
    rop2 = augment_with_data(rop, x=x, v=v)
    assert isinstance(rop2, ROPPolyhedron)
    assert len(rop2.b) > len(rop.b)  # should have added some facets

def test_from_B_basic():
    # Simple B: 2 species, 2 "regulated" reactions
    # B rows = binding complexes
    B = np.array([
        [1, 0],  # one complex binds 1 of species 0 for rxn 0
        [0, 2],  # another binds 2 of species 1 for rxn 1
    ])
    rop = build_rop_from_binding_stoichiometry(B=B)
    assert rop.dim == 2
    # Should have alpha0 <=1 , alpha1 <=2 , and >=0
    assert rop.is_feasible([0.9, 1.5])
    assert not rop.is_feasible([1.1, 0.5])
    assert not rop.is_feasible([0.5, 2.1])

def test_bos_hooks():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    # Should not crash
    publish_rop(rop)
    alpha = [0.1, 0.2, 0.3]
    safe = enforce_rop(alpha, rop)
    assert rop.is_feasible(safe)

if __name__ == "__main__":
    test_build_toy()
    test_projection()
    test_log_derivative()
    test_augment()
    test_from_B_basic()
    test_bos_hooks()
    print("All rop_polyhedron unit tests passed!")
