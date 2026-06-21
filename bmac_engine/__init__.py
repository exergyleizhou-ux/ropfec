"""
bmac_engine - BOS-BMAC Phase 0 implementation (per v1.0 spec)
"""
__version__ = "0.1.0"
from .rop_polyhedron import (
    build_rop_from_binding_stoichiometry,
    project_onto_ROP,
    is_in_ROP,
    ROPPolyhedron,
    publish_rop,
    enforce_rop,
    compute_log_derivative,
    augment_with_data,
)
from .fec_solver import solve_ROP_constrained_OCP, solve_ROP_constrained_OCP_casadi_skeleton, HAS_CASADI, TOY_N, compute_v_toy
from .bos_integration import run_fec_step
from .robust_extension import (
    IntervalROP,
    build_interval_rop,
    robust_fec_alpha,
    check_robust_test_suggestion,
)
from .multicell_agent import (
    MulticellularAgent,
    MulticellularSwarm,
)

__all__ = [
    "build_rop_from_binding_stoichiometry",
    "project_onto_ROP",
    "is_in_ROP",
    "ROPPolyhedron",
    "publish_rop",
    "enforce_rop",
    "compute_log_derivative",
    "augment_with_data",
    "solve_ROP_constrained_OCP",
    "solve_ROP_constrained_OCP_casadi_skeleton",
    "HAS_CASADI",
    "TOY_N",
    "compute_v_toy",
    "run_fec_step",
    "IntervalROP",
    "build_interval_rop",
    "robust_fec_alpha",
    "check_robust_test_suggestion",
    "MulticellularAgent",
    "MulticellularSwarm",
]
