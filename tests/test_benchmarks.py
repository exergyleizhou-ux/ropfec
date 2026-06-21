"""Required FBA/MM vs ROP benchmark gates."""

from bmac_engine.benchmarks import run_fba_mm_benchmark
from bmac_engine.rop_polyhedron import build_rop_from_binding_stoichiometry
from bos_platform import DigitalTwin


def test_fba_mm_benchmark_rop_beats_fba_on_toy():
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    dt = DigitalTwin([1.0, 1.0, 1.0], noise=0.05)
    stats = run_fba_mm_benchmark(rop, dt, n_runs=30)
    assert stats["red_fba_vs_rop"] > 1.0, "ROP mean DT cost should beat FBA on toy"
    assert stats["rop_mean"] < stats["fba_mean"]
    assert stats["n_runs"] == 30.0
