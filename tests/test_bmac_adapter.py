"""Integration tests for ~/Projects/bos-platform/bmac_adapter (delegate mode)."""

import os
from pathlib import Path

import pytest

ADAPTER_ROOT = Path.home() / "Projects" / "bos-platform" / "bmac_adapter"
BMAC_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(not (ADAPTER_ROOT / "bos_platform" / "signal_control.py").is_file(),
                      reason="bmac_adapter not installed at ~/Projects/bos-platform/bmac_adapter")
def test_bmac_adapter_delegate_loads_and_runs_workflow():
    old_path = os.environ.get("BOS_PLATFORM_PATH")
    old_home = os.environ.get("BMAC_HOME")
    os.environ["BOS_PLATFORM_PATH"] = str(ADAPTER_ROOT)
    os.environ["BMAC_HOME"] = str(BMAC_ROOT)
    try:
        # Fresh loader read (avoid cached __init__ if already imported)
        from bos_platform.loader import load_bos_platform

        bundle = load_bos_platform(prefer_external=True)
        assert bundle.source.startswith("external:")
        assert "bmac_adapter" in bundle.source

        from bmac_engine.bos_integration import run_fec_step, simulate_fec_workflow
        from bmac_engine.rop_polyhedron import build_rop_from_binding_stoichiometry

        signal = bundle.SignalAPI()
        kalman = bundle.Kalman()
        temporal = bundle.TemporalWorkflow()
        opa = bundle.OPA(capacity=50.0)
        dt = bundle.DigitalTwin([1.0, 1.0, 1.0], noise=0.05)
        rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")

        wf = temporal.start_workflow("adapter_fec")
        run_fec_step(
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.8, 0.8],
            horizon=2,
            bos_api=signal,
            kalman=kalman,
            temporal_wf=temporal,
            workflow=wf,
            opa=opa,
            dt=dt,
        )
        assert len(wf["checkpoints"]) >= 1

        hist = simulate_fec_workflow(
            [1.0, 1.0, 1.0],
            [0.8, 0.8, 0.8],
            n_steps=2,
            horizon=2,
            bos_api=signal,
            kalman=kalman,
            temporal=temporal,
            opa=opa,
            dt=dt,
        )
        assert len(hist) == 2
        assert all(rop.is_feasible(a) for a in hist)
    finally:
        if old_path is None:
            os.environ.pop("BOS_PLATFORM_PATH", None)
        else:
            os.environ["BOS_PLATFORM_PATH"] = old_path
        if old_home is None:
            os.environ.pop("BMAC_HOME", None)
        else:
            os.environ["BMAC_HOME"] = old_home
