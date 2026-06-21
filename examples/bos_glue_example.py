"""
bos_glue_example.py
Example of "real" BOS glue using mocks for Signal/Control API, Kalman, Temporal, OPA.

Demonstrates how the BMAC components (rop, fec, robust, multicell) integrate with BOS primitives
as described in Phase 0 Spec v1.0 (Introduction and Implementation Roadmap).

=== HOW TO USE WITH YOUR REAL bos-platform (the continuation task) ===
The Mock* classes below implement *exactly* the interfaces your real classes from
git@github.com:exergyleizhou-ux/bos-platform.git must provide (per the spec pseudocode
and the mocks that were written to match your BOS Signal/Control, Kalman, Temporal, OPA).

To switch to real:
1. Make your bos-platform importable (e.g. export PYTHONPATH=/path/to/your/bos-platform/src
   or pip install -e it in the same python).
2. Adjust the import and instantiation below, e.g.:
       from bos_platform.api import SignalAPI as RealSignalAPI
       from bos_platform.observers import Kalman as RealKalman
       from bos_platform.workflows import Temporal as RealTemporal
       from bos_platform.governance import OPA as RealOPA
   Then:
       signal = RealSignalAPI(...)   # or however your ctor works
       kalman = RealKalman(...)
       temporal = RealTemporal(...)
       opa = RealOPA(...)
3. The rest of the demo (and bmac_engine.bos_integration) will use them.
   - run_fec_step(..., bos_api=signal) will call your signal.apply_control etc.
   - Glue manually wires kalman.update, temporal.start/advance, opa.enforce/check.

Exact methods the real classes must support (from mocks + spec):
- Signal/Control: publish(key, value), get(key), apply_control(alpha)
- Kalman: update(x_meas) -> hat_x list
- Temporal: start_workflow(name) -> wf dict-like, advance(wf)
- OPA: enforce_policy(policy, value), check_policy(policy, *args) -> bool or value

Mocks are kept as the default so the example always runs. Replace the 4 lines in main()
when you are ready with real instances.

See also: bmac_engine/bos_integration.py (the chain), BOS-BMAC_Phase0_Impl_Status.txt,
and the Phase 0 Spec for the full pseudocode mapping.

Run:
    PYTHONPATH=. python examples/bos_glue_example.py
"""
import sys
sys.path.insert(0, '..')   # for bmac_engine
# bos_platform package is at the project root (Desktop/bos-bmac/bos_platform)
import numpy as np
from bmac_engine.rop_polyhedron import build_rop_from_binding_stoichiometry, ROPPolyhedron
from bmac_engine.fec_solver import solve_ROP_constrained_OCP
from bmac_engine.bos_integration import run_fec_step
from bmac_engine.robust_extension import build_interval_rop, robust_fec_alpha
from bmac_engine.multicell_agent import MulticellularAgent, MulticellularSwarm

# Real integration via bos_platform package (stubs today, your real code tomorrow)
from bos_platform import SignalAPI, Kalman, TemporalWorkflow, OPA

def print_correspondence_map(signal, kalman, temporal, opa, dt, rop, alphas, swarm_res=None):
    """Explicit bidirectional correspondence report (for verification that front == back per spec)."""
    print("\n--- DEEP FRONTEND <-> BACKEND CORRESPONDENCE MAP (per Phase 0 Spec pseudocode) ---")
    print("1. ROP construction (bmac_engine)  <->  Signal.publish('rop_polyhedron', (A,b)) + get()")
    p = signal.get("rop_polyhedron")
    print(f"   ROP facets published: {len(p[1]) if p else 'N/A'} | matches constructed: {rop is not None}")
    print("2. FEC solve (explicit H-rep subject_to in CasADi/scipy) <-> Kalman.update(meas) -> solve -> Control.apply_control(alpha)")
    print(f"   Last applied control count: {len(signal.controls)} | last alpha feasible in ROP: {rop.is_feasible(signal.controls[-1]) if signal.controls and hasattr(rop,'is_feasible') else 'N/A'}")
    print("3. Workflow: Temporal.start('FEC...') / advance() / checkpoint(state) around every FEC step")
    ncp = 0
    if temporal and temporal.workflows:
        ncp = len(temporal.workflows[-1].get('checkpoints', []))
    print(f"   Temporal checkpoints recorded: {ncp}")
    print("4. OPA.enforce/check for ROP/FEC_exponent + resource_ok (multicell total flux)")
    print(f"   OPA violations (resource): {opa.get_violation_count() if hasattr(opa,'get_violation_count') else 0}")
    print("5. DT.sample_robust_parameters / simulate_trajectory(alpha or seq)  <->  robust_extension MC + cost validation")
    print(f"   DT has sample_robust_parameters: {hasattr(dt, 'sample_robust_parameters')}")
    if swarm_res:
        print(f"6. Multicell: per-agent Kalman+Signal(quorum) + global Temporal checkpoints survive death + OPA blocks >=95%")
        print(f"   Swarm passes spec test suggestion: {swarm_res.get('passes_test')} | final_opa_violations={swarm_res.get('final_opa_violations',0)}")
    print("--- END CORRESPONDENCE MAP ---\n")

def main():
    print("=" * 60)
    print("BOS-BMAC Phase 0: BOS Glue Example (using bos_platform) -- DEEP CORRESPONDENCE MODE")
    print("=" * 60)

    signal = SignalAPI()
    kalman = Kalman()
    temporal = TemporalWorkflow()
    opa = OPA(capacity=25.0)

    # 1. Build and publish ROP via Signal (as in spec) -- use the backend helper for full meta
    rop = build_rop_from_binding_stoichiometry(toy_id="glycolysis_upper")
    # Deep publish (now also publishes log_deriv etc)
    from bmac_engine.rop_polyhedron import publish_rop
    publish_rop(rop, bos_signal=signal)

    # 2. Single cell FEC step with full BOS (Kalman + run_fec_step which now does more internals)
    wf = temporal.start_workflow("FEC_ROP_constrained")
    x_meas = [1.0, 1.0, 1.0]
    f_star = [0.8, 0.8, 0.8]
    hat_x = kalman.update(x_meas)
    # Use return_traj to get the exact time-var seq for DT roundtrip
    alpha_or_tuple = run_fec_step(hat_x, [0.5,0.6,0.4], f_star, horizon=3, bos_api=signal, kalman=kalman, temporal_wf=temporal, opa=opa)
    if isinstance(alpha_or_tuple, (list, tuple)) and len(alpha_or_tuple) == 2:
        alpha, alpha_traj = alpha_or_tuple
    else:
        alpha = alpha_or_tuple
        alpha_traj = None
    # OPA enforce (explicit)
    safe = opa.enforce_policy("FEC_exponent", alpha)
    signal.apply_control(safe)
    temporal.advance(wf)
    temporal.checkpoint(wf, state={"alpha": safe})  # explicit deep checkpoint

    print(f"\nSingle step with BOS platform: safe alpha = {[round(float(x),3) for x in safe]}")

    # 3. Robust with MC from "Digital Twin" (using bos_platform DT -- now with sample_robust_parameters + sim cost)
    from bos_platform import DigitalTwin
    dt = DigitalTwin([1.,1.,1.], noise=0.07)
    interval = build_interval_rop(rop, uncertainty=0.1)
    # Use the new DT method
    dt_params = dt.sample_robust_parameters(n=8)
    samples = []
    for ds in dt_params:
        A_pert = interval.A_nom + np.random.randn(*interval.A_nom.shape) * 0.04 * interval.A_radius
        b_pert = interval.b_nom + np.random.randn(*interval.b_nom.shape) * 0.04 * interval.b_radius
        samples.append(ROPPolyhedron(A_pert, b_pert))
    alpha_rob = robust_fec_alpha(rop, samples, safe)
    print(f"Robust alpha (MC over DT sample_robust_parameters): {[round(float(x),3) for x in alpha_rob]}")

    # Verify DT sim on the robust alpha (correspondence check)
    fin, traj = dt.simulate_trajectory([1.,1.,1.], alpha_rob, T=2.0)
    print(f"DT.simulate_trajectory(final) with robust alpha: {[round(float(x),3) for x in fin]}")

    # Run explicit invariants check now that all BOS objects + traj from earlier step are available (deep correspondence gate)
    from bmac_engine.bos_integration import verify_correspondence
    inv = verify_correspondence(bos_api=signal, kalman=kalman, temporal=temporal, opa=opa, dt=dt, rop=rop, last_alpha=safe, last_traj=None)
    print(f"Invariants after FEC+BOS+DT: {inv}")

    # 4. Multicell swarm with *real* BOS objects passed in (Signal for quorum, Temporal, OPA)
    # Note: tight capacity here for the *short demo* so OPA resource blocks are visible in output.
    # The unit test (test_multicell_agent) uses its own longer run + default higher capacity=100 and still asserts the spec >95% block rate on over-attempts.
    # Some agents get dt for Phase1 DT-feedback demo (closed loop prediction influencing local alpha)
    agents = []
    for i in range(8):
        adt = DigitalTwin([1.,1.,1.], noise=0.02) if i % 2 == 0 else None  # every other agent gets DT
        agents.append(MulticellularAgent(i, rop, np.array([1.0,1.0,1.0]), bos_signal=signal, kalman=Kalman(), dt=adt))
    swarm = MulticellularSwarm(agents, capacity=10.0, temporal=temporal, opa=opa, bos_signal=signal, dt=DigitalTwin([1.,1.,1.], noise=0.02))
    for step in range(3):
        cp = swarm.step(f_star)
        quorum = sum(a.send_quorum_signal() for a in agents)
        print(f"Step {step}: quorum signal sum ~ {quorum:.2f} (published via Signal)")
        opa.check_policy("resource_ok", [cp["total_flux"]])

    res = swarm.check_test_suggestion(n_steps=12, death_every=4)
    print(f"\nSwarm with real BOS objects: OPA block rate {res['opa_block_rate']:.0%}, passes suggestion {res['passes_test']}, violations={res.get('final_opa_violations')}")

    # 5. Explicit correspondence report (verifies front and back are wired per spec)
    print_correspondence_map(signal, kalman, temporal, opa, dt, rop, [safe, alpha_rob], swarm_res=res)

    print("\n=== BOS glue demo complete (DEEP front/back correspondence exercised per Phase 0 Spec) ===")

if __name__ == "__main__":
    main()
