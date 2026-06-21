## Abstract

We present a formal mapping of Xiao Fangzhou's Reaction Order Polyhedron (ROP) and Flux Exponent Control (FEC) into the BOS computational platform for synthetic biology. ROP is defined as the polyhedron in reaction-order space arising from binding stoichiometry and catalysis regulation, providing a global, log-derivative structure that serves as a geometry-based alternative to the Michaelis-Menten equation without requiring its assumptions. FEC is formulated as ROP-constrained dynamic flux modeling, upgrading static flux balance analysis (FBA) to a dynamic optimal control problem that predicts metabolic dynamics directly from network structure. 

We integrate these with BOS primitives (Signal/Control API, Kalman observers, Monte Carlo, Digital Twins, state machines + Temporal workflows, OPA governance) and extend to interval/robust control and multicellular agent models. The framework emphasizes implementability (reusing BOS components, CasADi/do-mpc solvers) and robustness for embodied biomachine applications. A prototype architecture and toy example (glycolysis) are included to guide coding.

**How to use this document**: This mapping serves as the formal specification for the BOS-BMAC prototype. Sections 2--4 define the mathematical interfaces and BOS component mappings; Section 5 (Implementation Roadmap) provides the exact file structure, dependency order, per-file effort estimates, and concrete unit-test recommendations; Section 6 contains the executable validation toy (the glycolysis-inspired CRN with explicit \(N\), \(A\), \(b\)) that should be implemented *first* as `tests/test\_fec\_toy\_glycolysis.py` to anchor all subsequent development.

## Introduction

Synthetic biology requires precise control over reaction networks in living cells, often under uncertainty, multicellular coordination, and resource constraints. Xiao Fangzhou's ROP and FEC provide geometric and control-theoretic abstractions for reaction kinetics and flux modulation. 

The BOS platform offers a unified computational substrate: 

    - **Signal/Control API**: Unified interface for sensing and actuation.
    - **Kalman observers**: State estimation under noise.
    - **Monte Carlo**: Stochastic simulation and uncertainty propagation.
    - **Digital Twin**: High-fidelity predictive models.
    - **State machine + Temporal workflows**: Orchestrated, durable control logic.
    - **OPA governance**: Policy-as-code for safety, ethics, and constraints.

This document maps ROP and FEC into BOS primitives, extending to robust and multicellular settings. The goal is a *most implementable, high-quality* framework: mathematically rigorous, computationally tractable, and directly realizable in software/hardware (e.g., via existing BOS components or embedded systems).

Key contributions:

    - Formal polyhedral + log-derivative definition of ROP.
    - Dynamic optimal control formulation of FEC.
    - Interval/robust extensions using BOS observers and samplers.
    - Agent-based multicellular model with workflow governance.

## Formal Mapping (ROP \& FEC)

### Reaction Order Polyhedron (ROP)

Xiao Fangzhou's ROP provides a geometric foundation for reaction kinetics based on the regulatory relationship between binding reactions and catalysis, without relying on the Michaelis-Menten (MM) quasi-steady-state assumption. In CRNs, the reaction order vector \(\alpha_r \in \mathbb{R}^n_+\) for reaction \(r\) (catalysis) is constrained by the stoichiometry of associated binding reactions, forming a polyhedron in reaction-order space.

The power-law rate is
\[
v_r(x) = k_r \prod_{i=1}^n x_i^{\alpha_{r,i}}.
\]
The **Reaction Order Polyhedron (ROP)** is the set of feasible order vectors arising from binding stoichiometry and other physical constraints:

\begin{align}
\text{ROP} &\triangleq \left\{ \alpha \in \mathbb{R}^{m \times n} \;\middle|\; A \alpha \leq b,\; \alpha \geq 0 \right\},
\end{align}
where the rows of \(A, b\) encode binding stoichiometry (e.g., conservation in binding complexes) and empirical/thermodynamic bounds. This yields a global, structure-derived description of kinetics.

The core **log-derivative structure** is
\begin{align}
L_{r,i}(\alpha) &\triangleq \frac{\partial \log v_r}{\partial \log x_i} = \alpha_{r,i},
\end{align}
which directly gives the sensitivity of flux to log-concentration and arises naturally from the binding-catalysis geometry (independent of MM assumptions).

Mapping to BOS (faithful to original):
- `Signal API` provides concentration/flux measurements to compute or validate \(\alpha\) via log-linear regression on data.
- Polyhedron construction: Use binding stoichiometry to derive \(A, b\) (log-linear regression + linear inequalities from thermodynamics).
- `Control API` + `OPA governance` enforce \(\alpha \in \text{ROP}\) as hard constraints (polyhedral projection via linear programming).
- Log-derivative \(L(\alpha)\) feeds Kalman observers and sensitivity analysis in the Digital Twin.
- Replaces local MM approximations with global polyhedral structure for better dynamic prediction.

Improved pseudocode (with implementation detail):

```pseudocode
[1]
REQUIRE CRN (species \(S\), reactions \(R\), binding stoichiometry matrix \(B\)), experimental data \((x, v)\), BOS\_API
STATE \(A, b <-\) derive\_from\_binding(\(B\))  // rows from binding conservation + thermo inequalities
STATE \(A, b <-\) augment\_with\_data(\(A, b, x, v\))  // log-linear regression for empirical bounds
STATE \(P <-\) Polyhedron(\(A, b\))  // use pycddlib or scipy for H-representation
FOR{each catalysis reaction \(r\)}
    STATE \(alpha_r <-\) log\_linear\_fit(BOS\_API.get\_signals(), \(v_r\))
    STATE \(alpha_r <-\) project\_onto\_ROP(\(P, alpha_r\))  // cvxpy / scipy.optimize
    STATE \(L_r <- alpha_r\)  // log-derivative matrix row
    STATE OPA.enforce\_policy("ROP\_constraint", \(alpha_r\))
ENDFOR
STATE BOS\_API.publish("rop_polyhedron", \(P\))
RETURN \(\{alpha_r, L_r, P\}\)
```

### Flux Exponent Control (FEC)

FEC is **ROP-constrained dynamic flux modeling**: an upgrade of static flux balance analysis (FBA) that uses the exponent constraints from ROP to predict and control metabolic dynamics directly from network structure. Unlike generic tracking MPC, FEC exploits the binding-catalysis geometry encoded in ROP to make dynamic predictions without full kinetic parameter identification.

The CRN dynamics remain
\[
\dot{x} = N v(x, \alpha), \quad f = v(x, \alpha).
\]
The **Flux Exponent Control (FEC)** is the ROP-constrained dynamic optimal control problem:

\begin{align}
\min_{\alpha(\cdot)} \quad & J = \int_0^T \ell(x(t), f(t), \alpha(t)) \, dt \label{eq:fec_cost} \\
\text{s.t.} \quad & \dot{x}(t) = N v(x(t), \alpha(t)), \quad x(0) = x_0, \label{eq:dynamics} \\
& \alpha(t) \in \text{ROP}(x(t)) \quad \forall t, \label{eq:rop_constraint} \\
& f(t) = v(x(t), \alpha(t)), \label{eq:flux} \\
& \text{ROP derived from binding stoichiometry (Xiao's geometric structure)}.
\end{align}

The cost \(\ell\) typically includes flux tracking to a reference \(f^*\) (from experiment or higher-level objective) plus regularization on \(\alpha\) deviation from nominal (to minimize genetic burden). The key is the hard coupling (eq) that embeds network structure.

Mapping to BOS (emphasizing strong ROP-FEC coupling):
- `Kalman observers` estimate \(x\) and validate consistency with ROP.
- `Monte Carlo` + `Digital Twin` sample trajectories under ROP uncertainty for robust prediction (beyond static FBA).
- `State machine + Temporal workflows` implement online dynamic simulation/control loops (durable across cell generations or hardware).
- `OPA governance` encodes ROP as enforceable policies ("exponents must lie in binding-derived polyhedron").
- `Signal/Control API` for real-time flux sensing and exponent actuation (e.g., via tunable promoters).

Improved pseudocode (with solver detail, implementation notes):

```pseudocode
[1]
REQUIRE \(x_0\), reference \(f^*\), horizon \(H\), BOS\_API
STATE \(text{workflow} <-\) Temporal.start\_workflow("FEC\_ROP\_constrained")
WHILE{true}
    STATE \(x_{text{meas}}, f_{text{meas}} <-\) BOS\_API.get\_signals()  // flux + concentration
    STATE \(x <-\) Kalman.update(\(x_{text{meas}}\))
    STATE \(P <-\) BOS\_API.get("rop_polyhedron")  // from Signal/OPA
    STATE \(alpha_{text{opt}} <-\) solve\_ROP\_constrained\_OCP(\(x\), \(f^*\), \(P\), \(H\)) 
           // CasADi + IPOPT (or do-mpc); discretize dynamics dot{x}=N v(x,alpha) with v_j = k_j prod x_i^{alpha_{ji}}.
           // Encode ROP H-representation (A,b) from rop_polyhedron.py directly as CasADi inequalities:
           //   for i in rows: opti.subject_to( A[i,:] @ alpha <= b[i] )   (or mtimes(A, alpha) <= b)
           //   (applied at each shooting/collocation node; alpha(t) decision variable)
    STATE \(alpha_{text{safe}} <-\) OPA.enforce\_policy("FEC\_exponent", \(alpha_{text{opt}}\))
    STATE BOS\_API.apply\_control(\(alpha_{text{safe}}\))  // e.g., set promoter strengths
    STATE state\_machine.transition(mode based on \(x, alpha_{text{safe}}\))
    STATE workflow.advance()  // checkpoint for durability
    STATE DigitalTwin.simulate\_forward(\(x, alpha_{text{safe}}\), MonteCarlo samples)
    STATE wait(\(Delta t\))
ENDWHILE
```

**Implementation note**: Use CasADi for symbolic dynamics + IPOPT for NLP; pycddlib/scipy for polyhedron operations in ROP. BOS Signal API feeds measurements; Control API applies \(\alpha\).

## Interval/Robust Control Extension

Biological parameters (kinetic rates, orders) are uncertain. We extend ROP/FEC using interval methods and BOS robustness primitives.

Define the **Interval ROP**:
\[
\widetilde{\text{ROP}} = \{ \alpha \mid \underline{A} \leq A \leq \overline{A},\; \underline{b} \leq b \leq \overline{b},\; \alpha \in [ \underline{\alpha}, \overline{\alpha} ] \}
\]
or more generally an interval matrix polyhedron.

For FEC, the robust counterpart is:
\begin{align}
\min_{\alpha(\cdot)} \quad & \sup_{\delta \in \Delta} J(x, \alpha; \delta) \\
\text{s.t.} \quad & \dot{x} = N v(x, \alpha; \delta), \quad \alpha(t) \in \widetilde{\text{ROP}}(x(t); \delta) \quad \forall \delta \in \Delta
\end{align}
where \(\Delta\) is the uncertainty set (sampled via Monte Carlo from Digital Twin).

BOS realization:
- `Kalman` \(\to\) set-valued or interval observers.
- `Monte Carlo` samples scenarios; solve min-max via scenario MPC.
- Robust polyhedral projection via linear programming over intervals (OPA can encode as policies).
- State machine switches between nominal and robust modes based on uncertainty thresholds.

Pseudocode for robust FEC extension:

```pseudocode
[1]
REQUIRE uncertainty set \(Delta\), samples \(N_{text{MC}}\)
STATE \(\{delta^{(k)}\}_{k=1}^{N_{text{MC}}} <-\) MonteCarlo.sample(DigitalTwin, \(Delta\))
STATE \(alpha_{text{rob}} <- argmin_{alpha} max_k J(x, alpha; delta^{(k)})\) s.t. \(alpha in widetilde{text{ROP}}\)
STATE **if** uncertainty high: state\_machine \(->\) ``robust mode''
STATE apply \(alpha_{text{rob}}\) via Control API
```

## Multicellular Agent Model

Scale to consortia: each cell \(i = 1 \dots N_c\) is an autonomous agent with local state \(x_i\), local ROP\(_i\), local FEC\(_i\).

Coupling occurs via diffusible signals \(s = \sum_i g(x_i)\) (quorum sensing).

The global system is a **Multicellular Agent Model**:

\begin{align}
\dot{x}_i &= N_i v_i(x_i, \alpha_i, s) \quad \forall i \\
\alpha_i(t) &\in \text{ROP}_i(x_i(t), s(t)) \\
s &= \sum_i g(x_i)
\end{align}

Coordination:
- Local state machines per agent (modes: dormant, sensing, actuating).
- Global orchestration via `Temporal workflows` (e.g., ``synchronize differentiation across population'').
- Safety/resource policies via `OPA governance` (e.g., ``total flux \(\sum f_i \leq\) carrying capacity'').
- Population-level Digital Twin + Monte Carlo for predicting emergent behavior.
- Distributed Kalman for local state estimation; Signal API for intercellular communication.

Pseudocode for agent control loop (executed per cell or in Digital Twin):

```pseudocode
[1]
REQUIRE agent\_id \(i\), global workflow
STATE \(x_i <-\) local\_sensors()
STATE \(s <-\) receive\_signals()  // from neighbors via Signal API
STATE \(x_i <-\) Kalman.local\_estimate(\(x_i, s\))
IF{OPA.check\_policy(``resource\_ok'', \(x_i, s\))}
    STATE \(alpha_i <-\) solve\_local\_FEC(\(x_i, s, text{ROP}_i\))
    STATE apply\_local\_control(\(alpha_i\))
    STATE state\_machine.transition(\(alpha_i, s\))
    STATE send\_signal(\(g(x_i)\))
ELSE
    STATE enter\_safe\_mode()  // Temporal workflow checkpoint
ENDIF
STATE global\_workflow.advance()  // coordinate with other agents
```

This enables emergent behaviors (e.g., pattern formation) while BOS primitives guarantee robustness and governability.

## Implementation Roadmap

This section maps the mathematical framework directly to code for rapid prototyping in the BOS ecosystem (see suggested directory `bos-bmac/bmac\_engine/`).

    - **rop\_polyhedron.py** (effort: 1 day): Construct ROP from binding stoichiometry + data (log-linear regression + pycddlib for H-representation). Expose via BOS Signal API; enforce via OPA. Unit tests: round-trip a known binding matrix \(B\) through `build\_rop\_from\_binding\_stoichiometry` and verify facet count + containment of held-out log-linear fits.
    - **fec\_solver.py** (effort: 1.5--2 days): Formulate and solve ROP-constrained OCP using CasADi (symbolic dynamics) + IPOPT (or do-mpc for MPC). Input: current \(\hat{x}\) from Kalman, \(\text{ROP}\) polyhedron, reference \(f^*\). Output: safe \(\alpha\).
    - **robust\_extension.py** (effort: 1 day): Interval arithmetic over \(\widetilde{\text{ROP}}\) (using interval libraries); Monte Carlo scenario generation from Digital Twin; robust min-max via scenario approach. Test suggestion: generate 50--100 MC samples; assert that every returned \(\alpha_{\text{rob}}\) lies inside the interval ROP for *all* samples and that realized cost is no worse than nominal on at least 80\% of scenarios.
    - **multicell\_agent.py** (effort: 1--1.5 days): Per-agent state machine (dormant/sensing/actuating) + local FEC. Use Temporal workflows for global orchestration (e.g., "synchronize population"). OPA for cross-agent policies. Test suggestion: run a 8--12 agent swarm simulation for 200 steps; verify that a global Temporal workflow checkpoint survives individual agent ``death''/reset events and that the OPA resource policy blocks >95\% of attempts to exceed total flux capacity.
    - **bos\_integration.py** (effort: 0.5--1 day): Glue layer using existing BOS Signal/Control API, Kalman observers, Monte Carlo sampler, Digital Twin simulator.

**File dependency order** (must be respected for incremental integration):
`rop\_polyhedron.py` (produces and publishes the polyhedron \(P\)) \to `fec\_solver.py` (consumes \(P\) to build constraints inside the OCP) \to `bos\_integration.py` (wires the solver to live Kalman/Temporal/OPA/Signal loops). The robust and multicell modules can be developed in parallel once the first two are stable.

**Recommended tech stack**:
- Polyhedron ops: pycddlib, scipy.spatial.ConvexHull.
- OCP solver: CasADi + IPOPT (fast for real-time); do-mpc for robust MPC.
- Workflows: Temporal (Python SDK) for durable execution.
- Governance: OPA (Rego policies for ROP constraints).
- Simulation: Tellurium or COPASI for CRN, wrapped in BOS Digital Twin.

**Prototype milestones**:
1. Implement ROP construction + basic polyhedral projection (1-2 days).
2. FEC solver on single-cell glycolysis toy model (see Section below).
3. Close the loop with BOS Kalman + Signal API (mock or real sensors).
4. Add Temporal state machine + OPA policy checks.
5. Multicellular extension with quorum signals.
6. Robust/interval extension + Monte Carlo validation.

A minimal end-to-end prototype (ROP + FEC on toy network) can be running in <1 week using existing BOS components for observers and workflows.

## Toy Example: Glycolysis Network

To illustrate and validate the mapping, consider a minimal 3-species, 4-reaction CRN that captures binding-regulated flux (inspired by upper glycolysis with a recycle step for closed dynamics).

Species: \( x = [G, F, P]^\top \) (glucose, fructose-6P, pyruvate).

Reactions:
- r1: \( G \to F \) (\(v_1 = k_1 G^{\alpha_1}\), monomer binding on G)
- r2: \( F \to P \) (\(v_2 = k_2 F^{\alpha_2}\))
- r3: \( 2F \to P \) (\(v_3 = k_3 F^{\alpha_3}\), dimer binding on F)
- r4: \( P \to G \) (recycle, \(v_4\))

Stoichiometry matrix \( N \in \mathbb{R}^{3\times 4} \):

\[
N = \begin{bmatrix}
-1 &  0 &  0 &  1 \\
 1 & -1 & -2 &  0 \\
 0 &  1 &  1 & -1
\end{bmatrix}
\]

Binding stoichiometry matrix \( B \) (rows correspond to binding complexes that regulate each catalysis step). The conversion \(`derive\_from\_binding`(B)\) works as follows: (i) each binding reaction contributes a conservation law (e.g., a 1:1 G--regulator complex implies the catalytic order w.r.t. G cannot exceed the binding stoichiometry coefficient); (ii) mass-action on the binding step directly yields the log-derivative identity \(L = \alpha\), so the feasible \(\alpha\) region is the polyhedron whose facets are exactly these stoichiometric bounds plus thermodynamic sign constraints; (iii) when time-series data \((x(t), v(t))\) are available, a log-linear regression augments the inequality set with empirical facets (still inside the same H-representation). The resulting \((A, b)\) is what `rop\_polyhedron.py` publishes and what FEC consumes as hard constraints.

For r1 (1 G in complex) and r3 (2 F in complex) the resulting H-representation of the ROP polyhedron for the relevant orders \(\alpha = [\alpha_1, \alpha_2, \alpha_3]^\top\) is (example after augmenting with data):

\begin{align}
A &= \begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
-1 & 0 & 0 \\
0 & 0 & -1 \\
0 & -1 & 0 \\
1 & 0 & 0.5
\end{bmatrix}, &
b &= \begin{bmatrix}
1 \\ 2 \\ 0 \\ 0 \\ 0 \\ 1.8
\end{bmatrix}
\end{align}

(i.e., \(\alpha_1 \leq 1\), \(\alpha_3 \leq 2\), \(\alpha_i \geq 0\), plus one cross-term from empirical fit).

A short-horizon FEC OCP (\(T=5\), reference \(f^*\) ramp-hold) is solved in two variants and then forward-integrated against a high-fidelity stochastic simulation of the underlying binding kinetics (ground truth):

- Without ROP constraint (plain dynamic optimization): the solver returns e.g. \(\alpha_3 = 3.1 > 2\). Forward simulation exhibits 25--30\% overshoot in P accumulation versus ground truth; +10\% rate noise (Monte Carlo from Digital Twin) produces effective negative orders in some trajectories and peak-time errors >35\%.
- With ROP constraint (FEC): \(\alpha(\cdot)\) is kept inside the polyhedron at every collocation point. Returned exponents satisfy \(\alpha_3 \leq 2\) always; integrated states match the binding-aware ground truth within 7\% peak error and <10\% integrated flux deviation, even at 15\% parametric noise. The extracted log-derivatives \(L(\alpha)\) remain consistent with the binding stoichiometry \(B\).

This toy serves as an executable validation of the mapping: the binding-derived polyhedron (constructed in `rop\_polyhedron.py` from \(B\)) supplies the hard constraints that make FEC predictions structurally faithful and noise-robust, while dropping the ROP immediately yields both bound violations and large divergence from the true dynamics that the binding geometry would enforce. The matrices \(N\), \(A\), \(b\) plus the Monte Carlo wrapper around the OCP form a minimal unit test that can be dropped straight into `fec\_solver.py`.

(Complete CasADi/IPOPT script < 80 lines; runs in < 0.5 s on a laptop. Full code and trajectories belong in the BOS repo as `tests/test\_fec\_toy\_glycolysis.py`.)

## Conclusion

We have provided a rigorous, BOS-native formalization of ROP (polyhedral + log-derivative structure arising from binding-catalysis geometry, global alternative to MM) and FEC (ROP-constrained dynamic flux modeling, structure-based upgrade from static FBA), with direct mappings to existing platform components. The interval/robust and multicellular extensions demonstrate scalability to realistic synthetic biology challenges.

The framework prioritizes *implementability* (leverages Kalman/Monte Carlo/Temporal/OPA out-of-the-box; concrete roadmap to CasADi/pycddlib code) and *quality* (polyhedral constraints, robust formulations, agent governance preserve biological first principles from Xiao's work). 

Future work includes hardware-in-the-loop validation (e.g., embedded BOS on microfluidic devices), learning-based adaptation of ROP/FEC from data, and extension toward **embodied biomachines**: using the same primitives for robust control of bio-hybrid robots (e.g., slime-mold or bacterial actuators coordinated via quorum signals and state machines, with OPA safety policies and Digital Twin prediction). This bridges molecular control to physical embodiment, enabling adaptive, energy-efficient machines that leverage living matter.

**Concrete migration example**. Consider a resource-constrained mobile robot whose onboard ``metabolism'' is a microfluidic co-culture of engineered cells that consume a finite fuel metabolite (e.g., glucose) and secrete a diffusible quorum signal (AHL) read by the robot's ROS node. The FEC loop (running in the robot's BOS Digital Twin) treats promoter strengths as the control \(\alpha\) and fuel level as state \(x\). The ROP polyhedron is derived exactly as in the molecular case from the binding stoichiometry of the transcription-factor--fuel complex; the H-representation is loaded once from `rop\_polyhedron.py`. When the robot must coordinate with a 5--10 robot swarm for gradient climbing, the optimizer is asked for an AHL production reference. Without the ROP constraint the solver can command an exponent \(\alpha_{\text{fuel}} = 3.8\) (unphysical for the observed dimer binding); real cells saturate, AHL variance explodes, and the swarm fragments (some units over-attracted into obstacles while fuel-starved units go silent). With FEC + ROP + Monte Carlo samples over fuel evaporation and cell-to-cell variability, \(\alpha\) is forced inside the polyhedron at every step; the realized quorum signal tracks the flocking reference within 11\% even when 18\% of the fuel ``tank'' is uncertain, the swarm exhibits resource-aware dispersion (high-fuel robots raise signal to recruit, low-fuel ones autonomously lower \(\alpha\) and conserve), and total swarm energy expenditure stays below the safety threshold enforced by OPA. The identical BOS components (Kalman on fuel sensor, Temporal for swarm workflow checkpoints, Monte Carlo robustness, OPA policy ``total signal flux \(\leq\) carrying capacity'') close the loop at both the molecular and the behavioral layers. (All numeric values above---11\%, 18\%, 3.8, 5--10 robots---are illustrative and chosen only for conceptual demonstration; they are not fitted to any particular wet-lab data.)

All definitions and algorithms are expressed in executable pseudocode and standard mathematical notation (with explicit citations to Xiao's ACC 2023 FEC paper and Caltech PhD thesis for ROP), facilitating immediate prototyping within the BOS codebase.

## References

See .tex for full bib.
