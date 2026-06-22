# ropfec

**An open, tested reference implementation of reaction-order-polytope (ROP) constrained
flux-exponent control (FEC) for staged bioconversion — with a built-in falsification of
structural priors.**

`ropfec` makes the ROP/FEC framework of Xiao, Li & Doyle *runnable and testable*: it
constructs the reaction-order polytope from binding stoichiometry, solves ROP-constrained
FEC as a CasADi/IPOPT optimal-control problem, and ships robust/interval, multicellular, and
data-driven-uncertainty modules plus validated re-implementations of two published
glycolytic oscillators (Sel'kov; Wolf–Heinrich).

**Honest scope.** This is a research reference implementation. No novel scientific result is
claimed — ROP, FEC and the log-derivative geometry are the prior work of Xiao and colleagues.
The contribution is the open, tested, reproducible implementation plus two re-runnable
results:

1. **A clarifying identity** — the exponents FEC controls are exactly the elasticity
   coefficients of metabolic control analysis (MCA).
2. **A built-in falsification** — tested against a count-matched random-facet null, the
   binding-derived ROP does **not** improve data-driven estimation of reaction orders. The
   transferable thesis: reaction-order structure is a *control-admissibility set, not an
   estimation prior*.

All demonstrations are **in-silico**, with reference components and published or synthetic
ground truth. The `bos_platform/` package here is an **in-tree reference stub** of the
estimation/governance/twin interfaces, not a production platform.

## Install

```bash
pip install -e .            # NumPy, SciPy, Matplotlib, pytest
# CasADi + IPOPT are optional; without them the FEC solver uses a SciPy projection fallback.
```

## Test (44 tests: 33 engine + 11 case-study)

```bash
PYTHONPATH=. pytest -q tests/ case_studies/
```

## Reproduce the paper's data figures and numbers

```bash
PYTHONPATH=. python3 paper/make_softwarex_figs.py
```

This regenerates the three data figures (toy mechanism; Sel'kov + Wolf–Heinrich oscillators;
the negative result) and prints the numbers used in the paper. Full mapping in
[`paper/reproduce.md`](paper/reproduce.md).

## Structure

| Path | Role |
|------|------|
| `bmac_engine/rop_polyhedron.py` | ROP H-representation, projection, log-derivative matrix |
| `bmac_engine/fec_solver.py` | ROP-constrained FEC optimal-control problem (CasADi/IPOPT + fallback) |
| `bmac_engine/robust_extension.py` | interval ROP + sampled scenario robust FEC |
| `bmac_engine/multicell_agent.py` | consortium loop with a population resource policy |
| `bmac_engine/order_uncertainty.py` | data-consistent order polytopes (the falsification) |
| `bmac_engine/bos_integration.py` | narrow interfaces to the reference components |
| `bos_platform/` | in-tree reference stub (SignalAPI, Kalman, OPA, DigitalTwin, Temporal) |
| `case_studies/` | Sel'kov & Wolf–Heinrich validated re-implementations |
| `paper/` | the SoftwareX manuscript (`softwarex.tex`) + figure generators |
| `tests/` | 44 automated tests |

## Citation

If you use `ropfec`, please cite the software paper (Paper B) and the companion empirical
paper (Paper A):

- L. Zhou. *ropfec: a tested, reproducible implementation of reaction-order-constrained
  flux-exponent control for staged bioconversion, with a built-in falsification of
  structural priors.* (SoftwareX, under review.)
- L. Zhou et al. *Decoupling waste deconstruction from nutrient recovery: a staged insect
  Biological Operating System for circular valorization of agro-industrial waste.*
  (J. Cleaner Production, under review.)

This work implements and builds upon the reaction order polytope and flux exponent control
framework of Fangzhou Xiao, Jing Shuang Li, and John C. Doyle.

## License

MIT — see [`LICENSE`](LICENSE).
