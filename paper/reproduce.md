# Reproducing the `ropfec` paper

Every number and figure in `ropfec.tex` is regenerated from the engine. No value
is hand-entered.

## Environment

- Python 3.9 with `numpy`, `scipy`, `casadi`, `matplotlib` (see `requirements.txt`).
- Verified on macOS (Apple Silicon), 2026-06-21.

## One command (tests + all figures/numbers)

```bash
cd ..              # repo root (bos-bmac)
PYTHONPATH=. python3 examples/run_all.py
```

This runs the 29-test suite and every example, writing figures to
`examples/figures/` (copied into `paper/figures/` for the manuscript).

## Mapping: paper claim -> source

| Paper claim | Value | Source |
|---|---|---|
| ROP enforce vs violate, trajectory error | 3.1% vs 9.9% (~3.2x) | `examples/numerical_toy_validation.py` |
| FEC solver admissible alpha, final-state err | [0.6,0.84,1.62], 0.0054 | `examples/numerical_toy_validation.py` |
| FBA/MM/FEC/scenario final-state cost (50 runs) | 1.283 / 1.090 / 1.011 / 1.011 | `bmac_engine/benchmarks.py::run_fba_mm_benchmark` -> `paper/figures/fba_mm_error_reduction.csv` |
| Robust worst-case nominal vs scenario | 1.008 vs 1.011 | `paper/figures/scenario_mpc_cost.csv` |
| End-to-end invariants all ok | `ok: True` | `examples/end_to_end_toy.py` / `correspondence_verification.py` |
| 29 tests pass | 29 passed | `pytest -q` |
| Multicell OPA block rate | 100% over-capacity | `correspondence_verification.py` |

## Figures used by the manuscript

`paper/figures/`: `toy_traj_comparison.png` (Fig 2), `fba_mm_error_reduction.png`
(Fig 3), `scenario_mpc_cost.png` (Fig 4), `sensitivity_L_cost.png` (Fig 5).
Figure 1 is a TikZ schematic compiled from `ropfec.tex`.

## Build the PDF

```bash
cd paper
tectonic ropfec.tex     # XeTeX engine; runs bibtex automatically
# or:  latexmk -pdf ropfec.tex   (with a full TeX install)
```
