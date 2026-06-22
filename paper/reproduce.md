# Reproducing the `ropfec` SoftwareX paper (`softwarex.tex`)

Every number and data figure in `softwarex.tex` is regenerated from the engine.
No value is hand-entered. License: MIT (see top-level `LICENSE`).

## Environment

- Python 3.9+ with `numpy`, `scipy`, `matplotlib`, `pytest` (see `requirements.txt`).
- `casadi` + IPOPT are optional; without them the FEC solver uses a SciPy projection
  fallback (the toy and negative-result numbers do not require IPOPT).

```
pip install -e .          # or: pip install -r requirements.txt
```

## Tests (44 total: 33 engine + 11 case-study)

```
PYTHONPATH=. pytest -q tests/ case_studies/
```

## Data figures and numbers

One command regenerates the three **data** figures and prints the numbers used in
the paper (Figure 1, the architecture, is a TikZ schematic compiled with the
manuscript, not a data figure):

```
PYTHONPATH=. python3 paper/make_softwarex_figs.py
```

| Manuscript item | Source | Value(s) |
|---|---|---|
| Figure 1 (architecture) | `figures/ropfec_architecture.pdf` (TikZ) | schematic |
| Figure 2 (toy mechanism) | `make_softwarex_figs.py::fig_toy` | 9.9% (out-of-ROP) vs 3.1% (ROP) |
| Example 1 tracking costs | `examples/correspondence_verification.py` | FEC 1.01; FBA 1.28; MM 1.09 |
| FEC round-trip exponent | `examples/numerical_toy_validation.py` | within ~2% of the prescribed orders |
| Figure 3 (oscillators) | `make_softwarex_figs.py::fig_osc` | Sel'kov + Wolf--Heinrich dynamics |
| Figure 4 (negative result) | `make_softwarex_figs.py::fig_neg` | informative 0.04/0.04/0.04; data-poor 2.72/2.31/2.34 |

## Build the manuscript

```
tectonic paper/softwarex.tex
```
