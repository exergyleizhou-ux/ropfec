# Submission-readiness design — ROP/FEC reproducible-implementation paper

**Date:** 2026-06-21
**Author (paper):** Lei Zhou (Zhejiang A&F University), single author + AI-assistance disclosure
**Status:** design approved in substance (positioning, venue, name, authorship); pending user review of this spec
**Working tree:** `~/Desktop/bos/bos-bmac` (now under local git; baseline `678220e`)

---

## 1. Goal

Take the rough "BMAC" draft (`~/bos-rop-fec.tex`, authored placeholder "Grok 4.3",
results marked "(simulated)") to a **solidly submission-ready methods/software paper**
plus a **preprint**, positioned honestly around what the work actually is.

Target (user-approved): **computational-methods / open-source-software journal**
(e.g. PLOS Computational Biology — Methods/Software; or JOSS for the software +
GigaScience-style), **with a bioRxiv/arXiv preprint first**.

## 2. Honest positioning

The work is **not new theory**. Xiao's ROP (Reaction Order Polyhedron) and FEC
(Flux Exponent Control) are the foundation (verified real — see §6). The honest,
defensible contribution is an **open, reproducible implementation + closed-loop
control/governance integration + in-silico validation** of that framework.

This framing is what makes it "稳稳" — matching the claim to the evidence avoids
desk-rejection on novelty.

## 3. Naming (RESOLVED)

- **Drop "BMAC"** — it collides with Xiao's own **B**iomachine **A**rchitecture &
  **C**ontrol (BMAC) lab at Westlake University. Building on his work *and* taking his
  lab's acronym is unacceptable.
- **Paper title (descriptive):** *"An open, reproducible implementation of
  reaction-order-polyhedron–constrained flux exponent control for closed-loop
  metabolic network control"* (final wording tunable).
- **Software package:** `ropfec`.

## 4. Authorship & disclosure (RESOLVED)

- **Author:** Lei Zhou, Zhejiang A&F University (corresponding).
- **AI-assistance disclosure** in a Declaration section: drafting was assisted by AI
  tools (xAI Grok; Anthropic Claude); the author verified and is responsible for all
  content. AI is **not** an author.

## 5. Contributions (each gated on a novelty check vs Xiao's full corpus)

- **C1 — Reproducible reference implementation.** Open ROP H-rep construction +
  ROP-constrained FEC OCP (CasADi/IPOPT); 29 passing tests; deterministic seeds.
- **C2 — Closed-loop integration.** FEC wrapped in sense→estimate→optimize→actuate→
  govern loop via clean interfaces (Signal/Control, Kalman, Monte Carlo, OPA policy,
  Temporal durability), shipped with **reference (stub) implementations** of those
  interfaces. *(Honesty: results use the self-contained reference impl, not a live
  production platform — do not overclaim a deployed "BOS platform".)*
- **C3 — Robust/interval ROP + multicellular agent extensions.** Claimed novel **only
  after** confirming Xiao didn't already publish them.
- **C4 — In-silico validation** (real, reproducible — see §6).

## 6. Verified foundations & real results (the evidence base)

**Citations verified real (2026-06-21 web check):**
- FEC: F. Xiao, J. S. Li, J. C. Doyle, "Flux exponent control predicts metabolic
  dynamics from network structure," *2023 American Control Conference (ACC)*,
  pp. 1189–1194. bioRxiv 10.1101/2023.03.23.533708; IEEE Xplore 10156281.
- ROP: F. Xiao, PhD thesis, Caltech, 2022 — **correct title** =
  *"Biocontrol of Biomolecular Systems: Polyhedral Constraints on Binding's Regulation
  of Catalysis from Biocircuits to Metabolism."* (draft's title was wrong — FIX.)
- TODO during writing: complete Xiao's full publication list for the reference section
  (CV PDF couldn't be parsed locally; fetch via Caltech feeds / Westlake / Scholar).

**Real numerics reproduced locally (`numerical_toy_validation.py`, 29 tests green):**
- ROP-constrained FEC reduces trajectory prediction error **~3.2×** vs unconstrained
  (rel. err **3.1%** vs **9.9%**; peak overshoot −2.2% → matched).
- FBA→MM→ROP-constrained error: **1.2832 → 1.0899 → 1.0113** (`fba_mm_error_reduction.csv`).
- FEC solver (CasADi/IPOPT) finds feasible in-ROP α = [0.6, 0.84, 1.62], sim err vs
  ground truth **0.0054**.
- Robustness: nominal-worst 1.0082 vs scenario-robust-worst 1.0113 (`scenario_mpc_cost.csv`).
- Figures already generated: `toy_traj_comparison`, `fba_mm_error_reduction`,
  `phase0_traj_comparison`, `mc_cost_dist`, `mc_dt_cost_with_mean`, `sensitivity_L_cost`,
  `scenario_mpc_cost`, `correspondence_*`, `phase3_quant_demo`.

## 7. Manuscript structure (methods/software journal)

1. Title · Author · Abstract (~250 w) · (optional Author Summary)
2. **Introduction** — motivation; Xiao's ROP/FEC; gap (no open reproducible impl, no
   closed-loop integration); explicit contributions list.
3. **Background** — ROP, FEC, CRN dynamics; faithfully attributed to Xiao.
4. **Methods / Implementation** — ROP construction · FEC OCP · closed-loop integration
   (with honest stub caveat) · robust/interval ROP · multicellular agent · software
   architecture + reproducibility.
5. **Results (in-silico)** — error reduction vs unconstrained/FBA/MM (table + fig) ·
   feasible in-ROP tracking · robustness under uncertainty · log-derivative sensitivity ·
   reproducibility (tests, seeds).
6. **Discussion** — relation to FBA/MM and to Xiao; what's novel here; **limitations:
   in-silico & single toy network, no wet-lab, stub platform**; future work.
7. **Outlook** (brief) — embodied biomachines, kept short and clearly labeled
   speculative/illustrative (numbers not fitted to data).
8. Conclusion · Code & Data Availability (GitHub + Zenodo DOI pending) · Declaration
   (AI assistance, competing interests, funding) · Acknowledgments · References.

## 8. Integrity fixes (checklist)

- [ ] Rename away from "BMAC"; software → `ropfec`.
- [ ] Real author (Lei Zhou) + AI-assistance disclosure.
- [ ] Fix Xiao thesis title; add ACC pages/DOI; complete Xiao reference list.
- [ ] Replace every "(simulated)" / "would be in supplementary" with real engine output.
- [ ] Honest "reference stub" framing for platform integration (no production overclaim).
- [ ] Explicit Xiao-vs-novel delineation throughout.
- [ ] Honest limitations section (in-silico/toy/no wet-lab).
- [ ] Keep speculative embodied content brief + clearly labeled.

## 9. Execution plan (phases)

- **A — Lock evidence.** Re-run engine; freeze the exact numbers + figures the paper
  cites; curate `examples/figures` into a `paper/figures/` set; write a `reproduce.md`.
- **B — Skeleton.** New LaTeX manuscript in `paper/` (rename, author, structure §7),
  drawing the good math/pseudocode from `~/bos-rop-fec.tex`.
- **C — Write from data.** Methods + Results + Discussion grounded in real §6 numbers;
  honest caveats.
- **D — Apparatus.** References (verified Xiao list + FBA/MM/CRN classics), declarations,
  Code/Data availability, reproducibility appendix.
- **E — Polish & compile.** Abstract, language pass, compile to PDF, optional cover
  letter + preprint checklist (bioRxiv/arXiv categories).

## 10. Reproducibility

- Python 3.9, numpy/scipy/casadi/matplotlib (all present). Deterministic seeds.
- `ropfec` package + 29 tests + `reproduce.md` (one command regenerates every figure/number).
- Zenodo archive DOI minted from a tagged GitHub release before submission.

## 11. Out of scope (v1, honest)

- Wet-lab validation (stated as future work).
- Live integration with the production bos-platform (interfaces shown; results use
  reference stubs).
- The full embodied-biomachine swarm study (Outlook only).

## 12. Parallel track — JCP Paper 1

Separate paper (insect bioconversion, J. Clean. Prod). Manuscript `.docx` lives only on
the user's Windows desktop — **not reachable from this Mac**. Blocked until the user
copies the file here or pastes the text. Repo evidence layer (`~/Projects/bos-platform`)
is already at `v0.9.1-paper-final`.
