Dear Editors,

We submit "ropfec: reaction-order-polytope flux-exponent control for staged bioconversion" as an Original Software Publication for SoftwareX.

ropfec is an open, tested Python package implementing the reaction-order-polytope (ROP) / flux-exponent control (FEC) framework for power-law (S-system) metabolic models: it constructs the ROP from binding stoichiometry, projects reaction-order vectors onto it, and solves the FEC optimal-control problem (CasADi/IPOPT, with a SciPy fallback). Beyond a reference implementation it encodes two first-class, re-runnable results: (i) a clarifying identity — the reaction-order exponents that FEC regulates are exactly the elasticity coefficients of metabolic control analysis; and (ii) a built-in falsification — tested against a count-matched random-facet null, the binding-derived ROP does not materially improve data-driven estimation of reaction orders. We report this negative result honestly rather than overclaiming. Validation is in-silico on the Sel'kov and Wolf–Heinrich glycolytic oscillators; the package ships 44 automated tests and one-command regeneration of every figure.

The software is released under the MIT license at https://github.com/exergyleizhou-ux/ropfec and permanently archived on Zenodo (https://doi.org/10.5281/zenodo.20805373, v1.0.1). The work has not been published previously and is not under consideration elsewhere; the author declares no competing interests and discloses AI-assisted drafting and coding, for which the author takes full responsibility.

We would be grateful if you would consider the following as potential reviewers, with expertise in control theory for biological systems and metabolic modelling: (1) Prof. Brian Ingalls, University of Waterloo, Canada — bingalls@uwaterloo.ca; (2) Prof. Vassily Hatzimanikatis, EPFL, Switzerland — vassily.hatzimanikatis@epfl.ch; (3) Dr. Steffen Waldherr, University of Vienna, Austria — steffen.waldherr@univie.ac.at. (The framework's originator, Prof. Fangzhou Xiao, Westlake University, is a domain expert the editor may also wish to consider, though as the method's inventor he may be considered close to the work.)

Thank you for your consideration.

Sincerely,
Lei Zhou
Zhejiang A&F University, Hangzhou, China — exergyleizhou@gmail.com
ORCID 0009-0000-9073-1349
