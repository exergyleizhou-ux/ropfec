"""Qualitative-dynamics tests for the single-cell Wolf-Heinrich (2000) oscillator.

These tests assert the KNOWN qualitative behaviour -- a stable limit cycle
(sustained glycolytic oscillations) -- WITHOUT matching exact trajectory
values. The core check: after discarding an initial transient, the key
oscillating variable keeps a clear peak-to-peak amplitude over a long window
(i.e. it does NOT decay to a fixed point).
"""

import numpy as np

from case_studies.wolf_heinrich import (
    STATE_NAMES,
    default_params,
    default_x0,
    dxdt,
    simulate,
)


def _peak_to_peak(series):
    return float(np.max(series) - np.min(series))


def test_sustained_oscillation_no_decay():
    """A3 (ATP) sustains a large limit-cycle amplitude; it does not decay.

    Strategy: integrate long, split the post-transient window into an EARLY
    and a LATE half. Both halves must show a large peak-to-peak amplitude, and
    the late amplitude must not have collapsed relative to the early one. A
    decaying spiral toward a stable focus would fail the late-window check.
    """
    t, X = simulate(t_end=120.0, n_points=12000)
    a3 = X[STATE_NAMES.index("A3")]

    # Discard the first third as transient.
    transient = len(t) // 3
    post = a3[transient:]

    mid = len(post) // 2
    early = post[:mid]
    late = post[mid:]

    p2p_early = _peak_to_peak(early)
    p2p_late = _peak_to_peak(late)

    # Sustained, large-amplitude oscillation (not a fixed point).
    assert p2p_late > 1.0, f"A3 amplitude collapsed: late p2p={p2p_late:.4f}"

    # No appreciable decay between early and late windows (allow numerical slack).
    assert p2p_late > 0.8 * p2p_early, (
        f"A3 oscillation is decaying: early p2p={p2p_early:.4f}, "
        f"late p2p={p2p_late:.4f}"
    )


def test_multiple_variables_oscillate():
    """Several glycolytic intermediates oscillate with clear amplitude."""
    t, X = simulate(t_end=120.0, n_points=12000)
    transient = len(t) // 3

    thresholds = {"S2": 0.5, "A3": 1.0, "N2": 0.02}
    for name, thresh in thresholds.items():
        series = X[STATE_NAMES.index(name)][transient:]
        p2p = _peak_to_peak(series)
        assert p2p > thresh, f"{name} not oscillating: p2p={p2p:.4f} <= {thresh}"


def test_multiple_extrema_indicate_periodicity():
    """The limit cycle produces many oscillation cycles, not a single hump.

    Count interior local maxima of A3 after the transient. A sustained
    oscillation over a long window yields many peaks; a decaying or monotone
    approach to a fixed point would yield few or none.
    """
    t, X = simulate(t_end=120.0, n_points=12000)
    transient = len(t) // 3
    a3 = X[STATE_NAMES.index("A3")][transient:]

    # Local maxima with a small prominence relative to the signal range.
    rng = _peak_to_peak(a3)
    prominence = 0.1 * rng
    peaks = 0
    for i in range(1, len(a3) - 1):
        if a3[i] > a3[i - 1] and a3[i] >= a3[i + 1]:
            local_min_left = np.min(a3[max(0, i - 50):i + 1])
            if a3[i] - local_min_left > prominence:
                peaks += 1

    assert peaks >= 5, f"Expected many oscillation peaks, found {peaks}"


def test_solution_stays_bounded_and_physical():
    """Concentrations remain bounded and non-negative on the limit cycle."""
    t, X = simulate(t_end=120.0, n_points=12000)
    transient = len(t) // 3
    post = X[:, transient:]

    # Bounded (no blow-up).
    assert np.all(np.isfinite(post)), "Non-finite values in solution"
    assert np.max(post) < 1e3, "Solution unbounded"

    # Physical: intermediates and cofactors stay (essentially) non-negative.
    assert np.min(post) > -1e-6, f"Negative concentration: min={np.min(post):.3e}"

    # Conserved-moiety partners stay within their pools.
    p = default_params()
    n2 = X[STATE_NAMES.index("N2")][transient:]
    a3 = X[STATE_NAMES.index("A3")][transient:]
    assert np.all(n2 <= p["N"] + 1e-6), "NADH exceeds total NAD pool"
    assert np.all(a3 <= p["A"] + 1e-6), "ATP exceeds total adenine pool"


def test_rhs_shape_and_determinism():
    """dxdt returns the right shape and is deterministic (autonomous)."""
    p = default_params()
    x0 = default_x0()
    d1 = dxdt(0.0, x0, p)
    d2 = dxdt(123.4, x0, p)  # autonomous: time must not matter
    assert d1.shape == (len(STATE_NAMES),)
    assert np.allclose(d1, d2)


def test_decoupled_core_is_not_oscillatory():
    """Sanity: with kappa=0 (isolated core, no S4 outflow) there is no
    large sustained limit cycle -- confirms the oscillation is a genuine
    property of the published parameter regime, not an artefact.
    """
    p = default_params()
    p["kappa"] = 0.0
    t, X = simulate(t_end=120.0, n_points=12000, params=p)
    transient = len(t) // 3
    a3 = X[STATE_NAMES.index("A3")][transient:]
    # The isolated core settles toward a focus: amplitude is small.
    assert _peak_to_peak(a3) < 0.5
