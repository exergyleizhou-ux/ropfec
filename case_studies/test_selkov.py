"""Qualitative-dynamics tests for the Sel'kov glycolytic oscillator.

These tests assert the KNOWN qualitative behaviour without brittle exact-value
matching:

  * With default parameters (a=0.06, b=0.6), inside the Hopf window, the system
    settles onto a STABLE LIMIT CYCLE -- after discarding the transient, the
    peak-to-peak amplitude of x (ADP) stays well above a clear threshold and does
    NOT decay (sustained oscillation).

  * With the contrast set (a=0.1, b=1.0), the system spirals into a STABLE FIXED
    POINT -- the late-time amplitude collapses toward zero.

  * The analytic fixed point and Hopf trace condition match the implementation.

Runtime is a few seconds total.
"""

from __future__ import annotations

import numpy as np

from selkov import (
    PARAMS,
    PARAMS_FIXED_POINT,
    X0,
    dxdt,
    fixed_point,
    simulate,
)


def _late_peak_to_peak(t, X, var=0, frac=0.5):
    """Peak-to-peak amplitude of column `var` over the last `1-frac` of the run."""
    mask = t >= (t[0] + frac * (t[-1] - t[0]))
    series = X[mask, var]
    return float(series.max() - series.min())


def _jacobian_trace(params):
    """trace(J) at the fixed point; >0 implies the Hopf/limit-cycle regime."""
    a, b = params["a"], params["b"]
    xs, ys = fixed_point(params)
    j11 = -1.0 + 2.0 * xs * ys
    j22 = -(a + xs * xs)
    return j11 + j22


def test_default_params_sustained_limit_cycle():
    """Default (a=0.06, b=0.6): sustained oscillation, large persistent amplitude."""
    t, X = simulate(params=PARAMS, x0=X0, t_span=(0.0, 400.0), n_points=4000)

    # After discarding the initial transient, amplitude must be clearly nonzero.
    ptp_late = _late_peak_to_peak(t, X, var=0, frac=0.5)
    assert ptp_late > 0.5, f"expected sustained oscillation, got ptp={ptp_late:.4f}"


def test_oscillation_does_not_decay():
    """Amplitude in a late window is comparable to a mid window (no decay to a point)."""
    t, X = simulate(params=PARAMS, x0=X0, t_span=(0.0, 400.0), n_points=4000)

    # Mid window (after transient) vs final window: a true limit cycle keeps a
    # roughly constant amplitude; a damped spiral would shrink markedly.
    mid_mask = (t >= 200.0) & (t < 300.0)
    end_mask = t >= 300.0
    ptp_mid = float(X[mid_mask, 0].max() - X[mid_mask, 0].min())
    ptp_end = float(X[end_mask, 0].max() - X[end_mask, 0].min())

    assert ptp_mid > 0.5 and ptp_end > 0.5
    # Final amplitude must not have collapsed relative to the mid window.
    assert ptp_end > 0.8 * ptp_mid, (
        f"amplitude decayed: mid={ptp_mid:.4f} end={ptp_end:.4f}"
    )


def test_contrast_params_decay_to_fixed_point():
    """Contrast (a=0.1, b=1.0): trajectory spirals into the fixed point (no LC)."""
    t, X = simulate(params=PARAMS_FIXED_POINT, x0=X0, t_span=(0.0, 400.0), n_points=4000)

    ptp_late = _late_peak_to_peak(t, X, var=0, frac=0.75)
    assert ptp_late < 0.02, f"expected decay to fixed point, got ptp={ptp_late:.4f}"

    # Late-time state should sit at the analytic fixed point.
    xs = fixed_point(PARAMS_FIXED_POINT)
    assert np.allclose(X[-1], xs, atol=1e-2)


def test_hopf_trace_condition_consistency():
    """The analytic Hopf condition matches each regime's observed behaviour."""
    # Oscillatory regime: unstable spiral -> trace(J) > 0.
    assert _jacobian_trace(PARAMS) > 0.0
    # Damped regime: stable spiral -> trace(J) < 0.
    assert _jacobian_trace(PARAMS_FIXED_POINT) < 0.0


def test_rhs_and_fixed_point_are_consistent():
    """dxdt vanishes at the analytic fixed point for both parameter sets."""
    for params in (PARAMS, PARAMS_FIXED_POINT):
        xs = fixed_point(params)
        deriv = dxdt(0.0, xs, params)
        assert np.allclose(deriv, 0.0, atol=1e-12), (
            f"rhs nonzero at fixed point for {params}: {deriv}"
        )
