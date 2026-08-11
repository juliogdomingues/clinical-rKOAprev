"""Tests for calibration, threshold metrics, and decision-curve analysis.

Synthetic data with analytically known answers, so these verify the maths
rather than just that the code runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from koa_screening.clinical_utility import (
    calibration_curve_points,
    calibration_slope_intercept,
    calibration_with_ci,
    decision_curve,
    threshold_metrics,
    youden_threshold,
)


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    n = 4000
    groups = np.repeat(np.arange(n // 2), 2)  # 2 knees per participant
    lp = rng.normal(-2.0, 1.2, size=n)        # true linear predictor
    p = 1 / (1 + np.exp(-lp))
    y = rng.binomial(1, p)
    return y, p, lp, groups


def test_perfect_predictions_give_slope_1_and_citl_0(synth):
    y, p, _, _ = synth
    slope, citl = calibration_slope_intercept(y, p)
    assert abs(slope - 1.0) < 0.12, f"slope {slope} should be ~1 for calibrated predictions"
    assert abs(citl) < 0.12, f"CITL {citl} should be ~0 for calibrated predictions"


def test_overconfident_predictions_give_slope_below_1(synth):
    y, _, lp, _ = synth
    p_extreme = 1 / (1 + np.exp(-lp * 1.6))  # too extreme
    slope, _ = calibration_slope_intercept(y, p_extreme)
    assert slope < 0.85, f"slope {slope} should be <1 when predictions are too extreme"


def test_inflated_risk_gives_negative_citl(synth):
    y, _, lp, _ = synth
    p_high = 1 / (1 + np.exp(-(lp + 0.8)))  # systematically over-predicts risk
    _, citl = calibration_slope_intercept(y, p_high)
    assert citl < -0.3, f"CITL {citl} should be clearly negative when risk is over-predicted"


def test_calibration_ci_brackets_point_estimate(synth):
    y, p, _, g = synth
    res = calibration_with_ci(y, p, g, n_boot=150)
    assert res["slope_ci_low"] <= res["calibration_slope"] <= res["slope_ci_high"]
    assert res["citl_ci_low"] <= res["calibration_in_the_large"] <= res["citl_ci_high"]
    assert res["n_boot_used"] > 100


def test_calibration_curve_points_are_monotone_and_bounded(synth):
    y, p, _, _ = synth
    cp = calibration_curve_points(y, p, n_bins=10)
    assert len(cp) == 10
    assert cp["mean_predicted"].is_monotonic_increasing
    assert ((cp["observed"] >= 0) & (cp["observed"] <= 1)).all()
    assert (cp["obs_ci_low"] <= cp["observed"]).all() and (cp["observed"] <= cp["obs_ci_high"]).all()


def test_decision_curve_reference_strategies(synth):
    """Textbook properties: treat-none NB == 0 always; treat-all NB == 0 when the
    threshold equals prevalence; and treat-all NB == prevalence at threshold ~0."""
    y, p, _, _ = synth
    prev = float(y.mean())
    dc = decision_curve(y, p, [0.001, round(prev, 3), 0.4])
    assert (dc["net_benefit_treat_none"] == 0).all()
    at_prev = dc[np.isclose(dc["threshold"], round(prev, 3))].iloc[0]
    assert abs(at_prev["net_benefit_treat_all"]) < 0.01, "treat-all NB should vanish at pt=prevalence"
    near_zero = dc.iloc[0]
    assert abs(near_zero["net_benefit_treat_all"] - prev) < 0.01


def test_decision_curve_model_beats_defaults_somewhere(synth):
    y, p, _, _ = synth
    dc = decision_curve(y, p)
    assert (dc["net_benefit_gain_vs_best_default"] > 0).any(), "a good model should beat treat-all/none somewhere"


def test_threshold_metrics_bounds_and_monotonicity(synth):
    y, p, _, g = synth
    tm = threshold_metrics(y, p, g, [0.05, 0.2, 0.5], n_boot=100)
    for c in ["sensitivity", "specificity", "ppv", "npv"]:
        assert tm[c].between(0, 1).all()
    # raising the threshold flags fewer, lowering sensitivity and raising specificity
    assert tm["sensitivity"].is_monotonic_decreasing
    assert tm["specificity"].is_monotonic_increasing
    assert tm["n_flagged"].is_monotonic_decreasing
    assert (tm["sens_ci_low"] <= tm["sensitivity"]).all() and (tm["sensitivity"] <= tm["sens_ci_high"]).all()


def test_youden_threshold_is_a_probability(synth):
    y, p, _, _ = synth
    thr = youden_threshold(y, p)
    assert 0.0 <= thr <= 1.0
