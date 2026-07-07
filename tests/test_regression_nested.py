"""Structural regression checks for the nested-CV outputs (scripts/12).

Nested-CV AUCs are deterministic at seed 42 but locking exact values would be
brittle across sklearn/xgboost versions, so this checks structure and sane
invariants rather than exact numbers:
  - every scenario/model row is present, CIs ordered, AUC in a plausible band,
  - the paired-difference file is well-formed (p in [0,1], CI brackets the point,
    the LR-minus-ML difference matches the two summary AUCs),
  - per-fold feature lists and best-hyperparameter records exist.

Marked ``requires_data`` (only exists after scripts/12 has run); skipped otherwise.
"""
from __future__ import annotations

import pytest

from koa_screening.config import RESULTS_COMPARISON

pytestmark = pytest.mark.requires_data

SUMMARY = RESULTS_COMPARISON / "nested_cv_summary.csv"
PAIRED = RESULTS_COMPARISON / "nested_cv_paired_diff.csv"
LR_FEATS = RESULTS_COMPARISON / "nested_cv_lr_fold_features.csv"
ML_PARAMS = RESULTS_COMPARISON / "nested_cv_ml_fold_params.csv"


def _load(path):
    import pandas as pd

    if not path.exists():
        pytest.skip(f"{path} not generated yet — run scripts/12_nested_cv.py first")
    return pd.read_csv(path)


def test_summary_structure_and_bounds():
    s = _load(SUMMARY)
    got = set(zip(s["Scenario"], s["Model"]))
    for scen in ("Without Symptoms", "With Symptoms"):
        for m in ("Stepwise LR", "XGBoost", "Random Forest", "Neural Network"):
            assert (scen, m) in got, f"missing nested row {scen}/{m}"
    # Virtual Maximum is ML-only by design (LR skipped)
    assert ("Virtual Maximum", "Stepwise LR") not in got
    assert (s["AUC_CI_Low"] <= s["AUC"]).all() and (s["AUC"] <= s["AUC_CI_High"]).all()
    assert (s["Brier_CI_Low"] <= s["Brier"]).all() and (s["Brier"] <= s["Brier_CI_High"]).all()
    assert s["AUC"].between(0.5, 1.0).all(), "a nested AUC fell below 0.5"


def test_paired_diff_wellformed():
    import pandas as pd

    p = _load(PAIRED)
    s = _load(SUMMARY).set_index(["Scenario", "Model"])["AUC"]
    assert p["p_value"].between(0.0, 1.0).all()
    assert (p["CI_Low"] <= p["delta_AUC"]).all() and (p["delta_AUC"] <= p["CI_High"]).all()
    assert (p["p_value"] > 0).all(), "add-one smoothing should keep p strictly > 0"
    # the reported delta equals AUC(LR) - AUC(ML) from the summary
    for _, r in p.iterrows():
        ml = r["Comparison"].replace("Stepwise LR - ", "")
        expected = s[(r["Scenario"], "Stepwise LR")] - s[(r["Scenario"], ml)]
        assert abs(expected - r["delta_AUC"]) < 1e-9


def test_transparency_records_exist():
    lr = _load(LR_FEATS)
    ml = _load(ML_PARAMS)
    assert lr["outer_fold"].nunique() == 5 and lr["n_features"].min() >= 1
    assert ml["outer_fold"].nunique() == 5 and ml["best_params"].str.len().min() > 2
