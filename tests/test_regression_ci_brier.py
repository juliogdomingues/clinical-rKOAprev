"""Regression checks for scripts/10 output (AUC CIs + Brier per model).

Reads the committed ``summary_all_models_ci_brier.csv`` and asserts:
  - its AUC point estimates match the canonical ``summary_all_models.csv``,
  - every CI is ordered (low <= point <= high) for both AUC and Brier,
  - the expected model/scenario rows are all present.

Marked ``requires_data`` because the CSV only exists after the pipeline runs;
skipped otherwise (see conftest). Does not re-run anything.
"""
from __future__ import annotations

import pytest

from koa_screening.config import RESULTS_COMPARISON

pytestmark = pytest.mark.requires_data

CI = RESULTS_COMPARISON / "summary_all_models_ci_brier.csv"
CANON = RESULTS_COMPARISON / "summary_all_models.csv"


def _load():
    import pandas as pd

    if not CI.exists():
        pytest.skip(f"{CI} not generated yet — run scripts/10_model_ci_brier.py first")
    return pd.read_csv(CI)


def test_auc_points_match_canonical():
    import pandas as pd

    ci = _load()
    canon = pd.read_csv(CANON)
    merged = canon.merge(ci[["Scenario", "Model", "AUC"]], on=["Scenario", "Model"], suffixes=("_c", "_ci"))
    assert len(merged) == len(canon), "ci_brier is missing canonical Scenario/Model rows"
    assert (merged["AUC_c"] - merged["AUC_ci"]).abs().max() < 1e-9


def test_confidence_intervals_are_ordered():
    ci = _load()
    assert (ci["AUC_CI_Low"] <= ci["AUC"]).all() and (ci["AUC"] <= ci["AUC_CI_High"]).all()
    assert (ci["Brier_CI_Low"] <= ci["Brier"]).all() and (ci["Brier"] <= ci["Brier_CI_High"]).all()


def test_all_models_present():
    ci = _load()
    got = set(zip(ci["Scenario"], ci["Model"]))
    for scen in ("Without Symptoms", "With Symptoms"):
        for model in ("Stepwise (Full)", "XGBoost", "Random Forest", "Neural Network"):
            assert (scen, model) in got, f"missing {scen} / {model}"
    # Virtual Maximum has no Stepwise row by design
    assert ("Virtual Maximum", "XGBoost") in got
    assert ("Virtual Maximum", "Stepwise (Full)") not in got
