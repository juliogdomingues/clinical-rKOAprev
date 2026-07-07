"""Regression tests that lock the manuscript's numerical results.

These compare the on-disk canonical outputs in ``results/`` against the
frozen fixtures captured before the refactor. They are bit-for-bit (within
1e-6 numerical tolerance) so any refactor that subtly shifts an AUC or an OR
will fail loudly.

Marked ``requires_data`` so they only run when the analysis has been
reproduced locally (i.e. ``results/comparison/summary_all_models.csv`` and
``results/final_analysis/final_model_or_raw_ci.csv`` exist).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from koa_screening.config import RESULTS_COMPARISON, RESULTS_FINAL

FIX = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.requires_data


def _require_actual(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path} not generated yet — run scripts/03_run_comparison.py first")
    return pd.read_csv(path)


def test_summary_aucs_match_manuscript():
    """4 models x 3 scenarios AUCs match the frozen baseline to 1e-6."""
    actual = _require_actual(RESULTS_COMPARISON / "summary_all_models.csv")
    expected = pd.read_csv(FIX / "expected_summary_all_models.csv")
    merged = expected.merge(actual, on=["Scenario", "Model"], suffixes=("_exp", "_act"))
    assert len(merged) == len(expected), (
        f"row count mismatch: expected {len(expected)} rows, joined {len(merged)} "
        f"— missing rows: {set(expected['Model'] + '|' + expected['Scenario']) - set(merged['Model'] + '|' + merged['Scenario'])}"
    )
    diffs = (merged["AUC_act"] - merged["AUC_exp"]).abs()
    assert diffs.max() < 1e-6, (
        "AUC drift detected:\n"
        + merged.assign(delta=diffs)[["Scenario", "Model", "AUC_exp", "AUC_act", "delta"]].to_string(index=False)
    )


def test_final_model_or_matches_baseline():
    """5-variable model raw ORs + 95% CIs match within 1e-6."""
    actual_path = RESULTS_FINAL / "final_model_or_raw_ci.csv"
    actual = _require_actual(actual_path)
    expected = pd.read_csv(FIX / "expected_final_model_or.csv")
    merged = expected.merge(actual, on="Feature", suffixes=("_exp", "_act"))
    assert len(merged) == len(expected)
    for col in ["d_OR_Raw", "d_OR_Raw_Low", "d_OR_Raw_High"]:
        delta = (merged[f"{col}_act"] - merged[f"{col}_exp"]).abs()
        assert delta.max() < 1e-6, f"{col} drift: max abs diff = {delta.max()}"


@pytest.mark.parametrize(
    "scenario,fixture",
    [
        ("Without_Symptoms", "expected_or_raw_Without_Symptoms.csv"),
        ("With_Symptoms", "expected_or_raw_With_Symptoms.csv"),
    ],
)
def test_raw_or_per_scenario_matches_baseline(scenario, fixture):
    """Manuscript Table 1 raw ORs match within 1e-6."""
    actual = _require_actual(RESULTS_COMPARISON / f"or_raw_{scenario}.csv")
    expected = pd.read_csv(FIX / fixture)
    merged = expected.merge(actual, on="Feature", suffixes=("_exp", "_act"))
    assert len(merged) == len(expected), f"feature set differs for {scenario}"
    for col in ["OR", "2.5%", "97.5%"]:
        delta = (merged[f"{col}_act"] - merged[f"{col}_exp"]).abs()
        assert delta.max() < 1e-6, f"{scenario} / {col} drift: max abs diff = {delta.max()}"


@pytest.mark.parametrize(
    "scenario,fixture",
    [
        ("Without_Symptoms", "expected_or_standardized_Without_Symptoms.csv"),
        ("With_Symptoms", "expected_or_standardized_With_Symptoms.csv"),
    ],
)
def test_standardized_or_per_scenario_matches_baseline(scenario, fixture):
    """Manuscript Table 1 standardized ORs match within 1e-6."""
    actual = _require_actual(RESULTS_COMPARISON / f"or_standardized_{scenario}.csv")
    expected = pd.read_csv(FIX / fixture)
    merged = expected.merge(actual, on="Feature", suffixes=("_exp", "_act"))
    assert len(merged) == len(expected)
    for col in ["OR", "2.5%", "97.5%"]:
        delta = (merged[f"{col}_act"] - merged[f"{col}_exp"]).abs()
        assert delta.max() < 1e-6, f"{scenario} / {col} drift: max abs diff = {delta.max()}"
