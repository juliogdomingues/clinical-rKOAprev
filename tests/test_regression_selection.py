"""Regression test: the feature-selection step reproduces its core outputs.

``scripts/02_feature_selection.py`` (via ``features.run_analysis``) must
regenerate, byte-for-byte at seed 42, the three intermediate files the rest
of the pipeline consumes:

  - stepwise_mpms_clinical.csv      (MPMS stepwise order -> Table 1, AUC summary)
  - mpms_features_for_ci.csv        (MPMS clinical feature set)
  - final_5var_features_for_ci.csv  (top-5 final model)

This guards the reproducibility wiring: if a future change perturbs the
LASSO/MPMS/stepwise selection, the downstream manuscript numbers would
silently drift. Catching it here is cheaper than discovering it in the
AUC/OR regression tests.

Marked ``requires_data`` (needs the real CSV); runs the selection once
(~1 min) into a temp dir and diffs against the frozen fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from koa_screening import data, features
from koa_screening.config import RAW_CSV

FIX = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.requires_data

CORE_FILES = {
    "stepwise_mpms_clinical.csv": "expected_stepwise_mpms_clinical.csv",
    "mpms_features_for_ci.csv": "expected_mpms_features_for_ci.csv",
    "final_5var_features_for_ci.csv": "expected_final_5var_features_for_ci.csv",
}


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> Path:
    outdir = tmp_path_factory.mktemp("selection")
    df = data.load_and_prep_data(str(RAW_CSV), outdir=str(outdir))
    features.run_analysis(df, outdir=str(outdir))
    return outdir


@pytest.mark.parametrize("produced,fixture", list(CORE_FILES.items()))
def test_selection_file_matches_fixture(regenerated, produced, fixture):
    actual = pd.read_csv(regenerated / produced)
    expected = pd.read_csv(FIX / fixture)
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
        obj=produced,
    )


def test_final5_is_top5_of_stepwise(regenerated):
    """The final-5 list must equal the first 5 rows of the stepwise order.
    This invariant is what makes the two files mutually consistent."""
    stepwise = pd.read_csv(regenerated / "stepwise_mpms_clinical.csv")
    final5 = pd.read_csv(regenerated / "final_5var_features_for_ci.csv")
    assert final5["feature"].tolist() == stepwise["Variable"].tolist()[:5]
