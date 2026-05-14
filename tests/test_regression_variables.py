"""Regression tests that lock the variable inventory.

These assert that the new layout produces exactly the same column set as the
original code did, at every stage: full post-prep DataFrame, per-scenario
feature lists, and the set of columns dropped by the 50%-missingness filter.

This is the primary safeguard against future edits silently dropping any
input variable.

Marked ``requires_data`` because they need the real ELSA CSV at
``data/raw/stataToCsvMG.csv``. Skipped automatically when absent (see
``conftest.py``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from koa_screening import data
from koa_screening.config import (
    BASE_EXCLUDE,
    MISSING_COL_THRESHOLD,
    RAW_CSV,
    SYMPTOM_VARS,
)

FIX = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.requires_data


def _read_fixture_lines(name: str) -> set[str]:
    return {line.strip() for line in (FIX / name).read_text(encoding="utf-8").splitlines() if line.strip()}


@pytest.fixture(scope="module")
def prepped(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("regression_audit")
    df = data.load_and_prep_data(str(RAW_CSV), outdir=str(outdir))
    return df.sort_values("idelsa").reset_index(drop=True)


def test_post_prep_columns_match_fixture(prepped):
    """The columns returned by ``load_and_prep_data`` are exactly the frozen
    set. Any column added, renamed, or removed will fail this test."""
    actual = set(prepped.columns)
    expected = _read_fixture_lines("expected_columns_post_prep.txt")
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"variables LOST in post-prep: {sorted(missing)}"
    assert not extra, f"unexpected new variables in post-prep: {sorted(extra)}"


def test_scenario_without_symptoms_features(prepped):
    """The Screening scenario uses exactly the locked feature set."""
    all_cols = [c for c in prepped.columns if c not in BASE_EXCLUDE]
    actual = {c for c in all_cols if c not in SYMPTOM_VARS}
    expected = _read_fixture_lines("expected_columns_scenario_without.txt")
    assert actual == expected, (
        f"Screening features changed. "
        f"lost={sorted(expected - actual)} added={sorted(actual - expected)}"
    )


def test_scenario_with_symptoms_features(prepped):
    """The Case Finding scenario uses exactly the locked feature set."""
    actual = {c for c in prepped.columns if c not in BASE_EXCLUDE}
    expected = _read_fixture_lines("expected_columns_scenario_with.txt")
    assert actual == expected, (
        f"With-Symptoms features changed. "
        f"lost={sorted(expected - actual)} added={sorted(actual - expected)}"
    )


def test_scenario_virtual_max_features(prepped):
    """The Virtual Maximum scenario uses exactly the locked feature set."""
    actual = {c for c in prepped.columns if c not in BASE_EXCLUDE}
    expected = _read_fixture_lines("expected_columns_scenario_virtual.txt")
    assert actual == expected


def test_no_silent_drops_from_missingness_filter(prepped):
    """The 50%-missingness filter drops exactly the locked set of columns
    (today: zero). Catches the case where a future change introduces a
    column with >50% missing and that column silently disappears."""
    all_cols = [c for c in prepped.columns if c not in BASE_EXCLUDE]
    X_full = prepped[all_cols].copy()
    thresh = int(np.ceil(len(prepped) * MISSING_COL_THRESHOLD))
    X_after = X_full.dropna(axis=1, thresh=thresh)
    actual_dropped = set(X_full.columns) - set(X_after.columns)
    expected_dropped = _read_fixture_lines("expected_dropped_high_missing.txt")
    assert actual_dropped == expected_dropped, (
        f"missingness filter drop-set changed. "
        f"lost={sorted(expected_dropped - actual_dropped)} "
        f"newly_dropped={sorted(actual_dropped - expected_dropped)}"
    )


def test_sample_counts_match_metadata(prepped):
    """Row, participant, and prevalence counts match the frozen metadata."""
    import json

    meta = json.loads((FIX / "fixture_metadata.json").read_text(encoding="utf-8"))
    assert len(prepped) == meta["n_rows_post_prep"]
    assert prepped["idelsa"].nunique() == meta["n_participants_post_prep"]
    assert prepped.shape[1] == meta["n_cols_post_prep"]
    assert abs(prepped["oa_knee"].mean() - meta["prevalence_oa_knee"]) < 1e-12
