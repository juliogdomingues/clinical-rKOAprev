"""Synthetic-data tests for the sensitivity filters.

Hand-built dataframes with known structure so the assertions speak directly
to the filter semantics. No real data needed; runs anywhere.
"""
from __future__ import annotations

import pandas as pd
import pytest

from koa_screening.sensitivity import (
    drop_isolated_pf_knees,
    drop_isolated_pf_participants,
    redefine_outcome_tf_only,
)


def _build_df() -> pd.DataFrame:
    """Six participants, knee-level rows, covering every relevant pattern.

    p1: both knees TF-positive (kl>=2)         -> stay
    p2: both knees PF-only (kl<2 & oapf=1)     -> knee-level: drop both;
                                                   participant-level: drop
    p3: one knee TF, one knee PF-only          -> knee-level: drop the PF-only
                                                   one; participant-level: keep
    p4: KOA-negative both knees                -> stay
    p5: one knee TF, one knee negative         -> stay (canonical OA)
    p6: one knee PF-only, one knee negative    -> knee-level: drop PF-only;
                                                   participant-level: drop
    """
    rows = []
    def add(id_, kl_d, oapf_d, kl_e, oapf_e):
        rows.append({"idelsa": id_, "side": "D", "kl": kl_d, "oapf": oapf_d})
        rows.append({"idelsa": id_, "side": "E", "kl": kl_e, "oapf": oapf_e})

    add("p1", 2, 0, 3, 1)         # both TF
    add("p2", 0, 1, 1, 1)         # both PF-only
    add("p3", 2, 0, 0, 1)         # mixed
    add("p4", 0, 0, 1, 0)         # negative
    add("p5", 2, 1, 0, 0)         # one TF (with PF too), one negative
    add("p6", 1, 1, 0, 0)         # one PF-only, one negative

    df = pd.DataFrame(rows)
    df["oa_knee"] = ((df["kl"] >= 2) | (df["oapf"] == 1)).astype(int)
    return df


@pytest.fixture
def df() -> pd.DataFrame:
    return _build_df()


def test_knee_level_drops_only_pf_only_rows(df):
    res = drop_isolated_pf_knees(df)
    # Rows where kl<2 & oapf=1 should be gone:
    surviving = res.df
    pf_only_remaining = ((surviving["kl"].fillna(0) < 2) & (surviving["oapf"] == 1)).sum()
    assert pf_only_remaining == 0

    # Expected drops: both p2 knees, p3 left knee, p6 right knee = 4 rows
    assert res.n_rows_before - res.n_rows_after == 4

    # p3 must still have its TF-positive right knee
    assert "p3" in set(surviving["idelsa"])
    # p1 / p4 / p5 untouched
    assert (surviving["idelsa"] == "p1").sum() == 2
    assert (surviving["idelsa"] == "p4").sum() == 2
    assert (surviving["idelsa"] == "p5").sum() == 2


def test_participant_level_drops_only_pf_only_participants(df):
    res = drop_isolated_pf_participants(df)
    remaining = set(res.df["idelsa"])

    # p2 (both knees PF-only, no TF) -> dropped
    # p6 (one PF-only + one negative; no TF; has KOA) -> dropped
    assert "p2" not in remaining
    assert "p6" not in remaining

    # p3 has one TF-positive knee -> kept entirely (both rows)
    assert (res.df["idelsa"] == "p3").sum() == 2

    # KOA-negative participants kept regardless
    assert "p4" in remaining


def test_tf_only_outcome_keeps_all_rows(df):
    res = redefine_outcome_tf_only(df)
    assert len(res.df) == len(df)
    # Redefined outcome equals (kl>=2) exactly:
    assert ((res.df["oa_knee"] == (res.df["kl"] >= 2).astype(int)).all())
    # CRITICAL: the canonical outcome must NOT survive as a column.
    # Leaving it in would let ML models train on the label.
    assert "oa_knee_canonical" not in res.df.columns
    assert set(res.df.columns) == set(df.columns)


def test_audit_row_shape(df):
    res = drop_isolated_pf_knees(df)
    row = res.audit_row()
    for k in [
        "sensitivity", "description",
        "n_rows_before", "n_rows_after", "n_rows_dropped",
        "n_participants_before", "n_participants_after", "n_participants_dropped",
        "prevalence_before", "prevalence_after",
    ]:
        assert k in row
    assert row["n_rows_dropped"] == 4


def test_filters_dont_mutate_input(df):
    snapshot = df.copy()
    drop_isolated_pf_knees(df)
    drop_isolated_pf_participants(df)
    redefine_outcome_tf_only(df)
    pd.testing.assert_frame_equal(df, snapshot)
