"""Sensitivity-analysis data filters.

Reviewer-driven companions to the canonical analysis. Each function takes a
DataFrame already produced by :func:`koa_screening.data.load_and_prep_data`
and returns a filtered (or re-derived) copy plus a one-line audit row
explaining what changed. None of these touch the original data-prep
pipeline; the canonical regression tests therefore continue to pass.

Three variants of "isolated patellofemoral KOA exclusion" are provided so a
reviewer response can present them side by side:

1. ``drop_isolated_pf_knees`` -- knee-level: remove knee-rows where
   ``oapf==1 & kl<2`` (PF-only knees). The contralateral knee, if any, is
   kept. Outcome formula unchanged.
2. ``drop_isolated_pf_participants`` -- participant-level: remove whole
   participants whose KOA is *only* PF (no knee with KL>=2). Outcome
   formula unchanged.
3. ``redefine_outcome_tf_only`` -- no row removal; ``oa_knee`` is redefined
   as ``(kl>=2)`` so OAPF is not used in the outcome.

Each function is pure and side-effect-free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SensitivityResult:
    df: pd.DataFrame
    name: str  # short id used for the results folder
    description: str
    n_rows_before: int
    n_rows_after: int
    n_participants_before: int
    n_participants_after: int
    prevalence_before: float
    prevalence_after: float

    def audit_row(self) -> dict:
        return {
            "sensitivity": self.name,
            "description": self.description,
            "n_rows_before": self.n_rows_before,
            "n_rows_after": self.n_rows_after,
            "n_rows_dropped": self.n_rows_before - self.n_rows_after,
            "n_participants_before": self.n_participants_before,
            "n_participants_after": self.n_participants_after,
            "n_participants_dropped": self.n_participants_before - self.n_participants_after,
            "prevalence_before": round(self.prevalence_before, 4),
            "prevalence_after": round(self.prevalence_after, 4),
        }


def _summarise(df: pd.DataFrame, filtered: pd.DataFrame, name: str, description: str) -> SensitivityResult:
    return SensitivityResult(
        df=filtered.reset_index(drop=True),
        name=name,
        description=description,
        n_rows_before=int(len(df)),
        n_rows_after=int(len(filtered)),
        n_participants_before=int(df["idelsa"].nunique()),
        n_participants_after=int(filtered["idelsa"].nunique()) if len(filtered) else 0,
        prevalence_before=float(df["oa_knee"].mean()) if len(df) else float("nan"),
        prevalence_after=float(filtered["oa_knee"].mean()) if len(filtered) else float("nan"),
    )


def drop_isolated_pf_knees(df: pd.DataFrame) -> SensitivityResult:
    """Drop knee-rows where structural OA is driven *only* by patellofemoral
    involvement (``oapf==1`` and ``kl<2``). The contralateral knee, if any,
    is unaffected. This is the most common operationalisation in the
    literature.

    Note that the outcome formula is unchanged; for the surviving rows it
    collapses to ``oa_knee == (kl>=2)`` because no row has both
    ``oapf==1`` and ``kl<2`` anymore.
    """
    if "kl" not in df.columns or "oapf" not in df.columns:
        raise ValueError("df must contain 'kl' and 'oapf' columns")

    kl = pd.to_numeric(df["kl"], errors="coerce")
    oapf = pd.to_numeric(df["oapf"], errors="coerce")
    isolated = (oapf == 1) & (kl.fillna(0) < 2)
    filtered = df.loc[~isolated].copy()
    return _summarise(
        df,
        filtered,
        name="drop_isolated_pf_knees",
        description="Drop knee-rows with oapf==1 and kl<2 (knee-level isolated-PF exclusion).",
    )


def drop_isolated_pf_participants(df: pd.DataFrame) -> SensitivityResult:
    """Drop entire participants whose only KOA-positive knees are PF-only.

    A participant is excluded iff (a) they have at least one ``oa_knee==1``
    knee and (b) none of their knees has ``kl>=2``. Participants with no
    KOA at all are *not* dropped; only those whose KOA signal is entirely
    PF-driven.
    """
    if "idelsa" not in df.columns:
        raise ValueError("df must contain 'idelsa' column")

    kl = pd.to_numeric(df["kl"], errors="coerce")
    has_tf = df.assign(_has_tf=(kl >= 2)).groupby("idelsa")["_has_tf"].max()
    has_any_oa = df.groupby("idelsa")["oa_knee"].max()
    isolated_ids = set(has_any_oa.index[(has_any_oa == 1) & (~has_tf.astype(bool))])
    filtered = df.loc[~df["idelsa"].isin(isolated_ids)].copy()
    return _summarise(
        df,
        filtered,
        name="drop_isolated_pf_participants",
        description=(
            "Drop participants whose KOA is PF-only: oa_knee==1 in at least one "
            "knee but no knee with kl>=2 (participant-level isolated-PF exclusion)."
        ),
    )


def redefine_outcome_tf_only(df: pd.DataFrame) -> SensitivityResult:
    """Keep every row; redefine ``oa_knee`` as ``(kl>=2)`` only (drop PF
    contribution from the outcome).

    This answers the reviewer question "does the screening signal hold for
    tibiofemoral KOA specifically?" without changing the sample.
    """
    if "kl" not in df.columns:
        raise ValueError("df must contain 'kl' column")

    kl = pd.to_numeric(df["kl"], errors="coerce")
    redefined = df.copy()
    # IMPORTANT: do NOT add an audit copy of the old outcome to the
    # returned dataframe -- if anything downstream auto-discovers features
    # by exclusion, an oa_knee_canonical column would leak the canonical
    # label into the feature matrix. Prevalence shift is captured in the
    # SensitivityResult instead.
    redefined["oa_knee"] = (kl >= 2).astype(int)
    return _summarise(
        df,
        redefined,
        name="outcome_tf_only",
        description="Redefine oa_knee as (kl>=2); PF contribution removed from outcome.",
    )


# Registry consumed by scripts/07_sensitivity_isolated_pf.py
ISOLATED_PF_VARIANTS = {
    "drop_isolated_pf_knees": drop_isolated_pf_knees,
    "drop_isolated_pf_participants": drop_isolated_pf_participants,
    "outcome_tf_only": redefine_outcome_tf_only,
}
