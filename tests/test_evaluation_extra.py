"""Synthetic tests for the OOF-prediction + Brier helpers added to evaluation.py.

These run anywhere (no real data). The key invariant is that cv_oof_predictions
yields predictions whose pooled AUC equals cv_roc_auc's AUC -- i.e. the new
function is a faithful, reusable decomposition of the regression-tested path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from koa_screening.evaluation import (
    auc_ci_bootstrap_by_group,
    brier_ci_bootstrap_by_group,
    cv_oof_predictions,
    cv_roc_auc,
)


def _toy():
    rng = np.random.default_rng(0)
    n = 200
    groups = np.repeat(np.arange(n // 2), 2)  # 2 knees per participant
    x1 = rng.normal(size=n)
    y = (x1 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": rng.normal(size=n)})
    return X, y, groups


def test_oof_auc_matches_cv_roc_auc():
    X, y, groups = _toy()
    p1 = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=500, random_state=42))
    _, _, auc, _ = cv_roc_auc(p1, X, y, groups)
    p2 = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=500, random_state=42))
    yt, yp, gp = cv_oof_predictions(p2, X, y, groups)
    assert abs(roc_auc_score(yt, yp) - auc) < 1e-12
    assert set(np.unique(gp)).issubset(set(np.unique(groups)))
    assert len(yt) == len(y)


def test_brier_and_auc_ci_bounds():
    X, y, groups = _toy()
    pipe = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=500, random_state=42))
    yt, yp, gp = cv_oof_predictions(pipe, X, y, groups)
    b, b_lo, b_hi = brier_ci_bootstrap_by_group(yt, yp, gp, n_boot=200)
    assert 0.0 <= b <= 1.0
    assert b_lo <= b <= b_hi
    a, a_lo, a_hi = auc_ci_bootstrap_by_group(yt, yp, gp, n_boot=200)
    assert a_lo <= a <= a_hi


def test_brier_ci_is_deterministic():
    X, y, groups = _toy()
    pipe = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=500, random_state=42))
    yt, yp, gp = cv_oof_predictions(pipe, X, y, groups)
    r1 = brier_ci_bootstrap_by_group(yt, yp, gp, n_boot=200, seed=42)
    r2 = brier_ci_bootstrap_by_group(yt, yp, gp, n_boot=200, seed=42)
    assert r1 == r2
