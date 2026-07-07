"""Nested cross-validation with inner-loop learning for a leak-free, symmetric
model comparison.

The single flat GroupKFold in :mod:`koa_screening.evaluation` scores a feature
set (LR) or fixed hyperparameters (ML) that were chosen on the *whole* dataset,
so the test folds helped choose them. This module puts BOTH arms' learning
inside the outer folds:

  - LR arm: LASSO pre-filter + forward-stepwise selection are re-run on each
    outer TRAINING split only; the resulting model predicts the untouched outer
    test split.
  - ML arm: a RandomizedSearchCV hyperparameter search (inner GroupKFold) runs
    on each outer training split; the best model predicts the outer test split.

The pooled out-of-fold predictions give an unbiased performance estimate that
is directly comparable across arms (both learned only from training data), and
supports a paired AUC-difference test. See docs/METHODOLOGY.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features
from .config import RND


# ---------------------------------------------------------------------------
# LR arm: selection inside each outer training fold
# ---------------------------------------------------------------------------
def _lr_pipe(seed: int = RND):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight=None, random_state=seed),
    )


def nested_lr(df: pd.DataFrame, pool: list[str], y, groups, *, n_outer: int = 5, seed: int = RND):
    """Nested-CV for the stepwise LR. Per outer fold: LASSO -> forward-stepwise
    prefix selection on the TRAIN split, fit LR, predict the TEST split.

    Returns (y_true, y_pred, groups_oof, per_fold_features).
    """
    cv = GroupKFold(n_splits=n_outer)
    Xpool = df[pool]
    yt: list = []
    yp: list = []
    gp: list = []
    fold_feats: list[list[str]] = []

    for tr, te in cv.split(Xpool, y, groups):
        Xtr, ytr, gtr = Xpool.iloc[tr], y[tr], groups[tr]
        # ---- selection on TRAIN ONLY (no test rows seen); grouped by participant ----
        lasso_vars = features.run_lasso(Xtr, ytr, groups=gtr)  # L1 pre-filter (grouped C selection)
        sel = features.run_mpms(Xtr, ytr, gtr, lasso_vars)  # forward prefix by inner GroupKFold CV-AUC
        sel = [f for f in sel if f in pool]
        if not sel:  # degenerate guard
            sel = (lasso_vars or list(pool))[:5]
        pipe = _lr_pipe(seed)
        pipe.fit(Xtr[sel], ytr)
        prob = pipe.predict_proba(Xpool.iloc[te][sel])[:, 1]
        yt.extend(y[te])
        yp.extend(prob)
        gp.extend(np.asarray(groups)[te])
        fold_feats.append(sel)

    return np.asarray(yt), np.asarray(yp), np.asarray(gp), fold_feats


# ---------------------------------------------------------------------------
# ML arm: inner-loop hyperparameter search inside each outer training fold
# ---------------------------------------------------------------------------
def _ml_pipe_and_dist(model_name: str, seed: int = RND):
    """Return (pipeline, param_distributions) for RandomizedSearchCV. Param keys
    are prefixed with the make_pipeline step name (lowercased class name)."""
    imp = SimpleImputer(strategy="median")
    if model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        pipe = make_pipeline(imp, RandomForestClassifier(random_state=seed, n_jobs=-1))
        dist = {
            "randomforestclassifier__n_estimators": [200, 400, 600],
            "randomforestclassifier__max_depth": [4, 6, 8, 10, None],
            "randomforestclassifier__min_samples_leaf": [1, 2, 5, 10, 20],
            "randomforestclassifier__max_features": ["sqrt", 0.3, 0.5, None],
            "randomforestclassifier__class_weight": ["balanced", "balanced_subsample", None],
        }
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier

        pipe = make_pipeline(imp, XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=-1))
        dist = {
            "xgbclassifier__n_estimators": [100, 200, 400],
            "xgbclassifier__max_depth": [2, 3, 4, 6],
            "xgbclassifier__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "xgbclassifier__subsample": [0.7, 0.85, 1.0],
            "xgbclassifier__colsample_bytree": [0.7, 0.85, 1.0],
            "xgbclassifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "xgbclassifier__min_child_weight": [1, 3, 5],
        }
    elif model_name == "Neural Network":
        from sklearn.neural_network import MLPClassifier

        pipe = make_pipeline(
            imp,
            StandardScaler(),
            MLPClassifier(max_iter=800, early_stopping=True, random_state=seed),
        )
        dist = {
            "mlpclassifier__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (64, 32, 16)],
            "mlpclassifier__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "mlpclassifier__learning_rate_init": [1e-3, 5e-4, 3e-3],
            "mlpclassifier__activation": ["relu", "tanh"],
        }
    else:
        raise ValueError(model_name)
    return pipe, dist


def nested_ml(X: pd.DataFrame, y, groups, model_name: str, *, n_outer: int = 5,
              n_inner: int = 3, n_iter: int = 25, seed: int = RND):
    """Nested-CV for an ML model with an inner RandomizedSearchCV (GroupKFold)
    hyperparameter search on each outer training split.

    Returns (y_true, y_pred, groups_oof, per_fold_best_params).
    """
    cv = GroupKFold(n_splits=n_outer)
    yt: list = []
    yp: list = []
    gp: list = []
    fold_params: list[dict] = []

    for fold_i, (tr, te) in enumerate(cv.split(X, y, groups)):
        Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
        pipe, dist = _ml_pipe_and_dist(model_name, seed)
        inner = GroupKFold(n_splits=n_inner)
        search = RandomizedSearchCV(
            pipe, dist, n_iter=n_iter, scoring="roc_auc", cv=inner,
            # vary the sampled candidates per outer fold (seed + fold) so the
            # search explores n_outer x n_iter distinct configs overall, not the
            # same n_iter every fold.
            random_state=seed + fold_i, n_jobs=-1, refit=True, error_score=np.nan,
        )
        search.fit(Xtr, ytr, groups=gtr)  # groups drives the inner GroupKFold
        prob = search.predict_proba(X.iloc[te])[:, 1]
        yt.extend(y[te])
        yp.extend(prob)
        gp.extend(np.asarray(groups)[te])
        fold_params.append({k.split("__")[-1]: v for k, v in search.best_params_.items()})

    return np.asarray(yt), np.asarray(yp), np.asarray(gp), fold_params


# ---------------------------------------------------------------------------
# Paired AUC-difference test (cluster bootstrap of the difference)
# ---------------------------------------------------------------------------
def paired_auc_diff(y_true, pred_a, pred_b, groups, *, n_boot: int = 2000,
                    alpha: float = 0.05, seed: int = RND) -> dict:
    """Cluster bootstrap of AUC(a) - AUC(b) on the SAME resampled participants.

    a and b are two models' OOF predictions on the identical rows. Returns the
    point difference, CI, and a two-sided bootstrap p-value for H0: diff == 0.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pa = np.asarray(pred_a)
    pb = np.asarray(pred_b)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {u: np.where(groups == u)[0] for u in uniq}

    diff_point = float(roc_auc_score(y_true, pa) - roc_auc_score(y_true, pb))
    diffs: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[u] for u in sampled])
        yt = y_true[rows]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(roc_auc_score(yt, pa[rows]) - roc_auc_score(yt, pb[rows]))
    diffs_arr = np.asarray(diffs, dtype=float)
    lo = float(np.quantile(diffs_arr, alpha / 2))
    hi = float(np.quantile(diffs_arr, 1 - alpha / 2))
    # Two-sided add-one-smoothed bootstrap p-value. Non-strict inequalities so
    # the all-ties case (two identical models -> every diff == 0) correctly
    # gives p = 1, while +1/(n+1) smoothing floors the smallest reportable p at
    # the resampling resolution (never exactly 0). Ties at exactly 0 are
    # negligible for continuous AUC differences.
    n_used = int(len(diffs_arr))
    cnt_le = int(np.sum(diffs_arr <= 0))
    cnt_ge = int(np.sum(diffs_arr >= 0))
    p_value = float(min(1.0, 2 * (min(cnt_le, cnt_ge) + 1) / (n_used + 1)))
    return {"diff": diff_point, "ci_low": lo, "ci_high": hi, "p_value": p_value, "n_boot_used": n_used}
