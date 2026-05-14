"""Cross-validated AUC + bootstrap confidence intervals.

Consolidates what used to live in ``oarsi_utils.py`` and
``auc_ci_bootstrap_eval.py``. The 5-fold GroupKFold AUC routine here is the
single source of truth for every CV evaluation in the pipeline.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold


def cv_roc_auc(model, X, y, groups, n_splits: int = 5):
    """5-fold GroupKFold OOF AUC, with group=participant to prevent
    bilateral-knee leakage. Returns (fpr, tpr, auc, fitted_last_fold_model).
    """
    cv = GroupKFold(n_splits=n_splits)
    y_true_all: list = []
    y_pred_all: list = []

    for tr, te in cv.split(X, y, groups):
        model.fit(X.iloc[tr], y[tr])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X.iloc[te])[:, 1]
        else:
            probs = model.predict(X.iloc[te])

        y_true_all.extend(y[te])
        y_pred_all.extend(probs)

    fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
    auc = roc_auc_score(y_true_all, y_pred_all)
    return fpr, tpr, auc, model


def auc_ci_from_folds(aucs: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """t-interval CI from per-fold AUCs. Returns (mean, ci_low, ci_high)."""
    a = np.asarray(aucs, dtype=float)
    k = len(a)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if k > 1 else 0.0
    se = sd / math.sqrt(k) if k > 0 else float("nan")

    try:
        from scipy.stats import t

        tcrit = float(t.ppf(1 - alpha / 2, df=k - 1))
    except Exception:
        tcrit = 1.96

    return mean, mean - tcrit * se, mean + tcrit * se


def auc_ci_bootstrap_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Cluster bootstrap CI for AUC: resample participants (not knees) with
    replacement. Returns (auc_point, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    auc_point = float(roc_auc_score(y_true, y_pred))

    boot: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(groups, sampled)
        yt, yp = y_true[mask], y_pred[mask]
        if len(np.unique(yt)) < 2:
            continue
        boot.append(roc_auc_score(yt, yp))

    boot_arr = np.asarray(boot, dtype=float)
    lo = float(np.quantile(boot_arr, alpha / 2))
    hi = float(np.quantile(boot_arr, 1 - alpha / 2))
    return auc_point, lo, hi
