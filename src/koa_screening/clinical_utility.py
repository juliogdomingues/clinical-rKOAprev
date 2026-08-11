"""Calibration, operating-point metrics, and decision-curve analysis.

Discrimination (AUC) alone does not establish that a model is usable. This
module adds the three things a prediction-model reviewer expects, all computed
from the **nested-CV out-of-fold predictions** (so they inherit the leak-free
estimate rather than in-sample optimism):

1. **Calibration** — calibration slope and calibration-in-the-large (CITL),
   plus the points for a calibration curve. The Brier score conflates
   calibration and discrimination; slope/intercept separate them.
2. **Operating-point metrics** — sensitivity, specificity, PPV, NPV at chosen
   probability thresholds, with cluster-bootstrap CIs.
3. **Decision-curve analysis** — net benefit across threshold probabilities
   against the treat-all and treat-none strategies.

All uncertainty uses the same participant-level cluster bootstrap as the rest
of the pipeline (resampling participants with multiplicity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import RND

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _cluster_indices(groups: np.ndarray) -> dict:
    groups = np.asarray(groups)
    return {u: np.where(groups == u)[0] for u in np.unique(groups)}


def _boot_rows(rng, idx_by_group, uniq):
    sampled = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([idx_by_group[u] for u in sampled])


# ---------------------------------------------------------------------------
# 1. Calibration
# ---------------------------------------------------------------------------
def calibration_slope_intercept(y_true, y_pred) -> tuple[float, float]:
    """Return (calibration_slope, calibration_in_the_large).

    slope: coefficient of logit(p_hat) in ``y ~ a + b*logit(p_hat)``; ideal 1.
           b<1 => predictions too extreme; b>1 => too conservative.
    CITL : intercept of ``y ~ a + offset(logit(p_hat))``; ideal 0.
           >0 => under-prediction of risk overall.
    """
    y = np.asarray(y_true, dtype=int)
    lp = _logit(y_pred).reshape(-1, 1)

    # Effectively unpenalised. A large finite C avoids both the deprecated
    # penalty=None and the "C is ignored" warning that C=inf triggers.
    slope_model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    slope_model.fit(lp, y)
    slope = float(slope_model.coef_[0][0])

    # CITL: logistic regression with the linear predictor as a fixed offset.
    # Fit by 1-D Newton/scan on the intercept (no sklearn offset support).
    off = lp.ravel()

    def neg_ll(a):
        z = a + off
        # stable log-loss
        return float(np.sum(np.logaddexp(0, z) - y * z))

    lo, hi = -10.0, 10.0
    for _ in range(200):  # golden-section-free simple ternary search
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if neg_ll(m1) < neg_ll(m2):
            hi = m2
        else:
            lo = m1
    citl = float((lo + hi) / 2)
    return slope, citl


def calibration_with_ci(y_true, y_pred, groups, n_boot: int = 2000, alpha: float = 0.05, seed: int = RND) -> dict:
    """Calibration slope and CITL with participant cluster-bootstrap CIs."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    slope, citl = calibration_slope_intercept(y_true, y_pred)

    rng = np.random.default_rng(seed)
    idx_by_group = _cluster_indices(groups)
    uniq = np.array(list(idx_by_group.keys()))
    slopes, citls = [], []
    for _ in range(n_boot):
        rows = _boot_rows(rng, idx_by_group, uniq)
        yt = y_true[rows]
        if len(np.unique(yt)) < 2:
            continue
        try:
            s, c = calibration_slope_intercept(yt, y_pred[rows])
        except Exception:
            continue
        slopes.append(s)
        citls.append(c)

    q = lambda a, arr: float(np.quantile(arr, a))  # noqa: E731
    return {
        "calibration_slope": slope,
        "slope_ci_low": q(alpha / 2, slopes), "slope_ci_high": q(1 - alpha / 2, slopes),
        "calibration_in_the_large": citl,
        "citl_ci_low": q(alpha / 2, citls), "citl_ci_high": q(1 - alpha / 2, citls),
        "n_boot_used": len(slopes),
    }


def calibration_curve_points(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted risk by equal-count (quantile) bins."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=n_bins, labels=False)
    out = df.groupby("bin").agg(
        n=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean"),
    ).reset_index()
    # Wilson interval for the observed proportion
    z = 1.959963985
    n, p = out["n"].to_numpy(), out["observed"].to_numpy()
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    out["obs_ci_low"] = np.clip(centre - half, 0, 1)
    out["obs_ci_high"] = np.clip(centre + half, 0, 1)
    return out


# ---------------------------------------------------------------------------
# 2. Operating-point metrics
# ---------------------------------------------------------------------------
def _confusion_rates(y, p, thr):
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return sens, spec, ppv, npv, tp, fp, fn, tn


def threshold_metrics(y_true, y_pred, groups, thresholds, n_boot: int = 1000,
                      alpha: float = 0.05, seed: int = RND) -> pd.DataFrame:
    """Sensitivity/specificity/PPV/NPV at each threshold, with cluster CIs."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)
    idx_by_group = _cluster_indices(groups)
    uniq = np.array(list(idx_by_group.keys()))

    boots = [_boot_rows(rng, idx_by_group, uniq) for _ in range(n_boot)]
    rows = []
    for thr in thresholds:
        sens, spec, ppv, npv, tp, fp, fn, tn = _confusion_rates(y_true, y_pred, thr)
        acc = {"sens": [], "spec": [], "ppv": [], "npv": []}
        for r in boots:
            s, sp, pv, nv, *_ = _confusion_rates(y_true[r], y_pred[r], thr)
            acc["sens"].append(s); acc["spec"].append(sp); acc["ppv"].append(pv); acc["npv"].append(nv)
        def ci(k):
            a = np.asarray(acc[k], dtype=float)
            a = a[~np.isnan(a)]
            return (float(np.quantile(a, alpha / 2)), float(np.quantile(a, 1 - alpha / 2))) if a.size else (np.nan, np.nan)
        s_lo, s_hi = ci("sens"); sp_lo, sp_hi = ci("spec")
        p_lo, p_hi = ci("ppv"); n_lo, n_hi = ci("npv")
        rows.append({
            "threshold": thr, "n_flagged": int(tp + fp), "pct_flagged": (tp + fp) / len(y_true) * 100,
            "sensitivity": sens, "sens_ci_low": s_lo, "sens_ci_high": s_hi,
            "specificity": spec, "spec_ci_low": sp_lo, "spec_ci_high": sp_hi,
            "ppv": ppv, "ppv_ci_low": p_lo, "ppv_ci_high": p_hi,
            "npv": npv, "npv_ci_low": n_lo, "npv_ci_high": n_hi,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return pd.DataFrame(rows)


def youden_threshold(y_true, y_pred) -> float:
    """Threshold maximizing Youden's J (sensitivity + specificity - 1)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_true, y_pred)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


# ---------------------------------------------------------------------------
# 3. Decision-curve analysis
# ---------------------------------------------------------------------------
def decision_curve(y_true, y_pred, thresholds=None) -> pd.DataFrame:
    """Net benefit of the model vs treat-all and treat-none.

    NB(model) = TP/n - (FP/n) * pt/(1-pt);  NB(treat-all) uses everyone flagged.
    Net benefit is on the scale of "true positives per patient", so the strategy
    with the highest curve at a clinically relevant threshold is preferred.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=float)
    n = len(y)
    prev = y.mean()
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.51, 0.01), 3)

    rows = []
    for pt in thresholds:
        w = pt / (1 - pt)
        pred = p >= pt
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        nb_model = tp / n - (fp / n) * w
        nb_all = prev - (1 - prev) * w
        rows.append({
            "threshold": float(pt),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": 0.0,
        })
    out = pd.DataFrame(rows)
    # standardised net benefit + net reduction in avoidable imaging per 100
    out["net_benefit_gain_vs_best_default"] = out["net_benefit_model"] - out[
        ["net_benefit_treat_all", "net_benefit_treat_none"]].max(axis=1)
    return out
