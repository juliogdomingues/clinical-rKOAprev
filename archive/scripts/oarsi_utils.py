import math
import numpy as np
from sklearn.metrics import roc_auc_score

def auc_ci_from_folds(aucs: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Returns: (mean_auc, ci_low, ci_high)
    CI via t-interval on fold AUCs: mean ± t * sd/sqrt(k)
    """
    a = np.asarray(aucs, dtype=float)
    k = len(a)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if k > 1 else 0.0
    se = sd / math.sqrt(k) if k > 0 else float("nan")

    # Prefer t critical if SciPy available; else normal approx
    try:
        from scipy.stats import t
        tcrit = float(t.ppf(1 - alpha / 2, df=k - 1))
    except Exception:
        tcrit = 1.96  # approx

    lo = mean - tcrit * se
    hi = mean + tcrit * se
    return mean, lo, hi

def auc_ci_bootstrap_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap AUC resampling participants (groups) with replacement.
    Returns: (auc_point, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    auc_point = float(roc_auc_score(y_true, y_pred))

    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(groups, sampled)
        yt = y_true[mask]
        yp = y_pred[mask]
        # need both classes
        if len(np.unique(yt)) < 2:
            continue
        boot.append(roc_auc_score(yt, yp))

    boot = np.asarray(boot, dtype=float)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return auc_point, lo, hi
