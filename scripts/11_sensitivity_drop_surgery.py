"""Step 11: sensitivity analysis dropping history of knee surgery.

Reviewers note that history of knee surgery (raw OR ~8.10) is close to
circular for PREVALENT radiographic KOA -- surgery is largely a consequence
or marker of established disease, not an independent antecedent. This runner
quantifies how much discrimination survives without it, in the Screening
(Without Symptoms) scenario, across three configurations:

  - canonical            : surgery retained (reproduces the headline numbers)
  - drop_surgery         : history_surgery removed from the feature/MPMS set
  - drop_surgery_trauma  : history_surgery AND history_trauma removed

For each, it reports the Stepwise LR and the three ML models' OOF AUC with
cluster-bootstrap 95% CIs, and re-estimates the raw + standardized ORs of the
Stepwise model (so you can see whether the other coefficients inflate once the
near-circular variable is gone).

Outputs -> results/sensitivity_drop_surgery/ . Canonical results untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from koa_screening import data
from koa_screening.config import BASE_EXCLUDE, BIO_VARS, RAW_CSV, RESULTS_DIR, RESULTS_FINAL, SYMPTOM_VARS, WOMAC_VARS
from koa_screening.evaluation import auc_ci_bootstrap_by_group, cv_oof_predictions
from koa_screening.models import get_lr_pipe, get_pipeline
from koa_screening.runner import calculate_odds_ratios, run_stepwise_mpms

N_BOOT = 2000
ML_MODELS = ["XGBoost", "Random Forest", "Neural Network"]
OUTDIR = RESULTS_DIR / "sensitivity_drop_surgery"

CONFIGS = [
    ("canonical", []),
    ("drop_surgery", ["history_surgery"]),
    ("drop_surgery_trauma", ["history_surgery", "history_trauma"]),
]


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: raw CSV not found at {RAW_CSV}", file=sys.stderr)
        return 1
    mpms_file = RESULTS_FINAL / "stepwise_mpms_clinical.csv"
    if not mpms_file.exists():
        print(f"ERROR: {mpms_file} missing -- run scripts/02_feature_selection.py first", file=sys.stderr)
        return 2
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = data.load_and_prep_data(str(RAW_CSV), outdir=str(RESULTS_FINAL))
    df = df.sort_values("idelsa").reset_index(drop=True)
    y = df["oa_knee"].values
    groups = df["idelsa"].values
    mpms_vars = pd.read_csv(mpms_file)["Variable"].tolist()

    all_cols = [c for c in df.columns if c not in BASE_EXCLUDE and c not in WOMAC_VARS and c not in BIO_VARS]
    cols_without = [c for c in all_cols if c not in SYMPTOM_VARS]

    rows = []
    for config_name, drop in CONFIGS:
        drop_set = set(drop)
        feats = [c for c in cols_without if c not in drop_set]
        X = df[feats]
        valid_mpms = [v for v in mpms_vars if v in feats and v not in drop_set]
        step_res = run_stepwise_mpms(X, y, groups, valid_mpms)
        features_used = step_res["Added Variable"].tolist()

        print(f"\n=== {config_name} === Stepwise features: {features_used}")

        # Stepwise LR
        yt, yp, gp = cv_oof_predictions(get_lr_pipe(), X[features_used], y, groups)
        auc, lo, hi = auc_ci_bootstrap_by_group(yt, yp, gp, n_boot=N_BOOT)
        rows.append({"config": config_name, "Model": "Stepwise Logistic Regression",
                     "n_features": len(features_used), "AUC": auc, "AUC_CI_Low": lo, "AUC_CI_High": hi})

        # ML models on the full reduced Without-Symptoms set
        for m in ML_MODELS:
            yt, yp, gp = cv_oof_predictions(get_pipeline(m), X, y, groups)
            a, l, h = auc_ci_bootstrap_by_group(yt, yp, gp, n_boot=N_BOOT)
            rows.append({"config": config_name, "Model": m, "n_features": X.shape[1],
                         "AUC": a, "AUC_CI_Low": l, "AUC_CI_High": h})

        # ORs for the Stepwise model in this configuration
        calculate_odds_ratios(X, y, groups, features_used, f"drop_surgery_{config_name}", OUTDIR)

    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "summary.csv", index=False)
    print(f"\nWrote {OUTDIR / 'summary.csv'} and OR tables to {OUTDIR}")

    pivot = out.assign(
        AUC_str=out.apply(lambda r: f"{r['AUC']:.3f} ({r['AUC_CI_Low']:.3f}-{r['AUC_CI_High']:.3f})", axis=1)
    ).pivot(index="Model", columns="config", values="AUC_str")
    print("\nAUC (95% CI) by configuration:")
    print(pivot.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
