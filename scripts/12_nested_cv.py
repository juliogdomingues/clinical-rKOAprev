"""Step 12: nested cross-validation with inner-loop learning (leak-free comparison).

For each scenario, runs:
  - the Stepwise LR with LASSO+forward-stepwise selection re-run inside each
    outer training fold (koa_screening.nested.nested_lr), and
  - XGBoost / Random Forest / MLP with an inner RandomizedSearchCV
    hyperparameter search inside each outer training fold (nested_ml),
then reports the pooled out-of-fold AUC (cluster-bootstrap 95% CI) and Brier for
every model, plus a paired AUC-difference test (LR vs each ML) on the identical
out-of-fold rows.

This is the honest, symmetric comparison: both arms learn only from training
data. Outputs:
  results/comparison/nested_cv_summary.csv          (AUC/CI/Brier per model)
  results/comparison/nested_cv_paired_diff.csv      (LR - ML difference + CI + p)
  results/comparison/nested_cv_lr_fold_features.csv (features chosen per fold)
  results/comparison/nested_cv_ml_fold_params.csv   (best hyperparameters per fold)
"""
from __future__ import annotations

import sys

import pandas as pd

from koa_screening import data
from koa_screening.config import (
    BASE_EXCLUDE,
    BIO_VARS,
    RAW_CSV,
    RESULTS_COMPARISON,
    RESULTS_FINAL,
    SYMPTOM_VARS,
    WOMAC_VARS,
)
from koa_screening.evaluation import auc_ci_bootstrap_by_group, brier_ci_bootstrap_by_group
from koa_screening.nested import nested_lr, nested_ml, paired_auc_diff

ML_MODELS = ["XGBoost", "Random Forest", "Neural Network"]
N_ITER = 40  # RandomizedSearchCV candidates per outer fold (x5 folds x distinct seeds = 200 configs explored)


def _scenarios(df):
    all_cols = [c for c in df.columns if c not in BASE_EXCLUDE and c not in WOMAC_VARS]
    base_pool = [c for c in all_cols if c not in BIO_VARS]
    without = [c for c in base_pool if c not in SYMPTOM_VARS]
    # (name, feature_pool, run_stepwise_lr)
    return [
        ("Without Symptoms", without, True),
        ("With Symptoms", list(base_pool), True),
        ("Virtual Maximum", list(all_cols), False),  # ML-only bio contrast; LR skipped
    ]


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: raw CSV not found at {RAW_CSV}", file=sys.stderr)
        return 1

    df = data.load_and_prep_data(str(RAW_CSV), outdir=str(RESULTS_FINAL))
    df = df.sort_values("idelsa").reset_index(drop=True)
    y = df["oa_knee"].values
    groups = df["idelsa"].values

    summary_rows = []
    paired_rows = []
    lr_feat_rows = []
    ml_param_rows = []

    for scen, pool, run_lr in _scenarios(df):
        print(f"\n=== Nested CV: {scen} ({len(pool)} candidate features) ===")
        oof = {}  # model -> (yt, yp, gp)

        if run_lr:
            print("  Stepwise LR (selection inside each outer fold) ...")
            yt, yp, gp, feats = nested_lr(df, pool, y, groups)
            oof["Stepwise LR"] = (yt, yp, gp)
            for k, fs in enumerate(feats, 1):
                lr_feat_rows.append({"Scenario": scen, "outer_fold": k, "n_features": len(fs), "features": ",".join(fs)})

        for m in ML_MODELS:
            print(f"  {m} (inner hyperparameter search) ...")
            yt, yp, gp, params = nested_ml(df[pool], y, groups, m, n_iter=N_ITER)
            oof[m] = (yt, yp, gp)
            for k, pr in enumerate(params, 1):
                ml_param_rows.append({"Scenario": scen, "Model": m, "outer_fold": k, "best_params": str(pr)})

        for model, (yt, yp, gp) in oof.items():
            auc, alo, ahi = auc_ci_bootstrap_by_group(yt, yp, gp)
            brier, blo, bhi = brier_ci_bootstrap_by_group(yt, yp, gp)
            summary_rows.append({
                "Scenario": scen, "Model": model,
                "AUC": auc, "AUC_CI_Low": alo, "AUC_CI_High": ahi,
                "Brier": brier, "Brier_CI_Low": blo, "Brier_CI_High": bhi,
            })

        # Paired LR - ML differences on identical OOF rows
        if "Stepwise LR" in oof:
            yt_lr, yp_lr, gp_lr = oof["Stepwise LR"]
            for m in ML_MODELS:
                yt_m, yp_m, gp_m = oof[m]
                d = paired_auc_diff(yt_lr, yp_lr, yp_m, gp_lr)
                paired_rows.append({
                    "Scenario": scen, "Comparison": f"Stepwise LR - {m}",
                    "delta_AUC": d["diff"], "CI_Low": d["ci_low"], "CI_High": d["ci_high"],
                    "p_value": d["p_value"], "n_boot_used": d["n_boot_used"],
                })

        # Write incrementally after each scenario so a late crash keeps progress
        RESULTS_COMPARISON.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(RESULTS_COMPARISON / "nested_cv_summary.csv", index=False)
        pd.DataFrame(paired_rows).to_csv(RESULTS_COMPARISON / "nested_cv_paired_diff.csv", index=False)
        pd.DataFrame(lr_feat_rows).to_csv(RESULTS_COMPARISON / "nested_cv_lr_fold_features.csv", index=False)
        pd.DataFrame(ml_param_rows).to_csv(RESULTS_COMPARISON / "nested_cv_ml_fold_params.csv", index=False)
        print(f"  [{scen}] written.")

    s = pd.DataFrame(summary_rows)
    s["AUC (95% CI)"] = s.apply(lambda r: f"{r['AUC']:.3f} ({r['AUC_CI_Low']:.3f}-{r['AUC_CI_High']:.3f})", axis=1)
    print("\n=== NESTED-CV AUC (95% CI) ===")
    print(s[["Scenario", "Model", "AUC (95% CI)"]].to_string(index=False))
    p = pd.DataFrame(paired_rows)
    if len(p):
        p["delta (95% CI)"] = p.apply(lambda r: f"{r['delta_AUC']:+.3f} ({r['CI_Low']:+.3f},{r['CI_High']:+.3f}) p={r['p_value']:.3f}", axis=1)
        print("\n=== PAIRED AUC DIFFERENCE (Stepwise LR - ML) ===")
        print(p[["Scenario", "Comparison", "delta (95% CI)"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
