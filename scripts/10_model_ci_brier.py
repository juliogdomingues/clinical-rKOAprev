"""Step 10: AUC confidence intervals + Brier score for EVERY model.

Closes the reviewer's "asymmetric comparison" critique: the canonical
summary_all_models.csv reports a bare AUC point estimate per model, while the
manuscript only gave a CI/Brier for the logistic model. This runner computes,
for every model in every scenario, the out-of-fold AUC with a cluster
bootstrap 95% CI (resampling participants) and the Brier score with its own
bootstrap CI, plus n / events / prevalence.

It reuses the SAME scenario/feature construction as scripts/03_run_comparison.py
(runner.run_comparison), so the AUC point estimates reproduce
results/comparison/summary_all_models.csv exactly; only CIs + Brier are added.
Promotes capability that previously lived only in archive/scripts/auc_ci_bootstrap_eval.py.

Output: results/comparison/summary_all_models_ci_brier.csv
This does NOT modify the canonical summary_all_models.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
from koa_screening.evaluation import (
    auc_ci_bootstrap_by_group,
    brier_ci_bootstrap_by_group,
    cv_oof_predictions,
)
from koa_screening.models import get_lr_pipe, get_pipeline
from koa_screening.runner import run_stepwise_mpms

N_BOOT = 2000
ML_MODELS = ["XGBoost", "Random Forest", "Neural Network"]


def _scenarios(df):
    # Mirror runner.run_comparison: exclude WOMAC everywhere; bioimpedance only
    # in Virtual Maximum.
    all_cols = [c for c in df.columns if c not in BASE_EXCLUDE and c not in WOMAC_VARS]
    base_pool = [c for c in all_cols if c not in BIO_VARS]
    cols_without = [c for c in base_pool if c not in SYMPTOM_VARS]
    return [
        ("Virtual Maximum", list(all_cols), False),   # (name, feats, run_stepwise)
        ("With Symptoms", list(base_pool), True),
        ("Without Symptoms", cols_without, True),
    ]


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: raw CSV not found at {RAW_CSV}", file=sys.stderr)
        return 1
    mpms_file = RESULTS_FINAL / "stepwise_mpms_clinical.csv"
    if not mpms_file.exists():
        print(f"ERROR: {mpms_file} missing -- run scripts/02_feature_selection.py first", file=sys.stderr)
        return 2

    df = data.load_and_prep_data(str(RAW_CSV), outdir=str(RESULTS_FINAL))
    df = df.sort_values("idelsa").reset_index(drop=True)
    y = df["oa_knee"].values
    groups = df["idelsa"].values
    mpms_vars = pd.read_csv(mpms_file)["Variable"].tolist()

    n_knees = int(len(df))
    n_participants = int(df["idelsa"].nunique())
    n_events = int(y.sum())
    prevalence = round(float(y.mean()), 4)

    rows = []
    for scenario_name, feat_list, run_stepwise in _scenarios(df):
        valid_feats = [f for f in feat_list if f in df.columns]
        X = df[valid_feats]

        combos = []  # (model_label, X_for_model, pipe_factory)
        if run_stepwise:
            valid_mpms = [v for v in mpms_vars if v in valid_feats]
            step_res = run_stepwise_mpms(X, y, groups, valid_mpms)
            features_used = step_res["Added Variable"].tolist()
            combos.append(("Stepwise (Full)", X[features_used], get_lr_pipe))
        for m in ML_MODELS:
            combos.append((m, X, lambda m=m: get_pipeline(m)))

        for model_label, X_model, factory in combos:
            print(f"  {scenario_name} / {model_label} ({X_model.shape[1]} features) ...")
            yt, yp, gp = cv_oof_predictions(factory(), X_model, y, groups)
            auc, auc_lo, auc_hi = auc_ci_bootstrap_by_group(yt, yp, gp, n_boot=N_BOOT)
            brier, brier_lo, brier_hi = brier_ci_bootstrap_by_group(yt, yp, gp, n_boot=N_BOOT)
            rows.append({
                "Scenario": scenario_name,
                "Model": model_label,
                "n_knees": n_knees,
                "n_participants": n_participants,
                "n_events": n_events,
                "prevalence": prevalence,
                "n_features": int(X_model.shape[1]),
                "AUC": auc,
                "AUC_CI_Low": auc_lo,
                "AUC_CI_High": auc_hi,
                "Brier": brier,
                "Brier_CI_Low": brier_lo,
                "Brier_CI_High": brier_hi,
            })

    out = pd.DataFrame(rows)
    RESULTS_COMPARISON.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_COMPARISON / "summary_all_models_ci_brier.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Consistency check against the canonical AUC summary (point estimates must match)
    canon_path = RESULTS_COMPARISON / "summary_all_models.csv"
    if canon_path.exists():
        canon = pd.read_csv(canon_path)
        merged = canon.merge(out[["Scenario", "Model", "AUC"]], on=["Scenario", "Model"], suffixes=("_canon", "_new"))
        if len(merged) != len(canon):
            print(f"WARNING: consistency check matched {len(merged)}/{len(canon)} canonical rows "
                  f"(Scenario/Model labels drifted?) -- point-estimate check is incomplete.", file=sys.stderr)
        if len(merged):
            max_delta = (merged["AUC_canon"] - merged["AUC_new"]).abs().max()
            status = "OK" if max_delta < 1e-9 else "MISMATCH"
            print(f"Consistency vs summary_all_models.csv: max |AUC diff| = {max_delta:.2e} ({status})")
            if status == "MISMATCH":
                print("WARNING: AUC point estimates diverge from the canonical summary.", file=sys.stderr)
    else:
        print("NOTE: summary_all_models.csv not found -- skipped consistency check.", file=sys.stderr)

    show = out.copy()
    show["AUC (95% CI)"] = show.apply(lambda r: f"{r['AUC']:.3f} ({r['AUC_CI_Low']:.3f}-{r['AUC_CI_High']:.3f})", axis=1)
    show["Brier (95% CI)"] = show.apply(lambda r: f"{r['Brier']:.3f} ({r['Brier_CI_Low']:.3f}-{r['Brier_CI_High']:.3f})", axis=1)
    print("\n" + show[["Scenario", "Model", "n_features", "AUC (95% CI)", "Brier (95% CI)"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
