"""Sensitivity analyses: isolated patellofemoral KOA exclusion.

Reviewer-driven supplement. Runs the canonical 4-model x Without-Symptoms
comparison on three independent operationalisations of "remove isolated
patellofemoral KOA":

  1. Knee-level: drop knees where oapf=1 & kl<2
  2. Participant-level: drop participants whose KOA is PF-only
  3. Outcome-redefined: oa_knee = (kl>=2) only

Outputs land in ``results/sensitivity_<variant>/`` (one folder per variant)
plus a single side-by-side summary in
``results/sensitivity_summary_isolated_pf.csv`` for inclusion in either
the main text or supplementary materials.

This script does **not** modify any canonical results. The canonical
regression suite continues to pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from koa_screening import data
from koa_screening.config import (
    BASE_EXCLUDE,
    BIO_VARS,
    RAW_CSV,
    RESULTS_DIR,
    RESULTS_FINAL,
    SYMPTOM_VARS,
    WOMAC_VARS,
)
from koa_screening.evaluation import cv_roc_auc
from koa_screening.models import get_lr_pipe, get_pipeline
from koa_screening.runner import run_stepwise_mpms
from koa_screening.sensitivity import ISOLATED_PF_VARIANTS


def _run_one(df: pd.DataFrame, outdir: Path, mpms_vars: list[str]) -> list[dict]:
    """Run the canonical Without-Symptoms scenario on one (possibly filtered) df.

    Returns a list of {Model, AUC, n_rows, n_participants, prevalence} dicts.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("idelsa").reset_index(drop=True)
    y = df["oa_knee"].values
    groups = df["idelsa"].values

    all_cols = [c for c in df.columns if c not in BASE_EXCLUDE and c not in WOMAC_VARS and c not in BIO_VARS]
    feat_list = [c for c in all_cols if c not in SYMPTOM_VARS]
    X = df[feat_list]
    print(f"  Without-Symptoms features: {len(feat_list)}; rows: {len(df)}; participants: {df['idelsa'].nunique()}")

    results = []
    roc_fig, roc_ax = plt.subplots(figsize=(8, 7))

    # Stepwise LR (canonical MPMS feature order)
    valid_mpms = [v for v in mpms_vars if v in feat_list]
    if valid_mpms:
        step_res = run_stepwise_mpms(X, y, groups, valid_mpms)
        features_used = step_res["Added Variable"].tolist()
        X_mpms = X[features_used]
        fpr, tpr, auc, _ = cv_roc_auc(get_lr_pipe(), X_mpms, y, groups)
        results.append(
            {
                "Model": "Stepwise Logistic Regression",
                "AUC": auc,
                "k_features": len(features_used),
                "features": ",".join(features_used),
            }
        )
        roc_ax.plot(fpr, tpr, lw=2, label=f"Stepwise LR (k={len(features_used)}, AUC={auc:.3f})")
        step_res.to_csv(outdir / "stepwise_trajectory.csv", index=False)

    # The 3 ML models on the full Without-Symptoms feature set
    for model_name in ["XGBoost", "Random Forest", "Neural Network"]:
        pipe = get_pipeline(model_name)
        fpr, tpr, auc, _ = cv_roc_auc(pipe, X, y, groups)
        results.append({"Model": model_name, "AUC": auc, "k_features": len(feat_list), "features": ""})
        roc_ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={auc:.3f})")

    roc_ax.plot([0, 1], [0, 1], "k--", lw=1)
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title(f"Sensitivity ROC: {outdir.name}")
    roc_ax.legend(loc="lower right")
    roc_ax.grid(True, alpha=0.3)
    roc_fig.tight_layout()
    roc_fig.savefig(outdir / "roc_comparison.png", dpi=200)
    plt.close(roc_fig)

    pd.DataFrame(results).to_csv(outdir / "summary.csv", index=False)
    return results


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: raw CSV not found at {RAW_CSV}", file=sys.stderr)
        return 1

    print("Loading data (canonical preprocessing) ...")
    df = data.load_and_prep_data(str(RAW_CSV), outdir=str(RESULTS_FINAL))

    mpms_file = RESULTS_FINAL / "stepwise_mpms_clinical.csv"
    if not mpms_file.exists():
        print(f"WARN: {mpms_file} missing -- run scripts/02_feature_selection.py first")
        return 2
    mpms_vars = pd.read_csv(mpms_file)["Variable"].tolist()

    # 1) Canonical (no filter) -- so the side-by-side has a baseline row
    print("\n=== Canonical (baseline) ===")
    canon_outdir = RESULTS_DIR / "sensitivity_canonical"
    canonical = _run_one(df, canon_outdir, mpms_vars)
    for r in canonical:
        r.update(
            {
                "sensitivity": "canonical",
                "description": "No filter; reproduces results/comparison numbers.",
                "n_rows": len(df),
                "n_participants": df["idelsa"].nunique(),
                "prevalence": round(df["oa_knee"].mean(), 4),
            }
        )

    # 2) The three sensitivity variants
    audit_rows: list[dict] = []
    all_results = list(canonical)
    for variant_name, filter_fn in ISOLATED_PF_VARIANTS.items():
        print(f"\n=== Sensitivity: {variant_name} ===")
        res = filter_fn(df)
        audit_rows.append(res.audit_row())
        print(
            f"  n_rows: {res.n_rows_before} -> {res.n_rows_after} "
            f"(dropped {res.n_rows_before - res.n_rows_after})"
        )
        print(
            f"  n_participants: {res.n_participants_before} -> {res.n_participants_after}"
        )
        print(f"  prevalence: {res.prevalence_before:.3f} -> {res.prevalence_after:.3f}")

        sub_outdir = RESULTS_DIR / f"sensitivity_{variant_name}"
        rows = _run_one(res.df, sub_outdir, mpms_vars)
        for r in rows:
            r.update(
                {
                    "sensitivity": variant_name,
                    "description": res.description,
                    "n_rows": res.n_rows_after,
                    "n_participants": res.n_participants_after,
                    "prevalence": round(res.prevalence_after, 4),
                }
            )
        all_results.extend(rows)

    audit_path = RESULTS_DIR / "sensitivity_isolated_pf_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    print(f"\nAudit written: {audit_path}")

    summary_df = pd.DataFrame(all_results)[
        ["sensitivity", "description", "Model", "AUC", "k_features",
         "n_rows", "n_participants", "prevalence", "features"]
    ]
    summary_path = RESULTS_DIR / "sensitivity_summary_isolated_pf.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary written: {summary_path}\n")

    pivot = summary_df.pivot_table(
        index=["sensitivity", "n_rows", "n_participants", "prevalence"],
        columns="Model",
        values="AUC",
    ).reset_index()
    print("Side-by-side AUCs:")
    print(pivot.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
