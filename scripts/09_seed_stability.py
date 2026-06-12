"""Sensitivity analysis: seed stability of canonical AUCs.

Re-runs the canonical 4-model x Without-Symptoms comparison at seeds
0..9 and reports mean +/- SD per model. Lets a reviewer see that the
published numbers (seed 42) are not cherry-picked.

This script does **not** modify any canonical results. Output goes to
``results/sensitivity_seed_stability/``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from koa_screening import data
from koa_screening.config import (
    BASE_EXCLUDE,
    RAW_CSV,
    RESULTS_DIR,
    RESULTS_FINAL,
    SYMPTOM_VARS,
)
from koa_screening.evaluation import cv_roc_auc

N_SEEDS = 10
OUTDIR = RESULTS_DIR / "sensitivity_seed_stability"


def _make_pipes(seed: int):
    """Rebuild the four canonical pipelines at an explicit seed.

    Each pipeline is reconstructed (rather than imported from
    ``koa_screening.models``) because those factories close over the
    package-level ``RND`` constant; here we want full control.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    return {
        "Stepwise (Full)": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight=None, random_state=seed),
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=seed, eval_metric="logloss",
            ),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200, max_depth=10,
                random_state=seed, class_weight="balanced", n_jobs=-1,
            ),
        ),
        "Neural Network": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                alpha=0.0001, batch_size="auto", learning_rate_init=0.001,
                max_iter=500, random_state=seed, early_stopping=True,
            ),
        ),
    }


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: raw CSV not found at {RAW_CSV}", file=sys.stderr)
        return 1
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Loading data (canonical preprocessing) ...")
    df = data.load_and_prep_data(str(RAW_CSV), outdir=str(RESULTS_FINAL))
    df = df.sort_values("idelsa").reset_index(drop=True)
    y = df["oa_knee"].values
    groups = df["idelsa"].values

    all_cols = [c for c in df.columns if c not in BASE_EXCLUDE]
    feat_list = [c for c in all_cols if c not in SYMPTOM_VARS]
    X = df[feat_list]

    # MPMS features (so Stepwise uses the same input set as canonical)
    mpms_file = RESULTS_FINAL / "stepwise_mpms_clinical.csv"
    if mpms_file.exists():
        mpms_vars = pd.read_csv(mpms_file)["Variable"].tolist()
        valid_mpms = [v for v in mpms_vars if v in feat_list]
        X_stepwise = df[valid_mpms]
    else:
        print(f"WARN: {mpms_file} missing; using full feature set for Stepwise too")
        X_stepwise = X

    rows = []
    for seed in range(N_SEEDS):
        print(f"\n--- seed {seed} ---")
        pipes = _make_pipes(seed)
        for name, pipe in pipes.items():
            X_for_model = X_stepwise if name == "Stepwise (Full)" else X
            _, _, auc, _ = cv_roc_auc(pipe, X_for_model, y, groups)
            print(f"  {name:>32}: AUC={auc:.4f}")
            rows.append({"seed": seed, "Model": name, "AUC": auc})

    df_long = pd.DataFrame(rows)
    df_long.to_csv(OUTDIR / "per_seed_aucs.csv", index=False)

    summary = (
        df_long.groupby("Model")["AUC"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(4)
        .reset_index()
    )
    summary.columns = ["Model", "AUC_mean", "AUC_std", "AUC_min", "AUC_max", "n_seeds"]
    summary.to_csv(OUTDIR / "summary_mean_sd.csv", index=False)

    print("\n=== Seed-stability summary (seeds 0..{0}, Without Symptoms) ===".format(N_SEEDS - 1))
    print(summary.to_string(index=False))
    print(f"\nFiles in {OUTDIR}:")
    for p in sorted(OUTDIR.iterdir()):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
