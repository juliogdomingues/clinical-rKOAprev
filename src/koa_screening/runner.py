"""End-to-end orchestration: 4 models x 3 scenarios.

This is the canonical comparison driver behind Table 1 / Figure 2 of the
manuscript. Extracted verbatim from the original
``run_comprehensive_comparison.py``; logic is unchanged (only the module
boundaries moved and config now comes from :mod:`koa_screening.config`).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from . import data as p
from .config import (
    BASE_EXCLUDE,
    LABEL_MAP,
    RAW_CSV,
    RESULTS_COMPARISON,
    RESULTS_FINAL,
    RND,
    SYMPTOM_VARS,
)
from .evaluation import cv_roc_auc
from .models import get_lr_pipe, get_pipeline


def run_stepwise_mpms(X, y, groups, mpms_features):
    start_features: list[str] = []
    results = []
    for k, feat in enumerate(mpms_features, 1):
        if feat not in X.columns:
            continue
        start_features.append(feat)
        X_sub = X[start_features]
        model = get_lr_pipe()
        _, _, auc, _ = cv_roc_auc(model, X_sub, y, groups)
        results.append({"k": k, "AUC": auc, "Added Variable": feat})
    return pd.DataFrame(results)


def plot_stepwise_detailed(df_results: pd.DataFrame, scenario_name: str, outdir: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(df_results["k"], df_results["AUC"], "-", color="gray", lw=2, alpha=0.5, zorder=1)

    cmap = matplotlib.colormaps["viridis"]
    num_steps = len(df_results)

    for i, row in df_results.iterrows():
        k = int(row["k"])
        auc = row["AUC"]
        color = cmap(i / max(1, num_steps - 1))
        ax.scatter(k, auc, s=250, facecolors=color, marker="o", edgecolors="none", linewidth=2, zorder=10)

        var_name = row["Added Variable"]
        var_display = LABEL_MAP.get(var_name, var_name)
        annot_text = f"{var_display}\nAUC: {auc:.3f}"

        if k % 2 == 1:
            xytext, va = (0, -60), "top"
        else:
            xytext, va = (0, 60), "bottom"

        ax.annotate(
            annot_text,
            (k, auc),
            xytext=xytext,
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9, lw=1),
            arrowprops=dict(arrowstyle="-", color="gray", lw=1.5, alpha=0.6),
        )

    ax.set_xlabel("Number of Variables (k)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Area Under the ROC Curve (AUC)", fontsize=16, fontweight="bold")
    ax.set_title(f"Incremental Gain (Stepwise Selection) - {scenario_name}", fontsize=20, fontweight="bold", pad=20)

    if not df_results.empty:
        ax.set_xticks(df_results["k"])
        min_auc = df_results["AUC"].min()
        max_auc = df_results["AUC"].max()
        pad = (max_auc - min_auc) * 0.4
        ax.set_ylim(min_auc - pad, max_auc + pad)

    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(str(outdir), f"stepwise_trajectory_{scenario_name.replace(' ', '_')}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Stepwise Plot: {plot_path}")


def calculate_odds_ratios(X, y, groups, features, scenario_name: str, outdir: str | Path) -> None:
    """Compute raw + standardized ORs with cluster-robust SEs (statsmodels Logit)."""
    print(f"   Calculating ORs for {len(features)} features in {scenario_name}...")
    valid_feats = [f for f in features if f in X.columns]
    X_sub = X[valid_feats].copy()
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_sub), columns=valid_feats, index=X_sub.index)

    X_raw = sm.add_constant(X_imputed)
    try:
        res_raw = sm.Logit(y, X_raw).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
        conf = res_raw.conf_int()
        conf["OR"] = res_raw.params
        conf.columns = ["2.5%", "97.5%", "OR"]
        res_df_raw = np.exp(conf)
        res_df_raw["P-value"] = res_raw.pvalues
        res_df_raw["Feature"] = res_df_raw.index
        res_df_raw = res_df_raw[res_df_raw["Feature"] != "const"]
        res_df_raw["OR (95% CI)"] = res_df_raw.apply(
            lambda x: f"{x['OR']:.2f} ({x['2.5%']:.2f}-{x['97.5%']:.2f})", axis=1
        )
        res_df_raw = res_df_raw[["Feature", "OR", "2.5%", "97.5%", "P-value", "OR (95% CI)"]]
        res_df_raw["Feature Label"] = res_df_raw["Feature"].map(LABEL_MAP).fillna(res_df_raw["Feature"])
        raw_path = os.path.join(str(outdir), f"or_raw_{scenario_name.replace(' ', '_')}.csv")
        res_df_raw.to_csv(raw_path, index=False)
        print(f"   Saved Raw ORs to {raw_path}")
    except Exception as e:
        print(f"   Error calculating Raw ORs: {e}")

    scaler = StandardScaler()
    X_std = pd.DataFrame(scaler.fit_transform(X_imputed), columns=valid_feats, index=X_imputed.index)
    X_std = sm.add_constant(X_std)
    try:
        res_std = sm.Logit(y, X_std).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
        conf = res_std.conf_int()
        conf["OR"] = res_std.params
        conf.columns = ["2.5%", "97.5%", "OR"]
        res_df_std = np.exp(conf)
        res_df_std["P-value"] = res_std.pvalues
        res_df_std["Feature"] = res_df_std.index
        res_df_std = res_df_std[res_df_std["Feature"] != "const"]
        res_df_std["OR (95% CI)"] = res_df_std.apply(
            lambda x: f"{x['OR']:.2f} ({x['2.5%']:.2f}-{x['97.5%']:.2f})", axis=1
        )
        res_df_std = res_df_std[["Feature", "OR", "2.5%", "97.5%", "P-value", "OR (95% CI)"]]
        res_df_std["Feature Label"] = res_df_std["Feature"].map(LABEL_MAP).fillna(res_df_std["Feature"])
        std_path = os.path.join(str(outdir), f"or_standardized_{scenario_name.replace(' ', '_')}.csv")
        res_df_std.to_csv(std_path, index=False)
        print(f"   Saved Standardized ORs to {std_path}")
    except Exception as e:
        print(f"   Error calculating Standardized ORs: {e}")


def get_feature_importance(model, X, y, model_name: str):
    try:
        model.fit(X, y)
        importances = None
        feature_names = X.columns
        estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        if model_name in {"Random Forest", "XGBoost"}:
            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
        elif model_name == "Neural Network":
            r = permutation_importance(model, X, y, n_repeats=5, random_state=RND, n_jobs=-1)
            importances = r.importances_mean
        if importances is not None:
            df_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
                "importance", ascending=False
            )
            df_imp["feature_label"] = df_imp["feature"].map(LABEL_MAP).fillna(df_imp["feature"])
            return df_imp
    except Exception as e:
        print(f"Error calculating importance for {model_name}: {e}")
    return None


def run_comparison(
    *,
    csv_path: str | Path | None = None,
    outdir: str | Path | None = None,
    mpms_file: str | Path | None = None,
    run_models: dict[str, bool] | None = None,
    run_virtual_max: bool = True,
    run_with_symptoms: bool = True,
    run_without_symptoms: bool = True,
    show_importance_plots: bool = True,
    show_stepwise_plot: bool = True,
    calculate_or: bool = True,
) -> pd.DataFrame:
    """The canonical comparison: 4 models x 3 scenarios with 5-fold OOF AUC.

    Returns the summary DataFrame and writes the full set of CSVs + PNGs to
    ``outdir`` (defaults to ``results/comparison``).
    """
    if csv_path is None:
        csv_path = RAW_CSV
    if outdir is None:
        outdir = RESULTS_COMPARISON
    if mpms_file is None:
        mpms_file = RESULTS_FINAL / "stepwise_mpms_clinical.csv"
    if run_models is None:
        run_models = {
            "Stepwise Logistic Regression": True,
            "XGBoost": True,
            "Random Forest": True,
            "Neural Network": True,
        }

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = p.load_and_prep_data(str(csv_path), outdir=str(RESULTS_FINAL))
    df = df.sort_values("idelsa").reset_index(drop=True)

    y = df["oa_knee"].values
    groups = df["idelsa"].values
    all_cols = [c for c in df.columns if c not in BASE_EXCLUDE]

    symptom_cols = [c for c in all_cols if c in SYMPTOM_VARS]
    cols_without_symptoms = [c for c in all_cols if c not in symptom_cols]
    cols_with_symptoms = list(all_cols)
    cols_virtual_max = list(all_cols)

    mpms_vars: list[str] = []
    if os.path.exists(str(mpms_file)):
        mpms_df = pd.read_csv(str(mpms_file))
        mpms_vars = mpms_df["Variable"].tolist()
    else:
        print(f"Warning: MPMS file not found at {mpms_file}. Skipping Stepwise.")
        run_models["Stepwise Logistic Regression"] = False

    scenarios: list[tuple[str, list[str]]] = []
    if run_virtual_max:
        scenarios.append(("Virtual Maximum", cols_virtual_max))
    if run_with_symptoms:
        scenarios.append(("With Symptoms", cols_with_symptoms))
    if run_without_symptoms:
        scenarios.append(("Without Symptoms", cols_without_symptoms))

    summary_results: list[dict] = []

    for scenario_name, feat_list in scenarios:
        print(f"\n--- Scenario: {scenario_name} ---")
        valid_feats = [f for f in feat_list if f in df.columns]
        X = df[valid_feats]
        print(f"Features in set: {len(valid_feats)}")

        roc_fig = plt.figure(figsize=(10, 8))
        ax = roc_fig.add_subplot(111)

        for model_name, enabled in run_models.items():
            if not enabled:
                continue
            plt.figure(roc_fig.number)

            if model_name == "Stepwise Logistic Regression":
                if scenario_name == "Virtual Maximum":
                    continue
                valid_mpms = [v for v in mpms_vars if v in valid_feats]
                if not valid_mpms:
                    continue
                step_res = run_stepwise_mpms(X, y, groups, valid_mpms)
                features_used = step_res["Added Variable"].tolist()
                X_mpms = X[features_used]
                fpr, tpr, auc, _ = cv_roc_auc(get_lr_pipe(), X_mpms, y, groups)
                ax.plot(
                    fpr,
                    tpr,
                    lw=2,
                    label=f"Stepwise Logistic Regression (Full Set k={len(features_used)}, AUC={auc:.3f})",
                )
                if show_stepwise_plot:
                    plot_stepwise_detailed(step_res, scenario_name, outdir)
                if calculate_or:
                    calculate_odds_ratios(X, y, groups, features_used, scenario_name, outdir)
                summary_results.append({"Scenario": scenario_name, "Model": "Stepwise (Full)", "AUC": auc})

            else:
                print(f"   Running {model_name}...")
                pipe = get_pipeline(model_name)
                fpr, tpr, auc, _ = cv_roc_auc(pipe, X, y, groups)
                plt.figure(roc_fig.number)
                ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={auc:.3f})")
                summary_results.append({"Scenario": scenario_name, "Model": model_name, "AUC": auc})

                if show_importance_plots:
                    imp_df = get_feature_importance(pipe, X, y, model_name)
                    if imp_df is not None:
                        imp_fig = plt.figure(figsize=(12, 8))
                        top_n = imp_df.head(15)
                        y_pos = np.arange(len(top_n))
                        labels = top_n["feature_label"].tolist()
                        plt.barh(y_pos, top_n["importance"], align="center")
                        plt.yticks(y_pos, labels)
                        plt.gca().invert_yaxis()
                        plt.xlabel("Importance")
                        plt.title(f"Feature Importance: {model_name} ({scenario_name})")
                        plt.tight_layout()
                        imp_path = outdir / f"importance_{model_name.replace(' ', '')}_{scenario_name.replace(' ', '_')}.png"
                        plt.savefig(imp_path)
                        plt.close(imp_fig)

        plt.figure(roc_fig.number)
        ax.plot([0, 1], [0, 1], "k--", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=14)
        ax.set_ylabel("True Positive Rate", fontsize=14)
        ax.set_title(f"ROC Curves: {scenario_name}", fontsize=16)
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, alpha=0.3)
        roc_path = outdir / f"roc_comparison_{scenario_name.replace(' ', '_')}.png"
        plt.savefig(roc_path, dpi=300)
        plt.close(roc_fig)
        print(f"Saved ROC: {roc_path}")

    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(outdir / "summary_all_models.csv", index=False)
    print("\nDone.")
    return summary_df
