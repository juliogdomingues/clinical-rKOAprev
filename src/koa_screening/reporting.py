"""Audit-report writers for the data-prep and selection steps.

Pure side-effect helpers (prefixed ``_save_*``) that write the missingness,
outcome-exclusion, drop-reason, imputation-count, and LASSO-diagnostic CSVs/PNGs
under ``results/final_analysis/``. Called by ``data.load_and_prep_data`` and
``features.run_analysis``; they compute nothing the models consume.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _save_drop_reports(final_df: pd.DataFrame, *, id_col: str, outcome_cols: list[str], outdir: str) -> None:
    """
    Writes:
      - outcome_missingness_and_drops.csv
      - feature_missingness_and_imputation.csv
      - complete_case_drop_impact_by_feature.csv
    """
    os.makedirs(outdir, exist_ok=True)

    # --- Outcomes report ---
    outcome_rows = []
    n = int(len(final_df))
    for oc in outcome_cols:
        if oc not in final_df.columns:
            continue
        miss_mask = final_df[oc].isna()
        outcome_rows.append(
            {
                "variable": oc,
                "kind": "outcome",
                "n_rows_total": n,
                "n_rows_missing": int(miss_mask.sum()),
                "pct_rows_missing": (float(miss_mask.mean()) * 100.0) if n else 0.0,
                "n_participants_affected": int(final_df.loc[miss_mask, id_col].nunique()) if n else 0,
                "n_rows_dropped_due_to_rule": 0,  # rule handled separately below
                "drop_rule": "",
            }
        )

    # If your analysis rule is "drop row when BOTH kl and oapf missing"
    if all(c in final_df.columns for c in ["kl", "oapf"]):
        both_missing = final_df[["kl", "oapf"]].isna().all(axis=1)
        kl_only = final_df["kl"].isna() & final_df["oapf"].notna()
        oapf_only = final_df["oapf"].isna() & final_df["kl"].notna()

        outcome_rows.extend(
            [
                {
                    "variable": "kl&oapf",
                    "kind": "outcome_rule",
                    "n_rows_total": n,
                    "n_rows_missing": int(both_missing.sum()),
                    "pct_rows_missing": (float(both_missing.mean()) * 100.0) if n else 0.0,
                    "n_participants_affected": int(final_df.loc[both_missing, id_col].nunique()) if n else 0,
                    "n_rows_dropped_due_to_rule": int(both_missing.sum()),
                    "drop_rule": "drop if BOTH kl and oapf missing",
                },
                {
                    "variable": "kl_only_missing",
                    "kind": "outcome_pattern",
                    "n_rows_total": n,
                    "n_rows_missing": int(kl_only.sum()),
                    "pct_rows_missing": (float(kl_only.mean()) * 100.0) if n else 0.0,
                    "n_participants_affected": int(final_df.loc[kl_only, id_col].nunique()) if n else 0,
                    "n_rows_dropped_due_to_rule": 0,
                    "drop_rule": "",
                },
                {
                    "variable": "oapf_only_missing",
                    "kind": "outcome_pattern",
                    "n_rows_total": n,
                    "n_rows_missing": int(oapf_only.sum()),
                    "pct_rows_missing": (float(oapf_only.mean()) * 100.0) if n else 0.0,
                    "n_participants_affected": int(final_df.loc[oapf_only, id_col].nunique()) if n else 0,
                    "n_rows_dropped_due_to_rule": 0,
                    "drop_rule": "",
                },
            ]
        )

    pd.DataFrame(outcome_rows).to_csv(os.path.join(outdir, "outcome_missingness_and_drops.csv"), index=False)

    # --- Feature report ---
    exclude = set([id_col, "side"] + outcome_cols + ["oa_knee"])
    feature_cols = [c for c in final_df.columns if c not in exclude]

    feat_rows = []
    for c in feature_cols:
        miss_mask = final_df[c].isna()
        feat_rows.append(
            {
                "feature": c,
                "n_rows_total": n,
                "n_missing_cells_to_impute": int(miss_mask.sum()),
                "pct_missing_cells": (float(miss_mask.mean()) * 100.0) if n else 0.0,
                "n_participants_affected": int(final_df.loc[miss_mask, id_col].nunique()) if n else 0,
                "n_rows_dropped_due_to_feature_rule": 0,
                "note": "pipeline uses SimpleImputer(strategy='median'); no row drops for missing features",
            }
        )

    pd.DataFrame(feat_rows).sort_values(["n_missing_cells_to_impute", "feature"], ascending=[False, True]).to_csv(
        os.path.join(outdir, "feature_missingness_and_imputation.csv"),
        index=False,
    )

    # --- Optional: per-feature complete-case impact ---
    cc_rows = []
    for c in feature_cols:
        miss_mask = final_df[c].isna()
        cc_rows.append(
            {
                "feature": c,
                "n_rows_that_would_be_dropped_if_required": int(miss_mask.sum()),
                "pct_rows_that_would_be_dropped_if_required": (float(miss_mask.mean()) * 100.0) if n else 0.0,
                "n_participants_that_would_be_affected": int(final_df.loc[miss_mask, id_col].nunique()) if n else 0,
            }
        )

    pd.DataFrame(cc_rows).sort_values(
        ["n_rows_that_would_be_dropped_if_required", "feature"], ascending=[False, True]
    ).to_csv(os.path.join(outdir, "complete_case_drop_impact_by_feature.csv"), index=False)

def _save_outcome_exclusion_counts(
    *,
    df_before_rules: pd.DataFrame,
    df_after_drop6_before_recode: pd.DataFrame,
    outcome_missing_mask_after_recode: pd.Series,
    outdir: str,
    label: str = "outcome_exclusion_counts_mutually_exclusive",
) -> None:
    """
    Contagens MUTUAMENTE EXCLUSIVAS de joelhos excluídos.
    """
    os.makedirs(outdir, exist_ok=True)

    # --- Raw numeric from BEFORE rules (for code-6 drop) ---
    kl_raw_before = pd.to_numeric(df_before_rules.get("kl", np.nan), errors="coerce")
    oapf_raw_before = pd.to_numeric(df_before_rules.get("oapf", np.nan), errors="coerce")

    drop6_mask = (kl_raw_before == 6) | (oapf_raw_before == 6)
    n_drop_6 = int(drop6_mask.sum())

    # --- Among rows present AFTER drop6 (pre-recode), attribute the ones dropped by BOTH missing (post-recode) ---
    if "kl_raw_num" in df_after_drop6_before_recode.columns:
        kl_raw = pd.to_numeric(df_after_drop6_before_recode["kl_raw_num"], errors="coerce")
    else:
        kl_raw = pd.to_numeric(df_after_drop6_before_recode.get("kl", np.nan), errors="coerce")

    if "oapf_raw_num" in df_after_drop6_before_recode.columns:
        oapf_raw = pd.to_numeric(df_after_drop6_before_recode["oapf_raw_num"], errors="coerce")
    else:
        oapf_raw = pd.to_numeric(df_after_drop6_before_recode.get("oapf", np.nan), errors="coerce")

    dropped_by_both_missing = df_after_drop6_before_recode.loc[outcome_missing_mask_after_recode].copy()
    idx = dropped_by_both_missing.index

    kl_d = kl_raw.loc[idx]
    oapf_d = oapf_raw.loc[idx]

    raw_both_missing = kl_d.isna() & oapf_d.isna()
    has5 = (kl_d == 5) | (oapf_d == 5)
    has7 = (kl_d == 7) | (oapf_d == 7)
    has8 = (kl_d == 8) | (oapf_d == 8)
    has9 = (kl_d == 9) | (oapf_d == 9)

    # Mutually exclusive primary reason
    primary = np.select(
        [
            raw_both_missing,
            (~raw_both_missing) & has5,
            (~raw_both_missing) & (~has5) & has7,
            (~raw_both_missing) & (~has5) & (~has7) & has8,
            (~raw_both_missing) & (~has5) & (~has7) & (~has8) & has9,
        ],
        ["missing", "5", "7", "8", "9"],
        default="other",
    )
    primary_counts = (
        pd.Series(primary)
        .value_counts()
        .reindex(["missing", "5", "7", "8", "9", "other"], fill_value=0)
        .rename_axis("reason")
        .reset_index(name="n_knees")
    )

    out = pd.concat(
        [
            pd.DataFrame([{"reason": "6", "n_knees": n_drop_6}]),
            primary_counts,
        ],
        ignore_index=True,
    )

    out.to_csv(os.path.join(outdir, f"{label}.csv"), index=False)

    # Console output
    n_drop_both_missing = int(outcome_missing_mask_after_recode.sum())
    print("\n[OUTCOME EXCLUSIONS - MUTUALLY EXCLUSIVE]")
    print(f"  Excluídos por 6 (artroplastia em KL ou OAPF): {n_drop_6}")
    print(f"  Excluídos por BOTH missing após recode:      {n_drop_both_missing}")
    for r in ["missing", "5", "7", "8", "9", "other"]:
        n = int(primary_counts.loc[primary_counts["reason"] == r, "n_knees"].iloc[0])
        print(f"    - motivo={r:>7}: {n}")

    print(f"  CSV salvo em: {os.path.join(outdir, f'{label}.csv')}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    order = ["6", "missing", "5", "7", "8", "9", "other"]
    plot_df = out.set_index("reason").reindex(order).reset_index()
    ax.bar(plot_df["reason"], plot_df["n_knees"])
    ax.set_title("Outcome exclusions (mutually exclusive)")
    ax.set_xlabel("Reason")
    ax.set_ylabel("Count (knees)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{label}.png"), dpi=300)
    plt.close(fig)

def _save_missing_outcome_ids(df: pd.DataFrame, outdir: str, label: str) -> None:
    """
    Saves the IDs (idelsa) and knee-rows (idelsa+side) that have missing outcomes.
    """
    os.makedirs(outdir, exist_ok=True)

    required = {"idelsa", "side", "kl", "oapf"}
    if not required.issubset(df.columns):
        return

    kl_num = pd.to_numeric(df["kl"], errors="coerce")
    oapf_num = pd.to_numeric(df["oapf"], errors="coerce")

    both_missing = kl_num.isna() & oapf_num.isna()
    kl_only_missing = kl_num.isna() & oapf_num.notna()
    oapf_only_missing = oapf_num.isna() & kl_num.notna()

    rows = df.loc[both_missing | kl_only_missing | oapf_only_missing, ["idelsa", "side", "kl", "oapf"]].copy()
    rows["missing_pattern"] = np.select(
        [both_missing.loc[rows.index], kl_only_missing.loc[rows.index], oapf_only_missing.loc[rows.index]],
        ["both_missing", "kl_missing_only", "oapf_missing_only"],
        default="unknown",
    )
    rows.to_csv(os.path.join(outdir, f"missing_outcomes_{label}_rows.csv"), index=False)

    per_id = (
        rows.groupby("idelsa")
        .agg(
            n_knees_with_any_missing=("side", "count"),
            n_knees_both_missing=("missing_pattern", lambda s: int((s == "both_missing").sum())),
        )
        .reset_index()
        .sort_values(["n_knees_both_missing", "n_knees_with_any_missing"], ascending=False)
    )

    per_id["would_lose_both_knees_due_to_both_missing"] = per_id["n_knees_both_missing"].ge(2)
    per_id.to_csv(os.path.join(outdir, f"missing_outcomes_{label}_ids.csv"), index=False)

def _save_imputation_counts(X: pd.DataFrame, label: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    out = (
        X.isna().sum()
        .rename("n_imputed")
        .to_frame()
        .assign(pct_imputed=lambda d: (d["n_imputed"] / len(X)) * 100.0)
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values(["n_imputed", "feature"], ascending=[False, True])
    )
    out.to_csv(os.path.join(outdir, f"imputation_counts_{label}.csv"), index=False)

def _save_lasso_diagnostics(model, X, label, outdir):
    os.makedirs(outdir, exist_ok=True)

    coefs = pd.Series(model.coef_[0], index=X.columns)
    coef_df = (
        coefs.rename('coef')
        .to_frame()
        .assign(abs_coef=lambda d: d['coef'].abs())
        .assign(is_zero=lambda d: np.isclose(d['coef'].values, 0.0))
        .sort_values('abs_coef', ascending=False)
        .reset_index()
        .rename(columns={'index': 'feature'})
    )
    coef_df.to_csv(os.path.join(outdir, f'lasso_coefficients_{label}.csv'), index=False)

    zero_df = coef_df[coef_df['is_zero']].copy()
    zero_df.to_csv(os.path.join(outdir, f'lasso_zeroed_features_{label}.csv'), index=False)

    # CV scores per C (roc_auc).
    try:
        score_key = next(iter(model.scores_.keys()))
        scores = model.scores_[score_key]  # shape: (n_folds, n_Cs)
        cv_df = pd.DataFrame(
            {
                'C': model.Cs_,
                'mean_cv_auc': scores.mean(axis=0),
                'std_cv_auc': scores.std(axis=0),
            }
        )
        best_c = float(getattr(model, 'C_', [np.nan])[0])
        cv_df['is_selected_C'] = np.isclose(cv_df['C'].values.astype(float), best_c)
        cv_df.to_csv(os.path.join(outdir, f'lasso_cv_by_C_{label}.csv'), index=False)
    except Exception:
        pass
