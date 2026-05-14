import os
import time
import threading
import contextlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Fix for headless
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

# Import new modules
import oarsi_data as p
import oarsi_analysis as pa
import oarsi_utils as pu
import complete_finalatual as main_script # For configuration constants
import run_rf_xgb # For XGB helper

# Configuration
OUTDIR = main_script.OUTDIR
CSV_PATH = main_script.CSV_PATH
USE_WOMAC = main_script.USE_WOMAC
RND = main_script.RND


@contextlib.contextmanager
def progress_timer(label: str, every_s: float = 0.2, bar_width: int = 24):
    """
    TQDM-like *indeterminate* progress bar (no total available).
    Updates the same console line using carriage return.
    Note: if the wrapped function prints to stdout, the bar line will be interrupted,
    but it will resume updating afterwards.
    """
    stop = threading.Event()
    start = time.time()

    def _render(i: int, elapsed: int) -> str:
        # moving "block" inside an indeterminate bar
        pos = i % bar_width
        bar = [" "] * bar_width
        bar[pos] = "#"
        return f"\r[{label}] |{''.join(bar)}| {elapsed}s elapsed..."

    def _worker():
        i = 0
        last_len = 0
        while not stop.wait(every_s):
            elapsed = int(time.time() - start)
            s = _render(i, elapsed)
            # pad to clear remnants of previous longer line
            pad = max(0, last_len - len(s))
            sys.stdout.write(s + (" " * pad))
            sys.stdout.flush()
            last_len = len(s)
            i += 1

    t = threading.Thread(target=_worker, daemon=True)

    # start line
    sys.stdout.write(f"[{label}] started...\n")
    sys.stdout.flush()

    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=1.0)
        elapsed = int(time.time() - start)
        # print final line + newline
        sys.stdout.write(f"\r[{label}] done in {elapsed}s." + (" " * 10) + "\n")
        sys.stdout.flush()


def brier_ci_bootstrap_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap do Brier score reamostrando participantes (groups=idelsa) com reposição.
    Retorna: (brier_point, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    b_point = float(brier_score_loss(y_true, y_pred))

    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(groups, sampled)
        yt = y_true[mask]
        yp = y_pred[mask]
        # Brier não exige 2 classes, mas mantemos coerência
        if yt.size == 0:
            continue
        boot.append(brier_score_loss(yt, yp))

    boot = np.asarray(boot, dtype=float)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return b_point, lo, hi


def oof_predictions(df: pd.DataFrame, features: list[str], model_factory, *, n_splits: int = 5):
    """
    Generic OOF predictions using a model factory function.
    model_factory: callable that returns a fresh sklearn estimator (or pipeline)
    """
    y = df["oa_knee"].values
    groups = df["idelsa"].values
    cv = GroupKFold(n_splits=n_splits)

    aucs = []
    y_true_all, y_pred_all, groups_all = [], [], []

    for tr, te in cv.split(df, y, groups):
        model = model_factory() # Create fresh model
        model.fit(df.iloc[tr][features], y[tr])
        probs = model.predict_proba(df.iloc[te][features])[:, 1]
        aucs.append(roc_auc_score(y[te], probs))

        y_true_all.extend(y[te])
        y_pred_all.extend(probs)
        groups_all.extend(groups[te])

    return np.asarray(aucs, float), np.asarray(y_true_all), np.asarray(y_pred_all), np.asarray(groups_all)


def _brier_baseline_from_prevalence(p: float) -> float:
    # Baseline Brier for constant predictor = prevalence
    return float(p * (1.0 - p))


def _brier_skill(brier: float, brier_baseline: float) -> float:
    if brier_baseline <= 0:
        return float("nan")
    return float(1.0 - (brier / brier_baseline))


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) Data prep
    df = p.load_and_prep_data(CSV_PATH, outdir=OUTDIR)

    # 2) Define feature sets
    exclude_base = [
        "idelsa", "side", "kl", "oapf", "oa_knee",
        "kl_raw_num", "oapf_raw_num",
        "race_raw", "occupation", "smoking_status",
        "physical_activity_ipaq", "alcohol_use",
    ]
    womac_vars = ["womac_total", "womac_pain", "womac_stiffness", "womac_function"]
    exclude_womac = womac_vars if not USE_WOMAC else []

    exclude_complex = exclude_base + exclude_womac
    X_complex_cols = [c for c in df.columns if c not in exclude_complex]
    X_complex = df[X_complex_cols].dropna(thresh=len(df) * 0.5, axis=1)

    bio_vars = ["bone_mineral_content_kg", "mineral_mass_kg", "skeletal_muscle_mass_kg"]
    X_clinical_cols = [c for c in X_complex.columns if c not in bio_vars]
    X_clinical = df[X_clinical_cols].dropna(thresh=len(df) * 0.5, axis=1)
    
    print("   -> Input Features (Complex):", X_complex.shape[1])
    print("   -> Input Features (Clinical):", X_clinical.shape[1])

    # 3) Variable selection
    
    # ... (Selection code remains) ...

    # We will try to read the final feature file first to skip re-running heavy selection
    # if it exists and is recent. But user asked for analysis, so safe to re-run or rely on
    # the fact this script controls the pipeline.
    # Let's perform the selection steps as in the original script.
    
    with progress_timer("LASSO (full)", every_s=1.0):
        vars_full = pa.run_lasso(X_complex, df["oa_knee"], label="full_ci", outdir=OUTDIR)

    with progress_timer("LASSO (clinical)", every_s=1.0):
        vars_clin_lasso = pa.run_lasso(X_clinical, df["oa_knee"], label="clinical_ci", outdir=OUTDIR)

    with progress_timer("MPMS (clinical)", every_s=5.0):
        vars_clin_mpms = pa.run_mpms(X_clinical, df["oa_knee"], df["idelsa"], vars_clin_lasso)
        # NEW: Save MPMS variables for Permutation Importance script
        pd.Series(vars_clin_mpms, name="feature").to_csv(
             os.path.join(OUTDIR, "mpms_features_for_ci.csv"),
             index=False,
        )

    models_feats = {
        "1. Full (Lasso)": [f for f in vars_full if f in df.columns],
        "2. Clinical (Lasso)": [f for f in vars_clin_lasso if f in df.columns],
        "3. Clinical (MPMS)": [f for f in vars_clin_mpms if f in df.columns],
    }

    # Final 5 var model
    mpms_candidates = models_feats["3. Clinical (MPMS)"]
    final5_feats = []
    if mpms_candidates:
        with progress_timer("Stepwise (MPMS -> ordered vars)"):
            step_res = pa.run_stepwise_specific(df, "oa_knee", "idelsa", mpms_candidates)
        ordered = step_res["Variable"].tolist() if (step_res is not None and not step_res.empty) else []
        final5_feats = ordered[:5]
        
        # Save for calculating ORs
        pd.Series(final5_feats, name="feature").to_csv(
             os.path.join(OUTDIR, "final_5var_features_for_ci.csv"),
             index=False,
        )
        models_feats["4. Final (5 vars)"] = final5_feats

    # 4) Add RF and XGB to the definitions
    # They behave differently: they don't 'select' features in the same way,
    # they typically use the full clinical set.
    clinical_all = list(X_clinical.columns)
    
    # We will treat them as models that use "Clinical All" features but have specific model types
    # So we need to store (features, factory_name, factory_func)
    
    # Helper factories
    def make_logreg_balanced():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RND))
    
    def make_logreg_unweighted():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight=None, random_state=RND))
        
    def make_rf():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RandomForestClassifier(n_estimators=800, max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight="balanced_subsample", n_jobs=-1, random_state=RND)
        )
        
    # XGBoost setup
    n_pos = (df['oa_knee'] == 1).sum()
    n_neg = (df['oa_knee'] == 0).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    
    def make_xgb():
        try:
            return make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                run_rf_xgb._try_make_xgb(scale_pos_weight=spw, random_state=RND)
            )
        except:
            return None

    # Define all evaluation scenarios
    # Structure: Label -> (features, factory_func)
    scenarios = {}
    
    # Add LogReg models (Unweighted vs Weighted comparison)
    for label, feats in models_feats.items():
        if not feats: continue
        scenarios[f"{label} [LR]"] = (feats, make_logreg_unweighted)
        # Add Balanced version for key models to check user hypothesis
        if "MPMS" in label or "Final" in label:
            scenarios[f"{label} [LR Balanced]"] = (feats, make_logreg_balanced)
        
    # Add RF and XGB (using Clinical features)
    scenarios["5. Random Forest (Clinical)"] = (clinical_all, make_rf)
    scenarios["6. XGBoost (Clinical)"] = (clinical_all, make_xgb)

    # Save detailed feature lists to text file for user review
    with open(os.path.join(OUTDIR, "model_features_list.txt"), "w") as f:
        f.write("=== Model Feature Lists ===\n\n")
        f.write(f"Full Input Set ({len(X_complex.columns)} vars): {list(X_complex.columns)}\n\n")
        f.write(f"Clinical Input Set ({len(X_clinical.columns)} vars): {list(X_clinical.columns)}\n\n")
        for label, (feats, _) in scenarios.items():
            f.write(f"--- {label} ({len(feats)} vars) ---\n")
            f.write(f"{feats}\n\n")

    # 5) Evaluation Loop
    rows = []
    curves_data = [] # Store for secondary plotting
    plt.figure(figsize=(10, 8))
    
    print("\n[EVALUATION] Computing Bootstrap CIs (AUC & Brier) for all models...")
    
    for label, (feats, factory) in scenarios.items():
        # Check if model is valid (e.g. XGB might fail import)
        try:
            sample_model = factory()
            if sample_model is None:
                print(f"Skipping {label} (model not available)")
                continue
        except Exception as e:
            print(f"Skipping {label} (error creating model: {e})")
            continue
            
        print(f"Processing: {label} ({len(feats)} features)...")
        
        # Run OOF
        aucs, y_true, y_pred, groups_out = oof_predictions(df, feats, factory, n_splits=5)
        
        # Bootstrap AUC CI
        auc_point, auc_lo, auc_hi = pu.auc_ci_bootstrap_by_group(
            y_true=y_true, y_pred=y_pred, groups=groups_out, n_boot=5000, alpha=0.05, seed=RND
        )
        
        # Bootstrap Brier CI (Restored)
        brier_point, brier_lo, brier_hi = brier_ci_bootstrap_by_group(
            y_true=y_true, y_pred=y_pred, groups=groups_out, n_boot=5000, alpha=0.05, seed=RND
        )
        
        mean_fold_auc = np.mean(aucs)
        std_fold_auc = np.std(aucs)
        
        rows.append({
            "Model": label,
            "AUC": auc_point,
            "AUC_CI_Low": auc_lo,
            "AUC_CI_High": auc_hi,
            "Brier": brier_point,
            "Brier_CI_Low": brier_lo,
            "Brier_CI_High": brier_hi,
            "AUC_Mean_CV": mean_fold_auc,
            "AUC_Std_CV": std_fold_auc,
            "Features_Count": len(feats),
            "Features_List": str(feats) if len(feats) < 10 else f"See model_features_list.txt"
        })
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_point:.3f})")
        
        # Store ROC curve data for later plotting
        curves_data.append({
            "label": label,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_point
        })
        
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend(loc='lower right', fontsize=8)
    plt.title("ROC Comparison (OOF Predictions)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTDIR, "fig_roc_comparison_all_models.png"), dpi=300)
    plt.close()

    # --- NEW PLOT: Unweighted Only + Highlight Final ---
    plt.figure(figsize=(10, 8))
    
    # Filter for unweighted models and plot
    for curve_data in curves_data:
        label = curve_data["label"]
        fpr = curve_data["fpr"]
        tpr = curve_data["tpr"]
        auc = curve_data["auc"]

        # Only include unweighted Logistic Regression, RF, and XGB
        if "[LR Balanced]" in label:
            continue # Skip balanced models
        
        if "Final (5 vars) [LR]" in label:
            plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color='red', linewidth=3, alpha=0.9)
        elif "Random Forest" in label or "XGBoost" in label:
            plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color='steelblue', linewidth=1.5, alpha=0.7)
        else:
            plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color='grey', linewidth=1, alpha=0.6)

    plt.plot([0, 1], [0, 1], "k--") # Diagonal line
    plt.legend(loc='lower right', fontsize=8)
    plt.title("ROC Comparison (Unweighted Models, Final Highlighted)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTDIR, "fig_roc_comparison_unweighted_highlighted.png"), dpi=300)
    plt.close()

    # Save results
    res_df = pd.DataFrame(rows)
    # Sort by AUC
    # Sort by AUC
    res_df = res_df.sort_values("AUC", ascending=False)
    
    out_csv = os.path.join(OUTDIR, "final_all_models_auc_ci.csv")
    res_df.to_csv(out_csv, index=False)
    
    print("\n=== FINAL EVALUATION (AUC & Brier) ===")
    print_cols = ["Model", "AUC", "AUC_CI_Low", "AUC_CI_High", "Brier", "Features_Count"]
    print(res_df[print_cols].to_string(index=False))
    
    print(f"\nSaved results to: {out_csv}")
    print(f"Saved feature lists to: {os.path.join(OUTDIR, 'model_features_list.txt')}")
    print(f"Saved ROC plot to: {os.path.join(OUTDIR, 'fig_roc_comparison_all_models.png')}")


if __name__ == "__main__":
    main()