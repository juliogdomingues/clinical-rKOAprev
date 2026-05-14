import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier


def _try_make_xgb(scale_pos_weight: float, random_state: int):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError(
            "XGBoost não está disponível neste ambiente.\n"
            "Instale com: python -m pip install xgboost\n"
            f"Erro original: {e}"
        )

    return XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )


def _cv_eval_auc(pipe, X, y, groups, *, n_splits=5):
    cv = GroupKFold(n_splits=n_splits)
    aucs = []
    y_true_all = []
    y_pred_all = []

    for tr, te in cv.split(X, y, groups):
        pipe.fit(X.iloc[tr], y[tr])
        probs = pipe.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], probs))
        y_true_all.extend(y[te])
        y_pred_all.extend(probs)

    return float(np.mean(aucs)), float(np.std(aucs)), np.asarray(y_true_all), np.asarray(y_pred_all)


def main():
    ap = argparse.ArgumentParser(description="Run RandomForest and XGBoost using the same preprocessing as complete_finalatual.py.")
    ap.add_argument("--csv", default="./base_stata/stataToCsvMG.csv", help="Input CSV path.")
    ap.add_argument("--outdir", default="./results_tree_models", help="Output directory.")
    ap.add_argument("--feature-set", choices=["full", "clinical"], default="clinical", help="Which feature set to use.")
    ap.add_argument("--use-womac", action="store_true", help="Include WOMAC variables (default: excluded).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from common_preprocess import prepare_dataset

    prepared = prepare_dataset(args.csv, use_womac=args.use_womac, outdir=args.outdir)
    X = prepared.X_full if args.feature_set == "full" else prepared.X_clinical
    y = prepared.y
    groups = prepared.groups

    # Shared preprocessing for ALL models (same imputer/scaler)
    pre = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    rf_pipe = make_pipeline(pre, rf)

    # XGBoost (optional dependency)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    spw = (n_neg / max(n_pos, 1.0))
    try:
        xgb = _try_make_xgb(scale_pos_weight=spw, random_state=args.seed)
        xgb_pipe = make_pipeline(pre, xgb)
        have_xgb = True
    except RuntimeError as e:
        print(str(e))
        have_xgb = False

    results = []

    # Evaluate RF
    mean_auc, std_auc, y_true, y_pred = _cv_eval_auc(rf_pipe, X, y, groups)
    results.append({"Model": f"RandomForest({args.feature_set})", "AUC": mean_auc, "Std": std_auc, "n_features": int(X.shape[1])})
    _save_roc(args.outdir, f"roc_rf_{args.feature_set}", y_true, y_pred, mean_auc)

    # Evaluate XGB
    if have_xgb:
        mean_auc, std_auc, y_true, y_pred = _cv_eval_auc(xgb_pipe, X, y, groups)
        results.append({"Model": f"XGBoost({args.feature_set})", "AUC": mean_auc, "Std": std_auc, "n_features": int(X.shape[1])})
        _save_roc(args.outdir, f"roc_xgb_{args.feature_set}", y_true, y_pred, mean_auc)

    res_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
    res_df.to_csv(os.path.join(args.outdir, "tree_models_comparison.csv"), index=False)

    print("\n=== TREE MODELS RESULTS ===")
    print(res_df.to_string(index=False))
    print(f"\nArquivos salvos em: {args.outdir}")


def _save_roc(outdir: str, stem: str, y_true: np.ndarray, y_pred: np.ndarray, auc: float):
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title(stem)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{stem}.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()