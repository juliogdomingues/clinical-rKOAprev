
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

from . import data as p
from .config import RND, RAW_CSV, RESULTS_FINAL

CSV_PATH = str(RAW_CSV)
OUTDIR = str(RESULTS_FINAL)

def calculate_cv_permutation_importance(df, features, model_factory, target_col='oa_knee', group_col='idelsa', n_splits=5, n_repeats=5, random_state=42):
    """
    Calculates Permutation Importance using Cross-Validation.
    Returns DataFrame with [Feature, Importance_Mean, Importance_Std].
    """
    X = df[features]
    y = df[target_col].values
    groups = df[group_col].values
    
    cv = GroupKFold(n_splits=n_splits)
    importances = {f: [] for f in features}
    
    print(f"   Running CV Permutation Importance ({n_splits} folds, {n_repeats} repeats)...")
    
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        X_train, y_train = X.iloc[tr], y[tr]
        X_test, y_test = X.iloc[te], y[te]
        
        model = model_factory()
        model.fit(X_train, y_train)
        
        result = permutation_importance(
            model, X_test, y_test, 
            n_repeats=n_repeats, 
            random_state=random_state, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        for i, feature in enumerate(features):
            importances[feature].extend(result.importances[i])
            
    final_res = []
    for feat, imp_list in importances.items():
        final_res.append({
            'Feature': feat,
            'Importance_Mean': np.mean(imp_list),
            'Importance_Std': np.std(imp_list)
        })
        
    return pd.DataFrame(final_res).sort_values('Importance_Mean', ascending=False)

def make_logreg_unweighted():
    # Use fixed seed for reproducibility
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight=None, random_state=RND))

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    
    # 1. Load Data
    df = p.load_and_prep_data(CSV_PATH, outdir=OUTDIR)
    
    # 2. Define Models to Analyze
    # User requested MPMS model importance specifically.
    models_to_analyze = []
    
    # Check for MPMS features file
    mpms_file = os.path.join(OUTDIR, "mpms_features_for_ci.csv")
    if os.path.exists(mpms_file):
        mpms_feats = pd.read_csv(mpms_file)['feature'].tolist()
        models_to_analyze.append(("Clinical (MPMS)", mpms_feats, make_logreg_unweighted))
    else:
        print(f"Warning: {mpms_file} not found. Run the feature-selection step first.")

    # Check for Final 5-var features file (User requested plot for final model)
    final_file = os.path.join(OUTDIR, "final_5var_features_for_ci.csv")
    if os.path.exists(final_file):
        final_feats = pd.read_csv(final_file)['feature'].tolist()
        # Use unweighted as it's the primary final model choices
        models_to_analyze.append(("Final (5 vars)", final_feats, make_logreg_unweighted))
    else:
        print(f"Warning: {final_file} not found. Run the feature-selection step first.")

    # 3. Validation Loop
    for name, feats, factory in models_to_analyze:
        print(f"\nAnalyzing Importance for: {name}")
        
        # Valid features only
        valid_feats = [f for f in feats if f in df.columns]
        if len(valid_feats) < len(feats):
            print(f"   Warning: {len(feats) - len(valid_feats)} features missing in DF.")
            
        imp_df = calculate_cv_permutation_importance(
            df, valid_feats, factory, 
            n_splits=5, n_repeats=10, # Increased repeats for stability 
            random_state=RND
        )
        
        # Save CSV
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        csv_path = os.path.join(OUTDIR, f'permutation_importance_{safe_name}.csv')
        imp_df.to_csv(csv_path, index=False)
        print(f"   Saved CSV to {csv_path}")
        
        # Plot
        plt.figure(figsize=(10, 8))
        top_n = imp_df.head(20).copy()
        
        plt.barh(
            y=top_n['Feature'],
            width=top_n['Importance_Mean'],
            xerr=top_n['Importance_Std'].values,
            color=sns.color_palette("viridis", n_colors=len(top_n)),
            align='center',
            ecolor='black',
            capsize=3
        )
        plt.gca().invert_yaxis()
        plt.title(f'Permutation Importance (CV): {name}')
        plt.xlabel('Decrease in AUC score')
        plt.tight_layout()
        
        png_path = os.path.join(OUTDIR, f'fig_permutation_importance_{safe_name}.png')
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"   Saved Plot to {png_path}")

    print("\nAnalysis complete. Results in", OUTDIR)

if __name__ == "__main__":
    main()
