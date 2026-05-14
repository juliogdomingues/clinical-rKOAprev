import os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')

# Import project modules
import oarsi_data as p

# Configuration
CSV_PATH = "./base_stata/stataToCsvMG.csv"
OUTDIR = './results_final_analysis'
RND = 42
N_BOOT = 2000
ALPHA = 0.05

def bootstrap_or_ci_raw(df, features, target_col='oa_knee', group_col='idelsa', n_boot=2000, alpha=0.05, seed=42):
    """
    Cluster Bootstrap for RAW (Unstandardized) Odds Ratios.
    """
    print(f"Starting Cluster Bootstrap for RAW ORs ({n_boot} iterations)...")
    rng = np.random.default_rng(seed)
    groups = df[group_col].values
    uniq_groups = np.unique(groups)
    
    boot_coefs = []
    
    # NO SCALER for Raw ORs
    # Note: If variables have different scales, convergence might be slower, but for interpretation we want raw units.
    pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        # StandardScaler(),  <-- REMOVED
        LogisticRegression(max_iter=5000, class_weight=None) # Increased iter for safety
    )
    
    for i in range(n_boot):
        if (i + 1) % 100 == 0:
            print(f"   Iteration {i + 1}/{n_boot}...", end='\r')
            
        sampled_groups = rng.choice(uniq_groups, size=len(uniq_groups), replace=True)
        sampled_groups_df = pd.DataFrame({group_col: sampled_groups})
        boot_df = sampled_groups_df.merge(df, on=group_col, how='left')
        
        X_boot = boot_df[features]
        y_boot = boot_df[target_col]
        
        if y_boot.nunique() < 2:
            continue
            
        pipe.fit(X_boot, y_boot)
        model = pipe.named_steps['logisticregression']
        boot_coefs.append(model.coef_[0])
        
    print("\nBootstrap complete.")
    
    boot_coefs = np.array(boot_coefs)
    
    results = []
    for idx, feature in enumerate(features):
        coef_vals = boot_coefs[:, idx]
        or_vals = np.exp(coef_vals)
        or_median = np.median(or_vals)
        ci_low = np.percentile(or_vals, 100 * (alpha / 2))
        ci_high = np.percentile(or_vals, 100 * (1 - alpha / 2))
        
        results.append({
            'Feature': feature,
            'd_OR_Raw': or_median, # d_ prefix for handy sorting later if needed
            'd_OR_Raw_Low': ci_low,
            'd_OR_Raw_High': ci_high
        })
        
    return pd.DataFrame(results)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    
    # 1. Load Data
    df = p.load_and_prep_data(CSV_PATH, outdir=OUTDIR)
    
    # 2. Load Final Features
    feature_file = os.path.join(OUTDIR, "final_5var_features_for_ci.csv")
    if os.path.exists(feature_file):
        feats_df = pd.read_csv(feature_file)
        final_features = feats_df.iloc[:, 0].tolist()
    else:
        print("Feature file not found. Exiting.")
        return

    print(f"Features: {final_features}")
    
    # 3. Calculate RAW ORs
    or_df = bootstrap_or_ci_raw(
        df, 
        final_features, 
        target_col='oa_knee', 
        group_col='idelsa', 
        n_boot=N_BOOT, 
        alpha=ALPHA, 
        seed=RND
    )
    
    out_path = os.path.join(OUTDIR, 'final_model_or_raw_ci.csv')
    or_df.to_csv(out_path, index=False)
    
    print("\n=== RAW ODDS RATIOS (Unstandardized) ===")
    print_df = or_df.copy()
    print_df['OR_Raw (95% CI)'] = print_df.apply(
        lambda x: f"{x['d_OR_Raw']:.2f} ({x['d_OR_Raw_Low']:.2f}-{x['d_OR_Raw_High']:.2f})", 
        axis=1
    )
    print(print_df[['Feature', 'OR_Raw (95% CI)']].to_string(index=False))

if __name__ == "__main__":
    main()
