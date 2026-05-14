import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import fisher_exact, pearsonr, spearmanr
import oarsi_data

def run_investigation():
    outdir = './results_final_analysis'
    os.makedirs(outdir, exist_ok=True)
    
    # 1. Load Data
    print("Loading data...")
    df = oarsi_data.load_and_prep_data('./base_stata/stataToCsvMG.csv', outdir)
    
    if 'sex_female' not in df.columns:
        print("ERROR: 'sex_female' column not found in dataset!")
        return

    # 2. Descriptive Stats (Univariate)
    print("\n--- 2. Univariate Analysis (Sex vs OA) ---")
    
    # Contingency Table
    ct = pd.crosstab(df['sex_female'], df['oa_knee'])
    ct.index = ['Male (0)', 'Female (1)']
    ct.columns = ['No OA (0)', 'OA (1)']
    print("\nContingency Table:")
    print(ct)
    
    # Prevalence
    prev_m = ct.loc['Male (0)', 'OA (1)'] / ct.loc['Male (0)'].sum()
    prev_f = ct.loc['Female (1)', 'OA (1)'] / ct.loc['Female (1)'].sum()
    print(f"\nPrevalence in Males:   {prev_m:.2%}")
    print(f"Prevalence in Females: {prev_f:.2%}")
    
    # Odds Ratio
    odds, pvalue = fisher_exact(ct.loc[['Male (0)', 'Female (1)']].values) # Note: order matters for OR interpretation
    # Correction: fisher_exact expects [[a, b], [c, d]]. 
    # M_No, M_Yes
    # F_No, F_Yes
    # We want OR for Female (row 2) vs Male (row 1).
    # OR = (F_Yes / F_No) / (M_Yes / M_No)
    
    a = ct.loc['Female (1)', 'OA (1)']
    b = ct.loc['Female (1)', 'No OA (0)']
    c = ct.loc['Male (0)', 'OA (1)']
    d = ct.loc['Male (0)', 'No OA (0)']
    
    or_val = (a/b) / (c/d)
    print(f"\nOdds Ratio (Female vs Male): {or_val:.4f}")
    # Crude check
    
    # 3. Correlations with Top Features
    print("\n--- 3. Correlations with Model Features ---")
    top_features = ['age', 'bmi', 'history_surgery', 'frequent_symptoms', 'history_trauma']
    valid_top = [f for f in top_features if f in df.columns]
    
    corr_data = []
    for feat in valid_top:
        # Drop NaNs for correlation
        tmp = df[['sex_female', feat]].dropna()
        if len(tmp) < 10:
            continue
        corr, _ = spearmanr(tmp['sex_female'], tmp[feat])
        corr_data.append({'Feature': feat, 'Spearman_r_with_Sex': corr})
        
    print(pd.DataFrame(corr_data))
    
    # 4. LASSO Path Analysis
    print("\n--- 4. LASSO Path Analysis ---")
    
    # Prepare Clinical Dataset (same as main analysis)
    # Replicate main exclusion logic briefly
    exclude_base = [
        'idelsa', 'side', 'kl', 'oapf', 'oa_knee',
        'kl_raw_num', 'oapf_raw_num',
        'race_raw', 'occupation', 'smoking_status',
        'physical_activity_ipaq', 'alcohol_use',
        'bone_mineral_content_kg', 'mineral_mass_kg', 'skeletal_muscle_mass_kg' # Clinical exclusion
    ]
    cols = [c for c in df.columns if c not in exclude_base]
    # Drop sparse
    X = df[cols].dropna(thresh=len(df)*0.5, axis=1)
    y = df['oa_knee']
    
    # Impute & Scale
    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_filled = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    X_scaled = pd.DataFrame(scaler.fit_transform(X_filled), columns=X.columns)
    
    print(f"Running LASSO path on {X_scaled.shape[1]} features...")
    
    # We use LogisticRegression with l1 penalty over a range of C (inverse lambda)
    # Small C = High Regularization (Coefficients -> 0)
    # Large C = Low Regularization
    Cs = np.logspace(-4, 4, 10) # REDUCED TO 10
    
    coefs_list = []
    
    for i, C in enumerate(Cs):
        print(f"   Fitting C={C:.4f} ({i+1}/{len(Cs)})...")
        lr = LogisticRegression(penalty='l1', C=C, solver='liblinear', 
                                max_iter=1000, random_state=42, class_weight=None)
        lr.fit(X_scaled, y)
        coefs_list.append(lr.coef_[0])

    coefs_arr = np.array(coefs_list)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Identify index of 'sex_female'
    try:
        sex_idx = list(X.columns).index('sex_female')
        sex_col_valid = True
    except ValueError:
        sex_col_valid = False
        print("Warning: sex_female not in final X columns (maybe dropped due to missingness?)")
        
    # Plot all lines grey first
    for i in range(coefs_arr.shape[1]):
        plt.plot(np.log10(Cs), coefs_arr[:, i], color='lightgrey', alpha=0.5, linewidth=1)
        
    # Highlight Sex
    if sex_col_valid:
        plt.plot(np.log10(Cs), coefs_arr[:, sex_idx], color='red', linewidth=3, label='sex_female')
        
    # Highlight Top Features
    colors = ['blue', 'green', 'purple', 'orange', 'brown']
    for i, feat in enumerate(valid_top):
        if feat in X.columns:
            idx = list(X.columns).index(feat)
            plt.plot(np.log10(Cs), coefs_arr[:, idx], color=colors[i%len(colors)], linewidth=2, label=feat)
            
    plt.xlabel('log10(C)  [<- Strong Reg. | Weak Reg. ->]')
    plt.ylabel('Coefficient Value')
    plt.title('LASSO Path: Evolution of Coefficients vs Regularization Strength')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(outdir, 'lasso_path_sex_focus.png')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    
    # Check max coefficient for sex
    if sex_col_valid:
        max_sex_coef = np.max(np.abs(coefs_arr[:, sex_idx]))
        print(f"\nMax absolute coefficient for sex_female across all C: {max_sex_coef:.4f}")
        if max_sex_coef == 0:
            print("CONFIRMED: Sex variable is perfectly zeroed out even at weak regularization?")
        else:
            print("Note: Sex variable appears when regularization is weak (high C), but likely disappears at the optimal C chosen by CV.")

if __name__ == "__main__":
    run_investigation()
