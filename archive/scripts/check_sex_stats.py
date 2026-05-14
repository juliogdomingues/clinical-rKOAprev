import os
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, spearmanr
import oarsi_data

def run_stats():
    outdir = './results_final_analysis'
    os.makedirs(outdir, exist_ok=True)
    
    print("Loading data...")
    df = oarsi_data.load_and_prep_data('./base_stata/stataToCsvMG.csv', outdir)
    
    if 'sex_female' not in df.columns:
        print("ERROR: 'sex_female' column not found!")
        return

    # 2. Descriptive Stats (Univariate)
    print("\n--- 2. Univariate Analysis (Sex vs OA) ---")
    
    ct = pd.crosstab(df['sex_female'], df['oa_knee'])
    ct.index = ['Male (0)', 'Female (1)']
    ct.columns = ['No OA (0)', 'OA (1)']
    print("\nContingency Table:")
    print(ct)
    
    prev_m = ct.loc['Male (0)', 'OA (1)'] / ct.loc['Male (0)'].sum()
    prev_f = ct.loc['Female (1)', 'OA (1)'] / ct.loc['Female (1)'].sum()
    print(f"\nPrevalence in Males:   {prev_m:.2%}")
    print(f"Prevalence in Females: {prev_f:.2%}")
    
    # OR
    a = ct.loc['Female (1)', 'OA (1)']
    b = ct.loc['Female (1)', 'No OA (0)']
    c = ct.loc['Male (0)', 'OA (1)']
    d = ct.loc['Male (0)', 'No OA (0)']
    
    or_val = (a/b) / (c/d)
    print(f"\nOdds Ratio (Female vs Male): {or_val:.4f}")

    # 3. Correlations with Model Features (MPMS Ranked)
    print("\n--- 3. Correlations with Sex (Ranked by Redundancy) ---")
    
    mpms_path = './results_final_analysis/stepwise_mpms_clinical.csv'
    if os.path.exists(mpms_path):
        mpms_df = pd.read_csv(mpms_path)
        # Assuming 'Variable' is the column name based on oarsi_analysis.py
        target_features = mpms_df['Variable'].tolist()
    else:
        print("Warning: Stepwise MPMS results not found. Using default top features.")
        target_features = ['age', 'bmi', 'history_surgery', 'frequent_symptoms', 'history_trauma']
    
    valid_features = [f for f in target_features if f in df.columns]
    
    corr_data = []
    for feat in valid_features:
        tmp = df[['sex_female', feat]].dropna()
        if len(tmp) < 10: continue
        
        # Calculate Spearman correlation
        corr, p = spearmanr(tmp['sex_female'], tmp[feat])
        
        # We care about magnitude for redundancy
        corr_data.append({
            'Feature': feat, 
            'Correlation_with_Sex': corr,
            'Abs_Correlation': abs(corr),
            'P_Value': p
        })
        
    res_df = pd.DataFrame(corr_data).sort_values('Abs_Correlation', ascending=False)
    print(res_df.drop(columns=['Abs_Correlation']).to_string(index=False))

if __name__ == "__main__":
    run_stats()
