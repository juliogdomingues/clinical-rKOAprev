
import pandas as pd
import numpy as np
from scipy import stats
import oarsi_data as p

def format_cont(series):
    return f"{series.mean():.1f} ± {series.std():.1f}"

def format_cat(series, target=1):
    count = (series == target).sum()
    pct = (count / len(series)) * 100
    return f"{count} ({pct:.1f}%)"

def main():
    # Load data
    df = p.load_and_prep_data("./base_stata/stataToCsvMG.csv")
    
    # Aggregate to Participant Level
    # Logic: 
    # - oa_knee: max (if any knee has it, the person has it)
    # - Continuous/Invariant vars (age, bmi, sex_female): first (should be same for both knees)
    # - History vars: max (if recorded as yes for the person/knee, we count it. 
    #   Based on oarsi_data.py these seem to be sourced from single variables per person anyway)
    
    agg_funcs = {
        'oa_knee': 'max',
        'age': 'first',
        'bmi': 'first',
        'sex_female': 'first',
        'frequent_symptoms': 'max',
        'history_trauma': 'max',
        'history_surgery': 'max'
    }
    
    part_df = df.groupby('idelsa').agg(agg_funcs).reset_index()
    
    # Define Groups
    no_koa = part_df[part_df['oa_knee'] == 0]
    yes_koa = part_df[part_df['oa_knee'] == 1]
    
    n_total = len(part_df)
    n_no = len(no_koa)
    n_yes = len(yes_koa)
    
    rows = []
    
    # Helper to add row
    def add_row(name, no_series, yes_series, type='cont', cat_val=1):
        full_series = pd.concat([no_series, yes_series])
        
        if type == 'cont':
            val_total = format_cont(full_series)
            val_no = format_cont(no_series)
            val_yes = format_cont(yes_series)
            
            # T-test (indep)
            t_stat, p_val = stats.ttest_ind(no_series.dropna(), yes_series.dropna(), equal_var=False)
            
        elif type == 'cat':
            val_total = format_cat(full_series, cat_val)
            val_no = format_cat(no_series, cat_val)
            val_yes = format_cat(yes_series, cat_val)
            
            # Chi-square
            table = [
                [(no_series == cat_val).sum(), (no_series != cat_val).sum()],
                [(yes_series == cat_val).sum(), (yes_series != cat_val).sum()]
            ]
            chi2, p_val, dof, ex = stats.chi2_contingency(table)
            
        p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        
        return f"| {name} | {val_total} | {val_no} | {val_yes} | {p_str} |"

    # Header
    print(f"| Characteristic | Total (N={n_total}) | No rKOA (N={n_no}) | With rKOA (N={n_yes}) | P-value |")
    print("|---|---|---|---|---|")
    
    # Variables
    # Age
    print(add_row("Age (years)", no_koa['age'], yes_koa['age'], 'cont'))
    
    # Sex (Female)
    print(add_row("Sex (Female)", no_koa['sex_female'], yes_koa['sex_female'], 'cat', 1)) 
    # Note: user asked for "Sex", reporting % Female is standard but I can label it "Female"
    
    # BMI
    print(add_row("BMI (kg/m²)", no_koa['bmi'], yes_koa['bmi'], 'cont'))
    
    # Frequent Knee Symptoms
    print(add_row("Frequent Knee Symptoms", no_koa['frequent_symptoms'], yes_koa['frequent_symptoms'], 'cat'))
    
    # History of Knee Trauma
    print(add_row("History of Knee Trauma", no_koa['history_trauma'], yes_koa['history_trauma'], 'cat'))
    
    # History of Knee Surgery
    print(add_row("History of Knee Surgery", no_koa['history_surgery'], yes_koa['history_surgery'], 'cat'))


if __name__ == "__main__":
    main()
