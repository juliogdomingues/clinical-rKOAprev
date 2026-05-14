
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score

from . import data as p
from .config import RND, RAW_CSV, RESULTS_FINAL

OUTDIR = str(RESULTS_FINAL)

def get_lr_pipe():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight=None, random_state=RND)
    )

def get_rf_model():
    return RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RND, class_weight='balanced')

def get_xgb_model():
    return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RND, eval_metric='logloss')

def cv_roc(model, X, y, groups):
    cv = GroupKFold(n_splits=5)
    y_true_all = []
    y_pred_all = []
    
    for tr, te in cv.split(X, y, groups):
        if isinstance(model, matplotlib.pyplot.Figure): pass 
             
        model.fit(X.iloc[tr], y[tr])
        
        if hasattr(model, "predict_proba"):
             probs = model.predict_proba(X.iloc[te])[:, 1]
        else:
             probs = model.predict(X.iloc[te])
             
        y_true_all.extend(y[te])
        y_pred_all.extend(probs)
        
    # Calculate Pooled AUC to match the original analysis (0.815)
    pooled_auc = roc_auc_score(y_true_all, y_pred_all)
    fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
    return fpr, tpr, pooled_auc

def main():
    print("Loading data...")
    df = p.load_and_prep_data(str(RAW_CSV), outdir=OUTDIR)
    
    # Sort by ID to ensure GroupKFold deterministic behavior matches original analysis (AUC 0.815)
    df = df.sort_values('idelsa').reset_index(drop=True)
    
    y = df['oa_knee'].values
    groups = df['idelsa'].values
    
    # 1. Load Feature Sets
    print("Loading feature sets...")
    lasso_full = pd.read_csv(os.path.join(OUTDIR, 'lasso_coefficients_full.csv'))
    feats_full = lasso_full[~lasso_full['is_zero']]['feature'].tolist()
    
    lasso_clin = pd.read_csv(os.path.join(OUTDIR, 'lasso_coefficients_clinical.csv'))
    feats_clin = lasso_clin[~lasso_clin['is_zero']]['feature'].tolist()
    
    mpms_df = pd.read_csv(os.path.join(OUTDIR, 'stepwise_mpms_clinical.csv'))
    feats_mpms_all = mpms_df['Variable'].tolist() # Ordered list 1..9
    feats_mpms_best = feats_mpms_all[:9] # According to table, 9 vars selected
    feats_final_5 = feats_mpms_all[:5]
    
    # Define Inputs for RF/XGB (Clinical Input Set)
    exclude_base = [
        'idelsa', 'side', 'kl', 'oapf', 'oa_knee',
        'kl_raw_num', 'oapf_raw_num',
        'race_raw', 'occupation', 'smoking_status',
        'physical_activity_ipaq', 'alcohol_use'
    ]
    bio_vars = ['bone_mineral_content_kg', 'mineral_mass_kg', 'skeletal_muscle_mass_kg']
    X_cols = [c for c in df.columns if c not in exclude_base + bio_vars]
    X_clinical_input = df[X_cols].dropna(thresh=len(df)*0.5, axis=1) # Same filter as analysis
    feats_clinical_input = X_clinical_input.columns.tolist()

    # =========================================================================
    # COMBINED FIGURE: Panel A (6 Models) | Panel B (Incremental Gain)
    # =========================================================================
    print("\nGenerating Combined Abstract Figure...")
    
    # Even wider figure to allow Panel B to stretch
    fig = plt.figure(figsize=(28, 10)) 
    # Make Panel B significantly wider (almost 2x Panel A) to be rectangular
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.8], wspace=0.15) 
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    
    # --- PANEL A: 6 MODELS ROC ---
    print("   Panel A: 6 Models ROC...")
    models = {
        '1. Full (Lasso)': (get_lr_pipe(), feats_full),
        '2. Clinical (Lasso)': (get_lr_pipe(), feats_clin),
        '3. Clinical (MPMS)': (get_lr_pipe(), feats_mpms_best),
        '4. Final (5 vars)': (get_lr_pipe(), feats_final_5),
        '5. Random Forest': (make_pipeline(SimpleImputer(strategy='median'), get_rf_model()), feats_clinical_input),
        '6. XGBoost': (make_pipeline(SimpleImputer(strategy='median'), get_xgb_model()), feats_clinical_input)
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (name, (model, feats)) in enumerate(models.items()):
        print(f"      Running {name}...")
        valid_feats = [f for f in feats if f in df.columns]
        X_sub = df[valid_feats]
        fpr, tpr, auc = cv_roc(model, X_sub, y, groups)
        
        lw = 3 
        z = 5
        if 'Final' in name:
            lw = 5 
            z = 10
            
        ax0.plot(fpr, tpr, color=colors[i], lw=lw, label=f'{name} (AUC = {auc:.3f})', zorder=z)

    ax0.plot([0, 1], [0, 1], 'k--', lw=2)
    ax0.set_xlim([0.0, 1.0])
    ax0.set_ylim([0.0, 1.05])
    
    # BIG FONTS
    ax0.set_xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    ax0.set_ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    ax0.set_title('A. ROC Curve Comparison (6 Models)', fontsize=22, fontweight='bold', pad=15)
    ax0.tick_params(axis='both', which='major', labelsize=16)
    
    # LEGEND: "Even larger" -> 18
    ax0.legend(loc="lower right", fontsize=18, framealpha=0.95, edgecolor='black', borderpad=0.8)
    ax0.grid(True, alpha=0.3)
    
    
    # --- PANEL B: INCREMENTAL GAIN ---
    print("   Panel B: Incremental Gain...")
    
    cmap = matplotlib.colormaps['viridis']
    gains = []
    
    # Pre-calculate data
    print("      Calculating steps...")
    
    num_steps = min(9, len(feats_mpms_all))
    step_colors = []
    for k_idx in range(num_steps):
        # Use Viridis gradient for ALL steps
        c = cmap(k_idx / max(1, num_steps)) 
        step_colors.append(c)
        
    for k_idx in range(num_steps):
        k = k_idx + 1
        curr_feats = feats_mpms_all[:k]
        valid_feats = [f for f in curr_feats if f in df.columns]
        X_sub = df[valid_feats]
        
        fpr, tpr, auc = cv_roc(get_lr_pipe(), X_sub, y, groups)
        gains.append({'k': k, 'AUC': auc, 'var': feats_mpms_all[k-1]})
    
    gain_df = pd.DataFrame(gains)
    
    # Plot connecting line
    ax1.plot(gain_df['k'], gain_df['AUC'], '-', color='gray', lw=2, alpha=0.5, zorder=1)
    
    # Plot points
    for i, row in gain_df.iterrows():
        k = int(row['k'])
        auc = row['AUC']
        color = step_colors[i]
        
        is_selected = (k <= 5)
        
        size = 300 
        marker = 'o'
        
        if is_selected:
            facecolors = color
            edgecolors = 'none'
            alpha = 1.0
        else:
            facecolors = 'none'
            edgecolors = color
            alpha = 0.7
            
        ax1.scatter(k, auc, s=size, facecolors=facecolors, marker=marker, edgecolors=edgecolors, linewidth=3, zorder=10)
        
        # Annotation - Bigger and Explicit AUC
        var_name = row['var']
        var_display = var_name 
        
        # Increase offset to avoid overlap
        offset_val = 65 # Increased to separate labels
        if k % 2 == 1: 
             xytext = (0, -offset_val) 
             va = 'top'
        else: 
             xytext = (0, offset_val)
             va = 'bottom'
        
        fw = 'bold' if is_selected else 'normal'
        fs = 14 if is_selected else 12 
        
        # Explicit AUC caption
        annot_text = f"{var_display}\nAUC: {auc:.3f}"
             
        ax1.annotate(
            annot_text, 
            (k, auc),
            xytext=xytext, 
            textcoords='offset points',
            rotation=0,
            ha='center', va=va, 
            fontsize=fs, fontweight=fw,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9, lw=1),
            arrowprops=dict(arrowstyle='-', color='gray', lw=1.5, alpha=0.6)
        )
    
    # Add vertical line separating selected
    ax1.axvline(x=5.5, color='black', linestyle='--', alpha=0.5, lw=2)
    ax1.text(5.5, min(gain_df['AUC']), ' Final Model Cutoff ', rotation=90, va='bottom', ha='right', fontsize=14, color='black', fontweight='bold')

    # Explicit Axis Labels
    ax1.set_xlabel('Number of Variables (k)', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Area Under the ROC Curve (AUC)', fontsize=18, fontweight='bold') # Explicit
    ax1.set_title('B. Incremental Gain (Stepwise Selection)', fontsize=22, fontweight='bold', pad=15)
    ax1.set_xticks(gain_df['k'])
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(True, alpha=0.3)
    
    min_auc = gain_df['AUC'].min()
    max_auc = gain_df['AUC'].max()
    pad = (max_auc - min_auc) * 0.4 # Even more padding for bigger annotations
    ax1.set_ylim(min_auc - pad, max_auc + pad)

    plt.tight_layout()
    
    out_fig = os.path.join(OUTDIR, 'fig_abstract_combined.png')
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved {out_fig}")

if __name__ == "__main__":
    main()
