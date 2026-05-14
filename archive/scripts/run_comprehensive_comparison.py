
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc as calc_auc
from sklearn.inspection import permutation_importance

import oarsi_data as p

# =============================================================================
# CONFIGURATION
# =============================================================================
RND = 42
OUTDIR = './results_comparison'
MPMS_FILE = './results_final_analysis/stepwise_mpms_clinical.csv'

# Models to include
RUN_MODELS = {
    'Stepwise Logistic Regression': True, # Renamed from 'LASSO-MPMS'
    'XGBoost': True,
    'Random Forest': True,
    'Neural Network': True
}

# Scenarios to run
RUN_VIRTUAL_MAX = True      # All available vars (Symptoms + Bio + Clinical)
RUN_WITH_SYMPTOMS = True    # Clinical + Symptoms
RUN_WITHOUT_SYMPTOMS = True # Clinical only (Screening)

# Plotting
SHOW_IMPORTANCE_PLOTS = True
SHOW_STEPWISE_PLOT = True
CALCULATE_ODDS_RATIOS = True

# Variable Labels (Mapping for clearer plots/tables)
LABEL_MAP = {
    'occupation_4': 'Occupation (Category 4)', 
    'race_raw_3': 'Race (Category 3)',
    'family_history_knee_replacement': 'Family History of Knee Replacement',
    'frequent_symptoms': 'Frequent Knee Symptoms',
    'history_surgery': 'History of Knee Surgery',
    'history_trauma': 'History of Knee Trauma',
    'knee_disability': 'Knee Disability',
    'recent_pain_7d': 'Recent Knee Pain (7d)',
    'bmi': 'Body Mass Index (BMI)',
    'age': 'Age (years)',
    'sit_stand_test': 'Sit-to-Stand Test (s)',
    'abdominal_obesity': 'Abdominal Obesity',
    'waist_hip_ratio': 'Waist-Hip Ratio',
    'mineral_mass_kg': 'Mineral Mass (kg)',
    'bone_mineral_content_kg': 'Bone Mineral Content (kg)',
    'skeletal_muscle_mass_kg': 'Skeletal Muscle Mass (kg)',
    'occ_stairs': 'Occupational: Stairs',
    'occ_kneeling': 'Occupational: Kneeling',
    'occ_squatting': 'Occupational: Squatting'
}

# =============================================================================
# VARIABLE DEFINITIONS
# =============================================================================
BASE_EXCLUDE = [
    'idelsa', 'side', 'kl', 'oapf', 'oa_knee',
    'kl_raw_num', 'oapf_raw_num',
    'race_raw', 'occupation', 'smoking_status',
    'physical_activity_ipaq', 'alcohol_use'
]

BIO_VARS = [
    'bone_mineral_content_kg', 'mineral_mass_kg', 'skeletal_muscle_mass_kg', 
    'abdominal_obesity', 'waist_hip_ratio', 'bmi'
]

SYMPTOM_VARS = [
    'frequent_symptoms', 'recent_pain_7d', 'knee_disability'
]
# Note: occ variables are EXPOSURE, so allowed in "Without Symptoms"

# =============================================================================
# MODEL FACTORIES
# =============================================================================
def get_lr_pipe():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight=None, random_state=RND)
    )

def get_rf_model():
    return RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RND, class_weight='balanced', n_jobs=-1)

def get_xgb_model():
    return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RND, eval_metric='logloss')

def get_mlp_model():
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.0001, 
                         batch_size='auto', learning_rate_init=0.001, max_iter=500, random_state=RND, early_stopping=True)

def get_pipeline(model_type):
    if model_type == 'Neural Network':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), get_mlp_model())
    elif model_type == 'Stepwise Logistic Regression':
         return None
    elif model_type == 'Random Forest':
        return make_pipeline(SimpleImputer(strategy='median'), get_rf_model())
    elif model_type == 'XGBoost':
        return make_pipeline(SimpleImputer(strategy='median'), get_xgb_model())
    else:
        raise ValueError(f"Unknown model: {model_type}")

# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def cv_roc_auc(model, X, y, groups):
    cv = GroupKFold(n_splits=5)
    y_true_all = []
    y_pred_all = []
    
    for tr, te in cv.split(X, y, groups):
        model.fit(X.iloc[tr], y[tr])
        
        if hasattr(model, "predict_proba"):
             probs = model.predict_proba(X.iloc[te])[:, 1]
        else:
             probs = model.predict(X.iloc[te])
             
        y_true_all.extend(y[te])
        y_pred_all.extend(probs)
        
    fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
    auc = roc_auc_score(y_true_all, y_pred_all)
    return fpr, tpr, auc, model 

def get_feature_importance(model, X, y, model_name):
    try:
        model.fit(X, y)
        importances = None
        feature_names = X.columns
        
        if hasattr(model, 'steps'):
            estimator = model.steps[-1][1]
        else:
            estimator = model
            
        if model_name in ['Random Forest', 'XGBoost']:
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
        elif model_name == 'Neural Network':
            r = permutation_importance(model, X, y, n_repeats=5, random_state=RND, n_jobs=-1)
            importances = r.importances_mean
            
        if importances is not None:
             df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
             df_imp['feature_label'] = df_imp['feature'].map(LABEL_MAP).fillna(df_imp['feature'])
             return df_imp
    except Exception as e:
        print(f"Error calculating importance for {model_name}: {e}")
    return None

def run_stepwise_mpms(X, y, groups, mpms_features):
    start_features = []
    results = []
    for k, feat in enumerate(mpms_features, 1):
        if feat not in X.columns:
            continue
        start_features.append(feat)
        X_sub = X[start_features]
        model = get_lr_pipe()
        _, _, auc, _ = cv_roc_auc(model, X_sub, y, groups)
        results.append({'k': k, 'AUC': auc, 'Added Variable': feat})
    return pd.DataFrame(results)

def plot_stepwise_detailed(df_results, scenario_name):
    fig, ax = plt.subplots(figsize=(14, 8)) 
    ax.plot(df_results['k'], df_results['AUC'], '-', color='gray', lw=2, alpha=0.5, zorder=1)
    
    cmap = matplotlib.colormaps['viridis']
    num_steps = len(df_results)
    
    for i, row in df_results.iterrows():
        k = int(row['k'])
        auc = row['AUC']
        color = cmap(i / max(1, num_steps-1))
        
        facecolors = color
        edgecolors = 'none' 
        alpha = 1.0
        marker = 'o'
        size = 250
        
        ax.scatter(k, auc, s=size, facecolors=facecolors, marker=marker, edgecolors=edgecolors, linewidth=2, zorder=10)
        
        var_name = row['Added Variable']
        var_display = LABEL_MAP.get(var_name, var_name)
        
        annot_text = f"{var_display}\nAUC: {auc:.3f}"
        
        offset_val = 60
        if k % 2 == 1: 
             xytext = (0, -offset_val) 
             va = 'top'
        else: 
             xytext = (0, offset_val)
             va = 'bottom'
             
        ax.annotate(
            annot_text, 
            (k, auc),
            xytext=xytext, 
            textcoords='offset points',
            rotation=0,
            ha='center', va=va, 
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9, lw=1),
            arrowprops=dict(arrowstyle='-', color='gray', lw=1.5, alpha=0.6)
        )

    ax.set_xlabel('Number of Variables (k)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Area Under the ROC Curve (AUC)', fontsize=16, fontweight='bold')
    ax.set_title(f'Incremental Gain (Stepwise Selection) - {scenario_name}', fontsize=20, fontweight='bold', pad=20)
    
    if not df_results.empty:
        ax.set_xticks(df_results['k'])
        min_auc = df_results['AUC'].min()
        max_auc = df_results['AUC'].max()
        pad = (max_auc - min_auc) * 0.4
        ax.set_ylim(min_auc - pad, max_auc + pad)
        
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTDIR, f'stepwise_trajectory_{scenario_name.replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Stepwise Plot: {plot_path}")

def calculate_odds_ratios(X, y, groups, features, scenario_name):
    """
    Calculates Odds Ratios (Raw and Standardized) using Statsmodels GEE for cluster-robust SEs.
    """
    print(f"   Calculating ORs for {len(features)} features in {scenario_name}...")
    
    # Filter available features
    valid_feats = [f for f in features if f in X.columns]
    X_sub = X[valid_feats].copy() # Ensure copy
    
    # Impute missing (Median)
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X_sub), columns=valid_feats, index=X_sub.index)
    
    # 1. RAW Odds Ratios
    # Using GEE (Generalized Estimating Equations) to account for clustering (bilateral knees)
    # Family=Binomial, CovStruct=Exchangeable or Independence with clustered covariances.
    # Statsmodels Logit with cluster-robust SEs is simpler:
    
    X_raw = sm.add_constant(X_imputed)
    try:
        # Use simple Logit with cluster covariance
        model_raw = sm.Logit(y, X_raw)
        res_raw = model_raw.fit(disp=0, cov_type='cluster', cov_kwds={'groups': groups})
        
        # Extract
        params = res_raw.params
        conf = res_raw.conf_int()
        conf['OR'] = params
        conf.columns = ['2.5%', '97.5%', 'OR']
        
        # Calculate Exp
        res_df_raw = np.exp(conf)
        res_df_raw['P-value'] = res_raw.pvalues
        res_df_raw['Feature'] = res_df_raw.index
        res_df_raw = res_df_raw[res_df_raw['Feature'] != 'const'] # Drop constant from table
        
        # Format
        res_df_raw['OR (95% CI)'] = res_df_raw.apply(
            lambda x: f"{x['OR']:.2f} ({x['2.5%']:.2f}-{x['97.5%']:.2f})", axis=1
        )
        res_df_raw = res_df_raw[['Feature', 'OR', '2.5%', '97.5%', 'P-value', 'OR (95% CI)']]
        res_df_raw['Feature Label'] = res_df_raw['Feature'].map(LABEL_MAP).fillna(res_df_raw['Feature'])
        
        # Save
        raw_path = os.path.join(OUTDIR, f'or_raw_{scenario_name.replace(" ", "_")}.csv')
        res_df_raw.to_csv(raw_path, index=False)
        print(f"   Saved Raw ORs to {raw_path}")
        
    except Exception as e:
        print(f"   Error calculating Raw ORs: {e}")

    # 2. STANDARDIZED Odds Ratios
    scaler = StandardScaler()
    X_std = pd.DataFrame(scaler.fit_transform(X_imputed), columns=valid_feats, index=X_imputed.index)
    X_std = sm.add_constant(X_std)
    
    try:
        model_std = sm.Logit(y, X_std)
        res_std = model_std.fit(disp=0, cov_type='cluster', cov_kwds={'groups': groups})
        
        params = res_std.params
        conf = res_std.conf_int()
        conf['OR'] = params
        conf.columns = ['2.5%', '97.5%', 'OR']
        
        res_df_std = np.exp(conf)
        res_df_std['P-value'] = res_std.pvalues
        res_df_std['Feature'] = res_df_std.index
        res_df_std = res_df_std[res_df_std['Feature'] != 'const']
        
        res_df_std['OR (95% CI)'] = res_df_std.apply(
            lambda x: f"{x['OR']:.2f} ({x['2.5%']:.2f}-{x['97.5%']:.2f})", axis=1
        )
        res_df_std = res_df_std[['Feature', 'OR', '2.5%', '97.5%', 'P-value', 'OR (95% CI)']]
        res_df_std['Feature Label'] = res_df_std['Feature'].map(LABEL_MAP).fillna(res_df_std['Feature'])
        
        std_path = os.path.join(OUTDIR, f'or_standardized_{scenario_name.replace(" ", "_")}.csv')
        res_df_std.to_csv(std_path, index=False)
        print(f"   Saved Standardized ORs to {std_path}")
        
    except Exception as e:
        print(f"   Error calculating Standardized ORs: {e}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    
    print("Loading data...")
    try:
        df = p.load_and_prep_data("./base_stata/stataToCsvMG.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    df = df.sort_values('idelsa').reset_index(drop=True)
    
    y = df['oa_knee'].values
    groups = df['idelsa'].values
    all_cols = [c for c in df.columns if c not in BASE_EXCLUDE]
    
    symptom_cols = [c for c in all_cols if c in SYMPTOM_VARS]
    cols_without_symptoms = [c for c in all_cols if c not in symptom_cols]
    cols_with_symptoms = all_cols 
    cols_virtual_max = all_cols 
    
    if os.path.exists(MPMS_FILE):
        mpms_df = pd.read_csv(MPMS_FILE)
        mpms_vars = mpms_df['Variable'].tolist()
    else:
        print("Warning: MPMS file not found. Skipping Stepwise.")
        RUN_MODELS['Stepwise Logistic Regression'] = False

    scenarios = []
    if RUN_VIRTUAL_MAX:
        scenarios.append(('Virtual Maximum', cols_virtual_max))
    if RUN_WITH_SYMPTOMS:
        scenarios.append(('With Symptoms', cols_with_symptoms))
    if RUN_WITHOUT_SYMPTOMS:
        scenarios.append(('Without Symptoms', cols_without_symptoms))
        
    summary_results = []
    
    for scenario_name, feat_list in scenarios:
        print(f"\n--- Scenario: {scenario_name} ---")
        
        valid_feats = [f for f in feat_list if f in df.columns]
        X = df[valid_feats]
        print(f"Features in set: {len(valid_feats)}")
        
        # Initialize ROC Figure
        roc_fig = plt.figure(figsize=(10, 8))
        ax = roc_fig.add_subplot(111)
        
        for model_name, enabled in RUN_MODELS.items():
            if not enabled: continue
            
            plt.figure(roc_fig.number)
            
            if model_name == 'Stepwise Logistic Regression':
                if scenario_name == 'Virtual Maximum': continue 
                
                valid_mpms = [v for v in mpms_vars if v in valid_feats]
                if not valid_mpms: continue
                
                # Stepwise trajectory (reusing fixed MPMS order, filtering by availability)
                step_res = run_stepwise_mpms(X, y, groups, valid_mpms)
                
                features_used = step_res['Added Variable'].tolist()
                X_mpms = X[features_used]
                fpr, tpr, auc, _ = cv_roc_auc(get_lr_pipe(), X_mpms, y, groups)
                
                ax.plot(fpr, tpr, lw=2, label=f'Stepwise Logistic Regression (Full Set k={len(features_used)}, AUC={auc:.3f})')
                
                if SHOW_STEPWISE_PLOT:
                    plot_stepwise_detailed(step_res, scenario_name)
                    
                if CALCULATE_ODDS_RATIOS:
                     calculate_odds_ratios(X, y, groups, features_used, scenario_name)

                summary_results.append({'Scenario': scenario_name, 'Model': 'Stepwise (Full)', 'AUC': auc})
                
            else:
                print(f"   Running {model_name}...")
                pipe = get_pipeline(model_name)
                fpr, tpr, auc, fitted_model = cv_roc_auc(pipe, X, y, groups)
                
                plt.figure(roc_fig.number)
                ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC={auc:.3f})')
                summary_results.append({'Scenario': scenario_name, 'Model': model_name, 'AUC': auc})
                
                if SHOW_IMPORTANCE_PLOTS:
                    imp_df = get_feature_importance(pipe, X, y, model_name)
                    if imp_df is not None:
                        imp_fig = plt.figure(figsize=(12, 8)) # Wider for labels
                        top_n = imp_df.head(15)
                        y_pos = np.arange(len(top_n))
                        # Use Labels
                        labels = top_n['feature_label'].tolist()
                        
                        plt.barh(y_pos, top_n['importance'], align='center')
                        plt.yticks(y_pos, labels)
                        plt.gca().invert_yaxis() 
                        plt.xlabel('Importance')
                        plt.title(f'Feature Importance: {model_name} ({scenario_name})')
                        plt.tight_layout()
                        imp_path = os.path.join(OUTDIR, f'importance_{model_name.replace(" ", "")}_{scenario_name.replace(" ", "_")}.png')
                        plt.savefig(imp_path)
                        plt.close(imp_fig)

        # Finalize ROC Plot
        plt.figure(roc_fig.number)
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title(f'ROC Curves: {scenario_name}', fontsize=16)
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        roc_path = os.path.join(OUTDIR, f'roc_comparison_{scenario_name.replace(" ", "_")}.png')
        plt.savefig(roc_path, dpi=300)
        plt.close(roc_fig)
        print(f"Saved ROC: {roc_path}")
        
    pd.DataFrame(summary_results).to_csv(os.path.join(OUTDIR, 'summary_all_models.csv'), index=False)
    print("\nDone.")

if __name__ == "__main__":
    main()
