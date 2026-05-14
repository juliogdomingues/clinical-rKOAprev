
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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score

import oarsi_data as p

RND = 42

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

def get_mlp_model():
    # Shallow MLP for structured data, with regularization
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.0001, 
                         batch_size='auto', learning_rate='constant',  learning_rate_init=0.001, 
                         max_iter=500, random_state=RND, early_stopping=True)

def cv_roc(model, X, y, groups):
    cv = GroupKFold(n_splits=5)
    y_true_all = []
    y_pred_all = []
    
    for tr, te in cv.split(X, y, groups):
        # Handle pipeline vs model
        model.fit(X.iloc[tr], y[tr])
        
        if hasattr(model, "predict_proba"):
             probs = model.predict_proba(X.iloc[te])[:, 1]
        else:
             probs = model.predict(X.iloc[te])
             
        y_true_all.extend(y[te])
        y_pred_all.extend(probs)
        
    pooled_auc = roc_auc_score(y_true_all, y_pred_all)
    return pooled_auc

def main():
    print("Loading data...")
    df = p.load_and_prep_data("./base_stata/stataToCsvMG.csv")
    df = df.sort_values('idelsa').reset_index(drop=True)
    
    y = df['oa_knee'].values
    groups = df['idelsa'].values
    
    # Define Input Features (All Clinical for ML Comparison)
    exclude_base = [
        'idelsa', 'side', 'kl', 'oapf', 'oa_knee',
        'kl_raw_num', 'oapf_raw_num',
        'race_raw', 'occupation', 'smoking_status',
        'physical_activity_ipaq', 'alcohol_use'
    ]
    bio_vars = ['bone_mineral_content_kg', 'mineral_mass_kg', 'skeletal_muscle_mass_kg']
    X_cols = [c for c in df.columns if c not in exclude_base + bio_vars]
    
    # Filter to valid cols
    X_clinical_input = df[X_cols].dropna(thresh=len(df)*0.5, axis=1) 
    feats_clinical_input = X_clinical_input.columns.tolist()
    
    current_feats = [f for f in feats_clinical_input if f in df.columns]
    X = df[current_feats]

    print("\n running comparison...")
    
    # Models to compare
    models = {
        'Logistic Regression (Baseline)': get_lr_pipe(),
        'Random Forest': make_pipeline(SimpleImputer(strategy='median'), get_rf_model()),
        'XGBoost': make_pipeline(SimpleImputer(strategy='median'), get_xgb_model()),
        'Neural Network (MLP)': make_pipeline(
            SimpleImputer(strategy='median'), 
            StandardScaler(), # Critical for MLP
            get_mlp_model()
        )
    }
    
    results = {}
    for name, model in models.items():
        print(f"Running {name}...")
        try:
            auc = cv_roc(model, X, y, groups)
            results[name] = auc
            print(f"  -> AUC: {auc:.4f}")
        except Exception as e:
            print(f"  -> Error: {e}")
            results[name] = "Error"

    print("\n--- Summary Results ---")
    for name, auc in results.items():
        print(f"{name}: {auc}")

if __name__ == "__main__":
    main()
