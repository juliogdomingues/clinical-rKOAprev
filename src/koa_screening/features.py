import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve

from . import reporting as oarsi_reporting
from . import utils as oarsi_utils  # noqa: F401  (re-exported for downstream consumers)

from .config import RND

def _cv_auc_for_features(df, features, target_col='oa_knee', group_col='idelsa'):
    valid = [f for f in features if f in df.columns]
    if not valid:
        return np.nan
    y = df[target_col].values
    groups = df[group_col].values
    cv = GroupKFold(n_splits=5)
    pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        # NOTE: unweighted metrics (better calibrated probabilities; consistent with final model export)
        LogisticRegression(max_iter=2000, class_weight=None, random_state=RND),
    )
    aucs = []
    for tr, te in cv.split(df, y, groups):
        pipe.fit(df.iloc[tr][valid], y[tr])
        probs = pipe.predict_proba(df.iloc[te][valid])[:, 1]
        aucs.append(roc_auc_score(y[te], probs))
    return float(np.mean(aucs))

def run_lasso(X, y, label=None, outdir=None):
    print("   -> Seleção LASSO...")
    pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), 
                         LogisticRegressionCV(cv=3, penalty='l1', solver='liblinear', 
                                              max_iter=5000, scoring='roc_auc', random_state=RND))
    pipe.fit(X, y)
    model = pipe.named_steps['logisticregressioncv']
    coefs = pd.Series(model.coef_[0], index=X.columns)
    selected = coefs[coefs != 0].abs().sort_values(ascending=False).index.tolist()

    if label and outdir:
        oarsi_reporting._save_lasso_diagnostics(model, X, label=label, outdir=outdir)

    return selected

def run_mpms(X, y, groups, candidate_vars):
    print(f"   -> Otimização MPMS em {len(candidate_vars)} variáveis...")
    results = []
    limit = min(20, len(candidate_vars))
    
    for k in range(1, limit + 1):
        current_vars = candidate_vars[:k]
        pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            # NOTE: unweighted metrics
            LogisticRegression(max_iter=1000, class_weight=None, random_state=RND),
        )
        cv = GroupKFold(n_splits=5)
        scores = cross_val_score(pipe, X[current_vars], y, groups=groups, cv=cv, scoring='roc_auc')
        mean_auc = scores.mean()
        score_penalized = mean_auc - (0.001 * k) 
        results.append({'k': k, 'AUC': mean_auc, 'Penalized': score_penalized, 'Vars': current_vars})
        
    res_df = pd.DataFrame(results)
    best_model = res_df.sort_values('Penalized', ascending=False).iloc[0]
    return best_model['Vars']

def run_stepwise_specific(df, target_col, group_col, candidates):
    print("\n[STEPWISE] Ordenando variáveis do modelo final...")
    y = df[target_col].values
    groups = df[group_col].values
    
    selected = []
    remaining = candidates.copy()
    history = []
    current_auc = 0.5
    step = 1
    
    while remaining:
        best_auc = -1
        best_var = None
        for var in remaining:
            trial = selected + [var]
            pipe = make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler(),
                # NOTE: unweighted metrics
                LogisticRegression(max_iter=1000, class_weight=None, random_state=RND),
            )
            cv = GroupKFold(n_splits=5)
            scores = cross_val_score(pipe, df[trial], y, groups=groups, cv=cv, scoring='roc_auc')
            auc = scores.mean()
            if auc > best_auc:
                best_auc = auc
                best_var = var
        
        gain = best_auc - current_auc
        print(f"   Passo {step}: + {best_var} (AUC {best_auc:.4f})")
        history.append({'Step': step, 'Variable': best_var, 'AUC': best_auc, 'Gain': gain})
        selected.append(best_var)
        remaining.remove(best_var)
        current_auc = best_auc
        step += 1
        
    return pd.DataFrame(history)

def run_analysis(df, outdir='./results_final_analysis', use_womac=False):
    print("[2/5] Definindo Modelos...")
    os.makedirs(outdir, exist_ok=True)

    exclude_base = [
        'idelsa', 'side', 'kl', 'oapf', 'oa_knee',
        # NEW: prevent leakage from raw outcome helper columns
        'kl_raw_num', 'oapf_raw_num',
        'race_raw', 'occupation', 'smoking_status',
        'physical_activity_ipaq', 'alcohol_use'
    ]

    womac_vars = ['womac_total', 'womac_pain', 'womac_stiffness', 'womac_function']
    exclude_womac = womac_vars if not use_womac else []

    # MODELO COMPLEXO (TUDO)
    exclude_complex = exclude_base + exclude_womac
    X_complex_cols = [c for c in df.columns if c not in exclude_complex]
    X_complex = df[X_complex_cols].dropna(thresh=len(df) * 0.5, axis=1)

    # MODELO CLÍNICO (SEM BIOIMPEDÂNCIA)
    bio_vars = ['bone_mineral_content_kg', 'mineral_mass_kg', 'skeletal_muscle_mass_kg']
    X_clinical_cols = [c for c in X_complex_cols if c not in bio_vars]
    X_clinical = df[X_clinical_cols].dropna(thresh=len(df) * 0.5, axis=1)

    oarsi_reporting._save_imputation_counts(X_complex, "full_feature_set", outdir)
    oarsi_reporting._save_imputation_counts(X_clinical, "clinical_feature_set", outdir)

    print(f"   -> # Variáveis: {len(X_complex.columns)}")
    print(f"   -> # Variáveis Clínicas: {len(X_clinical.columns)}")
    
    # 2. Treinamento e Seleção
    print("[3/5] Selecionando Variáveis...")
    
    # A. Complexo (LASSO)
    vars_complex = run_lasso(X_complex, df['oa_knee'], label='full', outdir=outdir)
    
    # B. Clínico Full (LASSO)
    vars_clinical_lasso = run_lasso(X_clinical, df['oa_knee'], label='clinical', outdir=outdir)
    
    # C. Clínico MPMS (Otimização dentro do Clínico)
    vars_clinical_mpms = run_mpms(X_clinical, df['oa_knee'], df['idelsa'], vars_clinical_lasso)
    
    models = {
        '1. Full': vars_complex,
        '2. Clinical (- bioimpedance)': vars_clinical_lasso,
        '3. Clinical MPMS': vars_clinical_mpms,
    }
    
    # 3. Comparação (ROC)
    print("[4/5] Gerando Curvas ROC Comparativas...")
    results = []
    y = df['oa_knee'].values
    groups = df['idelsa'].values
    cv = GroupKFold(n_splits=5)
    
    plt.figure(figsize=(10, 8))
    for name, feats in models.items():
        if not feats:
            continue
        valid = [f for f in feats if f in df.columns]
        pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            # NOTE: unweighted metrics
            LogisticRegression(max_iter=2000, class_weight=None, random_state=RND),
        )
        aucs = []
        y_true_all, y_pred_all = [], []
        
        for tr, te in cv.split(df, y, groups):
            pipe.fit(df.iloc[tr][valid], y[tr])
            probs = pipe.predict_proba(df.iloc[te][valid])[:, 1]
            aucs.append(roc_auc_score(y[te], probs))
            y_true_all.extend(y[te])
            y_pred_all.extend(probs)
            
        mean_auc = np.mean(aucs)
        results.append({'Model': name, 'AUC': mean_auc, 'Std': np.std(aucs), 'Vars': str(valid)})
        
        fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
        plt.plot(fpr, tpr, label=f'{name} (AUC={mean_auc:.3f})')

    plt.plot([0,1], [0,1], 'k--')
    plt.legend()
    plt.title('ROC Curve Comparison Across Models')
    plt.savefig(os.path.join(outdir, 'fig_roc_comparison.png'))
    plt.close()
    
    # 4. Stepwise no Vencedor (MPMS)
    print("[5/5] Gerando Gráfico Stepwise do MPMS Clínico...")
    step_res = run_stepwise_specific(df, 'oa_knee', 'idelsa', vars_clinical_mpms)
    step_res.to_csv(os.path.join(outdir, 'stepwise_mpms_clinical.csv'), index=False)

    # Curvas ROC por tamanho do modelo (1..N variáveis do MPMS, em ordem stepwise)
    mpms_ordered_vars = step_res['Variable'].tolist() if not step_res.empty else []
    roc_mpms_dir = os.path.join(outdir, 'roc_mpms_by_k')
    os.makedirs(roc_mpms_dir, exist_ok=True)

    if mpms_ordered_vars:
        mpms_k_perf = []
        mpms_roc_curves = []
        y = df['oa_knee'].values
        groups = df['idelsa'].values
        cv = GroupKFold(n_splits=5)

        for k in range(1, len(mpms_ordered_vars) + 1):
            feats_k = mpms_ordered_vars[:k]
            pipe = make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler(),
                # NOTE: unweighted metrics
                LogisticRegression(max_iter=2000, class_weight=None, random_state=RND),
            )

            aucs = []
            y_true_all, y_pred_all = [], []
            for tr, te in cv.split(df, y, groups):
                pipe.fit(df.iloc[tr][feats_k], y[tr])
                probs = pipe.predict_proba(df.iloc[te][feats_k])[:, 1]
                aucs.append(roc_auc_score(y[te], probs))
                y_true_all.extend(y[te])
                y_pred_all.extend(probs)

            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs))
            mpms_k_perf.append(
                {
                    'k': k,
                    'AUC': mean_auc,
                    'Std': std_auc,
                    'Vars': str(feats_k),
                }
            )

            fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
            pooled_auc_k = roc_auc_score(y_true_all, y_pred_all)
            mpms_roc_curves.append({'k': k, 'fpr': fpr, 'tpr': tpr, 'auc': mean_auc, 'pooled_auc': pooled_auc_k})
            
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(fpr, tpr, color='navy', linewidth=2, label=f'k={k} (AUC={mean_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_title(f'ROC Curve: Clinical MPMS (k={k})')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(roc_mpms_dir, f'fig_roc_mpms_k{k:02d}.png'))
            plt.close(fig)

        pd.DataFrame(mpms_k_perf).to_csv(os.path.join(outdir, 'mpms_k_performance.csv'), index=False)

        # Gráfico único: sobreposição das curvas ROC (k=1..N), destacando k=5
        fig, ax = plt.subplots(figsize=(9, 7))
        cmap = plt.get_cmap('viridis')
        n_curves = max(1, len(mpms_roc_curves))
        for i, curve in enumerate(mpms_roc_curves):
            k = int(curve['k'])
            is_final = (k == 5)
            color = 'crimson' if is_final else cmap(i / (n_curves - 1) if n_curves > 1 else 0.5)
            lw = 3.0 if is_final else 1.5
            alpha = 1.0 if is_final else 0.8
            
            # Use stored pooled AUC
            label_auc = curve['pooled_auc']
            label = f'Final (k=5) AUC={label_auc:.3f}' if is_final else f'k={k} AUC={label_auc:.3f}'
            ax.plot(curve['fpr'], curve['tpr'], color=color, linewidth=lw, alpha=alpha, label=label)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_title('ROC Curves: Clinical MPMS (k=1..N)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'fig_roc_mpms_overlay_v2.png'))
        plt.close(fig)

        # Coeficientes do modelo final (k=5) para predição
        if len(mpms_ordered_vars) >= 5:
            feats_5 = mpms_ordered_vars[:5]
            pipe_5 = make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight=None, random_state=RND),  # already unweighted
            )
            pipe_5.fit(df[feats_5], df['oa_knee'].values)

            imputer = pipe_5.named_steps['simpleimputer']
            scaler = pipe_5.named_steps['standardscaler']
            lr = pipe_5.named_steps['logisticregression']

            # NEW: imputation report for the final 5 features (counts of nulls that would be imputed)
            impute_counts = df[feats_5].isna().sum().rename("n_imputed").to_frame().reset_index().rename(columns={"index": "feature"})
            impute_counts["pct_imputed"] = (impute_counts["n_imputed"] / len(df)) * 100.0
            impute_counts.to_csv(os.path.join(outdir, "final_5var_imputation_counts.csv"), index=False)

            coef_df = pd.DataFrame(
                {
                    'feature': feats_5,
                    'imputer_median': imputer.statistics_.tolist(),
                    'scaler_mean': scaler.mean_.tolist(),
                    'scaler_scale': scaler.scale_.tolist(),
                    'coef_on_scaled': lr.coef_[0].tolist(),
                }
            )

            # SINGLE CSV (canonical)
            combined_rows = [
                {
                    'param_type': 'intercept',
                    'feature': '__INTERCEPT__',
                    'intercept': float(lr.intercept_[0]),
                    'positive_class': 1,
                    'notes': 'Prediction uses: z = (x_imputed - scaler_mean) / scaler_scale; logit = intercept + sum(coef_on_scaled * z)',
                    'imputer_median': np.nan,
                    'scaler_mean': np.nan,
                    'scaler_scale': np.nan,
                    'coef_on_scaled': np.nan,
                }
            ]

            for _, r in coef_df.iterrows():
                combined_rows.append(
                    {
                        'param_type': 'feature',
                        'feature': r['feature'],
                        'intercept': float(lr.intercept_[0]),
                        'positive_class': 1,
                        'notes': '',
                        'imputer_median': float(r['imputer_median']),
                        'scaler_mean': float(r['scaler_mean']),
                        'scaler_scale': float(r['scaler_scale']),
                        'coef_on_scaled': float(r['coef_on_scaled']),
                    }
                )

            combined_df = pd.DataFrame(combined_rows)
            combined_df.to_csv(os.path.join(outdir, 'final_5var_model.csv'), index=False)

    # Evidência para o motivo de sexo não entrar no modelo (impacto em AUC)
    sex_var = 'sex_female'
    if sex_var in df.columns:
        base_feats = [f for f in vars_clinical_mpms if f != sex_var]
        auc_sex_only = _cv_auc_for_features(df, [sex_var])
        auc_base = _cv_auc_for_features(df, base_feats)
        auc_with_sex = _cv_auc_for_features(df, base_feats + [sex_var])
        sex_report = pd.DataFrame(
            [
                {
                    'sex_feature': sex_var,
                    'auc_sex_only': auc_sex_only,
                    'auc_base_model': auc_base,
                    'auc_base_plus_sex': auc_with_sex,
                    'delta_auc_add_sex': (auc_with_sex - auc_base) if np.isfinite(auc_with_sex) and np.isfinite(auc_base) else np.nan,
                }
            ]
        )
        sex_report.to_csv(os.path.join(outdir, 'sex_feature_evidence.csv'), index=False)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(step_res['Step'], step_res['AUC'], 'o-', linewidth=2, color='navy')
    for i, row in step_res.iterrows():
        ax.annotate(
            row['Variable'],
            (row['Step'], row['AUC']),
            xytext=(0, 10),
            textcoords='offset points',
            rotation=45,
            fontsize=9,
            ha='left',
            va='bottom',
        )

    # Evita sobreposição das anotações com o título: adiciona "headroom" no eixo Y
    if not step_res.empty:
        y_min = float(step_res['AUC'].min())
        y_max = float(step_res['AUC'].max())
        pad = max(0.03, (y_max - y_min) * 0.25)
        ax.set_ylim(y_min - pad * 0.15, min(1.0, y_max + pad))

    ax.set_title('Incremental Gain: Clinical MPMS Variables', pad=14)
    ax.set_ylabel('AUC')
    ax.set_xlabel('Number of variables')
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, 'fig_stepwise_mpms.png'))
    plt.close(fig)
    
    return pd.DataFrame(results)
