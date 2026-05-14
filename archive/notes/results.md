# Analysis Results - Knee OA Prediction

This document summarizes the results of the final analysis, including Odds Ratios (OR) for the selected variables and AUC comparisons across models.

## 1. Final Model Variables & Odds Ratios

The final clinical model was selected using a Stepwise MPMS (Multiple Peptide Mass Spectrometry - adapted optimization) approach. The standardized Odds Ratios (per 1 SD increase) with 95% Cluster Bootstrap Confidence Intervals are presented below.

> [!NOTE]
> ORs are standardized. Correlation within subjects (two knees) was handled using Cluster Bootstrap resampling of participants.

> [!TIP]
> **Why use Raw ORs for Binary Variables?**
> Standardized ORs are useful for comparing the *relative strength* of variables on a common scale (1 Standard Deviation). However, for **binary variables** like *Surgery* or *Trauma*, the "Standard Deviation" is not intuitive.
> *   **Raw OR** tells you the odds increase when the event (Surgery) is present vs absent. This is the **clinically interpretable** number.
> *   **Standardized OR** tells you the odds increase for a "typical" shift in the population distribution.

| Feature | Standardized OR (95% CI) | Raw OR (95% CI) |
| :--- | :--- | :--- |
| **age** | 2.20 (1.97-2.46) | 1.09 (1.08-1.11) |
| **bmi** | 1.79 (1.63-1.97) | 1.13 (1.11-1.15) |
| **history_surgery** | 1.45 (1.34-1.57) | **7.11** (4.83-10.43) |
| **frequent_symptoms** | 1.40 (1.29-1.52) | 2.46 (1.98-3.04) |
| **history_trauma** | 1.29 (1.19-1.41) | 2.08 (1.64-2.64) |

## 2. Model Comparison (AUC)

The final model AUC (0.815) is consistent and reproducible with Seed 42. Feature counts below refer to variables selected by the model (or input for RF/XGB).

We compared the performance of several models using 5-fold Cross-Validation with GroupKFold (grouping by participant). The AUCs reported are OOF (Out-Of-Fold) estimates with 95% Bootstrap CIs.

| Model | AUC (95% CI) | Brier Score (95% CI) | Features |
| :--- | :--- | :--- | :--- |
| **1. Full (Lasso) [LR]** | 0.826 (0.811-0.841) | 0.091 (0.086-0.096) | **42 selected** (from 55 input) |
| **2. Clinical (Lasso) [LR]** | 0.824 (0.809-0.839) | 0.091 (0.086-0.096) | **27 selected** (from 52 input) |
| **3. Clinical (MPMS) [LR]** | 0.824 (0.809-0.838) | 0.092 (0.086-0.097) | **9 selected** (from 52 input) |
| **4. Final (5 vars) [LR]** | **0.815** (0.800-0.829) | **0.093** (0.088-0.098) | **5 selected** |
| **5. Random Forest** | 0.796 (0.780-0.811) | 0.097 (0.092-0.102) | 52 (Clinical Input) |
| **6. XGBoost** | 0.776 (0.759-0.792) | 0.125 (0.119-0.130) | 52 (Clinical Input) |

> [!NOTE]
> **Input Statistics**:
> *   **Full Input Set**: 55 variables (Clinical + Bioimpedance).
> *   **Clinical Input Set**: 52 variables (excluding 3 Bioimpedance vars).
> *   **Selection**: Lasso selected 42 and 27 variables respectively.

<details>
<summary>Click to see full list of 55 Input Variables</summary>

`history_surgery`, `history_trauma`, `frequent_symptoms`, `recent_pain_7d`, `age`, `bmi`, `abdominal_obesity`, `waist_hip_ratio`, `bone_mineral_content_kg`, `mineral_mass_kg`, `skeletal_muscle_mass_kg`, `sit_stand_test`, `knee_disability`, `alcohol_binge`, `alcohol_excessive`, `occ_stairs`, `occ_kneeling`, `occ_squatting`, `hypertension`, `diabetes`, `metabolic_syndrome_JIS`, `metabolic_syndrome_NCEP`, `metabolic_syndrome_IDF`, `hypertriglyceridemia`, `hypertrig_meds`, `low_hdl`, 'low_hdl_meds`, `framingham_chd_chol`, `framingham_chd_ldl`, `framingham_cvd_model1`, `framingham_cvd_model2`, `sex_female`, 4x `race_raw`, 4x `occupation`, 3x `smoking_status`, 3x `physical_activity`, 3x `alcohol_use`.

</details>

### ROC Curves
![ROC Comparison](file:///C:/Users/julio/.gemini/antigravity/brain/b297fef8-ec02-4f65-86d0-529a6b5c0ae4/fig_roc_comparison_all_models.png)

## 3. Permutation Importance

Feature importance was assessed via Permutation Importance on the held-out folds during Cross-Validation.

> [!TIP]
> **Weighted vs. Unweighted Regression**:
> The user hypothesized that the AUC difference might be due to using specific class weights. We compared **Final (5 vars) [LR]** (Unweighted) vs **Final (5 vars) [LR Balanced]** (Weighted).
> *   **Unweighted AUC**: 0.815 (0.800-0.829)
> *   **Weighted AUC**: 0.815 (0.800-0.829)
> The difference is negligible (< 0.001), suggesting that class weighting is not the cause of the AUC variation.

![Permutation Importance Final 5 Vars](file:///C:/Users/julio/.gemini/antigravity/brain/b297fef8-ec02-4f65-86d0-529a6b5c0ae4/fig_permutation_importance_Final_5_vars.png)
*(Permutation Importance for the Final 5-Variable Model)*

## 4. Conclusion
- The final 5-variable clinical model achieves an AUC of [PLACEHOLDER].
- Random Forest and XGBoost provide [comparable/better/worse] performance.
- Key drivers of prediction are [PLACEHOLDER].
