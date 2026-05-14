OARSI 2026 Late-Breaking Abstract

Title: Clinical-Epidemiological Screening Tool For Prevalent Radiographic Knee Osteoarthritis In A Large Population-Based Cohort

*Authors: Domingues JG, Veloso AA, Telles RW, Barreto SM.*

## Purpose

To develop and internally validate a parsimonious clinical–epidemiological screening tool to estimate the probability of prevalent radiographic knee osteoarthritis (KOA) using routinely obtainable variables in a large population-based cohort.

## Methods

This cross-sectional analysis used data from the ELSA-Brasil Musculoskeletal Study (2012–2014). Participants underwent standardized bilateral knee radiography (fixed-flexion PA and lateral views). The outcome was prevalent radiographic KOA, defined as Kellgren–Lawrence grade ≥2 in the tibiofemoral compartment and/or definite patellofemoral osteoarthritis.

From 5,660 radiographed knees in the ELSA-Brasil MSK dataset, 5,652 were included (8 excluded due to arthroplasty or technical artifacts). Candidate variables spanning demographic, anthropometric, clinical, mechanical, metabolic, and lifestyle domains were evaluated. We conducted a knee-level analysis with participant-level clustering.

Variable selection followed a staged, interpretability-oriented strategy. First, L1-penalized logistic regression (LASSO) was used to reduce a high-dimensional set of potentially correlated variables, improving stability and limiting overfitting. Second, the Minimal Predictive Model Search (MPMS) algorithm was applied to identify a minimal subset with near-maximal discrimination. Third, variables were ordered via a forward stepwise procedure within the MPMS set, and the final model size was chosen by balancing parsimony and performance: variables were added only while the incremental AUC gain remained clinically meaningful (≤0.5% improvement threshold), prioritizing variables that are routinely and reliably obtainable.

Internal validation used 5-fold cross-validation with participant-level grouping to account for bilateral dependency. Discrimination was quantified by the area under the ROC curve (AUC), with confidence intervals derived from bootstrapping pooled predictions. Calibration was additionally assessed using the Brier score. Tree-based learners (Random Forest and XGBoost) were also evaluated under the same cross-validation framework. Analyses were performed in Python 3.12.1 using NumPy (2.3.5), pandas (2.3.3), scikit-learn (1.8.0), SciPy (1.17.0), statsmodels (0.14.6), Matplotlib (3.10.8), seaborn (0.13.2), and XGBoost (3.1.3).

## Results

A five-variable logistic regression model comprising age, body mass index (BMI), frequent knee symptoms, history of knee surgery, and history of knee trauma was selected. This parsimonious model demonstrated good discrimination (pooled AUC 0.815; 95% CI 0.799–0.829) and calibration (Brier score 0.093). Expanding the model beyond five variables yielded marginal gains (<0.5% AUC per variable). Furthermore, complex machine learning approaches did not improve performance, as Random Forest (AUC 0.796) and XGBoost (AUC 0.776) failed to outperform the simple logistic model in this dataset.

All selected variables were independently associated with prevalent radiographic KOA (p < 0.0001). History of knee surgery conferred the highest conditional risk (raw OR 7.11; 95% CI 4.83–10.43), followed by frequent symptoms (raw OR 2.46) and past trauma (raw OR 2.08). When measuring contribution to discriminative power on a population scale, standardized ORs identified Age (sOR 2.20; 95% CI 1.97–2.46) and BMI (sOR 1.79; 95% CI 1.63–1.97) as the strongest drivers. Sex was not selected in the final model, indicating that once mechanical and constitutional factors are accounted for, it provided no significant incremental discrimination.

## Conclusions

A simple, transparent clinical–epidemiological tool using five routinely available variables effectively identifies knees with a high probability of prevalent radiographic KOA. The integration of clinical symptoms with mechanical history and constitutional factors provides robust discrimination, rivaling the performance of complex algorithmic models. This tool may support efficient screening and prioritization of imaging in epidemiological studies and healthcare settings with limited access to radiography. A web-based calculator implementing this screening tool has been developed to facilitate open access and clinical use.

## 

## Figures

Figure 1. **Discriminative performance of the parsimonious screening tool.** Receiver operating characteristic (ROC) curves comparing the final 5-variable logistic regression model (Red line) against the full high-dimensional model, clinical-only baselines, and machine learning benchmarks (Random Forest, XGBoost). The final model achieves an AUC of 0.815 (95% CI 0.799–0.829), enabling effective screening without the complexity of the full variable set.

![](./results_final_analysis/fig_roc_comparison_unweighted_highlighted.png)

Figure 2. **Web-based Risk Calculator Interface.** Screenshot of the open-access clinical decision support tool allowing practitioners to estimate probability of prevalent radiographic OA by entering the five selected predictors: Age, BMI, History of Surgery, History of Trauma, and Frequent Knee Symptoms.

![](./screenshot.png)

## Other figures and tables (not for the abstract)

### Figure - ROC curves MPMS

Figure S1. **Variable Selection Trajectory (MPMS).** ROC curves illustrating the sequential discrimination performance as variables are added during the Minimal Predictive Model Search (MPMS). Note the rapid saturation of performance, with minimal gains observed after the first few clinically salient variables.

![](./results_final_analysis/fig_roc_mpms_overlay.png)

### Figure - Incremental gain for MPMS Variables

Figure S2. **Diminishing Returns in Model Complexity.** The bar chart displays the incremental AUC gain for each variable added during the forward stepwise selection. The red dashed line marks the <0.005 (0.5%) improvement threshold used to define the stopping point, justifying the selection of the 5-variable model over larger subsets.

![](./results_final_analysis/fig_stepwise_mpms.png)

### Figure - Permutation importance (final model)

Figure S3. **Relative Feature Importance.** Contribution of each predictor to the final model's discriminative power, measured by the mean decrease in AUC when the feature is randomly permuted. Age and BMI act as the primary constitutional drivers, while surgical history and symptoms provide critical specific risk signals.

![](./results_final_analysis/fig_permutation_importance_Final_5_vars.png)

### Table - All Models AUC and Brier

Table S1. **Comparative Performance of Candidate Models.** Evaluation of discrimination (AUC) and calibration (Brier Score) across different modeling strategies and feature sets. The final 5-variable model is highlighted, showing optimal balance between parsimony and performance compared to high-dimensional Lasso and ML approaches.

| Model | AUC (95% CI) | Brier Score (95% CI) | Features |
| :---- | :---- | :---- | :---- |
| **1\. Full (Lasso) \[LR\]** | 0.826 (0.811-0.841) | 0.091 (0.086-0.096) | 42 selected (from 55 input) |
| **2\. Clinical (Lasso) \[LR\]** | 0.824 (0.809-0.839) | 0.091 (0.086-0.096) | 27 selected (from 52 input) |
| **3\. Clinical (MPMS) \[LR\]** | 0.824 (0.809-0.838) | 0.092 (0.086-0.097) | 9 selected (from 52 input) |
| **4\. Final (5 vars) \[LR\]** | **0.815** (0.800-0.829) | **0.093** (0.088-0.098) | 5 selected |
| **5\. Random Forest** | 0.796 (0.780-0.811) | 0.097 (0.092-0.102) | 52 (Clinical Input) |
| **6\. XGBoost** | 0.776 (0.759-0.792) | 0.125 (0.119-0.130) | 52 (Clinical Input) |

### Table - OR of final model features

Table S2. **Multivariable Predictors of Prevalent Radiographic KOA.** Adjusted Odds Ratios (OR) for the five variables in the final screening tool. Standardized ORs allow comparison of effect size for continuous variables (Age, BMI) per standard deviation unit, while Raw ORs provide intuitive risk interpretation for binary history variables.

| Feature | Standardized OR (95% CI) | Raw OR (95% CI) |
| :---- | :---- | :---- |
| **age** | 2.20 (1.97-2.46) | 1.09 (1.08-1.11) |
| **bmi** | 1.79 (1.63-1.97) | 1.13 (1.11-1.15) |
| **history\_surgery** | 1.45 (1.34-1.57) | 7.11 (4.83-10.43) |
| **frequent\_symptoms** | 1.40 (1.29-1.52) | 2.46 (1.98-3.04) |
| **history\_trauma** | 1.29 (1.19-1.41) | 2.08 (1.64-2.64) |

### All Variables list

52 Clinical-Epidemiological Variables

`history_surgery`, `history_trauma`, `frequent_symptoms`, `recent_pain_7d`, `age`, `bmi`, `abdominal_obesity`, `waist_hip_ratio`, `sit_stand_test`, `knee_disability`, `alcohol_binge`, `alcohol_excessive`, `occ_stairs`, `occ_kneeling`, `occ_squatting`, `hypertension`, `diabetes`, `metabolic_syndrome_JIS`, `metabolic_syndrome_NCEP`, `metabolic_syndrome_IDF`, `hypertriglyceridemia`, `hypertrig_meds`, `low_hdl`, `low_hdl_meds`, `framingham_chd_chol`, `framingham_chd_ldl`, `framingham_cvd_model1`, `framingham_cvd_model2`, `sex_female`.

*Categorical variables (one-hot encoded):* `race_raw` (6 levels), `occupation` (5 levels), `smoking_status` (4 levels), `physical_activity` (4 levels), `alcohol_use` (4 levels).
*(Note: Level counts include missing/unknown categories modeled as distinct features)*

3 Bioimpedance Variables

`bone_mineral_content_kg`, `mineral_mass_kg`, `skeletal_muscle_mass_kg`.

### Main Variable Definitions

**History of Knee Surgery**: "Have you ever undergone any type of knee surgery, including arthroscopy, meniscus or ligament repair, or a total joint replacement?"

**History of Knee Trauma**: "Have you ever sustained a knee injury or trauma that resulted in difficulty walking for at least one week?"

**Frequent Knee Symptoms**: “In the last 12 months, have you had pain, discomfort, or stiffness in your knee that lasted for most days for at least one month?”