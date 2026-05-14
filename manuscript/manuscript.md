# TITLE PAGE

**Title**  
Clinical-Epidemiological Screening Tool for Prevalent Radiographic Knee Osteoarthritis: Comparison of Traditional and Machine Learning Approaches in the ELSA-Brasil Cohort

**Running Head**  
Clinical Identification of Radiographic Knee Osteoarthritis

**Authors**

Júlio Guerra Domingues \[1\]

Adriano Alonso Veloso \[2\]

Rosa Weiss Telles \[3\]

Sandhi Maria Barreto \[4\]

**Affiliations:**

1. Department of Anatomy and Medical Imaging, UFMG, Belo Horizonte, Brazil  
2. Department of Computer Science, UFMG, Belo Horizonte, Brazil  
3. Department of Clinical Medicine, UFMG, Belo Horizonte, Brazil  
4. Department of Preventive and Social Medicine, UFMG, Belo Horizonte, Brazil

Corresponding Author:

Júlio Guerra Domingues

Email: juliogdomingues@gmail.com

# ABSTRACT

## Objective

To develop and compare simple versus complex modeling approaches for identifying prevalent radiographic knee osteoarthritis (KOA) in a large population-based cohort, evaluating the necessity of advanced machine learning algorithms and extensive variable sets.

## Design

Cross-sectional analysis within the ELSA-Brasil Musculoskeletal Study (2012–2014). The sample included 2,830 participants (5,655 knees). The outcome was prevalent radiographic KOA (Kellgren–Lawrence grade ≥2 or definite patellofemoral OA). We compared a traditional Stepwise Logistic Regression approach against three machine learning algorithms (XGBoost, Random Forest, Neural Network) across two scenarios: (1) "Screening" using only demographic and clinical history variables, and (2) "Case Finding" adding patient-reported knee symptoms. A "Virtual Maximum" scenario including bioimpedance and advanced anthropometry was also evaluated. Models were assessed using 5-fold nested cross-validation (AUC-ROC).

## Results

Simple Stepwise Logistic Regression matched or outperformed complex models in all scenarios. In the "Screening" scenario (without symptoms), the Stepwise model achieved an AUC of 0.810, comparable to XGBoost (0.803) and superior to Random Forest (0.785) and Neural Networks (0.742). Adding specific knee symptoms ("Case Finding") marginally improved performance (Stepwise AUC 0.824), but the complex models did not gain additional advantage. The inclusion of bioimpedance variables provided no performance benefit. Key predictors included age, BMI, history of knee surgery (OR 8.10), and history of knee trauma (OR 2.47).

## Conclusions

A simple, transparent logistic regression model using routinely obtainable clinical variables effectively identifies prevalent radiographic KOA, performing as well as complex "black box" machine learning methods. This supports the use of interpretable, low-cost screening tools to prioritize imaging in resource-limited settings without the need for advanced computational or diagnostic technologies.

**Keywords**  
Knee Osteoarthritis; Screening; Machine Learning; Epidemiology; ELSA-Brasil.

# Introduction

Osteoarthritis (OA) is a leading cause of disability worldwide, with knee osteoarthritis (KOA) contributing significantly to this burden [\[1,2\]](https://www.zotero.org/google-docs/?RKLGFn). While symptoms guide clinical management, radiographic assessment remains the standard for defining structural disease in epidemiological and research contexts. However, in large population-based studies or resource-constrained health systems, obtaining radiographs for all individuals is often logistically and financially prohibitive. This creates a need for efficient "screening" or "case-finding" tools that can identify individuals with a high probability of structural disease using only easily obtainable non-imaging data [\[6\]](https://www.zotero.org/google-docs/?suiQSf).

Technological advances have spurred interest in using complex machine learning (ML) algorithms and novel biomarkers (e.g., bioimpedance, advanced anthropometry) to improve predictive accuracy [\[7,8\]](https://www.zotero.org/google-docs/?EkQux4). These "black box" models often promise superior performance by capturing non-linear interactions but come at the cost of interpretability and computational complexity [\[9,10\]](https://www.zotero.org/google-docs/?2fMQug). A critical, often unanswered question is whether these complex approaches actually offer meaningful advantages over traditional, transparent statistical methods when applied to standard clinical-epidemiological data.

Furthermore, the utility of such models depends on the screening context. In a primary care setting, patients typically present with symptoms, and the goal is to confirm diagnosis ("Case Finding"). In contrast, general population screening might aim to identify at-risk individuals regardless of current complaint status ("Screening").

Using data from the ELSA-Brasil Musculoskeletal Study (ELSA-Brasil MSK) [\[11\]](https://www.zotero.org/google-docs/?pFr0UF), we aimed to: (1) Develop and validate models to identify prevalent radiographic KOA using routinely available clinical variables; (2) Systematically compare the performance of a simple Stepwise Logistic Regression against complex ML algorithms (XGBoost, Random Forest, Neural Networks); and (3) Evaluate the incremental value of adding symptom data and objective bioimpedance measures to the screening performance.

# Methods

## Study Design and Sample

This cross-sectional study utilized data from the ELSA-Brasil MSK cohort (2012–2014), an ancillary study of the Brazilian Longitudinal Study of Adult Health. The analytical sample comprised 2,830 participants (5,655 knees) aged 38–79 years [\[11\]](https://www.zotero.org/google-docs/?lAwcGQ).

## Outcome Definition

The primary outcome was prevalent radiographic Knee Osteoarthritis (KOA), defined as Kellgren–Lawrence (KL) grade ≥2 in the tibiofemoral compartment and/or definite patellofemoral osteoarthritis [\[11\]](https://www.zotero.org/google-docs/?RfbENC). Knees with total arthroplasty were excluded from the analysis.

## Predictor Variables and Scenarios

We curated a comprehensive set of candidate variables spanning demographics, anthropometry, clinical history, and occupational exposures (Supplementary Table S1). Based on these, we defined three analysis scenarios:

1.  **Screening (Without Symptoms):** Included only variables obtainable without asking about current knee pain/symptoms. This simulates a general health check context. Variables included:
    *   **Demographics:** Age, Sex, Race/skin color, Occupation.
    *   **Anthropometry:** BMI.
    *   **Clinical History:** History of knee trauma, History of knee surgery, Family history of knee replacement (FDR4_LB).
    *   **Occupational Exposure:** History of squatting, kneeling, or climbing stairs for work.
2.  **Case Finding (With Symptoms):** Added specific knee symptom variables: Frequent knee symptoms (pain/stiffness on most days in past month), Recent knee pain (past 7 days), and Knee disability.
3.  **Virtual Maximum:** Added objective bioimpedance and advanced anthropometric measures (Skeletal Muscle Mass, Bone Mineral Content, Waist-Hip Ratio) to test if "high-tech" measurements add value.

## Statistical Analysis and Modeling

We compared four modeling approaches:
1.  **Stepwise Logistic Regression:** A traditional, interpretable generalized linear model using bidirectional stepwise selection (based on AIC/BIC optimizations).
2.  **XGBoost:** A gradient-boosted decision tree algorithm known for high performance on tabular data.
3.  **Random Forest:** An ensemble of decision trees robust to overfitting.
4.  **Neural Network (MLP):** A multi-layer perceptron (deep learning capability) to capture complex non-linear patterns.

All models were evaluated using **5-fold nested cross-validation** with grouping at the participant level to prevent data leakage between bilateral knees. Performance was assessed using the Area Under the Receiver Operating Characteristic Curve (**AUC-ROC**). 

For the final logistic model, we calculated both **Raw Odds Ratios** (to interpret biological magnitude) and **Standardized Odds Ratios** (to compare variable importance per standard deviation). Analyses were performed in Python 3.12 using scikit-learn and statsmodels.

# Results

## Model Comparison: Simple vs. Complex

The simple **Stepwise Logistic Regression** performed as well as or better than the complex machine learning models across all scenarios (Table 1).

In the **"Screening" (Without Symptoms)** scenario:
*   **Stepwise Logistic Regression:** **AUC 0.810**
*   XGBoost: AUC 0.803
*   Random Forest: AUC 0.785
*   Neural Network: AUC 0.742

In the **"Case Finding" (With Symptoms)** scenario:
*   **Stepwise Logistic Regression:** **AUC 0.824**
*   XGBoost: AUC 0.809
*   Random Forest: AUC 0.789
*   Neural Network: AUC 0.765

The "Virtual Maximum" scenario, which added bioimpedance and advanced body composition metrics, yielded identical performance to the "With Symptoms" scenario (XGBoost 0.809), indicating no incremental value from these specialized measurements.

## Variable Importance and Association

Key predictors identified by the Stepwise model in the "Screening" scenario included age, BMI, history of knee surgery, and history of knee trauma.

**Table 1. Multivariable Association (Screening Model - Without Symptoms)**
*Estimates from the Stepwise Logistic Regression model.*

| Variable | Raw Odds Ratio (95% CI) | Standardized OR (95% CI)* |
| :--- | :--- | :--- |
| **Age** (per year) | 1.10 (1.09–1.11) | 2.34 (2.09–2.61) |
| **BMI** (per kg/m²) | 1.14 (1.12–1.17) | 1.89 (1.71–2.09) |
| **History of Knee Surgery** (Yes vs No) | 8.10 (5.46–12.03) | 1.48 (1.37–1.59) |
| **History of Knee Trauma** (Yes vs No) | 2.47 (1.95–3.13) | 1.38 (1.27–1.50) |
| **Race** (Category 3 vs Ref) | 0.69 (0.55–0.87) | 0.83 (0.74–0.93) |
| **Occupation** (Category 4 vs Ref) | 0.66 (0.53–0.83) | 0.82 (0.73–0.91) |

*\*Standardized OR reflects the change in odds per 1 Standard Deviation increase for continuous variables.*

History of knee surgery was the strongest individual marker (OR 8.10), likely serving as a proxy for established severe disease. However, at the population level (Standardized OR), **Age** (OR 2.34) and **BMI** (OR 1.89) were the most dominant drivers of risk identification due to their high prevalence and continuous effect.

Adding symptoms (Scenario 2) increased the AUC from 0.810 to 0.824. While statistically detectable, this suggests that the majority of the information regarding *structural* disease prevalence is already captured by the demographic and history variables alone. Frequency of symptoms (OR 1.58) and recent pain (OR 1.61) were significant but did not displace age and BMI as the primary determinants.

# Discussion

Our results demonstrate that a simple, interpretable Stepwise Logistic Regression model is sufficient for identifying prevalent radiographic knee osteoarthritis in a general adult population. Complex machine learning algorithms (XGBoost, Random Forest, Neural Networks) offered no performance advantage, and in some cases (Neural Networks) performed worse, likely due to overfitting on this sample size or the fundamentally linear nature of the risk associations.

## Implications for Screening

The strong performance of the "Screening" model (AUC 0.810)—which relies solely on variables obtainable without querying current pain status—is particularly relevant for opportunistic screening. It suggests that electronic health records (containing Age, BMI, and surgical history) could automatically flag individuals at high probability of structural OA for further evaluation, even before they present with complaints.

## The Role of "High-Tech" Variables

The "Virtual Maximum" analysis highlighted that adding objective measurements of body composition (muscle mass, bone mineral content) provided **zero** incremental benefit over simple BMI. This is a crucial finding for resource-limited settings, confirming that basic anthropometry is sufficient for risk stratification.

## Strengths and Limitations

Strengths of this study include the large sample size, high-quality radiographic assessment, and the rigorous comparison of multiple modeling techniques. A limitation is the cross-sectional design; these models identify *existing* (prevalent) disease, not future risk. Additionally, the specific Odds Ratios for variables like "History of Surgery" are high because they effectively identify "treated" disease, which is appropriate for a case-finding tool but should be distinguished from etiological risk factors.

# Conclusions

For the identification of prevalent radiographic knee osteoarthritis, simple linear models perform as well as complex machine learning approaches. A screening tool based on Age, BMI, and Clinical History can effectively prioritize individuals for radiographic evaluation with high discrimination (AUC > 0.80), supporting efficient resource allocation without the need for advanced diagnostics or "black box" algorithms.

# References
[References unchanged from original]
