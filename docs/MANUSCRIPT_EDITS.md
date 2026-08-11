# Manuscript edit list — reconciling `newmanuscript.docx` with the current analysis

The `.docx` (last modified 13 May) predates the analysis re-baseline, so every
quantitative claim below has drifted. Each row is *manuscript text → replacement*.
Values verified against the committed outputs on the date of this file.

Source of truth: `results/comparison/nested_cv_summary.csv` (headline AUCs),
`nested_cv_paired_diff.csv`, `or_raw_*/or_standardized_*.csv` (Table 2),
`results/final_analysis/data_prep_summary.csv` (sample), `lasso_coefficients_clinical.csv` (Table 4).

---

## A. Sample, prevalence, denominators

| Manuscript | Replace with |
|---|---|
| "5,652 knee radiographs" (Abstract Results, Methods, Table 2 note, Table 3 header) | **5,650** |
| "(8 excluded due to arthroplasty or technical artifacts)" | recheck: 7 knees excluded for arthroplasty (KL/PF code 6); 573 knee-rows dropped for missing outcome in both compartments |
| "The prevalence of radiographic KOA was 13.2%" (Abstract) | **14.0%** (knee-level) |
| "Of the 5,652 knee radiographs … **18.1%** met the criteria for radiographic KOA" (Results) | **This conflates denominators.** Knee-level prevalence is **14.0%**; the participant-level figure is **19.1%** (540/2,830). Rewrite: "Of the 5,650 knees analysed, 14.0% met the criteria for rKOA; at the participant level, 540 of 2,830 participants (19.1%) had rKOA in at least one knee." |
| "26.8% (n=1,514) of the evaluated knees were symptomatic, among which the prevalence of radiographic KOA was 26.1%" | **26.8% (n=1,513)**; rKOA among symptomatic knees **28.0%** |
| "mean 56.0 years, SD 8.9", "aged 38–79" | unchanged ✅ (mean 56.1, SD 8.9, range 38–79) |

## B. Abstract & Results — model performance (now nested CV)

| Manuscript | Replace with |
|---|---|
| "six constitutional variables (age, body mass index, race, occupation, history of trauma, and history of surgery)" | **seven** variables — add **waist–hip ratio** |
| "achieved an Area Under the Curve (AUC) of 0.810 (95% CI 0.795–0.825)" | **0.809 (95% CI 0.789–0.828)** |
| "XGBoost (AUC 0.803), Random Forest (AUC 0.789), and MLP (AUC 0.742)" | **XGBoost 0.799 (0.779–0.818), Random Forest 0.796 (0.775–0.816), MLP 0.776 (0.753–0.799)** |
| "improved the logistic regression AUC to 0.824" | **0.820 (95% CI 0.800–0.839)**; XGBoost 0.813, RF 0.812, MLP 0.808 |
| "a Brier score of 0.091" / "(Brier score 0.090)" | **0.098** (Screening) / **0.095** (Case-Finding) |
| Figure 1 caption "begins with Age (AUC 0.690) and reaches a plateau of 0.810 with the inclusion of six variables (Age, BMI, History of Surgery, History of Trauma, Occupation, and Race)" | Age **0.691**; the trajectory now runs age → bmi → history_surgery → frequent_symptoms → history_trauma → occupation → knee_disability → waist–hip ratio → recent pain → race (0.830). Rewrite for the current order and the k you report. |
| Figure 2/3 captions (AUCs) | update to the nested values above |

**ADD (new, and the strongest addition):** the paired ΔAUC tests.
> In the Screening scenario the logistic model was superior to all three tuned
> algorithms (ΔAUC vs XGBoost +0.010, 95% CI +0.001 to +0.018, p=0.034; vs Random
> Forest +0.013, +0.003 to +0.022, p=0.011; vs MLP +0.033, +0.017 to +0.049,
> p=0.001). In the Case-Finding scenario it was statistically indistinguishable
> from XGBoost (+0.007, −0.001 to +0.014, p=0.090) and Random Forest (+0.008,
> −0.001 to +0.017, p=0.063), and superior to the MLP (+0.012, +0.003 to +0.022,
> p=0.015).

## C. Table 2 (Constitutional model ORs) — now 7 rows

| Feature | Raw OR (95% CI) | Standardized OR (95% CI) |
|---|---|---|
| Age (per year) | 1.11 (1.10–1.12) | 2.54 (2.27–2.85) |
| BMI (per kg/m²) | 1.17 (1.15–1.20) | 2.13 (1.92–2.37) |
| History of Knee Surgery | 8.69 (5.87–12.86) | 1.49 (1.39–1.61) |
| History of Knee Trauma | 2.62 (2.06–3.32) | 1.41 (1.29–1.53) |
| Occupation (non-routine non-manual) | 0.66 (0.53–0.83) | 0.82 (0.73–0.91) |
| **Waist–hip ratio** (new) | 0.06 (0.01–0.21) → **report per-SD instead** | **0.78 (0.69–0.87)** |
| Race (White vs others) | 0.69 (0.55–0.86) | 0.83 (0.74–0.93) |

⚠️ **Do not report the waist–hip ratio raw OR per 1 unit.** WHR has SD ≈ 0.087
(range 0.68–1.25), so "per 1-unit" extrapolates ~11.5 SD — far beyond the data,
which is why the OR looks extreme (0.06). Report **per SD (0.78, 0.69–0.87)** and
note that, conditional on BMI (r = 0.44), the association is negative and should
be interpreted cautiously (possible collinearity/suppression).

Text: "Age (sOR 2.34) and BMI (sOR 1.89)" → **age sOR 2.54 (2.27–2.85), BMI sOR
2.13 (1.92–2.37)**. Symptom-Augmented: "Frequent Symptoms (OR 1.58; 1.15–2.17)"
→ **1.65 (1.21–2.25)**; "Age (sOR 2.33) and BMI (sOR 1.74)" → recompute from
`or_standardized_With_Symptoms.csv`.

## D. Table 4 / Supplementary S2 (LASSO)

"the LASSO model identified **27** variables with non-zero coefficients" →
**32** non-zero (of 65 candidates). Regenerate the table from
`results/final_analysis/lasso_coefficients_clinical.csv`.

## E. Methods — statements that no longer match the code

| Manuscript | Replace with |
|---|---|
| Outcome: "KL grade ≥ 2 for the tibiofemoral compartment and/or patellofemoral (PF) OA (defined as a definitive osteophyte …)" | Both compartments now use KL: **"KL grade ≥ 2 in the tibiofemoral compartment and/or KL grade ≥ 2 in the patellofemoral compartment"** |
| Radiographic assessment | **Add the revised-readings paragraph** (two-step protocol, κ=0.755, then re-review by a calibrated radiologist at the second-visit reading with a third calibrated reader when needed, allowing revision of the original KL classification). |
| "missing WOMAC scores were imputed to 0 assuming asymptomatic knees" | **WOMAC was excluded from all models** (~44% missing; symptom information is instead captured by the discrete symptom items). Also state that non-response on the binary history/symptom items is coded as absent. |
| "5-fold nested cross-validation" | ✅ **now accurate** — but describe it: outer 5-fold GroupKFold; within each outer training fold the LASSO+forward-stepwise selection is re-run for the logistic model and a randomised hyperparameter search (40 configurations, grouped inner CV) is run for each ML model. |
| RF "200 decision trees, maximum depth of 10"; XGB "100 estimators, learning rate 0.1, max depth 3"; MLP "(64, 32) … alpha=0.0001" | These are no longer fixed — **hyperparameters were tuned inside each training fold**. Replace with the search description + cite Supplementary Table Sy (`supplementary/ml_tuning.md`). |
| "L1-penalized Logistic Regression (LASSO) with 3-fold cross-validation" | **participant-grouped** 3-fold CV |
| "forward stepwise selection maximizing the cross-validated AUC" | ✅ keep (never reintroduce "MPMS" — it is not a real technique) |
| "Python 3.12.1 … scikit-learn (v1.8.0) and statsmodels (v0.14.6)" | **Python 3.13.0**, scikit-learn 1.8.0, statsmodels 0.14.6, xgboost 3.1.3 |
| Virtual Maximum "added … Skeletal Muscle Mass, Bone Mineral Content, Waist-Hip Ratio" | WHR is a **base** predictor, not a bioimpedance add-on. The Virtual Maximum adds only the three bioimpedance measures. |
| Bioimpedance "offered no incremental discriminative value" | **"a negligible increment (~0.008 AUC)"** — it is now a real contrast, not identical models |

## F. Discussion / framing

- Add the **tuning-converges-to-linear** paragraph (`supplementary/ml_tuning.md`).
- Add the **isolated-PF sensitivity** (`supplementary/pf_ablation.md`).
- Add the **SES negative finding**: education and income were candidate predictors
  but were never selected — occupation and race already capture the gradient.
- Add **selection stability**: per-fold selections share a stable core (age, BMI,
  surgery, trauma, occupation) with a variable periphery
  (`nested_cv_lr_fold_features.csv`).
- Add **drop-surgery sensitivity**: Screening AUC 0.815 → 0.797 (surgery removed)
  → 0.766 (surgery + trauma removed), remaining above the ML models.
- Soften: "performs equivalently" → "was at least as discriminative"; "no
  discriminative advantage" → "no clinically meaningful advantage"; "validating a
  transparent approach" → "internally evaluating"; temper EHR/triage deployment
  language for a cross-sectional prevalent-disease model.
- Add limitations: internal validation only (single cohort, no external/temporal
  validation); post-selection inference (ORs/CIs from the same data used for
  selection are optimistic); missing→0 coding for history items; calibration
  assessed by Brier only.
