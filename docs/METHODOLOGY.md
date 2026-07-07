# Methodology & Design Decisions

This document explains **what each pipeline step does** and **why each modeling
decision was made**, including known limitations. It is the authoritative
reference for reviewers and future maintainers; the code is the source of truth,
and this file explains the intent behind it. Where the current code and the
manuscript disagree, that is flagged explicitly under *Known caveats*.

All results are produced at random seed **42** (`koa_screening.config.RND`,
overridable via the `KOA_SEED` env var for sensitivity runs).

---

## 1. Pipeline steps (`scripts/`)

| # | Script | Purpose | Key outputs |
|---|--------|---------|-------------|
| 01 | `01_prepare_data.py` | Load + validate the raw ELSA CSV; write data-prep audit reports | `results/final_analysis/data_prep_summary.csv`, missingness/exclusion audits |
| 02 | `02_feature_selection.py` | LASSO → MPMS → forward-stepwise variable selection (`features.run_analysis`) | `stepwise_mpms_clinical.csv`, `mpms_features_for_ci.csv`, `final_5var_features_for_ci.csv`, `lasso_coefficients_*.csv` |
| 03 | `03_run_comparison.py` | 4 models × 3 scenarios, 5-fold GroupKFold OOF AUC | `results/comparison/summary_all_models.csv`, ROC/importance/OR figures |
| 04 | `04_final_model_or.py` | Cluster-bootstrap raw ORs for the final model | `final_model_or_raw_ci.csv` |
| 05 | `05_permutation_importance.py` | CV permutation importance | `permutation_importance_*.csv`, figures |
| 06 | `06_table1_descriptives.py` | Participant-level Table 1 (mean±SD, n%, tests) | stdout (paste into manuscript) |
| 07 | `07_figures.py` | Composite abstract figure | `fig_abstract_combined.png` |
| 08 | `08_sensitivity_isolated_pf.py` | Isolated-patellofemoral exclusion (3 variants) | `results/sensitivity_*` |
| 09 | `09_seed_stability.py` | AUC stability over seeds 0–9 | `results/sensitivity_seed_stability/` |
| 10 | `10_model_ci_brier.py` | AUC 95% CI + Brier for **every** model/scenario | `summary_all_models_ci_brier.csv` |
| 11 | `11_sensitivity_drop_surgery.py` | AUC + ORs with `history_surgery` (± trauma) removed | `results/sensitivity_drop_surgery/` |

Steps 02→07 must run in order (02 writes the selection files 03–07 read). 08–11
are independent post-hoc analyses.

---

## 2. Data preparation (`src/koa_screening/data.py`)

- **Unit of analysis:** the **knee** (up to 2 per participant). 5,652 knees /
  2,830 participants after exclusions. Because a participant contributes two
  correlated knees, every evaluation groups by participant (`idelsa`) — see §5.
- **Outcome** (`oa_knee`): Kellgren–Lawrence grade ≥ 2 **or** definite
  patellofemoral OA. Knees with total arthroplasty (code 6) are dropped; knees
  missing **both** KL and PF outcomes are dropped. Knee-level prevalence is
  **13.2%** (≈746/5,652); participant-level prevalence is **18.1%** (512/2,830).
  *These two rates are different — do not interchange them.*
- **Continuous predictors:** left as `NaN` when missing and **median-imputed
  inside each CV training fold** (`SimpleImputer` is a pipeline step, refit per
  fold — leak-safe; see §5).
- **Binary history/symptom predictors** (`history_surgery`, `history_trauma`,
  `frequent_symptoms`, `recent_pain_7d`): derived by `get_bin()`, which codes
  **missing → 0** ("event not reported" ⇒ treated as absent), applied **globally
  at prep time** (not fold-wise). See *Known caveats*.
- **Categoricals** (race, occupation, smoking, activity, alcohol): expanded to
  dummy columns during prep; the raw columns are then excluded from modeling.

---

## 3. Feature selection (`src/koa_screening/features.py`)

Two-stage hybrid, run **once on the full dataset**:

1. **LASSO screen** — L1-penalized `LogisticRegressionCV` (3-fold) to drop
   near-zero coefficients and reduce dimensionality/collinearity.
2. **MPMS forward stepwise** — from the LASSO survivors, variables are added one
   at a time in the order that maximizes the 5-fold GroupKFold CV AUC. "MPMS"
   is an internal label for this greedy forward-selection-by-CV-AUC routine (it
   is **not** a published acronym; `run_mpms`/`run_stepwise_specific`).

The selected order is frozen in `stepwise_mpms_clinical.csv`; the final
**Constitutional model** uses the first 6 non-symptom variables (age, BMI,
history of surgery, history of trauma, occupation, race); the
**Symptom-Augmented model** adds the symptom variables (9 total). The 5-variable
"deployable" subset is a further truncation (see *Known caveats: k=5*).

**WOMAC** subscales are excluded from this selection by design (`config.WOMAC_VARS`),
because they encode symptom severity that would leak into a structural-disease
model.

---

## 4. Models (`src/koa_screening/models.py`)

Fixed hyperparameters (no tuning search) — pinned to reproduce the published run:

| Model | Configuration | Imbalance handling |
|-------|---------------|--------------------|
| Stepwise Logistic Regression | L2, `C=1.0`, `max_iter=3000` | none (`class_weight=None`) |
| Random Forest | 200 trees, `max_depth=10` | **`class_weight='balanced'`** |
| XGBoost | 100 trees, depth 3, `lr=0.1`, logloss | none |
| MLP | hidden (64, 32), ReLU, Adam `lr=0.001`, `alpha=1e-4`, early stopping | none |

All models sit behind a `SimpleImputer(median)` (and `StandardScaler` for
LR/MLP) inside one sklearn `Pipeline`, so preprocessing is refit per fold.

---

## 5. Evaluation (`src/koa_screening/evaluation.py`)

- **Cross-validation:** a single **5-fold `GroupKFold`** grouped by participant,
  so both knees of a person are always in the same fold (prevents bilateral-knee
  leakage). AUC is computed on the **pooled out-of-fold (OOF)** predictions.
  *This is a single CV loop, not a nested one* (there is no inner
  tuning/selection loop — see caveats).
- **Confidence intervals:** cluster bootstrap resampling **participants**
  (`idelsa`) with replacement, `n_boot=2000`. The resampler expands row indices
  so a participant drawn *k* times contributes its rows *k* times (a correct
  cluster bootstrap; an earlier version used `np.isin` and under-dispersed the
  CIs — fixed).
- **Brier score** (calibration summary) is computed the same way, per model.
- **Odds ratios** (`runner.calculate_odds_ratios`, `odds_ratios.py`): reported as
  **raw** and **standardized**, from an **unpenalized `statsmodels` Logit** with
  **cluster-robust SEs** at the participant level. This auxiliary inferential
  model differs from the penalized sklearn model used for discrimination (a
  standard, deliberate choice — the OR model is for interpretation, not the
  0.810 AUC).

---

## 6. Scenarios

| Scenario | Intent | Feature set (as coded) |
|----------|--------|------------------------|
| Without Symptoms ("Constitutional") | demographic + history only | all candidates minus `SYMPTOM_VARS` |
| With Symptoms ("Symptom-Augmented") | + patient-reported symptoms | all candidates |
| Virtual Maximum | + bioimpedance/advanced anthropometry | all candidates |

The **Stepwise LR** honors these definitions (its features come from the
WOMAC-free MPMS selection). The **ML models** in `runner.run_comparison` are
given the full candidate pool minus only `BASE_EXCLUDE`/`SYMPTOM_VARS` — see the
two caveats immediately below.

---

## 7. Known caveats (read before citing the numbers)

These are documented honestly so the repo is self-consistent. Items marked
**[affects reported numbers]** would change published values if fixed and are
therefore left as author decisions, not silently changed.

1. **Selection runs once on the full data (not nested).** LASSO→MPMS→stepwise is
   fit on all rows, then GroupKFold only refits LR *coefficients* per fold. The
   Stepwise AUC is thus mildly optimistic relative to a fully nested procedure,
   and the manuscript's "nested cross-validation" wording is inaccurate — it is
   a single 5-fold GroupKFold. Magnitude is likely small (few strong candidates)
   but is not measured. **[affects reported numbers if nested CV is implemented]**
2. **WOMAC leaks into the ML scenarios.** `runner.run_comparison` does not
   exclude `WOMAC_VARS`, so XGBoost/RF/MLP receive WOMAC pain/stiffness/function
   in *all* scenarios — including the "symptom-free" Constitutional arm — while
   the Stepwise LR does not. This contradicts the scenario definition and gives
   the ML models symptom data the LR lacked (yet they still underperform).
   **[affects reported ML AUCs]**
3. **Virtual Maximum ≡ With Symptoms.** As coded, both scenarios use the same
   full candidate list (bioimpedance is already in the base features), so their
   AUCs are byte-identical. The "bioimpedance adds no incremental value" claim is
   therefore a tautology rather than a contrast — a proper test needs an
   explicit no-bioimpedance baseline. **[affects the bioimpedance claim]**
4. **ML hyperparameters are fixed, not tuned.** "Increasing complexity did not
   help" is not a tuned-vs-tuned comparison; the MLP in particular may be
   undertuned. **[affects fairness of the comparison]**
5. **Missing → 0 for binary history/symptom items.** Non-response on surgery/
   trauma/symptom questions is coded as "absent." For `history_surgery` (the
   OR-8.10 headline) this can manufacture negatives from missing data. A
   drop-surgery sensitivity (`scripts/11`) shows the AUC holds (0.810→0.792→0.760)
   and age/BMI ORs are stable. **[interpretation]**
6. **No paired test of AUC differences.** "Equivalent or superior" rests on
   overlapping-CI point estimates, not a DeLong / paired cluster-bootstrap test.
7. **Calibration is summarized by Brier only** (no slope/intercept); the
   committed `fig_calibration_*.png` have no active producer.
8. **Internal validation only** — single cohort (ELSA-Brasil MSK); no external
   or temporal validation.
9. **OR model ≠ discrimination model** — Table 2 ORs come from an unpenalized
   Logit; the discrimination AUC comes from the penalized sklearn model.

The reviewer-response status of each item and the exact code locations are
tracked in `docs/REVIEW_NOTES.md`.

---

## 8. Reproducibility

`pip install -e ".[dev]"`, place the ELSA CSV at `data/raw/stataToCsvMG.csv`,
then run `scripts/01`→`07` (core) and `08`→`11` (sensitivity). `pytest -q`
locks the pipeline against regressions (`tests/`); real-data regression tests
require the CSV and are skipped otherwise. Fixtures are regenerated with
`python tests/fixtures/_make_fixtures.py` after an intentional behavior change.
