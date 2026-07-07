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
| 02 | `02_feature_selection.py` | LASSO L1 screen → forward-stepwise variable selection (`features.run_analysis`) | `stepwise_mpms_clinical.csv`, `mpms_features_for_ci.csv`, `final_5var_features_for_ci.csv`, `lasso_coefficients_*.csv` (the `*_mpms_*` names are a legacy tag — they hold the **forward-stepwise** outputs) |
| 03 | `03_run_comparison.py` | 4 models × 3 scenarios, 5-fold GroupKFold OOF AUC | `results/comparison/summary_all_models.csv`, ROC/importance/OR figures |
| 04 | `04_final_model_or.py` | Cluster-bootstrap raw ORs for the final model | `final_model_or_raw_ci.csv` |
| 05 | `05_permutation_importance.py` | CV permutation importance | `permutation_importance_*.csv`, figures |
| 06 | `06_table1_descriptives.py` | Participant-level Table 1 (mean±SD, n%, tests) | stdout (paste into manuscript) |
| 07 | `07_figures.py` | Composite abstract figure | `fig_abstract_combined.png` |
| 08 | `08_sensitivity_isolated_pf.py` | Isolated-patellofemoral exclusion (3 variants) | `results/sensitivity_*` |
| 09 | `09_seed_stability.py` | AUC stability over seeds 0–9 | `results/sensitivity_seed_stability/` |
| 10 | `10_model_ci_brier.py` | AUC 95% CI + Brier for **every** model/scenario (single-CV) | `summary_all_models_ci_brier.csv` |
| 11 | `11_sensitivity_drop_surgery.py` | AUC + ORs with `history_surgery` (± trauma) removed | `results/sensitivity_drop_surgery/` |
| 12 | `12_nested_cv.py` | **Nested CV** (leak-free, tuned ML) + paired ΔAUC tests — the headline comparison | `nested_cv_summary.csv`, `nested_cv_paired_diff.csv`, per-fold features/params |

Steps 02→07 must run in order (02 writes the selection files 03–07 read). 08–12
are independent post-hoc analyses. **Step 12 (nested CV) is the primary,
leak-free model comparison** (see §5); step 03's single-CV comparison is a
faster secondary view whose LR AUC is mildly optimistic (selection ran on the
full sample).

---

## 2. Data preparation (`src/koa_screening/data.py`)

- **Unit of analysis:** the **knee** (up to 2 per participant). ~5,650 knees /
  2,830 participants after exclusions. Because a participant contributes two
  correlated knees, every evaluation groups by participant (`idelsa`) — see §5.
- **Outcome** (`oa_knee`): from the **revised radiographic readings** (merged
  from `data/raw/Base_complementar_1_julio.dta` on `idelsa`), a knee is positive
  if **either compartment** is Kellgren–Lawrence grade ≥ 2 — tibiofemoral
  (`b_klpad`/`b_klpae`, revised PA view) **or** patellofemoral (`b_klpd`/`b_klpe`,
  Perfil view). KL grades 5/6/7/8/9 (doubtful/prosthesis/non-gradeable) are
  excluded; arthroplasty (6) drops the knee; knees missing **both** compartments
  are dropped. Knee-level prevalence is **~14.0%**; participant-level ~19.1%.
  *These two rates differ — do not interchange them.* (If the complementary
  `.dta` is absent, `data.py` falls back to the legacy TF-KL + binary PF-OA
  outcome.)
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

1. **LASSO screen** — L1-penalized `LogisticRegressionCV` with **grouped**
   (participant) CV to drop near-zero coefficients and reduce collinearity.
2. **Forward stepwise** (`run_forward_stepwise`) — from the LASSO survivors,
   variables are added one at a time in the order that maximizes the 5-fold
   GroupKFold CV AUC (standard forward selection with a CV-AUC criterion; a
   legacy internal name for this was "MPMS", kept only as an alias / in some
   result filenames — describe it as *forward stepwise selection*, not "MPMS").

The selected order is frozen in `stepwise_mpms_clinical.csv` (forward-stepwise
output). Under the revised outcome the **Constitutional model** is data-driven
(≈7–8 non-symptom variables: age, BMI, history of surgery, history of trauma,
occupation, waist-hip ratio, race); the **Symptom-Augmented model** adds the
discrete symptom items. The 5-variable
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

## 5. Evaluation (`src/koa_screening/evaluation.py`, `src/koa_screening/nested.py`)

- **Primary comparison = nested cross-validation** (`nested.py`, `scripts/12`).
  An outer 5-fold `GroupKFold` (by participant) holds out each test fold; inside
  each outer *training* split, **both arms learn**:
    - LR: LASSO (grouped C-selection) + forward-stepwise re-run on the training
      split only,
    - XGBoost/RF/MLP: an inner `RandomizedSearchCV` (grouped) hyperparameter
      search (40 candidates × distinct per-fold seed ≈ 200 configs explored).
  The pooled outer-test predictions give an **unbiased, symmetric** estimate: no
  test row informs any selection/tuning, and both arms are handicapped
  identically. This is the number the manuscript's "nested cross-validation"
  refers to.
- **Secondary single-CV view** (`evaluation.cv_roc_auc`, `scripts/03`): one flat
  5-fold `GroupKFold` scoring a feature set / fixed hyperparameters chosen on the
  full data. Faster, but the LR AUC is mildly optimistic; kept as a cross-check.
- **Paired AUC-difference test** (`nested.paired_auc_diff`): cluster bootstrap of
  AUC(LR) − AUC(ML) on the *same* resampled participants, with an add-one-smoothed
  two-sided p-value — the formal evidence for "comparable / superior".
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

All arms use the same pool per scenario (`WOMAC_VARS` excluded everywhere;
`BIO_VARS` reserved for Virtual Maximum only):

| Scenario | Intent | Feature pool |
|----------|--------|--------------|
| Without Symptoms ("Constitutional") | demographic + anthropometric + history + occupational | `base_pool` − `SYMPTOM_VARS` |
| With Symptoms ("Symptom-Augmented") | + patient-reported symptoms | `base_pool` (= all − WOMAC − BIO) |
| Virtual Maximum | + bioimpedance/advanced anthropometry | `base_pool` + `BIO_VARS` |

`WOMAC_VARS` are excluded from every model (symptom-severity instrument, ~44%
missing) — the Symptom-Augmented arm instead uses the discrete symptom items
(`frequent_symptoms`, `recent_pain_7d`, `knee_disability`). Missing-category
dummies (`_-1`, created by the `fillna(-1)` one-hot step) are dropped so the
model can't use *missingness* as a predictor. Socioeconomic predictors
(education dummies; family/per-capita income, continuous — merged from the
complementary `.dta`) are in the pool but are not selected by the LR (occupation
and race already capture the SES gradient).

---

## 7. Known caveats (read before citing the numbers)

These are documented honestly so the repo is self-consistent. Items marked
**[affects reported numbers]** would change published values if fixed and are
therefore left as author decisions, not silently changed.

1. **Selection runs once on the full data (not nested).** LASSO→MPMS→stepwise is
   fit on all rows, then GroupKFold only refits LR *coefficients* per fold — so
   the **single-CV (step 03)** LR AUC is mildly optimistic. ✅ **RESOLVED** by the
   nested CV (step 12), which re-runs selection inside each outer fold and is the
   headline number; the single-CV view is kept only as a cross-check.
2. **WOMAC in the ML arm.** ✅ **RESOLVED** — `WOMAC_VARS` are now excluded from
   every model (§6), so the ML and LR arms see the same symptom-free pool in the
   Constitutional scenario.
3. **Virtual Maximum ≡ With Symptoms.** ✅ **RESOLVED** — `BIO_VARS` are now
   reserved for Virtual Maximum only, so it is a genuine "does bioimpedance add
   incremental value?" contrast (the answer is a small ~0.008 AUC increment —
   report as *negligible*, not *zero*).
4. **ML hyperparameters fixed, not tuned.** ✅ **RESOLVED** in the nested CV —
   each ML model gets an inner `RandomizedSearchCV` search per outer fold; even
   tuned, they do not outperform the LR.
5. **Missing → 0 for binary history/symptom items.** Non-response on surgery/
   trauma/symptom questions is coded "absent"; for `history_surgery` this can
   manufacture negatives. The drop-surgery sensitivity (`scripts/11`) shows the
   AUC holds and age/BMI ORs are stable. **[open — interpretation / text]**
6. **Paired AUC-difference test.** ✅ **RESOLVED** — `nested.paired_auc_diff`
   reports ΔAUC(LR − ML) with cluster-bootstrap CI + p-value per scenario.
7. **Calibration summarized by Brier only** (no slope/intercept); the committed
   `fig_calibration_*.png` have no active producer. **[open]**
8. **Internal validation only** — single cohort (ELSA-Brasil MSK); no external or
   temporal validation. **[open — acknowledge in text]**
9. **OR model ≠ discrimination model** — Table 2 ORs come from an unpenalized
   Logit; discrimination from the penalized sklearn model (standard, disclose).
10. **Stepwise-selection instability** — the nested per-fold feature lists
    (`nested_cv_lr_fold_features.csv`) show a stable core (age, BMI, surgery,
    trauma, occupation) with an unstable periphery; report this. **[open — text]**

The reviewer-response status of each item and the exact code locations are
tracked in `docs/REVIEW_NOTES.md`.

---

## 8. Reproducibility

`pip install -e ".[dev]"`, place the ELSA CSV at `data/raw/stataToCsvMG.csv`,
then run `scripts/01`→`07` (core) and `08`→`11` (sensitivity). `pytest -q`
locks the pipeline against regressions (`tests/`); real-data regression tests
require the CSV and are skipped otherwise. Fixtures are regenerated with
`python tests/fixtures/_make_fixtures.py` after an intentional behavior change.
