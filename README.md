# Knee OA Screening (ELSA-Brasil MSK)

Reproducibility supplement for the manuscript

> **Clinical-Epidemiological Screening Tool for Prevalent Radiographic Knee Osteoarthritis: Comparison of Traditional and Machine Learning Approaches in the ELSA-Brasil Cohort**
> Domingues JG, Veloso AA, Telles RW, Barreto SM. 2026.

The study asks whether complex machine learning models (XGBoost, Random
Forest, Neural Network) actually beat simple Stepwise Logistic Regression
for identifying prevalent radiographic knee OA from routinely available
clinical-epidemiological variables. Across Screening, Case Finding, and
Virtual Maximum scenarios (n=2,830 participants / 5,652 knees), the simple
model matches or beats the complex ones (AUC 0.810 vs 0.803 / 0.785 / 0.742
in the Screening scenario).

The manuscript is in [`manuscript/`](manuscript/) — the current version is
`newmanuscript.docx`; `manuscript.md` is an earlier draft.

**Before citing the numbers, read [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)**
(what each step does + every design decision and known caveat),
[`docs/REVIEW_NOTES.md`](docs/REVIEW_NOTES.md) (paper↔code reconciliation and
open items), and **[`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md)** — the
resume anchor: the full decision log, current state, open next steps, and how
to run everything.

## Layout

```
src/koa_screening/    Library code (data prep, models, evaluation, plots)
scripts/01..11        Thin CLI runners (01-07 core pipeline, 08-11 sensitivity/robustness)
data/raw/             Restricted ELSA-Brasil CSV (not committed)
data/codebook/        Variable dictionary (CSV + original xlsx)
results/comparison/   Manuscript Table 1 + Figure 2 outputs
results/final_analysis/  Audit reports, OR tables, permutation importance
supplementary/        KL/PF/symptom distribution plots
tests/                Synthetic-data smoke tests + real-data regression suite
archive/              Superseded code and earlier drafts (see archive/README.md)
manuscript/           Submitted manuscript (.md and .docx)
```

## Install

Tested on Python 3.13 (3.12 also supported).

```sh
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell
# . .venv/bin/activate            # macOS/Linux
pip install -e ".[dev]"
```

## Get the data

`data/raw/stataToCsvMG.csv` is access-controlled (ELSA-Brasil). See
[`data/README.md`](data/README.md) for how to obtain it. Drop the CSV at
that exact path; everything else is automatic.

## Reproduce the manuscript numbers

Run in order:

```sh
python scripts/01_prepare_data.py            # validate + audit reports
python scripts/02_feature_selection.py       # LASSO -> MPMS -> stepwise selection (writes intermediates)
python scripts/03_run_comparison.py          # 4 models x 3 scenarios -> results/comparison/
python scripts/04_final_model_or.py          # raw OR + 95% CI for Table 1
python scripts/05_permutation_importance.py  # feature importance for final model
python scripts/06_table1_descriptives.py     # Table 1 descriptives (stdout)
python scripts/07_figures.py                 # composite manuscript figures
```

Step 02 must run before 03-07: it writes the selection intermediates
(`stepwise_mpms_clinical.csv`, `mpms_features_for_ci.csv`,
`final_5var_features_for_ci.csv`, `lasso_coefficients_*.csv`) that the later
steps consume. These are committed to `results/final_analysis/` so the
numbers are inspectable without re-running, but a from-scratch reproduction
must regenerate them with step 02 first.

**The headline comparison is the nested CV (step 12)** — leak-free (selection
and hyperparameter tuning happen inside each outer fold) and symmetric. The
sensitivity / robustness analyses:

```sh
python scripts/12_nested_cv.py                # HEADLINE: nested CV + tuned ML + paired ΔAUC tests
python scripts/08_sensitivity_isolated_pf.py  # isolated-PF exclusion (3 variants)
python scripts/09_seed_stability.py           # AUC stability over seeds 0..9
python scripts/10_model_ci_brier.py           # single-CV AUC 95% CI + Brier for every model
python scripts/11_sensitivity_drop_surgery.py # AUC + ORs without history_surgery
```

Headline nested-CV AUC (`results/comparison/nested_cv_summary.csv`):

| Scenario | Stepwise LR | XGBoost | Random Forest | Neural Network |
| --- | --- | --- | --- | --- |
| Screening (Without Symptoms) | **0.809** (0.789–0.828) | 0.799 | 0.796 | 0.776 |
| Case-Finding (With Symptoms) | **0.820** (0.800–0.839) | 0.813 | 0.812 | 0.808 |

Paired ΔAUC (LR − ML): Screening LR > all three (p=0.034/0.011/0.001);
Case-Finding LR comparable to XGB/RF (p=0.090/0.063), > MLP (p=0.015). The
single-CV `summary_all_models.csv` is a **diagnostic cross-check** only (see
`results/comparison/README.md`).

Constitutional-model Odds Ratios (`results/comparison/or_raw_Without_Symptoms.csv`):

| Variable | Raw OR (95% CI) |
| --- | --- |
| Age (per year) | 1.11 (1.10–1.12) |
| BMI (per kg/m^2) | 1.17 (1.15–1.20) |
| History of Knee Surgery | 8.69 (5.87–12.86) |
| History of Knee Trauma | 2.62 (2.06–3.32) |
| Waist–Hip Ratio | 0.06 (0.01–0.21) |
| Occupation (Category 4) | 0.66 (0.53–0.83) |
| Race (Category 3) | 0.69 (0.55–0.86) |

All numbers are locked at random seed `42`. Override with the `KOA_SEED`
environment variable for sensitivity analyses (write to a separate
`results/sensitivity_<name>/` folder so you don't overwrite the canonical
run).

## Tests

```sh
pytest -q -m "not requires_data"   # synthetic-data smoke tests (no real CSV needed)
pytest -q -m requires_data         # regression tests against the frozen baseline
```

The `requires_data` suite enforces, against snapshots in
`tests/fixtures/`, that any future change preserves:

- the full set of 70 post-prep columns (no variable is silently dropped);
- the per-scenario feature lists (57 / 60 / 60);
- the set of columns excluded by the 50% missingness filter (currently
  empty for this dataset);
- every AUC in `summary_all_models.csv` to +-1e-3;
- every Odds Ratio + 95% CI for the Without/With Symptoms scenarios;
- the feature-selection step's three core outputs
  (`stepwise_mpms_clinical.csv`, `mpms_features_for_ci.csv`,
  `final_5var_features_for_ci.csv`) byte-for-byte.

If you intentionally change behaviour, regenerate the fixtures with
`python tests/fixtures/_make_fixtures.py` (and re-snapshot the selection
fixtures from `results/final_analysis/` if the selection changed).

## Adding sensitivity analyses

Additional analyses (reviewer revisions, secondary analyses) are added as
their own runner under `scripts/`, writing to a dedicated
`results/<analysis_name>/` folder, without touching the canonical scripts.
See `scripts/08_sensitivity_isolated_pf.py` and `scripts/09_seed_stability.py`
for the pattern; the filters live in `src/koa_screening/sensitivity.py`.

> Note: `SCENARIOS`/`MODELS` in `src/koa_screening/config.py` are reference
> definitions of the canonical run; the comparison runner currently builds
> its scenario list internally rather than reading them, so editing those
> lists does not by itself change behaviour.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT - see [`LICENSE`](LICENSE).
