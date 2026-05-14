# Knee OA Screening (ELSA-Brasil MSK)

Reproducibility supplement for the manuscript

> **Clinical-Epidemiological Screening Tool for Prevalent Radiographic Knee Osteoarthritis: Comparison of Traditional and Machine Learning Approaches in the ELSA-Brasil Cohort**
> Domingues JG, Veloso AA, Telles RW, Barreto SM. 2026.

The study asks whether complex machine learning models (XGBoost, Random
Forest, Neural Network) actually beat simple Stepwise Logistic Regression
for identifying prevalent radiographic knee OA from routinely available
clinical-epidemiological variables. Across Screening, Case Finding, and
Virtual Maximum scenarios (n=2,830 participants / 5,655 knees), the simple
model matches or beats the complex ones (AUC 0.810 vs 0.803 / 0.785 / 0.742
in the Screening scenario).

The manuscript is in [`manuscript/`](manuscript/).

## Layout

```
src/koa_screening/    Library code (data prep, models, evaluation, plots)
scripts/01..06        Thin CLI runners — the reproduction recipe
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
python scripts/02_run_comparison.py          # 4 models x 3 scenarios -> results/comparison/
python scripts/03_final_model_or.py          # raw OR + 95% CI for Table 1
python scripts/04_permutation_importance.py  # feature importance for final model
python scripts/05_table1_descriptives.py     # Table 1 descriptives (stdout)
python scripts/06_figures.py                 # composite manuscript figures
```

Expected outputs (`results/comparison/summary_all_models.csv`):

| Scenario          | Stepwise LR | XGBoost | Random Forest | Neural Network |
| ----------------- | ----------- | ------- | ------------- | -------------- |
| Without Symptoms  | **0.810**   | 0.803   | 0.785         | 0.742          |
| With Symptoms     | **0.824**   | 0.809   | 0.789         | 0.765          |
| Virtual Maximum   | -           | 0.809   | 0.789         | 0.765          |

Final model Odds Ratios (`results/comparison/or_raw_Without_Symptoms.csv`):

| Variable                | Raw OR (95% CI)     |
| ----------------------- | ------------------- |
| Age (per year)          | 1.10 (1.09-1.11)    |
| BMI (per kg/m^2)        | 1.14 (1.12-1.17)    |
| History of Knee Surgery | 8.10 (5.46-12.03)   |
| History of Knee Trauma  | 2.47 (1.95-3.13)    |
| Race (Category 3)       | 0.69 (0.55-0.87)    |
| Occupation (Category 4) | 0.66 (0.53-0.83)    |

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
- every Odds Ratio + 95% CI for the Without/With Symptoms scenarios.

If you intentionally change behaviour, regenerate the fixtures with
`python tests/fixtures/_make_fixtures.py`.

## Adding sensitivity analyses

The plan is for additional tests (reviewer revisions, secondary analyses) to
be added without touching the canonical scripts. Add a new entry to
`SCENARIOS` in `src/koa_screening/config.py` and a new runner under
`scripts/`, writing to its own `results/<analysis_name>/` folder.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT - see [`LICENSE`](LICENSE).
