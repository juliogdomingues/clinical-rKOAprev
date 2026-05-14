# Test fixtures

These files are the **frozen snapshot of the original analysis behaviour**.
They were generated on 2026-05-13 from the canonical pipeline at random
seed 42, *before* any of the refactor moved files around. The regression
suite in `tests/test_*` asserts that the new layout produces byte-identical
results to this snapshot.

If you intentionally change behaviour (different feature engineering,
different model hyperparameters, etc.), regenerate the fixtures:

```sh
python tests/fixtures/_make_fixtures.py
```

That overwrites every file here. Commit the diff alongside the code change
so reviewers can see what shifted.

## Files

| File | What it locks |
| --- | --- |
| `expected_columns_post_prep.txt` | The 70 columns produced by `load_and_prep_data` (sorted). Catches accidental column drops/renames anywhere in the preprocessing pipeline. |
| `expected_columns_scenario_without.txt` | 57 features the Screening / Without-Symptoms scenario uses. |
| `expected_columns_scenario_with.txt` | 60 features the Case-Finding / With-Symptoms scenario uses. |
| `expected_columns_scenario_virtual.txt` | 60 features for the Virtual Maximum scenario. |
| `expected_dropped_high_missing.txt` | The columns dropped by the 50%-missingness filter. Currently empty — locks that no column is silently lost. |
| `expected_summary_all_models.csv` | The 11 AUCs that appear in the manuscript (4 models x 3 scenarios, minus the stepwise/virtual-max combination which is not run). |
| `expected_final_model_or.csv` | Raw OR + 95% CI for the 5-variable final model (`results/final_analysis/final_model_or_raw_ci.csv`). |
| `expected_or_raw_Without_Symptoms.csv` | Table 1 raw ORs for the Screening scenario. |
| `expected_or_standardized_Without_Symptoms.csv` | Table 1 standardized ORs for the Screening scenario. |
| `expected_or_raw_With_Symptoms.csv` | Table 1 raw ORs for the Case-Finding scenario. |
| `expected_or_standardized_With_Symptoms.csv` | Table 1 standardized ORs for the Case-Finding scenario. |
| `fixture_metadata.json` | Generation date, source CSV path, n_rows / n_cols / prevalence, seed. |
| `_make_fixtures.py` | The generator script. |
