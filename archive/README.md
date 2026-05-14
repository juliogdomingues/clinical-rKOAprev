# Archive

Everything in this folder is **superseded or exploratory** and is *not* part
of the canonical analysis published in the manuscript. Kept on disk for
historical reference and because future reviewers may ask about specific
sensitivity analyses. The canonical code lives in `src/koa_screening/` and
`scripts/` at the repo root.

## `scripts/` — superseded Python files

| File | What it was | Replaced by |
| --- | --- | --- |
| `oarsi_data.py` | Original data-prep module | `src/koa_screening/data.py` (identical logic, package import) |
| `oarsi_reporting.py` | Audit/report writers | `src/koa_screening/reporting.py` |
| `oarsi_utils.py` | AUC CI helpers | merged into `src/koa_screening/evaluation.py` |
| `oarsi_analysis.py` | LASSO + MPMS feature selection | `src/koa_screening/features.py` |
| `common_preprocess.py` | `PreparedData` wrapper | `src/koa_screening/preprocess.py` |
| `run_comprehensive_comparison.py` | The original master driver | split into `src/koa_screening/{models,evaluation,runner}.py` + `scripts/02_run_comparison.py` |
| `calculate_or_raw.py` | Cluster bootstrap of raw ORs | `src/koa_screening/odds_ratios.py` + `scripts/03_final_model_or.py` |
| `permutation_importance.py` | Permutation importance script | `src/koa_screening/importance.py` + `scripts/04_permutation_importance.py` |
| `generate_table1.py` | Descriptive Table 1 | `scripts/05_table1_descriptives.py` |
| `final_figure_abstract.py` | Composite manuscript figures | `src/koa_screening/plots.py` + `scripts/06_figures.py` |
| `auc_ci_bootstrap_eval.py` | Separate AUC-bootstrap experiment | merged into `src/koa_screening/evaluation.py` |
| `calculate_or_ci.py` | Earlier OR-CI implementation | superseded by `calculate_or_raw.py` (now in `src/koa_screening/odds_ratios.py`) |
| `experiment_neural_network.py` | Early NN prototyping | superseded by `runner.py` |
| `run_rf_xgb.py` | Standalone RF/XGBoost run | superseded by `runner.py` |
| `investigate_sex_lasso.py` | Sex-stratified LASSO sensitivity | exploratory; not in manuscript |
| `check_prevalence.py` | Prevalence sanity check | one-off |
| `check_sex_stats.py` | Sex × KOA univariate tests | exploratory |
| `list_variables.py` | Variable inventory print-out | replaced by `data/codebook/variable_codebook.csv` |
| `complete_finalatual.py` | Intermediate orchestration glue | superseded by `scripts/02_run_comparison.py` |

## `notes/` — superseded prose

| File | What it was |
| --- | --- |
| `ELSA screening tool KOA - abstract OARSI fev 26.md` | Earlier OARSI abstract draft |
| `results.md` | Earlier results writeup; numbers do *not* match the current manuscript |
| `abstract` | Earlier abstract draft |
| `thesis_proposal_revised.md` | Thesis proposal (different scope from this paper) |
| `debug_last_run.txt`, `debug_output.txt` | Stdout dumps from earlier runs |

## `media/`

| File | What it was |
| --- | --- |
| `screenshot.png` | Earlier results screenshot |
| `WhatsApp Audio 2026-02-11 at 00.44.01.mp4` | Voice note from co-author discussion |

## `results_tree_models/`

Output folder from an intermediate tree-model run. Superseded by
`results/comparison/` (which is what the manuscript cites).
