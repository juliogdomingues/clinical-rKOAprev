# `results/comparison/` — which file is the headline?

**Report the nested-CV numbers.** The nested cross-validation is the leak-free,
symmetric, tuned comparison; the single-CV files are a secondary cross-check.

| File | Role | Use |
|---|---|---|
| `nested_cv_summary.csv` | **HEADLINE** — nested CV, tuned ML, AUC + 95% CI + Brier | Report these AUCs |
| `nested_cv_paired_diff.csv` | **HEADLINE** — paired ΔAUC (LR − ML) with CI + p | Report these tests |
| `nested_cv_lr_fold_features.csv`, `nested_cv_ml_fold_params.csv` | Transparency — per-fold selections / hyperparameters | Supplementary |
| `or_raw_*.csv`, `or_standardized_*.csv` | **Table 2** — Odds Ratios (CV-scheme-independent) | Report |
| `summary_all_models.csv`, `summary_all_models_ci_brier.csv` | **Diagnostic** — single-CV AUCs (mildly optimistic) | Cross-check only; do **not** report as the headline |
| `roc_comparison_*.png`, `importance_*.png`, `stepwise_trajectory_*.png` | Figures | Optional |

Single-CV vs nested (Screening LR): 0.815 → **0.809** — the ~0.006 gap is the
selection optimism the nested CV removes.
