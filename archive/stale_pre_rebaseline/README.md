# Stale results — PRE-RE-BASELINE. Do not cite these numbers.

These files were produced (Jan–Feb 2026) by the **old analysis**, before:

- the outcome was revised to `(tibiofemoral KL>=2) OR (patellofemoral KL>=2)`
  using the revised radiographic readings (prevalence 13.2% -> 14.0%,
  5,652 -> 5,650 knees);
- WOMAC was excluded from the models and the missing-category dummies dropped;
- the Virtual Maximum scenario became a genuine bioimpedance contrast;
- the headline moved to the leak-free **nested** cross-validation with tuned ML.

They have **no producer** in the current pipeline (`scripts/01..13`) — their
producers live in `archive/scripts/`. They are kept only for provenance.

**Every number in these files is superseded.** Current results are in
`results/comparison/` and `results/final_analysis/`; see
`results/comparison/README.md` for which file is the headline, and
`docs/MANUSCRIPT_EDITS.md` for the current values to use in the paper.

Notably superseded here: `final_all_models_auc_ci.csv`, `final_comparison_table*.csv`,
`fig_abstract_6models.png`, `fig_abstract_mpms_*.png`, `fig_calibration_oof_*.png`
(replaced by the reproducible `results/comparison/fig_calibration.png` from
`scripts/13_clinical_utility.py`), `fig_roc_comparison_*.png`, and the
`*_ci.csv` LASSO diagnostics.
