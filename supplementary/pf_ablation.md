# Supplementary: Isolated Patellofemoral OA Sensitivity Analysis

Because the primary outcome counts a knee as positive when **either** the
tibiofemoral (TF) **or** the patellofemoral (PF) compartment reaches
Kellgren–Lawrence (KL) grade ≥ 2, we tested whether the model comparison is
robust to how isolated PF disease is handled. Three operationalisations were
evaluated, each re-running the full Screening (Constitutional) comparison
(`scripts/08_sensitivity_isolated_pf.py`).

**Supplementary Table Sx. Model discrimination (AUC) under isolated-PF sensitivity analyses (Screening scenario).**

| Definition | n knees | Prevalence | Stepwise LR | XGBoost | Random Forest | Neural Net |
|---|---|---|---|---|---|---|
| Primary analysis (TF **or** PF KL≥2) | 5,650 | 14.0% | **0.815** | 0.799 | 0.778 | 0.764 |
| Exclude isolated-PF **knees** (positive only via PF, TF KL<2; −86 knees) | 5,564 | 12.7% | **0.827** | 0.810 | 0.789 | 0.738 |
| Exclude isolated-PF **participants** (KOA only via PF; −50 participants) | 5,550 | 13.1% | **0.827** | 0.810 | 0.789 | 0.784 |
| Redefine outcome as **TF-only** (KL≥2, PF ignored) | 5,650 | 12.5% | **0.824** | 0.807 | 0.788 | 0.699 |

**Interpretation.** The parsimonious logistic model remained the most
discriminative approach under every definition, and its discrimination was
stable or slightly higher when isolated-PF disease was removed
(AUC 0.815 → 0.824–0.827), indicating that the primary result is not driven by
patellofemoral-only cases. The hierarchy relative to the complex algorithms was
preserved throughout. (AUCs are pooled out-of-fold estimates from the same
5-fold participant-grouped cross-validation used in the primary analysis.)
