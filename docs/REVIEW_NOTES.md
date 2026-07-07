# Review Notes — reviewer-response tracking & paper↔code reconciliation

Record of the critical review (code + manuscript) and the decisions taken. Split
into (A) code/doc fixes already applied, (B) paper numbers/wording to reconcile,
and (C) scientific-judgment items that change a reported result and therefore
await the author's decision.

The **current paper is `manuscript/newmanuscript.docx`** (newer than the tracked
`manuscript/manuscript.md`, which is a stale earlier draft). Numbers below refer
to the `.docx`.

---

## A. Applied in code/docs (safe — no scientific claim changed)

| Item | What was wrong | Fix |
|------|----------------|-----|
| Cluster bootstrap under-dispersed CIs | `auc/brier_ci_bootstrap_by_group` used `np.isin`, collapsing duplicate participant draws → CIs ~25–33% too narrow | Index-expansion resampling preserving multiplicity (`evaluation.py`); `utils.py` de-duplicated to re-export |
| CIs regenerated | scripts 10 & 11 carried the too-narrow CIs | Re-ran both; corrected CIs committed |
| `clean_sex` returned `None` | out-of-domain sex codes returned Python `None`, not `NaN` | added `return np.nan` (`data.py`) |
| `plots.py` no-op | `if isinstance(model, ...Figure): pass` dead line | removed |
| `scripts/10` silent self-check | consistency check passed on a 0-row merge | warns/flags incomplete or mismatched merges |
| `_make_fixtures.py` broken | imported pre-refactor `oarsi_data`, wrote only 7/14 fixtures | rewritten to current layout; emits all 14; fails loudly if source results missing |
| Stale doc references | README "01..06", test skip msg "02_run_comparison", archive off-by-one | corrected to 01..11 / 03 / 03–07 |
| Docs added | no single doc explained steps/decisions | `docs/METHODOLOGY.md` (this pass) |

Regenerated CIs (corrected cluster bootstrap, Constitutional / Without-Symptoms):

| Model | AUC | 95% CI (corrected) | Brier | Brier 95% CI |
|-------|-----|--------------------|-------|--------------|
| Stepwise LR | 0.810 | **0.790–0.830** (was 0.795–0.825) | 0.094 | 0.087–0.101 |
| XGBoost | 0.803 | 0.783–0.823 | 0.095 | 0.088–0.102 |
| Random Forest | 0.785 | 0.764–0.805 | 0.115 | 0.110–0.121 |
| Neural Network | 0.742 | 0.715–0.766 | 0.109 | 0.101–0.117 |

---

## B. Paper ↔ code reconciliation (edit the manuscript)

| Manuscript | Value in code/output | Action |
|------------|----------------------|--------|
| Headline LR CI "0.795–0.825" | corrected bootstrap → **0.790–0.830** | update the CI (point 0.810 unchanged) |
| "18.1% [of 5,652 knee radiographs] met the criteria" (Results) | 18.1% is the **participant** rate; knee-level is **13.2%** (matches abstract) | reword to name the denominator |
| Constitutional Brier "0.091" / Symptom "0.090" | reproducible values for those exact models are **0.094** / **0.092** | use the reproducible values, or state which model the Brier is for |
| Supplementary Table S2 "27 non-zero LASSO predictors" | current file has **28** (LASSO refresh correctly added `family_history`, coef 0.010) | regenerate S2 from `lasso_coefficients_clinical.csv` (28 rows) |
| "5-fold **nested** cross-validation" (×2) | single flat GroupKFold, no inner loop | "5-fold GroupKFold cross-validation"; disclose selection-on-full-data |
| "bidirectional stepwise (AIC/BIC)" | forward-only greedy max-CV-AUC after LASSO screen | describe the actual procedure |
| "missing WOMAC imputed to 0" | WOMAC is **median-imputed** in the pipeline (it is not 0-filled); the 0-coding applies to the binary FDR/DME items instead | correct the missing-data sentence |
| Virtual Maximum "added ... Waist-Hip Ratio" | WHR is an ordinary feature, not in `BIO_VARS` | fix the VM variable list |
| Confirmed **correct** | N=5,652; mean age 56.0±8.9; Table 1; Table 2 raw+std ORs; all 8 AUCs; κ=0.755; frequent_symptoms OR 1.58 | no action |

---

## C. Scientific-judgment items (await author decision — each changes a result/claim)

Do **not** silently apply; each alters a reported number or scientific claim.

1. **WOMAC in the ML "Constitutional" arm** — ML models received WOMAC symptom
   subscales the LR did not (contradicts the scenario definition). Fixing
   (exclude `WOMAC_VARS` in `runner.run_comparison`) would change the ML
   Screening/Case-Finding AUCs — likely *widening* the LR advantage.
2. **Virtual Maximum tautology** — VM ≡ With-Symptoms as coded; either build VM
   as With-Symptoms *plus* an otherwise-excluded `BIO_VARS`, or drop the
   bioimpedance-incremental-value claim.
3. **Surgery/trauma missing→0** — source items are heavily missing with no "no"
   category; report true missingness and a responders-only/missing-indicator
   sensitivity (drop-surgery already run in `scripts/11`).
4. **ML not tuned** — add inner-CV hyperparameter search, or state models used
   fixed defaults (weakens "complexity did not help").
5. **True nested CV** — move selection inside outer folds to measure the
   optimism, or caveat the current single-CV design and drop "nested."
6. **Paired AUC-difference test** — needed to support "equivalent or superior".
7. **Report the deployed 5-var model's AUC** (0.8167 mean-of-folds) distinctly
   from the 6/9-var full-set model.
8. **Class-weight consistency** — only RF uses `class_weight='balanced'`.
9. **Overclaims** — "validate" (only internal), EHR auto-flag deployment,
   "in all scenarios" (LR not run in VM), OR-scale mixing in the Discussion.
10. **Clinical-utility metrics** — sensitivity/specificity/PPV/NPV at a threshold
    + decision-curve analysis for a screening/triage claim.

See `docs/METHODOLOGY.md` §7 for the code locations of each caveat.
