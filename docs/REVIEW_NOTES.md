# Review Notes — reviewer-response tracking & paper↔code reconciliation

Record of the critical review and the decisions taken, updated to the
**re-baseline** (revised TF+PF-KL outcome, leak-free nested CV, tuned ML). The
current paper is `manuscript/newmanuscript.docx`; `manuscript.md` is an earlier
draft.

---

## Headline numbers (final — nested CV, leak-free, tuned ML)

| Scenario | Model | Nested AUC (95% CI) |
|---|---|---|
| **Screening** | **Stepwise LR** | **0.809 (0.789–0.828)** |
| | XGBoost | 0.799 (0.779–0.818) |
| | Random Forest | 0.796 (0.775–0.816) |
| | Neural Network | 0.776 (0.753–0.799) |
| **Case-Finding** | **Stepwise LR** | **0.820 (0.800–0.839)** |
| | XGBoost | 0.813 (0.793–0.832) |
| | Random Forest | 0.812 (0.792–0.831) |
| | Neural Network | 0.808 (0.787–0.828) |

Paired ΔAUC (LR − ML): Screening LR **> all three** ML (p=0.034/0.011/0.001);
Case-Finding LR comparable to XGB/RF (p=0.090/0.063), > MLP (p=0.015).

Prevalence **14.0%** knee-level / **19.1%** participant-level (540/2,830); 5,650
knees. Bioimpedance (Virtual Maximum vs Case-Finding) adds only ~0.008 AUC.

---

## A. Applied in code (safe / methodological, all committed)

- **New outcome** from revised readings: `oa_knee = (TF KL≥2) OR (PF KL≥2)`.
- **Nested CV + inner-loop ML tuning** (`nested.py`, `scripts/12`) — leak-free,
  symmetric; audited for leakage. Paired ΔAUC test with cluster-bootstrap CI + p.
- **WOMAC excluded** from every model; **`_-1` missing dummies dropped**;
  **Virtual Maximum** a real bioimpedance contrast; **grouped LASSO** C-selection.
- **SES predictors added** (education, income) — not selected (occupation/race
  already capture the SES gradient); reportable negative finding.
- Cluster-bootstrap CI/Brier bug fixed (was ~25% too narrow); `clean_sex`,
  `run_mpms` empty guard, p-value smoothing, dead code removed.
- `run_mpms` → `run_forward_stepwise` (alias kept); "MPMS" is **not** a real
  technique — describe as *forward stepwise selection*. (Current .docx is
  already MPMS-free.)

## B. Paper ↔ code reconciliation (edit the manuscript — see the edit list I provided)

Sample 5,650 knees / prevalence 14.0%; Table 1 (n=540 rKOA); Table 2 = 7-var
Constitutional model (age, BMI, surgery, trauma, occupation, waist-hip ratio,
race); nested AUCs + paired tests above; reading-protocol Methods (revised
readings); "forward stepwise" not "MPMS"; bioimpedance "negligible" not "no
value"; reframe "nested CV" (now true) and soften "equivalent" per the paired
tests.

## C. Reviewer critique — final status

Resolved by the re-baseline: all-model CIs, drop-surgery quantified, bioimpedance
reworded, event counts, nested CV (leakage), tuned ML, paired difference test,
WOMAC scenario definition, grouped CV everywhere.

Remaining (text / optional code): calibration slope-intercept + reproducible
plot; sensitivity/specificity/PPV/NPV + decision-curve analysis (for the
"screening" claim); external/temporal validation (acknowledge — single cohort);
stepwise-instability caveat (per-fold features saved); OR-scale consistency;
humbler framing; surgery reverse-causation acknowledgment.
