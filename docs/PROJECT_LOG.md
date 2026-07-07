# Project Log — decisions, state, and how to resume

Resume anchor for this project (clinical-rKOAprev). Read this + `METHODOLOGY.md`
(what/why of the pipeline) + `REVIEW_NOTES.md` (reviewer-response tracking) to
pick up where we left off. Chronological, with rationale for each decision.

## What this is
Reproducibility supplement for a paper comparing a parsimonious **Stepwise
Logistic Regression** against **XGBoost / Random Forest / MLP** for identifying
prevalent radiographic knee OA (ELSA-Brasil MSK). Repo:
`github.com/juliogdomingues/clinical-rKOAprev` (public; link redacted in the
manuscript for double-blind review). Manuscript: `manuscript/newmanuscript.docx`
(current; `manuscript.md` is an old draft).

## Current headline numbers (nested CV — the number to report)
| Scenario | LR | XGB | RF | MLP |
|---|---|---|---|---|
| Screening | **0.809** (0.789–0.828) | 0.799 | 0.796 | 0.776 |
| Case-Finding | **0.820** (0.800–0.839) | 0.813 | 0.812 | 0.808 |

Paired ΔAUC (LR−ML): Screening LR **>** all three (p=0.034/0.011/0.001);
Case-Finding LR comparable to XGB/RF (p=0.090/0.063), **>** MLP (p=0.015).
Prevalence 14.0% knee / 19.1% participant (540/2,830); **5,650 knees**.

## Decision log (with rationale)

1. **Repo reorganization** — flat script pile → `src/koa_screening/` package +
   numbered `scripts/` runners + `tests/` + `archive/` (superseded code). *Why:*
   publishable, testable reproducibility supplement.
2. **Data handling** — raw ELSA CSV/`.dta`/xlsx gitignored (access-controlled);
   results kept (aggregate, safe). Regression fixtures freeze behaviour.
3. **Reviewer critique addressed** across several rounds (see REVIEW_NOTES §C).
4. **Cluster-bootstrap CI bug** — `np.isin` dropped resampled duplicates → CIs
   ~25% too narrow. Fixed to index-expansion (preserves multiplicity).
5. **Revised outcome (major)** — switched from `TF-KL≥2 OR binary-PF-OA` to
   `(revised TF KL≥2) OR (PF KL≥2)`, merging revised readings from
   `Base_complementar_1_julio.dta` (b_klpad/b_klpae=TF PA, b_klpd/b_klpe=PF
   Perfil). *Why:* the revised second-visit readings supply KL for both
   compartments. Chose **revised** TF (not original) to match the Methods.
   Prevalence 13.2%→14.0%.
6. **WOMAC excluded from all models** — was leaking into the ML "symptom-free"
   arm. *Why:* it's a symptom-severity instrument (~44% missing); the
   Symptom-Augmented arm uses the discrete symptom items instead.
7. **Virtual Maximum made a real contrast** — `BIO_VARS` reserved for VM only
   (was identical to With-Symptoms = tautology). Bioimpedance adds ~0.008 AUC
   → report as "negligible", not "no value".
8. **Missing-category `_-1` dummies dropped** — so the model can't use
   missingness as a predictor.
9. **SES predictors added** (education dummies, income continuous) from the
   `.dta`. *Decision:* education as dummies (pipeline-consistent). *Finding:*
   neither is selected — occupation/race already capture the SES gradient
   (clean negative result).
10. **Constitutional pool** — showed unconstrained vs restricted; with SES in the
    pool the selection lands on a clean 7-var model on its own (age, BMI,
    surgery, trauma, occupation, waist-hip ratio, race) — no metabolic var.
11. **Grouped LASSO** — the L1 penalty (C) selection is now GroupKFold
    (was ungrouped `cv=3`) — consistent with everything else.
12. **Nested CV + inner-loop tuning (headline)** — `nested.py`, `scripts/12`.
    Outer GroupKFold; inside each training fold the LR re-runs LASSO+forward-
    stepwise and the ML models run RandomizedSearchCV (40 configs × per-fold
    seed). Paired ΔAUC test. *Audited for leakage → confirmed leak-free.* *Why:*
    makes "nested CV" true, fair to the ML arm, and answers the paired-test
    demand. Tuning lifted RF/MLP (~+0.02) but they still don't beat LR.
    *Finding:* tuning drives all ML toward maximal regularization (≈linear).
13. **"MPMS" is not a real technique** — invented internal label (expanded
    inconsistently in old drafts as "Minimal Predictive Model Search" and
    "Multiple Peptide Mass Spectrometry"). Renamed `run_mpms`→
    `run_forward_stepwise` (alias kept). It's standard forward stepwise
    selection. Manuscript is already MPMS-free — keep it that way.
14. **Single-CV demoted** — `scripts/03` / `summary_all_models.csv` is a
    diagnostic cross-check; report the nested CV. See
    `results/comparison/README.md`.
15. **Isolated-PF ablation fixed + rerun** — filter used old binary `oapf==1`;
    now `oa_knee==1 & TF-KL<2` (coding-agnostic). Draft in
    `supplementary/pf_ablation.md`. LR robust (0.815→0.824–0.827).
16. **DATA-EXPOSURE fix** — `missing_outcomes_*_{ids,rows}.csv` held real
    `idelsa` IDs + per-knee outcomes; removed from tracking, gitignored, and
    **purged from all git history** (git-filter-repo + force-push); one example
    ID in an archived comment redacted. Remote history verified clean.

## Open / next steps (not yet done)
- **Manuscript `.docx` edits** — from the edit list (numbers, Methods reading
  protocol, paired-test table, 7-var Table 2, framing). Author task.
- Paste `supplementary/pf_ablation.md` into the manuscript supplementary.
- **Optional new analyses** (higher value than more ML tuning):
  calibration slope/intercept + reproducible plot; sensitivity/specificity/
  PPV/NPV at a threshold + decision-curve analysis (for the "screening" claim).
- Optional: bounded grid-edge tuning check (confirm ML ceiling); acknowledge
  external/temporal validation limitation in text.
- **Do NOT** keep tuning ML to try to beat the LR (p-hacking).

## How to resume / run
```sh
pip install -e ".[dev]"
# place restricted data: data/raw/stataToCsvMG.csv + Base_complementar_1_julio.dta
python scripts/01_prepare_data.py
python scripts/02_feature_selection.py   # must precede 03-07,12
python scripts/03_run_comparison.py       # ORs + figures (diagnostic AUCs)
python scripts/04_final_model_or.py
python scripts/05_permutation_importance.py
python scripts/06_table1_descriptives.py
python scripts/07_figures.py
python scripts/12_nested_cv.py            # HEADLINE (slow: nested CV + tuning)
python scripts/08_ / 09_ / 10_ / 11_      # sensitivity / robustness
pytest -q                                 # 35 tests; requires_data ones need the CSVs
python tests/fixtures/_make_fixtures.py    # regenerate fixtures after an intended change
```
Seed 42 (config.RND; `KOA_SEED` to override). Key modules: `data.py` (outcome/
prep), `features.py` (selection), `nested.py` (headline CV), `runner.py`
(single-CV + ORs), `sensitivity.py` (ablations), `config.py` (all constants).
