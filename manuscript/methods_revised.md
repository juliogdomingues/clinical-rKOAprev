# Methods — revised draft

Drop-in replacement for the Methods section, incorporating the revised
radiographic readings, the nested cross-validation, hyperparameter tuning,
missing-data handling, calibration and clinical-utility analyses, and the
reporting requirements of Osteoarthritis and Cartilage.

Terminology follows two constraints. The analysis is cross-sectional and
identifies prevalent radiographic disease; the words "predict", "prediction" and
"predictor" are therefore not used, and candidate variables are described as
such. The model is not presented as a screening instrument, since screening
implies application to an asymptomatic population followed over time; the
analysis instead quantifies how accurately routinely available clinical and
demographic information classifies knees with respect to concurrent radiographic
status.

Text marked **[NEW]** does not exist in the current manuscript. Text marked
**[REVISED]** replaces existing content.

---

## Study Design and Sample

This was a cross-sectional analysis of clinical, demographic and radiographic
data obtained at the first visit of the ELSA-Brasil Musculoskeletal Study
(ELSA-Brasil MSK, 2012–2014), an ancillary study of the Brazilian Longitudinal
Study of Adult Health. The baseline ELSA-Brasil MSK sample comprised 2,901
active and retired civil servants aged 38–79 years, unselected for musculoskeletal
or other medical conditions [13].

**[REVISED]** Participants were eligible for the present analysis if bilateral
knee radiographs had been obtained and were interpretable. Knees with total knee
arthroplasty were excluded, as were knees that could not be graded in either
compartment because of image quality or the presence of alterations precluding
classification. The analytical sample comprised 5,650 knees from 2,830
participants. Because both knees of a participant contribute correlated
observations, the knee was the unit of analysis and the participant was the unit
of clustering in all analyses.

**[NEW]** No a priori sample-size calculation was performed; all eligible
participants of the ELSA-Brasil MSK baseline were included. The analytical
sample contained 791 knees with prevalent radiographic knee osteoarthritis,
corresponding to approximately 113 events per variable for the final
seven-variable model and approximately 13 events per variable relative to the
full set of candidate variables considered during selection.

## Radiographic Assessment

**[REVISED]** Bilateral knee radiographs were acquired using a standardised
non-fluoroscopic digital protocol comprising a weight-bearing posteroanterior
fixed-flexion view and a lateral view. A positioning device developed for the
study was used for the fixed-flexion view to ensure reproducibility of joint
space width measurement [26].

Images were interpreted blinded to participant characteristics, following a
validated two-step protocol [13]. In the first step, two trained radiology
technologists independently reviewed all radiographs to identify possible knee
osteoarthritis. In the second step, all images flagged by at least one
technologist were reviewed by an experienced musculoskeletal radiologist, who
established the definitive diagnosis and grading. This process was calibrated
against an external expert reader, with substantial inter-observer agreement
(κ = 0.755; 95% CI 0.663, 0.847) [13]. Subsequently, at the reading of the
radiographs obtained at the second visit of ELSA-Brasil MSK, all images were
re-examined by a calibrated radiologist and discussed with a third calibrated
reader when required, with the possibility of revising the original
Kellgren–Lawrence classification. The revised classifications were used in the
present analysis.

## Outcome Definition

**[REVISED]** The outcome was prevalent radiographic knee osteoarthritis,
defined at the knee level as a Kellgren–Lawrence grade of 2 or higher in the
tibiofemoral compartment, assessed on the posteroanterior view, or a
Kellgren–Lawrence grade of 2 or higher in the patellofemoral compartment,
assessed on the lateral view. Grades denoting prosthesis, doubtful
significance, alterations precluding assessment, inadequate image quality or
unavailable images were treated as non-gradeable. Knees non-gradeable in both
compartments were excluded. The outcome reflects the structural status of the
joint at the time of imaging, irrespective of symptoms, and does not represent
incident or progressive disease.

## Candidate Variables

**[REVISED]** Candidate variables were selected from the domains identified in
the literature review and comprised demographic characteristics, anthropometry,
clinical and occupational history, lifestyle, metabolic and cardiovascular
markers, socioeconomic position, and bioimpedance-derived body composition.
Operational definitions and questionnaire wording are given in the Supplementary
Methods. Socioeconomic position was represented by participant and maternal
educational attainment and by household and per-capita income.

**[NEW]** The Western Ontario and McMaster Universities Osteoarthritis Index
(WOMAC) subscales were not included as candidate variables. They were
administered conditionally and were missing for approximately 44% of knees, and
they quantify symptom severity rather than the clinical and demographic
information the analysis was designed to evaluate. Knee symptoms were instead
represented by three discrete items: frequent knee symptoms, knee symptoms in
the preceding seven days, and knee-related activity limitation.

Three sets of candidate variables were defined a priori. The Constitutional set
comprised demographic, anthropometric, clinical-history and occupational
variables obtainable without enquiry about current knee symptoms. The
Symptom-Augmented set added the three symptom items. A third set additionally
included the bioimpedance-derived measures (skeletal muscle mass, bone mineral
content and mineral mass), which were otherwise withheld, in order to quantify
their incremental contribution.

## Statistical Analysis

### Data structure and missing data

**[REVISED]** Continuous variables with missing values were imputed with the
median of the corresponding training partition and the same value was applied to
the held-out partition, so that no information from held-out observations
entered the imputation. Categorical variables were represented as indicator
variables; indicators denoting a missing category were removed, so that
missingness could not itself contribute to classification. For the binary
clinical-history and symptom items, which were recorded only when the
corresponding event was reported, absence of a report was coded as absence of the
event. The proportion of missing values for every candidate variable is reported
in Supplementary Table S1. Complete-case analysis was not performed.

### Model development

**[REVISED]** Variable selection proceeded in two stages. First, an
L1-penalised logistic regression was fitted to reduce dimensionality and
attenuate collinearity, with the penalty selected by three-fold cross-validation
in which the folds were defined by participant. Second, variables retained by
the penalised fit were entered one at a time by forward stepwise selection, in
the order that maximised the cross-validated area under the receiver operating
characteristic curve, with a small penalty proportional to model size to favour
parsimony. The final model was an L2-penalised logistic regression using the
selected subset.

**[REVISED]** Three comparator algorithms were fitted: a random forest, an
extreme gradient boosting model, and a multilayer perceptron. To avoid
attributing differences in performance to inadequate configuration, the
hyperparameters of each comparator were tuned by randomised search over a
pre-specified space of 40 candidate configurations, evaluated by cross-validation
with folds defined by participant. The searched ranges and the configurations
selected are reported in Supplementary Table [n].

### Internal validation

**[NEW]** Performance was estimated by nested cross-validation. The data were
partitioned into five outer folds, with all knees from a given participant
assigned to the same fold. Within each outer training partition, and using those
data only, the full analytical procedure was repeated: penalised selection and
forward stepwise selection for the logistic model, and hyperparameter search for
each comparator algorithm. The resulting models were applied once to the
corresponding held-out partition. Estimates therefore reflect the performance of
the entire modelling procedure rather than of a fixed set of variables, and the
logistic and comparator models were subject to the same constraint. No external
or temporal validation was performed.

### Performance measures

**[REVISED]** Discrimination was quantified by the area under the receiver
operating characteristic curve computed on the pooled held-out estimates.
Calibration was assessed by the calibration slope and calibration-in-the-large,
obtained by regressing the observed outcome on the linear predictor, and by a
calibration plot of observed proportions against model-estimated probabilities by
decile. A calibration slope of 1 and a calibration-in-the-large of 0 denote
agreement between estimated and observed probabilities. The Brier score was
reported as an overall measure.

**[NEW]** The consequences of using model-estimated probabilities to select
knees for radiographic examination were examined in two ways. Sensitivity,
specificity, and positive and negative predictive values were computed at a
range of probability thresholds and at the threshold maximising the Youden
index. Decision-curve analysis was used to quantify net benefit across threshold
probabilities relative to the alternatives of examining all knees or examining
none, thereby weighting correctly identified cases against unnecessary
examinations at each threshold.

**[NEW]** Confidence intervals for all performance measures were obtained by
bootstrap resampling of participants with replacement (2,000 resamples), so that
the correlation between knees of the same participant was preserved.

### Comparison between modelling approaches

**[NEW]** Differences in discrimination between the logistic model and each
comparator were estimated as the difference in the area under the curve computed
on identical held-out observations, with confidence intervals obtained by
resampling participants and recomputing both quantities within each resample.
Six such comparisons were performed, corresponding to three algorithms in each of
two variable sets. These comparisons were specified in advance and are
interpreted jointly; no adjustment for multiplicity was applied, and the
confidence intervals should be interpreted accordingly.

### Measures of association

**[REVISED]** For the variables retained in the final model, odds ratios were
estimated by unpenalised logistic regression with standard errors clustered at
the participant level. Odds ratios are reported per standard deviation for
continuous variables, since a one-unit change is not interpretable for variables
whose observed range spans less than one unit, and per category for binary
variables. Because the same data were used to select and to estimate the model,
these estimates and their confidence intervals do not account for the selection
step and are presented as descriptive measures of association rather than as
inferential quantities.

**[NEW]** The assumptions of the logistic model were examined by inspecting the
linearity of the association between each continuous variable and the log-odds of
the outcome, and by assessing collinearity among the retained variables.

### Sensitivity analyses

**[NEW]** Three sets of sensitivity analyses were performed and are reported in
the Supplementary Material. First, because the outcome combined two compartments,
the analysis was repeated after excluding knees classified as affected solely
through the patellofemoral compartment, after excluding participants whose
disease was confined to that compartment, and after restricting the outcome to
the tibiofemoral compartment. Second, because a history of knee surgery may
reflect established disease rather than antecedent exposure, the analysis was
repeated with that variable removed, and with both surgical and traumatic history
removed. Third, the stability of the estimates with respect to the pseudorandom
number generator was assessed across ten seeds.

### Software

**[REVISED]** Analyses were performed in Python 3.13.0 using scikit-learn
1.8.0, statsmodels 0.14.6 and xgboost 3.1.3. The analysis code, the coefficients
of the final model and the aggregate result files required to reproduce all
tables and figures are openly available (see Data and code availability).

---

## Notes for the authors

1. **Terminology must be made consistent beyond the Methods.** The Abstract,
   Results, Discussion and Conclusions currently use "predict", "predictive",
   "screening" and "triage". Replacements that preserve meaning: *identify*,
   *identification*, *classify*, *classification*, *discriminate*; and, for the
   application, *selection of knees for radiographic examination*. The keyword
   "Screening" should be replaced, for example by "Diagnosis" or
   "Radiography".

2. **TRIPOD remains the applicable reporting guideline.** It distinguishes
   diagnostic models, which concern a condition present at the time of
   assessment, from prognostic models, which concern future events. The present
   study is of the former type. This can be stated once in the Methods, which
   also justifies citing TRIPOD without describing the work as prediction.

3. **Events per variable.** The figure of approximately 13 events per candidate
   variable relates to the full candidate set before selection and should be
   reported alongside the value for the final model, since the selection step
   consumed part of the available information.

4. **Supplementary Table [n]** for the tuned hyperparameters is drafted in
   `supplementary/ml_tuning.md`; the sensitivity analyses referred to above are
   drafted in `supplementary/pf_ablation.md` and the clinical-utility analyses in
   `supplementary/clinical_utility.md`.

5. **Word count.** This draft is 1,556 words and replaces a Methods section of
   approximately 1,100 words, a net increase of about 450 words. The current
   body text is approximately 3,233 words against a limit of 4,000, so the
   material can be accommodated, but the Discussion should be reviewed for
   redundancy once the Results are updated.

6. **Two residual uses of the word "predict" are retained deliberately.** Both
   are fixed technical terms in which the word does not denote forecasting:
   *linear predictor*, the standard designation of the quantity Xβ in a
   generalised linear model, and *positive and negative predictive value*, the
   established names of the two post-test probabilities. Substituting either
   would obscure the meaning. No other instance remains in the draft, and neither
   term describes the study as predicting future disease.

7. **Figures for the added analyses.** Calibration and decision-curve figures
   exist (`results/comparison/fig_calibration.png`,
   `fig_decision_curve.png`). The combined limit is eight tables and figures and
   the manuscript currently contains seven. If both are placed in the main text
   the limit is exceeded; assigning them to the Supplementary Material is the
   more economical option and is consistent with their supporting role.
