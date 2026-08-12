# Supplementary: Calibration and clinical utility

Discrimination alone does not establish that a model is usable for deciding who
should receive a radiograph. All results below are computed from the **nested
cross-validation out-of-fold predictions** (`scripts/13_clinical_utility.py`),
so they carry no in-sample optimism. Uncertainty uses the same participant-level
cluster bootstrap as the main analysis.

## 1. Calibration

**Supplementary Table Sz. Calibration of each model (nested-CV out-of-fold predictions).**

| Scenario | Model | Calibration slope (95% CI) | Calibration-in-the-large (95% CI) |
|---|---|---|---|
| Screening | **Stepwise LR** | **0.95 (0.87–1.04)** | **−0.02 (−0.13, +0.09)** |
| Screening | XGBoost | 1.05 (0.96–1.14) | +0.01 (−0.10, +0.11) |
| Screening | Random Forest | 1.16 (1.05–1.26) | −0.01 (−0.11, +0.09) |
| Screening | Neural Network | 0.79 (0.70–0.89) | −0.33 (−0.44, −0.22) |
| Case-Finding | **Stepwise LR** | **0.95 (0.87–1.03)** | **−0.01 (−0.12, +0.10)** |
| Case-Finding | XGBoost | 1.07 (0.98–1.16) | +0.01 (−0.10, +0.11) |
| Case-Finding | Random Forest | 1.15 (1.05–1.25) | −0.00 (−0.11, +0.10) |
| Case-Finding | Neural Network | 0.80 (0.73–0.87) | −0.01 (−0.12, +0.11) |

Ideal values are slope = 1 and calibration-in-the-large = 0.

**Interpretation.** The logistic model was well calibrated in both scenarios:
the calibration slope was consistent with 1 and calibration-in-the-large with 0
(both 95% CIs include the ideal value), and the calibration plot follows the
45° line across the full range of predicted risk (Supplementary Figure Sz1).
Among the complex models, XGBoost was also well calibrated; the Random Forest
was slightly under-dispersed (slope 1.16, 95% CI 1.05–1.26, excluding 1) and the
neural network was miscalibrated in the Screening scenario, producing both
over-dispersed and systematically over-estimated risks (slope 0.79, 0.70–0.89;
calibration-in-the-large −0.33, −0.44 to −0.22). The neural network was
therefore the weakest model on **both** discrimination and calibration.

## 2. Operating characteristics (Screening model)

**Supplementary Table Sz2. Performance of the Screening (Constitutional) logistic model at candidate decision thresholds.**

| Threshold | Knees referred (%) | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|---|
| 0.10 | 43.4 | 0.83 | 0.63 | 0.27 | 0.96 |
| **0.146 (Youden)** | **31.3** | **0.73** | **0.76** | **0.33** | **0.95** |
| 0.15 | 30.3 | 0.72 | 0.77 | 0.33 | 0.94 |
| 0.20 | 22.1 | 0.58 | 0.84 | 0.37 | 0.93 |
| 0.25 | 17.0 | 0.49 | 0.88 | 0.40 | 0.91 |
| 0.30 | 12.7 | 0.43 | 0.92 | 0.47 | 0.91 |
| 0.40 | 7.4 | 0.29 | 0.96 | 0.55 | 0.89 |
| 0.50 | 4.9 | 0.21 | 0.98 | 0.61 | 0.88 |

(95% CIs for every cell are in `results/comparison/threshold_metrics.csv`.)

**Interpretation.** At the Youden-optimal threshold (0.146) the model would refer
31% of knees for radiography while detecting 73% of prevalent rKOA, with a
negative predictive value of 95%. The consistently high NPV (0.88–0.96) is the
clinically relevant property for an imaging-prioritisation tool: a low predicted
probability reliably rules out structural disease. The modest PPV (0.27–0.61)
is an expected consequence of the 14% prevalence in this population-based sample.

## 3. Decision-curve analysis

**Scope and limitations of this analysis.** Net benefit expresses the trade-off
between correctly identifying prevalent disease and performing unnecessary
examinations, using the decision threshold to fix the rate of exchange between
the two errors. Two conditions limit its interpretation here. First, no
threshold probability has been established at which a knee radiograph would be
obtained in order to detect structural osteoarthritis, and current guidance holds
that imaging frequently does not alter management; the range of thresholds
examined is therefore illustrative rather than clinically anchored. Second, the
analysis treats the knee as the decision unit, whereas a radiographic
examination is requested per person and images both knees. The results below
should accordingly be read as a description of the model's operating behaviour,
not as a recommendation for practice, which would in any case require external
validation.

Across the full range of clinically plausible threshold probabilities
(0.02–0.50), using the model to select knees for radiography yielded higher net
benefit than either radiographing all knees or radiographing none
(Supplementary Figure Sz2). The maximum gain over the best default strategy
occurred at a threshold of 0.14 (net benefit +0.067), i.e. approximately seven
additional true cases identified per 100 knees imaged without any increase in
unnecessary radiographs. Above a threshold of ~0.15, "radiograph all" becomes
harmful (negative net benefit) while the model retains positive net benefit,
which is the situation in which a prioritisation rule is most useful.

---

## 4. Discrimination under a metric weighted toward the minority class

Because the outcome was present in 14% of knees, discrimination was additionally
summarised by average precision, the area under the precision–recall curve,
which weights performance on the minority class more heavily than the area under
the ROC curve. The baseline value of average precision equals the prevalence,
here 0.14.

**Supplementary Table Sz3. Area under the ROC curve and average precision (nested cross-validation).**

| Scenario | Model | ROC AUC | Average precision (95% CI) |
|---|---|---|---|
| Constitutional | Logistic regression | 0.809 | **0.452 (0.407, 0.495)** |
| Constitutional | XGBoost | 0.799 | 0.431 (0.388, 0.476) |
| Constitutional | Neural network | 0.776 | 0.431 (0.385, 0.474) |
| Constitutional | Random forest | 0.796 | 0.416 (0.373, 0.460) |
| Symptom-Augmented | Logistic regression | 0.820 | **0.476 (0.432, 0.521)** |
| Symptom-Augmented | Neural network | 0.808 | 0.464 (0.419, 0.507) |
| Symptom-Augmented | XGBoost | 0.813 | 0.452 (0.407, 0.495) |
| Symptom-Augmented | Random forest | 0.812 | 0.438 (0.395, 0.482) |

**Interpretation.** The logistic model had the highest average precision in both
variable sets, so the principal finding does not depend on the choice between the
two threshold-free measures. All models achieved roughly three times the
baseline value of 0.14. The ordering among the comparators, however, was not
preserved: the multilayer perceptron ranked lowest on the area under the ROC
curve but second on average precision in the Symptom-Augmented set, indicating
relatively better separation among the highest-scoring knees and relatively
poorer separation elsewhere. Statements about the relative performance of the
comparator algorithms should therefore be confined to their comparison with the
logistic model, which is consistent across measures. Average precision is not
reported in the main text because its baseline is the prevalence, which makes it
incomparable across populations with different prevalence and therefore
unsuitable as the primary summary in a study emphasising the need for external
validation.

---

## Draft text for the manuscript

**Methods (add to Statistical Analysis):**
> Calibration was assessed by the calibration slope and calibration-in-the-large,
> estimated from the out-of-fold predictions, together with a calibration plot of
> observed versus predicted risk by decile. Clinical utility was evaluated with
> decision-curve analysis, comparing the net benefit of model-based referral with
> the strategies of radiographing all or no knees, and by reporting sensitivity,
> specificity, and predictive values at candidate decision thresholds. Confidence
> intervals were obtained by bootstrap resampling of participants.

**Results (add after the discrimination paragraph):**
> The logistic model was well calibrated, with a calibration slope of 0.95
> (95% CI 0.87–1.04) and calibration-in-the-large of −0.02 (−0.13 to +0.09);
> predicted and observed risks agreed closely across the full range
> (Supplementary Figure Sz1). At the Youden-optimal threshold of 0.15, the model
> referred 31% of knees for radiography while identifying 73% of prevalent rKOA
> (specificity 76%, negative predictive value 95%). Decision-curve analysis
> showed a higher net benefit than radiographing all or no knees across all
> threshold probabilities between 0.02 and 0.50 (Supplementary Figure Sz2).
