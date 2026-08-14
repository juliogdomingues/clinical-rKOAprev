# TITLE PAGE

**Title**
Comparative performance of linear and non-linear algorithms for identifying prevalent radiographic knee osteoarthritis: a cross-sectional analysis of the ELSA-Brasil Musculoskeletal Study

**Running title**
Linear vs. complex models for knee OA (37 characters)

**Authors**
Júlio Guerra Domingues [1], ORCID [ADD]
Adriano Alonso Veloso [2], ORCID [ADD]
Rosa Weiss Telles [3], ORCID [ADD]
Sandhi Maria Barreto [4], ORCID [ADD]

**Affiliations**
1. Department of Anatomy and Medical Imaging, Universidade Federal de Minas Gerais, Belo Horizonte, Brazil
2. Department of Computer Science, Universidade Federal de Minas Gerais, Belo Horizonte, Brazil
3. Department of Internal Medicine, Universidade Federal de Minas Gerais, Belo Horizonte, Brazil
4. Department of Preventive and Social Medicine, Universidade Federal de Minas Gerais, Belo Horizonte, Brazil

[ADD email address for every co-author, as required by the Guide for Authors §1.9.1.]

**Corresponding author**
Júlio Guerra Domingues. [ADD full postal address and telephone number.]
juliogdomingues@gmail.com

---

# ABSTRACT

**Objective.** To develop a parsimonious model for identifying prevalent radiographic knee osteoarthritis (rKOA) from routinely available clinical and demographic information, and to compare its discrimination with that of three machine-learning algorithms.

**Design.** Cross-sectional analysis of 5,650 knees from 2,830 participants of the ELSA-Brasil Musculoskeletal Study (2012–2014), a cohort of civil servants aged 38–79 years unselected for medical conditions. The outcome was rKOA, defined as a Kellgren–Lawrence grade of 2 or higher in the tibiofemoral or the patellofemoral compartment. Candidate variables spanned demographic, anthropometric, clinical-history, occupational, metabolic, socioeconomic and body-composition domains. A logistic model was developed by penalised selection followed by forward stepwise selection, and compared with a random forest, extreme gradient boosting and a multilayer perceptron. Discrimination and calibration were estimated by nested cross-validation with folds defined by participant, so that variable selection and hyperparameter tuning occurred within training partitions only.

**Results.** rKOA was present in 14.0% of knees and in 540 of 2,830 participants (19.1%). A model containing age, body mass index, history of knee surgery, history of knee trauma, occupational nature, waist–hip ratio and race achieved an area under the curve of 0.809 (95% CI 0.789, 0.828). The comparator algorithms did not achieve higher discrimination: extreme gradient boosting 0.799 (0.779, 0.818), random forest 0.796 (0.775, 0.816) and multilayer perceptron 0.776 (0.753, 0.799); differences relative to the logistic model were 0.010 (0.001, 0.018), 0.013 (0.003, 0.022) and 0.033 (0.017, 0.049), respectively. Adding self-reported symptoms increased the area under the curve to 0.820 (0.800, 0.839). The logistic model was well calibrated (calibration slope 0.95, 95% CI 0.87, 1.04), as was extreme gradient boosting (1.05, 0.96, 1.14); the random forest (1.16, 1.05, 1.26) and the multilayer perceptron (0.79, 0.70, 0.89) departed from the ideal value of 1.

**Conclusions.** A model of seven routinely available characteristics identified prevalent rKOA with discrimination at least equal to that of substantially more complex algorithms, and with adequate calibration. Increasing algorithmic complexity conferred no advantage in this setting. External validation is required before any application to clinical decision-making.

**Keywords.** Osteoarthritis, knee; Radiography; Epidemiology; Logistic models; Machine learning; ELSA-Brasil

[Word count: 349. Limit 350.]

---

# INTRODUCTION

Osteoarthritis is among the most prevalent musculoskeletal diseases worldwide, and knee involvement is a leading contributor to physical disability [1,2]. Symptoms such as pain, stiffness and loss of mobility carry substantial economic and social consequences [3–5].

Current guidance holds that knee osteoarthritis can often be identified from characteristic clinical features without routine imaging [6]. Radiographic assessment nevertheless remains a standard criterion for defining the structural presence of disease in research and epidemiological surveillance [7], and provides a morphological reference that correlates with subsequent joint degradation and functional decline [7,8]. Characterising structural disease therefore remains necessary for population-level monitoring, even where it does not alter individual clinical management.

Interest has grown in applying machine-learning algorithms and novel biomarkers, including bioimpedance-derived body composition, to the identification and classification of osteoarthritis [9,10]. Such models can represent non-linear structure and interactions, at the cost of interpretability and computational complexity [11,12]. Whether these methods offer a meaningful advantage over conventional regression when applied to standard clinical-epidemiological data has not been established for this outcome. Two features of the existing literature limit the available evidence: comparator algorithms are frequently reported without hyperparameter tuning, and variable selection for the regression comparator is often performed on the complete dataset before validation, which favours the regression model.

Population-based cohorts permit this comparison under conditions in which feasibility and interpretability are relevant. The ELSA-Brasil Musculoskeletal Study incorporates standardised knee radiography within a large, well-characterised sample of Brazilian adults [13].

The objectives of this study were to develop a parsimonious model for identifying prevalent radiographic knee osteoarthritis from routinely available clinical variables; to compare its discrimination and calibration with those of three machine-learning algorithms under an identical validation procedure; and to quantify the incremental contribution of self-reported symptoms and of bioimpedance-derived body composition.

---

# METHODS

## Study design and sample

This was a cross-sectional analysis of clinical, demographic and radiographic data obtained at the first visit of the ELSA-Brasil Musculoskeletal Study (ELSA-Brasil MSK, 2012–2014), an ancillary study of the Brazilian Longitudinal Study of Adult Health. The baseline ELSA-Brasil MSK sample comprised 2,901 active and retired civil servants aged 38–79 years, unselected for musculoskeletal or other medical conditions [13].

Participants were eligible if bilateral knee radiographs had been obtained and were interpretable. Knees with total knee arthroplasty were excluded, as were knees that could not be graded in either compartment because of image quality or the presence of alterations precluding classification. The analytical sample comprised 5,650 knees from 2,830 participants. Because both knees of a participant contribute correlated observations, the knee was the unit of analysis and the participant was the unit of clustering throughout.

No a priori sample-size calculation was performed; all eligible participants of the ELSA-Brasil MSK baseline were included. The analytical sample contained 791 knees with the outcome, corresponding to approximately 113 events per variable for the final seven-variable model and approximately 13 events per variable relative to the 59 candidate variables considered during selection.

## Ethical aspects

The principles of respect for persons, beneficence and justice were observed in the planning of ELSA-Brasil and ELSA-Brasil MSK, and the study was conducted in accordance with the Declaration of Helsinki. ELSA-Brasil was approved by the research ethics committees of the six participating institutions. The baseline of the ELSA-Brasil MSK ancillary study and its first follow-up were approved by the Research Ethics Committee of the Universidade Federal de Minas Gerais (amendment CAAE 0186.1.203.000-06 of proposal ETIC 186/06, 9 March 2012; and CAAE 47125015.4.1001.5149, opinion 1.897.023, 24 January 2017). All participants provided written informed consent.

## Radiographic assessment

Bilateral knee radiographs were acquired using a standardised non-fluoroscopic digital protocol comprising a weight-bearing posteroanterior fixed-flexion view and a lateral view. A positioning device developed for the study was used for the fixed-flexion view to ensure reproducibility of joint space width measurement [26].

Images were interpreted blinded to participant characteristics, following a validated two-step protocol [13]. In the first step, two trained radiology technologists independently reviewed all radiographs to identify possible knee osteoarthritis. In the second step, all images identified by at least one technologist were reviewed by an experienced musculoskeletal radiologist, who established the definitive diagnosis and grading. The process was calibrated against an external expert reader, with substantial inter-observer agreement (κ = 0.755; 95% CI 0.663, 0.847) [13]. Subsequently, at the reading of the radiographs obtained at the second visit of ELSA-Brasil MSK, all images were re-examined by a calibrated radiologist and discussed with a third calibrated reader when required, with the possibility of revising the original Kellgren–Lawrence classification. The revised classifications were used in the present analysis.

## Outcome definition

The outcome was prevalent radiographic knee osteoarthritis, defined at the knee level as a Kellgren–Lawrence grade of 2 or higher in the tibiofemoral compartment, assessed on the posteroanterior view, or a Kellgren–Lawrence grade of 2 or higher in the patellofemoral compartment, assessed on the lateral view. Grades denoting prosthesis, doubtful significance, alterations precluding assessment, inadequate image quality or unavailable images were treated as non-gradeable. Knees non-gradeable in both compartments were excluded. The outcome reflects the structural status of the joint at the time of imaging, irrespective of symptoms, and does not represent incident or progressive disease.

## Candidate variables

Candidate variables were identified from a review of the literature and comprised demographic characteristics, anthropometry, clinical and occupational history, lifestyle, metabolic and cardiovascular markers, socioeconomic position, and bioimpedance-derived body composition. Established associations of older age, female sex and higher body mass index with knee osteoarthritis [14–16], of previous knee injury and occupational mechanical exposure [14,15,20], of physical activity and high-impact sport [18–21], and of metabolic syndrome components [22–24] informed this selection. Family history of knee replacement was included as a proxy for shared genetic and environmental risk [25]. Socioeconomic position was represented by participant and maternal educational attainment and by household and per-capita income. Operational definitions and questionnaire wording are provided in the Supplementary Methods, and the distribution of every candidate variable is given in Supplementary Table S1.

The Western Ontario and McMaster Universities Osteoarthritis Index subscales were not included as candidate variables. They were administered conditionally and were missing for approximately 44% of knees, and they quantify symptom severity rather than the routinely available information the analysis was designed to evaluate. Knee symptoms were instead represented by three discrete items: frequent knee symptoms, knee symptoms in the preceding seven days, and knee-related activity limitation.

Three sets of candidate variables were specified in advance. The Constitutional set comprised demographic, anthropometric, clinical-history and occupational variables obtainable without enquiry about current knee symptoms. The Symptom-Augmented set added the three symptom items. A third set additionally included the bioimpedance-derived measures of skeletal muscle mass, bone mineral content and mineral mass, which were otherwise withheld, in order to quantify their incremental contribution.

## Statistical analysis

### Data structure and missing data

Continuous variables with missing values were imputed with the median of the corresponding training partition, and the same value was applied to the held-out partition, so that no information from held-out observations entered the imputation. Categorical variables were represented as indicator variables; indicators denoting a missing category were removed, so that missingness could not itself contribute to classification. For the binary clinical-history and symptom items, which were recorded only when the corresponding event was reported, absence of a report was coded as absence of the event. The proportion of missing values for every candidate variable is reported in Supplementary Table S1. Complete-case analysis was not performed.

### Model development

Variable selection proceeded in two stages. First, an L1-penalised logistic regression was fitted to reduce dimensionality and attenuate collinearity, with the penalty selected by three-fold cross-validation in which folds were defined by participant. Second, variables retained by the penalised fit were entered one at a time by forward stepwise selection, in the order that maximised the cross-validated area under the receiver operating characteristic curve, with a small penalty proportional to model size to favour parsimony. The final model was an L2-penalised logistic regression using the selected subset.

Three comparator algorithms were fitted: a random forest, an extreme gradient boosting model and a multilayer perceptron. To avoid attributing differences in performance to inadequate configuration, the hyperparameters of each comparator were tuned by randomised search over a pre-specified space of 40 candidate configurations, evaluated by cross-validation with folds defined by participant. The searched ranges and the configurations selected are reported in Supplementary Table S3.

### Internal validation

Performance was estimated by nested cross-validation. The data were partitioned into five outer folds, with all knees of a given participant assigned to the same fold. Within each outer training partition, and using those data only, the entire analytical procedure was repeated: penalised selection and forward stepwise selection for the logistic model, and hyperparameter search for each comparator. The resulting models were applied once to the corresponding held-out partition. Estimates therefore reflect the performance of the whole modelling procedure rather than of a fixed set of variables, and both the logistic and the comparator models were subject to the same constraint. No external or temporal validation was performed.

### Performance measures

Discrimination was quantified by the area under the receiver operating characteristic curve computed on the pooled held-out estimates. Calibration was assessed at two levels [Van Calster et al.]. The calibration slope, obtained as the coefficient of the linear predictor when the observed outcome is regressed upon it, describes whether estimated probabilities are too extreme (slope below 1) or too conservative (slope above 1); the ideal value is 1. The calibration intercept, or calibration-in-the-large, obtained with the linear predictor entered as an offset, describes whether probabilities are systematically too high (negative values) or too low (positive values); the ideal value is 0. Agreement across the range of estimates was examined graphically by plotting observed proportions against model-estimated probabilities by decile. The Brier score was reported as an overall measure.

The consequences of using model-estimated probabilities to select knees for radiographic examination were examined in two ways. Sensitivity, specificity and positive and negative predictive values were computed across a range of probability thresholds and at the threshold maximising the Youden index. Decision-curve analysis quantified net benefit across threshold probabilities relative to examining all knees or examining none, weighting correctly identified cases against unnecessary examinations at each threshold.

Confidence intervals for all performance measures were obtained by bootstrap resampling of participants with replacement (2,000 resamples), preserving the correlation between knees of the same participant.

### Comparison between modelling approaches

Differences in discrimination between the logistic model and each comparator were estimated as the difference in areas under the curve computed on identical held-out observations, with confidence intervals obtained by resampling participants and recomputing both quantities within each resample. Six such comparisons were performed, corresponding to three algorithms in each of two variable sets. These comparisons were specified in advance and are interpreted jointly; no adjustment for multiplicity was applied, and the confidence intervals should be interpreted accordingly.

### Measures of association

For the variables retained in the final model, odds ratios were estimated by unpenalised logistic regression with standard errors clustered at the participant level. Odds ratios are reported per standard deviation for continuous variables, because a one-unit change is not interpretable for variables whose observed range spans less than one unit, and per category for binary variables. Because the same data were used to select and to estimate the model, these estimates and their confidence intervals do not account for the selection step and are presented as descriptive measures of association rather than as inferential quantities. The assumptions of the logistic model were examined by inspecting the linearity of the association between each continuous variable and the log-odds of the outcome, and by assessing collinearity among the retained variables.

### Sensitivity analyses

Three sets of sensitivity analyses were performed and are reported in the Supplementary Material. First, because the outcome combined two compartments, the analysis was repeated after excluding knees affected solely through the patellofemoral compartment, after excluding participants whose disease was confined to that compartment, and after restricting the outcome to the tibiofemoral compartment. Second, because a history of knee surgery may reflect established disease rather than antecedent exposure, the analysis was repeated with that variable removed, and with both surgical and traumatic history removed. Third, the stability of estimates with respect to the pseudorandom number generator was assessed across ten seeds.

### Software

Analyses were performed in Python 3.13.0 using scikit-learn 1.8.0, statsmodels 0.14.6 and xgboost 3.1.3. The analysis code, the coefficients of the final model and the aggregate result files required to reproduce all tables and figures are openly available (see Data and code availability).

---

# RESULTS

## Participants

Of the 5,650 knees analysed from 2,830 participants, 791 (14.0%) met the criteria for prevalent radiographic knee osteoarthritis. At the participant level, 540 of 2,830 (19.1%) had the outcome in at least one knee. Participants with the outcome were older (60.7 years, SD 8.2, versus 55.0, SD 8.8) and had higher body mass index (29.3 kg/m², SD 5.2, versus 26.4, SD 4.4). Frequent knee symptoms were reported by 45.6% of participants with the outcome and 18.2% of those without; a history of knee trauma by 43.5% and 19.6%; and a history of knee surgery by 21.3% and 2.9%. The distribution of female sex did not differ appreciably between groups (55.4% versus 52.1%, p = 0.192). Characteristics are presented in Table 1, and the distributions of all candidate variables in Supplementary Table S1.

Of the 5,650 knees, 1,513 (26.8%) were symptomatic by at least one of the three symptom items; the outcome was present in 28.0% of these.

## Variables associated with prevalent rKOA

The penalised fit retained 32 of the 59 candidate variables (Supplementary Table S2). Forward stepwise selection identified seven variables for the Constitutional model: age, body mass index, history of knee surgery, history of knee trauma, occupational nature, waist–hip ratio and race. Their associations with the outcome are shown in Table 2.

Expressed per standard deviation, age (odds ratio 2.54; 95% CI 2.27, 2.85) and body mass index (2.13; 1.92, 2.37) showed the largest associations. Among the binary variables, a history of knee surgery showed the largest association (8.69; 5.87, 12.86), followed by a history of knee trauma (2.62; 2.06, 3.32). Non-routine non-manual occupation (0.66; 0.53, 0.83) and self-reported White race (0.69; 0.55, 0.86) were inversely associated with the outcome. Waist–hip ratio was inversely associated with the outcome once body mass index was included (0.78 per standard deviation; 0.69, 0.87); given the correlation between the two measures (r = 0.44), this estimate should be interpreted as conditional on body mass index rather than as an independent association.

Educational attainment and income were available as candidate variables but were not retained at any stage of selection, indicating that the socioeconomic gradient in this sample was already represented by occupational nature and race.

The incremental contribution of each variable is shown in Figure 1. Age alone yielded a cross-validated area under the curve of 0.692, rising to 0.756 with body mass index and 0.797 with a history of knee surgery. The four remaining variables together added 0.018, of which history of knee trauma accounted for 0.010 and race for 0.001. These values are the cross-validated estimates that guided selection and were computed on the full sample; the corresponding validated estimate, obtained by nested cross-validation, is reported below and is lower.

## Discrimination

Discrimination of the four modelling approaches under nested cross-validation is shown in Table 3 and Figure 2. In the Constitutional set, the logistic model achieved an area under the curve of 0.809 (95% CI 0.789, 0.828). The comparator algorithms did not achieve higher values: extreme gradient boosting 0.799 (0.779, 0.818), random forest 0.796 (0.775, 0.816) and multilayer perceptron 0.776 (0.753, 0.799). Differences relative to the logistic model, computed on identical held-out observations, were 0.010 (0.001, 0.018), 0.013 (0.003, 0.022) and 0.033 (0.017, 0.049) respectively.

In the Symptom-Augmented set the logistic model achieved 0.820 (0.800, 0.839). Differences relative to extreme gradient boosting and random forest were 0.007 (−0.001, 0.014) and 0.008 (−0.001, 0.017); the corresponding interval excluded zero only for the multilayer perceptron, 0.012 (0.003, 0.022). Adding self-reported symptoms therefore increased the area under the curve of the logistic model by 0.011.

Adding the bioimpedance-derived measures to the Symptom-Augmented set changed the area under the curve of extreme gradient boosting from 0.813 to 0.821, of the random forest from 0.812 to 0.815 and of the multilayer perceptron from 0.808 to 0.815; the largest of these increments was 0.008.

Hyperparameter tuning selected the most heavily constrained configurations available for each comparator: gradient-boosted trees of depth 2 with the maximum penalty in the search space, random forests with large terminal nodes and few variables per split, and single-hidden-layer networks (Supplementary Table S3).

## Calibration

The logistic model was well calibrated in both variable sets. In the Constitutional set the calibration slope was 0.95 (95% CI 0.87, 1.04) and calibration-in-the-large was −0.02 (−0.13, 0.09), both intervals including the values denoting agreement, and observed proportions followed model-estimated probabilities across the range of estimates (Supplementary Figure S1). The Brier score was 0.098. Among the comparators, extreme gradient boosting was similarly calibrated (slope 1.05; 0.96, 1.14), the random forest produced estimates that were insufficiently dispersed (1.16; 1.05, 1.26) and the multilayer perceptron was miscalibrated in both respects (slope 0.79; 0.70, 0.89; calibration-in-the-large −0.33; −0.44, −0.22). The algorithm with the lowest discrimination was therefore also the least well calibrated.

Operating characteristics at a range of probability thresholds, and an analysis of net benefit relative to examining all or no knees, are reported in the Supplementary Material. These quantities describe the consequences that would follow from using the model to select knees for radiographic examination; they are presented for completeness and are not intended to define a decision rule, since no threshold has been established for this decision and the analysis was conducted at the level of the knee rather than the person.

## Sensitivity analyses

Excluding knees affected solely through the patellofemoral compartment, excluding participants whose disease was confined to that compartment, and restricting the outcome to the tibiofemoral compartment produced areas under the curve of 0.827, 0.827 and 0.824 for the logistic model, each exceeding the corresponding comparator values (Supplementary Table S5). Removing history of knee surgery reduced the area under the curve from 0.815 to 0.797, and removing both surgical and traumatic history reduced it to 0.766; the ordering of the modelling approaches was unchanged. Estimates varied by less than 0.01 across ten seeds.

---

# DISCUSSION

In this cross-sectional analysis of a large population-based sample, a logistic model containing seven routinely obtainable characteristics identified prevalent radiographic knee osteoarthritis with an area under the curve of 0.809, and three machine-learning algorithms did not achieve higher discrimination. The differences favoured the logistic model in the Constitutional set and were compatible with no difference for two of three comparators once symptoms were included. All differences were smaller than 0.035, which is unlikely to be material for any practical purpose.

Two features of the analysis strengthen this comparison relative to previous reports. First, variable selection and hyperparameter tuning were performed within training partitions only, so that neither approach was advantaged by access to held-out observations; when selection is performed on the complete dataset the regression comparator is systematically favoured. Second, the comparator algorithms were tuned rather than fixed at default settings, so their performance cannot be attributed to inadequate configuration. The behaviour of the tuning procedure is itself informative: it selected the most constrained configurations available in each search space, that is, configurations approximating an additive, approximately linear structure. This is consistent with the interpretation that the association between these clinical characteristics and structural disease contains little non-additive structure for the algorithms to exploit.

The dominance of age and body mass index accords with established evidence identifying ageing and mechanical loading as principal determinants of knee osteoarthritis prevalence [14–17]. The large association observed for a history of knee surgery requires cautious interpretation. In a cross-sectional analysis of prevalent disease, previous surgery is more plausibly a consequence or marker of established joint pathology than an antecedent exposure, and part of this association reflects reverse causation. The sensitivity analysis excluding this variable is therefore informative: discrimination fell from 0.815 to 0.797, and the relative ordering of the modelling approaches was unchanged, indicating that the principal findings do not depend on it.

Metabolic and body-composition variables contributed little. Although metabolic factors are associated with the incidence and progression of knee osteoarthritis [22–24], they did not improve discrimination beyond age and body mass index in this sample, and the bioimpedance-derived measures produced increments of no more than 0.008 in the area under the curve. Educational attainment and income were likewise not retained. These are statements about incremental discrimination in this population, conditional on the variables already included, and not about aetiological relevance.

The model was well calibrated, which is a prerequisite for interpreting its output as a probability rather than as a ranking. The operating characteristics reported in the Supplementary Material show a high negative predictive value across thresholds, indicating that low model-estimated probabilities correspond reliably to the absence of radiographic disease in this population, whereas the modest positive predictive value is an expected consequence of a prevalence of 14%. Whether such a model would be useful in practice is a separate question from the one addressed here, and would require both an established indication for imaging and validation in the population of intended use.

Several limitations qualify these findings. The analysis is cross-sectional: the models classify structural status at the time of imaging and provide no information about future disease. Validation was internal only; no external or temporal validation was performed, and the estimates reported here may not transport to populations differing in age structure, body mass index distribution or the composition of occupational and racial groups. ELSA-Brasil comprises civil servants and excludes the extremes of the socioeconomic spectrum, although the distribution of major chronic disease risk factors has been shown to be comparable to that of the general Brazilian adult population [27], and representativeness is not a prerequisite for valid aetiological inference [28]. Variables selected by stepwise procedures are known to be unstable across resamples; the variables retained across folds shared a consistent core of age, body mass index and history of surgery and trauma, with variation in the remaining positions. Because selection and estimation used the same data, the odds ratios and their confidence intervals do not account for the selection step and are reported as descriptive. Non-response to the clinical-history and symptom items was coded as absence of the event, which may misclassify a proportion of knees. Finally, the discordance between radiographic findings and symptoms means that knees classified as having a high probability of structural disease do not necessarily require symptomatic treatment.

---

# CONCLUSIONS

A model of seven routinely obtainable characteristics identified prevalent radiographic knee osteoarthritis with discrimination at least equal to that of tuned machine-learning algorithms, and with adequate calibration. Increasing algorithmic complexity conferred no advantage for this outcome in this population, and the tuning procedure converged on configurations approximating linear, additive structure. These findings support the use of transparent regression models for characterising structural knee osteoarthritis in population-based data. They concern the classification of disease already present; extension to clinical decision-making would require an established indication for imaging and validation in independent populations.

---

# ACKNOWLEDGEMENTS

[See `OAC_submission_sections.md` §A1.]

# AUTHOR CONTRIBUTIONS

[See `OAC_submission_sections.md` §A2.]

# ROLE OF THE FUNDING SOURCE

[See `OAC_submission_sections.md` §A3.]

# COMPETING INTEREST STATEMENT

[See `OAC_submission_sections.md` §A4.]

# DATA AND CODE AVAILABILITY

[See `OAC_submission_sections.md` §A5.]

# DECLARATION OF GENERATIVE AI AND AI-ASSISTED TECHNOLOGIES IN THE WRITING PROCESS

[See `OAC_submission_sections.md` §A6. Determine which of the two situations applies before submission.]

---

# REFERENCES

[Retain the existing 28 references, renumbered if the citation order changes. Convert to the Elsevier format specified in Guide for Authors §1.9.12: numbers in square brackets, journal names abbreviated per the List of Title Word Abbreviations.]

---

# TABLES

**Table 1.** Characteristics of the study participants, by prevalent radiographic knee osteoarthritis (rKOA).
Values are mean (SD) or n (%). Participants were classified as having rKOA if at least one knee met the criteria; for knee-specific variables, participants were classified as positive if at least one knee was affected. p-values from Welch's t-test for continuous variables and the chi-squared test for categorical variables.
[Insert `results/manuscript/table1_participants.csv`.]

**Table 2.** Associations of the selected variables with prevalent radiographic knee osteoarthritis.
Odds ratios from unpenalised logistic regression with standard errors clustered at the participant level (5,650 knees from 2,830 participants), reported per standard deviation for continuous variables and per category for binary variables. Estimates do not account for the preceding variable-selection step.
[Insert `results/manuscript/table2_final_model_or.csv`.]

**Table 3.** Discrimination of the four modelling approaches under nested cross-validation.
[Insert from `results/comparison/nested_cv_summary.csv` and `nested_cv_paired_diff.csv`; report the area under the curve with 95% CI for each approach and set, and the difference relative to the logistic model.]

# FIGURE LEGENDS

**Figure 1.** Incremental change in the area under the receiver operating characteristic curve as variables are added to the Constitutional model by forward stepwise selection. Values are the cross-validated estimates that served as the selection criterion, computed on the full sample; they are not the validated estimates of model performance, which were obtained by nested cross-validation and are reported in Table 3.
[File: `results/manuscript/figure1_constitutional_trajectory.png`]

**Figure 2.** Receiver operating characteristic curves for the four modelling approaches in the Constitutional variable set, computed from pooled held-out estimates under nested cross-validation.
[File: `results/comparison/fig_roc_nested.png`]

# SUPPLEMENTARY MATERIAL

- **Supplementary Table S1.** Distribution and missingness of all candidate variables. [`results/manuscript/table3_candidate_variables.csv`]
- **Supplementary Table S2.** Variables retained by the penalised fit. [`results/manuscript/table4_lasso_coefficients.csv`]
- **Supplementary Table S3.** Hyperparameter search spaces and selected configurations. [`supplementary/ml_tuning.md`]
- **Supplementary Table S4.** Operating characteristics at candidate probability thresholds. [`supplementary/clinical_utility.md`]
- **Supplementary Table S5.** Sensitivity analyses for isolated patellofemoral disease. [`supplementary/pf_ablation.md`]
- **Supplementary Table S6.** Coefficients of the final model. [`results/manuscript/final_model_coefficients.csv`]
- **Supplementary Figure S1.** Calibration plot. [`results/comparison/fig_calibration.png`]
- **Supplementary Figure S2.** Decision curve. [`results/comparison/fig_decision_curve.png`]
- **Supplementary Methods.** Operational definitions of the variables. [Retain existing text.]
