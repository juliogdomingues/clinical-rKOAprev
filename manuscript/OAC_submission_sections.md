# Osteoarthritis and Cartilage — end-matter sections and reporting-checklist mapping

Prepared for submission of "Comparative Performance of Linear and Non-Linear
Algorithms for Identifying Prevalent Radiographic Knee Osteoarthritis".
Text in `[SQUARE BRACKETS]` requires author input and must not be submitted as
written. Sections A1–A6 are placed after the Discussion and before the
References, in the order specified in Guide for Authors §1.2.

---

## PART A — Sections to add to the manuscript

### A1. Acknowledgements

> The authors thank the participants and the field staff of the ELSA-Brasil
> Musculoskeletal Study for their contribution to data collection.
> [ADD: individuals who contributed but do not meet authorship criteria, with
> their role (for example, technical assistance, radiographic reading, or
> statistical advice). Written permission must be obtained from each person
> named. If writing assistance was received, it must be disclosed here together
> with the responsible entity and the funder of that assistance.]

### A2. Author contributions

Guide for Authors §1.9.8 requires a declaration against the four ICMJE criteria
and the nomination of at least one author, with an email address, who takes
responsibility for the integrity of the work as a whole.

> Conception and design: [initials]. Acquisition of data: [initials].
> Analysis and interpretation of the data: [initials]. Statistical expertise:
> [initials]. Drafting of the article: [initials]. Critical revision of the
> article for important intellectual content: all authors. Obtaining of funding:
> [initials]. Administrative, technical, or logistic support: [initials].
> Final approval of the article: all authors.
>
> All authors have read and approved the final version of the manuscript.
> J.G. Domingues (juliogdomingues@gmail.com) takes responsibility for the
> integrity of the work as a whole, from inception to finished article.

[CONFIRM the allocation above with each co-author before submission. The
existing contributions statement in the manuscript should be replaced by this
expanded version, which is the format the journal requires.]

### A3. Role of the funding source

> [ADD funding sources for ELSA-Brasil, ELSA-Brasil MSK, and any personal
> support (scholarships or fellowships) received by the authors. The exact
> wording used in the cohort profile update (reference 13, Telles et al., Int J
> Epidemiol 2022) should be reproduced to ensure consistency with previous
> publications from the cohort.]
>
> The study sponsors had no role in the study design; in the collection,
> analysis and interpretation of data; in the writing of the manuscript; or in
> the decision to submit the manuscript for publication.

[If any sponsor did have such a role, that role must be described instead of the
sentence above.]

### A4. Competing interest statement

> [ADD, for every author: financial and personal relationships with other people
> or organisations that could inappropriately influence the work. Categories to
> consider are employment, consultancies, stock ownership, honoraria, paid expert
> testimony, patent applications or registrations, and research grants or other
> funding.]
>
> The authors declare that they have no competing interests.

[The final sentence is appropriate only if it is accurate for all four authors.
Each co-author must additionally complete the ICMJE disclosure form, which is
uploaded separately at submission.]

### A5. Data and code availability

The journal requests a statement on data availability (§1.8.3) and, for
prediction-model studies, states that the model should be sufficiently open to
permit external validation (§1.1.4).

> Individual-level data from the ELSA-Brasil Musculoskeletal Study are not
> publicly available because they contain potentially identifying participant
> information and are governed by the ELSA-Brasil data access policy. Access may
> be requested through the ELSA-Brasil Steering Committee
> (https://www.elsa.org.br/). The complete analysis code, including the data
> preparation, variable selection, model fitting, cross-validation, calibration
> and decision-curve routines, together with the fitted coefficients of the final
> model and the aggregate result files required to reproduce every table and
> figure, is openly available at [REPOSITORY URL; deposit a tagged release and
> cite the archived DOI]. The repository does not contain participant-level data.

[The link is currently withheld in the Methods. Confirm whether OAC operates
double-blind review before retaining that sentence; the Guide for Authors does
not state that it does. If review is single-blind, cite the repository normally
in this statement and delete the redaction sentence from the Methods.]

### A6. Declaration of generative AI and AI-assisted technologies in the writing process

The journal's policy (§1.9.11) applies to the **writing process only** and
explicitly excludes the use of AI tools to analyse data or draw insights as part
of the research process. Two situations are therefore distinguished.

**(i) If generative AI was used to draft or edit manuscript prose**, the
following statement is required, placed immediately before the References:

> During the preparation of this work the author(s) used [NAME OF TOOL AND
> VERSION] in order to [SPECIFIC PURPOSE, for example: draft and revise sections
> of the Methods for clarity]. After using this tool, the author(s) reviewed and
> edited the content as needed and take(s) full responsibility for the content of
> the publication.

**(ii) If generative AI was used only for software development and data
analysis**, no declaration is required under §1.9.11, and the policy states this
explicitly. Basic grammar and spelling checkers are also exempt. Computational
tooling of this kind is nevertheless best described in the Methods or in the
repository documentation, where the analysis pipeline is already recorded.

[Determine which situation applies and include the corresponding text. AI tools
must not be listed as authors.]

---

## PART B — STROBE mapping (cross-sectional studies)

A completed STROBE checklist must be submitted with the manuscript (§1.1.2).
The table records where each item is addressed and what remains outstanding.
Page and line numbers must be inserted after the manuscript is paginated.

| # | STROBE item | Location / action required |
|---|---|---|
| 1a | Study design in title or abstract | **Action:** the design is not named. Add "cross-sectional analysis" to the title or to the first line of the abstract. |
| 1b | Informative, balanced abstract | Abstract, structured as Objective / Design / Results / Conclusions. **Action:** reduce to the OAC limit and remove the equivalence claim. |
| 2 | Background and rationale | Introduction, paragraphs 1–4. |
| 3 | Objectives | Introduction, final paragraph. |
| 4 | Study design | Methods, "Study Design and Sample". |
| 5 | Setting, locations, dates | Methods, "Study Design and Sample" (ELSA-Brasil MSK, 2012–2014). |
| 6a | Eligibility criteria, sources, selection | Methods, "Study Design and Sample". **Action:** state exclusion criteria explicitly (arthroplasty; knees non-gradeable in both compartments). |
| 7 | Variables: outcome, predictors, confounders | Methods, "Outcome Definition" and "Candidate Predictors"; Supplementary Methods. |
| 8 | Data sources and measurement | Methods, "Radiographic Assessment" and Supplementary Methods (questionnaire wording). |
| 9 | Bias | **Action: absent.** Add a short paragraph covering blinded outcome assessment, the participant-level clustering of bilateral knees, the coding of non-response on history items as absence of the event, and the possibility of reverse causation for history of knee surgery. |
| 10 | Study size | **Action: absent.** State that the sample comprised all eligible participants of the ELSA-Brasil MSK baseline, that no a priori sample-size calculation was performed, and report the number of events per candidate variable. |
| 11 | Quantitative variables | Methods, "Statistical Analysis". **Action:** state that continuous predictors were analysed on their original scale and standardised for the comparison of effect sizes. |
| 12a | Statistical methods, confounding | Methods, "Statistical Analysis". |
| 12b | Subgroups and interactions | Not applicable; no subgroup analyses were performed. State this. |
| 12c | Missing data | **Action:** the current description is incomplete. Report the proportion of missing values, the within-fold median imputation, the exclusion of WOMAC, and the coding of non-response on the binary history and symptom items. |
| 12d | Sampling strategy | Not applicable (no complex survey design). |
| 12e | Sensitivity analyses | **Action:** cite the isolated-patellofemoral analyses, the analysis excluding history of knee surgery, and the seed-stability analysis, all reported in the Supplementary Material. |
| 13a–c | Participant flow | Methods and Results. **Action:** add a flow diagram or an explicit sequence of exclusions from eligible participants to the 5,650 analysed knees. |
| 14a | Descriptive data | Table 1; Supplementary Table S1. |
| 14b | Missing data per variable | **Action:** report in Supplementary Table S1. |
| 15 | Outcome data | Results, first paragraph. **Action:** report the number of events at both the knee and participant level, using unambiguous denominators. |
| 16a | Estimates with confidence intervals | Table 2 (odds ratios with 95% confidence intervals). |
| 16b | Category boundaries for continuous variables | Applicable to the reporting thresholds in the clinical-utility analysis; specify them. |
| 17 | Other analyses | Supplementary Material (sensitivity and stability analyses). |
| 18 | Key results | Discussion, first paragraph. |
| 19 | Limitations | Discussion, final paragraph. **Action:** add internal validation only, post-selection inference, instability of stepwise selection, and the missing-data coding. |
| 20 | Interpretation | Discussion. **Action:** address multiplicity across the model comparisons. |
| 21 | Generalisability | Discussion (civil-servant sample). |
| 22 | Funding | Section A3 above. |

---

## PART C — TRIPOD mapping (prediction-model reporting)

The journal accepts TRIPOD for prediction-model studies (§1.1.4). Items are
those of the TRIPOD statement for model development with internal validation;
items specific to external validation are not applicable.

| # | TRIPOD item | Location / action required |
|---|---|---|
| 1 | Title identifies the study as developing a prediction model, with target population and outcome | **Action:** the current title describes an algorithm comparison. Consider naming the target population and the outcome, and the fact that a model is developed. |
| 2 | Structured abstract | Abstract. **Action:** state that internal validation only was performed. |
| 3a | Rationale, reference to existing models | Introduction. **Action:** cite existing KOA prediction models (references 9–12 are available). |
| 3b | Objectives | Introduction, final paragraph. |
| 4a | Source of data and design | Methods. |
| 4b | Key study dates | Methods (2012–2014). |
| 5a–b | Setting and eligibility | Methods. |
| 6a | Outcome definition and assessment | Methods, "Outcome Definition". **Action:** update to Kellgren–Lawrence grade ≥ 2 in the tibiofemoral or the patellofemoral compartment, per the revised readings. |
| 6b | Blinding of outcome assessment | Methods, "Radiographic Assessment" (readers blinded to participant characteristics). Retain this sentence; it satisfies the item directly. |
| 7a | Predictor definitions and timing | Methods and Supplementary Methods. |
| 7b | Blinding of predictor assessment | **Action: absent.** State that predictors were obtained by interview and examination independently of, and blinded to, the radiographic classification. |
| 8 | Sample size | **Action: absent.** See STROBE item 10; report events per candidate predictor. |
| 9 | Missing data | **Action:** as STROBE item 12c. |
| 10a | Handling of predictors in the analysis | Methods, "Statistical Analysis". |
| 10b | Model type, model-building procedure, internal validation | Methods. **Action:** describe the nested cross-validation explicitly: an outer five-fold participant-grouped split, with variable selection for the logistic model and hyperparameter search for the machine-learning models performed within each training fold only. |
| 10d | Performance measures | Methods. **Action:** add calibration (slope and calibration-in-the-large) and decision-curve analysis. |
| 11 | Risk groups | Supplementary clinical-utility table (threshold-based classification). |
| 13a | Participant flow | As STROBE 13a. |
| 13b | Participant characteristics including predictors and outcome | Table 1; Supplementary Table S1. |
| 14a | Number of participants and events | Results. |
| 14b | Unadjusted associations of candidate predictors | Supplementary Table S1 provides distributions. **Action:** consider adding unadjusted odds ratios. |
| **15a** | **Full model: all regression coefficients and the intercept** | **Action: absent from the manuscript and required.** The fitted intercept, coefficients, imputation medians and standardisation parameters are already produced by the analysis pipeline (`final_5var_model.csv`). Present them as a supplementary table so that the model can be applied and externally validated. |
| 15b | Explanation of how to use the model | **Action: absent.** Add one sentence giving the linear predictor and the transformation to a probability. |
| 16 | Model performance with confidence intervals | Results; Supplementary clinical-utility tables. |
| 18 | Limitations | Discussion. |
| 19b | Overall interpretation | Discussion. |
| 20 | Implications for practice and research | Discussion and Conclusions. **Action:** temper claims regarding deployment, and state that external validation is required first. |
| 21 | Supplementary information: protocol, code, data | Section A5 above. |
| 22 | Funding | Section A3 above. |

Item 15a is the single most consequential omission. Without the model
coefficients the study cannot be externally validated, which is the specific
capability the journal identifies as imperative for prediction-model
submissions. The required content is already available in the analysis output
(`results/final_analysis/final_5var_model.csv`) and is reproduced below in the
form of a supplementary table.

**Supplementary Table [n]. Coefficients of the final logistic model.**

Predictors are standardised before entering the linear predictor. For a knee
with covariate vector *x*, the predicted probability of prevalent radiographic
knee osteoarthritis is obtained as

    z_j    = (x_j − mean_j) / SD_j
    logit  = −2.3288 + Σ_j β_j · z_j
    p      = 1 / (1 + exp(−logit))

| Predictor | Mean | SD | β (per SD) |
|---|---|---|---|
| Intercept | — | — | −2.3288 |
| Age, years | 56.050 | 8.937 | 0.8062 |
| Body mass index, kg/m² | 26.966 | 4.719 | 0.6135 |
| History of knee surgery (yes = 1) | 0.0356 | 0.1852 | 0.3761 |
| Frequent knee symptoms (yes = 1) | 0.1665 | 0.3726 | 0.3582 |
| History of knee trauma (yes = 1) | 0.1473 | 0.3544 | 0.2607 |

[VERIFY before submission: these values correspond to the five-variable model as
currently fitted. If the model presented in the manuscript is the seven-variable
Screening model, regenerate this table for that specification. Median values used
for imputation of missing predictors are given in the same output file and should
be reported alongside the table.]

---

## PART D — Submission requirements

**Files.** Three separate uploads are required: a file containing the title page
and abstract; a file containing the main text and references; and each figure as
a separate image file. Figures must not be embedded in the main text file.

**Title page.** Affiliations, ORCID iD and email address for every co-author,
full contact details for the corresponding author, and a running title of no
more than 40 characters including spaces. The present running head, "Linear vs.
Complex Models for Radiographic KOA", is 46 characters and must be shortened;
"Linear vs. complex models for knee OA" (37 characters) is one option.

**Manuscript preparation.** Double spacing throughout, including abstract and
references; sequential page numbers and line numbers on every page, including
the title page; metric and SI units; abbreviations defined at first use and then
used consistently.

**Abstract.** Maximum 350 words, structured as Objective, Design, Results,
Conclusions. Followed by three to six keywords.

**Limits.** 4,000 words excluding title page, abstract, tables, figure legends,
acknowledgements, contributions and references; a combined maximum of eight
tables and figures; 50 references. The current manuscript contains
approximately 3,233 words of body text, four tables and three figures, and 28
references, and is within all limits.

**References.** Numbered in square brackets in order of first citation, journal
names abbreviated according to the List of Title Word Abbreviations, in the
Elsevier format given in §1.9.12.

**Accompanying documents.** Cover letter; completed STROBE checklist; ICMJE
disclosure form for each author; author disclosure form. A submission fee of USD
50 applies.

**Statistical reporting.** The journal's Ten Recommendations (§1.1) impose three
requirements that the current text does not meet:

1. The terms "significant" and "significance" must not be used to classify
   results (recommendation 8).
2. Statistical non-significance must not be presented as evidence of equivalence
   (recommendation 9a). The paired differences in the area under the curve, with
   their confidence intervals, should be reported in place of any statement that
   the models performed equivalently.
3. P values must be reported numerically without categorisation; where the
   computed value is smaller than 0.0001 it is written as p < 0.0001
   (§1.9.5). Tables 1 and 2 currently report "< 0.001".

Recommendations 6 and 9c additionally require a statement on the examination of
model assumptions and a stated strategy for multiplicity across the model
comparisons.
