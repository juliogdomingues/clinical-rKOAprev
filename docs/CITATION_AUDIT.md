# Citation audit

Every claim in `manuscript_revised.md` carrying a numbered citation was checked
against the corresponding source in `references/`. Sixteen cited claims were
examined against 25 of the 28 references; three sources were not supplied
(reference 4, Jin et al.; reference 19, Felson et al.; reference 28, Rothman,
a textbook) and could not be checked.

Eleven claims are supported. Five require attention, one of them urgently.

---

## 1. Requires resolution before submission

### 1.1 The inter-observer agreement statistic has no identifiable source

**Claim.** "The process was calibrated against an external expert reader, with
substantial inter-observer agreement (κ = 0.755; 95% CI 0.663, 0.847) [13]."

**Finding.** The value does not appear in Telles et al. 2022. A search of the
full text of all 25 available references for "0.755", "0.663" and "0.847"
returned no match, and Telles 2022 contains no occurrence of "kappa",
"inter-observer" or "reproducibility".

What Telles 2022 does report is a different validation of the reading protocol:

> "The screening protocol was tested by having a random selection of 108 images
> rated by the two screeners as 'no possible knee OA' evaluated by the
> experienced radiologist, with no misclassification being found."

**Action.** Identify the source of κ = 0.755 and cite it, or replace the
sentence with the validation actually reported in reference 13. A specific
statistic attributed to a source that does not contain it is the kind of error
that reviewers check.

---

## 2. Claims not supported by the cited sources

### 2.1 Radiographic findings and subsequent decline

**Claim.** "Radiographic assessment ... provides a morphological reference that
correlates with subsequent joint degradation and functional decline [7,8]."

**Reference 7 (Yoshikawa et al. 2026).** The cited paper reports the opposite of
what the sentence implies. Its stated conclusion is that "3D-JSW on WBCT did not
outperform 2D-JSWx on radiography for predicting knee pain and functional
worsening", and the discrimination of radiographic joint space width for
24-month worsening was close to chance: AUC 0.511 for WOMAC pain, with
comparable values for the 20-metre walk, sit-to-stand and WOMAC function. The
paper is a null comparison between imaging modalities, not evidence that
radiographic findings track functional decline.

**Reference 8 (Miguel et al. 2019).** Described in its own abstract as "a
cross-sectional study of the diagnostic accuracy of different knee OA
classification criteria". A cross-sectional study cannot support a claim about
subsequent decline.

**Action.** Either cite sources that examine the longitudinal relationship
between radiographic status and functional outcome, or restrict the sentence to
what these two sources do support: that radiographic classification is a
standard criterion for defining structural disease, and that classification
criteria have been compared within this cohort.

### 2.2 Bioimpedance-derived body composition

**Claim.** "Interest has grown in applying machine-learning algorithms and novel
biomarkers, including bioimpedance-derived body composition, to the
identification and classification of osteoarthritis [9,10]."

**Finding.** Neither reference 9 (Ramazanian et al. 2023) nor reference 10
(Joseph et al. 2025) contains any occurrence of "bioimpedance" or "body
composition". Both support the statement about machine learning; neither
supports the clause about bioimpedance.

**Action.** Cite a source for the bioimpedance clause, or remove it and
introduce bioimpedance where the candidate variables are described, which is
where the study's own rationale for including it belongs.

---

## 3. Claims requiring more precise wording

### 3.1 Comparability of ELSA-Brasil to the Brazilian population

**Claim.** "the distribution of major chronic disease risk factors has been shown
to be comparable to that of the general Brazilian adult population [27]".

**Finding.** Schmidt et al. 2015 compares ELSA-Brasil with VIGITEL, described in
that paper as "producing representative data for adults living in Brazil's 27
state capitals and Federal District". The comparison is therefore with the urban
capital population, not with the general Brazilian adult population, which
includes rural and non-capital areas.

**Action.** Reword to reflect the actual comparator, for example "comparable to
that of adults living in Brazilian state capitals [27]". The point being made is
unaffected, and the narrower statement is defensible.

### 3.2 Occupational mechanical exposure

**Claim.** "of previous knee injury and occupational mechanical exposure
[14,15,20]".

**Finding.** References 14 (Zhang and Jordan) and 15 (Silverwood et al.) support
both elements. Reference 20 (Driban et al.) is a systematic review of sports
participation; "occupation" appears in it only in a background list of known
risk factors and in the title of a work it cites. It does not provide evidence
about occupational exposure.

**Action.** Remove reference 20 from this citation. It is correctly cited later
for high-impact sport.

### 3.3 Ageing and mechanical loading as principal determinants

**Claim.** "The dominance of age and body mass index accords with established
evidence identifying ageing and mechanical loading as the strongest determinants
of knee osteoarthritis prevalence [14–17]."

**Finding.** References 14, 15 and 16 support this. Reference 17 (Szilagyi et
al.) is a systematic review of sex differences in risk factors, concluding that
"more good quality studies are needed to assess sex differences in risk factors
for KOA". It does not establish ageing or mechanical loading as principal
determinants.

**Action.** Cite 14–16 here. Reference 17 is better placed where sex is
discussed.

---

## 4. Two further observations

**Terminology.** Telles 2022 refers to "two independent radiographers"; the
manuscript says "two trained radiology technologists". Aligning the wording with
the source avoids an apparent discrepancy.

**Outcome definition.** Telles 2022 defines radiographic knee osteoarthritis in
this cohort as tibiofemoral Kellgren–Lawrence grade ≥ 2 "and/or patellofemoral
OA (definitive osteophyte in the PF joint or definitive [joint space narrowing])".
The present analysis instead uses Kellgren–Lawrence grade ≥ 2 in the
patellofemoral compartment, which the revised readings make available. This is a
departure from the previously published definition for the cohort and should be
stated explicitly in the Outcome Definition, with the reason, so that readers
comparing the two are not misled.

---

## 5. Claims verified as supported

| Claim | Reference | Evidence |
|---|---|---|
| Osteoarthritis prevalence and disability burden | 1, 2 | Both address prevalence and disability directly |
| Economic and social consequences of symptoms | 3, 5 | Sharma covers symptoms; Li addresses socioeconomic burden. Reference 4 not supplied |
| Clinical identification without routine imaging | 6 | "a clinical diagnosis of osteoarthritis is sufficient to initiate treatment ... Laboratory testing and imaging are not needed for diagnosis" |
| Machine-learning models and interpretability | 11, 12 | Interpretability discussed in both |
| ELSA-Brasil MSK sample: 2,901, aged 38–79, civil servants | 13 | "ELSA-Brasil MSK comprises 2901 active/retired civil servants, both sexes, aged 38 to 79 years old at inception (2012-14)" |
| Two-step reading protocol, blinded | 13 | "interpreted, blind to participants' characteristics"; "two-step protocol: (i) preliminary screening ... (ii) diagnosis ... by an experienced radiologist" |
| Positioning device and repeatability of joint space width | 26 | Title: "produced highly repeatable measurements of joint space width" |
| Age, sex and body mass index as risk factors | 14–16 | All three address these |
| Physical activity and high-impact sport | 18–21 | Reference 19 not supplied; 18, 20, 21 support the claim |
| Metabolic syndrome components | 22–24 | All three address metabolic factors |
| Genetic contribution, family history as proxy | 25 | "the genetic component of OA to be between 40 and 80%" from family and twin studies |
