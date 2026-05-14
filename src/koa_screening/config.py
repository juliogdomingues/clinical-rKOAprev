"""Centralised configuration for the KOA screening analysis.

Everything that used to be scattered across module-level constants
(`RND=42`, `BASE_EXCLUDE`, `BIO_VARS`, `SYMPTOM_VARS`, `LABEL_MAP`,
hard-coded output paths) lives here so the rest of the code reads
from a single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root — resolved at import time. Scripts can override via env var.
REPO_ROOT = Path(os.environ.get("KOA_REPO_ROOT", Path(__file__).resolve().parents[2]))

# Random seed. Locked at 42 to match the published manuscript. Override with
# KOA_SEED for sensitivity analyses (those should write to their own results
# subdirectory rather than overwriting the canonical run).
RND = int(os.environ.get("KOA_SEED", "42"))

# Paths
DATA_DIR = REPO_ROOT / "data"
RAW_CSV = Path(os.environ.get("KOA_RAW_CSV", DATA_DIR / "raw" / "stataToCsvMG.csv"))
CODEBOOK_CSV = DATA_DIR / "codebook" / "variable_codebook.csv"

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_COMPARISON = RESULTS_DIR / "comparison"
RESULTS_FINAL = RESULTS_DIR / "final_analysis"

# ---------------------------------------------------------------------------
# Variable exclusion lists — exactly the lists that produced the manuscript.
# Locked in tests/fixtures/expected_columns_*.txt; do not edit without
# regenerating the fixtures.
# ---------------------------------------------------------------------------

# IDs, outcome columns, and raw categoricals that have already been expanded
# into dummies. These are NEVER inputs to a model.
BASE_EXCLUDE: list[str] = [
    "idelsa", "side", "kl", "oapf", "oa_knee",
    "kl_raw_num", "oapf_raw_num",
    "race_raw", "occupation", "smoking_status",
    "physical_activity_ipaq", "alcohol_use",
]

# WOMAC subscales are excluded by default (they leak symptom information into
# the structural-disease outcome). Set ``use_womac=True`` in scenarios to
# include them as a sensitivity analysis.
WOMAC_VARS: list[str] = [
    "womac_total", "womac_pain", "womac_stiffness", "womac_function",
]

# Bioimpedance + advanced anthropometry — the variables that the manuscript's
# "Virtual Maximum" scenario tests for incremental value.
BIO_VARS: list[str] = [
    "bone_mineral_content_kg", "mineral_mass_kg", "skeletal_muscle_mass_kg",
]

# Three patient-reported symptom variables. Their presence is what
# distinguishes "Case Finding (With Symptoms)" from "Screening (Without
# Symptoms)" in the manuscript.
SYMPTOM_VARS: list[str] = [
    "frequent_symptoms", "recent_pain_7d", "knee_disability",
]

# Column-wise missingness threshold for feature filtering. Columns with more
# than (1 - threshold) missing rows are dropped from the feature matrix.
# With the locked seed and current data this drops zero columns; the test
# suite asserts that.
MISSING_COL_THRESHOLD: float = 0.5

# Display labels for plots/tables (technical column name -> reader-friendly).
LABEL_MAP: dict[str, str] = {
    "occupation_4": "Occupation (Category 4)",
    "race_raw_3": "Race (Category 3)",
    "family_history_knee_replacement": "Family History of Knee Replacement",
    "frequent_symptoms": "Frequent Knee Symptoms",
    "history_surgery": "History of Knee Surgery",
    "history_trauma": "History of Knee Trauma",
    "knee_disability": "Knee Disability",
    "recent_pain_7d": "Recent Knee Pain (7d)",
    "bmi": "Body Mass Index (BMI)",
    "age": "Age (years)",
    "sit_stand_test": "Sit-to-Stand Test (s)",
    "abdominal_obesity": "Abdominal Obesity",
    "waist_hip_ratio": "Waist-Hip Ratio",
    "mineral_mass_kg": "Mineral Mass (kg)",
    "bone_mineral_content_kg": "Bone Mineral Content (kg)",
    "skeletal_muscle_mass_kg": "Skeletal Muscle Mass (kg)",
    "occ_stairs": "Occupational: Stairs",
    "occ_kneeling": "Occupational: Kneeling",
    "occ_squatting": "Occupational: Squatting",
}

# Scenarios. Each entry is (scenario_id, human_label, drop_set_relative_to_all_features).
# Add a new analysis here rather than copy-pasting a runner script.
SCENARIOS: list[tuple[str, str, list[str]]] = [
    ("without_symptoms", "Without Symptoms", SYMPTOM_VARS),
    ("with_symptoms", "With Symptoms", []),
    ("virtual_maximum", "Virtual Maximum", []),
]

MODELS: list[str] = [
    "Stepwise Logistic Regression",
    "XGBoost",
    "Random Forest",
    "Neural Network",
]
