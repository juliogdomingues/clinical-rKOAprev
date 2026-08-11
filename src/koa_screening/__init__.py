"""Knee OA radiographic screening: clinical-epidemiological screening tool.

This package backs the ELSA-Brasil MSK prevalent radiographic KOA paper.
Public entry points:

    from koa_screening.data import load_and_prep_data      # raw CSV -> analysis dataset
    from koa_screening.features import run_analysis         # LASSO + forward stepwise
    from koa_screening.nested import nested_lr, nested_ml   # headline leak-free CV
    from koa_screening.clinical_utility import decision_curve, calibration_with_ci
    from koa_screening.config import RND, BASE_EXCLUDE, SYMPTOM_VARS, WOMAC_VARS, BIO_VARS

Scripts under ``scripts/`` are the canonical runners; this package is the
library code they share.
"""
from __future__ import annotations

__version__ = "0.1.0"
