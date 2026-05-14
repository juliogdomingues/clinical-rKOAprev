"""Knee OA radiographic screening: clinical-epidemiological screening tool.

This package backs the ELSA-Brasil MSK prevalent radiographic KOA paper.
Public entry points:

    from koa_screening.data import load_and_prep_data
    from koa_screening.preprocess import prepare_dataset
    from koa_screening.config import RND, SCENARIOS, BASE_EXCLUDE, SYMPTOM_VARS

Scripts under ``scripts/`` are the canonical runners; this package is the
library code they share.
"""
from __future__ import annotations

__version__ = "0.1.0"
