"""Regenerate the regression-test fixtures that lock current behaviour.

Run from the repo root, with the real ELSA-Brasil CSV at
``data/raw/stataToCsvMG.csv`` and after the core pipeline has produced its
results (``scripts/02_feature_selection.py`` and ``scripts/03_run_comparison.py``,
plus ``scripts/04_final_model_or.py`` for the OR fixture):

    python tests/fixtures/_make_fixtures.py

Overwrites ALL fixture files consumed by tests/ (14 total):
    - column inventories (post-prep + 3 scenarios + high-missing drop set)
    - the 3 selection fixtures (stepwise MPMS order + the two _for_ci lists)
    - AUC summary + 5 OR tables (final-model raw, and per-scenario raw/std)
    - fixture_metadata.json

Snapshotting today's behaviour is how the regression suite detects any future
edit that drops a variable, reorders dummies, or shifts an AUC/OR.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FIX = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from koa_screening import data  # noqa: E402
from koa_screening.config import (  # noqa: E402
    BASE_EXCLUDE,
    BIO_VARS,
    MISSING_COL_THRESHOLD,
    RAW_CSV,
    RESULTS_COMPARISON,
    RESULTS_FINAL,
    SYMPTOM_VARS,
    WOMAC_VARS,
)

# (source result file under results/, fixture filename) — copied verbatim.
COPY_FIXTURES = [
    (RESULTS_COMPARISON / "summary_all_models.csv", "expected_summary_all_models.csv"),
    (RESULTS_FINAL / "final_model_or_raw_ci.csv", "expected_final_model_or.csv"),
    (RESULTS_FINAL / "stepwise_mpms_clinical.csv", "expected_stepwise_mpms_clinical.csv"),
    (RESULTS_FINAL / "mpms_features_for_ci.csv", "expected_mpms_features_for_ci.csv"),
    (RESULTS_FINAL / "final_5var_features_for_ci.csv", "expected_final_5var_features_for_ci.csv"),
    (RESULTS_COMPARISON / "or_raw_Without_Symptoms.csv", "expected_or_raw_Without_Symptoms.csv"),
    (RESULTS_COMPARISON / "or_standardized_Without_Symptoms.csv", "expected_or_standardized_Without_Symptoms.csv"),
    (RESULTS_COMPARISON / "or_raw_With_Symptoms.csv", "expected_or_raw_With_Symptoms.csv"),
    (RESULTS_COMPARISON / "or_standardized_With_Symptoms.csv", "expected_or_standardized_With_Symptoms.csv"),
]


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hash_first_kb(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(1024))
    return h.hexdigest()[:16]


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: input CSV not found at {RAW_CSV}", file=sys.stderr)
        return 1

    outdir = FIX / "_prep_audit"
    outdir.mkdir(parents=True, exist_ok=True)

    df = data.load_and_prep_data(str(RAW_CSV), outdir=str(outdir))
    df = df.sort_values("idelsa").reset_index(drop=True)

    # --- column inventories (mirror runner.run_comparison: WOMAC excluded
    # everywhere; bioimpedance only in Virtual Maximum) ---
    write_lines(FIX / "expected_columns_post_prep.txt", sorted(df.columns.tolist()))
    all_cols = [c for c in df.columns if c not in BASE_EXCLUDE and c not in WOMAC_VARS]
    base_pool = [c for c in all_cols if c not in BIO_VARS]
    cols_without = [c for c in base_pool if c not in SYMPTOM_VARS]
    write_lines(FIX / "expected_columns_scenario_without.txt", sorted(cols_without))
    write_lines(FIX / "expected_columns_scenario_with.txt", sorted(base_pool))
    write_lines(FIX / "expected_columns_scenario_virtual.txt", sorted(all_cols))

    X_full = df[all_cols].copy()
    thresh = int(np.ceil(len(df) * MISSING_COL_THRESHOLD))
    dropped = sorted(set(X_full.columns) - set(X_full.dropna(axis=1, thresh=thresh).columns))
    write_lines(FIX / "expected_dropped_high_missing.txt", dropped)

    # --- copy the result-derived fixtures verbatim (fail loudly if a source is missing) ---
    missing = []
    for src, dest in COPY_FIXTURES:
        if src.exists():
            shutil.copyfile(src, FIX / dest)
        else:
            missing.append(str(src.relative_to(REPO)))
    if missing:
        print("ERROR: these result files are missing -- run the pipeline first "
              "(scripts 02, 03, 04):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 2

    # --- metadata ---
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(RAW_CSV.relative_to(REPO)) if RAW_CSV.is_relative_to(REPO) else str(RAW_CSV),
        "csv_sha_first_kb": _hash_first_kb(RAW_CSV),
        "n_rows_post_prep": int(len(df)),
        "n_cols_post_prep": int(df.shape[1]),
        "n_participants_post_prep": int(df["idelsa"].nunique()),
        "n_cols_scenario_without": len(cols_without),
        "n_cols_scenario_with": len(base_pool),
        "n_cols_scenario_virtual": len(all_cols),
        "n_dropped_high_missing": len(dropped),
        "prevalence_oa_knee": float(df["oa_knee"].mean()),
        "seed": 42,
    }
    (FIX / "fixture_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    shutil.rmtree(outdir, ignore_errors=True)
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    print(f"\nAll {len(COPY_FIXTURES) + 6} fixtures written to {FIX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
