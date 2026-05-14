"""Generate the regression-test fixtures that lock current behaviour.

Run once from the repo root, with the real ELSA-Brasil CSV at
``base_stata/stataToCsvMG.csv``, *before* the refactor:

    python tests/fixtures/_make_fixtures.py

Outputs (overwrites):
    tests/fixtures/expected_columns_post_prep.txt
    tests/fixtures/expected_columns_scenario_without.txt
    tests/fixtures/expected_columns_scenario_with.txt
    tests/fixtures/expected_columns_scenario_virtual.txt
    tests/fixtures/expected_dropped_high_missing.txt
    tests/fixtures/expected_summary_all_models.csv
    tests/fixtures/expected_final_model_or.csv
    tests/fixtures/fixture_metadata.json

The point is to snapshot today's behaviour so any future edit that
inadvertently drops a variable, reorders dummies, or changes an AUC
breaks the regression suite loudly.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FIX = Path(__file__).resolve().parent
CSV = REPO / "base_stata" / "stataToCsvMG.csv"

sys.path.insert(0, str(REPO))

import oarsi_data  # noqa: E402

BASE_EXCLUDE = [
    "idelsa", "side", "kl", "oapf", "oa_knee",
    "kl_raw_num", "oapf_raw_num",
    "race_raw", "occupation", "smoking_status",
    "physical_activity_ipaq", "alcohol_use",
]
SYMPTOM_VARS = ["frequent_symptoms", "recent_pain_7d", "knee_disability"]


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not CSV.exists():
        print(f"ERROR: input CSV not found at {CSV}")
        return 1

    outdir = REPO / "tests" / "fixtures" / "_prep_audit"
    outdir.mkdir(parents=True, exist_ok=True)

    df = oarsi_data.load_and_prep_data(str(CSV), outdir=str(outdir))
    df = df.sort_values("idelsa").reset_index(drop=True)

    write_lines(FIX / "expected_columns_post_prep.txt", sorted(df.columns.tolist()))

    all_cols = [c for c in df.columns if c not in BASE_EXCLUDE]
    cols_without = [c for c in all_cols if c not in SYMPTOM_VARS]
    cols_with = list(all_cols)
    cols_virtual = list(all_cols)

    write_lines(FIX / "expected_columns_scenario_without.txt", sorted(cols_without))
    write_lines(FIX / "expected_columns_scenario_with.txt", sorted(cols_with))
    write_lines(FIX / "expected_columns_scenario_virtual.txt", sorted(cols_virtual))

    X_full = df[all_cols].copy()
    thresh = int(np.ceil(len(df) * 0.5))
    X_after = X_full.dropna(axis=1, thresh=thresh)
    dropped = sorted(set(X_full.columns) - set(X_after.columns))
    write_lines(FIX / "expected_dropped_high_missing.txt", dropped)

    summary_src = REPO / "results_comparison" / "summary_all_models.csv"
    if summary_src.exists():
        pd.read_csv(summary_src).to_csv(FIX / "expected_summary_all_models.csv", index=False)
    else:
        print(f"WARN: {summary_src} not found; AUC fixture skipped")

    or_src = REPO / "results_final_analysis" / "final_model_or_raw_ci.csv"
    if or_src.exists():
        pd.read_csv(or_src).to_csv(FIX / "expected_final_model_or.csv", index=False)
    else:
        print(f"WARN: {or_src} not found; OR fixture skipped")

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(CSV.relative_to(REPO)),
        "csv_sha_first_kb": _hash_first_kb(CSV),
        "n_rows_post_prep": int(len(df)),
        "n_cols_post_prep": int(df.shape[1]),
        "n_participants_post_prep": int(df["idelsa"].nunique()),
        "n_cols_scenario_without": len(cols_without),
        "n_cols_scenario_with": len(cols_with),
        "n_cols_scenario_virtual": len(cols_virtual),
        "n_dropped_high_missing": len(dropped),
        "prevalence_oa_knee": float(df["oa_knee"].mean()),
        "seed": 42,
    }
    (FIX / "fixture_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    for k, v in metadata.items():
        print(f"  {k}: {v}")
    print(f"\nFixtures written to {FIX}")
    return 0


def _hash_first_kb(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(1024))
    return h.hexdigest()[:16]


if __name__ == "__main__":
    sys.exit(main())
