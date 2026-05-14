"""Step 1: prepare the analysis dataset from the raw ELSA CSV.

Calls :func:`koa_screening.data.load_and_prep_data` and writes audit reports
(``data_prep_summary.csv``, ``data_missingness_after_prep.csv``,
``data_value_exclusions_audit.csv``, ``row_drop_reasons.csv``, etc.) to
``results/final_analysis/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from koa_screening import data
from koa_screening.config import RAW_CSV, RESULTS_FINAL


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: raw CSV not found at {RAW_CSV}", file=sys.stderr)
        print("Place the ELSA-Brasil MSK CSV at that path. See data/README.md.", file=sys.stderr)
        return 1
    RESULTS_FINAL.mkdir(parents=True, exist_ok=True)
    df = data.load_and_prep_data(str(RAW_CSV), outdir=str(RESULTS_FINAL))
    print(
        f"\nPrepared dataset: {len(df)} knee-rows, "
        f"{df['idelsa'].nunique()} participants, {df.shape[1]} columns."
    )
    print(f"Audit reports in: {RESULTS_FINAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
