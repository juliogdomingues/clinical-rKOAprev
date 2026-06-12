"""Step 2: variable selection (LASSO -> MPMS -> stepwise).

Runs ``koa_screening.features.run_analysis`` to produce the intermediate
files the rest of the pipeline depends on:

  - stepwise_mpms_clinical.csv      (MPMS stepwise order; read by step 3, 8, 9)
  - mpms_features_for_ci.csv        (MPMS clinical feature set; read by step 5)
  - final_5var_features_for_ci.csv  (top-5 final model; read by steps 4 & 5)
  - lasso_coefficients_{full,clinical}.csv  (read by step 7 figures)
  - mpms_k_performance.csv, final_5var_model.csv, fig_stepwise_mpms.png, ...

The first three are byte-for-byte deterministic at seed 42 and are locked
by tests/test_regression_selection.py. This step MUST run before steps
3-9; previously these files were only available because they were committed,
which broke a from-scratch reproduction.
"""
from __future__ import annotations

import sys

from koa_screening import data, features
from koa_screening.config import RAW_CSV, RESULTS_FINAL


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: raw CSV not found at {RAW_CSV}", file=sys.stderr)
        print("See data/README.md for how to obtain it.", file=sys.stderr)
        return 1
    RESULTS_FINAL.mkdir(parents=True, exist_ok=True)
    df = data.load_and_prep_data(str(RAW_CSV), outdir=str(RESULTS_FINAL))
    features.run_analysis(df, outdir=str(RESULTS_FINAL))
    print(f"\nSelection intermediates written to: {RESULTS_FINAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
