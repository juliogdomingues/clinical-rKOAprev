"""Step 3: 4 models x 3 scenarios — the manuscript's main comparison.

Produces ``results/comparison/summary_all_models.csv`` plus per-scenario ROC
curves, OR tables, importance plots, and the stepwise trajectory plot.
Outputs match the manuscript exactly at seed 42.

Requires the feature-selection step (02) to have produced
``results/final_analysis/stepwise_mpms_clinical.csv``; without it the
Stepwise model is silently skipped.
"""
from __future__ import annotations

import sys

from koa_screening.runner import run_comparison


def main() -> int:
    summary = run_comparison()
    print("\nSummary:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
