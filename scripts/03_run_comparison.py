"""Step 3: 4 models x 3 scenarios — single-CV comparison + OR tables/figures.

NOTE: this single (non-nested) GroupKFold comparison is a **secondary /
diagnostic** view. The headline, leak-free model comparison is the nested CV in
`scripts/12_nested_cv.py` (`nested_cv_summary.csv`) — its LR AUC is the number
to report; the `summary_all_models.csv` AUCs here are mildly optimistic (feature
selection ran on the full sample) and are kept only as a cross-check.

This step is still required in the pipeline because it also produces the
**Odds-Ratio tables** (`or_raw_*`, `or_standardized_*` -> manuscript Table 2)
and the per-scenario ROC / importance / stepwise-trajectory figures, none of
which depend on the CV scheme.

Requires the feature-selection step (02) to have produced
``results/final_analysis/stepwise_mpms_clinical.csv``.
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
