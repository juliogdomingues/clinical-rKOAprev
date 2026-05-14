"""Step 3: cluster-bootstrap raw Odds Ratios for the final 5-variable model.

Reproduces ``results/final_analysis/final_model_or_raw_ci.csv``. Requires that
step 2 has already produced ``final_5var_features_for_ci.csv``.
"""
from __future__ import annotations

import sys

from koa_screening import odds_ratios


def main() -> int:
    odds_ratios.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
