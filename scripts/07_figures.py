"""Step 7: composite manuscript figures.

Wraps :mod:`koa_screening.plots`, which generates the abstract/summary
figures (ROC overlays, 6-model panel, MPMS composite). Reads the LASSO
coefficient files written by the feature-selection step (02).
"""
from __future__ import annotations

import sys

from koa_screening import plots


def main() -> int:
    plots.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
