"""Step 4: cross-validated permutation importance for the final and MPMS models.

Writes ``permutation_importance_*.csv`` and ``fig_permutation_importance_*.png``
to ``results/final_analysis/``.
"""
from __future__ import annotations

import sys

from koa_screening import importance


def main() -> int:
    importance.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
