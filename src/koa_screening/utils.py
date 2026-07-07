"""Backward-compatible shim.

The AUC/CI helpers now live in :mod:`koa_screening.evaluation` (single source
of truth). This module used to carry its own copies; they were removed to
avoid a second, separately-maintained implementation of the cluster bootstrap.
Import from ``koa_screening.evaluation`` in new code.
"""
from __future__ import annotations

from .evaluation import (  # noqa: F401
    auc_ci_bootstrap_by_group,
    auc_ci_from_folds,
)
