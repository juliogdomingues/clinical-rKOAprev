"""Pytest configuration: skip ``@requires_data`` tests when the raw CSV is absent.

Synthetic-data smoke tests run anywhere; the regression tests against the
frozen baseline (`tests/fixtures/expected_*`) need the access-controlled
ELSA CSV to be sitting at ``data/raw/stataToCsvMG.csv``.
"""
from __future__ import annotations

import pytest

from koa_screening.config import RAW_CSV


def pytest_collection_modifyitems(config, items):
    if RAW_CSV.exists():
        return
    skip_reason = pytest.mark.skip(
        reason=f"raw CSV not found at {RAW_CSV} — see data/README.md"
    )
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip_reason)
