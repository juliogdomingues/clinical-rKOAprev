"""Smoke tests on a tiny synthetic CSV.

These run on any machine — they do not need the real ELSA-Brasil CSV. They
prove the pipeline is wired up correctly (data prep -> feature matrices ->
model fit -> AUC) and catch import / shape / NaN-handling regressions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic CSV that mimics the columns ``load_and_prep_data`` reads from
# the real STATA export. Two waves of variables (a_ / b_ / c_), with two
# rows per participant (one per knee side). Outcomes vary so we get both
# classes in every fold.
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)
N_PARTICIPANTS = 80


def _synthetic_df() -> pd.DataFrame:
    ids = [f"id{i:03d}" for i in range(N_PARTICIPANTS)]
    rng = RNG

    base = pd.DataFrame(
        {
            "idelsa": ids,
            "idadea": rng.integers(40, 75, N_PARTICIPANTS),
            "idadeb": rng.integers(40, 80, N_PARTICIPANTS),
            "rcta8": rng.integers(1, 3, N_PARTICIPANTS),
            "vifa29": rng.integers(0, 5, N_PARTICIPANTS),
            "b_imc1": rng.normal(28, 5, N_PARTICIPANTS).round(1),
            "b_imc2": rng.integers(0, 2, N_PARTICIPANTS),
            "b_imc3": rng.integers(0, 2, N_PARTICIPANTS),
            "b_rcq": rng.normal(0.9, 0.1, N_PARTICIPANTS).round(2),
            "b_obesidadeabdominal": rng.integers(0, 2, N_PARTICIPANTS),
            "biob11": rng.normal(3.0, 0.3, N_PARTICIPANTS).round(2),
            "biob130": rng.normal(2.5, 0.3, N_PARTICIPANTS).round(2),
            "biob27": rng.normal(25.0, 4.0, N_PARTICIPANTS).round(1),
            "TAL_MEAN2TRIALSSG_LB": rng.normal(12.0, 2.0, N_PARTICIPANTS).round(1),
            "DME49_LB": rng.integers(0, 2, N_PARTICIPANTS),
            "b_fumante": rng.integers(0, 3, N_PARTICIPANTS),
            "b_ativfisica": rng.integers(1, 4, N_PARTICIPANTS),
            "b_binge": rng.integers(0, 2, N_PARTICIPANTS),
            "b_bebexcessivo": rng.integers(0, 2, N_PARTICIPANTS),
            "b_usodealcool": rng.integers(0, 3, N_PARTICIPANTS),
            "FDR2a_LB": rng.integers(0, 4, N_PARTICIPANTS),
            "FDR3a_LB": rng.integers(0, 4, N_PARTICIPANTS),
            "FDR4_LB": rng.integers(0, 2, N_PARTICIPANTS),
            "FDR5_LB": rng.integers(0, 2, N_PARTICIPANTS),
            "FDR6_LB": rng.integers(0, 2, N_PARTICIPANTS),
            "FDR7_LB": rng.integers(0, 2, N_PARTICIPANTS),
            "DME47a_LB": rng.integers(0, 4, N_PARTICIPANTS),
            "DME51_2_LB": rng.integers(0, 4, N_PARTICIPANTS),
            "b_has2_2": rng.integers(0, 2, N_PARTICIPANTS),
            "b_dm_3": rng.integers(0, 2, N_PARTICIPANTS),
            "b_smj_as": rng.integers(0, 2, N_PARTICIPANTS),
            "b_smj_an": rng.integers(0, 2, N_PARTICIPANTS),
            "b_smj_eu": rng.integers(0, 2, N_PARTICIPANTS),
            "b_hipertrig": rng.integers(0, 2, N_PARTICIPANTS),
            "b_hipertrigmed": rng.integers(0, 2, N_PARTICIPANTS),
            "b_baixohdl": rng.integers(0, 2, N_PARTICIPANTS),
            "b_baixohdlmed": rng.integers(0, 2, N_PARTICIPANTS),
            "b_framingham_chd_chol_2": rng.normal(8, 3, N_PARTICIPANTS).round(1),
            "b_framingham_chd_ldl_2": rng.normal(8, 3, N_PARTICIPANTS).round(1),
            "b_framingham_cvd_model1_2": rng.normal(10, 4, N_PARTICIPANTS).round(1),
            "b_framingham_cvd_model2_2": rng.normal(10, 4, N_PARTICIPANTS).round(1),
            "a_nat_todos": rng.integers(0, 5, N_PARTICIPANTS),
            # Revised readings (current outcome source): KL for tibiofemoral
            # (b_klpad/b_klpae, PA) and patellofemoral (b_klpd/b_klpe, Perfil).
            "b_klpad": rng.choice([0, 1, 2, 3, 4], size=N_PARTICIPANTS, p=[0.45, 0.2, 0.2, 0.1, 0.05]),
            "b_klpae": rng.choice([0, 1, 2, 3, 4], size=N_PARTICIPANTS, p=[0.45, 0.2, 0.2, 0.1, 0.05]),
            "b_klpd": rng.choice([0, 1, 2, 3, 4], size=N_PARTICIPANTS, p=[0.55, 0.25, 0.12, 0.05, 0.03]),
            "b_klpe": rng.choice([0, 1, 2, 3, 4], size=N_PARTICIPANTS, p=[0.55, 0.25, 0.12, 0.05, 0.03]),
            # Legacy columns (present but ignored when the revised ones exist)
            "b_kld": rng.choice([0, 1, 2, 3], size=N_PARTICIPANTS, p=[0.5, 0.2, 0.2, 0.1]),
            "b_kle": rng.choice([0, 1, 2, 3], size=N_PARTICIPANTS, p=[0.5, 0.2, 0.2, 0.1]),
            "b_oapfd": rng.choice([0, 1], size=N_PARTICIPANTS, p=[0.85, 0.15]),
            "b_oapfe": rng.choice([0, 1], size=N_PARTICIPANTS, p=[0.85, 0.15]),
            # Socioeconomic (education ordinal 1-4; income continuous)
            "a_escolar": rng.integers(1, 5, N_PARTICIPANTS),
            "b_escolar": rng.integers(1, 5, N_PARTICIPANTS),
            "a_escolarmae": rng.integers(1, 5, N_PARTICIPANTS),
            "b_vifb43_pmcat": rng.normal(4000, 2000, N_PARTICIPANTS).round(1),
            "b_rendapercapita": rng.normal(2000, 1000, N_PARTICIPANTS).round(1),
            # WOMAC subscales (loaded but excluded from features by default)
            "WOMTOTD_LB": rng.normal(10, 5, N_PARTICIPANTS).round(1),
            "WOMTOTE_LB": rng.normal(10, 5, N_PARTICIPANTS).round(1),
            "WOMDORD_LB": rng.normal(2, 1, N_PARTICIPANTS).round(1),
            "WOMDORE_LB": rng.normal(2, 1, N_PARTICIPANTS).round(1),
            "WOMRIGD_LB": rng.normal(1, 0.5, N_PARTICIPANTS).round(1),
            "WOMRIGE_LB": rng.normal(1, 0.5, N_PARTICIPANTS).round(1),
            "WOMFUND_LB": rng.normal(8, 4, N_PARTICIPANTS).round(1),
            "WOMFUNE_LB": rng.normal(8, 4, N_PARTICIPANTS).round(1),
        }
    )
    return base


@pytest.fixture(scope="module", autouse=True)
def _disable_comp_merge():
    """Point COMP_KL_DTA at a nonexistent path so the synthetic prep uses the
    revised KL columns already in the synthetic CSV instead of merging the real
    (id-mismatched) complementary .dta."""
    from pathlib import Path as _P

    from koa_screening import config

    orig = config.COMP_KL_DTA
    config.COMP_KL_DTA = _P("__no_such_comp_file__.dta")
    yield
    config.COMP_KL_DTA = orig


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory) -> Path:
    csv = tmp_path_factory.mktemp("synthetic") / "synthetic.csv"
    _synthetic_df().to_csv(csv, index=False)
    return csv


@pytest.fixture(scope="module")
def prepped(synthetic_csv, tmp_path_factory):
    from koa_screening import data

    outdir = tmp_path_factory.mktemp("audit")
    return data.load_and_prep_data(str(synthetic_csv), outdir=str(outdir))


def test_load_and_prep_returns_long_dataset(prepped):
    """Two rows per participant (one per knee) and the participant id is preserved."""
    assert "idelsa" in prepped.columns
    assert "side" in prepped.columns
    assert prepped["side"].isin(["D", "E"]).all()
    # Each participant contributes up to 2 rows; >=1 always.
    assert prepped.groupby("idelsa").size().max() <= 2


def test_load_and_prep_creates_binary_outcome(prepped):
    """The derived ``oa_knee`` outcome is 0/1, with both classes present."""
    assert "oa_knee" in prepped.columns
    assert set(prepped["oa_knee"].unique()).issubset({0, 1})
    assert prepped["oa_knee"].nunique() == 2


def test_load_and_prep_creates_dummies(prepped):
    """Raw categoricals (race_raw, occupation, ...) are expanded into dummies."""
    dummy_prefixes = ["race_raw_", "occupation_", "smoking_status_", "alcohol_use_"]
    for pfx in dummy_prefixes:
        matches = [c for c in prepped.columns if c.startswith(pfx)]
        assert matches, f"no dummies created for {pfx}"


def test_scenario_pools_exclude_womac_and_gate_bio(prepped):
    """The scenario pools follow the canonical rule: WOMAC excluded from every
    model; bioimpedance only in Virtual Maximum; symptoms only in Case-Finding."""
    from koa_screening.config import BASE_EXCLUDE, BIO_VARS, SYMPTOM_VARS, WOMAC_VARS

    all_cols = [c for c in prepped.columns if c not in BASE_EXCLUDE and c not in WOMAC_VARS]
    base_pool = [c for c in all_cols if c not in BIO_VARS]
    without = [c for c in base_pool if c not in SYMPTOM_VARS]

    assert not set(WOMAC_VARS) & set(all_cols), "WOMAC must not reach any model"
    assert set(BIO_VARS) & set(all_cols), "bio vars belong to the Virtual Maximum pool"
    assert not set(BIO_VARS) & set(base_pool), "bio vars must be gated out of the base pool"
    assert not set(SYMPTOM_VARS) & set(without), "symptoms must be gated out of Screening"
    assert set(SYMPTOM_VARS) & set(base_pool), "symptoms belong to Case-Finding"


def test_smoke_pipeline_runs_one_fold(prepped):
    """One-fold smoke run: each model can fit and produce a sane AUC."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from koa_screening.config import BASE_EXCLUDE
    from koa_screening.models import get_pipeline

    y = prepped["oa_knee"].values
    groups = prepped["idelsa"].values
    feats = [c for c in prepped.columns if c not in BASE_EXCLUDE]
    X = prepped[feats]

    cv = GroupKFold(n_splits=2)
    tr, te = next(cv.split(X, y, groups))

    for model_name in ["XGBoost", "Random Forest", "Neural Network"]:
        pipe = get_pipeline(model_name)
        pipe.fit(X.iloc[tr], y[tr])
        probs = pipe.predict_proba(X.iloc[te])[:, 1]
        auc = roc_auc_score(y[te], probs)
        # Permissive bound — synthetic data has no signal; we only care that
        # the pipeline produces a valid AUC.
        assert 0.0 <= auc <= 1.0, f"{model_name} AUC out of range: {auc}"
