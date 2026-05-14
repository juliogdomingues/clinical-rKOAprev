import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from . import data as p

@dataclass(frozen=True)
class PreparedData:
    df: pd.DataFrame
    X_full: pd.DataFrame
    X_clinical: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray


def prepare_dataset(
    csv_path: str,
    *,
    use_womac: bool = False,
    missing_col_threshold: float = 0.5,
    outdir: str | None = None,
) -> PreparedData:
    """
    Uses the *existing* data prep from oarsi_data.py and applies the SAME feature filtering
    logic used in run_analysis():

      - exclude outcome/id columns + raw categorical columns used to create dummies
      - optionally exclude WOMAC
      - drop columns with > (1-missing_col_threshold) missingness (same as dropna(thresh=len(df)*0.5))
      - create "clinical" set by removing bioimpedance variables

    Returns a PreparedData with (df, X_full, X_clinical, y, groups).
    """
    if outdir is None:
        outdir = "./results_final_analysis"
    os.makedirs(outdir, exist_ok=True)

    df = p.load_and_prep_data(csv_path, outdir=outdir)

    exclude_base = [
        "idelsa",
        "side",
        "kl",
        "oapf",
        "oa_knee",
        # raw categorical sources (dummies are created during prep)
        "race_raw",
        "occupation",
        "smoking_status",
        "physical_activity_ipaq",
        "alcohol_use",
        # safety: if any helper cols exist, exclude them
        "kl_raw_num",
        "oapf_raw_num",
    ]

    womac_vars = ["womac_total", "womac_pain", "womac_stiffness", "womac_function"]
    exclude_womac = womac_vars if not use_womac else []

    exclude = set(exclude_base + exclude_womac)

    X_cols = [c for c in df.columns if c not in exclude]
    X_full = df[X_cols].copy()

    # Drop columns with too much missingness (same intent as your pipeline)
    thresh = int(np.ceil(len(df) * missing_col_threshold))
    X_full = X_full.dropna(axis=1, thresh=thresh)

    bio_vars = ["bone_mineral_content_kg", "mineral_mass_kg", "skeletal_muscle_mass_kg"]
    X_clinical = X_full.drop(columns=[c for c in bio_vars if c in X_full.columns], errors="ignore").copy()

    y = df["oa_knee"].to_numpy()
    groups = df["idelsa"].to_numpy()

    # Optional: write the final feature lists
    pd.Series(X_full.columns, name="feature").to_csv(os.path.join(outdir, "features_full.csv"), index=False)
    pd.Series(X_clinical.columns, name="feature").to_csv(os.path.join(outdir, "features_clinical.csv"), index=False)

    return PreparedData(df=df, X_full=X_full, X_clinical=X_clinical, y=y, groups=groups)