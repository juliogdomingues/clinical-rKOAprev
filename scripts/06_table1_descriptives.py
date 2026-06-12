"""Step 6: Table 1 — participant-level descriptives by rKOA status.

Aggregates from knee-level to participant-level and reports mean +/- SD for
continuous variables, n (%) for categorical, with Welch t-tests and
chi-square tests.
"""
from __future__ import annotations

import sys

import pandas as pd
from scipy import stats

from koa_screening import data
from koa_screening.config import RAW_CSV, RESULTS_FINAL


def format_cont(series: pd.Series) -> str:
    return f"{series.mean():.1f} +/- {series.std():.1f}"


def format_cat(series: pd.Series, target: int = 1) -> str:
    count = int((series == target).sum())
    pct = (count / len(series)) * 100
    return f"{count} ({pct:.1f}%)"


def add_row(name: str, no_series: pd.Series, yes_series: pd.Series, kind: str = "cont", cat_val: int = 1) -> str:
    full_series = pd.concat([no_series, yes_series])
    if kind == "cont":
        val_total = format_cont(full_series)
        val_no = format_cont(no_series)
        val_yes = format_cont(yes_series)
        _, p_val = stats.ttest_ind(no_series.dropna(), yes_series.dropna(), equal_var=False)
    elif kind == "cat":
        val_total = format_cat(full_series, cat_val)
        val_no = format_cat(no_series, cat_val)
        val_yes = format_cat(yes_series, cat_val)
        table = [
            [int((no_series == cat_val).sum()), int((no_series != cat_val).sum())],
            [int((yes_series == cat_val).sum()), int((yes_series != cat_val).sum())],
        ]
        _, p_val, _, _ = stats.chi2_contingency(table)
    else:
        raise ValueError(kind)
    p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
    return f"| {name} | {val_total} | {val_no} | {val_yes} | {p_str} |"


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: raw CSV not found at {RAW_CSV}", file=sys.stderr)
        return 1

    df = data.load_and_prep_data(str(RAW_CSV), outdir=str(RESULTS_FINAL))
    agg_funcs = {
        "oa_knee": "max",
        "age": "first",
        "bmi": "first",
        "sex_female": "first",
        "frequent_symptoms": "max",
        "history_trauma": "max",
        "history_surgery": "max",
    }
    part_df = df.groupby("idelsa").agg(agg_funcs).reset_index()
    no_koa = part_df[part_df["oa_knee"] == 0]
    yes_koa = part_df[part_df["oa_knee"] == 1]

    print(
        f"| Characteristic | Total (N={len(part_df)}) | No rKOA (N={len(no_koa)}) | "
        f"With rKOA (N={len(yes_koa)}) | P-value |"
    )
    print("|---|---|---|---|---|")
    print(add_row("Age (years)", no_koa["age"], yes_koa["age"], "cont"))
    print(add_row("Sex (Female)", no_koa["sex_female"], yes_koa["sex_female"], "cat", 1))
    print(add_row("BMI (kg/m^2)", no_koa["bmi"], yes_koa["bmi"], "cont"))
    print(add_row("Frequent Knee Symptoms", no_koa["frequent_symptoms"], yes_koa["frequent_symptoms"], "cat"))
    print(add_row("History of Knee Trauma", no_koa["history_trauma"], yes_koa["history_trauma"], "cat"))
    print(add_row("History of Knee Surgery", no_koa["history_surgery"], yes_koa["history_surgery"], "cat"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
