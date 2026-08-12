"""Step 14: generate the manuscript tables and the final model specification.

Produces, in results/manuscript/:
  table1_participants.csv        participant characteristics by rKOA status
  table2_final_model_or.csv      odds ratios for the Constitutional model
  table3_candidate_variables.csv distributions of all candidate variables
  table4_lasso_coefficients.csv  variables retained by the penalised fit
  final_model_coefficients.csv   intercept + standardised coefficients (TRIPOD 15a)

The final model specification is what allows the model to be applied and
externally validated; it is required for prediction-model submissions and is not
otherwise exported for the Constitutional model.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from koa_screening import data
from koa_screening.config import (
    BASE_EXCLUDE,
    BIO_VARS,
    RAW_CSV,
    RESULTS_DIR,
    RESULTS_FINAL,
    RND,
    SYMPTOM_VARS,
    WOMAC_VARS,
)

OUT = RESULTS_DIR / "manuscript"

# Constitutional model, in the order selected by the forward stepwise procedure.
FINAL_FEATURES = [
    "age", "bmi", "history_surgery", "history_trauma",
    "occupation_4", "waist_hip_ratio", "race_raw_3",
]

LABELS = {
    "age": "Age, years",
    "bmi": "Body mass index, kg/m2",
    "history_surgery": "History of knee surgery",
    "history_trauma": "History of knee trauma",
    "occupation_4": "Occupation: non-routine non-manual",
    "waist_hip_ratio": "Waist-hip ratio",
    "race_raw_3": "Race and skin colour: White",
    "frequent_symptoms": "Frequent knee symptoms",
    "recent_pain_7d": "Knee symptoms in the previous 7 days",
    "knee_disability": "Knee-related activity limitation",
}


def _fmt_p(p: float) -> str:
    """Numeric p-values without categorisation (journal Guide for Authors)."""
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def table1(df: pd.DataFrame) -> pd.DataFrame:
    part = df.groupby("idelsa").agg(
        oa=("oa_knee", "max"), age=("age", "first"), bmi=("bmi", "first"),
        sex_female=("sex_female", "first"), race=("race_raw", "first"),
        occupation=("occupation", "first"),
        frequent_symptoms=("frequent_symptoms", "max"),
        history_trauma=("history_trauma", "max"),
        history_surgery=("history_surgery", "max"),
        whr=("waist_hip_ratio", "first"),
    ).reset_index()
    no, yes = part[part.oa == 0], part[part.oa == 1]
    rows = []

    def cont(label, col, unit=""):
        a, b = no[col].dropna(), yes[col].dropna()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({
            "Characteristic": label,
            f"Total (n={len(part)})": f"{part[col].mean():.1f} ({part[col].std():.1f})",
            f"Without rKOA (n={len(no)})": f"{a.mean():.1f} ({a.std():.1f})",
            f"With rKOA (n={len(yes)})": f"{b.mean():.1f} ({b.std():.1f})",
            "p": _fmt_p(p),
        })

    def binary(label, col):
        tab = [[int((no[col] == 1).sum()), int((no[col] != 1).sum())],
               [int((yes[col] == 1).sum()), int((yes[col] != 1).sum())]]
        _, p, _, _ = stats.chi2_contingency(tab)
        f = lambda d: f"{int((d[col] == 1).sum())} ({(d[col] == 1).mean()*100:.1f})"  # noqa: E731
        rows.append({
            "Characteristic": label,
            f"Total (n={len(part)})": f(part),
            f"Without rKOA (n={len(no)})": f(no),
            f"With rKOA (n={len(yes)})": f(yes),
            "p": _fmt_p(p),
        })

    def categorical(label, col, mapping):
        # data.py stores the raw categoricals as strings ('1', '2', ...) after
        # dummy creation; compare on the string form and drop the '-1' missing code.
        part[col] = part[col].astype(str)
        sub = part[part[col].notna() & (part[col] != "-1")]
        tab = pd.crosstab(sub[col], sub.oa)
        _, p, _, _ = stats.chi2_contingency(tab)
        rows.append({"Characteristic": label, f"Total (n={len(part)})": "",
                     f"Without rKOA (n={len(no)})": "", f"With rKOA (n={len(yes)})": "", "p": _fmt_p(p)})
        for code, name in mapping.items():
            key = str(code)
            f = lambda d, k=key: f"{int((d[col] == k).sum())} ({(d[col] == k).mean()*100:.1f})"  # noqa: E731
            rows.append({
                "Characteristic": f"  {name}",
                f"Total (n={len(part)})": f(part),
                f"Without rKOA (n={len(no)})": f(no),
                f"With rKOA (n={len(yes)})": f(yes),
                "p": "",
            })

    cont("Age, years", "age")
    binary("Female sex", "sex_female")
    categorical("Race and skin colour", "race",
                {1: "Black", 2: "Brown", 3: "White", 4: "Asian", 5: "Indigenous"})
    categorical("Occupational nature", "occupation",
                {1: "Routine manual", 2: "Non-routine manual",
                 3: "Routine non-manual", 4: "Non-routine non-manual"})
    cont("Body mass index, kg/m2", "bmi")
    cont("Waist-hip ratio", "whr")
    binary("Frequent knee symptoms", "frequent_symptoms")
    binary("History of knee trauma", "history_trauma")
    binary("History of knee surgery", "history_surgery")
    return pd.DataFrame(rows)


def table2_and_model(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.api as sm

    feats = [f for f in FINAL_FEATURES if f in df.columns]
    y = df["oa_knee"].values
    groups = df["idelsa"].values
    X = df[feats]
    imp = SimpleImputer(strategy="median")
    Xi = pd.DataFrame(imp.fit_transform(X), columns=feats, index=X.index)

    # Standardised odds ratios (per SD for continuous, per category for binary)
    sc = StandardScaler()
    Xs = pd.DataFrame(sc.fit_transform(Xi), columns=feats, index=Xi.index)
    res_s = sm.Logit(y, sm.add_constant(Xs)).fit(disp=0, cov_type="cluster",
                                                 cov_kwds={"groups": groups})
    res_r = sm.Logit(y, sm.add_constant(Xi)).fit(disp=0, cov_type="cluster",
                                                 cov_kwds={"groups": groups})

    binary_vars = {"history_surgery", "history_trauma", "occupation_4", "race_raw_3"}
    rows = []
    for f in feats:
        # Report per category for binary variables, per SD for continuous ones.
        r = res_r if f in binary_vars else res_s
        ci = r.conf_int()
        or_ = float(np.exp(r.params[f]))
        lo, hi = float(np.exp(ci.loc[f, 0])), float(np.exp(ci.loc[f, 1]))
        rows.append({
            "Variable": LABELS.get(f, f),
            "Scale": "per category" if f in binary_vars else "per SD",
            "Odds ratio (95% CI)": f"{or_:.2f} ({lo:.2f}, {hi:.2f})",
            "p": _fmt_p(float(r.pvalues[f])),
        })
    t2 = pd.DataFrame(rows)

    # Final model specification (TRIPOD item 15a)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight=None, random_state=RND))
    pipe.fit(X, y)
    lr = pipe.named_steps["logisticregression"]
    imp2 = pipe.named_steps["simpleimputer"]
    sc2 = pipe.named_steps["standardscaler"]
    spec = pd.DataFrame({
        "variable": feats,
        "imputation_median": imp2.statistics_,
        "mean": sc2.mean_,
        "sd": sc2.scale_,
        "beta_per_sd": lr.coef_[0],
    })
    spec.loc[len(spec)] = ["(Intercept)", np.nan, np.nan, np.nan, float(lr.intercept_[0])]
    return t2, spec


def figure1(df: pd.DataFrame) -> pd.DataFrame:
    """Incremental change in cross-validated AUC for the Constitutional model.

    The values are the cross-validated estimates that guided forward selection,
    computed on the full sample. They are not the validated estimates: those come
    from the nested procedure and are reported separately. The distinction is
    stated in the figure caption so the two sets of numbers are not conflated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from koa_screening.runner import run_stepwise_mpms

    feats = [f for f in FINAL_FEATURES if f in df.columns]
    traj = run_stepwise_mpms(df[feats], df["oa_knee"].values, df["idelsa"].values, feats)
    traj["Label"] = traj["Added Variable"].map(lambda f: LABELS.get(f, f))
    traj.to_csv(OUT / "figure1_constitutional_trajectory.csv", index=False)

    # Short labels keep the annotation boxes from overlapping each other and the
    # axes; the full variable names are given in Table 2.
    short = {
        "age": "Age", "bmi": "BMI", "history_surgery": "Knee surgery",
        "history_trauma": "Knee trauma", "occupation_4": "Occupation",
        "waist_hip_ratio": "Waist-hip ratio", "race_raw_3": "Race",
    }
    traj["Short"] = traj["Added Variable"].map(lambda f: short.get(f, f))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(traj["k"], traj["AUC"], "o-", color="#1f4e79", lw=2, markersize=7)
    for _, r in traj.iterrows():
        k = int(r["k"])
        ha = "left" if k == 1 else ("right" if k == len(traj) else "center")
        dx = 10 if k == 1 else (-10 if k == len(traj) else 0)
        ax.annotate(f"{r['Short']}\n{r['AUC']:.3f}", (r["k"], r["AUC"]),
                    xytext=(dx, -38 if k % 2 == 0 else 22), textcoords="offset points",
                    ha=ha, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.95))
    ax.set_xticks(traj["k"])
    ax.set_xlim(0.5, len(traj) + 0.5)
    ax.set_xlabel("Number of variables")
    ax.set_ylabel("Cross-validated area under the curve")
    lo, hi = traj["AUC"].min(), traj["AUC"].max()
    ax.set_ylim(lo - (hi - lo) * 0.30, hi + (hi - lo) * 0.22)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figure1_constitutional_trajectory.png", dpi=300)
    plt.close(fig)
    return traj


def table3(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in BASE_EXCLUDE and c not in WOMAC_VARS]
    rows = []
    for c in sorted(cols):
        s = df[c]
        miss = s.isna().mean() * 100
        uniq = set(pd.unique(s.dropna()))
        if uniq <= {0, 1} and len(uniq) <= 2:
            rows.append({"Variable": LABELS.get(c, c), "Type": "binary",
                         "Summary": f"{int((s == 1).sum())} ({(s == 1).mean()*100:.1f})",
                         "Missing, %": f"{miss:.1f}"})
        else:
            rows.append({"Variable": LABELS.get(c, c), "Type": "continuous",
                         "Summary": f"{s.mean():.1f} ({s.std():.1f})",
                         "Missing, %": f"{miss:.1f}"})
    return pd.DataFrame(rows)


def table4() -> pd.DataFrame:
    p = RESULTS_FINAL / "lasso_coefficients_clinical.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    d = d[~d["is_zero"]].copy()
    d["Variable"] = d["feature"].map(lambda f: LABELS.get(f, f))
    d["Coefficient"] = d["coef"].round(3)
    return d[["Variable", "Coefficient"]].reset_index(drop=True)


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: raw CSV not found at {RAW_CSV}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    df = data.load_and_prep_data(str(RAW_CSV), outdir=str(RESULTS_FINAL))
    df = df.sort_values("idelsa").reset_index(drop=True)

    t1 = table1(df)
    t1.to_csv(OUT / "table1_participants.csv", index=False)
    t2, spec = table2_and_model(df)
    t2.to_csv(OUT / "table2_final_model_or.csv", index=False)
    spec.to_csv(OUT / "final_model_coefficients.csv", index=False)
    traj = figure1(df)
    table3(df).to_csv(OUT / "table3_candidate_variables.csv", index=False)
    t4 = table4()
    if len(t4):
        t4.to_csv(OUT / "table4_lasso_coefficients.csv", index=False)

    print("=== Table 1 ===");  print(t1.to_string(index=False))
    print("\n=== Table 2 ==="); print(t2.to_string(index=False))
    print("\n=== Final model specification ==="); print(spec.round(4).to_string(index=False))
    print("\n=== Figure 1: Constitutional trajectory ===")
    print(traj[["k", "Label", "AUC"]].round(4).to_string(index=False))
    print(f"\n=== Table 4: {len(t4)} variables retained by the penalised fit ===")
    print(f"\nWritten to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
