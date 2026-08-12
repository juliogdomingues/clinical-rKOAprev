"""Step 13: calibration, operating-point metrics, and decision-curve analysis.

Consumes the nested-CV out-of-fold predictions written by
`scripts/12_nested_cv.py` (`nested_cv_oof_predictions.csv`, gitignored) so every
number here inherits the leak-free nested estimate.

Outputs (results/comparison/):
  calibration_metrics.csv        calibration slope + CITL (cluster-bootstrap CIs)
  calibration_curve_points.csv   observed vs predicted by decile
  threshold_metrics.csv          sens/spec/PPV/NPV at clinical thresholds + Youden
  decision_curve.csv             net benefit vs treat-all / treat-none
  fig_calibration.png            calibration plot (replaces the orphaned figures)
  fig_decision_curve.png         decision curve

Together these answer "is the model clinically usable?", which AUC + Brier alone
cannot (TRIPOD+AI requires calibration; a screening/triage claim requires
operating-point and net-benefit evidence).
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from koa_screening.clinical_utility import (
    calibration_curve_points,
    calibration_with_ci,
    decision_curve,
    threshold_metrics,
    youden_threshold,
)
from koa_screening.config import RESULTS_COMPARISON

OOF = RESULTS_COMPARISON / "nested_cv_oof_predictions.csv"
# Clinically motivated thresholds for "who should get a radiograph?" plus Youden.
FIXED_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
PRIMARY = ("Without Symptoms", "Stepwise LR")  # the deployable screening model


def main() -> int:
    if not OOF.exists():
        print(f"ERROR: {OOF} not found — run scripts/12_nested_cv.py first.", file=sys.stderr)
        return 1
    oof = pd.read_csv(OOF)

    cal_rows, curve_rows, thr_rows, dca_rows = [], [], [], []

    for (scen, model), g in oof.groupby(["Scenario", "Model"], sort=False):
        y, p, cl = g["y_true"].to_numpy(), g["y_pred"].to_numpy(), g["cluster"].to_numpy()
        print(f"  {scen} / {model} (n={len(y)}, events={int(y.sum())}) ...")

        cal = calibration_with_ci(y, p, cl)
        cal.update({"Scenario": scen, "Model": model, "n": len(y), "events": int(y.sum())})
        cal_rows.append(cal)

        cp = calibration_curve_points(y, p)
        cp.insert(0, "Model", model); cp.insert(0, "Scenario", scen)
        curve_rows.append(cp)

        thrs = sorted(set(FIXED_THRESHOLDS) | {round(youden_threshold(y, p), 3)})
        tm = threshold_metrics(y, p, cl, thrs)
        tm.insert(0, "Model", model); tm.insert(0, "Scenario", scen)
        tm["is_youden"] = np.isclose(tm["threshold"], round(youden_threshold(y, p), 3))
        thr_rows.append(tm)

        dc = decision_curve(y, p)
        dc.insert(0, "Model", model); dc.insert(0, "Scenario", scen)
        dca_rows.append(dc)

    cal_df = pd.DataFrame(cal_rows)[
        ["Scenario", "Model", "n", "events", "calibration_slope", "slope_ci_low", "slope_ci_high",
         "calibration_in_the_large", "citl_ci_low", "citl_ci_high", "n_boot_used"]]
    cal_df.to_csv(RESULTS_COMPARISON / "calibration_metrics.csv", index=False)
    pd.concat(curve_rows, ignore_index=True).to_csv(RESULTS_COMPARISON / "calibration_curve_points.csv", index=False)
    thr_all = pd.concat(thr_rows, ignore_index=True)
    thr_all.to_csv(RESULTS_COMPARISON / "threshold_metrics.csv", index=False)
    dca_all = pd.concat(dca_rows, ignore_index=True)
    dca_all.to_csv(RESULTS_COMPARISON / "decision_curve.csv", index=False)

    # ---- Figures for the primary screening model ----
    scen, model = PRIMARY
    sub = oof[(oof.Scenario == scen) & (oof.Model == model)]
    if len(sub):
        y, p = sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()

        cp = calibration_curve_points(y, p)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, cp["mean_predicted"].max() * 1.1], [0, cp["mean_predicted"].max() * 1.1],
                "k--", lw=1, label="Perfect calibration")
        ax.errorbar(cp["mean_predicted"], cp["observed"],
                    yerr=[cp["observed"] - cp["obs_ci_low"], cp["obs_ci_high"] - cp["observed"]],
                    fmt="o-", color="navy", capsize=3, label="Observed (decile, 95% CI)")
        ax.set_xlabel("Predicted probability of rKOA")
        ax.set_ylabel("Observed proportion with rKOA")
        ax.set_title(f"Calibration — {model}, {scen}\n(nested-CV out-of-fold predictions)")
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(RESULTS_COMPARISON / "fig_calibration.png", dpi=300); plt.close(fig)

        # Main-text ROC figure, built from the SAME nested out-of-fold estimates
        # as the reported areas under the curve, so the figure and the text agree.
        # (The roc_comparison_*.png files come from the single-CV analysis and
        # show different values; they must not be used for the main text.)
        from sklearn.metrics import roc_auc_score, roc_curve

        fig, ax = plt.subplots(figsize=(7.5, 7))
        order = ["Stepwise LR", "XGBoost", "Random Forest", "Neural Network"]
        names = {"Stepwise LR": "Logistic regression"}
        for m in order:
            g = oof[(oof.Scenario == scen) & (oof.Model == m)]
            if not len(g):
                continue
            fpr, tpr, _ = roc_curve(g["y_true"], g["y_pred"])
            auc = roc_auc_score(g["y_true"], g["y_pred"])
            ax.plot(fpr, tpr, lw=2, label=f"{names.get(m, m)} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
        ax.legend(loc="lower right", frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS_COMPARISON / "fig_roc_nested.png", dpi=300)
        plt.close(fig)

        dc = decision_curve(y, p)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(dc["threshold"], dc["net_benefit_model"], lw=2.5, color="crimson", label=f"{model}")
        ax.plot(dc["threshold"], dc["net_benefit_treat_all"], lw=1.5, color="grey", label="Radiograph all")
        ax.axhline(0, color="black", lw=1, ls="--", label="Radiograph none")
        ax.set_ylim(min(-0.02, dc["net_benefit_model"].min()), dc["net_benefit_model"].max() * 1.25 + 0.01)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title(f"Decision-curve analysis — {model}, {scen}\n(nested-CV out-of-fold predictions)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(RESULTS_COMPARISON / "fig_decision_curve.png", dpi=300); plt.close(fig)

    print("\n=== CALIBRATION (slope ideal 1.0, CITL ideal 0.0) ===")
    show = cal_df.copy()
    show["slope (95% CI)"] = show.apply(lambda r: f"{r.calibration_slope:.2f} ({r.slope_ci_low:.2f}-{r.slope_ci_high:.2f})", axis=1)
    show["CITL (95% CI)"] = show.apply(lambda r: f"{r.calibration_in_the_large:+.2f} ({r.citl_ci_low:+.2f},{r.citl_ci_high:+.2f})", axis=1)
    print(show[["Scenario", "Model", "slope (95% CI)", "CITL (95% CI)"]].to_string(index=False))

    print(f"\n=== OPERATING POINTS — {model}, {scen} ===")
    t = thr_all[(thr_all.Scenario == scen) & (thr_all.Model == model)].copy()
    for c in ["sensitivity", "specificity", "ppv", "npv"]:
        t[c] = t[c].round(3)
    print(t[["threshold", "pct_flagged", "sensitivity", "specificity", "ppv", "npv", "is_youden"]].to_string(index=False))

    d = dca_all[(dca_all.Scenario == scen) & (dca_all.Model == model)]
    best = d.loc[d["net_benefit_gain_vs_best_default"].idxmax()]
    print(f"\n=== DECISION CURVE — {model}, {scen} ===")
    print(f"  Largest net-benefit gain over treat-all/none at threshold "
          f"{best['threshold']:.2f}: {best['net_benefit_gain_vs_best_default']:+.4f}")
    pos = d[d["net_benefit_gain_vs_best_default"] > 0]["threshold"]
    if len(pos):
        print(f"  Model is the preferred strategy for thresholds {pos.min():.2f}-{pos.max():.2f}")
    print(f"\nWrote calibration/threshold/decision-curve outputs to {RESULTS_COMPARISON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
